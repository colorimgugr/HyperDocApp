import os
import subprocess
import tempfile
import scipy.io
import numpy as np
import glob

def find_matlab_executable():
    base_dir = r"C:\Program Files\MATLAB"
    if not os.path.isdir(base_dir):
        return None

    # Recherche toutes les versions installées
    versions = sorted(
        glob.glob(os.path.join(base_dir, "R20*", "bin", "matlab.exe")),
        reverse=True
    )

    if versions:
        return versions[0]  # la version la plus récente
    return None


def load_mat_file_with_matlab_obj(filepath, debug=False):
    """
    Charge un fichier .mat contenant un objet `cube` depuis MATLAB sans utiliser matlab.engine,
    en accédant à cube.DataCube, cube.Wavelength et Metadata, et en exportant dans un .mat lisible.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    tmp_dir = tempfile.mkdtemp()
    script_path = os.path.join(tmp_dir, "extract_cube_fields.m")
    output_path = os.path.join(tmp_dir, "cube_fields_output.mat")

    filepath_matlab = filepath.replace("\\", "/")
    output_path_matlab = output_path.replace("\\", "/")

    # Création du script MATLAB
    with open(script_path, "w") as f:
        f.write(f"""
        try
            disp('Trying to load you cube .mat from Matlab environment.');
            disp('This may take one minute. Please wait.');

            load('{filepath_matlab}');

            disp('Getting from cube : DataCube, Wavelength, Metadata...');
            data = cube.DataCube;

            try
                wl = cube.Wavelength;
            catch
                wl = [];
                disp(' [ !!! ] Wavelength not found');
            end

            try
                meta = Metadata;
            catch
                meta = [];
                disp(' [ !!! ] Metadata not found');
            end

            disp('Sending cube to python...');
            save('{output_path_matlab}', 'data', 'wl', 'meta');

            disp('[ :-) ]Sended to python with succes.');
        catch ME
            disp('[ !!!] An errror occured :');
            disp(getReport(ME));
            exit(1);
        end
        exit;
        """)

    # Exécution de MATLAB
    try:
        matlab_cmd = find_matlab_executable() or "matlab"
        result = subprocess.run(
            [matlab_cmd, "-batch", f"run('{script_path.replace('\\\\', '/')}')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print("❌ MATLAB error")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("MATLAB returned an error")
    except Exception as e:
        raise RuntimeError(f"MATLAB batch execution failed: {e}")

    if not os.path.exists(output_path):
        raise FileNotFoundError("MATLAB did not produce expected output")

    # Lecture finale en Python
    out = scipy.io.loadmat(output_path)
    data = np.array(out.get("data"))
    wl = np.array(out.get("wl")).squeeze() if "wl" in out else None
    meta_raw = out.get("meta", None)

    # Conversion de meta MATLAB struct → dict Python
    meta = None
    if meta_raw is not None and meta_raw.dtype.names:
        meta = {field: meta_raw[field][0, 0] for field in meta_raw.dtype.names}

    # Nettoyage
    if not debug:
        try:
            os.remove(script_path)
            os.remove(output_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    return data, wl, meta