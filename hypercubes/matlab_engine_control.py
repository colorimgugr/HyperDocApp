import os
import sys
import numpy as np

def configure_matlab_env():
    """Ajoute le moteur MATLAB au sys.path et à l'environnement si nécessaire (Windows uniquement)."""
    base = r"C:\Program Files\MATLAB"
    if not os.path.isdir(base):
        return False

    versions = sorted(
        [v for v in os.listdir(base) if v.startswith("R")],
        reverse=True
    )

    for version in versions:
        base_path  = os.path.join(base, version)
        extern_bin = os.path.join(base_path, "extern", "bin", "win64")
        core_bin   = os.path.join(base_path, "bin", "win64")
        python_eng = os.path.join(base_path, "extern", "engines", "python")

        if all(os.path.isdir(p) for p in [extern_bin, core_bin, python_eng]):
            for p in [extern_bin, core_bin]:
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

            if python_eng not in sys.path:
                sys.path.insert(0, python_eng)

            return True
    return False


def get_matlab_engine():
    """Retourne le moteur MATLAB prêt à être utilisé, ou lève une erreur claire."""
    if not configure_matlab_env():
        raise RuntimeError("⚠️ MATLAB non trouvé ou chemin incorrect.")

    try:
        import matlab.engine
    except ImportError:
        raise RuntimeError("⚠️ Le module `matlab.engine` est introuvable. Exécute `cd extern/engines/python && python setup.py install`.")

    return matlab.engine


def load_mat_file_with_engine(filepath):
    """Charge un fichier .mat avec un objet `cube` via MATLAB Engine."""
    eng = get_matlab_engine().start_matlab()

    try:
        eng.eval(f"load('{filepath}')", nargout=0)

        try:
            data = eng.eval("cube.DataCube", nargout=1)
        except:
            raise ValueError("❌ Impossible de trouver cube.DataCube")

        try:
            wl = eng.eval("cube.Wavelength", nargout=1)
        except:
            wl = None

        try:
            metadata = eng.eval("Metadata", nargout=1)
        except:
            metadata = None

    finally:
        eng.quit()

    return np.array(data), np.array(wl).squeeze() if wl is not None else None, metadata
