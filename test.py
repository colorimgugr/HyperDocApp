import h5py
import numpy as np


def inspect_hdf5(path_mat):
    with h5py.File(path_mat, 'r') as f:

        # 1. attributs globaux du fichier
        if f.attrs:
            print("== Attributs fichiers ==")
            for k, v in f.attrs.items():
                print(f"  {k!r}  →  {v}")

        # 2. parcourir récursivement tous les objets
        def print_attrs(name, obj):
            if obj.attrs:
                print(f"\n-- Attributs de {name} --")
                for k, v in obj.attrs.items():
                    print(f"  {k!r}  →  {v}")
        f.visititems(print_attrs)

def load_mat_v73_hypercube(filepath):
    with h5py.File(filepath, 'r') as f:
        # 1) Récupérer la référence à l'objet hypercube
        refs = f['cube'][()]           # array dtype=object ou uint32
        obj_ref = refs.flat[0]         # HDF5 object reference

        # 2) Aller chercher le groupe “cube” réel
        cube_grp = f[obj_ref]          # c'est un Group HDF5

        # 3) Lire vos vraies données
        data_ref = cube_grp['DataCube'][()].flat[0]
        wavelengths_ref = cube_grp['Wavelength'][()].flat[0]
        data        = f[data_ref][()]          # numpy.ndarray
        wavelengths = f[wavelengths_ref][()]   # numpy.ndarray

        # 4) Charger les métadonnées
        metadata = {}
        for name, ds in f['Metadata'].items():
            arr = ds[()]
            mclass = ds.attrs.get('MATLAB_class', b'').decode()
            if mclass == 'char':
                bb = np.asarray(arr, np.uint8).flatten()
                bb = bb[bb != 0]
                metadata[name] = bb.tobytes().decode('utf-8')
            else:
                metadata[name] = arr

    return data, wavelengths, metadata

# Usage
import re

def get_matlab_version(filepath):
    """
    Retourne la version MATLAB qui a créé le .mat :
     - '7.3' si HDF5 (v7.3+)
     - '5.0', '7.2', etc. pour les autres
     - None si indéterminé
    """
    # signature HDF5
    HDF5_SIG = b'\x89HDF\r\n\x1a\n'

    with open(filepath, 'rb') as f:
        sig = f.read(8)
        if sig == HDF5_SIG:
            return '7.3 (HDF5)'
        # pas HDF5 : relire le début en ASCII
        f.seek(0)
        header = f.read(116).decode('ascii', errors='ignore')
        # chercher "MATLAB X.Y MAT-file"
        m = re.search(r'MATLAB\s+(\d+\.\d+)\s+MAT-file', header)
        if m:
            return m.group(1)
    return None


sample='MPD41a_SWIR.mat'
folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
filepath=folder_cube+sample
version = get_matlab_version(filepath)
print(f"Version MATLAB détectée : {version}")

# inspect_hdf5(filepath)
data, wl, meta = load_mat_v73_hypercube(filepath)
print("shape hypercube:", data.shape)
print("shape wl       :", wl.shape)
print("metadata keys  :", list(meta.keys()))
# for k, v in meta.items():
#     print(f"{k:20s} → {v}")
