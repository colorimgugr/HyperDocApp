import os
import numpy as np
import h5py
from spectral.io import envi
from PyQt5.QtWidgets import QApplication, QFileDialog,QMessageBox
from PyQt5.QtWidgets import QDialog
from hypercubes.save_window import Ui_Save_Window
import cv2
from scipy.io import savemat

# TODO : look at ENVI open with lowercase etc...
# TODO : hypercube class to enforce in all tools

class Hypercube:
    def __init__(self, filepath=None,data=None,wl=None,metadata=None,load_init=False):
        self.filepath = filepath
        self.data = data
        self.wl = wl
        self.metadata = metadata

        if load_init:
            if  self.filepath:
                self.open_hyp(self.filepath,open_dialog=False)
            else:
                self.open_hyp(open_dialog=True)

    def reinit_cube(self):
        self.filepath = None
        self.data = None
        self.wl = None
        self.metadata = None

    def get_rgb_image(self, indices):
        if self.data is None:
            return None
        return self.data[:, :, indices]

    def open_hyp(self,default_path: str = "",
                       open_dialog: bool = True,show_except : bool=True ):

        # 1) Ask user to select a file if requested
        if open_dialog:
            app = QApplication.instance() or QApplication([])
            filepath, _ = QFileDialog.getOpenFileName(
                None,
                "Open hyperspectral cube",
                default_path,
                "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not filepath:
                return
            else: self.filepath=filepath

        else:
            filepath = default_path

        ext = os.path.splitext(filepath)[1].lower()

        # 2) MATLAB .mat (HDF5-based)
        if ext == ".mat":
            try:
                with h5py.File(filepath, "r") as f:
                    # Your internal layout: '#refs#/d' → wavelengths, '#refs#/c' → data
                    self.wl = f["#refs#/d"][:].flatten()
                    raw = f["#refs#/c"][:]               # shape e.g. (bands, samples, lines)
                    self.data = np.transpose(raw, (2, 1, 0))  # → (lines, samples, bands)
                    self.metadata = {}
                    if "Metadata" in f:
                        meta_grp = f["Metadata"]
                        for name, ds in meta_grp.items():
                            arr = ds[()]
                            mclass = ds.attrs.get("MATLAB_class", b"").decode()

                            if mclass == "char":
                                # convertir codes ASCII en str
                                bb = np.asarray(arr, np.uint8).flatten()
                                bb = bb[bb != 0]  # retirer les zéros
                                self.metadata[name] = bb.tobytes().decode("utf-8", errors="ignore")

                            elif mclass == "logical":
                                self.metadata[name] = np.asarray(arr, dtype=bool)

                            else:
                                # double, single, int, etc.
                                self.metadata[name] = arr
                    return
            except Exception:
                if show_except:
                    QMessageBox.critical(
                        None, "Error",
                        "Unsupported MATLAB file structure.\nPlease contact support."
                    )
                self.reinit_cube()
                return

        # 3) HDF5 .h5 / .hdf5
        elif ext in (".h5", ".hdf5"):
            try:
                with h5py.File(filepath, "r") as f:
                    raw = f["DataCube"][:]               # e.g. (bands, samples, lines)
                    self.data = np.transpose(raw, (2, 1, 0))
                    self.metadata = {k: f.attrs[k] for k in f.attrs}
                    self.wl = self.metadata.get("wl")
                    return
            except Exception as e:
                if show_except:
                    QMessageBox.critical(None, "Error", f"Failed to read HDF5 file:\n{e}")
                self.reinit_cube()
                return

        # 4) ENVI (.hdr + raw .dat/.img)
        elif ext == ".hdr":
            try:
                # Open the ENVI header (automatically finds the raw)
                img = envi.open(filepath)
                # Load the full cube into memory
                self.data = img.load().astype(np.float32)  # shape: (lines, samples, bands)
                self.metadata = img.metadata.copy()

                # Extract wavelengths (key may be 'wavelength' or 'wavelengths')
                self.wl = self.metadata.get("wavelength", self.metadata.get("wavelengths"))
                if isinstance(self.wl, str):
                    # Convert a string "{400,410,...}" into a numeric array
                    wl = self.wl.strip("{}").split(",")
                    self.wl = np.array(wl, dtype=np.float32)
                else:
                    self.wl = np.array(self.wl, dtype=np.float32) if self.wl is not None else None

                return

            except Exception as e:
                if show_except:
                    QMessageBox.critical(None, "Error", f"Failed to read ENVI file:\n{e}")
                self.reinit_cube()
                return

        # 5) Unsupported extension
        else:
            if show_except:
                QMessageBox.critical(
                    None, "Unsupported format",
                    f"The extension '{ext}' is not supported."
                )
            self.reinit_cube()
            return

    def save_hyp(self,filepath=None,fmt='HDF5'):

        if filepath is None:
            filepath, _ = QFileDialog.getSaveFileName(
                parent=None,
                caption="Save cube As…")
            if filepath is None:
                print('Problem getting filepath to save')
                return

        if fmt=='HDF5':
            self.save_hdf5_cube(filepath)

    def save_hdf5_cube(self,filepath: str):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("DataCube", data=self.data)
            for key, val in self.metadata.items():
                f.attrs[key] = val

    def save_envi_cube(self,filepath: str,
                       interleave: str = "bil",
                       dtype_code: int = 4):

        os.makedirs(filepath, exist_ok=True)
        hdr_meta = {
            "lines": self.data.shape[0],
            "samples": self.data.shape[1],
            "bands": self.data.shape[2],
            "data type": dtype_code,
            "interleave": interleave
        }
        if self.metadata is not None:
           hdr_meta.update(self.metadata)
        envi.save_image(filepath, self.data.astype(np.float32), metadata=hdr_meta)

    def save_matlab_cube(self,filepath: str):
        from scipy.io import savemat
        tosave = {'DataCube': filepath}
        if self.metadata:
            for key, value in self.metadata.items():
                tosave[key] = value

        savemat(filepath, tosave)

class SaveWindow(QDialog, Ui_Save_Window):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Accept on Save
        self.pushButton_save_cube_final.clicked.connect(self.accept)
        # Reject on Cancel
        self.pushButton_Cancel.clicked.connect(self.reject)

    def closeEvent(self, event):
        """
        Si l'utilisateur ferme la fenêtre via la croix,
        on rejette aussi le dialogue.
        """
        self.reject()
        super().closeEvent(event)

    def get_options(self):
        # ... ta méthode existante ...
        opts = {}
        opts['cube_format']    = self.comboBox_cube_format.currentText()
        opts['save_both']      = self.radioButton_both_cube_save.isChecked()
        opts['crop_cube']      = self.checkBox_minicube_save.isChecked()
        opts['export_images']  = self.checkBox_export_images.isChecked()
        if opts['export_images']:
            opts['image_format']   = self.comboBox_image_format.currentText()
            opts['image_mode_rgb'] = self.radioButton_RGB_save_image.isChecked()
        else:
            opts['image_format']   = None
            opts['image_mode_rgb'] = False
        opts['modify_metadata'] = self.checkBox_modif_metadata.isChecked()
        return opts



def save_images(dirpath: str,
                fixed_img: np.ndarray,
                aligned_img: np.ndarray,
                image_format: str = "png",
                rgb: bool = False):

    os.makedirs(dirpath, exist_ok=True)
    ext = image_format.lower()
    def write(name, img):
        out = os.path.join(dirpath, f"{name}.{ext}")
        if img.ndim == 2 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out, img)
    write("fixed", fixed_img)
    write("aligned", aligned_img)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    sample='MPD41a_SWIR.mat'
    folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    filepath=folder_cube+sample
    filepath=None
    try:
        cube=Hypercube(filepath,load_init=True)
        hyp=cube.data
        plt.figure()
        plt.imshow(hyp[:, :, [50, 30, 10]]/np.max(hyp[:, :, [50, 30, 10]]))
        plt.axis('off')
    except: print('Hyp not loaded')

