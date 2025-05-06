import os
import numpy as np
import h5py
from spectral.io import envi
from scipy.io import savemat, loadmat
import cv2

from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QDialog, QTreeWidgetItem
)
from hypercubes.save_window import Ui_Save_Window
from hypercubes.HDF5BrowserDialog import Ui_HDF5BrowserDialog

#TODO : finish to check transpose at loading and add other clasical open idem than saving option in matlab

class HDF5BrowserDialog(QDialog, Ui_HDF5BrowserDialog):
    """
    Dialog for browsing HDF5 or legacy MAT (<v7.3) files:
    lets the user pick the cube path, wavelength path, and metadata path.
    """
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(f"HDF5 / MAT Browser — {os.path.basename(filepath)}")
        # ensure 3 columns are visible
        self.treeWidget.setColumnCount(3)
        self.treeWidget.setHeaderLabels(['Name', 'Type', 'Path'])
        self.filepath = filepath

        # Connect buttons
        self.btn_select_cube.clicked.connect(lambda: self._assign(self.le_cube))
        self.btn_select_wl.clicked.connect(lambda: self._assign(self.le_wl))
        self.btn_select_meta.clicked.connect(lambda: self._assign(self.le_meta))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # Populate the tree
        self._load_file()

    def _is_hdf5_file(self, path: str) -> bool:
        """True if `path` can be opened by h5py (including .mat v7.3)."""
        try:
            with h5py.File(path, 'r'):
                return True
        except (OSError, IOError):
            return False

    def _load_file(self):
        """Fill the QTreeWidget with either legacy-MAT variables or HDF5 contents."""
        self.treeWidget.clear()
        root = self.treeWidget.invisibleRootItem()
        ext = os.path.splitext(self.filepath)[1].lower()

        is_mat     = (ext == '.mat')
        is_hdf5    = (ext in ('.h5', '.hdf5')) or (is_mat and self._is_hdf5_file(self.filepath))

        print(f'is_hdf5 ?: {is_hdf5}')
        print(f'is_mat ?: {is_mat}')

        if is_mat and not is_hdf5:
            # Legacy MATLAB (< v7.3) via mat4py
            mat_dict = loadmat(self.filepath)
            for name, val in mat_dict.items():
                # skip Python‐side globals
                # if name.startswith('__'):
                #     continue
                self._add_mat_node(name, val, root, name)
        else:
            # HDF5 or MATLAB v7.3 via h5py
            with h5py.File(self.filepath, 'r') as f:
                def recurse(group, parent, path):
                    # show attributes
                    if group.attrs:
                        attrs_node = QTreeWidgetItem(['<Attributes>', 'Group', path or '/'])
                        parent.addChild(attrs_node)
                        for key in group.attrs:
                            attrs_node.addChild(
                                QTreeWidgetItem([key, 'Attribute', f"{path}@{key}"])
                            )
                    # show groups/datasets
                    for name, obj in group.items():
                        full  = f"{path}/{name}" if path else name
                        kind  = 'Group'   if isinstance(obj, h5py.Group) else 'Dataset'
                        node  = QTreeWidgetItem([name, kind, full])
                        parent.addChild(node)
                        if isinstance(obj, h5py.Group):
                            recurse(obj, node, full)

                recurse(f, root, "")

        # expand everything so you immediately see the branches
        self.treeWidget.expandAll()

    def _add_mat_node(self, name, val, parent, path):
        """
        Recursively add legacy-MAT nodes:
          • Struct → dive fields
          • Variable → leaf
        """
        if isinstance(val, dict):
            # mat4py represents structs as Python dicts
            node = QTreeWidgetItem([name, 'Struct', path])
            parent.addChild(node)
            for field, fldval in val.items():
                self._add_mat_node(field, fldval, node, f"{path}/{field}")
        else:
            # list, number, etc.
            node = QTreeWidgetItem([name, 'Variable', path])
            parent.addChild(node)

    def _assign(self, line_edit):
        """Copy the selected item's Path into the given QLineEdit."""
        item = self.treeWidget.currentItem()
        if item:
            line_edit.setText(item.text(2))

    def get_selection(self):
        """Return {'cube_path','wl_path','meta_path'} (or None)."""
        return {
            'cube_path': self.le_cube.text() or None,
            'wl_path':   self.le_wl.text()   or None,
            'meta_path': self.le_meta.text() or None,
        }

class Hypercube:
    """
    Hyperspectral cube loader supporting:
      • Legacy .mat (<v7.3) via mat4py.loadmat
      • HDF5 (.h5/.hdf5 or .mat v7.3) via h5py
      • ENVI (.hdr + raw) via spectral.io.envi
    """
    def __init__(self, filepath=None, data=None, wl=None, metadata=None, load_init=False):
        self.filepath = filepath
        self.data     = data
        self.wl       = wl
        self.metadata = metadata or {}

        if load_init:
            if self.filepath:
                self.open_hyp(self.filepath, open_dialog=False)
            else:
                self.open_hyp(open_dialog=True)

    @staticmethod
    def _is_hdf5_file(path: str) -> bool:
        """True if `path` can be opened by h5py (including .mat v7.3)."""
        try:
            with h5py.File(path, 'r'):
                return True
        except (OSError, IOError):
            return False

    def reinit_cube(self):
        """Reset all attributes."""
        self.filepath = None
        self.data     = None
        self.wl       = None
        self.metadata = {}

    def get_rgb_image(self, indices):
        """Return an RGB composite from three-band indices, or None."""
        if self.data is None:
            return None
        return self.data[:, :, indices]

    def open_hyp(self, default_path="", open_dialog=True, show_exception=True):
        """
        Open a hyperspectral cube file:
          • .mat or .h5/.hdf5 → browser dialog to pick cube/wl/metadata
          • .hdr → ENVI reader
        """
        # 1) File selection
        if open_dialog:
            app = QApplication.instance() or QApplication([])
            filepath, _ = QFileDialog.getOpenFileName(
                None, "Open Hyperspectral Cube", default_path,
                "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not filepath:
                return
            self.filepath = filepath
        else:
            filepath = default_path

        ext = os.path.splitext(filepath)[1].lower()

        # 1) Try your old “automatic” loaders first
        if ext == ".mat":
            try:
                # v7.3 .mat are HDF5 → this will fail for legacy .mat
                with h5py.File(filepath, "r") as f:
                    # your hard‐coded layout
                    self.wl = f["#refs#/d"][:].flatten()
                    raw = f["#refs#/c"][:]
                    self.data = np.transpose(raw, (2, 1, 0))
                    # rebuild metadata if any
                    self.metadata = {}
                    if "Metadata" in f:
                        grp = f["Metadata"]
                        for k, ds in grp.items():
                            self.metadata[k] = ds[()]
                    return  # success → no dialog

            except Exception:
                # either not HDF5 at all, or missing your hard‐coded paths
                pass

        elif ext in (".h5", ".hdf5"):
            try:
                with h5py.File(filepath, "r") as f:
                    raw = f["DataCube"][:]
                    self.data = np.transpose(raw, (2, 1, 0))
                    self.metadata = dict(f.attrs)
                    self.wl = self.metadata.get("wl")
                    return  # success → no dialog

            except Exception:
                pass

        # 2) If we get here, automatic failed → show browser dialog
        if ext in (".mat", ".h5", ".hdf5"):
            try:
                app = QApplication.instance() or QApplication([])
                dlg = HDF5BrowserDialog(filepath)
                if dlg.exec_() != QDialog.Accepted:
                    self.reinit_cube()
                    return

                sel = dlg.get_selection()
                try:
                    mat_dict = loadmat(filepath)

                    # data cube
                    arr = mat_dict.get(sel['cube_path'])
                    if arr is None:
                        QMessageBox.critical(None, "Error",
                                             f"Variable '{sel['cube_path']}' not found.")
                        self.reinit_cube()
                        return
                    data_arr = np.array(arr)
                    if data_arr.ndim != 3:
                        QMessageBox.critical(None, "Error",
                                             f"Expected 3D array, got shape {data_arr.shape}.")
                        self.reinit_cube()
                        return
                    self.data = np.transpose(data_arr, (2, 1, 0))

                    # wavelengths
                    if sel['wl_path']:
                        wl_arr = mat_dict.get(sel['wl_path'])
                        self.wl = np.array(wl_arr).flatten() if wl_arr is not None else None
                    else:
                        self.wl = None

                    # metadata
                    self.metadata = {}
                    if sel['meta_path']:
                        meta_val = mat_dict.get(sel['meta_path'])
                        if meta_val is not None:
                            key = sel['meta_path'].split('/')[-1]
                            self.metadata[key] = meta_val

                except:
                    #HDF5 or MAT v7.3
                    with h5py.File(filepath, 'r') as f:
                        raw = f[sel['cube_path']][:]
                        self.data = np.transpose(raw, (2, 1, 0))

                        # wavelengths
                        wl_sel = sel.get('wl_path') or ""
                        if wl_sel:
                            if '@' in wl_sel:
                                grp, _, attr = wl_sel.rpartition('@')
                                vals = (f.attrs[attr] if not grp or grp == '/'
                                        else f[grp].attrs[attr])
                            else:
                                vals = f[wl_sel][:]
                            self.wl = np.array(vals).flatten()
                        else:
                            self.wl = None

                        # metadata
                        self.metadata = {}
                        meta_sel = sel.get('meta_path') or ""
                        if meta_sel == '/':
                            self.metadata = {k: f.attrs[k] for k in f.attrs}
                        elif '@' in meta_sel:
                            grp, _, attr = meta_sel.rpartition('@')
                            self.metadata[attr] = (
                                f.attrs[attr] if not grp or grp == '/'
                                else f[grp].attrs[attr]
                            )
                        elif meta_sel:
                            obj = f[meta_sel]
                            if isinstance(obj, h5py.Group):
                                for k, ds in obj.items():
                                    self.metadata[k] = ds[()]
                            else:
                                name = meta_sel.split('/')[-1]
                                self.metadata[name] = obj[()]

                return

            except Exception as e:
                if show_exception:
                    QMessageBox.critical(None, "Error", f"Failed to read file: {e}")
                self.reinit_cube()
                return

        # 3) ENVI (.hdr + raw)
        elif ext == '.hdr':
            try:
                img = envi.open(filepath)
                self.data = img.load().astype(np.float32)
                self.metadata = img.metadata.copy()

                wl = self.metadata.get('wavelength', self.metadata.get('wavelengths'))
                if isinstance(wl, str):
                    wl_list = wl.strip('{}').split(',')
                    self.wl = np.array(wl_list, dtype=np.float32)
                else:
                    self.wl = np.array(wl, dtype=np.float32) if wl is not None else None

                return

            except Exception as e:
                if show_exception:
                    QMessageBox.critical(
                        None, "Error",
                        f"Failed to read ENVI file:\n{e}"
                    )
                self.reinit_cube()
                return

        # 4) Unsupported
        else:
            if show_exception:
                QMessageBox.critical(
                    None, "Unsupported Format",
                    f"The extension '{ext}' is not supported."
                )
            self.reinit_cube()
            return

    def save(self,filepath=None,fmt=None):
        filters = (
            "HDF5 files (*.h5);;"
            "MATLAB files (*.mat);;"
            "ENVI header (*.hdr)"
        )
        default_filter = "HDF5 files (*.h5)"

        if filepath is None:
            app = QApplication.instance() or QApplication([])
            filepath,selected_filter= QFileDialog.getSaveFileName(
                parent=None,
                caption="Save cube As…",
                filter=filters,
                initialFilter=default_filter
            )

            if filepath is None:
                print('Problem getting filepath to save')
                return

            sel = selected_filter.lower()
            if "hdf5" in sel:
                fmt = 'HDF5'
                if not filepath.lower().endswith(".h5"):
                    filepath += ".h5"

            elif "matlab" in sel:
                fmt = 'MATLAB'
                if not filepath.lower().endswith(".mat"):
                    filepath += ".mat"
            elif "envi" in sel:
                fmt = 'ENVI'
                # pour ENVI, on sauvegarde l'en-tête (.hdr) ; l'utilisateur devra
                # gérer son fichier raw associé (.img ou .dat)
                if not filepath.lower().endswith(".hdr"):
                    filepath += ".hdr"


        if fmt=='HDF5':
            self.save_hdf5_cube(filepath)

        elif fmt=='MATLAB':
            self.save_matlab_cube(filepath)

        elif fmt=='ENVI':
            self.save_envi_cube(filepath)


    def save_hdf5_cube(self,filepath: str):
        with h5py.File(filepath+'.h5', "w") as f:
            f.create_dataset("DataCube", data=self.data.transpose(2,1,0))
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
        tosave = {
            'DataCube': self.data,
            'wl': self.wl,
            'metadata': self.metadata or {}
        }
        savemat(filepath, tosave)

class SaveWindow(QDialog, Ui_Save_Window):
    """Dialog to configure saving options."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.pushButton_save_cube_final.clicked.connect(self.accept)
        self.pushButton_Cancel.clicked.connect(self.reject)

    def closeEvent(self, event):
        self.reject()
        super().closeEvent(event)

    def get_options(self):
        opts = {
            'cube_format':   self.comboBox_cube_format.currentText(),
            'save_both':     self.radioButton_both_cube_save.isChecked(),
            'crop_cube':     self.checkBox_minicube_save.isChecked(),
            'export_images': self.checkBox_export_images.isChecked(),
        }
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
                image_format: str = 'png',
                rgb: bool = False):
    """Save 'fixed' and 'aligned' images side by side."""
    os.makedirs(dirpath, exist_ok=True)
    ext = image_format.lower()

    def write(name, img):
        out = os.path.join(dirpath, f"{name}.{ext}")
        if img.ndim == 2 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out, img)

    write('fixed', fixed_img)
    write('aligned', aligned_img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    # Example usage:
    # sample   = '00001-VNIR-mock-up.mat'
    # folder   = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Samples\minicubes'
    sample = 'jabon_guillermo_final.mat'
    folder = r'C:\Users\Usuario\Downloads'
    filepath = os.path.join(folder, sample)
    filepath = None  # force dialog

    try:
        cube = Hypercube(filepath, load_init=True)
        if cube.data is not None:
            print('cube.data is not None')
            rgb_img = cube.get_rgb_image([50, 30, 10])
            plt.figure()
            plt.imshow(rgb_img / np.max(rgb_img))
            plt.axis('off')
            plt.show()
    except Exception:
        print("Failed to load or display the hyperspectral cube.")
