import os
import h5py
from spectral.io import envi
from scipy.io import loadmat
import cv2

from PyQt5.QtWidgets import (QWidget,
    QApplication, QFileDialog, QMessageBox, QDialog, QTreeWidgetItem
)
from PyQt5.QtCore    import pyqtSignal, QEventLoop
from registration.save_window_register_tool import Ui_Save_Window
from hypercubes.hdf5_browser_tool import Ui_HDF5BrowserWidget
import sys

from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np

# TODO : sortir HDF5BrowserWidget du fichier de la classe hypercube ?
# TODO : si metadata à la racine, alors on garde toute la racine sauf le cube (ou une selection ?)
# todo : before closing or removing cubes, check if modif have been made by comparing cube.cubeInfo.metadataTemp and cube.metadata

@dataclass
class CubeInfoTemp:
    """
    Container for per-cube working data.
    """
    filepath: str = None    #filepath of cube
    data_path: Optional[str] = None # data location in the file
    metadata_path: Optional[str] = None # metadata location in the path
    wl_path: Optional[str] = None #wl location in the path
    metadata_temp: dict = field(default_factory=dict) # all metadatas modified in the app before saving
    data_shape: Optional[Union[List[float], np.ndarray]] = None # cube shape [width, height, bands]
    wl_trans:Optional[str]= None # if need to transpose wl dim from dim 1 to dim 3

    # because only one filepath for one cube...and one cube for one filepath, let's define the cubeInfo equality
    def __eq__(self, other):
        if not isinstance(other, CubeInfoTemp):
            return False
        return self.filepath == other.filepath

class Hypercube:

    """
    Hyperspectral cube loader for format :
    MATLAB .mat (<v7.3) via scipy.io.loadmat
    HDF5 (.h5/.hdf5 or .mat v7.3) via h5py
    ENVI (.hdr + raw) via spectral.io.envi
    """

    def __init__(self, filepath=None, data=None, wl=None, metadata=None, load_init=False,cube_info=None):
        self.data     = data
        self.wl       = wl
        self.metadata = metadata or {}

        if cube_info is None:
            cube_info = CubeInfoTemp()
        self.cube_info = cube_info

        if self.cube_info.filepath is None:
            self.cube_info.filepath=filepath

        if load_init:
            if filepath is not None:
                self.open_hyp(default_path=filepath, open_dialog=False)
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

        flag_loaded=False

        if QApplication.instance() is None: # to open Qt app if used as main without other Qapp opened
            self._qt_app = QApplication([])

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

        self.filepath = filepath

        ext = os.path.splitext(filepath)[1].lower()

        #test with matlab new format
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

                    self.cube_info.data_path = "#refs#/d"
                    self.cube_info.wl_path = "#refs#/c"
                    self.cube_info.metadata_path = "Metadata"
                    self.cube_info.metadata_temp = self.metadata.copy()
                    self.cube_info.wl_dim = True
                    if self.data is not None:
                        self.cube_info.data_shape = self.data.shape

                flag_loaded=True  # success → no dialog

            except Exception:
                pass

        elif ext in (".h5", ".hdf5"):
            try:
                with h5py.File(filepath, "r") as f:
                    raw = f["DataCube"][:]
                    self.data = np.transpose(raw, (2, 1, 0))
                    self.metadata = dict(f.attrs)
                    self.wl = self.metadata.get("wl")

                    self.cube_info.data_path = "DataCube"
                    self.cube_info.wl_path = "@wl"
                    self.cube_info.metadata_path = "/"
                    self.cube_info.metadata_temp = self.metadata.copy()
                    self.cube_info.wl_trans = True
                    if self.data is not None:
                        self.cube_info.data_shape = self.data.shape

                flag_loaded=True  # success → no dialog

            except Exception:
                pass


        # 2) If we get here, automatic failed → show browser dialog
        # TODO : remember choice made by user for future loading ?

        if ext in (".mat", ".h5", ".hdf5") and not flag_loaded:

            try:

                widget = HDF5BrowserWidget(cube_info=self.cube_info,
                                           filepath = filepath,closable = True)
                loop = QEventLoop()

                widget.accepted.connect(lambda ci: loop.quit())
                widget.rejected.connect(loop.quit)
                widget.show()
                loop.exec_()

                # after loop, check whether user clicked OK
                if not widget._accepted:
                    # user cancelled
                    self.reinit_cube()
                    return

                # else retrieve their paths

                try:
                    mat_dict = loadmat(filepath)

                    # data cube
                    arr = mat_dict.get(self.cube_info.data_path)
                    if arr is None:
                        QMessageBox.critical(None, "Error",
                                             f"Variable '{self.cube_info.data_path}' not found.")
                        self.reinit_cube()
                        return
                    data_arr = np.array(arr)
                    if data_arr.ndim != 3:
                        QMessageBox.critical(None, "Error",
                                             f"Expected 3D array, got shape {data_arr.shape}.")
                        self.reinit_cube()
                        return

                    if self.cube_info.wl_trans:
                        self.data = np.transpose(data_arr, (2, 1, 0))
                    else:
                        self.data=data_arr

                    # wavelengths
                    if self.cube_info.wl_path:
                        wl_arr = mat_dict.get(self.cube_info.wl_path)
                        self.wl = np.array(wl_arr).flatten() if wl_arr is not None else None
                    else:
                        self.wl = None

                    # metadata
                    self.metadata = {}
                    if self.cube_info.metadata_path:
                        meta_val = mat_dict.get(self.cube_info.metadata_path)
                        if meta_val is not None:
                            key = sel['meta_path'].split('/')[-1]
                            self.metadata[key] = meta_val

                    if self.data is not None:
                        self.cube_info.data_shape = self.data.shape

                except:
                    #HDF5 or MAT v7.3
                    with h5py.File(filepath, 'r') as f:
                        raw = f[self.cube_info.data_path][:]
                        self.data = np.transpose(raw, (2, 1, 0))

                        # wavelengths
                        wl_sel = self.cube_info.wl_path or ""
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
                        meta_sel = self.cube_info.metadata_path or ""
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

                        if self.data is not None:
                            self.cube_info.data_shape = self.data.shape

                flag_loaded=True

            except Exception as e:
                if show_exception:
                    QMessageBox.critical(None, "Error", f"Failed to read file: {e}")
                self.reinit_cube()
                return

        # 3) ENVI (.hdr + raw)
        # TODO : cube info fill with ENVI files
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
            if not flag_loaded:
                if show_exception:
                    QMessageBox.critical(
                        None, "Unsupported Format",
                        f"The extension '{ext}' is not supported."
                    )
                self.reinit_cube()
                return

        if flag_loaded:
            if 'name' not in self.cube_info.metadata_temp:
                self.cube_info.metadata_temp['name']=self.filepath.split('/')[-1].split('.')[0]

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
            if not filepath.lower().endswith(".h5"):
                filepath += ".h5"
            self.save_hdf5_cube(filepath)

        elif fmt=='MATLAB':
            if not filepath.lower().endswith(".mat"):
                filepath += ".mat"
            self.save_matlab_cube(filepath)

        elif fmt=='ENVI':
            if not filepath.lower().endswith(".hdr"):
                filepath += ".hdr"
            self.save_envi_cube(filepath)

    # def save_hdf5_cube(self, filepath: str):
    #     from PyQt5.QtWidgets import QMessageBox, QCheckBox
    #
    #     full_meta = self.metadata.copy()
    #     filtered_meta = {}
    #     ask_all = False
    #     default_drop = False
    #
    #     with h5py.File(filepath, "w") as f:
    #         # création du dataset principal
    #         f.create_dataset("DataCube", data=self.data.transpose(2, 1, 0))
    #
    #         for key, val in full_meta.items():
    #             print(f'{key}: {val}')
    #
    #         # itération sur les métadonnées
    #         for key, val in full_meta.items():
    #             # si on a déjà coché “Do this for all”, on applique la même décision
    #             if ask_all and default_drop:
    #                 # on drop sans rien écrire
    #                 continue
    #             print(key)
    #
    #             try:
    #                 if key =='wl':
    #                     if len(self.wl)==self.data.shape[2]:
    #                        val=self.wl
    #                 print(val)
    #                 f.attrs[key] = val
    #                 filtered_meta[key] = val
    #
    #             except OSError as e:
    #                 # seulement si c’est bien le header trop gros
    #                 if "object header message is too large" not in str(e):
    #                     raise  # autre erreur → on remonte
    #
    #                 try:
    #                     f.create_dataset(f"Meta_{key}", data=val)
    #                 except Exception:
    #                     pass
    #
    #                 # prompt interactif if we want to choose if new dataset
    #
    #                 # msg = QMessageBox()
    #                 # msg.setIcon(QMessageBox.Question)
    #                 # msg.setWindowTitle("Metadata too large")
    #                 # msg.setText(f"The metadata '{key}' is too large to save in HDF5.")
    #                 # msg.setInformativeText("Do you want to save without this metadata?")
    #                 # yes_btn = msg.addButton("Yes", QMessageBox.AcceptRole)
    #                 # no_btn = msg.addButton("No", QMessageBox.RejectRole)
    #                 # dataset_btn = msg.addButton("Save as dataset", QMessageBox.ActionRole)
    #                 # cancel_btn = msg.addButton("Cancel", QMessageBox.DestructiveRole)
    #                 # cb = QCheckBox("Do this for all")
    #                 # msg.setCheckBox(cb)
    #                 # msg.exec_()
    #
    #                 # # gestion de la réponse
    #                 # if msg.clickedButton() is cancel_btn:
    #                 #     # on annule tout le save
    #                 #     f.close()
    #                 #     os.remove(filepath)  # ou on laisse le fichier incomplet
    #                 #     return
    #                 #
    #                 # if msg.clickedButton() is dataset_btn:
    #                 #     try:
    #                 #         f.create_dataset(f"Meta_{key}", data=val)
    #                 #     except Exception:
    #                 #         pass
    #                 #     filtered_meta[key] = val
    #                 #     if cb.isChecked():
    #                 #         ask_all = True
    #                 #         default_drop = False
    #                 #     continue
    #                 #
    #                 # drop = (msg.clickedButton() is yes_btn)
    #                 # if cb.isChecked():
    #                 #     ask_all = True
    #                 #     default_drop = drop
    #                 # if not drop:
    #                 #     # user chose "Keep as attribute"
    #                 #     try:
    #                 #         f.attrs[key] = val
    #                 #         filtered_meta[key] = val
    #                 #     except:
    #                 #         pass
    #
    #         # Optionnel : mettre à jour self.metadata pour refléter ce qui a été écrit
    #         self.metadata = filtered_meta

    def save_hdf5_cube(self, filepath: str):
        # dtype HDF5 « variable-length UTF-8 string »
        import traceback
        str_dt = h5py.string_dtype(encoding='utf-8')
        filtered_meta = {}

        try:
            with h5py.File(filepath, "w") as f:
                # 1) Enregistrer le cube principal (bandes, lignes, colonnes)
                f.create_dataset("DataCube", data=self.data.transpose(2, 1, 0))

                # 2) Parcourir tous les champs de metadata
                for key, val in self.metadata.items():

                    try:
                        if key == 'wl':
                            if len(self.wl)==self.data.shape[2]:
                                val=self.wl
                                f.attrs[key] = val
                                filtered_meta[key] = val

                        else:
                            if isinstance(val, str):
                                f.attrs.create(key, val, dtype=str_dt)

                            # b) si c'est un np.ndarray de chaînes Unicode ou d’objets
                            elif isinstance(val, np.ndarray) and val.dtype.kind in ('U', 'O'):
                                arr = np.array(val, dtype=object)
                                f.attrs.create(key, arr, dtype=str_dt)

                            # c) tout le reste (scalars, arrays numériques, listes de nombres…)
                            else:
                                f.attrs.create(key, val)

                            filtered_meta[key] = val

                    except Exception as attr_err:
                        try:
                            if isinstance(val, str):
                                f.create_dataset(f"meta/{key}", data=np.array(val, dtype=object), dtype=str_dt)
                            elif isinstance(val, (list, tuple)) and all(isinstance(v, str) for v in val):
                                f.create_dataset(f"meta/{key}", data=np.array(val, dtype=object), dtype=str_dt)
                            else:
                                f.create_dataset(f"meta/{key}", data=val)

                            filtered_meta[key] = val
                        except Exception as ds_err:
                            print(f"[Warning] Impossible to sava metadatum '{key}' "
                                  f"attribute error : {attr_err}\n"
                                  f"dataset error : {ds_err}")
                            traceback.print_exc()

                # 3) Mettre à jour self.metadata pour ne garder que ce qui a été écrit
                self.metadata = filtered_meta

        except Exception as e:
            QMessageBox.critical(
                None,
                "Erreur de sauvegarde",
                f"Échec de l'enregistrement du cube HDF5 : {e}"
            )

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

class HDF5BrowserWidget(QWidget, Ui_HDF5BrowserWidget):
    """
    Embeddable widget for browsing HDF5 / legacy MAT files.
    Emits `accepted` when OK is clicked, `rejected` on Cancel.
    """

    accepted = pyqtSignal(object)
    rejected = pyqtSignal()

    def __init__(self,cube_info: CubeInfoTemp, filepath:Optional[str]=None, parent=None,closable=False):
        super().__init__(parent)
        self.setupUi(self)

        # attributes
        self.filepath = filepath
        self.cube_info  = cube_info
        self._accepted = False
        self.closable=closable

        # tree
        self.treeWidget.setColumnCount(4)
        self.treeWidget.setHeaderLabels(['Name', 'Type', 'Path', 'Size'])

        # Connect buttons
        self.btn_select_cube.clicked.connect(lambda: self._assign(self.le_cube))
        self.btn_select_wl.clicked.connect(lambda: self._assign(self.le_wl))
        self.btn_select_meta.clicked.connect(lambda: self._assign(self.le_meta))
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self._on_reject)

        # Populate the tree
        if filepath is not None:
            self._load_file()

        # 2) Prérémplir depuis cube_info
        if cube_info.data_path:
            self.le_cube.setText(cube_info.data_path)
        if cube_info.wl_path:
            self.le_wl.setText(cube_info.wl_path)
        if cube_info.metadata_path:
            self.le_meta.setText(cube_info.metadata_path)

    def _is_hdf5_file(self, path: str) -> bool:
        """True if `path` can be opened by h5py (including .mat v7.3)."""
        try:
            with h5py.File(path, 'r'):
                return True
        except (OSError, IOError):
            return False

    def _update(self,path=None):
        if path is not None:
            self.filepath=path
            self._load_file()

    def _load_file(self):
        """Fill the QTreeWidget with either legacy-MAT variables or HDF5 contents."""
        self.treeWidget.clear()
        root = self.treeWidget.invisibleRootItem()
        ext = os.path.splitext(self.filepath)[1].lower()
        self.label_filename.setText(self.filepath.split('/')[-1])

        is_mat     = (ext == '.mat')
        is_hdf5    = (ext in ('.h5', '.hdf5')) or (is_mat and self._is_hdf5_file(self.filepath))

        if is_mat and not is_hdf5:
            # Legacy MATLAB (< v7.3) via mat4py
            mat_dict = loadmat(self.filepath)
            for name, val in mat_dict.items():
                # skip Python‐side globals
                if name.startswith('__'):
                    continue
                self._add_mat_node(name, val, root, name)
        else:
            # HDF5 or MATLAB v7.3 via h5py
            with h5py.File(self.filepath, 'r') as f:
                def recurse(group, parent, path):
                    # show attributes
                    if group.attrs:
                        attrs_node = QTreeWidgetItem(['<Attributes>', 'Group', path or '/', ''])
                        parent.addChild(attrs_node)

                        for key, val in group.attrs.items():
                            size = self._format_bytes(self._get_attr_size(val))
                            attrs_node.addChild(
                                QTreeWidgetItem([key, 'Attribute', f"{path}@{key}", size])
                            )
                    # show groups/datasets
                    for name, obj in group.items():
                        full  = f"{path}/{name}" if path else name
                        kind  = 'Group'   if isinstance(obj, h5py.Group) else 'Dataset'
                        raw_size = self._get_obj_size(obj)
                        size = self._format_bytes(raw_size)
                        node = QTreeWidgetItem([name, kind, full, size])
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
            node = QTreeWidgetItem([name, 'Struct', path, ''])
            parent.addChild(node)
            for field, fldval in val.items():
                self._add_mat_node(field, fldval, node, f"{path}/{field}")
        else:
            size = self._format_bytes(self._get_attr_size(val))
            node = QTreeWidgetItem([name, 'Variable', path, size])
            parent.addChild(node)

    def _assign(self, line_edit):
        """Copy the selected item's Path into the given QLineEdit."""
        item = self.treeWidget.currentItem()
        if item:
            line_edit.setText(item.text(2))

    def _on_accept(self):
        # Met à jour directement l'objet dataclass
        self.cube_info.data_path = self.le_cube.text().strip() or None
        self.cube_info.wl_path = self.le_wl.text().strip() or None
        self.cube_info.metadata_path = self.le_meta.text().strip() or None
        self.cube_info.wl_trans = self.comboBox_channel_wl.currentText()=='First'

        self._accepted = True

        self.accepted.emit(self.cube_info)

        if self.closable:
            self.close()

    def _on_reject(self):
        self._accepted = False
        self.rejected.emit()
        if self.closable:
            self.close()

    def load_file(self, filepath):
        """
        Point the browser at a new file and rebuild the tree.
        """
        self.filepath = filepath
        self.le_cube.clear()
        self.le_wl.clear()
        self.le_meta.clear()
        self._load_file()

    def _format_bytes(self, num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi']:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def _get_obj_size(self, obj):
        """Storage size for Group or Dataset via HDF5 API."""
        try:
            return obj.id.get_storage_size()
        except Exception:
            return 0

    def _get_attr_size(self, val):
        """Approximate byte-size for an HDF5 attribute value."""
        try:
            if isinstance(val, np.ndarray):
                return val.nbytes
            if isinstance(val, (bytes, bytearray)):
                return len(val)
            if isinstance(val, str):
                return len(val.encode('utf-8'))
            # pour int, float, bool, listes Python...
            return sys.getsizeof(val)
        except Exception:
            return 0

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
    import matplotlib
    matplotlib.use('Qt5Agg')  # backend matplotlib → Qt5
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    # Example usage:

    folder=r'C:\Users\Usuario\Documents\GitHub\Hypertool\metadata/'
    file_name='model.h5'
    filepath = folder + file_name

    try:
        cube = Hypercube(filepath=filepath, load_init=True)
        if cube.data is not None:
            print('cube.data is not None')
            rgb_img = cube.get_rgb_image([50, 30, 10])
            plt.figure()
            plt.imshow(rgb_img / np.max(rgb_img))
            plt.axis('off')
            plt.show()
            for key in cube.metadata:
                if key not in ['GTLabels','gtlabels','pixels_averaged','position','wl','GT_cmap','spectra_mean','spectra_std']:
                    print(f'{key} : {cube.metadata[key]}')

    except Exception:
        print("Failed to load or display the hyperspectral cube.")
