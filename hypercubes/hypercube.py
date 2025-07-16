import os
from copy import deepcopy

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
import copy

from pathlib import Path

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
            return NotImplemented
        return Path(self.filepath).resolve() == Path(other.filepath).resolve()

    def __hash__(self):
        # if in dic or set
        return hash(Path(self.filepath).resolve())

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

        if filepath is not None:
            self.cube_info.filepath=filepath

        if load_init:
            if filepath is not None:
                self.open_hyp(default_path=filepath, open_dialog=False)
            else:
                self.open_hyp(open_dialog=True)

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

    def get_data_mat7p3_or_h5(self,filepath=None):
        if filepath is None:
            try :
                filepath=self.cube_info.filepath
            except:
                pass

        with h5py.File(filepath, "r") as f:
            raw = f[self.cube_info.data_path][:]
            self.data = np.transpose(raw, (2, 1, 0))

            wl_sel = self.cube_info.wl_path
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

            self.metadata = {}
            meta_sel = self.cube_info.metadata_path
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
                    self.meta_from_gnl_v7p3(obj)

                else:
                    name = meta_sel.split('/')[-1]
                    self.metadata[name] = obj[()]

    def meta_from_gnl_v7p3(self,meta_grp):
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

    def open_hyp(self, default_path="", open_dialog=True, cube_info=None, show_exception=True):
        """
        Open a hyperspectral cube file: try with .h5 . mat with different methods and with .hdr for ENVI.
        """

        if cube_info is not None:
            self.cube_info=cube_info

        flag_loaded=False

        if QApplication.instance() is None: # to open Qt app if used as main without other Qapp opened
            self._qt_app = QApplication([])

        if open_dialog:
            app = QApplication.instance() or QApplication([])
            filepath, _ = QFileDialog.getOpenFileName(
                None, "Open Hyperspectral Cube", default_path,
                "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not filepath:
                return

        else:
            filepath = default_path

        self.filepath = filepath

        ext = os.path.splitext(filepath)[1].lower()

        #test with matlab
        if ext == ".mat":
            is_v7p3=False
            try:
                with h5py.File(filepath, "r") as f:
                    is_v7p3 = True
            except:
                pass
            print(f'.mat is v7.3 : {is_v7p3}')

            if is_v7p3 :  # new format
               try:#automatic hypercube_classic_save of CIL
                    with h5py.File(filepath, "r") as f:

                        self.wl = f["#refs#/d"][:].flatten()
                        raw = f["#refs#/c"][:]
                        self.data = np.transpose(raw, (2, 1, 0))
                        # rebuild metadata if any
                        self.metadata = {}
                        if "Metadata" in f:
                            grp=f["Metadata"]
                            self.meta_from_gnl_v7p3(grp)

                        self.cube_info.data_path = "#refs#/d"
                        self.cube_info.wl_path = "#refs#/c"
                        self.cube_info.metadata_path = "Metadata"
                        self.cube_info.wl_trans = True
                        flag_loaded=True  # success
                        print('loaded .mat v7p3 with classic hypercube look')
               except Exception:
                   pass

               if not flag_loaded:
                   try:
                       if self.cube_info.data_path is None :
                           print('try wih browser')
                           ans=self.look_with_browser(filepath)
                           print(ans)
                           if not ans:
                               return

                       try:
                            self.get_data_mat7p3_or_h5(filepath)
                            self.cube_info.wl_trans = True
                            flag_loaded=True
                            print('loaded .mat v7p3 fro known cube_info.pathS')
                       except:
                           pass
                   except:
                       pass

            else :
                # with old format classic save
                try :
                    mat_dict = loadmat(filepath)
                    self.data=mat_dict['DataCube']
                    self.wl=mat_dict['wl']
                    try :
                        metadata_struct=mat_dict['metadata'][0]
                        self.metadata={name: metadata_struct[name][0] for name in metadata_struct.dtype.names}
                    except:
                        pass

                    self.cube_info.wl_trans = True

                    flag_loaded=True
                    print('Loaded .mat NO v7p3 with classic')
                except:
                    pass

                if not flag_loaded:
                    if self.look_with_browser(filepath=filepath):

                        try:
                            mat_dict = loadmat(filepath)
                            # data cube
                            arr = mat_dict[self.cube_info.data_path]
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
                                self.data = data_arr

                            # wavelengths
                            if self.cube_info.wl_path:
                                wl_arr = mat_dict[self.cube_info.wl_path]
                                self.wl = np.array(wl_arr).flatten() if wl_arr is not None else None
                            else:
                                self.wl = None

                            # metadata
                            self.metadata = {}
                            if self.cube_info.metadata_path:
                                try:
                                    metadata_struct = mat_dict[self.cube_info.metadata_path][0]
                                    self.metadata = {name: metadata_struct[name][0] for name in
                                                     metadata_struct.dtype.names}
                                except:
                                    pass

                            print('loaded .mat NO v7p3 with BROWSER')
                            flag_loaded=True
                        except:
                            pass

            if not flag_loaded:
                # try:
                #     self.look_with_matlab_engine(filepath)
                #     flag_loaded=True
                # except:
                #     print('Matlab.engine loading failed')

                if not flag_loaded:
                    try:
                        from hypercubes.matlab_subprocess import load_mat_file_with_matlab_obj

                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Information)
                        msg_box.setWindowTitle("Loading Cube via MATLAB")
                        msg_box.setText(
                            "MATLAB is running in the background and may open a console.\n\n"
                            "This may take up to one minute. Please wait..."
                        )
                        msg_box.setStandardButtons(QMessageBox.Ok)
                        msg_box.show()
                        from PyQt5.QtCore import QTimer
                        QTimer.singleShot(10000, msg_box.close) #close after a time in ms

                        data, wl, meta = load_mat_file_with_matlab_obj(filepath)
                        self.data = data
                        self.wl = wl
                        self.metadata = meta

                        flag_loaded = True

                    except Exception as e:
                        if show_exception:
                            QMessageBox.critical(None, "MATLAB Subprocess Error",
                                                 f"MATLAB Subprocess loading failed:\n{e}")

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
                    self.cube_info.metadata_temp = copy.deepcopy(self.metadata)
                    self.cube_info.wl_trans = True
                    self.cube_info.filepath=filepath

                    if self.data is not None:
                        self.cube_info.data_shape = self.data.shape

                flag_loaded=True  # success → no dialog
                print('.h5 loaded from classic')

            except Exception:
                pass

            if not flag_loaded:

                if self.look_with_browser(filepath=filepath):
                    try:
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
                        print('loaded .h5 from browser')
                    except:
                        pass

        # 3) ENVI (.hdr + raw)
        # TODO : cube info fill with ENVI files
        elif ext == '.hdr':
            try:
                img = envi.open(filepath)
                self.data = img.load().astype(np.float32)
                self.metadata = copy.deepcopy(img.metadata)

                wl = self.metadata.get('wavelength')
                if wl is None:
                    wl=self.metadata.get('wl')

                if isinstance(wl, str):
                    wl_list = wl.strip('{}').split(',')
                    self.wl = np.array(wl_list, dtype=np.float32)
                else:
                    self.wl = np.array(wl, dtype=np.float32) if wl is not None else None

                flag_loaded=True

            except Exception as e:
                if show_exception:
                    QMessageBox.critical(
                        None, "Error",
                        f"Failed to read ENVI file:\n{e}"
                    )
                self.reinit_cube()

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
            metadata_raw=copy.deepcopy(self.metadata)
            self.metadata = {k: self.simplify_metadata_value(v) for k, v in metadata_raw.items()}
            self.cube_info.metadata_temp = copy.deepcopy(self.metadata)
            self.cube_info.metadata_temp['wl'] = self.wl
            self.cube_info.filepath = filepath
            if self.data is not None:
                self.cube_info.data_shape = self.data.shape

            if 'name' not in self.cube_info.metadata_temp:
                self.cube_info.metadata_temp['name']=self.filepath.split('/')[-1].split('.')[0]

    def look_with_matlab_engine(self,filepath=None):
        from hypercubes.matlab_engine_control import get_matlab_engine, load_mat_file_with_engine

        get_matlab_engine()

        try:
            data, wl, meta = load_mat_file_with_engine(filepath)
            self.data = data
            self.wl = wl
            self.metadata = meta
            self.cube_info.data_shape = self.data.shape
            self.cube_info.filepath = filepath
            self.cube_info.metadata_temp = copy.deepcopy(self.metadata)
            flag_loaded = True

        except Exception as e:
            QMessageBox.critical(None, "MATLAB Engine Error",
                                     f"MATLAB Engine loading cube failed :\n{e}")

    def simplify_metadata_value(self,value):
        # if scalar numpy ->  Python type
        if isinstance(value, np.generic):
            return value.item()

        elif isinstance(value, np.ndarray):
            if value.shape == ():  #  pur scalar
                return self.simplify_metadata_value(value.item())
            elif value.size == 1 :
                return self.simplify_metadata_value(value.item())
            else:
                return value

        # if not scalar numpy or line numpy array
        return value

    def look_with_browser(self,filepath=None):
        if filepath is None:
            try:
                filepath=self.cube_info.filepath
            except:
                pass

        widget = HDF5BrowserWidget(cube_info=self.cube_info,
                                   filepath=filepath, closable=True)
        loop = QEventLoop()

        widget.accepted.connect(lambda ci: loop.quit())
        widget.rejected.connect(loop.quit)
        widget.show()
        loop.exec_()

        # after loop, check whether user clicked OK
        if not widget._accepted:
            # user cancelled
            self.reinit_cube()
            return False

        return True

    def save(self,filepath=None,fmt=None,meta_from_cube_info=False):
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

            if meta_from_cube_info:
                self.metadata=copy.deepcopy(self.cube_info.metadata_temp)

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

        if fmt is None:
            if filepath.split('.')[-1] in ['h5','h5p']:
                fmt='HDF5'
            elif filepath.split('.')[-1] in ['mat']:
                fmt='MATLAB'
            elif filepath.split('.')[-1] in ['hdr']:
                fmt='ENVI'
            else:
                fmt='HDF5'


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

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        hdr_meta = {
            "lines": self.data.shape[0],
            "samples": self.data.shape[1],
            "bands": self.data.shape[2],
            "data type": dtype_code,
            "interleave": interleave
        }
        if self.metadata is not None:
           hdr_meta.update(self.metadata)
        print(filepath)
        envi.save_image(filepath, self.data, dtype=np.float32, metadata=hdr_meta)

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

    def closeEvent(self, event):
        self._on_reject()
        event.accept()

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

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    # fname = '00189-VNIR-mock-up.h5'

    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Thin_film\CITIC\241216\PikaL'
    fname = 'c1318_12.bil.hdr'

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria'
    # fname = 'MPD41a_VNIR.mat'
    # fname = 'MPD41a_SWIR_cube_NOv7p3.mat'
    # fname = 'MPD41a_SWIR_sruct_NOv7p3.mat'
    # fname = 'MPD41a_SWIR_struct_YESv7p3.mat'

    import os

    filepath = os.path.join(folder, fname)

    app = QApplication(sys.argv)

    cube = Hypercube(filepath=filepath, load_init=True)
    for key in cube.metadata:
        print(f' {key} (type: {type(cube.metadata[key]).__name__})-> {cube.metadata[key]}')

    try:
        import cv2
        mid=int(cube.data.shape[2]/2)
        image_np=cube.data[:,:,mid]
        cv2.imshow("Image", image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print('imshow crashed')
