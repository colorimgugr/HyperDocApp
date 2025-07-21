import os
import copy
import sys

from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

import h5py
import numpy as np
from spectral.io import envi
from scipy.io import loadmat

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QWidget,QSizePolicy,
    QApplication, QFileDialog, QMessageBox, QDialog, QTreeWidgetItem,QVBoxLayout,QHBoxLayout,
)
from PyQt5.QtCore   import pyqtSignal, QEventLoop, QRectF,QTimer

from hypercubes.hdf5_browser_tool import Ui_HDF5BrowserWidget
from hypercubes.white_calibration_window import Ui_dialog_white_calibration
from interface.some_widget_for_interface import*

@dataclass
class CubeInfoTemp:
    """
    Container for per-cube working data.
    """
    _filepath: str = None    #filepath of cube
    data_path: Optional[str] = None # data location in the file
    metadata_path: Optional[str] = None # metadata location in the path
    wl_path: Optional[str] = None #wl location in the path
    metadata_temp: dict = field(default_factory=dict) # all metadatas modified in the app before saving
    data_shape: Optional[Union[List[float], np.ndarray]] = None # cube shape [width, height, bands]
    wl_trans:Optional[str]= None # if need to transpose wl dim from dim 1 to dim 3

    @property
    def filepath(self):
        return self._filepath

    # protect filepath modification
    @filepath.setter
    def filepath(self, val):
        print(f"[DEBUG] Changing filepath from {self._filepath} to {val}")
        self._filepath = val

    # because only one filepath for one cube...and one cube for one filepath, let's define the cubeInfo equality
    def __eq__(self, other):
        if not isinstance(other, CubeInfoTemp):
            return NotImplemented
        return Path(self.filepath).resolve() == Path(other.filepath).resolve()

    def __hash__(self):
        # if in dic or set
        return hash(Path(self.filepath).resolve())

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.fig.tight_layout()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.draw()
        super().resizeEvent(event)

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

        self.save_calib=False # for hdr files

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

    def open_hyp(self, default_path="", open_dialog=True, cube_info=None, show_exception=True,ask_calib=True):
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

                ans=QMessageBox.question(None,'Try with installed MATLAB','Hyperdoc Tool can try to open the cube using your Matlab installed remotely. \nBut it can take up to one minute.\nDo you want to test ?', QMessageBox.Yes|QMessageBox.No)
                if ans!=QMessageBox.Yes:
                    return
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

                try:
                    self.metadata['reflectance_data_from']
                except:
                    if ask_calib:
                        ans=QMessageBox.question(None,'Calibration ','Do you need to process the white calibration ?',QMessageBox.Yes | QMessageBox.No)
                        if ans==QMessageBox.Yes:
                            self.calibrating_from_image_extract()

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

            if self.save_calib:
                self.save(filepath=filepath,ask_newfilename=True)

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

    def save(self,filepath=None,fmt=None,meta_from_cube_info=False,ask_newfilename=False):
        filters = (
            "Supported files (*.h5 *.mat *.hdr);;"
            "HDF5 files (*.h5);;"
            "MATLAB files (*.mat);;"
            "ENVI header (*.hdr)"
        )
        default_filter = "Supported files (*.h5 *.mat *.hdr);;"

        if filepath is None or ask_newfilename:
            app = QApplication.instance() or QApplication([])
            filepath,selected_filter= QFileDialog.getSaveFileName(
                parent=None,
                caption="Save cube As…",
                directory=filepath,
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
        envi.save_image(filepath, self.data, dtype=np.float32, metadata=hdr_meta,force=True)

    def save_matlab_cube(self,filepath: str):
        from scipy.io import savemat
        tosave = {
            'DataCube': self.data,
            'wl': self.wl,
            'metadata': self.metadata or {}
        }
        savemat(filepath, tosave)

    def calibrating_from_image_extract(self):
        """
            Opens an interactive white calibration dialog.
            Allows manual region selection or loading an external white reference.
            Calibration is applied according to chosen mode and reference.
        """
        class WhiteCalibrationDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.ui = Ui_dialog_white_calibration()
                self.ui.setupUi(self)
                self.cube_calib=None
                self.other_white_capture=False # if other file loaded.

                # Add ZoomableGraphicsView into frame_image
                self.viewer = ZoomableGraphicsView()
                layout = QVBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.viewer)
                self.ui.frame_image.setLayout(layout)

                # Add matplotlib for quick white spectrum visualisation
                self.canvas_white = MatplotlibCanvas(parent=self.ui.frame_spectra_white)
                layout = QVBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.canvas_white)
                # self.canvas_white.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.ui.frame_spectra_white.setLayout(layout)

                self.ui.comboBox_white_Reference_choice.currentIndexChanged.connect(self.toggle_manual_value_fields)
                self.toggle_manual_value_fields()

                self.viewer.selectionChanged.connect(self.update_selection_overlay)

                self.ui.pushButton_load_white_capture.clicked.connect(self.load_other_white)
                self.ui.pushButton_valid_calibration.clicked.connect(self.accept)
                self.ui.pushButton_load_personal_white_ref.clicked.connect(self.load_white_corection_file)
                self.ui.doubleSpinBox_R_manual.valueChanged.connect(self.update_white_spectrum)

                self.ui.radioButton_horizontal_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_vertical_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_full_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_no_flat_field.toggled.connect(self.update_selection_overlay)

                QTimer.singleShot(0, self.update_white_spectrum) # to force spectra draw after layout processed

            def get_selected_rect(self):
                return self.viewer.get_rect_coords()

            def get_calibration_mode(self):
                if self.ui.radioButton_horizontal_flat_field.isChecked():
                    return "horizontal"
                elif self.ui.radioButton_vertical_flat_field.isChecked():
                    return "vertical"
                elif self.ui.radioButton_full_flat_field.isChecked():
                    return "full"
                else:
                    return "mean"

            def get_white_ref_name(self):
                return self.ui.comboBox_white_Reference_choice.currentText()

            def load_other_white(self):
                filename,_=QFileDialog.getOpenFileName(self,"Open White Reference Capture",os.path.dirname(self.cube_calib.cube_info.filepath))
                if filename:
                    self.cube_calib = Hypercube()
                    self.cube_calib.open_hyp(default_path=filename, ask_calib=False)
                    self.set_cube(self.cube_calib)
                    self.other_white_capture = True

            def toggle_manual_value_fields(self):
                index = self.ui.comboBox_white_Reference_choice.currentIndex()
                is_manual = (index == 0)  # "Constant value (from field)" is index 0

                self.ui.label_R_manual.setEnabled(is_manual)
                self.ui.doubleSpinBox_R_manual.setEnabled(is_manual)

                self.update_white_spectrum()

            def update_selection_overlay(self):
                coords = self.get_selected_rect()
                if not coords or self.cube_calib is None:
                    self.viewer.clear_selection_overlay()
                    return

                x, y, dx, dy = coords
                h, w, _ = self.cube_calib.data.shape

                if self.ui.radioButton_horizontal_flat_field.isChecked():
                    rect = QRectF(0, y, w, dy)
                elif self.ui.radioButton_vertical_flat_field.isChecked():
                    rect = QRectF(x, 0, dx, h)
                elif self.ui.radioButton_full_flat_field.isChecked():
                    rect = QRectF(0, 0, w, h)
                else:  # mean (no extension)
                    rect = QRectF(x, y, dx, dy)

                self.viewer.add_selection_overlay(rect)

            def set_cube(self,cube):
                self.cube_calib=cube
                # RGB image from data
                channels = [0, len(self.cube_calib.wl) // 2, len(self.cube_calib.wl) - 1]
                img_rgb = self.cube_calib.data[:, :, channels]
                img_rgb = (img_rgb * 255 / img_rgb.max()).clip(0, 255).astype(np.uint8)
                self.viewer.setImage(np_to_qpixmap(img_rgb))
                self.toggle_manual_value_fields()

            def load_white_corection_file(self):
                QMessageBox.information(self,"Ups","Not implemented yet. Wait for update")

            def plot_white_reflectance(self, wavelengths, reflectance,name='white_ref'):
                self.canvas_white.ax.clear()
                self.canvas_white.ax.plot(wavelengths, reflectance)
                self.canvas_white.ax.tick_params(axis='both', labelsize=6)
                self.canvas_white.ax.set_ylabel(name,fontsize=6)
                self.canvas_white.ax.grid(True)
                self.canvas_white.ax.set_ylim(0, 1.1)
                self.canvas_white.fig.tight_layout()
                self.canvas_white.draw()

            def update_white_spectrum(self):
                try :
                    self.cube_calib.wl
                except:
                    return

                name = self.get_white_ref_name()
                if name == "Constant value (from field)":
                    R_val=int(self.ui.doubleSpinBox_R_manual.value()*100)/100
                    reflectance=R_val*np.ones(len(self.cube_calib.wl))
                    label=f'Constant R = {R_val}'
                    name=label
                    self.plot_white_reflectance(self.cube_calib.wl, reflectance, name)
                    return

                elif name == "Personal (load file)":
                    self.canvas_white.ax.clear()
                    self.canvas_white.ax.grid(True)
                    self.canvas_white.ax.set_ylim(0, 1.1)
                    self.canvas_white.ax.set_xlim(self.cube_calib.wl[0], self.cube_calib.wl[-1])
                    self.canvas_white.draw()
                    return

                reflectance = self.cube_calib.get_ref_white(name)
                if 'Sphere Optics' in name:
                    name=name.replace('Sphere Optics','SphOpt')
                self.plot_white_reflectance(self.cube_calib.wl, reflectance,name)

            def accept(self):
                coords = self.get_selected_rect()
                mode = self.get_calibration_mode()

                # si pas de sélection et pas de mode full, refuser
                if not coords and not self.other_white_capture and mode != "full":
                    QMessageBox.warning(
                        self,
                        "Selection Required",
                        "Please select a region before validating."
                    )
                    return  # n'appelle pas super().accept()

                # sinon, fermeture acceptée
                super().accept()

        app = QApplication.instance() or QApplication(sys.argv)
        dialog = WhiteCalibrationDialog()
        dialog.set_cube(cube=self)

        if dialog.exec_() == QDialog.Accepted:
            white_ref_name = dialog.get_white_ref_name()
            if white_ref_name == "Constant value (from field)":
                white_ref_values = dialog.ui.doubleSpinBox_R_manual.value()
            else:
                white_ref_values = self.get_ref_white(white_ref_name)

            print(f'[Hypercube_calib] white_ref_name : {white_ref_name}')

            calib_mode = dialog.get_calibration_mode()
            coords = dialog.get_selected_rect()

            if not coords:
                if calib_mode!='full':
                    QMessageBox.warning(None, "No Selection", "No region was selected.")
            else :
                x, y, dx, dy = coords

            if calib_mode == "mean":
                selected = dialog.cube_calib.data[y:y + dy, x:x + dx, :]
                mean_white = np.mean(selected, axis=(0, 1))
                mean_white[mean_white == 0] = 1e-6
                self.data /= (mean_white / white_ref_values)
                self.metadata['calibration_type'] = 'mean_rectangle'
            elif calib_mode == "horizontal":
                selected = dialog.cube_calib.data[y:y + dy, :, :]
                mean_white = np.mean(selected, axis=0)
                mean_white[mean_white == 0] = 1e-6
                self.data /= (mean_white[ None,:, :] / white_ref_values)
                self.metadata['calibration_type'] = 'horizontal_flat'
            elif calib_mode == "vertical":
                selected = dialog.cube_calib.data[:, x:x + dx, :]
                mean_white = np.mean(selected, axis=1)
                mean_white[mean_white == 0] = 1e-6
                self.data /= (mean_white[:,None, :] / white_ref_values)
                self.metadata['calibration_type'] = 'vertical_flat'
            elif calib_mode == "full":
                selected=dialog.cube_calib.data
                selected[selected == 0] = 1e-6
                mean_white=selected
                self.data /= (self / white_ref_values)
                self.metadata['calibration_type'] = 'full_flat'

            try :
                print(f'[Hypercube_calib] mean_white shape: {mean_white.shape}')
                print(f'[Hypercube_calib] cube shape: {self.data.shape}')
            except:
                pass

            self.metadata['calibration_reflectance_values'] = white_ref_values
            self.metadata['white_reference'] = white_ref_name
            self.metadata['reflectance_data_from'] = 'selected white in image ' + white_ref_name
            print('Calibration done using selected region and chosen white reference.')

            ans = QMessageBox.question(None, 'Save with calibration',
                                       'Do you want to save the reflectance cube?',
                                       QMessageBox.Yes | QMessageBox.No)

            if ans == QMessageBox.Yes:
                self.save_calib = True

    def get_ref_white(self,white_name):
        from scipy.io import loadmat
        from scipy.interpolate import interp1d

        multi_labels = ['Sphere Optics Multi 90 (CIL)', 'Sphere Optics Multi 50 (CIL)', 'Sphere Optics Multi 20 (CIL)', 'Sphere Optics Multi 5 (CIL)']

        if getattr(sys, 'frozen', False):
            # if from .exe de pyinstaler
            BASE_DIR = sys._MEIPASS
        else:
            # from Python script
            CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
            BASE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

        white_folder = os.path.join(BASE_DIR, "hypercubes", "white_ref_reflectance_data")

        if white_name == 'Sphere Optics Full White (CIL)':
            white_path = os.path.join(white_folder, 'SphereOptics_standard_white.mat')
            data = loadmat(white_path)['white_reflectance']
            wl_white = data[:, 0].squeeze()
            reflectance_white = data[:, 1].squeeze()

        elif white_name in multi_labels:
            white_path = os.path.join(white_folder, 'SO_multiwhite.mat')
            data = loadmat(white_path)
            wl_white = data['wl'].squeeze()
            reflectance_dic = {multi_labels[i]: data['sphereoptics'][:, i] for i in range(len(multi_labels))}
            reflectance_white = reflectance_dic[white_name].squeeze()

        elif white_name == 'Teflon':
            white_path = os.path.join(white_folder, 'Teflon_full_range.mat')
            data = loadmat(white_path)
            wl_white = data['wl'].squeeze()
            reflectance_white = data['spectrum'].squeeze()

        elif white_name == 'White Hyspex (Finland)':
            white_path = os.path.join(white_folder, 'multi90_finland.mat')
            data = loadmat(white_path)
            wl_white = data['wl_multi_90white'].squeeze()
            reflectance_white = data['Multi_90white'].squeeze()

        else:
            print('Unknown white reference label')
            return

        interp_ref_func = interp1d(wl_white, reflectance_white, kind='cubic',
                                   bounds_error=False,
                                   fill_value=(reflectance_white[0], reflectance_white[-1]))

        reflectance_interp = interp_ref_func(self.wl)
        return reflectance_interp

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

    def _on_accept(self):
        print("[DEBUG] on_accept called")

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
        if not self._accepted:
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
    matplotlib.use('Qt5Agg')  # 'TkAgg' or 'Qt5Agg'
    import matplotlib.pyplot as plt

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    # fname = '00189-VNIR-mock-up.h5'
    #
    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Thin_film\CITIC\241216\PikaL'
    # fname = 'c1318_12.bil.hdr' # test pikaL

    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\hypspex'
    fname = 'P1A_SWIR_384_SN3189_2335us_2023-09-05T205402_raw_rad.hdr' # test pikaL

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST'
    # fname = 'P1A_SWIR_384_SN3189_2335us_2023-09-05T205402_raw_rad.hdr' # test specim
    # fname = 'P1A_SWIR_384_SN3189_2335us_2023-09-05T205402_raw_rad_calib.hdr' # test specim calib

    # folder = r'G:\Mi unidad\CIMLab\Proyectos y OTRI\Hyperdoc\Datos\Muestras controladas 22-23\NTNU captures\UGS\VNIR'
    # fname = 'P1A_VNIR_1800_SN00841_12004us_2023-09-05T220145_raw.hdr' # test specim

    # fname = 'MPD41a_SWIR.mat'
    # fname = 'MPD41a_SWIR_cube_NOv7p3.mat'
    # fname = 'MPD41a_SWIR_sruct_NOv7p3.mat'
    # fname = 'MPD41a_SWIR_struct_YESv7p3.mat'

    import os

    filepath = os.path.join(folder, fname)

    # app = QApplication(sys.argv)

    # cube = Hypercube(filepath=filepath, load_init=True)
    cube = Hypercube(filepath=filepath, cube_info=None, load_init=True)

    # cube.calibrating_from_image_extract()

    # path=cube.cube_info.filepath.replace('.','_calib.')

    # for key in cube.cube_info.metadata_temp:
    #     print(f' {key} -> {cube.cube_info.metadata_temp[key]}')

    mid=int(cube.data.shape[2]/2)
    chan=[0,mid,cube.data.shape[2]-1]
    image_np=cube.data[:,:,chan]
    print(f'max data : {np.max(cube.data)}')


    fig,ax=plt.subplots()
    ax.imshow(image_np)
    plt.show()
