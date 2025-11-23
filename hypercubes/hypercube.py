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
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


from PyQt5.QtWidgets import (QWidget,QSizePolicy,QMainWindow,QWidget,QCheckBox,
    QApplication, QFileDialog, QMessageBox, QDialog, QTreeWidgetItem,QVBoxLayout,QHBoxLayout,
)
from PyQt5.QtCore   import pyqtSignal, QEventLoop, QRectF,QTimer,pyqtSlot

from hypercubes.hdf5_browser_window import Ui_HDF5BrowserWidget
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
    gtmap_path:Optional[str] = None # Ground Truth map location in the path (if present)
    metadata_temp: dict = field(default_factory=dict) # all metadatas modified in the app before saving
    data_shape: Optional[Union[List[float], np.ndarray]] = None # cube shape [width, height, bands]
    wl_trans:Optional[str]= None # if need to transpose wl dim from dim 1 to dim 3

    @property
    def filepath(self):
        return self._filepath

    # protect filepath modification
    def filepath(self, val):
        # Normalise pour éviter que deux bouts de chaîne écrites différemment soient considérées différentes
        old = Path(self._filepath).resolve() if self._filepath else None
        new = Path(val).resolve()
        if old != new:
            # print(f"[DEBUG] Changing filepath from {self._filepath} to {val}")
            self._filepath = val

    # because only one filepath for one cube...and one cube for one filepath, let's define the cubeInfo equality
    def __eq__(self, other):
        if not isinstance(other, CubeInfoTemp):
            return NotImplemented
        return Path(self.filepath).resolve() == Path(other.filepath).resolve()

    def __hash__(self):
        # if in dic or set
        return hash(Path(self.filepath).resolve())

import numpy as np

def averagefilter(image, window=(3, 3), padding='constant'):
    """
    2D mean filtering using integral image method (fast summed-area table).
    Parameters
    ----------
    image : ndarray
        Input 2D image.
    window : tuple of int
        (m, n) window size.
    padding : str or scalar
        Padding mode ('constant', 'edge', 'reflect', 'wrap') or a constant value.
    Returns
    -------
    image_filt : ndarray
        Filtered image, same dtype as input.
    """
    m, n = window

    # Ensure odd dimensions
    if m % 2 == 0:
        m -= 1
    if n % 2 == 0:
        n -= 1

    if image.ndim != 2:
        raise ValueError("Input image must be 2D (grayscale).")

    rows, cols = image.shape

    # Map MATLAB's padding names to numpy.pad modes
    pad_map = {
        'replicate': 'edge',
        'symmetric': 'reflect',
        'circular': 'wrap',
        'constant': 'constant'
    }
    pad_mode = pad_map.get(padding, padding)  # keep numeric if scalar

    # Pad before and after
    pad_pre = ((m+1)//2, (n+1)//2)
    pad_post = ((m-1)//2, (n-1)//2)

    if isinstance(pad_mode, (int, float)):
        img_pad_pre = np.pad(image, pad_pre, mode='constant', constant_values=pad_mode)
        img_pad = np.pad(img_pad_pre, pad_post, mode='constant', constant_values=pad_mode)
    else:
        img_pad_pre = np.pad(image, pad_pre, mode=pad_mode)
        img_pad = np.pad(img_pad_pre, pad_post, mode=pad_mode)

    # Convert to double for computation
    img_d = img_pad.astype(float)

    # Integral image
    t = np.cumsum(np.cumsum(img_d, axis=0), axis=1)

    # Window sum extraction
    imageI = (
        t[m:rows+m, n:cols+n] +
        t[0:rows, 0:cols] -
        t[m:rows+m, 0:cols] -
        t[0:rows, n:cols+n]
    )
    # Average
    imageI = imageI / (m * n)

    return imageI.astype(image.dtype)

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

    def normalize_spectral(self, train_wl, *, min_wl=400.0, max_wl=1700.0,
                           interp_kind="linear", in_place=False):
        """
        Clamp to [min_wl, max_wl] and interpolate onto `train_wl` (e.g. np.arange(400,1705,5)).
        - Validates that wl exists, is finite, and increasing (sorts data/bands if needed).
        - Intersects desired bounds with cube coverage and training grid.
        - Uses get_interpolate_cube(wl_interp=...) to resample.
        Returns a NEW Hypercube unless in_place=True.
        Raises ValueError with a clear message if normalization is impossible.
        """
        import numpy as np

        if self.wl is None:
            raise ValueError("Cube has no wavelengths; cannot normalize.")
        wl = np.asarray(self.wl, dtype=float)
        if wl.ndim != 1 or wl.size < 2:
            raise ValueError(f"Unexpected wl shape/size: {wl.shape if hasattr(wl, 'shape') else type(wl)}")

        # Remove non-finite wavelengths and corresponding bands
        finite = np.isfinite(wl)
        if not finite.all():
            wl = wl[finite]
            if wl.size < 2:
                raise ValueError("Too few valid wavelengths after cleaning.")
            self.data = self.data[:, :, finite]

        # Enforce strictly increasing order (sort bands if needed)
        if not np.all(np.diff(wl) > 0):
            order = np.argsort(wl)
            wl = wl[order]
            self.data = self.data[:, :, order]

        # Compute overlap with desired bounds and with the cube’s own coverage
        lo = max(min_wl, float(wl.min()))
        hi = min(max_wl, float(wl.max()))
        if not (lo < hi):
            raise ValueError(f"No spectral overlap with {min_wl}–{max_wl} nm.")

        train_wl = np.asarray(train_wl, dtype=float)
        if train_wl.ndim != 1 or train_wl.size < 2:
            raise ValueError("`train_wl` must be a 1D array of target wavelengths.")

        target = train_wl[(train_wl >= lo) & (train_wl <= hi)]
        if target.size < 2:
            raise ValueError("Not enough target wavelengths inside the overlapped range.")

        # Interpolate onto the target grid (uses your existing helper)
        data_i, wl_i = self.get_interpolate_cube(wl_interp=target, interp_kind=interp_kind)

        if in_place:
            self.data = data_i
            self.wl = wl_i
            # keep metadata/cube_info as-is
            return self
        else:
            new_cube = Hypercube(data=data_i, wl=wl_i, metadata=dict(self.metadata), cube_info=self.cube_info)
            return new_cube

    def get_interpolate_cube(self, _wl_step=None, wl_interp=None, interp_kind='linear'):
        """
        Interpole le cube le long de l'axe spectral.
        - Soit on fournit un pas (_wl_step) -> interpole entre wl[0] et wl[-1] avec ce pas
        - Soit on fournit explicitement une grille (wl_interp) -> interpole exactement sur ces lambda
        Renvoie: (data_interpolated, wl_interp)
        """
        if _wl_step is None and wl_interp is None:
            raise ValueError("Provide either _wl_step or wl_interp")

        # f(λ) vectorisé sur l'axe spectral
        f = interp1d(
            self.wl.astype(float),
            self.data,  # shape: (H, W, B)
            kind=interp_kind,
            axis=2,
            bounds_error=False,
            fill_value=(self.data[:, :, 0], self.data[:, :, -1])  # extrap à plateaux
        )

        if wl_interp is None:
            wl_start = int(self.wl[0] / _wl_step) * _wl_step
            wl_end = int(self.wl[-1] / _wl_step + 0.5) * _wl_step
            wl_interp = np.arange(wl_start, wl_end + _wl_step, _wl_step, dtype=float)
        else:
            wl_interp = np.asarray(wl_interp, dtype=float)
            # garde-fous utiles
            if wl_interp.ndim != 1:
                raise ValueError("wl_interp must be 1D")
            if wl_interp.size < 2:
                raise ValueError("wl_interp needs at least 2 wavelengths")
            # Trie + unique (interp1d exige croissant strict)
            wl_interp = np.unique(np.sort(wl_interp))

        data_interp = f(wl_interp)

        return data_interp, wl_interp

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
                        self.cube_info.gtmap_path= self.cube_info.metadata_path+"/gt_index_map"
                        self.cube_info.wl_trans = True
                        flag_loaded=True  # success
                        # print('loaded .mat v7p3 with classic hypercube look')
               except Exception:
                   pass

               if not flag_loaded:
                   try:
                       if self.cube_info.data_path is None :
                           # print('try wih browser')
                           ans=self.look_with_browser(filepath)
                           # print(ans)
                           if not ans:
                               return

                       try:
                            self.get_data_mat7p3_or_h5(filepath)
                            self.cube_info.wl_trans = True
                            flag_loaded=True
                            # print('loaded .mat v7p3 fro known cube_info.pathS')
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
                    # print('Loaded .mat NO v7p3 with classic')
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

                            # print('loaded .mat NO v7p3 with BROWSER')
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
                    self.cube_info.gtmap_path = self.cube_info.metadata_path+"gt_index_map"
                    self.cube_info.metadata_temp = copy.deepcopy(self.metadata)
                    self.cube_info.wl_trans = True
                    self.cube_info.filepath=filepath

                    if self.data is not None:
                        self.cube_info.data_shape = self.data.shape

                flag_loaded=True  # success → no dialog
                # print('.h5 loaded from classic')

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
                        # print('loaded .h5 from browser')
                    except:
                        pass

        # 3) ENVI (.hdr + raw)
        # TODO : cube info fill with ENVI files
        elif ext == '.hdr':
            try:
                img = envi.open(filepath)
                self.metadata = copy.deepcopy(img.metadata)
                # dtype = self.metadata.get('data type')
                self.data = img.load().astype(np.float32)

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
        # print(filepath)
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
                self.cube_calib_init=None
                self.reflectance_white=None
                self.wl_white=None
                self.other_white_capture=False # if other file loaded.
                self.use_unity_white = False

                self.cube_main = None
                self.cube_white= None
                self.cube_dark= None

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
                self.ui.pushButton_load_dark.clicked.connect(self.load_dark)

                self.ui.pushButton_valid_calibration.clicked.connect(self.accept)
                self.ui.pushButton_load_personal_white_ref.clicked.connect(self.load_white_corection_file)
                self.ui.doubleSpinBox_R_manual.valueChanged.connect(self.update_white_spectrum)
                self.ui.checkBox_crop_ref_1.clicked.connect(self.crop_ref_to_one)
                self.ui.checkBox_use_white_capture.toggled.connect(self.on_use_white_capture_toggled)

                self.ui.radioButton_horizontal_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_vertical_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_full_flat_field.toggled.connect(self.update_selection_overlay)
                self.ui.radioButton_no_flat_field.toggled.connect(self.update_selection_overlay)

                QTimer.singleShot(0, self.update_white_spectrum) # to force spectra draw after layout processed

            def crop_ref_to_one(self):

                if self.ui.checkBox_crop_ref_1.isChecked():
                    self.set_cube(self.cube_calib_init,renorm=True)
                else:
                    self.set_cube(self.cube_calib_init, renorm=False)

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

            def load_dark(self):
                filters = (
                    "Supported files (*.h5 *.mat *.hdr);;"
                    "HDF5 files (*.h5);;"
                    "MATLAB files (*.mat);;"
                    "ENVI header (*.hdr)"
                )
                default_filter = "Supported files (*.h5 *.mat *.hdr);;"

                try:
                    dir = os.path.dirname(self.cube_calib.cube_info.filepath)
                except:
                    dir = None

                filename, _ = QFileDialog.getOpenFileName(
                    parent=None,
                    caption="Open Dark Capture",
                    directory=dir,
                    filter=filters,
                    initialFilter=default_filter
                )

                if not filename:
                    return
                self.cube_dark = Hypercube()
                self.cube_dark.open_hyp(default_path=filename, open_dialog=False, ask_calib=False)
                self.ui.pushButton_load_dark.setText(f'Dark Loaded {self.cube_dark.data.shape}')

            def load_other_white(self):
                filters = (
                    "Supported files (*.h5 *.mat *.hdr);;"
                    "HDF5 files (*.h5);;"
                    "MATLAB files (*.mat);;"
                    "ENVI header (*.hdr)"
                )
                default_filter = "Supported files (*.h5 *.mat *.hdr);;"

                try :
                    dir=os.path.dirname(self.cube_calib.cube_info.filepath)
                except:
                    dir=None

                filename, _ = QFileDialog.getOpenFileName(
                    parent=None,
                    caption="Open White Reference Capture",
                    directory=dir,
                    filter=filters,
                    initialFilter=default_filter
                )

                if not filename:
                    return
                self.cube_white = Hypercube()
                self.cube_white.open_hyp(default_path=filename, open_dialog=False,ask_calib=False)
                self.other_white_capture = True


                cube=self.cube_white
                self.cube_calib = cube

                self.cube_calib_init = Hypercube(
                    data=cube.data if cube.data is not None else None,
                    wl=cube.wl if cube.wl is not None else None,
                    metadata=cube.metadata,
                    cube_info=cube.cube_info,
                )
                print('[CALIBRATION] : cube filepath : ',self.cube_main.cube_info.filepath)
                print('[CALIBRATION] : cube size : ',self.cube_main.data.shape)

                print('[CALIBRATION] : white filepath : ',self.cube_white.cube_info.filepath)
                print('[CALIBRATION] : white size : ',self.cube_white.data.shape)

                print('[CALIBRATION] : calib filepath : ',self.cube_calib.cube_info.filepath)
                print('[CALIBRATION] : calib size : ',self.cube_calib.data.shape)

                print('[CALIBRATION] : init filepath : ', self.cube_calib_init.cube_info.filepath)
                print('[CALIBRATION] : init size : ', self.cube_calib_init.data.shape)

                self.set_cube(self.cube_white, renorm=False)

                print('--------------SET CUBE-----------------')

                print('[CALIBRATION] : cube filepath : ',self.cube_main.cube_info.filepath)
                print('[CALIBRATION] : cube size : ',self.cube_main.data.shape)

                print('[CALIBRATION] : white filepath : ',self.cube_white.cube_info.filepath)
                print('[CALIBRATION] : white size : ',self.cube_white.data.shape)

                print('[CALIBRATION] : calib filepath : ',self.cube_calib.cube_info.filepath)
                print('[CALIBRATION] : calib size : ',self.cube_calib.data.shape)

                print('[CALIBRATION] : init filepath : ', self.cube_calib_init.cube_info.filepath)
                print('[CALIBRATION] : init size : ', self.cube_calib_init.data.shape)

                n_bands = self.cube_calib.wl.size
                channels = [0, n_bands // 2, n_bands - 1]
                img_rgb = self.cube_calib.data[:, :, channels]

                if img_rgb.max() > 0:
                    img_rgb = (img_rgb * 255.0 / img_rgb.max()).clip(0, 255).astype(np.uint8)
                else:
                    img_rgb = np.zeros_like(img_rgb, dtype=np.uint8)

                self.viewer.setImage(np_to_qpixmap(img_rgb))

                self.toggle_manual_value_fields()

                self.ui.checkBox_use_white_capture.setChecked(True)

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

            def set_cube(self, cube,renorm=False,init=False):
                from PyQt5.QtWidgets import QMessageBox
                import numpy as np

                if init:

                    self.cube_calib = cube
                    self.cube_calib_init = Hypercube(
                        data=cube.data.copy() if cube.data is not None else None,
                        wl=cube.wl.copy() if cube.wl is not None else None,
                        metadata=copy.deepcopy(cube.metadata),
                        cube_info=cube.cube_info,
                    )
                    self.cube_main = cube

                    # 1) S'assurer que les données sont chargées
                    if getattr(self.cube_calib, "data", None) is None:
                        ci = getattr(self.cube_calib, "cube_info", None)
                        fp = getattr(ci, "filepath", None) if ci else None

                        if not fp:
                            QMessageBox.warning(
                                self,
                                "No data",
                                "Cube has no data loaded and no filepath. "
                                "Cannot perform white calibration."
                            )
                            return

                        try:
                            # recharge le cube sans redemander de fichier
                            self.cube_calib.open_hyp(
                                default_path=fp,
                                open_dialog=False,
                                cube_info=ci
                            )
                        except Exception as e:
                            QMessageBox.critical(
                                self,
                                "Error loading cube",
                                f"Could not load cube data for calibration:\n{e}"
                            )
                            return

                    # 2) Déterminer un axe spectral robuste
                    wl = getattr(self.cube_calib, "wl", None)
                    if wl is None:
                        # pas de wl -> axe artificiel 0..B-1
                        n_bands = self.cube_calib.data.shape[2]
                        wl = np.arange(n_bands, dtype=float)
                        self.cube_calib.wl = wl
                    else:
                        wl = np.asarray(wl).ravel()
                        n_bands = wl.size

                else:

                    wl=self.cube_calib.wl
                    n_bands = wl.size

                    if renorm :
                        self.cube_calib.data = np.clip(self.cube_calib_init.data, 0, 1)
                    else:
                        self.cube_calib.data=self.cube_calib_init.data

                # 3) Construire l’image RGB à partir des indices de bande
                channels = [0, n_bands // 2, n_bands - 1]
                img_rgb = self.cube_calib.data[:, :, channels]

                if img_rgb.max() > 0:
                    img_rgb = (img_rgb * 255.0 / img_rgb.max()).clip(0, 255).astype(np.uint8)
                else:
                    img_rgb = np.zeros_like(img_rgb, dtype=np.uint8)

                self.viewer.setImage(np_to_qpixmap(img_rgb))

                # 4) Mettre à jour l’état de l’UI
                self.toggle_manual_value_fields()

            def load_white_corection_file(self):
                """
                Load a 2-column text/CSV file containing (wavelength, reflectance) or
                (reflectance, wavelength), let the user confirm column order and
                whether to skip the first row, then set self.wl_white / self.reflectance_white
                and refresh the preview plot.
                """
                # ---- 1) Ask for a file ----
                start_dir = os.path.dirname(
                    self.cube_calib.cube_info.filepath) if self.cube_calib and self.cube_calib.cube_info and self.cube_calib.cube_info.filepath else ""
                fname, _ = QFileDialog.getOpenFileName(
                    self, "Open white correction file", start_dir,
                    "Text/CSV files (*.txt *.csv);;All files (*)"
                )
                if not fname:
                    return

                # ---- 2) Read file and detect delimiter ----
                def _detect_delimiter(sample_line: str):

                    if ';' in sample_line:
                        return ';'
                    elif '\t' in sample_line:
                        return '\t'
                    elif ',' in sample_line:
                        return ','

                    return None  # means split on any whitespace

                # Read lines (ignore empties)
                try:
                    with open(fname, 'r', encoding='utf-8') as f:
                        raw_lines = [ln.strip() for ln in f.readlines()]
                except UnicodeDecodeError:
                    with open(fname, 'r', encoding='latin-1') as f:
                        raw_lines = [ln.strip() for ln in f.readlines()]
                lines = [ln for ln in raw_lines if ln != ""]
                if not lines:
                    QMessageBox.warning(self, "Empty file", "The selected file is empty.")
                    return

                delim = _detect_delimiter(lines[0])

                # Tokenize all rows to preview (keep original text tokens as strings)
                def _tokenize(ln):
                    return ln.split(delim) if delim else ln.split()

                tokens = [_tokenize(ln) for ln in lines]
                # Keep only rows with at least 2 tokens
                tokens = [row for row in tokens if len(row) >= 2]
                if not tokens:
                    QMessageBox.warning(self, "Invalid file", "Could not find rows with at least two columns.")
                    return

                # Heuristic: if first row has any non-float token, default skip_first_row = True
                def _is_float(s):
                    try:
                        float(s.replace(',', '.'))
                        return True
                    except Exception:
                        return False

                first_row_has_nonnumeric = (not _is_float(tokens[0][0])) or (not _is_float(tokens[0][1]))
                default_skip = first_row_has_nonnumeric

                # ---- 3) Small preview dialog with options ----
                class TablePreviewDialog(QDialog):
                    def __init__(self, parent, fname, tokens, default_skip):
                        super().__init__(parent)
                        self.setWindowTitle("Preview white correction table")
                        self.setMinimumWidth(520)
                        self.setMinimumHeight(420)

                        self.tokens = tokens
                        self.skip_first = default_skip
                        self.swap_cols = False

                        layout = QVBoxLayout(self)

                        layout.addWidget(QLabel(f"File: {os.path.basename(fname)}"))

                        # Table preview (first 50 rows)
                        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
                        self.table = QTableWidget(self)
                        n_show = min(50, len(tokens))
                        self.table.setRowCount(n_show)
                        self.table.setColumnCount(2)
                        self.table.setHorizontalHeaderLabels(["Column 0", "Column 1"])
                        for i in range(n_show):
                            c0 = QTableWidgetItem(str(tokens[i][0]))
                            c1 = QTableWidgetItem(str(tokens[i][1]))
                            self.table.setItem(i, 0, c0)
                            self.table.setItem(i, 1, c1)
                        self.table.resizeColumnsToContents()
                        layout.addWidget(self.table)

                        # Options
                        opts = QHBoxLayout()
                        self.cb_skip = QCheckBox("Skip first row")
                        self.cb_skip.setChecked(default_skip)
                        self.cb_swap = QCheckBox("Swap columns (col0=reflectance, col1=wavelength)")
                        self.cb_swap.setChecked(False)
                        opts.addWidget(self.cb_skip)
                        opts.addWidget(self.cb_swap)
                        opts.addStretch()
                        layout.addLayout(opts)

                        # Info/blurb
                        info = QLabel("Default: Column 0 = wavelength (nm), Column 1 = reflectance (0–1).\n"
                                      "Enable 'Swap columns' if your file has the opposite order.")
                        info.setStyleSheet("color: gray;")
                        layout.addWidget(info)

                        # Buttons
                        from PyQt5.QtWidgets import QPushButton
                        btns = QHBoxLayout()
                        btn_ok = QPushButton("Accept")
                        btn_cancel = QPushButton("Cancel")
                        btn_ok.clicked.connect(self.accept)
                        btn_cancel.clicked.connect(self.reject)
                        btns.addStretch()
                        btns.addWidget(btn_cancel)
                        btns.addWidget(btn_ok)
                        layout.addLayout(btns)

                    def get_choices(self):
                        return self.cb_skip.isChecked(), self.cb_swap.isChecked()

                dlg = TablePreviewDialog(self, fname, tokens, default_skip)
                if dlg.exec_() != QDialog.Accepted:
                    return
                skip_first, swap_cols = dlg.get_choices()

                # ---- 4) Build numeric arrays based on user choices ----
                start_idx = 1 if skip_first and len(tokens) > 1 else 0
                usable = tokens[start_idx:]

                # Convert to float with best effort (drop rows that fail)
                wl_vals = []
                r_vals = []
                for row in usable:
                    try:
                        # Accept decimal commas like "0,411" or "0," -> "0.411" / "0."
                        c0 = row[0].strip().replace(',', '.')
                        c1 = row[1].strip().replace(',', '.')

                        # Handle weird trailing dots (e.g. "0.")
                        if c0.endswith('.'):
                            c0 = c0[:-1] or '0'
                        if c1.endswith('.'):
                            c1 = c1[:-1] or '0'

                        a = float(c0)
                        b = float(c1)

                        if swap_cols:
                            wl_vals.append(b)
                            r_vals.append(a)
                        else:
                            wl_vals.append(a)
                            r_vals.append(b)
                    except Exception:
                        # silently skip malformed lines
                        continue

                import numpy as _np
                wl_vals = _np.asarray(wl_vals, dtype=float)
                r_vals = _np.asarray(r_vals, dtype=float)

                # Basic validation
                if wl_vals.size < 2 or r_vals.size < 2:
                    QMessageBox.warning(self, "Not enough data",
                                        "After applying your options, fewer than two valid rows remain.")
                    return
                if wl_vals.size != r_vals.size:
                    n = min(wl_vals.size, r_vals.size)
                    wl_vals = wl_vals[:n]
                    r_vals = r_vals[:n]

                # Optional clamp reflectance to [0, 1.1]
                r_vals = _np.clip(r_vals, 0.0, 1.1)

                # Sort by wavelength if needed
                order = _np.argsort(wl_vals)
                wl_vals = wl_vals[order]
                r_vals = r_vals[order]

                # ---- 5) Store & refresh ----
                self.wl_white = wl_vals
                self.reflectance_white = r_vals

                # Notify the user and refresh plot
                QMessageBox.information(self, "Loaded",
                                        f"Loaded {wl_vals.size} rows.\n"
                                        f"Column order: {'(reflectance, wavelength)' if swap_cols else '(wavelength, reflectance)'}\n"
                                        f"{'Skipped first row' if skip_first else 'Used first row'}.")

                self.ui.label.setText(fname.split('/')[-1])
                self.ui.comboBox_white_Reference_choice.setCurrentText("Personal (load file)")
                self.update_white_spectrum()

            def on_use_white_capture_toggled(self, checked):
                from PyQt5.QtWidgets import QMessageBox

                if checked:
                    if self.cube_white is None:
                        QMessageBox.warning(
                            self,
                            "No white capture loaded",
                            "You must load a white reference capture first."
                        )
                        # revenir à l’état décoché
                        self.ui.checkBox_use_white_capture.setChecked(False)
                        return

                    # 👉 utiliser le cube de white capture
                    cube = self.cube_white
                    renorm = False


                else:
                    # 👉 revenir au cube principal
                    cube = self.cube_main
                    renorm = self.ui.checkBox_crop_ref_1.isChecked()

                self.cube_calib = cube
                self.cube_calib_init = Hypercube(
                    data=cube.data if cube.data is not None else None,
                    wl=cube.wl if cube.wl is not None else None,
                    metadata=cube.metadata,
                    cube_info=cube.cube_info,
                )

                self.set_cube(self.cube_calib, renorm=renorm)

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
                # pas de cube -> rien à faire
                if self.cube_calib is None or self.cube_calib.data is None:
                    return

                # Axe spectral robuste
                wl = getattr(self.cube_calib, "wl", None)
                if wl is None:
                    n_bands = self.cube_calib.data.shape[2]
                    wl = np.arange(n_bands, dtype=float)
                    self.cube_calib.wl = wl  # on l’enregistre pour les autres fonctions
                else:
                    wl = np.asarray(wl).ravel()
                    n_bands = wl.size

                name = self.get_white_ref_name()

                # --- Cas "valeur constante" ---
                if name == "Constant value (from field)":
                    R_val = int(self.ui.doubleSpinBox_R_manual.value() * 100) / 100
                    reflectance = R_val * np.ones(n_bands)
                    label = f'Constant R = {R_val}'
                    self.plot_white_reflectance(wl, reflectance, label)
                    return

                # --- Cas "fichier perso" ---
                if name == "Personal (load file)":
                    try:
                        if self.wl_white is None:
                            QMessageBox.warning(self, "Load white correction file first",
                                                "Please load a white correction file first.")
                            return
                        else:
                            reflectance = self.cube_calib.get_ref_white(
                                name, self.wl_white, self.reflectance_white
                            )
                    except Exception:
                        return
                else:
                    # Sphere Optics, Teflon, etc.
                    reflectance = self.cube_calib.get_ref_white(name)

                if 'Sphere Optics' in name:
                    name = name.replace('Sphere Optics', 'SphOpt')

                self.plot_white_reflectance(wl, reflectance, name)

            def accept(self):
                from PyQt5.QtWidgets import QMessageBox

                coords = self.get_selected_rect()
                mode = self.get_calibration_mode()

                # Si pas de sélection, pas d'autre capture, et pas en mode "full"
                if not coords and not self.other_white_capture and mode != "full":
                    msg = QMessageBox(self)
                    msg.setWindowTitle("No white region selected")
                    msg.setText(
                        "No region was selected for the white zone.\n\n"
                        "What do you want to do?"
                    )
                    msg.setIcon(QMessageBox.Question)

                    btn_select = msg.addButton("Select a region", QMessageBox.AcceptRole)
                    btn_unity = msg.addButton("No calibration from white zone", QMessageBox.DestructiveRole)

                    msg.exec_()

                    if msg.clickedButton() is btn_select:
                        # On reste dans la fenêtre, l'utilisateur va sélectionner une zone
                        return

                    if msg.clickedButton() is btn_unity:
                        # On valide le dialog, et on signalera plus tard qu’on veut un spectre de 1
                        self.use_unity_white = True
                        super().accept()
                        return

                    # sécurité, au cas où
                    return

                # sinon, fermeture acceptée normalement
                super().accept()

        app = QApplication.instance() or QApplication(sys.argv)
        dialog = WhiteCalibrationDialog()
        dialog.set_cube(cube=self,init=True)

        if dialog.exec_() == QDialog.Accepted:

            # --- 0) DARK CORRECTION (cube + white) ---
            dark_cube = getattr(self, "cube_dark", None)
            if dark_cube is not None and getattr(dark_cube, "data", None) is not None:
                try:
                    # 0a) Dark pour le cube principal
                    dark_main = self._make_dark_map(dark_cube.data, self.data.shape)
                    self.data = np.maximum(
                        self.data.astype(np.float32) - dark_main,
                        0.0
                    )

                    # 0b) Dark pour le cube white (si différent du cube principal)
                    if dialog.cube_calib is not None and dialog.cube_calib is not self \
                            and getattr(dialog.cube_calib, "data", None) is not None:
                        try:
                            dark_white = self._make_dark_map(
                                dark_cube.data,
                                dialog.cube_calib.data.shape
                            )
                            dialog.cube_calib.data = np.maximum(
                                dialog.cube_calib.data.astype(np.float32) - dark_white,
                                0.0
                            )
                        except Exception:
                            # Si la forme est vraiment incompatible, on ignore pour le white
                            pass

                    # traces dans les métadonnées
                    self.metadata["dark_correction"] = True
                    self.metadata["dark_shape"] = np.array(dark_cube.data.shape, dtype=int)

                except ValueError as e:
                    QMessageBox.warning(
                        None, "Dark correction ignored",
                        f"Dark cube is incompatible with main cube:\n{e}"
                    )

            # --- 1) White reference + mode de calibration ---
            white_ref_name = dialog.get_white_ref_name()
            if white_ref_name == "Constant value (from field)":
                white_ref_values = dialog.ui.doubleSpinBox_R_manual.value()
            else:
                white_ref_values = self.get_ref_white(white_ref_name)

            calib_mode = dialog.get_calibration_mode()
            coords = dialog.get_selected_rect()

            if not coords:
                if calib_mode != 'full':
                    QMessageBox.warning(None, "No Selection", "No region was selected.")
                    return
            else:
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

            self.metadata['calibration_reflectance_values'] = white_ref_values
            self.metadata['white_reference'] = white_ref_name
            self.metadata['reflectance_data_from'] = 'selected white in image ' + white_ref_name
            # print('Calibration done using selected region and chosen white reference.')

            ans = QMessageBox.question(None, 'Save with calibration',
                                       'Do you want to save the reflectance cube?',
                                       QMessageBox.Yes | QMessageBox.No)

            if ans == QMessageBox.Yes:
                self.save_calib = True

    def _make_dark_map(self, dark_data: np.ndarray, target_shape):
        """
        Construit une carte de dark (H, W, B) compatible avec target_shape
        à partir de dark_data, selon les règles :
          - si même H, W, B -> carte complète
          - si même H et W=1 -> profil vertical répété (horizontal flat)
          - si même W et H=1 -> profil horizontal répété (vertical flat)
          - sinon -> spectre moyen (H,W,B) identique partout
        """
        if dark_data is None or dark_data.ndim != 3:
            raise ValueError("Dark cube must be 3D (H, W, B).")

        H, W, B = target_shape
        Hd, Wd, Bd = dark_data.shape

        if Bd != B:
            raise ValueError(
                f"Incompatible dark cube: {Bd} bands vs cube {B} bands."
            )

        # Cas 1 : dimensions identiques -> carte complète
        if Hd == H and Wd == W:
            return dark_data.astype(np.float32)

        # Cas 2 : même hauteur, largeur=1 -> horizontal flat field
        if Hd == H and Wd == 1:
            line = dark_data[:, 0, :]  # (H, B)
            return np.repeat(line[:, None, :], W, axis=1)  # (H, W, B)

        # Cas 2 bis : même largeur, hauteur=1 -> vertical flat field
        if Wd == W and Hd == 1:
            col = dark_data[0, :, :]  # (W, B)
            return np.repeat(col[None, :, :], H, axis=0)  # (H, W, B)

        # Cas 3 : ni largeur ni hauteur -> spectre moyen global
        spec = dark_data.mean(axis=(0, 1))  # (B,)
        return np.broadcast_to(spec[None, None, :], (H, W, B)).astype(np.float32)

    def get_ref_white(self,white_name,_wl_white=None,_reflectance_white=None):
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

        elif white_name == 'White Hyspex (Norway)':
            white_path = os.path.join(white_folder, 'multi90_finland.mat')
            data = loadmat(white_path)
            wl_white = data['wl_multi_90white'].squeeze()
            reflectance_white = data['Multi_90white'].squeeze()

        elif 'Personal' in white_name :
            wl_white=_wl_white
            reflectance_white=_reflectance_white

        else:
            print('[HYP Calibration] Unknown white reference label')
            return

        interp_ref_func = interp1d(wl_white, reflectance_white, kind='cubic',
                                   bounds_error=False,
                                   fill_value=(reflectance_white[0], reflectance_white[-1]))

        reflectance_interp = interp_ref_func(self.wl)
        return reflectance_interp

    def get_best_band(self):
        """
        return band with smaller SNR ratio
        """
        hypercube = self.data
        bands_number = hypercube.shape[2]
        noise_values = np.zeros(bands_number)

        for k in range(bands_number):
            band = hypercube[:, :, k]
            average = np.mean(band)
            std_dev = np.std(band)
            noise_values[k] = 10 * np.log10((average ** 2) / (std_dev ** 2))

        best_band = np.argmin(noise_values)
        return best_band

    def binarize_niblack(self, image, param={}):

        try:
            k_threshold=param['k']
        except:
            k_threshold = -0.2

        X = image.astype(float)
        filt_radius = X.shape[0] // 3
        y, x = np.ogrid[-filt_radius:filt_radius + 1, -filt_radius:filt_radius + 1]
        mask = x ** 2 + y ** 2 <= filt_radius ** 2
        mask = mask / np.sum(mask)

        local_mean = uniform_filter(X, size=filt_radius * 2 + 1, mode='reflect')
        local_std = np.sqrt(uniform_filter(X ** 2, size=filt_radius * 2 + 1, mode='reflect'))
        return X <= (local_mean + k_threshold * local_std)

    def binarize_otsu(self, image):
        thresh = threshold_otsu(image)
        return image < thresh

    def binarize_sauvola(self, image,param={}):

        try:
            k=param['k']
        except:
            k=0.4

        try:
            win_div=param['window']
        except:
            win_div=3

        try:
            padding=param['padding']
        except:
            padding = 'replicate'

        image = image.astype(float)
        h, w = image.shape
        window = (h // win_div, w // win_div)

        #Local mean
        mean_p = averagefilter(image, window, padding)

        # sq local mean
        mean_sq = averagefilter(image ** 2, window, padding)
        var = np.maximum(mean_sq - mean_p ** 2, 0.0)
        deviation = np.sqrt(var, dtype=np.float64)

        # Sauvola
        R = float(np.max(deviation))

        if not np.isfinite(R) or R <= 0:
            threshold = mean_p
        else:
            threshold = mean_p * (1 + k * (deviation / R - 1))

        BW = image < threshold
        return BW

    def binarize_wolf(self, image,param={},):
        try:
            k = param['k']
        except:
            k = 0.5

        try:
            win_div = param['window']
        except:
            win_div = 3

        try:
            padding = param['padding']
        except:
            padding = 'reflect'


        X = image.astype(float)
        window = (X.shape[0] // win_div, X.shape[1] // win_div)

        m = averagefilter(X, window=window, padding=padding)
        m_sq = averagefilter(X ** 2, window=window, padding=padding)
        var = np.maximum(m_sq - m ** 2, 0.0)
        std = np.sqrt(var, dtype=np.float64)

        R = float(np.max(std))
        M = np.min(X)
        thresh = (1 - k) * m + k * std / R * (m - M) + k * M
        return X < thresh

    def binarize_bradley(self, image,param={}):

        try:
            k = param['k']
        except:
            k = 0.1

        try:
            win_size = param['window']
        except:
            win_size = 15

        try:
            padding = param['padding']
        except:
            padding = 'replicate'

        window=(win_size,win_size)
        mean_p = averagefilter(image, window, padding)
        BW = np.ones_like(image, dtype=bool)
        BW[image >= mean_p * (1 - k)] = 0
        return BW

    def get_binary_from_best_band(self, algorithm="Bradley",param={},idx=None):
        """
        return a binary map from
        """
        if idx is None:
            idx = self.get_best_band()
        image = self.data[:, :, idx]

        if algorithm.lower() == "niblack":
            return self.binarize_niblack(image,param)
        elif algorithm.lower() == "otsu":
            return self.binarize_otsu(image)
        elif algorithm.lower() == "sauvola":
            return self.binarize_sauvola(image,param)
        elif algorithm.lower() == "wolf":
            return self.binarize_wolf(image,param)
        elif algorithm.lower() == "bradley":
            return self.binarize_bradley(image,param)
        else:
            raise ValueError(f"Unknown binarization algorithm: {algorithm}")

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
        self.btn_select_gtmap.clicked.connect(lambda: self._assign(self.le_gtmap))

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
        if cube_info.gtmap_path:
            self.le_gtmap.setText(cube_info.gtmap_path)

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
        """Fill the QTreeWidget with HDF5, MAT or ENVI metadata."""
        self.treeWidget.clear()
        root = self.treeWidget.invisibleRootItem()
        ext = os.path.splitext(self.filepath)[1].lower()
        self.label_filename.setText(os.path.basename(self.filepath))

        is_mat = (ext == '.mat')
        is_hdr = (ext == '.hdr')
        is_hdf5 = (ext in ('.h5', '.hdf5')) or (is_mat and self._is_hdf5_file(self.filepath))

        if is_hdr:
            try:
                img = envi.open(self.filepath)
                meta = img.metadata
                dims = (meta.get("lines"), meta.get("samples"), meta.get("bands"))
                root.addChild(QTreeWidgetItem(["DataCube", "ENVI", "<ENVI>", f"{dims}"]))
                for k, v in meta.items():
                    val = v if isinstance(v, str) else str(v)
                    node = QTreeWidgetItem([k, "Metadata", f"meta@{k}", f"{len(val)} chars"])
                    root.addChild(node)

                # disable cube selection, set wl_path if present
                self.le_cube.setText("<fixed: ENVI cube>")
                self.le_cube.setEnabled(False)

                if 'wavelength' in meta:
                    self.le_wl.setText("meta@wavelength")
                    self.cube_info.wl_path = "meta@wavelength"

                self.le_meta.setText("<ENVI metadata>")
                self.cube_info.metadata_path = "<ENVI metadata>"

            except Exception as e:
                QMessageBox.critical(None, "ENVI Error", f"Failed to read ENVI file:\n{e}")
                return

        elif is_mat and not is_hdf5:
            mat_dict = loadmat(self.filepath)
            for name, val in mat_dict.items():
                if name.startswith('__'):
                    continue
                self._add_mat_node(name, val, root, name)

        else:
            with h5py.File(self.filepath, 'r') as f:
                def recurse(group, parent, path):
                    if group.attrs:
                        attrs_node = QTreeWidgetItem(['<Attributes>', 'Group', path or '/', ''])
                        parent.addChild(attrs_node)
                        for key, val in group.attrs.items():
                            size = self._format_bytes(self._get_attr_size(val))
                            attrs_node.addChild(QTreeWidgetItem([key, 'Attribute', f"{path}@{key}", size]))

                    for name, obj in group.items():
                        full = f"{path}/{name}" if path else name
                        kind = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
                        size = self._format_bytes(self._get_obj_size(obj))
                        node = QTreeWidgetItem([name, kind, full, size])
                        parent.addChild(node)
                        if isinstance(obj, h5py.Group):
                            recurse(obj, node, full)

                recurse(f, root, "")

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
        self.cube_info.gtmap_path = self.le_gtmap.text().strip() or None

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

    @pyqtSlot(CubeInfoTemp)
    def load_from_cubeinfo(self, cube_info: CubeInfoTemp):
        """
        update CubeInfoTemp from other tools
        """
        self.cube_info = cube_info
        # Recharge l'arborescence si besoin
        self._update(self.cube_info.filepath)

        # Mets à jour les champs
        self.le_cube.setText(cube_info.data_path or "")
        self.le_wl.setText(cube_info.wl_path or "")
        self.le_meta.setText(cube_info.metadata_path or "")
        self.le_gtmap.setText(cube_info.gtmap_path or "")

        # channel wl
        mode = 'First' if cube_info.wl_trans else 'Last'
        idx = self.comboBox_channel_wl.findText(mode)
        if idx != -1:
            self.comboBox_channel_wl.setCurrentIndex(idx)

if __name__ == '__main__':

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\Samples\minicubes/'
    # fname = '00279-VNIR-mock-up.h5'
    folder=(r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\From_Cameras\Javierille 19_11_2025\Javierille 19_11_2025\white light 0, only UV')
    fname='unmix_scan_0338_calib_app.hdr'
    # fname='unmix_scan_0338.hdr'


    import os
    filepath = os.path.join(folder, fname)

    cube = Hypercube(filepath=filepath, cube_info=None, load_init=True)

