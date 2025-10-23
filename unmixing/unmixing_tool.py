# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import cv2
from PIL import Image
import h5py

import numpy as np
from PyQt5.QtWidgets import (QApplication, QSizePolicy, QSplitter,QHeaderView,QProgressBar,
                            QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QDialogButtonBox, QCheckBox, QScrollArea, QWidget, QFileDialog, QMessageBox,
                             QRadioButton,QInputDialog,QTableWidget, QTableWidgetItem,QHeaderView,
                             )

from PyQt5.QtGui import QPixmap, QImage,QGuiApplication,QStandardItemModel, QStandardItem,QColor
from PyQt5.QtCore import Qt,QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot, QRectF

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import traceback

from unmixing.unmixing_backend import*
from unmixing.unmixing_window import Ui_GroundTruthWidget
from hypercubes.hypercube import Hypercube
from interface.some_widget_for_interface import ZoomableGraphicsView
from identification.load_cube_dialog import Ui_Dialog

class LoadCubeDialog(QDialog):
    """
    Simple dialog to load exactly two cubes (VNIR + SWIR),
    show their filepaths and spectral ranges, and validate coverage softly.
    """
    def __init__(self, parent=None,vnir_cube=None, swir_cube=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Internal state
        self.cubes = {"VNIR": None, "SWIR": None}
        self._wl_ranges = {"VNIR": None, "SWIR": None}

        # Prefill from provided cubes (if any)
        def _set(kind, cube):
            if cube is None:
                return
            self.cubes[kind] = cube
            wl = getattr(cube, "wl", None)
            if isinstance(wl, np.ndarray) and wl.size > 0:
                self._wl_ranges[kind] = (float(wl.min()), float(wl.max()))

        _set("VNIR", vnir_cube)
        _set("SWIR", swir_cube)

        # Wire buttons
        self.ui.pushButton_load_cube_1.clicked.connect(lambda: self._load("VNIR"))
        self.ui.pushButton_load_cube_2.clicked.connect(lambda: self._load("SWIR"))
        self.ui.pushButton_valid.clicked.connect(self._on_accept)

        # Reset labels
        self._update_labels()

    # ---------------------- public API ----------------------
    def get_cubes(self):
        """Return (vnir_cube, swir_cube) or (None, None) if cancelled."""
        if self.exec_() == QDialog.Accepted:
            return self.cubes["VNIR"], self.cubes["SWIR"]
        return None, None

    # ---------------------- internals -----------------------
    def _load(self, kind: str):
        """
        Load VNIR or SWIR cube. On error, cleans state and UI.
        """
        start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {kind} cube",
            start_dir,
            "Hypercube files (*.mat *.h5 *.hdr)"
        )
        if not path:
            return

        cube = Hypercube()
        try:
            cube.open_hyp(default_path=path, open_dialog=False)
            # WL can be None for some sources; we tolerate that
            wl = cube.wl if cube.wl is not None else np.array([])
            self.cubes[kind] = cube
            self._wl_ranges[kind] = (float(wl.min()), float(wl.max())) if wl.size > 0 else None
        except Exception as e:
            # Robustness: clear state on failure
            QMessageBox.warning(self, "Load error",
                                f"Could not load {kind} cube:\n{e}")
            self.cubes[kind] = None
            self._wl_ranges[kind] = None

        self._update_labels()

    def _update_labels(self):
        """
        Updates filepath + spectral range labels for both cubes.
        """
        def fmt_path(c):
            return c.cube_info.filepath if c and c.cube_info and c.cube_info.filepath else "—"

        def fmt_range(r):
            if r is None:
                return "—"
            lo, hi = r
            return f"{int(round(lo))}–{int(round(hi))} nm"

        # VNIR labels
        self.ui.label_filepath_cube_1.setText(fmt_path(self.cubes["VNIR"]))
        self.ui.label_spec_range_cube_1.setText(fmt_range(self._wl_ranges["VNIR"]))

        # SWIR labels
        self.ui.label_filepath_cube_2.setText(fmt_path(self.cubes["SWIR"]))
        self.ui.label_spec_range_cube_2.setText(fmt_range(self._wl_ranges["SWIR"]))

        # Instruction summary
        instr = self._coverage_message()
        self.ui.label_instructions.setText(instr)

    def _coverage_message(self) -> str:
        """
        Build a human-readable coverage message. Does not block anything.
        """
        vnir = self._wl_ranges["VNIR"]
        swir = self._wl_ranges["SWIR"]

        parts = []
        if vnir:
            parts.append(f"VNIR: {int(vnir[0])}–{int(vnir[1])} nm")
        else:
            parts.append("VNIR: —")

        if swir:
            parts.append(f"SWIR: {int(swir[0])}–{int(swir[1])} nm")
        else:
            parts.append("SWIR: —")

        # Expected nominal coverage (soft check)
        expected = (400, 1700)
        covered_lo = min([vnir[0] for vnir in [vnir] if vnir] + [swir[0] for swir in [swir] if swir], default=None)
        covered_hi = max([vnir[1] for vnir in [vnir] if vnir] + [swir[1] for swir in [swir] if swir], default=None)

        if covered_lo is not None and covered_hi is not None:
            covered = f"Currently covered: {int(covered_lo)}–{int(covered_hi)} nm"
        else:
            covered = "Currently covered: —"

        return (
            "<html><body><p align='center'>"
            + " | ".join(parts)
            + "<br>"
            + covered
            + "</p></body></html>"
        )

    def _soft_validate_and_warn(self) -> None:
        """
        Show a non-blocking warning if coverage is partial, missing wl, or cubes missing.
        """
        vnir = self._wl_ranges["VNIR"]
        swir = self._wl_ranges["SWIR"]

        missing = []
        if self.cubes["VNIR"] is None:
            missing.append("VNIR")
        if self.cubes["SWIR"] is None:
            missing.append("SWIR")

        if missing:
            QMessageBox.information(
                self, "Heads-up",
                f"The whole spectral range is not covered.\n"
                f"You can still proceed, but classification performance may be degraded."
            )
            return

        # Any wl missing?
        if vnir is None or swir is None:
            QMessageBox.information(
                self, "Spectral coverage",
                "At least one cube has no wavelength axis available.\n"
                "You can proceed, but classification performance may be degraded."
            )
            return

        # Soft range check: nominal expectation only
        nominal = (400, 1700)
        lo = min(vnir[0], swir[0])
        hi = max(vnir[1], swir[1])
        # If there is a gap between VNIR and SWIR
        gap = (vnir[1] + 1) < swir[0] or (swir[1] + 1) < vnir[0]
        if lo > nominal[0] or hi < nominal[1] or gap:
            QMessageBox.information(
                self, "Spectral coverage",
                "The spectral range does not fully cover the nominal 400–1700 nm, "
                "or there is a gap between VNIR and SWIR.\n\n"
                "You can proceed, but classification performance may be reduced."
            )

    def _on_accept(self):
        # Only warn (never block)
        self._soft_validate_and_warn()
        self.accept()

def _safe_name_from(cube) -> str:
    md = getattr(cube, "metadata", {}) or {}
    if isinstance(md, dict):
        name = md.get("name")
        if name:
            return str(name)
    ci = getattr(cube, "cube_info", None)
    mt = getattr(ci, "metadata_temp", None) if ci else None
    if isinstance(mt, dict):
        name = mt.get("name")
        if name:
            return str(name)
    fp = getattr(ci, "filepath", "") if ci else ""
    if fp:
        path=os.path.splitext(os.path.basename(fp))[0]
        name=path.split('/')[-1]
        return name
    return "unknown"

def _safe_filename_from(cube) -> str:
    ci = getattr(cube, "cube_info", None)
    fp = getattr(ci, "filepath", "") if ci else ""
    return os.path.basename(fp) if fp else ""

def fused_cube(cube1, cube2, *, copy_common_meta: bool = True):
    """
    Return a fused Hypercube (VNIR+SWIR) with provenance metadata:
      metadata['source_roles'] = ['VNIR','SWIR' or single role]
      metadata['source_files'] = [...]
      metadata['source_names'] = [...]
    Also crops to VNIR[400–950] and SWIR[955–1700] before concatenation.
    """
    if cube1 is None and cube2 is None:
        raise ValueError("fused_cube: at least one cube is required")

    # Decide roles from wavelength starts
    if cube1 is None or cube2 is None:
        parent = cube1 or cube2
        fused = parent.__class__(data=parent.data.copy(), wl=parent.wl.copy(), cube_info=parent.cube_info)
        fused.metadata = dict(getattr(parent, "metadata", {}) or {})
        role = "VNIR" if (cube1 is not None and cube1.wl[0] < 800) or (cube2 is None and parent.wl[0] < 800) else "SWIR"
        fused.metadata["source_roles"]  = [role]
        fused.metadata["source_files"]  = [_safe_filename_from(parent)]
        fused.metadata["source_names"]  = [_safe_name_from(parent)]
        return fused

    # Classify inputs as VNIR/SWIR with a simple heuristic
    if cube1.wl[0] < 500 and cube2.wl[0] > 800:
        VNIR, SWIR = cube1, cube2
    elif cube1.wl[0] > 800 and cube2.wl[0] < 500:
        VNIR, SWIR = cube2, cube1
    else:
        raise ValueError("fused_cube: could not infer VNIR/SWIR from wavelength ranges")

    target = {'VNIR': (400, 950), 'SWIR': (955, 1700)}

    # Crop both ranges
    hyps_cut = {}
    for role, cube in (("VNIR", VNIR), ("SWIR", SWIR)):
        wl = cube.wl
        data = cube.data
        start_idx = int(np.argmin(np.abs(wl - target[role][0])))
        end_idx   = int(np.argmin(np.abs(wl - target[role][1])))
        data_cut = data[:, :, start_idx:end_idx + 1]
        wl_cut   = wl[start_idx:end_idx + 1]
        hyps_cut[role] = Hypercube(data=data_cut, wl=wl_cut, cube_info=cube.cube_info)

    # Spatial check
    h1, w1, _ = hyps_cut["VNIR"].data.shape
    h2, w2, _ = hyps_cut["SWIR"].data.shape
    if (h1 != h2) or (w1 != w2):
        raise ValueError(f"fused_cube: incompatible spatial dims VNIR={h1}x{w1}, SWIR={h2}x{w2}")

    data_fused = np.concatenate((hyps_cut['VNIR'].data, hyps_cut['SWIR'].data), axis=2)
    wl_fused   = np.concatenate((hyps_cut['VNIR'].wl,   hyps_cut['SWIR'].wl))

    fused = Hypercube(data=data_fused, wl=wl_fused, cube_info=VNIR.cube_info)
    fused.metadata = {}

    if copy_common_meta:
        for src in (VNIR, SWIR):
            md = dict(getattr(src, "metadata", {}) or {})
            for k, v in md.items():
                fused.metadata.setdefault(k, v)

    # Provenance (lists for HDF5-friendliness)
    fused.metadata["source_roles"]  = ["VNIR", "SWIR"]
    fused.metadata["source_files"]  = [_safe_filename_from(VNIR), _safe_filename_from(SWIR)]
    fused.metadata["source_names"]  = [_safe_name_from(VNIR),     _safe_name_from(SWIR)]

    return fused

# ------------------------------- Signals --------------------------------------
class UnmixingSignals(QObject):
    error = pyqtSignal(str)
    info = pyqtSignal(str)
    progress = pyqtSignal(int)
    # Results
    em_ready = pyqtSignal(np.ndarray, object, dict)  # E, labels, index_map
    unmix_ready = pyqtSignal(np.ndarray, np.ndarray, dict)  # A, E, maps_by_group

# ------------------------------- Workers --------------------------------------
@dataclass
class EndmemberJob:
    method: str  # 'Manual' | 'ATGP' | 'N-FINDR' | 'Library'
    p: int
    niter: int
    normalization: str  # 'None'|'L2'|'L1'
    # inputs
    manual_groups: Optional[Dict[str, np.ndarray]] = None  # {name: (L,p_g)}
    library_groups: Optional[Dict[str, np.ndarray]] = None  # same shape

class EndmemberWorker(QRunnable):
    def __init__(self, job: EndmemberJob,cube):
        super().__init__()
        self.job = job
        self.signals = UnmixingSignals()
        self.cube=cube

    @pyqtSlot()
    def run(self):
        try:
            # Normalize cube if needed
            data = normalize_cube(self.cube.data, mode=self.job.normalization)
            print('[ENDMEMBERS] work started with ',self.job.method)

            if self.job.method == 'Manual' and self.job.manual_groups:
                E, labels, index_map = build_dictionary_from_groups(self.job.manual_groups)
            elif self.job.method == 'Library' and self.job.library_groups:
                E, labels, index_map = build_dictionary_from_groups(self.job.library_groups)
            elif self.job.method == 'ATGP':
                E = extract_endmembers_atgp(data, self.job.p)
                labels = np.array([f"EM_{i:02d}" for i in range(E.shape[1])], dtype=object)
                index_map = {str(lbl): np.array([i]) for i, lbl in enumerate(labels)}
            elif self.job.method == 'N-FINDR':
                # In PySptools, `maxit` is a small integer; use job.niter
                E = extract_endmembers_nfindr(data, self.job.p, maxit=max(1, int(self.job.niter)))
                labels = np.array([f"EM_{i:02d}" for i in range(E.shape[1])], dtype=object)
                index_map = {str(lbl): np.array([i]) for i, lbl in enumerate(labels)}
            else:
                raise ValueError("Invalid endmember extraction settings.")

            # Match normalization if any groups were given not already normalized
            if self.job.normalization and self.job.normalization.lower() != 'none':
                # When E came from groups, assume caller already normalized consistently.
                # When E came from ATGP/N-FINDR (from the normalized cube), it's fine.
                pass

            self.signals.em_ready.emit(E, labels, index_map)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"Endmember extraction failed: {e}\n{tb}")

@dataclass
class UnmixJob:
    name : str # unique key shown in table
    model: str  # 'UCLS'|'NNLS'|'FCLS'|'SUnSAL'
    normalization: str  # 'None'|'L2'|'L1'
    max_iter: int
    tol: float
    # SUnSAL specific
    lam: float = 1e-3
    rho: float = 1.0
    anc: bool = True
    asc: bool = False
    # inputs
    E = None  # (L,p)
    rect : Optional[np.ndarray] = None  # (H,W) boolean (optional)
    progress: int = 0         # 0..100
    _t0: Optional[float] = field(default=None, repr=False)  # internal start time
    duration_s: Optional[float] = None    # whole classification duration
    abundance_maps: Optional[np.ndarray] = None     # raw classification map
    clean_map: Optional[np.ndarray] = None  # cleaned classification map
    clean_param = None  # clean parameters

class UnmixWorker(QRunnable):
    def __init__(self, job: UnmixJob):
        super().__init__()
        self.job = job
        self.signals = UnmixingSignals()

    @pyqtSlot()
    def run(self):
        try:
            H, W, L = self.job.cube.shape
            cube = normalize_cube(self.job.cube, mode=self.job.normalization)
            Y = vectorize_cube(cube)  # (L,N)

            # Optionally restrict to ROI
            if self.job.roi_mask is not None:
                mask = self.job.roi_mask.astype(bool).ravel()
                Y_work = Y[:, mask]
            else:
                mask = None
                Y_work = Y

            # Call selected solver
            model = self.job.model.upper()
            if model == 'UCLS':
                A_work = unmix_ucls(self.job.E, Y_work)
            elif model == 'NNLS':
                A_work = unmix_nnls(self.job.E, Y_work)
            elif model == 'FCLS':
                A_work = unmix_fcls(self.job.E, Y_work)
            elif model == 'SUNSAL':
                A_work = unmix_sunsal(
                    self.job.E, Y_work,
                    lam=self.job.lam,
                    positivity=self.job.anc or self.job.asc,
                    sum_to_one=self.job.asc,
                    rho=self.job.rho,
                    max_iter=self.job.max_iter,
                    tol=self.job.tol,
                )
            else:
                raise ValueError(f"Unknown model: {self.job.model}")

            # Re-inject into full image if ROI was used
            p = self.job.E.shape[1]
            A = np.zeros((p, H * W), dtype=A_work.dtype)
            if mask is None:
                A = A_work
            else:
                A[:, mask] = A_work

            # Build maps by group (if labels provided)
            maps_by_group = {}
            if self.job.labels is not None:
                maps_by_group = abundance_maps_by_group(A, self.job.labels, H, W)

            self.signals.unmix_ready.emit(A, self.job.E, maps_by_group)
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"Unmixing failed: {e}\n{tb}")

# ------------------------------- Main Widget ----------------------------------
class UnmixingTool(QWidget,Ui_GroundTruthWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Queue structures
        self.job_order: List[str] = []  # only job names in execution order
        self.jobs: Dict[str, UnmixJob] = {}  # name -> job

        # Table init
        # self._init_classification_table(self.tableWidget_classificationList)
        # self._init_cleaning_list()

        # Unmix as thread init
        self.threadpool = QThreadPool()
        self._running_idx: int = -1
        self._current_worker = None
        self._stop_all = False
        self.only_selected = False
        self.signals = UnmixingSignals()

        # Remplacer placeholders par ZoomableGraphicsView
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)
        self.viewer_left.enable_rect_selection = True
        self.viewer_right.enable_rect_selection = False # no rectangle selection for right view

        # for manual selection
        self.selecting_pixels = False  # mode selection ref activated
        self._pixel_selecting = False  # for manual pixel selection for dragging mode
        self.erase_selection = False  # erase mode on or off
        self._pixel_coords = []  # collected  (x,y) during dragging
        self._preview_mask = None  # temp mask during dragging pixel selection
        self._last_label = 0  # default to 0
        self.last_class_number = 3
        self.nclass_box.setValue(self.last_class_number)
        self.class_info = {}  # dictionnary of lists :  {key:[label, name GT,(R,G,B)]}
        self.class_colors = {}  # color of each class

        # data variable
        self._last_vnir = None
        self._last_swir = None
        self.cube = None
        self.data = None
        self.wl = None
        self.whole_range = None
        self.saved_rec = None
        self.alpha = self.horizontalSlider_overlay_transparency.value() / 100.0

        self.E : Dict[str,np.ndarray]  # {EN_i : (L,pi)}
        self.index_map: Optional[Dict[str, np.ndarray]] = None
        self.A: Optional[np.ndarray] = None  # (p, N)
        self.maps_by_group: Dict[str, np.ndarray] = {}

        # connection
        self.load_btn.clicked.connect(self.open_load_cube_dialog)

        self.horizontalSlider_overlay_transparency.valueChanged.connect(self.update_alpha)


        self.sliders_rgb = [
            self.horizontalSlider_red_channel,
            self.horizontalSlider_green_channel,
            self.horizontalSlider_blue_channel
        ]
        self.spinBox_rgb = [
            self.spinBox_red_channel,
            self.spinBox_green_channel,
            self.spinBox_blue_channel
        ]

        for sl, sp in zip(self.sliders_rgb, self.spinBox_rgb):
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(self.show_rgb_image)  # Mise à jour image si on change

        # --- Radios pour mode RGB/Gris ---
        self.radioButton_rgb_user.toggled.connect(self.update_rgb_controls)
        self.radioButton_rgb_default.toggled.connect(self.update_rgb_controls)
        self.radioButton_grayscale.toggled.connect(self.update_rgb_controls)
        self.radioButton_rgb_default.setChecked(True)
        self.update_rgb_controls()

        # --- Boutons de rotation / flip ---
        self.pushButton_rotate.clicked.connect(lambda: self.transform(np.rot90))
        self.pushButton_flip_h.clicked.connect(lambda: self.transform(np.fliplr))
        self.pushButton_flip_v.clicked.connect(lambda: self.transform(np.flipud))

        #Endmembers window
        self.run_btn.clicked.connect(self._on_extract_endmembers)
        self.pushButton_load_EM.clicked.connect(self._on_load_library_clicked)

        # Defaults values algorithm
        self.comboBox_unmix_algorithm.setCurrentText('SUnSAL')
        self.doubleSpinBox_unmix_lambda_3.setValue(-3.0)  # log10 lambda
        self.doubleSpinBox_unmix_lambda_2.setValue(-4.0)  # log10 tol
        self.doubleSpinBox_unmix_lambda_4.setValue(1.0)   # rho
        self.checkBox_unmix_ANC.setChecked(True)
        self.checkBox_unmix_ASC.setChecked(False)

    # <editor-fold desc="Visual elements">

    def reset_all(self):

        reply = QMessageBox.question(
            self, "Reset all ?",
            f"Are you sure to reset all current jobs ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Réinit internes
        self.job_order = []
        self.jobs = {}
        self._init_classification_table(self.tableWidget_classificationList)
        self._init_cleaning_list()

        self.threadpool = QThreadPool()
        self._running_idx = -1
        self._current_worker = None
        self._stop_all = False
        self.only_selected = False

        self.comboBox_clas_show_model.clear()
        self._refresh_show_model_combo()

        self._last_vnir = None
        self._last_swir = None
        self.cube = None
        self.data = None
        self.wl = None
        self.saved_rec = None
        self.whole_range = None

        # Vider les viewers
        if hasattr(self, "viewer_left"):
            self.viewer_left.setImage(QPixmap())  # plus d’image à gauche
        if hasattr(self, "viewer_right"):
            self.viewer_right.setImage(QPixmap())  # plus d’image à droite

        self.update_legend()
        self._set_info_rows()

    def bgr_to_rgb(self, bgr):
        return (bgr[2], bgr[1], bgr[0])

    def _replace_placeholder(self, name, widget_cls, **kwargs):
        placeholder = getattr(self, name)
        parent = placeholder.parent()
        # Determine if parent is a layout container or a splitter
        if parent.layout() is not None:
            layout = parent.layout()
            idx = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)
            layout.insertWidget(idx, widget)
        elif isinstance(parent, QSplitter):
            # Handle QSplitter parent
            idx = parent.indexOf(placeholder)
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)
            parent.insertWidget(idx, widget)
        else:
            # Fallback: direct replacement
            placeholder.deleteLater()
            widget = widget_cls(**kwargs) if kwargs else widget_cls()
            setattr(self, name, widget)

    def update_rgb_controls(self):
        """Active/désactive sliders/spinbox selon mode RGB choisi"""
        if self.data is None or self.wl is None:
            return
        max_wl = int(self.wl[-1])
        min_wl = int(self.wl[0])
        wl_step = int(self.wl[1] - self.wl[0])

        default = self.radioButton_rgb_default.isChecked()

        if self.radioButton_grayscale.isChecked():
            self.label_red_channel.setText('')
            self.label_green_channel.setText('')
            self.label_blue_channel.setText('Gray')
        else:
            self.label_red_channel.setText('Red')
            self.label_green_channel.setText('Green')
            self.label_blue_channel.setText('Blue')

        for i, element in enumerate(self.sliders_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(int(self.default_rgb_channels()[i]))
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                element.setEnabled(i == 2)
            else:
                element.setEnabled(True)

        for i, element in enumerate(self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(int(self.default_rgb_channels()[i]))
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                element.setEnabled(i == 2)
            else:
                element.setEnabled(True)

        self.show_rgb_image()

    def default_rgb_channels(self):
        """Renvoie les canaux RGB par défaut selon plage spectrale"""
        if self.wl[-1] < 1100 and self.wl[0] > 350:
            return [610, 540, 435]
        elif self.wl[-1] >= 1100 and self.wl[0] > 800:
            return [1605, 1205, 1005]
        else:
            if len(self.wl) > 7:
                sixieme = len(self.wl) // 6
                idx1, idx2, idx3 = sixieme, 3 * sixieme, 5 * sixieme
                return [self.wl[idx3], self.wl[idx2], self.wl[idx1]]
            else:
                mid = int(len(self.wl) / 2)
                return [self.wl[-1], self.wl[mid], self.wl[0]]

    def transform(self, trans_type):
        """Rotation / Flip du cube"""
        try:
            self.data = trans_type(self.data)
            self.cube.data = self.data

            # binary_map peut être None selon l’état de l’outil
            if getattr(self, "binary_map", None) is not None:
                bm = trans_type(self.binary_map)
                # Garantir une carte 2D (H, W)
                if bm.ndim == 3 and bm.shape[-1] == 1:
                    bm = bm[..., 0]
                self.binary_map = bm

            if getattr(self, "class_map", None) is not None:
                self.class_map = trans_type(self.class_map)

            for job in self.jobs.values():
                if hasattr(job, "_mask_indices"):
                    job._mask_indices = None
                if hasattr(job, "_shape"):
                    job._shape = None
                if getattr(job, "class_map", None) is not None:
                    job.class_map = trans_type(job.class_map)

        except Exception as e:
            print(f"[transform] Failed: {e}")
            return

        self.show_rgb_image()
        self.update_overlay

    def _np2pixmap(self, img):
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

    def _draw_current_rect(self, *, use_job=False, surface=False):
        """
        Dessine le rectangle de sélection sur les deux viewers.
        - use_job=True : prend le rect du job actuellement sélectionné dans la combo.
        - sinon : prend self.saved_rec.
        - surface=False : seulement le contour (plus discret pour l’overlay).
        """
        rect_tuple = None
        if use_job:
            idx = self.comboBox_clas_show_model.currentIndex()
            if idx >= 0:
                name = (self.comboBox_clas_show_model.itemData(idx, Qt.UserRole)
                        or self.comboBox_clas_show_model.currentText())
                job = self.jobs.get(name)
                if job:
                    rect_tuple = job.rect
        if rect_tuple is None and not use_job:
            rect_tuple = self.saved_rec

        qrect = self._rect_to_qrectf(rect_tuple)
        if qrect is None:
            # Nettoyer d’anciens overlays s’il y en a
            if hasattr(self.viewer_left, "clear_selection_overlay"): self.viewer_left.clear_selection_overlay()
            if hasattr(self.viewer_right, "clear_selection_overlay"): self.viewer_right.clear_selection_overlay()
            return

        # Affiche sur chaque viewer (méthode dispo dans ZoomableGraphicsView)
        self.viewer_left.add_selection_overlay(qrect, surface=surface)
        self.viewer_right.add_selection_overlay(qrect, surface=surface)

    def _get_selected_rect(self):
        """
        Renvoie (y, x, h, w) en indices numpy si un rectangle est sélectionné
        dans viewer_left, sinon None.
        """
        rc = getattr(self.viewer_left, "get_rect_coords", None)
        if not callable(rc):
            return None
        coords = rc()
        if coords is None:
            return None
        x_min, y_min, w, h = coords
        return (y_min, x_min, h, w)

    def _rect_to_qrectf(self, rect_tuple):
        if not rect_tuple:
            return None
        y, x, h, w = rect_tuple
        return QRectF(float(x), float(y), float(w), float(h))

    def show_rgb_image(self):

        if self.data is None:
            return
        if self.radioButton_rgb_default.isChecked():
            rgb_chan = self.default_rgb_channels()
        elif self.radioButton_grayscale.isChecked():
            val = self.spinBox_blue_channel.value()
            rgb_chan = [val, val, val]
        else:  # RGB user
            rgb_chan = [
                self.spinBox_red_channel.value(),
                self.spinBox_green_channel.value(),
                self.spinBox_blue_channel.value()
            ]
        idx = [np.argmin(np.abs(ch - self.wl)) for ch in rgb_chan]
        rgb = self.data[:, :, idx].astype(np.float32)
        max_val = float(np.max(rgb)) if rgb.size else 0.0
        if max_val > 0:
            rgb = (rgb / max_val) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        self.rgb_image = rgb

        self.viewer_left.setImage(self._np2pixmap(rgb))
        self.viewer_left.fitImage()
        self._draw_current_rect(surface=False)

    def update_overlay(self,only_images=False):
        # self.show_classification_result(only_images)
        pass

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.update_overlay(only_images=True)
    # </editor-fold>

    # <editor-fold desc="cube">
    def open_load_cube_dialog(self):

        if self.no_reset_jobs_on_new_cube():
            return

        dlg = LoadCubeDialog(self, vnir_cube=self._last_vnir, swir_cube=self._last_swir)
        if dlg.exec_() == QDialog.Accepted:
            # Prefer fusing VNIR+SWIR if both available; otherwise passthrough a single cube
            vnir = dlg.cubes.get("VNIR")
            swir = dlg.cubes.get("SWIR")

            self._last_vnir = vnir
            self._last_swir = swir

            if vnir is not None or swir is not None:
                self.cube = fused_cube(vnir, swir)
            else:
                QMessageBox.warning(self, "Error", "No cube loaded.")
                return

            # try:
            #     self.cube.normalize_spectral(self.TRAIN_WL, min_wl=400, max_wl=1700, interp_kind="linear",
            #                                  in_place=True)
            # except ValueError as e:
            #     QMessageBox.warning(self, "Spectral error", str(e))
            #     return

            # Ensure UI buffers are in sync
            self.data = self.cube.data
            self.wl = self.cube.wl
            self.update_rgb_controls()
            self.update_overlay()

    def load_cube(self, filepath=None, cube=None, cube_info=None, range=None):

        if self.no_reset_jobs_on_new_cube():
            return

        flag_loaded = False
        if cube is not None:
            try:
                if cube_info is not None:
                    cube.metadata = cube.cube_info.metadata_temp

                if range is None:
                    self.cube = cube
                    flag_loaded = True
                else:
                    print('Range : ', range)
                    if range == 'VNIR':
                        self._last_vnir = cube
                        self.cube = fused_cube(self._last_vnir, self._last_swir)

                        flag_loaded = True

                    elif range == 'SWIR':
                        self._last_swir = cube
                        self.cube = fused_cube(self._last_vnir, self._last_swir)

                        flag_loaded = True

                    else:
                        print('Problem with cube range in parameter')

            except:
                print('Problem with cube in parameter')

        if not flag_loaded:
            if not filepath:
                filepath, _ = QFileDialog.getOpenFileName(
                    self, "Open Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
                )
                if not filepath:
                    return
            try:
                cube = Hypercube(filepath=filepath, load_init=True)

                try:
                    if cube_info is not None:
                        cube.metadata = cube.cube_info.metadata_temp

                    if range is None:
                        self.cube = cube
                    else:
                        if range == 'VNIR':
                            self._last_vnir = cube
                            if self._last_swir is not None:
                                self.cube = fused_cube(self._last_vnir, self._last_swir)
                            else:
                                self.cube = cube

                        elif range == 'SWIR':
                            self._last_swir = cube
                            if self._last_vnir is not None:
                                self.cube = fused_cube(self._last_vnir, self._last_swir)
                            else:
                                self.cube = cube

                        else:
                            print('Problem with cube range in parameter')

                except:
                    print('Problem with cube in parameter')

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load cube: {e}")
                return

        # test spectral range

        self.data = self.cube.data
        self.wl = self.cube.wl
        self.update_rgb_controls()
        self.show_rgb_image()
        self.update_overlay()

    def no_reset_jobs_on_new_cube(self):
        for job in self.jobs:
            if self.jobs[job].status in ['Done', 'Running']:
                reply = QMessageBox.question(
                    self, "Reset Jobs ?",
                    f"Some jobs have been done and will be loosed.\n Are you sure to continue ? ",
                    QMessageBox.Yes | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    return True

                self.remove_all_jobs(ask_confirm=False)
                return False

        return False
    # </editor-fold>

    # <editor-fold desc="Endmembers">

    def _on_extract_endmembers(self):
        if self.cube is None:
            QMessageBox.warning(self, 'Unmixing', 'Load a cube first.')
            return
        method = self.comboBox_endmembers_get.currentText()
        print('[ENDMEMBERS] algorithm : ',method)
        p = int(self.nclass_box.value())
        niter = int(self.niter_box.value())
        norm = self._current_normalization()

        # Gather manual/library groups from host (signals or attributes)
        manual_groups = getattr(self, 'manual_groups', None)
        library_groups = getattr(self, 'library_groups', None)

        job = EndmemberJob(method=method, p=p, niter=niter, normalization=norm,
                           manual_groups=manual_groups,
                           library_groups=library_groups)

        worker = EndmemberWorker(job,self.cube)
        worker.signals.em_ready.connect(self._on_em_ready)
        worker.signals.error.connect(self._on_error)
        self.threadpool.start(worker)

    def _on_em_ready(self, E: np.ndarray, labels: np.ndarray, index_map: Dict[str, np.ndarray]):
        self.E, self.labels, self.index_map = E, labels, index_map
        # Populate EM combo for visualization (one by one)
        self.comboBox_viz_show_EM.clear()
        for i in range(E.shape[1]):
            self.comboBox_viz_show_EM.addItem(f"EM {i:02d}")
        # If you also have groups (labels), you can populate comboBox_viz_show_model too
        print('[ENDMEMBERS]',
            f"Endmembers ready: E shape {E.shape}, groups: {len(np.unique(labels)) if labels is not None else 0}")

    def _on_load_library_clicked(self):
        # Expect host app to set self.library_groups from a file dialog elsewhere
        if not getattr(self, 'library_groups', None):
            QMessageBox.information(self, 'Library', 'No library groups loaded in tool instance.')
        else:
            QMessageBox.information(self, 'Library', f"Library groups found: {list(self.library_groups.keys())}")

    def _on_error(self, msg: str):
        self.signals.error.emit(msg)
        print('[ERROR] : ',msg)

    # </editor-fold>

    # <editor-fold desc="Processing Data Helpers">
    def _current_normalization(self) -> str:
        txt = self.comboBox_normalisation.currentText().lower()
        if 'l2' in txt:
            return 'L2'
        if 'l1' in txt:
            return 'L1'
        return 'none'
    # </editor-fold>

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = UnmixingTool()
    w.show()

    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\identification/'
    fname1 = '01644-VNIR-genealogies.h5'
    fname2 = '01677-SWIR-genealogies.h5'

    import os
    filepath1 = os.path.join(folder, fname1)
    filepath2 = os.path.join(folder, fname2)
    cube=fused_cube(Hypercube(filepath1,load_init=True),Hypercube(filepath2,load_init=True))
    w.load_cube(cube=cube)
    sys.exit(app.exec_())
