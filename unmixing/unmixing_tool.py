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
                             QRadioButton,QInputDialog,QTableWidget, QTableWidgetItem,QHeaderView,QGraphicsView
                             )

from PyQt5.QtGui import QPixmap, QImage,QGuiApplication,QStandardItemModel, QStandardItem,QColor
from PyQt5.QtCore import Qt,QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot, QRectF,QEvent,QRect, QPoint, QSize


# Graphs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib import colormaps
from matplotlib.path import Path

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import traceback

from unmixing.unmixing_backend import*
from unmixing.unmixing_window import Ui_GroundTruthWidget
from hypercubes.hypercube import Hypercube
from interface.some_widget_for_interface import ZoomableGraphicsView
from identification.load_cube_dialog import Ui_Dialog

# <editor-fold desc="To do">
#todo : from librarie -> gerer les wl_lib et wl (cube)
#todo : unmixing -> select endmmembers AND if merge
#todo : viz spectra -> show/hide by clicking line or title (or ctrl+click)
#todo : save manual or auto EM selection
#todo : add ban selection widget
#todo : add ROI
#todo : add one pixel fusion

# </editor-fold>

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
        self.viewer_left.enable_rect_selection = False  # no rectangle selection for left view
        self.viewer_right.enable_rect_selection = False # no rectangle selection for right view
        self.viewer_left.viewport().installEventFilter(self)
        self.viewer_left.viewport().setMouseTracking(True)
        self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)

        # Promote spec_canvas placeholder to FigureCanvas
        self._promote_canvas('spec_canvas', FigureCanvas)
        self.spec_canvas_layout = self.spec_canvas.layout() if hasattr(self.spec_canvas, 'layout') else None
        self.init_spectrum_canvas()
        self.show_selection = True

        # for manual selection
        self.samples={}
        self.sample_coords = {}
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
        self.selected_bands=[] #band selection
        self.selected_span_patch=[] # rectangle patch of selected bands
        self.selection_mask_map = None

        # data variable
        self._last_vnir = None
        self._last_swir = None
        self.cube = None
        self.data = None
        self.wl = None
        self.whole_range = None
        self.saved_rec = None
        self.horizontalSlider_overlay_transparency.setValue(70)
        self.alpha = self.horizontalSlider_overlay_transparency.value() / 100.0

        #endmembers
        self.E = {} # {EN_i : (L,pi)}
        self.regions = {}  # dict: classe -> [ {'coords': set[(x,y)], 'mean': np.ndarray}, ... ]

        self.E_manual= {}
        self.wl_manual=None

        self.E_lib=Dict[str,np.ndarray]
        self.wl_lib=None

        self.E_auto=Dict[str,np.ndarray]
        self.wl_auto=None

        self.class_means = {}  # for spectra of classe
        self.class_stds = {}  # for spectra of classe
        self.class_ncount = {}  # for npixels classified
        self.class_colors ={}  # color of each class

        self.cls_map = None
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
        self.pushButton_class_selection.toggled.connect(self.on_toggle_selection)
        self.pushButton_erase_selected_pix.toggled.connect(self.on_toggle_erase)

        # Spectra window
        self.comboBox_endmembers_spectra.currentIndexChanged.connect(self.on_changes_EM_spectra_viz)
        self.checkBox_showLegend.toggled.connect(self.update_spectra)

        # Unmix window


        # Defaults values algorithm
        self.comboBox_unmix_algorithm.setCurrentText('SUnSAL')
        self.doubleSpinBox_unmix_lambda_3.setValue(-3.0)  # log10 lambda
        self.doubleSpinBox_unmix_lambda_2.setValue(-4.0)  # log10 tol
        self.doubleSpinBox_unmix_lambda_4.setValue(1.0)   # rho
        self.checkBox_unmix_ANC.setChecked(True)
        self.checkBox_unmix_ASC.setChecked(False)

        self._sunsal_last = {
            'anc': self.checkBox_unmix_ANC.isChecked(),
            'asc': self.checkBox_unmix_ASC.isChecked(),
            'merge': self.checkBox_unmix_merge_EM_groups.isChecked()
        }
        self.checkBox_unmix_ANC.toggled.connect(self._remember_sunsal)
        self.checkBox_unmix_ASC.toggled.connect(self._remember_sunsal)
        self.checkBox_unmix_merge_EM_groups.toggled.connect(self._remember_sunsal)

        self.comboBox_unmix_algorithm.currentIndexChanged.connect(self.on_unmix_algo_change)
        self.on_unmix_algo_change()

        self.comboBox_endmembers_get.currentIndexChanged.connect(self.on_algo_endmember_change)
        self.on_algo_endmember_change()

        # Job Queue
        self._init_classification_table(self.tableWidget_classificationList)
        self.pushButton_launch_unmixing.clicked.connect(self._on_add_unmix_job)
        self.pushButton_clas_remove.clicked.connect(self.remove_selected_job)
        self.pushButton_clas_remove_all.clicked.connect(self.remove_all_jobs)
        self.pushButton_clas_up.clicked.connect(lambda: self._move_job(-1))
        self.pushButton_clas_down.clicked.connect(lambda: self._move_job(+1))

        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(2, 5)
        self.splitter.setStretchFactor(1, 2)

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
        self.update_overlay()

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

    def _remember_sunsal(self):
        if self.comboBox_unmix_algorithm.currentText().upper() == 'SUNSAL':
            self._sunsal_last['anc'] = self.checkBox_unmix_ANC.isChecked()
            self._sunsal_last['asc'] = self.checkBox_unmix_ASC.isChecked()
            self._sunsal_last['merge'] = self.checkBox_unmix_merge_EM_groups.isChecked()

    def on_unmix_algo_change(self):
        algo = self.comboBox_unmix_algorithm.currentText().upper()

        # Raccourcis
        anc = self.checkBox_unmix_ANC
        asc = self.checkBox_unmix_ASC
        mrg = self.checkBox_unmix_merge_EM_groups

        # Paramètres numériques (SUnSAL uniquement)
        lam3 = self.doubleSpinBox_unmix_lambda_3
        lam2 = self.doubleSpinBox_unmix_lambda_2
        lam4 = self.doubleSpinBox_unmix_lambda_4
        maxit = self.spinBox

        # Bloquer les signaux pendant le changement
        for w in (anc, asc, mrg, lam3, lam2, lam4, maxit):
            w.blockSignals(True)

        try:
            if algo == 'UCLS':
                # UCLS = unconstrained least squares
                anc.setChecked(False)
                asc.setChecked(False)
                anc.setEnabled(False)
                asc.setEnabled(False)

                # Merge option libre
                mrg.setEnabled(True)

                # Paramètres SUnSAL désactivés
                for w in (lam3, lam2, lam4, maxit):
                    w.setEnabled(False)

            elif algo == 'NNLS':
                # NNLS = ANC uniquement
                anc.setChecked(True)
                asc.setChecked(False)
                anc.setEnabled(False)
                asc.setEnabled(False)
                mrg.setEnabled(True)

                for w in (lam3, lam2, lam4, maxit):
                    w.setEnabled(False)

            elif algo == 'FCLS':
                # FCLS = ANC + ASC
                anc.setChecked(True)
                asc.setChecked(True)
                anc.setEnabled(False)
                asc.setEnabled(False)
                mrg.setEnabled(True)

                for w in (lam3, lam2, lam4, maxit):
                    w.setEnabled(False)

            elif algo == 'SUNSAL':
                # SUnSAL : ANC/ASC libres + paramètres activés
                anc.setEnabled(True)
                asc.setEnabled(True)
                mrg.setEnabled(True)

                # Restaurer les derniers choix utilisateur
                anc.setChecked(bool(self._sunsal_last.get('anc', True)))
                asc.setChecked(bool(self._sunsal_last.get('asc', False)))
                mrg.setChecked(bool(self._sunsal_last.get('merge', mrg.isChecked())))

                for w in (lam3, lam2, lam4, maxit):
                    w.setEnabled(True)

            else:
                # Cas inattendu
                anc.setEnabled(False)
                asc.setEnabled(False)
                mrg.setEnabled(True)
                for w in (lam3, lam2, lam4, maxit):
                    w.setEnabled(False)

        finally:
            for w in (anc, asc, mrg, lam3, lam2, lam4, maxit):
                w.blockSignals(False)

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
        self.update_overlay()

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

    def _make_rgb_from_cube(self) -> np.ndarray:
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

    def show_rgb_image(self):
        self.update_overlay(recompute_rgb=True, preview=False)

    def update_overlay(self, recompute_rgb: bool = False, preview: bool = False):
        """
        1) (optionnel) Recalcule self.rgb_image depuis le cube
        2) Mélange les sélections (selection_mask_map) sur le RGB
        3) Push viewer_left (composite) et viewer_right (carte couleurs)
        """
        if self.data is None:
            return

        # 1) Garantir un RGB prêt
        if recompute_rgb or getattr(self, "rgb_image", None) is None:
            self._make_rgb_from_cube()

        base = self.rgb_image.copy()
        H, W = base.shape[:2]
        overlay = base

        # 2) Overlay des classes sélectionnées
        if getattr(self, "selection_mask_map", None) is not None and getattr(self, "show_selection", True):
            a = float(getattr(self, "alpha", 0.35))
            a = max(0.0, min(1.0, a))
            current = overlay.copy()
            for cls, (b, g, r) in getattr(self, "class_colors", {}).items():
                mask2d = (self.selection_mask_map == cls)
                if not np.any(mask2d):
                    continue
                layer = np.zeros_like(overlay, dtype=np.uint8)
                layer[:] = (b, g, r)
                blended = cv2.addWeighted(overlay, 1.0 - a, layer, a, 0.0)
                current = np.where(mask2d[:, :, None], blended, current)
            overlay = current

        # 3) Aperçu en cours de tracé (facultatif)
        if preview and getattr(self, "_preview_mask", None) is not None:
            layer = np.zeros_like(overlay, dtype=np.uint8)
            layer[:] = (0, 0, 255)
            mixed = cv2.addWeighted(overlay, 0.7, layer, 0.3, 0.0)
            overlay = np.where(self._preview_mask[:, :, None], mixed, overlay)

        # 4) Push viewers
        self.viewer_left.setImage(self._np2pixmap(overlay))

        if getattr(self, "selection_mask_map", None) is not None:
            seg = np.zeros((H, W, 3), dtype=np.uint8)
            for cls, (b, g, r) in getattr(self, "class_colors", {}).items():
                seg[self.selection_mask_map == cls] = (b, g, r)
            self.viewer_right.setImage(self._np2pixmap(seg))

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.update_overlay()

    def _promote_canvas(self, name, canvas_cls):
        placeholder = getattr(self, name)
        parent = placeholder.parent()
        from PyQt5.QtWidgets import QSplitter

        # Crée le nouveau canvas
        canvas = canvas_cls()
        # Supprime l’ancien placeholder
        placeholder.deleteLater()

        if isinstance(parent, QSplitter):
            # cas splitter : insère au même emplacement
            idx = parent.indexOf(placeholder)
            parent.insertWidget(idx, canvas)
        else:
            # cas layout classique
            layout = parent.layout() or self.verticalLayout
            layout.addWidget(canvas)

        # Conserve refs pour live spectrum
        self.spec_canvas = canvas
        self.spec_fig = getattr(canvas, 'figure', None) or Figure()
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')
        canvas.setVisible(False)

    def init_spectrum_canvas(self):
        placeholder = getattr(self, 'spec_canvas')
        parent = placeholder.parent()

        # Crée le canvas
        self.spec_fig = Figure(facecolor=(1, 1, 1, 0.1))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_facecolor((0.7,0.7,0.7,1))
        self.spec_ax.set_title('Spectra')
        self.spec_ax.grid()


        self.span_selector = SpanSelector(
            ax=self.spec_ax,  # votre axe “Spectrum”
            onselect=self._on_bandselect,  # callback
            direction="horizontal",  # sélection horizontale
            useblit=True,  # activer le “blitting”
            minspan=1.0,  # au moins 1 unité sur l’axe λ
            props=dict(alpha=0.3, facecolor='tab:blue')
        )

        self.span_selector.set_active(False)

        # Remplace dans le splitter ou dans le layout
        if isinstance(parent, QSplitter):
            idx = parent.indexOf(placeholder)
            placeholder.deleteLater()
            parent.insertWidget(idx, self.spec_canvas)
        elif parent.layout() is not None:
            layout = parent.layout()
            idx = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            layout.insertWidget(idx, self.spec_canvas)
        else:
            placeholder.deleteLater()
            self.verticalLayout.addWidget(self.spec_canvas)

    def update_spectra(self,x=None,y=None,maxR=0):
        self.spec_ax.clear()
        x_graph = self.wl

        if self.data is None:
            return

        if x is not None and y is not None:
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                spectrum = self.data[y, x, :]
                # Spectre du pixel
                self.spec_ax.plot(x_graph, spectrum, label='Pixel')
                if np.max(spectrum) > maxR: maxR= np.max(spectrum)

        # Spectres GT moyens ± std
        if hasattr(self, 'class_means'):
            for c, mu in self.class_means.items():
                std = self.class_stds[c]
                b, g, r = self.class_colors[c]
                col = (r/255.0, g/255.0, b/255.0)
                self.spec_ax.fill_between(
                    x_graph, mu - std, mu + std,
                    color=col, alpha=0.3, linewidth=0
                )
                self.spec_ax.plot(
                    x_graph, mu, '--',
                    color=col, label=self.get_class_name(c)
                )
                if np.max(mu + std) > maxR: maxR = np.max(mu + std)

            if self.checkBox_showLegend.isChecked():

                handles, labels = self.spec_ax.get_legend_handles_labels()
                if labels:
                    # jusqu’à 4 colonnes, ~8 items par colonne
                    ncol = min(4, max(1, (len(labels) // 8 +1)))
                    leg = self.spec_ax.legend(
                        handles, labels,
                        loc='upper left',
                        borderaxespad=0.,
                        frameon=True,
                        fontsize='small',
                        ncol=ncol
                    )
                    leg.set_draggable(True)  # tu peux la déplacer à la souris
                    self.spec_fig.subplots_adjust(right=0.95)  # laisse de la place à droite

            else:
                leg = self.spec_ax.get_legend()
                if leg is not None:
                    leg.remove()
                self.spec_fig.subplots_adjust(right=0.95)  # récupère l’espace

            # if self.spec_ax.get_legend_handles_labels()[1]:
            #     self.spec_ax.legend(loc='upper right', fontsize='small')

            self.spec_ax.set_title(f"Spectra")
            self.spec_ax.grid(color='black')
            self.spec_ax.set_ylim(0,maxR+0.05)
            self.spec_ax.set_xlim(x_graph[0],x_graph[-1])
            self.spec_ax.set_xlabel("wavelength (nm)")
            self.spec_ax.set_ylabel("Reflectance (a.u.)")

        for patch in self.selected_span_patch:
            # patch est un PolyCollection produit par axvspan()
            # On le remet dans l’axe courant :
            self.spec_ax.add_patch(patch)

            # 4) On rafraîchit le canvas
        self.spec_canvas.draw_idle()

    def on_changes_EM_spectra_viz(self):
        txt = self.comboBox_endmembers_spectra.currentText()
        if 'library' in txt:
            self._activate_endmembers('lib')
        elif 'Manual' in txt:
            self._activate_endmembers('manual')
        else:
            self._activate_endmembers('auto')

    def _assign_initial_colors(self,c=None):

        if c is not None :
            unique_labels=[c]
        elif getattr(self, 'class_means', None):  # NEW fallback
            unique_labels = list(self.class_means.keys())
        else:
            return

        if len(unique_labels) <= 10:
            cmap = colormaps.get_cmap('tab10')
        else:
            cmap = colormaps.get_cmap('tab20')

        n_colors = cmap.N

        for cls in unique_labels:
            if cls not in self.class_colors:
                # cmap renvoie un tuple RGBA avec floats 0..1
                color_idx = cls % n_colors
                r_f, g_f, b_f, _ = cmap(color_idx)
                # on convertit en entiers 0..255
                r, g, b = int(255 * r_f), int(255 * g_f), int(255 * b_f)
                # MAIS OpenCV attend BGR, donc on stocke (b,g,r)
                self.class_colors[cls] = (b, g, r)
                if cls not in self.class_info:
                    self.class_info[cls] = [cls,f"Class {cls}",(r, g, b)]
                self.class_info[cls][2]=(r,g,b)

    # </editor-fold>

    # <editor-fold desc="Cube">
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
            H, W = self.data.shape[:2]
            self.selection_mask_map = np.full((H, W), -1, dtype=int)
            self.samples = {}
            self.sample_coords = {}
            self.class_means = {}
            self.class_stds = {}
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
        H, W = self.data.shape[:2]
        self.selection_mask_map = np.full((H, W), -1, dtype=int)
        self.samples = {}
        self.sample_coords = {}
        self.class_means = {}
        self.class_stds = {}
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
    def on_algo_endmember_change(self):
        txt=self.comboBox_endmembers_get.currentText()
        if 'library' in txt:
            self.stackedWidget.setCurrentIndex(0)
            self._activate_endmembers('lib')
        elif 'Manual' in txt:
            self.stackedWidget.setCurrentIndex(1)
            self._activate_endmembers('manual')
        else:
            self.stackedWidget.setCurrentIndex(2)
            self._activate_endmembers('auto')

    def _on_extract_endmembers(self):
        if self.cube is None:
            QMessageBox.warning(self, 'Unmixing', 'Load a cube first.')
            return
        method = self.comboBox_endmembers_auto_algo.currentText()
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
        self.labels, self.index_map = labels, index_map
        for i, lab in enumerate(labels):
            name = str(lab)
            self.set_class_name(i, name)

        self.comboBox_viz_show_EM.clear()
        n=E.shape[1]
        self.E_auto={}
        for key in range(n):
            spec=E[:,key]
            self.E_auto[key]=spec
            name = self.get_class_name(key)
            self.comboBox_viz_show_EM.addItem(name)
        # If you also have groups (labels), you can populate comboBox_viz_show_model too
        print('[ENDMEMBERS]',
            f"Endmembers ready: E shape {len(self.E_auto)},{len(self.E_auto[0])}, groups: {len(np.unique(labels)) if labels is not None else 0}")

        self._activate_endmembers('auto')

        self._assign_initial_colors()
        self.update_spectra(maxR=0)
        self.comboBox_endmembers_spectra.setCurrentText('Auto')

    def _on_load_library_clicked(self):
        """
        Load a spectral library from a CSV file (wavelength + endmember columns).
        Expected format:
            Wavelength, Material1, Material2, ...
            400, 0.12, 0.18, ...
            ...
        Each column (after wavelength) becomes an endmember spectrum.
        """
        import pandas as pd

        if getattr(sys, 'frozen', False):
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        open_dir=os.path.join(BASE_DIR, "unmixing", "data")

        # 1) Sélection du fichier CSV
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open spectral library",
            open_dir,
            "Spectral libraries (*.csv *.txt)"
        )
        if not filepath:
            return

        try:
            # 2) Lecture du fichier CSV
            df = pd.read_csv(filepath)

            # Vérifie qu’il y a au moins deux colonnes (λ + 1 matériau)
            if df.shape[1] < 2:
                QMessageBox.warning(self, "Library error",
                                    "Invalid library format (need wavelength + ≥1 spectrum column).")
                return

            # 3) Extraction des longueurs d’onde et des spectres
            wl = df.iloc[:, 0].to_numpy(dtype=float)
            names = list(df.columns[1:])
            E = df.iloc[:, 1:].to_numpy(dtype=float)  # shape (L, p)

            for i, name in enumerate(names):
                self.set_class_name(i, name)

            # 4) Stockage dans l'objet
            self.library_path = filepath
            self.wl_lib = wl
            self.E_lib = {i: E[:, i] for i in range(E.shape[1])}
            self.library_groups = {name: E[:, i:i + 1] for i, name in enumerate(names)}

            # 5) Renseigne la combo des endmembers (si présente)
            self.comboBox_viz_show_EM.clear()
            for i, name in enumerate(names):
                self.comboBox_viz_show_EM.addItem(name)

            # 6) Appelle la pipeline d’activation / affichage
            self._activate_endmembers('lib')

            QMessageBox.information(
                self,
                "Library loaded",
                f"Loaded {len(names)} spectra from:\n{os.path.basename(filepath)}"
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            QMessageBox.warning(
                self,
                "Load error",
                f"Failed to read library:\n{e}\n\n{tb}"
            )
            print('[LIBRARY ERROR]', e)

    def _on_error(self, msg: str):
        self.signals.error.emit(msg)
        print('[ERROR] : ',msg)

    def _activate_endmembers(self, source: str):
        """
        Construit self.E à partir de la source ('manual'|'auto'|'lib'),
        puis met à jour les moyennes/std et rafraîchit le graphe.
        """
        if source == 'manual':
            # E_manual[c] : (n_regions_c, L)  ->  E[c] : (L, n_regions_c)
            self.E = {c: arr.T for c, arr in self.E_manual.items()} if self.E_manual else {}
        elif source == 'auto':
            # E_auto[c] : (L,) -> E[c] : (L,1)
            try :
                self.E = {c: np.asarray(v, dtype=float).reshape(-1, 1) for c, v in
                          self.E_auto.items()} if self.E_auto else {}
            except:
                self.E={}
        elif source == 'lib':
            try:
                self.E = {c: np.asarray(v, dtype=float).reshape(-1, 1) for c, v in
                      self.E_lib.items()} if self.E_auto else {}
            except:
                self.E = {}
        else:
            self.E = {}

        if not self.E:
            # Rien à afficher
            self.class_means, self.class_stds = {}, {}
            self.update_spectra()
            return

        # Met à jour moyennes/écarts-types par classe -> update_spectra les trace
        self.fill_means_std_classes()  # utilise self.E pour calculer mu/std par classe. :contentReference[oaicite:0]{index=0}
        self._assign_initial_colors()  # garde tes couleurs cohérentes
        self.update_spectra(maxR=0)  # rafraîchit la figure spectra. :contentReference[oaicite:1]{index=1}

    def get_class_name(self, cls: int) -> str:
        """Return human-readable name for class id."""
        if cls in self.class_info and len(self.class_info[cls]) > 1:
            name = self.class_info[cls][1]
            if name and name.strip():
                return name
        return f"Class {cls}"

    def set_class_name(self, cls: int, name: str):
        """Update or create readable name for a class."""
        if cls not in self.class_info:
            self.class_info[cls] = [cls, name, (0, 0, 0)]
        else:
            while len(self.class_info[cls]) < 3:
                self.class_info[cls].append(None)
            self.class_info[cls][1] = name

    # </editor-fold>

    # <editor-fold desc="Manual Selection">

    def eventFilter(self, source, event):
        mode = self.comboBox_pixel_selection_mode.currentText()


        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            return False      ## to dont block drag

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton and (self.selecting_pixels or self.erase_selection):

            if not (self.selecting_pixels or self.erase_selection):
                return False
            pos = self.viewer_left.mapToScene(event.pos())
            x0, y0 = int(pos.x()), int(pos.y())
            if mode == 'pixel':
                # on commence la collecte
                self._pixel_selecting = True
                self._pixel_coords = [(x0, y0)]
                return True
            elif mode == 'rectangle':
                # début du drag
                from PyQt5.QtWidgets import QRubberBand
                self.origin = event.pos()
                self.rubberBand = QRubberBand(QRubberBand.Rectangle,
                                              self.viewer_left.viewport())
                self.rubberBand.setGeometry(self.origin.x(),
                                            self.origin.y(), 1, 1)
                self.rubberBand.show()
                return True
            elif mode == 'ellipse':
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                from PyQt5.QtGui import QPen

                self.origin = event.pos()
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                self.ellipse_item = QGraphicsEllipseItem()
                self.ellipse_item.setPen(pen)
                self.ellipse_item.setBrush(Qt.transparent)
                self.viewer_left.scene().addItem(self.ellipse_item)
                return True

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            if not self.selecting_pixels:
                self.checkBox_live_spectra.toggle()
        # 2) Mouvement souris → mise à jour de la selection en cours
        if event.type() == QEvent.MouseMove and self._pixel_selecting and mode == 'pixel':
            if not (self.selecting_pixels or self.erase_selection):
                return False
            pos = self.viewer_left.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            if x < 0 or y < 0:
                return True

            if (x, y) not in self._pixel_coords:
                self._pixel_coords.append((x, y))
            if self._preview_mask is None:
                H, W = self.data.shape[:2]
                self._preview_mask = np.zeros((H, W), dtype=bool)

            H, W = self._preview_mask.shape
            if 0 <= y < H and 0 <= x < W:
                self._preview_mask[y, x] = True
                self.show_rgb_image()
                self.update_overlay(preview=True)

            return True

        if event.type() == QEvent.MouseMove and hasattr(self, 'rubberBand'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            self.rubberBand.setGeometry(
                QRect(self.origin, event.pos()).normalized()
            )
            return True

        if event.type() == QEvent.MouseMove and mode == 'ellipse' and hasattr(self, 'ellipse_item'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            sc_orig = self.viewer_left.mapToScene(self.origin)
            sc_now = self.viewer_left.mapToScene(event.pos())
            x0, y0 = sc_orig.x(), sc_orig.y()
            x1, y1 = sc_now.x(), sc_now.y()
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.ellipse_item.setRect(rect)
            return True

        # 3) Relâchement souris → calcul de la sélection

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton and mode == 'pixel' and self._pixel_selecting :
            if not (self.selecting_pixels or self.erase_selection):
                return False
            print('realeased OK')
            # get pixels
            coords = self._pixel_coords.copy()
            #  Si au moins 3 points, propose de fermer le cheminif min 3 points, propose contour
            if len(coords) >= 3:
                reply = QMessageBox.question(
                    self, "Close Path?",
                    "You have selected multiple pixels.\n"
                    "Do you want to close the path and include all pixels inside the contour?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    pts = np.array(coords)
                    poly = Path(pts)
                    x0, y0 = pts[:, 0].min().astype(int), pts[:, 1].min().astype(int)
                    x1, y1 = pts[:, 0].max().astype(int), pts[:, 1].max().astype(int)
                    filled = list(coords)
                    for yy in range(y0, y1 + 1):
                        for xx in range(x0, x1 + 1):
                            if poly.contains_point((xx, yy)):
                                filled.append((xx, yy))

                    # to avoid dobbles
                    seen = set()
                    coords = []
                    for p in filled:
                        if p not in seen:
                            seen.add(p)
                            coords.append(p)

            if self.erase_selection:
                self._handle_erasure(coords)
            else :
                self._handle_selection(coords) # close selection

            # ready to new selection
            self._pixel_selecting = False
            self._erase_selecting = False
            self._preview_mask = None
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'rubberBand'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            # coins en coordonnées image
            tl = self.viewer_left.mapToScene(rect.topLeft())
            br = self.viewer_left.mapToScene(rect.bottomRight())
            x0, y0 = int(tl.x()), int(tl.y())
            x1, y1 = int(br.x()), int(br.y())
            # liste de tous les pixels dans le rectangle
            coords = [
                (xx, yy)
                for yy in range(max(0, min(y0, y1)), min(self.data.shape[0], max(y0, y1) + 1))
                for xx in range(max(0, min(x0, x1)), min(self.data.shape[1], max(x0, x1) + 1))
            ]

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection

            del self.rubberBand
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'ellipse_item'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            rect = self.ellipse_item.rect()
            self.viewer_left.scene().removeItem(self.ellipse_item)
            del self.ellipse_item

            cx, cy = rect.center().x(), rect.center().y()
            rx, ry = rect.width() / 2, rect.height() / 2
            x0, x1 = int(rect.left()), int(rect.right())
            y0, y1 = int(rect.top()), int(rect.bottom())

            coords = []
            for yy in range(max(0, y0), min(self.data.shape[0], y1 + 1)):
                for xx in range(max(0, x0), min(self.data.shape[1], x1 + 1)):
                    if ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1:
                        coords.append((xx, yy))

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection
            return True

        # 4) Mouvement souris pour le live spectrum
        # if source is self.viewer_left.viewport() and event.type() == QEvent.MouseMove and         self.checkBox_live_spectra.isChecked():
        #     if self.checkBox_live_spectra.isChecked() and self.data is not None:
        #         pos = self.viewer_left.mapToScene(event.pos())
        #         x,y=int(pos.x()),int(pos.y())
        #         H, W = self.data.shape[0], self.data.shape[1]
        #         if 0 <= x < W and 0 <= y < H:
        #             self.update_spectra(x, y)
        #
        #     return True

        # return super().eventFilter(source, event)
        return False

    def _on_bandselect(self, lambda_min, lambda_max):
        """
        Callback  SpanSelector
        """

        if self._band_action is None:
            return

        # 1) S’assurer que lambda_min < lambda_max
        if lambda_min > lambda_max:
            lambda_min, lambda_max = lambda_max, lambda_min

        # 2) Conversion en indices d’onde
        idx_min = int(np.argmin(np.abs(self.wl - lambda_min)))
        idx_max = int(np.argmin(np.abs(self.wl - lambda_max)))

        # 3) update self.selected_bands
        if self._band_action == 'add':
            for idx in range(idx_min,idx_max+1):
                if idx not in self.selected_bands:
                    self.selected_bands.append(idx)

            print(f"Selected band : [{idx_min} → {idx_max}] "
                  f"({self.wl[idx_min]:.1f} → {self.wl[idx_max]:.1f} nm)")

            patch=self.spec_ax.axvspan(
                lambda_min, lambda_max,
                alpha=0.2, color='tab:blue'
            )

            self.selected_span_patch.append(patch)

        elif self._band_action == 'del':

            for idx in range(idx_min,idx_max+1):
                if idx in self.selected_bands:
                    self.selected_bands.remove(idx)

            self.selected_bands=sorted(self.selected_bands)

            for patch in self.selected_span_patch:  # reset all patch
                patch.remove()
                self.selected_span_patch = []

            bands={}
            i_band=0
            for i in range(len(self.selected_bands)-1): # get bands from index
                if (self.selected_bands[i+1] -self.selected_bands[i]) ==1:
                    try:
                        bands[i_band].append(self.selected_bands[i])
                    except:
                        bands[i_band]=[self.selected_bands[i]]
                else:
                    try:
                        bands[i_band].append(self.selected_bands[i])
                    except:
                        bands[i_band]=[self.selected_bands[i]]

                    i_band+=1


            # recreate patches
            for i_band in bands:
                lambda_min, lambda_max=self.wl[bands[i_band][0]],self.wl[bands[i_band][-1]]

                patch = self.spec_ax.axvspan(
                    lambda_min, lambda_max,
                    alpha=0.2, color='tab:blue'
                )

                self.selected_span_patch.append(patch)

        self.spec_canvas.draw_idle()

    def start_pixel_selection(self):

        self.show_selection = True
        self.pushButton_class_selection.setText("Stop Selection")
        self.pushButton_erase_selected_pix.setChecked(False)

        if len(self.samples) > 0:
            reply = QMessageBox.question(
                self, "Erase selection?",
                "Do you want to erase previous selection?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                H, W = self.selection_mask_map.shape
                # Réinitialise à -1 (aucune classe)
                self.selection_mask_map[:] = -1
                self.samples.clear()

        self.selecting_pixels = True
        # self.viewer_left.setDragMode(QGraphicsView.NoDrag)
        self.show_rgb_image()
        self.update_overlay()

    def stop_pixel_selection(self):

        self.selecting_pixels = False

        # ready to select
        self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)

        # remet le bouton à l'état initial
        self.pushButton_class_selection.setText("Start Selection")
        self.pushButton_class_selection.setChecked(False)

        # efface tout preview en cours
        self.selecting_pixels = False

        # enfin, on affiche l'image normale (sans preview ni sélection en cours)
        self.show_rgb_image()
        self.update_overlay()
        print(f'[MANUAL SELECTION] E shape : {len(self.samples)}')
        for key in self.E_manual:
            print(self.E_manual[key].shape)

        self._activate_endmembers('manual')
        self.update_spectra(maxR=0)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')

    def _handle_selection(self, coords):
        """Prompt for class and store spectra of the given coordinates."""
        n = self.nclass_box.value() - 1
        labels = [str(i) for i in range(n + 1)]

        default_label = max(0, min(self._last_label, n)) if n > 0 else 0

        # 2) Ouvrir un QInputDialog.getItem() au lieu de getInt()
        #    - on force l’édition à se faire via la liste déroulante
        cls_str, ok = QInputDialog.getItem(
            self,
            "Class",
            "Choose class label:",
            labels,
            default_label,  # index initial (par défaut on sélectionne “0”)
            False  # False = l’utilisateur ne peut pas taper autre chose que la liste
        )

        if not ok:
            return

        cls = int(cls_str)
        self._last_label = cls
        if cls not in self.class_info or not self.class_info[cls][1]:
            self.set_class_name(cls, f"Manual_{cls}")
        if cls not in self.class_colors:
            self._assign_initial_colors(cls)

        # append spectra

        for x, y in coords:
            if not (0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]):
                continue

            # A) s’il appartenait déjà à une autre classe, on l’enlève
            old = self.selection_mask_map[y, x]
            if old >= 0 and old != cls:
                # retirer coord de sample_coords[old] et de samples[old]
                if (x, y) in self.sample_coords.get(old, set()):
                    self.sample_coords[old].remove((x, y))
                # reconstruire la liste des spectres pour old
                self.samples[old] = [
                    self.data[yy, xx, :]
                    for (xx, yy) in self.sample_coords.get(old, ())
                ]

            # B) on (ré)assigne le pixel à la classe cls
            self.selection_mask_map[y, x] = cls
            # ajouter dans sample_coords et samples si pas déjà présent
            if (x, y) not in self.sample_coords.setdefault(cls, set()):
                self.sample_coords.setdefault(cls, set()).add((x, y))
                self.samples.setdefault(cls, []).append(self.data[y, x, :])

        self._add_region(cls, coords)
        self._activate_endmembers('manual')
        self.update_spectra(maxR=0)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')

        # 3) rafraîchir l’affichage
        self.show_rgb_image()
        self.update_overlay()

    def _handle_erasure(self, coords):
        if self.selection_mask_map is None or self.data is None:
            return
        H, W = self.selection_mask_map.shape

        # 1) masque des pixels à effacer
        erase_mask = np.zeros((H, W), dtype=bool)
        for x, y in coords:
            if 0 <= x < W and 0 <= y < H:
                erase_mask[y, x] = True

        # 2) applique à la carte
        self.selection_mask_map[erase_mask] = -1

        # 3) rebuild samples/sample_coords propres depuis la carte
        new_sample_coords = {}
        ys, xs = np.where(self.selection_mask_map >= 0)
        for y, x in zip(ys, xs):
            c = int(self.selection_mask_map[y, x])
            new_sample_coords.setdefault(c, set()).add((x, y))
        new_samples = {}
        for c, pts in new_sample_coords.items():
            new_samples[c] = [self.data[yy, xx, :] for (xx, yy) in pts]
        self.sample_coords = new_sample_coords
        self.samples = new_samples

        # 4) retirer ces coords des régions et mettre à jour E_manual
        self._erase_from_regions(coords)

        # 5) nettoyer couleurs/means/std SEULEMENT si la classe est totalement vide
        alive = set(self.sample_coords.keys())
        for c in list(self.class_means.keys()):
            if c not in alive:
                self.class_means.pop(c, None)
                self.class_stds.pop(c, None)
                self.class_colors.pop(c, None)

        self._activate_endmembers('manual')
        self.update_spectra(maxR=0)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')

        self.update_overlay()

    def toggle_show_selection(self):

        self.show_selection = self.checkBox_see_selection_overlay.isChecked()
        self.show_rgb_image()
        self.update_overlay()

    def on_toggle_erase(self, checked):
        self.erase_selection = checked

        if checked:
            self._pixel_selecting = False
            self.stop_pixel_selection()

            self.show_selection = True

            self.pushButton_erase_selected_pix.setText("Stop Erasing")
            self.pushButton_class_selection.setChecked(False)
            # self.viewer_left.setDragMode(QGraphicsView.NoDrag)

        else:
            self.pushButton_erase_selected_pix.setText("Erase Pixels")
            self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)

    def on_toggle_selection(self, checked: bool):
        if checked:
            self.erase_selection = False
            self.start_pixel_selection()

        else:
            # fin du mode sélection
            self.stop_pixel_selection()

    def _recompute_E_manual_for_class(self, cls: int):
        """Recalcule self.E_manual[cls] à partir des régions existantes de la classe."""
        regs = self.regions.get(cls, [])
        mats = []
        for reg in regs:
            if not reg.get('coords'):
                continue
            # moyenne de la région
            spectra = np.array([self.data[y, x, :] for (x, y) in reg['coords']])
            reg['mean'] = spectra.mean(axis=0)
            mats.append(reg['mean'][None, :])  # (1, L)
        if mats:
            self.E_manual[cls] = np.vstack(mats)  # (n_regions, L)
        else:
            self.E_manual.pop(cls, None)

    def _union_coords_of_class(self, cls: int):
        """Ensemble de tous les pixels déjà pris par des régions de cette classe."""
        s = set()
        for reg in self.regions.get(cls, []):
            s |= reg.get('coords', set())
        return s

    def _add_region(self, cls: int, coords):
        """Crée une nouvelle région (coords uniques vs régions existantes) et met à jour E_manual."""
        if cls not in self.regions:
            self.regions[cls] = []
        # éviter les doublons: ne garde que les pixels non déjà présents dans d'autres régions de la classe
        used = self._union_coords_of_class(cls)
        coords_unique = [(x, y) for (x, y) in coords if (x, y) not in used]
        if not coords_unique:
            # rien de nouveau pour cette classe -> juste recomposer E_manual au cas où
            self._recompute_E_manual_for_class(cls)
            return
        reg = {'coords': set(coords_unique), 'mean': None}
        self.regions[cls].append(reg)
        self._recompute_E_manual_for_class(cls)

    def _erase_from_regions(self, coords):
        """Retire coords des régions de toutes classes, supprime régions vides, et met à jour E_manual."""
        to_erase = set(coords)
        affected = set()
        for c, rlist in list(self.regions.items()):
            new_list = []
            for reg in rlist:
                n_before = len(reg['coords'])
                reg['coords'].difference_update(to_erase)
                if len(reg['coords']) > 0:
                    new_list.append(reg)
                if len(reg['coords']) != n_before:
                    affected.add(c)
            if new_list:
                self.regions[c] = new_list
            else:
                self.regions.pop(c, None)
                # si plus de régions => on enlèvera aussi E_manual[c] ci-dessous
                affected.add(c)
        for c in affected:
            self._recompute_E_manual_for_class(c)

    # </editor-fold>

    # <editor-fold desc="Processing Data Helpers">
    def _current_normalization(self) -> str:
        txt = self.comboBox_normalisation.currentText().lower()
        if 'l2' in txt:
            return 'L2'
        if 'l1' in txt:
            return 'L1'
        return 'none'

    def fill_means_std_classes(self):
        full_means = {}
        full_stds = {}
        for key in self.E:
            full_means[key] = self.E[key].mean(axis=1)
            full_stds[key] = self.E[key].std(axis=1)
        self.class_means = full_means
        self.class_stds = full_stds

    def prune_unused_classes(self):
        """
        Supprime de self.class_colors et self.class_info
        tous les labels qui ne figurent plus dans self.cls_map.
        """
        if self.class_colors is None:
            return

        labels_in_map = set(np.unique(self.cls_map))
        for d in (self.class_colors, self.class_info):
            for cls in list(d.keys()):
                if cls not in labels_in_map:
                    del d[cls]

    # </editor-fold>

    # <editor-fold desc="Unmixing Job Queue">
    def _init_classification_table(self, table):
        from PyQt5.QtWidgets import QTableWidgetItem
        headers = ["Name", "Algo", "EM source", "Params", "Status", "Progress", "Duration"]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSortingEnabled(False)
        table.setRowCount(0)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(table.SelectRows)
        table.setSelectionMode(table.SingleSelection)
        table.setEditTriggers(table.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        # largeur indicative
        table.setColumnWidth(0, 140)
        table.setColumnWidth(1, 90)
        table.setColumnWidth(2, 110)
        table.setColumnWidth(3, 320)
        table.setColumnWidth(4, 90)
        table.setColumnWidth(5, 120)

    def _make_progress_bar(self):
        from PyQt5.QtWidgets import QProgressBar
        pb = QProgressBar()
        pb.setRange(0, 100)
        pb.setValue(0)
        pb.setTextVisible(True)
        return pb

    def _ensure_unique_name(self, base: str) -> str:
        name = base
        i = 1
        while name in self.jobs:
            i += 1
            name = f"{base} ({i})"
        return name

    def _current_normalization(self) -> str:
        # UI: comboBox_normalisation -> "L2 (Euclidian)", "L1 (Sum=1)", "None"
        txt = self.comboBox_normalisation.currentText()
        if "L2" in txt: return "L2"
        if "L1" in txt: return "L1"
        return "None"

    def _collect_unmix_params(self) -> dict:
        # Widgets: voir unmixing_window.py (tol: 10^spin2, lam: 10^spin3, rho: spin4, max-iter: spinBox)
        algo = self.comboBox_unmix_algorithm.currentText()
        norm = self._current_normalization()
        anc = self.checkBox_unmix_ANC.isChecked()
        asc = self.checkBox_unmix_ASC.isChecked()
        tol = 10.0 ** float(self.doubleSpinBox_unmix_lambda_2.value())
        lam = 10.0 ** float(self.doubleSpinBox_unmix_lambda_3.value())
        rho = float(self.doubleSpinBox_unmix_lambda_4.value())
        max_iter = int(self.spinBox.value())  # label "Maximum iterations" dans l’UI

        em_src = self.comboBox_endmembers_use_for_unmixing.currentText()  # "From library" | "Manual" | "Auto"
        em_merge = self.checkBox_unmix_merge_EM_groups.isChecked()
        p = int(self.nclass_box.value())

        return dict(
            algo=algo, norm=norm, anc=anc, asc=asc, tol=tol, lam=lam, rho=rho,
            max_iter=max_iter, em_src=em_src, em_merge=em_merge, p=p
        )

    def _format_params_summary(self, P: dict) -> str:
        # court et lisible dans la table
        bits = [f"norm={P['norm']}"]
        if P["algo"] == "SUnSAL":
            bits += [f"λ={P['lam']:.1e}", f"tol={P['tol']:.1e}", f"ρ={P['rho']:.3g}",
                     f"ANC={'on' if P['anc'] else 'off'}", f"ASC={'on' if P['asc'] else 'off'}",
                     f"maxit={P['max_iter']}"]
        else:
            # tol/max_iter peuvent servir pour FCLS/NNLS si tu les utilises en backend
            bits += [f"tol={P['tol']:.1e}", f"maxit={P['max_iter']}"]
        bits += [f"p={P['p']}"]
        return " | ".join(bits)

    def _on_add_unmix_job(self):
        if getattr(self, "cube", None) is None:
            QMessageBox.warning(self, "Unmixing", "Load a cube first.")
            return

        P = self._collect_unmix_params()
        base = f"{P['algo']} ({P['em_src']})"
        name = self._ensure_unique_name(base)

        # Construire l’objet job (on ne calcule/attache pas encore E ici)
        job = UnmixJob(
            name=name,
            model=P["algo"],
            normalization=P["norm"],
            max_iter=P["max_iter"],
            tol=P["tol"],
            lam=P["lam"],
            rho=P["rho"],
            anc=P["anc"],
            asc=P["asc"],
        )
        # Tu pourras plus tard remplir job.E et/ou un tag sur la source EM au moment du run.

        self.jobs[name] = job
        self.job_order.append(name)

        # Insère la ligne dans la table
        self._insert_job_row(name, P)
        # (si tu préfères recalculer tout : self._refresh_table())

    def _insert_job_row(self, name: str, P: dict):
        from PyQt5.QtWidgets import QTableWidgetItem
        table = self.tableWidget_classificationList
        row = table.rowCount()
        table.insertRow(row)

        # Col 0: Name
        table.setItem(row, 0, QTableWidgetItem(name))
        # Col 1: Algo
        table.setItem(row, 1, QTableWidgetItem(P["algo"]))
        # Col 2: EM source (+ merge)
        em_txt = P["em_src"] + (" + merge" if P["em_merge"] else "")
        table.setItem(row, 2, QTableWidgetItem(em_txt))
        # Col 3: Params
        table.setItem(row, 3, QTableWidgetItem(self._format_params_summary(P)))
        # Col 4: Status
        table.setItem(row, 4, QTableWidgetItem("Queued"))
        # Col 5: Progress (widget)
        pb = self._make_progress_bar()
        table.setCellWidget(row, 5, pb)
        # Col 6: Duration
        table.setItem(row, 6, QTableWidgetItem("-"))

    # 4) (Optionnel) Refresh complet si tu modifies des jobs ailleurs

    def _refresh_table(self):
        from PyQt5.QtWidgets import QTableWidgetItem
        table = self.tableWidget_classificationList
        sorting = table.isSortingEnabled()
        if sorting: table.setSortingEnabled(False)
        table.setRowCount(0)

        for name in self.job_order:
            job = self.jobs.get(name)
            if not job: continue
            # reconstruit un mini dict pour afficher les params
            P = dict(
                algo=job.model, norm=job.normalization,
                anc=job.anc, asc=job.asc,
                tol=job.tol, lam=job.lam, rho=job.rho,
                max_iter=job.max_iter,
                p=getattr(self, "nclass_box", None).value() if hasattr(self, "nclass_box") else None,
                em_src=getattr(self, "comboBox_endmembers_use_for_unmixing", None).currentText() if hasattr(self,
                                                                                                            "comboBox_endmembers_use_for_unmixing") else "?",
                em_merge=getattr(self, "checkBox_unmix_merge_EM_groups", None).isChecked() if hasattr(self,
                                                                                                      "checkBox_unmix_merge_EM_groups") else False
            )
            self._insert_job_row(name, P)

        if sorting: table.setSortingEnabled(True)

    def _update_row_from_job(self, name: str):
        """Appelle ceci pendant l’exécution plus tard pour refléter status/progress/duration."""
        table = self.tableWidget_classificationList
        NAME_COL, STATUS_COL, PROG_COL, DUR_COL = 0, 4, 5, 6
        # Retrouve la row par le texte de la 1re colonne
        for row in range(table.rowCount()):
            item = table.item(row, NAME_COL)
            if not item or item.text() != name:
                continue
            job = self.jobs.get(name)
            if not job: return
            # Status
            table.setItem(row, STATUS_COL, QTableWidgetItem(getattr(job, "status", "Queued")))
            # Progress
            w = table.cellWidget(row, PROG_COL)
            if w:
                try:
                    w.setValue(int(job.progress))
                except:
                    pass
            # Duration
            dur_txt = "-" if job.duration_s is None else f"{job.duration_s:.1f}s"
            table.setItem(row, DUR_COL, QTableWidgetItem(dur_txt))
            break

    def remove_selected_job(self):
        row = self.tableWidget_classificationList.currentRow()
        if row < 0: return
        name = self.tableWidget_classificationList.item(row, 0).text()
        self.jobs.pop(name, None)
        try:
            self.job_order.remove(name)
        except ValueError:
            pass
        self.tableWidget_classificationList.removeRow(row)

    def remove_all_jobs(self, ask_confirm=True):
        if ask_confirm and any(getattr(self.jobs[n], "status", "") in ("Running", "Done") for n in self.job_order):
            from PyQt5.QtWidgets import QMessageBox
            if QMessageBox.question(self, "Reset Jobs ?",
                                    "Some jobs may be running/done. Continue?",
                                    QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
                return
        self.jobs.clear()
        self.job_order.clear()
        self._init_classification_table(self.tableWidget_classificationList)

    def _move_job(self, delta: int):
        row = self.tableWidget_classificationList.currentRow()
        if row < 0: return
        new_row = max(0, min(self.tableWidget_classificationList.rowCount() - 1, row + delta))
        if new_row == row: return
        name = self.tableWidget_classificationList.item(row, 0).text()
        # maj ordre
        idx = self.job_order.index(name)
        self.job_order.insert(idx + delta, self.job_order.pop(idx))
        # Rebuild table (simple)
        self._refresh_table()
        self.tableWidget_classificationList.selectRow(new_row)

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
