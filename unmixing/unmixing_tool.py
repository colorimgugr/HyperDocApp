# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import cv2
from PIL import Image
import h5py

import numpy as np
from PyQt5.QtWidgets import (QApplication, QSizePolicy, QSplitter,QHeaderView,QProgressBar,QColorDialog,
                            QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QDialogButtonBox, QCheckBox, QScrollArea, QWidget, QFileDialog, QMessageBox,
                             QRadioButton,QInputDialog,QTableWidget, QTableWidgetItem,QHeaderView,QGraphicsView,
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
#todo : add substrate to library by manual selection
#todo : color map of abundance map
#todo : export results en h5 or multiple png
#todo : show all classified by : name, total abundace, max abundance, other ?
#todo : check interaction with transtaprecny between EM and AMap
#todo : from library -> gerer les wl_lib et wl (cube) pour unmixing
#todo : unmixing -> select endmembers AND if merge
#todo : viz spectra -> show/hide by clicking line or title (or ctrl+click)
#todo : add one pixel fusion
#todo : select pixels of endmembers also with ctrl+clic left
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

class EMEditDialog(QDialog):
    """
    Editable table for Endmembers:
    Columns: [Index | Name | Color]
    - Name is edited inline (QLineEdit).
    - Color cell shows a swatch; clicking opens QColorDialog.
    Returns updated (names, BGR colors) via accessors if accepted.
    """
    def __init__(self, parent, rows_data):
        """
        rows_data: list of tuples (cls_index:int, name:str, color_bgr:tuple[int,int,int])
        """
        super().__init__(parent)
        self.setWindowTitle("Edit Endmembers")
        self.setModal(True)
        self.resize(560, 420)

        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Index", "Name", "Color"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked | QTableWidget.EditKeyPressed)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setRowCount(len(rows_data))

        # Fill rows
        for r, (cls_idx, name, bgr) in enumerate(rows_data):
            # Index (read-only)
            item_idx = QTableWidgetItem(str(cls_idx))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
            item_idx.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, item_idx)

            # Name (editable)
            item_name = QTableWidgetItem(name or f"class{cls_idx}")
            self.table.setItem(r, 1, item_name)

            # Color (button with swatch)
            btn = QPushButton("")
            btn.setObjectName(f"color_btn_{r}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, row=r: self.pick_color(row))
            self._apply_btn_color(btn, bgr)
            self.table.setCellWidget(r, 2, btn)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Layout
        lay = QVBoxLayout(self)
        lay.addWidget(self.table)
        lay.addWidget(buttons)

    def _apply_btn_color(self, btn: QPushButton, bgr):
        b, g, r = bgr
        btn.setFixedWidth(90)
        btn.setFixedHeight(22)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b}); border: 1px solid #555; }}"
        )
        btn.setProperty("bgr", (b, g, r))

    def pick_color(self, row):
        btn = self.table.cellWidget(row, 2)
        b, g, r = btn.property("bgr")
        initial = QColor(int(r), int(g), int(b))
        col = QColorDialog.getColor(initial, self, "Pick color")
        if col.isValid():
            new_bgr = (col.blue(), col.green(), col.red())  # keep BGR in storage
            self._apply_btn_color(btn, new_bgr)

    def result_rows(self):
        """Return list of (cls_index:int, name:str, color_bgr:tuple) from the table."""
        out = []
        for r in range(self.table.rowCount()):
            idx = int(self.table.item(r, 0).text())
            name = self.table.item(r, 1).text().strip()
            btn = self.table.cellWidget(r, 2)
            bgr = btn.property("bgr")
            out.append((idx, name, bgr))
        return out

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
    progress = pyqtSignal(int,np.ndarray)
    # Results
    em_ready = pyqtSignal(np.ndarray,np.ndarray, object, dict)  # E, labels, index_map
    unmix_ready = pyqtSignal(np.ndarray, np.ndarray, dict)  # A, E, maps_by_group

# ------------------------------- Workers --------------------------------------
@dataclass
class EndmemberJob:
    method: str  # 'Manual' | 'ATGP' | 'N-FINDR' | 'Library'
    p: int
    niter: int
    normalization: str  # 'None'|'L2'|'L1'

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

            if self.job.method == 'ATGP':
                E,em_idx = extract_endmembers_atgp(data, self.job.p)
                labels = np.array([f"EM_{i:02d}" for i in range(E.shape[1])], dtype=object)
                index_map = {str(lbl): np.array([i]) for i, lbl in enumerate(labels)}
            elif self.job.method == 'N-FINDR':
                # In PySptools, `maxit` is a small integer; use job.niter
                E,em_idx = extract_endmembers_nfindr(data, self.job.p, maxit=max(1, int(self.job.niter)))
                labels = np.array([f"EM_{i:02d}" for i in range(E.shape[1])], dtype=object)
                index_map = {str(lbl): np.array([i]) for i, lbl in enumerate(labels)}
            else:
                raise ValueError("Invalid endmember extraction settings.")

            # Match normalization if any groups were given not already normalized
            if self.job.normalization and self.job.normalization.lower() != 'none':
                pass

            self.signals.em_ready.emit(E,em_idx, labels, index_map)
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
    cube: Optional[np.ndarray] = None        # (H, W, L)
    E = None  # (L,p)
    roi_mask : Optional[np.ndarray] = None  # (H,W) boolean (optional)
    labels: Optional[np.ndarray] = None      # (p,) opcional (grupos)
    em_src: str = "Auto"

    preprocess = None
    merge_EM = False
    wl_job= None # wavelength kept for job
    wl_cube = None # initial wl of cube
    wl_em = None # initial wl of EM

    # Job progress
    progress: int = 0         # 0..100
    status: str = "Queued"                   # 'Queued'|'Running'|'Done'|'Failed'|'Cancelled'
    duration_s: Optional[float] = None    # whole classification duration
    _t0=None
    error_msg: Optional[str] = None
    reshape_order: str = 'C'  # 'C' (row-major) o 'F' (column-major), igual que en identification_tool
    chunk_size: int = 0  # 0 = auto; >0 fuerza tamaño de bloque en nº de píxeles

    #tOutputs
    abundance_maps: Optional[np.ndarray] = None     # raw classification map
    clean_map: Optional[np.ndarray] = None  # cleaned classification map
    clean_param = None  # clean parameters

    #to compare with others
    params: dict = field(default_factory=dict)
    _frozen_params: tuple = field(default=(), repr=False, compare=False)

class UnmixWorker(QRunnable):
    """
    UnmixWorker con chunking por bloques contiguos de píxeles aplanados tras un reshape.
    Respeta el orden de aplanado indicado en job.reshape_order ('C' o 'F').
    """

    def __init__(self, job):
        super().__init__()
        self.job = job
        self.signals = UnmixingSignals()
        self._cancelled = False

    # ---- API de cancelación desde la UI ----
    def cancel(self):
        self._cancelled = True

    def _check_cancel(self):
        if self._cancelled:
            raise RuntimeError("Cancelled")

    # ---- utilidades internas ----
    def _auto_chunk_size(self, p: int, dtype) -> int:
        """
        Elige tamaño de bloque (nº de píxeles) para usar ~64 MB por tile.
        Aproxima coste ~ (L + 3*p) * C * bytes. Redondea a múltiplos de 256.
        """
        bytes_per = np.dtype(dtype).itemsize
        L = int(self.job.E.shape[0])
        budget = 64 * 1024 * 1024  # 64 MB
        denom = max(1, (L + 3 * p) * bytes_per)
        C = max(1, int(budget // denom))
        return max(256, (C // 256) * 256)

    def _validate_inputs(self):
        if self.job.cube is None or not isinstance(self.job.cube, np.ndarray) or self.job.cube.ndim != 3:
            raise ValueError("cube must be a (H, W, L) ndarray.")
        if self.job.E is None:
            raise ValueError("E (endmembers) is required.")
        E = np.asarray(self.job.E)
        if E.ndim == 1:
            E = E.reshape(-1, 1)
        elif E.ndim != 2:
            raise ValueError(f"E invalid ndim={E.ndim}; expected (L, p).")
        H, W, L = self.job.cube.shape
        if E.shape[0] != L:
            raise ValueError(f"E and cube mismatch on L: E({E.shape[0]}) != cube({L}).")
        self.job.E = E  # normalizado a 2D
        return H, W, L

    def _vectorize_like_identification(self, data: np.ndarray, order: str) -> np.ndarray:
        """
        Convierte (H, W, L) -> (L, N) siguiendo el mismo 'order' que en identification_tool.
        """
        H, W, L = data.shape
        return np.reshape(data, (H * W, L), order=order).T  # (L, N)

    @pyqtSlot()
    def run(self):
        t0 = time.time()
        try:
            # 0) Validación
            H, W, L = self._validate_inputs()
            self._check_cancel()

            # 1) Normalización
            cube = normalize_cube(self.job.cube, mode=self.job.normalization)
            self._check_cancel()

            # 2) Vectorizado
            order = getattr(self.job, "reshape_order", "C")
            if order not in ("C", "F"):
                order = "C"
            Y = self._vectorize_like_identification(cube, order=order)  # (L, N)
            N = Y.shape[1]
            self._check_cancel()

            # 3) ROI -> índices en el espacio aplanado (alineados con el mismo 'order')
            if getattr(self.job, "roi_mask", None) is not None:
                mask = self.job.roi_mask.astype(bool).ravel(order=order)
                idx_work = np.flatnonzero(mask)  # píxeles seleccionados
            else:
                mask = None
                idx_work = np.arange(N, dtype=np.int64)
            Nw = int(idx_work.size)

            self._check_cancel()

            # 4) Preparar solver + chunking
            E = self.job.E  # (L, p)
            norm = (getattr(self.job, "normalization", "None") or "None").lower()
            if norm != "none":
                # same column-wise normalization applied to Y must be applied to E
                E = normalize_spectra(E, mode=norm)

            p = int(E.shape[1])
            # Decide tamaño de bloque
            user_chunk = int(getattr(self.job, "chunk_size", 0))
            if user_chunk > 0:
                chunk = user_chunk
            else:
                # ➜ 20 morceaux ≈ 5% chacun
                steps = 20
                chunk = max(1, int(np.ceil(Nw / steps)))

            # Destino global en espacio (p, N)
            # float32 para equilibrio precisión/memoria
            A = np.zeros((p, N), dtype=np.float32)
            self.signals.progress.emit(0, A)

            # Progreso granular 5 -> 95 durante el solver
            base_prog, end_prog = 5.0, 95.0  # 5%..95% pendant le solveur
            processed = 0

            m = (self.job.model or "").upper()
            for s in range(0, Nw, chunk):
                self._check_cancel()
                e = min(Nw, s + chunk)
                cols = idx_work[s:e]          # índices contiguos en el espacio aplanado
                Y_sub = Y[:, cols]            # (L, C)

                # Ejecuta solver para el bloque
                if m == 'UCLS':
                    A_sub = unmix_ucls(E, Y_sub)
                elif m == 'NNLS':
                    A_sub = unmix_nnls(E, Y_sub)
                elif m == 'FCLS':
                    A_sub = unmix_fcls(E, Y_sub)
                elif m == 'SUNSAL':
                    A_sub = unmix_sunsal(
                        E, Y_sub,
                        lam=self.job.lam,
                        positivity=self.job.anc or self.job.asc,
                        sum_to_one=self.job.asc,
                        rho=self.job.rho,
                        max_iter=self.job.max_iter,
                        tol=self.job.tol,
                    )
                else:
                    raise ValueError(f"Unknown model: {self.job.model}")

                # Reinyecta directamente en los índices globales
                A[:, cols] = A_sub.astype(np.float32, copy=False)

                # progreso por bloque
                processed += (e - s)
                frac = processed / max(1, Nw)
                prog = int(base_prog + (end_prog - base_prog) * frac)
                self.signals.progress.emit(prog,A)

            # 5) Post-procesado opcional: mapas por grupo
            maps_by_group = {}
            if getattr(self.job, "labels", None) is not None:
                # Si tu abundance_maps_by_group soporta 'order', pásalo; si no, deja tal cual.
                maps_by_group = abundance_maps_by_group(A, self.job.labels, H, W)

            # 6) Fin
            self.signals.progress.emit(100,A)
            # Entregamos A en espacio (p, N) + E y los mapas por grupo
            self.signals.unmix_ready.emit(A, E, maps_by_group)
            print(f'[UnmixWorker running] : end of job for {self.job.name}')
            print(f'[UnmixWorker running] labels of EM :')
            for lab in self.job.labels:
                print(lab)


        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"Unmixing failed: {e}\n{tb}")
# ------------------------------- Main Widget ----------------------------------
class UnmixingTool(QWidget,Ui_GroundTruthWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Queue structures pour unmixing
        self.job_order: List[str] = []  # only job names in execution order
        self.jobs: Dict[str, UnmixJob] = {}  # name -> job
        # self._init_cleaning_list()

        # Unmix as thread init
        self.threadpool = QThreadPool.globalInstance()
        self.threadpool.setMaxThreadCount(1)  # clé: 1 seul QRunnable en parallèle
        self._running_idx: int = -1
        self._current_worker = None
        self._stop_all = False
        self.only_selected = False
        self.signals = UnmixingSignals()

        # Remplacer placeholders par ZoomableGraphicsView
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)
        self.viewer_left.enable_rect_selection = True  # no rectangle selection for left view
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
        self.selected_bands=[] #band selection
        self.selected_span_patch=[] # rectangle patch of selected bands

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
        self.regions = {}  # dict: classe -> [ {'coords': set[(x,y)], 'mean': np.ndarray}, ... ]

        self.E_manual= {}
        self.wl_manual=None
        self.param_manual= {}

        self.E_lib= {}
        self.wl_lib=None
        self.param_lib={}

        self.E_auto={}
        self.wl_auto=None
        self.param_auto={}

        self.class_means = {}  # for spectra of classe
        self.class_stds = {}  # for spectra of classe
        self.class_ncount = {}  # for npixels classified

        self.class_info_manual = {}  # {cid: [label, name, (R,G,B),norm_params]}
        self.class_info_auto = {}
        self.class_info_lib = {}

        self.active_source = 'manual'  # 'manual' | 'auto' | 'lib'

        self.cls_map = None
        self.index_map: Optional[Dict[str, np.ndarray]] = None

        self.A: Optional[np.ndarray] = None  # (p, N)
        self.maps_by_group: Dict[str, np.ndarray] = {}

        # connection
        self.load_btn.clicked.connect(self.open_load_cube_dialog)
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
        self.pushButton_save_EM.clicked.connect(self.save_endmembers_spectra)
        self.pushButton_class_name_assign.clicked.connect(self.open_em_editor)

        # Spectra window
        self.comboBox_endmembers_spectra.currentIndexChanged.connect(self.on_changes_EM_spectra_viz)
        self.checkBox_showLegend.toggled.connect(self.update_spectra)
        self.checkBox_showGraph.toggled.connect(self.toggle_spectra)
        self.pushButton_band_selection.toggled.connect(self.band_selection)

        #Results viz window
        self.comboBox_viz_show_model.currentIndexChanged.connect(self._on_model_viz_change)
        self.comboBox_viz_show_EM.currentIndexChanged.connect(self._refresh_abundance_view)
        self.radioButton_view_abundance.toggled.connect(self._refresh_abundance_view)

        # Unmix window
        self.radioButton_view_em.toggled.connect(self.toggle_em_viz_stacked)
        self.horizontalSlider_em_position_size.valueChanged.connect(self.update_overlay)
        self.horizontalSlider_overlay_transparency.valueChanged.connect(self.update_alpha)

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
        self.pushButton_add_queue_unmixing.clicked.connect(self._on_add_unmix_job)
        self.pushButton_clas_start.clicked.connect(self._on_start_all)
        self.pushButton_clas_stop.clicked.connect(self._on_cancel_queue)
        self.pushButton_clas_start_selected.clicked.connect(self._on_start_selected_or_last)
        self.pushButton_clas_remove.clicked.connect(self.remove_selected_job)
        self.pushButton_clas_remove_all.clicked.connect(self.remove_all_jobs)
        self.pushButton_clas_up.clicked.connect(lambda: self._move_job(-1))
        self.pushButton_clas_down.clicked.connect(lambda: self._move_job(+1))

        mem_strech_factors={0:5,2:5,1:2}
        for key,val in mem_strech_factors.items():
            self.splitter.setStretchFactor(key,val)
        self.mem_sizes=self.splitter.sizes()

        self.fill_form_em('manual')
        self.fill_form_em('auto')
        self.fill_form_em('lib')

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

    def toggle_spectra(self):
        if self.checkBox_showGraph.isChecked():
            self.splitter.setSizes(self.mem_sizes)
        else:
            self.mem_sizes = self.splitter.sizes()
            sizes=[self.mem_sizes[0],self.mem_sizes[1],0]
            self.splitter.setSizes(sizes)

    def toggle_em_viz_stacked(self):
        if self.radioButton_view_em.isChecked():
            self.stackedWidget_2.setCurrentIndex(1)
        else:
            self.stackedWidget_2.setCurrentIndex(0)

        self.update_overlay()

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

            for attr in [
                "selection_mask_map_manual",
                "selection_mask_map_auto",
                "selection_mask_map_lib",
                "selection_mask_map",  # si elle existe
                "class_map",
                "_preview_mask",
            ]:
                if hasattr(self, attr):
                    m = getattr(self, attr)
                    if m is None:
                        continue
                    try:
                        m_rot = trans_type(m)
                        # si ça revient en (H,W,1), on squeeze
                        if m_rot.ndim == 3 and m_rot.shape[-1] == 1:
                            m_rot = m_rot[..., 0]
                        setattr(self, attr, m_rot)
                    except Exception as e:
                        print(f"[TRANSFORM] could not rotate {attr}: {e}")

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
        rect_tuple = None
        if use_job:
            ...
        if rect_tuple is None and not use_job:
            rect_tuple = self.saved_rec

        # ➜ fallback: lire le rect courant du viewer si rien en mémoire
        if rect_tuple is None and hasattr(self.viewer_left, "get_rect_coords"):
            coords = self.viewer_left.get_rect_coords()
            if coords:
                x_min, y_min, w, h = coords
                rect_tuple = (y_min, x_min, h, w)

        qrect = self._rect_to_qrectf(rect_tuple)
        if qrect is None:
            if hasattr(self.viewer_left, "clear_selection_overlay"): self.viewer_left.clear_selection_overlay()
            if hasattr(self.viewer_right, "clear_selection_overlay"): self.viewer_right.clear_selection_overlay()
            return

        self.viewer_left.add_selection_overlay(qrect, surface=surface)
        # self.viewer_right.add_selection_overlay(qrect, surface=surface)

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

    def _current_roi_mask(self):
        """Construit un masque bool (H,W) à partir du rectangle sélectionné dans viewer_left.
           Retourne None s’il n’y a pas de sélection."""
        if self.data is None:
            return None
        H, W = self.data.shape[:2]
        rect = self._get_selected_rect()  # (y, x, h, w) ou None
        if not rect:
            return None
        y, x, h, w = rect
        # bornes sûres
        y0 = max(0, int(y));
        x0 = max(0, int(x))
        y1 = min(H, int(y + h));
        x1 = min(W, int(x + w))
        if y1 <= y0 or x1 <= x0:
            return None
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

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
        if self.data is None:
            return

        # 1) Assurer le RGB à jour
        if recompute_rgb or getattr(self, "rgb_image", None) is None:
            self._make_rgb_from_cube()

        base = self.rgb_image.copy()
        H, W = base.shape[:2]

        overlay_img = base.copy()

        # On prépare aussi seg_img pour viewer_right
        seg_img = np.zeros((H, W, 3), dtype=np.uint8)
        if getattr(self, "selection_mask_map", None) is not None:
            for cls, param in getattr(self, "class_info", {}).items():
                b, g, r = param[2]
                seg_img[self.selection_mask_map == cls] = (b, g, r)

        # Lire alpha depuis slider transparence

        alpha_overlay = getattr(self, "alpha", 0.35)
        alpha_overlay = max(0.0, min(1.0, alpha_overlay))

        # ---- NOUVEAU : préparer les masques supplémentaires pour les croix EM auto ----
        # Dictionnaire {cls: mask_bool(H,W)} qui marquera les croix à afficher pour chaque classe
        extra_cross_masks = {}

        # Taille des croix
        try:
            cross_half_size = int(self.horizontalSlider_em_position_size.value())
        except Exception:
            cross_half_size = 0

        if (
                self.active_source == 'auto'
                and cross_half_size > 0
                and hasattr(self, "selection_mask_map_auto")
                and self.selection_mask_map_auto is not None
        ):
            # On va générer les croix par classe auto
            unique_cls = np.unique(self.selection_mask_map_auto)
            unique_cls = [c for c in unique_cls if c >= 0]

            for cls in unique_cls:
                ys, xs = np.where(self.selection_mask_map_auto == cls)
                if ys.size == 0:
                    continue

                # on prend juste la première position comme "endmember position"
                y0 = int(ys[0])
                x0 = int(xs[0])

                # créer le masque booléen pour cette classe si pas encore fait
                if cls not in extra_cross_masks:
                    extra_cross_masks[cls] = np.zeros((H, W), dtype=bool)

                m = extra_cross_masks[cls]

                # dessiner une croix dans m
                hs = max(1, cross_half_size)

                # segment horizontal
                x_min = max(0, x0 - hs)
                x_max = min(W - 1, x0 + hs)
                if 0 <= y0 < H:
                    m[y0, x_min:x_max + 1] = True

                # segment vertical
                y_min = max(0, y0 - hs)
                y_max = min(H - 1, y0 + hs)
                if 0 <= x0 < W:
                    m[y_min:y_max + 1, x0] = True

        # ------------------------------------------------------------------------------

        # 2) Overlay des classes sélectionnées sur l'image RGB (viewer_left)
        if getattr(self, "selection_mask_map", None) is not None and getattr(self, "show_selection", True):
            current = overlay_img.copy()

            for cls, param in self.class_info.items():
                # couleur de la classe
                b, g, r = param[2]

                # masque pixels de la classe
                mask2d = (self.selection_mask_map == cls)

                # ajouter les croix EM auto pour cette classe (si on en a)
                if cls in extra_cross_masks:
                    mask2d = np.logical_or(mask2d, extra_cross_masks[cls])

                if not np.any(mask2d):
                    continue

                # créer un layer uni couleur
                layer = np.zeros_like(overlay_img, dtype=np.uint8)
                layer[:] = (b, g, r)

                # mélange alpha
                blended = cv2.addWeighted(overlay_img, 1.0 - alpha_overlay, layer, alpha_overlay, 0.0)

                # appliquer le résultat sur les pixels de ce mask
                current = np.where(mask2d[:, :, None], blended, current)

            overlay_img = current

        # 3) Aperçu live temporaire (preview rouge)
        if preview and getattr(self, "_preview_mask", None) is not None:
            layer = np.zeros_like(overlay_img, dtype=np.uint8)
            layer[:] = (0, 0, 255)
            mixed = cv2.addWeighted(overlay_img, 0.7, layer, 0.3, 0.0)
            overlay_img = np.where(self._preview_mask[:, :, None], mixed, overlay_img)

        # 4) IMPORTANT : On veut aussi les croix sur viewer_right
        #    seg_img est déjà coloré classe par classe sans alpha.
        #    On va juste peindre les croix (opaques) par dessus.
        #    Ici on réutilise extra_cross_masks.
        for cls, cross_mask in extra_cross_masks.items():
            # couleur : si tu veux cohérence, essaie class_info_auto[cls][2] si dispo sinon class_info[cls][2]
            bgr_color = (0, 255, 255)
            if hasattr(self, "class_info_auto") and cls in self.class_info_auto:
                if len(self.class_info_auto[cls]) >= 3 and self.class_info_auto[cls][2] is not None:
                    bgr_color = self.class_info_auto[cls][2]
            elif cls in getattr(self, "class_info", {}):
                bgr_color = self.class_info[cls][2]

            seg_img[cross_mask] = bgr_color

        # 5) Push vers les viewers
        self.viewer_left.setImage(self._np2pixmap(overlay_img))
        self.viewer_right.setImage(self._np2pixmap(seg_img))
        self._draw_current_rect(surface=False)

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
        self.spec_ax.set_title('Endmembers Spectra')
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
        wl = {"manual": self.wl_manual,
              "auto": self.wl_auto,
              "lib": self.wl_lib,
              }.get(self.active_source, {})

        if wl is None:
            wl=self.wl

        x_graph = wl

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
                b, g, r = self.class_info[c][2]
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
            #     self.spec_ax.legend(loc='upper right', fontsize='small'

            try:
                self.spec_ax.set_title(f"Spectra")
                self.spec_ax.grid(color='black')
                self.spec_ax.set_ylim(0,maxR+0.05)
                self.spec_ax.set_xlim(x_graph[0],x_graph[-1])
                self.spec_ax.set_xlabel("wavelength (nm)")
                self.spec_ax.set_ylabel("Reflectance (a.u.)")
            except:
                print('[UPDATE SPECTRA] problem with x_graph')

        for patch in self.selected_span_patch:
            # patch est un PolyCollection produit par axvspan()
            # On le remet dans l’axe courant :
            self.spec_ax.add_patch(patch)

            # 4) On rafraîchit le canvas

        self.spec_canvas.draw_idle()

    def band_selection(self,checked):
        if checked:

            try:
                msg = QMessageBox(self)
                msg.setWindowTitle("Bands selection")
                msg.setText("Add or suppress bands ")
                add_button = msg.addButton("Add", QMessageBox.AcceptRole)
                remove_button = msg.addButton("Suppress", QMessageBox.AcceptRole)
                reset_button=msg.addButton("Clear all bands", QMessageBox.AcceptRole)
                cancel_button = msg.addButton(QMessageBox.Cancel)
                msg.setDefaultButton(add_button)
                msg.exec_()

                if msg.clickedButton() == add_button:
                    self._band_action = 'add'
                elif msg.clickedButton() == remove_button:
                    self._band_action = 'del'
                elif msg.clickedButton() == reset_button:
                    print('reset')
                    self._band_action = None
                    self.selected_bands = []

                    for patch in self.selected_span_patch:  # reset patch
                        patch.remove()
                        self.selected_span_patch = []

                    self.pushButton_band_selection.setChecked(False)
                    self.spec_canvas.draw_idle()
                    return

                else:
                    self.span_selector.set_active(False)
                    self.pushButton_band_selection.setChecked(False)
                    return

                self.span_selector.set_active(True)
                self.pushButton_band_selection.setText('STOP SELECTION')
            except:
                QMessageBox.warning(
                    self, "Warning",
                    "No band selection choice"
                )
                self.pushButton_band_selection.setChecked(False)

                return

        else:
            self.span_selector.set_active(False)
            self.pushButton_band_selection.setText('Band selection')

    def _rebuild_band_patches_from_selected(self):
        """A partir de self.selected_bands (lista de índices), compacta en intervalos contiguos
           y vuelve a crear parches (axvspan) no solapados."""
        # 1) limpiar parches previos
        for p in getattr(self, 'selected_span_patch', []):
            try:
                p.remove()
            except:
                pass
        self.selected_span_patch = []

        if not self.selected_bands:
            self.spec_canvas.draw_idle()
            return

        # 2) compactar índices consecutivos en bandas
        sb = sorted(set(self.selected_bands))
        bands = []
        start = prev = sb[0]
        for idx in sb[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                bands.append((start, prev))
                start = prev = idx
        bands.append((start, prev))  # último intervalo

        # 3) recrear parches no solapados
        for i0, i1 in bands:
            lam0, lam1 = float(self.wl[i0]), float(self.wl[i1])
            patch = self.spec_ax.axvspan(lam0, lam1, alpha=0.2, color='tab:blue')
            self.selected_span_patch.append(patch)

        self.spec_canvas.draw_idle()

    def on_changes_EM_spectra_viz(self):
        txt = self.comboBox_endmembers_spectra.currentText()
        if 'library' in txt:
            self._activate_endmembers('lib')
        elif 'Manual' in txt:
            self._activate_endmembers('manual')
        else:
            self._activate_endmembers('auto')

    def _assign_initial_colors(self, c=None):
        """
        Assigne une couleur aux classes qui n'en ont pas encore,
        sans jamais écraser le nom existant.
        """
        # Déterminer les classes concernées
        if c is not None:
            class_ids = [c]
        elif getattr(self, 'class_means', None):
            class_ids = list(self.class_means.keys())
        else:
            return  # rien à faire

        # Colormap : tab10 jusqu'à 10 classes, tab20 sinon
        cmap = colormaps.get_cmap('tab10' if len(class_ids) <= 10 else 'tab20')
        n_colors = cmap.N

        # Dictionnaire actif : manual / auto / lib selon active_source
        ci = self.class_info
        if ci is None or not isinstance(ci, dict):
            return

        for cls in class_ids:
            color_idx = cls % n_colors
            r_f, g_f, b_f, _ = cmap(color_idx)
            bgr = (int(255 * b_f), int(255 * g_f), int(255 * r_f))

            # Si l'entrée n'existe pas du tout → créer structure minimale SANS nom
            if cls not in ci:
                ci[cls] = [cls, None, bgr]  # [class_id, name=None, color=BGR]
                continue

            # L'entrée existe :
            # s'assurer qu'elle a 3 champs
            if len(ci[cls]) < 3:
                ci[cls] += [None] * (3 - len(ci[cls]))

            # Ne JAMAIS écraser le nom si ci[cls][1] existe
            # → on met à jour uniquement la couleur
            ci[cls][2] = bgr

    def _draw_cross(self, img, y, x, half_size, color=(0, 255, 255)):
        """
        Dessine une croix opaque dans `img` (BGR) en-place.
        img: np.uint8 (H,W,3) sur laquelle on dessine.
        """
        H, W = img.shape[:2]
        hs = int(max(1, half_size))

        # horizontal
        x_min = max(0, x - hs)
        x_max = min(W - 1, x + hs)
        if 0 <= y < H:
            img[y, x_min:x_max + 1, :] = color

        # vertical
        y_min = max(0, y - hs)
        y_max = min(H - 1, y + hs)
        if 0 <= x < W:
            img[y_min:y_max + 1, x, :] = color

    def fill_form_em(self, source: str):

        form = {
            "manual": self.formLayout_em_manual,
            "auto": self.formLayout_em_auto,
            "lib": self.formLayout_em_lib,
        }.get(source)

        if form is None :
            return

        # clear
        while form.count():
            item = form.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        class_info = {
            "manual": self.class_info_manual,
            "auto": self.class_info_auto,
            "lib": self.class_info_lib,
        }.get(source, {})

        param = {
            "manual": self.param_manual,
            "auto": self.param_auto,
            "lib": self.param_lib,
        }.get(source, {})

        wl={"manual": self.wl_manual,
        "auto": self.wl_auto,
        "lib": self.wl_lib,
        }.get(source, {})

        if param is None:
            param='None'

        if not class_info:
            form.addRow(QLabel("No endmembers yet"))
            return

        spectral_range=f'{int(wl[0])} - {int(wl[-1])}'
        form.addRow(QLabel("wavelength :"), QLabel(spectral_range))

        for key,item in param.items():
            form.addRow(QLabel(str(key)+" :"), QLabel(str(item)))

    def _on_model_viz_change(self):
        idx_job = self.comboBox_viz_show_model.currentIndex()
        if idx_job < 0:
            return

        job_name = self.comboBox_viz_show_model.itemText(idx_job)
        job = self.jobs.get(job_name)
        self.comboBox_viz_show_EM.clear()
        for name in job.labels:
            self.comboBox_viz_show_EM.addItem(name)

        self._refresh_abundance_view()

    def _refresh_abundance_view(self):
        """Affiche la carte d'abondance de l'endmember choisi pour le job sélectionné,
        en utilisant la couleur d'endmember définie."""

        try:
            # On ne fait rien si le mode n’est pas activé
            if not getattr(self, "radioButton_view_abundance", None) or not self.radioButton_view_abundance.isChecked():
                return
            if self.data is None:
                return

            # Récupérer le job sélectionné
            idx_job = self.comboBox_viz_show_model.currentIndex()
            if idx_job < 0:
                return
            job_name = self.comboBox_viz_show_model.itemText(idx_job)
            job = self.jobs.get(job_name)

            if job is None or getattr(job, "A", None) is None:
                return

            H, W = self.data.shape[:2]
            A = job.A  # (p, N) ou (H,W,p)

            # Remodeler en (H, W, p)
            if A.ndim == 2:
                p, N = A.shape
                if N != H * W:
                    # Taille incohérente : on stoppe proprement
                    return
                A3 = A.reshape(p, H, W).transpose(1, 2, 0)  # -> (H,W,p)
            elif A.ndim == 3:
                A3 = A
                # on s’assure d’être bien en (H,W,p)
                if A3.shape[0] == H and A3.shape[1] == W:
                    pass
                elif A3.shape[2] == H and A3.shape[1] == W:
                    A3 = np.transpose(A3, (2, 1, 0))
                else:
                    # si ça ne colle pas, on abandonne l’affichage
                    return
            else:
                return

            # Endmember sélectionné
            em_idx = self.comboBox_viz_show_EM.currentIndex()
            em_idx = max(0, min(em_idx, A3.shape[2] - 1))

            amap = np.asarray(A3[:, :, em_idx], dtype=float)
            # Normalisation pour affichage
            vmin, vmax = float(np.nanmin(amap)), float(np.nanmax(amap))
            if vmax > vmin:
                amap_norm = (amap - vmin) / (vmax - vmin)
            else:
                amap_norm = np.zeros_like(amap, dtype=float)

            # Couleur d’endmember (BGR) selon la source active
            # class_info: {cls: [id, name, (b,g,r)]}
            bgr = (0, 255, 255)  # fallback
            if hasattr(self, "class_info") and isinstance(self.class_info, dict):
                if em_idx in self.class_info and len(self.class_info[em_idx]) >= 3 and self.class_info[em_idx][
                    2] is not None:
                    bgr = self.class_info[em_idx][2]

            b, g, r = [int(x) for x in bgr]
            # Teinte: on applique la valeur d’abondance comme intensité sur la couleur
            # (on reste en BGR pour coller au pipeline _np2pixmap)
            color_vec = np.array([b, g, r], dtype=float).reshape(1, 1, 3)
            img = (amap_norm[..., None] * color_vec).clip(0, 255).astype(np.uint8)

            self.viewer_right.setImage(self._np2pixmap(img))
            self.current_unmix_job_name = job_name
        except Exception as e:
            print("[ABUNDANCE VIEW] error:", e)

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
            # self.selection_mask_map = np.full((H, W), -1, dtype=int)
            self.selection_mask_map_manual = np.full((H, W), -1, np.int32)
            self.selection_mask_map_auto = np.full((H, W), -1, np.int32)
            self.selection_mask_map_lib = np.full((H, W), -1, np.int32)
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
        # self.selection_mask_map = np.full((H, W), -1, dtype=int)
        self.selection_mask_map_manual = np.full((H, W), -1, np.int32)
        self.selection_mask_map_auto = np.full((H, W), -1, np.int32)
        self.selection_mask_map_lib = np.full((H, W), -1, np.int32)
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
        self.param_auto["normalization"]=norm
        self.param_auto["algorithm"] = method
        self.param_auto["number iteration"] = niter
        self.param_auto["number endmembers"] = p
        self.wl_auto=self.wl

        job = EndmemberJob(method=method, p=p, niter=niter, normalization=norm)

        worker = EndmemberWorker(job,self.cube)
        worker.signals.em_ready.connect(self._on_em_ready)
        worker.signals.error.connect(self._on_error)
        self.threadpool.start(worker)

    def _on_em_ready(self, E: np.ndarray,idx_em : np.ndarray, labels: np.ndarray, index_map: Dict[str, np.ndarray]):
        self.labels, self.index_map = labels, index_map

        self.active_source = 'auto'  # pour que class_info pointent sur *_auto

        for i, lab in enumerate(labels):
            name = str(lab)
            self.set_class_name(i, name)

        if self.selection_mask_map_auto is None:
            H, W = self.data.shape[:2]
            self.selection_mask_map_auto = np.full((H, W), -1, np.int32)
        else:
            self.selection_mask_map_auto.fill(-1)

        self.E_auto = {}

        H, W = self.data.shape[:2]
        idx_em = np.asarray(idx_em)
        if idx_em.ndim == 1:
            rows, cols = np.unravel_index(idx_em, (H, W))
        else:
            rows, cols = idx_em[:, 0], idx_em[:, 1]

        for cls, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= c < H and 0 <= r < W:
                self.selection_mask_map_auto[c, r] = cls
                spec = self.data[c,r]
                self.E_auto[cls] = spec
                name = self.get_class_name(cls)

            else :
                print(f'[ENDMEMBERS AUTO] pb idx EM {cls} with coords : ({c},{r})')
                print(f'Cube shape : ({H},{W})')

        print('[ENDMEMBERS]',
              f"Endmembers ready: E shape {len(self.E_auto)},{len(self.E_auto[0])}, groups: {len(np.unique(labels)) if labels is not None else 0}")

        self._activate_endmembers('auto')
        self._assign_initial_colors()
        self.update_spectra(maxR=0)
        self.comboBox_endmembers_spectra.setCurrentText('Auto')
        self.fill_form_em('auto')

    def _on_load_library_clicked(self):
        """
        Load a spectral library from a CSV file:
          - Column 0: wavelength (nm)
          - Columns 1..N: endmember spectra
        Multiple columns can share the same name; their spectra are stacked
        into the same class array with shape (L, n_regions).
        """
        import pandas as pd
        import re

        if getattr(sys, 'frozen', False):
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        open_dir = os.path.join(BASE_DIR, "unmixing", "data")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open spectral library (CSV)", open_dir, "CSV files (*.csv)"
        )
        if not filepath:
            return

        try:
            # 1) Read
            df = pd.read_csv(filepath)
            if df.shape[1] < 2:
                QMessageBox.warning(self, "Library error",
                                    "Invalid format: need wavelength + ≥1 spectrum column.")
                return

            # 2) Columns
            wl = df.iloc[:, 0].to_numpy(dtype=float)
            raw_names = list(df.columns[1:])
            mat = df.iloc[:, 1:].to_numpy(dtype=float)  # (L, P)

            # 3) Build name-keyed dict with stacking on duplicate names
            def canon(name: str) -> str:
                n = re.sub(r"\.\d+$", "", str(name))  # drop ".1", ".2", ...
                n = re.sub(r"\s+", " ", n.strip())  # tidy spaces
                return n

            can_names = [canon(n) for n in raw_names]

            E_by_name = {}  # {name: (L, n_regions)}
            for j, name in enumerate(can_names):
                col = mat[:, j].reshape(-1, 1)  # (L,1)
                if name in E_by_name:
                    E_by_name[name] = np.concatenate([E_by_name[name], col], axis=1)
                else:
                    E_by_name[name] = col

            # 4) Store (name-keyed); no library_groups anymore
            self.active_source = 'lib'  # ensure set_class_name writes into class_info_lib
            for i, name in enumerate(can_names):
                self.set_class_name(i, str(name))
            self.wl_lib = wl
            self.E_lib = E_by_name
            self.param_lib['file'] = filepath

            # 6) Activate LIB source for visualization/unmixing
            self._activate_endmembers('lib')
            self.fill_form_em('lib')
            self.update_spectra()

            QMessageBox.information(
                self, "Library loaded",
                f"Loaded {sum(A.shape[1] for A in E_by_name.values())} spectra "
                f"in {len(E_by_name)} classes from:\n{os.path.basename(filepath)}"
            )


        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.warning(self, "Load error", f"Failed to read library:\n{e}\n\n{tb}")
            print('[LIBRARY ERROR]', e)

    def _on_error(self, msg: str):
        self.signals.error.emit(msg)
        print('[ERROR] : ',msg)

    def _activate_endmembers(self, source: str):
        """
        Construit self.E à partir de la source ('manual'|'auto'|'lib'),
        puis met à jour les moyennes/std et rafraîchit le graphe.
        """
        assert source in ('manual', 'auto', 'lib')
        self.active_source = source

        if source == 'manual':
            # E_manual[c] : (n_regions_c, L)
            self.E = {c: arr for c, arr in (self.E_manual or {}).items()}

        elif source == 'auto':
            # E_auto[c] : (L,) -> E[c] : (L,1)
            try:
                self.E = {c: np.asarray(v, dtype=float).reshape(-1, 1)
                          for c, v in (self.E_auto or {}).items()}
            except Exception:
                self.E = {}

        elif source == 'lib':
            # self.E_lib: {name: (L, n_regions)} ou {name: (L,)}
            E_norm = {}
            for idx, (name, V) in enumerate((self.E_lib or {}).items()):
                A = np.asarray(V, dtype=float)
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                # Ensure shape (L, n_regions)
                if A.shape[0] < A.shape[1]:
                    A = A.T
                E_norm[idx] = A

                # --- PRÉSERVER LES NOMS EXISTANTS ---
                # On ne réécrit le nom que si l’entrée est absente ou vide
                current_name = None
                try:
                    # essaie de lire le nom existant côté 'lib'
                    current_name = self.get_class_name(idx, src='lib')
                except Exception:
                    pass

                if not current_name or current_name.startswith("Class "):
                    self.set_class_name(idx, str(name))

            self.E = E_norm

        else:
            self.E = {}

        if not self.E:
            self.class_means, self.class_stds = {}, {}
            self.update_spectra()
            self.update_overlay()
            return

        # mu/std + couleurs (les couleurs n’écrasent pas les noms)
        self.fill_means_std_classes()
        self._assign_initial_colors()
        self.update_spectra(maxR=1)
        self.update_overlay()

    def get_class_name(self, cls: int, src=None) -> str:
        prefer = src or getattr(self, "active_source", "lib")
        sources = [prefer, "lib", "auto", "manual"]
        dicts = {
            "manual": getattr(self, "class_info_manual", {}) or {},
            "auto": getattr(self, "class_info_auto", {}) or {},
            "lib": getattr(self, "class_info_lib", {}) or {},
        }
        # priorité à la source préférée, sinon fallback
        for s in sources:
            d = dicts.get(s, {})
            if isinstance(d, dict) and cls in d and len(d[cls]) >= 2 and d[cls][1]:
                return str(d[cls][1])
        return f"Class {cls}"

    def set_class_name(self, cls: int, name):
        """
        Store class name as string in class_info (per-source),
        keeping your [label, name, (B,G,R)] pattern.
        """
        sname = "" if name is None else str(name)

        info = self.class_info
        if not isinstance(info, dict):
            return

        entry = info.get(cls)
        if isinstance(entry, (list, tuple)):
            lst = list(entry)
            # ensure [label, name, (B,G,R)]
            if len(lst) == 0:
                lst = [cls, sname, (0, 255, 0)]
            elif len(lst) == 1:
                lst.append(sname)
                lst.append((0, 255, 0))
            else:
                lst[1] = sname
            info[cls] = lst
        else:
            # create fresh entry with default color
            info[cls] = [cls, sname, (0, 255, 0)]

    def save_endmembers_spectra(self):
        """
        Save endmembers from the active source (manual/auto/lib) into a CSV:
          Wavelength, ClassA, ClassA.1, ClassB, ClassC, ClassC.1, ...
        Shapes are normalized to (L, n_specs) per class before writing.
        """
        import pandas as pd
        from collections import defaultdict

        # 1) Pick source dict + wavelength + name getter for display
        src = (self.active_source or "").lower()

        if src == "manual":
            E_src = self.E_manual if isinstance(self.E_manual, dict) else {}
            wl = getattr(self, "wl", None)

            def _name_for(key):
                try:
                    return self.get_class_name(key)
                except Exception:
                    return f"class{key}"

        elif src == "auto":
            E_src = self.E_auto if isinstance(self.E_auto, dict) else {}
            wl = getattr(self, "wl_auto", None)

            def _name_for(key):
                try:
                    return self.get_class_name(key)
                except Exception:
                    return f"class{key}"

        elif src == "lib":
            # keys are already class names
            E_src = self.E_lib if isinstance(self.E_lib, dict) else {}
            wl = getattr(self, "wl_lib", None)

            def _name_for(key):
                return str(key)
        else:
            QMessageBox.warning(self, "Save endmembers", "Unknown active source.")
            return

        if wl is None or len(E_src) == 0:
            QMessageBox.warning(self, "Save endmembers", "No spectra to save for this source.")
            return

        # 2) Normalize every entry to shape (L, n_specs)
        def _to_LxN(arr):
            A = np.asarray(arr, dtype=float)
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            # ensure (L, n)
            if A.shape[0] < A.shape[1]:
                A = A.T
            return A

        try:
            # 3) Build a DataFrame
            df = pd.DataFrame({"Wavelength": np.asarray(wl, dtype=float)})
            name_counts = defaultdict(int)

            # To keep a stable order: numeric keys sorted; string keys alphabetical
            def _sort_key(k):
                return (0, int(k)) if isinstance(k, (int, np.integer)) else (1, str(k))

            for key in sorted(E_src.keys(), key=_sort_key):
                base_name = _name_for(key) or f"class{key}"
                A = _to_LxN(E_src[key])  # (L, n_specs)

                for j in range(A.shape[1]):
                    # First spectrum keeps base name; subsequent add .1, .2, ...
                    cnt = name_counts[base_name]
                    col_name = base_name if cnt == 0 else f"{base_name}.{cnt}"
                    name_counts[base_name] += 1

                    # Add column
                    df[col_name] = A[:, j]

            # 4) Ask where to save
            if getattr(sys, 'frozen', False):
                BASE_DIR = sys._MEIPASS
            else:
                BASE_DIR = os.path.dirname(os.path.dirname(__file__))

            default_dir = os.path.join(BASE_DIR, "unmixing", "data")
            default_name = f"endmembers_{src}.csv"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save endmembers as CSV", os.path.join(default_dir, default_name), "CSV files (*.csv)"
            )
            if not path:
                return

            # 5) Write CSV
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Save endmembers",
                                    f"Saved {df.shape[1] - 1} spectra to:\n{os.path.basename(path)}")

        except Exception as e:
            import traceback
            QMessageBox.warning(self, "Save endmembers", f"Failed to save:\n{e}\n\n{traceback.format_exc()}")

    def open_em_editor(self):
        """
        Open the EM editor on the current active source.
        Reads class ids, names, colors from class_info; writes back on Save.
        """
        # Collect rows from current source
        # We’ll use class_means (computed from E) to decide which classes exist/display;
        # then pull names/colors from class_info. If missing, default nicely.
        try:
            class_ids = sorted(self.class_means.keys())
        except Exception:
            class_ids = sorted(getattr(self, "E", {}).keys())

        rows = []
        for cls in class_ids:
            # Name
            try:
                name = self.get_class_name(cls)
            except Exception:
                name = f"class{cls}"

            # Color (your storage is BGR in class_info[cls][2])
            try:
                b, g, r = self.class_info[cls][2]  # BGR stored, as in your tool
            except Exception:
                # fallback soft colors
                b, g, r = (64 + (37 * cls) % 192, 64 + (91 * cls) % 192, 64 + (53 * cls) % 192)

            rows.append((cls, name, (b, g, r)))

        dlg = EMEditDialog(self, rows)
        if dlg.exec_() != QDialog.Accepted:
            return

        # Write back names/colors
        updated = dlg.result_rows()
        for cls, name, (b, g, r) in updated:
            # name
            self.set_class_name(cls, name)  # keeps your per-source mapping consistent
            # color (BGR)
            try:
                info = self.class_info.get(cls, None)
                if info is None:
                    # class_info entry: [label, name GT, (B,G,R)] in your pattern—preserve style if needed
                    self.class_info[cls] = [cls, name, (b, g, r)]
                else:
                    # ensure a triplet slot exists
                    if len(info) < 3:
                        while len(info) < 2:
                            info.append(name)
                        info.append((b, g, r))
                    else:
                        info[2] = (b, g, r)
            except Exception:
                # if class_info is None or not a dict, initialize minimally
                if not isinstance(self.class_info, dict):
                    # if the property returns a proxy, make sure assignment works
                    pass
                self.class_info[cls] = [cls, name, (b, g, r)]

        # Refresh UI that uses names/colors
        self.update_spectra()
        self.update_overlay()

        # If you list EMs in a combo, refresh labels there too
        try:
            cb = self.comboBox_viz_show_EM
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            for cls in class_ids:
                cb.addItem(self.get_class_name(cls))
            idx_restore = cb.findText(cur)
            cb.setCurrentIndex(idx_restore if idx_restore >= 0 else 0)
            cb.blockSignals(False)
        except Exception:
            pass

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
        """Callback del SpanSelector con merge automático de spans."""
        if self._band_action is None:
            return

        if lambda_min > lambda_max:
            lambda_min, lambda_max = lambda_max, lambda_min

        idx_min = int(np.argmin(np.abs(self.wl - lambda_min)))
        idx_max = int(np.argmin(np.abs(self.wl - lambda_max)))

        if self._band_action == 'add':
            for idx in range(idx_min, idx_max + 1):
                if idx not in self.selected_bands:
                    self.selected_bands.append(idx)

        elif self._band_action == 'del':
            # quitar los índices seleccionados de ese rango
            for idx in range(idx_min, idx_max + 1):
                if idx in self.selected_bands:
                    self.selected_bands.remove(idx)

        # Ordenar y reconstruir SIEMPRE: así evitamos solapes y hacemos el "merge".
        self.selected_bands = sorted(set(self.selected_bands))
        self._rebuild_band_patches_from_selected()

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
                self.param_manual={}

        self.selecting_pixels = True
        # self.viewer_left.setDragMode(QGraphicsView.NoDrag)
        self.show_rgb_image()
        self.update_overlay()
        self.viewer_left.enable_rect_selection = False

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
            print(key,'->',self.E_manual[key].shape)

        self._activate_endmembers('manual')
        self.update_spectra(maxR=1)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')
        self.viewer_left.enable_rect_selection = True

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
        self.update_spectra(maxR=1)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')

        # 3) rafraîchir l’affichage
        self.wl_manual=self.wl
        self.param_manual["number endmembers"]=len(self.E_manual)
        for key in self.E_manual:
            new_key=self.class_info_manual[key][1] + "  #spec "
            self.param_manual[new_key]=self.E_manual[key].shape[0]

        self.fill_form_em('manual')
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

        self._activate_endmembers('manual')
        self.update_spectra(maxR=1)
        self.comboBox_endmembers_spectra.setCurrentText('Manual')

        self.param_manual["number endmembers"] = len(self.E_manual)
        for key in self.E_manual:
            new_key = self.class_info_manual[key][1] + "  #spec "
            self.param_manual[new_key] = self.E_manual[key].shape[0]

        self.fill_form_em('manual')
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

        print(f'[MANUAL SELECTION] ONE SELEC E_manual shapes ')
        for key in self.E_manual:
            print(key, '->', self.E_manual[key].shape)

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

    def fill_means_std_classes(self):
        full_means = {}
        full_stds = {}

        for key, M in self.E.items():
            M = np.asarray(M)

            # On veut M shape = (L, n_regions)
            # Si on reçoit (1, L) au lieu de (L,1), on le remet d'équerre
            if M.ndim == 2:
                L, N = M.shape
                if L < N:
                    # ex: (1, 261) -> transpose en (261, 1)
                    M = M.T
            else:
                # cas dégénéré: vecteur 1D -> on force en (L,1)
                M = M.reshape(-1, 1)

            mu = M.mean(axis=1)  # (L,)
            sigma = M.std(axis=1)  # (L,)

            full_means[key] = mu
            full_stds[key] = sigma

        self.class_means = full_means
        self.class_stds = full_stds

    def prune_unused_classes(self):
        """
        Supprime self.class_info
        tous les labels qui ne figurent plus dans self.cls_map.
        """
        if self.class_info is None:
            return

        labels_in_map = set(np.unique(self.cls_map))
        for d in (self.class_info):
            for cls in list(d.keys()):
                if cls not in labels_in_map:
                    del d[cls]

    def _get_E(self):
        """Public method to retrieve E from a specified source."""
        return {
            'manual': self.E_manual,
            'auto': self.E_auto,
            'lib': self.E_lib}[self.active_source]

    def _set_E(self, E):
        if self.active_source == 'manual':
            self.E_manual = E
        elif self.active_source == 'auto':
            self.E_auto = E
        else:
            self.E_lib = E

    E = property(_get_E, _set_E)

    @property
    def selection_mask_map(self):
        return {'manual': self.selection_mask_map_manual,
                'auto': self.selection_mask_map_auto,
                'lib': self.selection_mask_map_lib}[self.active_source]

    @property
    def class_info(self):
        return {'manual': self.class_info_manual,
                    'auto': self.class_info_auto,
                    'lib': self.class_info_lib}[self.active_source]

    @property
    def norm_em(self):
        return {'manual': self.norm_manual,
                'auto': self.norm_auto,
                'lib': self.norm_lib}[self.active_source]

    # </editor-fold>

    # <editor-fold desc="Unmixing Job Queue">
    def _init_classification_table(self, table):
        from PyQt5.QtWidgets import QTableWidgetItem
        headers = ["Name","Status", "Progress", "Params", "Algo", "EM source", "Duration"]
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
        table.setColumnWidth(2, 90)
        table.setColumnWidth(3, 320)
        table.setColumnWidth(4, 90)
        table.setColumnWidth(5, 120)

    def _make_progress_bar(self,val=0):
        from PyQt5.QtWidgets import QProgressBar
        pb = QProgressBar()
        pb.setRange(0, 100)
        pb.setValue(val)
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
        txt = self.comboBox_normalisation_unmix.currentText()
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

        # --- Canonicalisation des paramètres pour comparaison robuste
        def _freeze_params(d):
            def _norm_val(v):
                if isinstance(v, float):
                    return round(v, 10)  # évite les micro-différences
                return v

            return tuple(sorted((k, _norm_val(v)) for k, v in d.items()))

        frozen = _freeze_params(P)

        # --- Vérification de doublon (mêmes params = même job)
        for exist in self.jobs.values():
            if getattr(exist, "_frozen_params", None) == frozen:
                QMessageBox.information(
                    self,
                    "Duplicate job",
                    f"An identical job ('{exist.name}') already exists.\nNo new job was added."
                )
                return

        base = f"{P['algo']} ({P['em_src']})"
        name = self._ensure_unique_name(base)

        # Construire l’objet job (on ne calcule/attache pas encore E ici)
        job = UnmixJob(
            name=name,
            em_src=P['em_src'],
            model=P["algo"],
            normalization=P["norm"],
            max_iter=P["max_iter"],
            tol=P["tol"],
            lam=P["lam"],
            rho=P["rho"],
            anc=P["anc"],
            asc=P["asc"],
            params=P,
            _frozen_params=frozen,
        )
        # Tu pourras plus tard remplir job.E et/ou un tag sur la source EM au moment du run.

        self.jobs[name] = job
        self.job_order.append(name)

        # Insère la ligne dans la table
        self._insert_job_row(name, P)
        self.tabWidget.setCurrentIndex(2)
        self._refresh_viz_model_combo(select_name=name)

    def _insert_job_row(self, name: str, P: dict):
        from PyQt5.QtWidgets import QTableWidgetItem
        table = self.tableWidget_classificationList
        row = table.rowCount()
        table.insertRow(row)

        job=self.jobs.get(name)

        # Col 0: Name
        table.setItem(row, 0, QTableWidgetItem(name))
        # Col 1: Status
        table.setItem(row, 1, QTableWidgetItem(job.status))
        # Col 2: Progress (widget)
        pb = self._make_progress_bar(val=job.progress)
        table.setCellWidget(row, 2, pb)
        # Col 3: Params
        table.setItem(row, 3, QTableWidgetItem(self._format_params_summary(P)))
        # Col 3: Algo
        table.setItem(row, 4, QTableWidgetItem(P["algo"]))
        # Col 4: EM source (+ merge)
        em_txt = P["em_src"] + (" + merge" if P["em_merge"] else "")
        table.setItem(row, 5, QTableWidgetItem(em_txt))
        # Col 6: Duration
        table.setItem(row, 6, QTableWidgetItem("-"))

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
                em_src=job.em_src,
                em_merge=job.merge_EM
            )
            self._insert_job_row(name, P)

        if sorting: table.setSortingEnabled(True)

        self._refresh_viz_model_combo()

    def _update_row_from_job(self, name: str):
        """Appelle ceci pendant l’exécution plus tard pour refléter status/progress/duration."""
        table = self.tableWidget_classificationList
        NAME_COL, STATUS_COL, PROG_COL, DUR_COL = 0, 1, 2, 6
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
        """Remove the selected job from the table, with confirmation if it's Done."""
        table = self.tableWidget_classificationList
        row = table.currentRow()
        if row < 0:
            return

        name_item = table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        job = self.jobs.get(name)

        # 🔒 Ask for confirmation if the job is Done
        if job and getattr(job, "status", "") == "Done":
            from PyQt5.QtWidgets import QMessageBox
            ans = QMessageBox.question(
                self,
                "Delete finished job",
                f"The job “{name}” is marked as Done.\n"
                f"Do you really want to delete it? Results stored in memory will be lost.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if ans != QMessageBox.Yes:
                return

        # Actual deletion
        self.jobs.pop(name, None)
        try:
            self.job_order.remove(name)
        except ValueError:
            pass
        table.removeRow(row)

        # Refresh UI
        self._refresh_viz_model_combo()
        self._refresh_table()

    def remove_all_jobs(self):
        """Clear all jobs, with confirmation if there are completed ones."""
        done_count = sum(
            1 for n in self.job_order
            if getattr(self.jobs.get(n, None), "status", "") == "Done"
        )

        if done_count > 0:
            from PyQt5.QtWidgets import QMessageBox
            ans = QMessageBox.question(
                self,
                "Delete completed jobs",
                f"There are {done_count} job(s) marked as Done.\n"
                f"Do you really want to delete them? All stored results will be lost.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if ans != QMessageBox.Yes:
                return

        # Actual reset
        self.jobs.clear()
        self.job_order.clear()
        self._init_classification_table(self.tableWidget_classificationList)
        self._refresh_viz_model_combo()
        self._refresh_table()

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

    def _refresh_viz_model_combo(self, select_name: str = None):
        """Rebuild comboBox_viz_show_model with ALL jobs present in job_order (Queued/Running/Done).
           Keep selection when possible; disable if empty."""
        cb = self.comboBox_viz_show_model  # UI: defined in unmixing_window.ui/py
        was_blocked = cb.blockSignals(True)
        try:
            # Mémorise la sélection actuelle
            current_text = cb.currentText() if cb.count() > 0 else None

            cb.clear()
            for name in self.job_order:
                if name in self.jobs:
                    cb.addItem(name)

            # Cible de sélection : priorité à select_name, sinon l'ancienne si encore présente
            target = select_name or current_text
            if target:
                idx = cb.findText(target)
                if idx != -1:
                    cb.setCurrentIndex(idx)

            cb.setEnabled(cb.count() > 0)
        finally:
            cb.blockSignals(was_blocked)

    # </editor-fold>

    # <editor-fold desc="Unmixing on work">
    def _on_start_all(self):
        if not self.job_order: return
        self._stop_all = False
        self._run_next_in_queue()
        self.radioButton_view_abundance.setChecked(True)
        self._refresh_abundance_view()

    def _on_start_selected_or_last(self):
        """
        Lanza el job seleccionado en la tabla, o si no hay selección,
        el último añadido. Prepara E/labels desde la fuente elegida
        y arranca el worker en el threadpool.
        """
        self._stop_all = False

        # 1) Resolver qué job lanzar (seleccionado o último)
        table = self.tableWidget_classificationList
        row = table.currentRow()
        if row < 0:
            # sin selección -> usa la última fila si existe
            if table.rowCount() == 0:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Unmixing", "No hay jobs en la cola.")
                return
            row = table.rowCount() - 1
            table.selectRow(row)

        name_item = table.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()
        job = self.jobs.get(name)
        if job is None:
            return

        # Evitar relanzar si está corriendo
        if getattr(job, "status", "") == "Running":
            return

        # 2) Preparar datos del job: cube y endmembers
        if self.data is None or self.wl is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Unmixing", "Carga un cubo primero.")
            return
        job.cube = self.data  # (H,W,L)  -> requerido por UnmixWorker.run() :contentReference[oaicite:0]{index=0}

        # Fuente EM desde la UI ("From library" | "Manual" | "Auto") :contentReference[oaicite:1]{index=1}
        src_txt = self.comboBox_endmembers_use_for_unmixing.currentText()
        if "library" in src_txt.lower():
            src = "lib"
        elif "manual" in src_txt.lower():
            src = "manual"
        else:
            src = "auto"

        # Activar esa fuente para tener self.E coherente y con formas estandarizadas
        # (_activate_endmembers rellena y normaliza self.E por fuente, y recalcula medias/std) :contentReference[oaicite:2]{index=2}
        try:
            self._activate_endmembers(src)  # asegura self.E = dict{clase: (L,n_reg) o (L,1)}
        except Exception:
            pass

        # Merge por grupos (media de regiones por clase) según checkbox de la UI :contentReference[oaicite:3]{index=3}
        merge_groups = self.checkBox_unmix_merge_EM_groups.isChecked()

        # 2.1) Construir matriz E (L,p) y vector de labels (p,) para agrupar mapas
        #      Usamos una función local que homogeniza (L,n) para cada clase
        def _to_LxN(arr):
            A = np.asarray(arr, dtype=float)
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            if A.shape[0] < A.shape[1]:
                A = A.T
            return A  # (L,n)

        E_cols = []
        labels = []

        # Mantener orden estable: claves numéricas primero ordenadas, luego texto alfabético
        def _sort_key(k):
            import numpy as _np
            return (0, int(k)) if isinstance(k, (int, _np.integer)) else (1, str(k))

        for cls in sorted(self.E.keys(), key=_sort_key):
            A = _to_LxN(self.E[cls])  # (L,n_cls)
            if merge_groups:
                mu = A.mean(axis=1, keepdims=True)  # (L,1)
                E_cols.append(mu)
                labels.append(int(cls) if isinstance(cls, (int, np.integer)) else len(labels))
            else:
                E_cols.append(A)  # (L,n_cls)
                labels.extend([int(cls) if isinstance(cls, (int, np.integer)) else len(labels)] * A.shape[1])

        if not E_cols:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Unmixing", "No hay endmembers en la fuente seleccionada.")
            return

        E_mat = np.concatenate(E_cols, axis=1)  # (L,p)
        job.E = E_mat
        job.labels = np.asarray(labels, dtype=int)
        job.roi_mask = None  # ROI opcional; aún no gestionado

        # Guardar trazas útiles en el job (por si luego necesitas mostrarlas)
        job.merge_EM = merge_groups
        job.wl_job = self.wl
        job.wl_cube = self.wl
        job.wl_em = {"manual": self.wl,
                     "auto": getattr(self, "wl_auto", self.wl),
                     "lib": getattr(self, "wl_lib", self.wl)}.get(src, self.wl)

        # 3) Actualizar estado en la tabla y lanzar el worker
        job.status = "Queued"
        job.progress = 0
        job._t0 = time.time()
        self._update_row_from_job(name)  # refresca status/progreso/duración en la fila :contentReference[oaicite:4]{index=4}

        # Si rien ne tourne, démarre la queue
        if self._current_worker is None:
            self._run_next_in_queue()

    def _run_next_in_queue(self):
        # Ne lance pas si on a stoppé la chaîne ou si un worker tourne déjà
        if self._stop_all or (self._current_worker is not None):
            return

        # Trouve le prochain job en attente
        next_name = None
        for n in self.job_order:
            st = getattr(self.jobs.get(n, None), "status", "")
            if st in ("", None, "Queued"):
                next_name = n
                break
        if next_name is None:
            return  # rien à exécuter

        job = self.jobs[next_name]

        # ---- Préparation des endmembers et des inputs ICI ----
        src_txt = job.em_src.lower()
        src = "lib" if "library" in src_txt else ("manual" if "manual" in src_txt else "auto")
        print('[RUN NEXT JOB] : ',src,' - ',src_txt)
        merge_groups = self.checkBox_unmix_merge_EM_groups.isChecked()

        # (ré)active la source sélectionnée -> remplit self.E standardisée
        self._activate_endmembers(src)

        # construit E_mat, labels (réinterpole si 'lib' nécessaire)
        E_mat, labels = self._prepare_job_inputs_from_current_E(src=src, merge_groups=merge_groups)

        # Remplit le job
        job.cube = np.asarray(self.cube.data, dtype=float)  # H×W×L
        job.E = E_mat  # L×p
        job.labels = labels  # (p,)
        job.roi_mask = self._current_roi_mask()
        job.wl_job = self.wl

        # Marque comme Running
        import time
        job.status = "Running"
        job.progress = 0
        job._t0 = time.time()
        self._update_row_from_job(next_name)
        self._refresh_viz_model_combo(select_name=next_name)
        self._on_model_viz_change()

        # Crée le worker et mémorise-le comme courant
        worker = UnmixWorker(job)
        self._current_worker = worker
        self._running_idx = self.job_order.index(next_name) if next_name in self.job_order else -1

        # Connexions (injecter le nom pour les handlers)
        worker.signals.progress.connect(lambda v,A, n=next_name: self._on_unmix_progress(n, v,A))
        worker.signals.error.connect(lambda msg, n=next_name: self._on_unmix_error(n, msg))
        worker.signals.unmix_ready.connect(lambda A, E, maps, n=next_name: self._on_unmix_ready(n, A, E, maps))

        # Lance (pool mono-thread => séquentiel)
        self.threadpool.start(worker)

    def _prepare_job_inputs_from_current_E(self, src: str, merge_groups: bool):
        """
        Usa self.E (dict clase->(L,n_regiones)) ya construida por _activate_endmembers(src)
        para devolver:
          - E_mat: (L, p) concatenando columnas por clase
          - labels: (p,) con el label de grupo por columna (nombre de clase si merge=True,
                    o 'Class i - #k' si no merge)
        Si la fuente es 'lib' y las WL no coinciden con el cubo, reinterpola E a self.wl.
        """
        # self.E: {cls_id: (L, n_regions)} ya normalizado de la fuente elegida

        # Gestionar wavelentghs por fuente
        wl_cube = np.asarray(self.wl, dtype=float)
        if src == "manual":
            wl_em = self.wl_manual
            E_dict=self.E_manual

        elif src == "auto":
            wl_em = self.wl_auto
            E_dict = self.E_auto
        else:  # 'lib'
            wl_em = np.asarray(self.wl_lib, dtype=float)
            E_dict = self.E_lib

        # 1) Reinterpolar a la rejilla del cubo si hace falta (solo lib)
        def _to_LxN(arr):
            A = np.asarray(arr, dtype=float)
            if A.ndim == 1:
                A = A.reshape(-1, 1)
            if A.shape[0] < A.shape[1]:
                A = A.T
            return A

        E_resampled = []
        col_labels = []

        # nombre legible de cada clase
        def _class_name(cid,src=None):
            try:
                return self.get_class_name(cid,src)
            except Exception:
                return f"Class {cid}"

        for cls_id in sorted(E_dict.keys()):
            A = _to_LxN(E_dict[cls_id])  # (L_em, n_reg)
            # resample si procede
            if src == "lib" and wl_em is not None and not np.array_equal(wl_em, wl_cube):
                # interp columna a columna
                L_target = wl_cube.size
                A_rs = np.empty((L_target, A.shape[1]), dtype=float)
                for j in range(A.shape[1]):
                    A_rs[:, j] = np.interp(wl_cube, wl_em, A[:, j])
                A = A_rs

            # labels por columna
            if merge_groups:
                grp = _class_name(cls_id)  # todas las columnas de esta clase comparten grupo
                col_labels.extend([grp] * A.shape[1])
            else:
                base = _class_name(cls_id)
                for j in range(A.shape[1]):
                    col_labels.append(f"{base} #{j + 1}")

            E_resampled.append(A)

        if not E_resampled:
            raise ValueError("No endmembers for this source.")

        # 2) Concatenar columnas
        E_mat = np.concatenate(E_resampled, axis=1)  # (L, p total)
        labels = np.asarray(col_labels, dtype=object)
        return E_mat, labels

    def _on_cancel_queue(self):
        self._stop_all = True
        if self._current_worker:
            try:
                self._current_worker.cancel()  # nécessite un flag côté worker
            except Exception:
                pass
        # la libération se fera dans le callback d’erreur/finish

    # -- Job progress UI -------------------------------------------------
    def _on_unmix_progress(self, name: str, value: int,A: np.ndarray):
        """MAJ douce de la barre de progression pour le job `name`."""
        job = self.jobs.get(name)
        if not job:
            return
        job.progress = max(0, min(100, int(value)))
        job.A = A
        self._refresh_abundance_view()


        self._update_row_from_job(name)  # met à jour Status/Progress/Durée dans la table
        if getattr(self, "radioButton_view_abundance", None) and self.radioButton_view_abundance.isChecked():
            self._refresh_abundance_view()

    # -- Job error UI ----------------------------------------------------
    def _on_unmix_error(self, name: str, message: str):
        job = self.jobs.get(name)
        print(f'[UNMIXING ERROR] with {name}')
        print(message)

        if job:
            txt = (message or "").lower()
            job.status = "Canceled" if ("canceled" in txt or "cancelled" in txt) else "Error"
            job.progress = 0 if job.status != "Done" else 100
            try:
                import time
                job.duration_s = time.time() - job._t0 if getattr(job, "_t0", None) else None
            except Exception:
                job.duration_s = None
            self._update_row_from_job(name)

        # >>> Libère et enchaîne
        self._current_worker = None
        self._run_next_in_queue()

    # -- Job success UI --------------------------------------------------
    def _on_unmix_ready(self, name: str, A: np.ndarray, E_used: np.ndarray, maps_by_group: dict):
        job = self.jobs.get(name)
        if not job:
            return

        # Stockage
        job.A = A
        job.E_used = E_used
        job.maps_by_group = maps_by_group or {}
        job.status = "Done"
        job.progress = 100

        try:
            import time
            job.duration_s = time.time() - job._t0 if getattr(job, "_t0", None) else None
        except Exception:
            job.duration_s = None

        self._update_row_from_job(name)
        self._refresh_abundance_view()

        # >>> Libère et enchaîne
        self._current_worker = None
        if not self._stop_all:
            self._run_next_in_queue()

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

    w.active_source='auto'
    w._on_extract_endmembers()
    w.comboBox_endmembers_use_for_unmixing.setCurrentText('Auto')

    sys.exit(app.exec_())