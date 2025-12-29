# identification_tool.py

import sys
import os
import numpy as np
import cv2
from PIL import Image
import h5py

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

from identification.identification_window import Ui_IdentificationWidget
from hypercubes.hypercube import Hypercube
from interface.some_widget_for_interface  import ZoomableGraphicsView, LoadingDialog, LoadCubeWorker
from identification.load_cube_dialog import Ui_Dialog

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

class ClassifySignals(QObject):
    finished = pyqtSignal()                          # fin (toujours émis)
    error = pyqtSignal(str)                          # message d'erreur
    progress = pyqtSignal(int)                       # 0..100
    result = pyqtSignal(np.ndarray)                  # prédictions finales (shape: [N,])
    result_partial = pyqtSignal(int, int, np.ndarray)  # (start, end, preds_chunk)

class ClassifyWorker(QRunnable):
    """
    Exécute une classification sur 'spectra' (np.ndarray [N_pixels, N_bands]) en CHUNKS.
    - classifier: scikit-learn pipeline/estim.
    - classifier_type: "sklearn" ou "cnn".
    - chunk_size: nb de pixels par batch (équilibre RAM/vitesse).
    - emit_partial: si True, émet result_partial(start, end, preds_chunk) à chaque chunk.
      -> permet d'afficher la class_map PROGRESSIVEMENT.
    """
    def __init__(self, spectra, classifier, classifier_type,chunk_size= 100_000, emit_partial= True):
        super().__init__()
        self.spectra = spectra
        self.classifier = classifier
        self.classifier_type = (classifier_type or "").lower()
        self.chunk_size = int(max(1, chunk_size))
        self.emit_partial = bool(emit_partial)
        self.signals = ClassifySignals()
        self._cancel = False

    def cancel(self):
        """Demande d'annulation (prise en compte entre deux chunks)."""
        self._cancel = True

    @pyqtSlot()
    def run(self):
        try:
            X = self.spectra
            if X is None or not isinstance(X, np.ndarray) or X.ndim != 2:
                raise ValueError("Invalid spectra: expected 2D numpy array [N_pixels, N_bands].")

            n = X.shape[0]
            if n == 0:
                self.signals.result.emit(np.empty((0,), dtype=np.int32))
                return

            preds = np.empty(n, dtype=np.int32)

            # Boucle par chunks
            for start in range(0, n, self.chunk_size):
                if self._cancel:
                    # On arrête proprement (pas de result final).
                    self.signals.error.emit("Classification canceled.")
                    self.signals.finished.emit()
                    return

                end = min(start + self.chunk_size, n)
                Xc = X[start:end, :]

                if self.classifier_type == "cnn":
                    # PyTorch CPU (pas de CUDA ici)
                    try:
                        import torch
                    except Exception as e:
                        raise RuntimeError(
                            "PyTorch not available for 'cnn' classifier_type. "
                            "Install torch (CPU build) or switch to 'sklearn'."
                        ) from e

                    with torch.no_grad():
                        t = torch.tensor(Xc, dtype=torch.float32)  # CPU
                        out = self.classifier(t)                   # shape [B, C]
                        pc = out.argmax(dim=1).cpu().numpy().astype(np.int32)
                else:
                    # scikit-learn (idéalement un Pipeline qui inclut StandardScaler)
                    pc = self.classifier.predict(Xc).astype(np.int32)

                preds[start:end] = pc

                # Résultat partiel pour affichage progressif (optionnel)
                if self.emit_partial:
                    self.signals.result_partial.emit(start, end, pc)

                # Progression
                pct = int((end * 100) / n)
                # clamp pour éviter >100
                self.signals.progress.emit(100 if pct > 100 else pct)

            # Émission du résultat complet
            self.signals.result.emit(preds)

        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n{tb}")
        finally:
            self.signals.finished.emit()

class TrainSignals(QObject):
    finished = pyqtSignal()           # always emitted
    error = pyqtSignal(str)
    progress = pyqtSignal(int)        # 0..100
    saved = pyqtSignal(str)           # path to saved model
    progress_indeterminate = pyqtSignal(bool)

class TrainWorker(QRunnable):
    """
    Trains a model for (clf_type, kind) on the local training set, then saves it to savepath.
    Emits .saved(savepath) on success.
    """
    def __init__(self, clf_type: str, kind: str, savepath: str, train_wl: np.ndarray):
        super().__init__()
        self.clf_type = (clf_type or "").upper()
        self.kind = kind
        self.savepath = savepath
        self.train_wl = np.asarray(train_wl, float)
        self.signals = TrainSignals()
        self._cancel = False

    def cancel(self): self._cancel = True

    def crop_training_to_target_wl(self,X_train):
        """
        Crops columns of X_train (assumed to be on 400..1700 nm @ 5 nm) so they match wl_tgt.
        Returns (X_crop, wl_used) with wl_used == wl_tgt.
        """
        BASE_MIN_NM = 400.0
        BASE_MAX_NM = 1700.0
        BASE_STEP_NM = 5.0

        wl_tgt=self.train_wl

        # quick sanity checks
        if wl_tgt.size == 0:
            raise ValueError("wl_tgt is empty.")
        if (wl_tgt.min() < BASE_MIN_NM - 1e-9) or (wl_tgt.max() > BASE_MAX_NM + 1e-9):
            raise ValueError(f"wl_tgt must be within [{BASE_MIN_NM}, {BASE_MAX_NM}] nm.")

        # Expected source wavelengths for the training dataset
        wl_src = np.arange(BASE_MIN_NM, BASE_MAX_NM + BASE_STEP_NM, BASE_STEP_NM, dtype=float)

        # Map target wavelengths to integer indices on the 5 nm grid
        # idx = (wl - 400) / 5  -> integer
        idx = np.rint((wl_tgt - BASE_MIN_NM) / BASE_STEP_NM).astype(int)

        # Bounds check
        if (idx < 0).any() or (idx >= wl_src.size).any():
            raise ValueError("Some target wavelengths are outside the training grid.")

        # Shape check: X_train columns must match full wl_src length
        if X_train.shape[1] != wl_src.size:
            raise ValueError(
                f"X_train has {X_train.shape[1]} bands but expected {wl_src.size} "
                "for 400..1700 nm @ 5 nm. Ensure you load the full training feature set."
            )

        X_crop = X_train[:, idx]
        # wl_src[idx] should equal wl_tgt (numerically)
        return X_crop, wl_tgt

    @pyqtSlot()
    def run(self):
        try:
            import joblib
            from pathlib import Path
            base = Path(__file__).resolve().parents[1] / "identification" / "data"

            # --- early bail
            if self._cancel:
                self.signals.error.emit("Training canceled.")
                return

            X_train = None; y_train = None
            if "background" in self.kind.lower():
                import h5py
                fname = "XY_train_background_10pct.h5" if "10pct" in self.kind.lower() else "XY_train_background.mat"
                f = h5py.File(base / fname, "r")
                X_train = np.array(f['X']).T
                y_train = np.array(f['Y'][0])
            else:
                import scipy.io
                data = scipy.io.loadmat(base / "XY_train_3classes.mat")
                X_train = data['X']
                y_train = data['Y'].ravel()

            if self._cancel:
                self.signals.error.emit("Training canceled.")
                return

            X_aligned, wl_used = self.crop_training_to_target_wl(X_train)

            if self._cancel:
                self.signals.error.emit("Training canceled.")
                return

            # ---- Build model pipeline (same choices as your UI) ----
            if self.clf_type == "KNN":
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import KNeighborsClassifier
                model = make_pipeline(StandardScaler(),
                                      KNeighborsClassifier(n_neighbors=1, metric='cosine', weights='uniform'))
            elif self.clf_type == "SVM":
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC
                model = make_pipeline(StandardScaler(),
                                      SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovo'))
            elif self.clf_type == "LDA":
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model = LinearDiscriminantAnalysis(solver='svd')

            elif self.clf_type == "RDF":
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(
                    n_estimators=0, max_features=None, max_leaf_nodes=751266,
                    bootstrap=True, warm_start=True, n_jobs=-1
                )
                # incremental build like your script
                import math
                n_total = 30
                for i in range(1, n_total+1):
                    if self._cancel:
                        self.signals.error.emit("Training canceled.")
                        self.signals.finished.emit()
                        return
                    rf.set_params(n_estimators=i)
                    rf.fit(X_train, y_train)
                    self.signals.progress.emit(int(i*100/n_total))
                model = rf
            else:
                raise ValueError(f"Training not supported for '{self.clf_type}'.")

            if self._cancel:
                self.signals.error.emit("Training canceled.")
                return

            if self.clf_type != "RDF":
                self.signals.progress_indeterminate.emit(True)
                model.fit(X_aligned, y_train)
                self.signals.progress_indeterminate.emit(False)
                self.signals.progress.emit(100)

            if self._cancel:
                self.signals.error.emit("Training canceled.")
                return

            # Save a payload that mirrors what load_classifier() expects
            payload = {
                "pipeline": model,
                "train_wl": wl_used.astype(float)
            }

            os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
            joblib.dump(payload, self.savepath)

            self.signals.saved.emit(self.savepath)

        except Exception as e:
            import traceback
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

@dataclass
class ClassificationJob:
    name: str                 # unique key shown in table (e.g., "SVM (RBF)")
    clf_type: str             # "knn" | "cnn" | "svm" etc...
    kind: str                 # e.g., ["Substrate","Ink 3 classes" ...]
    status: str = "Queued"    # "Queued" | "Running" | "Done" | "Canceled" | "Error" | "To Train" | "Training..."
    trained = True
    trained_path = 'Default'
    progress: int = 0         # 0..100
    _t0: Optional[float] = field(default=None, repr=False)  # internal start time
    duration_s: Optional[float] = None    # whole classification duration
    class_map: Optional[np.ndarray] = None     # raw classification map
    rect=None                  # y, x, h, w of selected rectangle. None if no selection
    clean_map : Optional[np.ndarray] = None     # cleaned classification map
    clean_param= None           # clean parameters
    binary_algo= None           # binary algo
    binary_param = None         # binary param
    spectral_range_used = None  # to save
    substrate_name = 'Unknown Substrate'

    def reinit(self):
        if self.trained:
            self.status = "Queued"
            self.progress = 0
            self.duration_s = None
            self._t0 = None
            self.class_map: Optional[np.ndarray] = None

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

        # Async loading state
        self._load_dialog = None
        self._load_worker = {"VNIR": None, "SWIR": None}


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
        Load VNIR or SWIR cube asynchronously (non-blocking UI), with a LoadingDialog.
        """
        if self._load_dialog is not None:
            return

        start_dir = ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {kind} cube",
            start_dir,
            "Hypercube files (*.mat *.h5 *.hdr)"
        )
        if not path:
            return

        # UI: disable buttons to avoid concurrent loads
        self._set_load_buttons_enabled(False)
        self._load_dialog = LoadingDialog(
            message=f"Loading {kind} cube…",
            filename=path,
            parent=self,
            cancellable=False,
        )

        self._load_dialog.cancel_requested.connect(lambda: self._on_load_canceled(kind))

        self._load_dialog.show()

        def _do_load():
            cube = Hypercube()
            cube.open_hyp(default_path=path, open_dialog=False)
            return cube

        worker = LoadCubeWorker(load_fn=_do_load)
        self._load_worker[kind] = worker

        self._load_dialog.cancel_requested.connect(worker.cancel)

        worker.signals.finished.connect(lambda cube: self._on_load_result(kind, cube))
        worker.signals.error.connect(lambda err: self._on_load_error(kind, err))
        worker.signals.canceled.connect(lambda: self._on_load_canceled(kind))

        # Always finalize UI state
        worker.signals.finished.connect(lambda _: self._on_load_finished(kind))
        worker.signals.canceled.connect(lambda: self._on_load_finished(kind))
        worker.signals.error.connect(lambda _: self._on_load_finished(kind))

        QThreadPool.globalInstance().start(worker)

    def _set_load_buttons_enabled(self, enabled: bool):
        self.ui.pushButton_load_cube_1.setEnabled(enabled)
        self.ui.pushButton_load_cube_2.setEnabled(enabled)
        self.ui.pushButton_valid.setEnabled(enabled)

    def _on_load_result(self, kind: str, cube):
        # Save cube
        self.cubes[kind] = cube

        # WL can be None for some sources; tolerate that
        wl = getattr(cube, "wl", None)
        if isinstance(wl, np.ndarray) and wl.size > 0:
            self._wl_ranges[kind] = (float(wl.min()), float(wl.max()))
        else:
            self._wl_ranges[kind] = None

        # Update labels in the dialog
        self._update_labels()

    def _on_load_error(self, kind: str, err):
        # Robustness: clear state on failure
        self.cubes[kind] = None
        self._wl_ranges[kind] = None
        self._update_labels()

        msg = str(err) if err is not None else "Unknown error"
        QMessageBox.warning(self, "Load error", f"Failed to load {kind} cube.\n\n{msg}")

    def _on_load_canceled(self, kind: str):
        # If canceled before completion, keep previous state as-is.
        # (We do not forcibly clear cubes[kind] here.)
        pass

    def _on_load_finished(self, kind: str):
        # Close dialog and restore UI
        if self._load_dialog is not None:
            try:
                self._load_dialog.close()
            except Exception:
                pass
            self._load_dialog = None

        self._load_worker[kind] = None
        self._set_load_buttons_enabled(True)


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

class SaveClassMapDialog(QDialog):
    """
    Dialog to select save options for classification maps:
      - Save formats (HDF5 / PNG)
      - Models to include
      - Base filename
    """
    def __init__(self, model_names, default_base_name="result", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Classification Map(s)")
        self.setModal(True)

        self.chk_h5  = QCheckBox("HDF5 (.h5)")
        self.chk_png = QCheckBox("PNG (.png)")
        self.chk_h5.setChecked(True)
        self.chk_png.setChecked(True)

        self.radio_raw = QRadioButton("RAW map")
        self.radio_clean = QRadioButton("Cleaned map")
        self.radio_raw.setChecked(True)  # valeur par défaut

        row_maptype = QHBoxLayout()
        row_maptype.addWidget(self.radio_raw)
        row_maptype.addWidget(self.radio_clean)

        # Base name input
        self.base_name_edit = QLineEdit(default_base_name)

        # Model checkboxes in scrollable area
        self.model_checks = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        vbox_models = QVBoxLayout(inner)

        for name in model_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.model_checks.append(cb)
            vbox_models.addWidget(cb)

        scroll.setWidget(inner)

        # OK / Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        # Layout
        form = QFormLayout()
        form.addRow(QLabel("<b>Save formats</b>"))
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(self.chk_h5)
        row_fmt.addWidget(self.chk_png)
        form.addRow(row_fmt)
        form.addRow("Base name:", self.base_name_edit)

        form.addRow(QLabel("<b>Map to save</b>"))
        form.addRow(row_maptype)

        main = QVBoxLayout(self)
        main.addLayout(form)
        main.addWidget(QLabel("<b>Models</b>"))
        main.addWidget(scroll)
        main.addWidget(buttons)

        self._selected_models = []
        self._base_name = None
        self._want_h5 = True
        self._want_png = True
        self._want_clean = True

    def _on_accept(self):
        base = self.base_name_edit.text().strip()
        if not base:
            QMessageBox.warning(self, "Missing name", "Please enter a base name.")
            return

        selected = [cb.text() for cb in self.model_checks if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No model selected", "Please select at least one model.")
            return

        if not (self.chk_h5.isChecked() or self.chk_png.isChecked()):
            QMessageBox.warning(self, "No format selected", "Please select at least one format (HDF5 or PNG).")
            return

        self._selected_models = selected
        self._base_name = base
        self._want_h5 = self.chk_h5.isChecked()
        self._want_png = self.chk_png.isChecked()
        self._want_clean = self.radio_clean.isChecked()
        self.accept()

    @property
    def selected_models(self):
        return self._selected_models

    @property
    def base_name(self):
        return self._base_name

    @property
    def want_h5(self):
        return self._want_h5

    @property
    def want_png(self):
        return self._want_png

    @property
    def want_clean(self):
        return self._want_clean

def _ensure_unique_path(folder, filename_no_ext, ext):
    """Ensure unique filename by appending (n) if needed."""
    candidate = os.path.join(folder, f"{filename_no_ext}{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(folder, f"{filename_no_ext} ({n}){ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1

def _write_h5_class_map(path, class_map, metadata: dict):
    """Sauvegarde une carte de classification + métadonnées unifiées."""
    str_dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(path, "w") as f:
        # dataset principal
        f.create_dataset("class_map", data=np.asarray(class_map, dtype=np.int32), compression="gzip")

        # groupe unique pour toutes les métadonnées
        meta = f.require_group("Metadata")

        for key, val in (metadata or {}).items():
            try:
                if isinstance(val, str):
                    meta.attrs.create(key, val, dtype=str_dt)
                elif isinstance(val, np.ndarray) and val.dtype.kind in ("U", "O"):
                    meta.create_dataset(key, data=np.array(val, dtype=object), dtype=str_dt)
                elif isinstance(val, (list, tuple)):
                    if all(isinstance(x, str) for x in val):
                        meta.create_dataset(key, data=np.array(val, dtype=object), dtype=str_dt)
                    else:
                        meta.create_dataset(key, data=np.asarray(val))
                elif isinstance(val, (int, float, np.integer, np.floating, bool, np.bool_)):
                    meta.attrs[key] = val
                elif isinstance(val, np.ndarray):
                    meta.create_dataset(key, data=val)
                elif isinstance(val, dict):
                    sub = meta.require_group(key)
                    _write_dict_to_group(sub, val, str_dt)  # récursion
                else:
                    # fallback : texte
                    meta.attrs.create(key, str(val), dtype=str_dt)
            except Exception as e:
                s = str(val)
                meta.attrs.create(key, s, dtype=str_dt)

def _write_dict_to_group(group, d: dict, str_dt):
    for k, v in d.items():
        # même logique que ci-dessus, mais récursive
        if isinstance(v, dict):
            sub = group.require_group(k)
            _write_dict_to_group(sub, v, str_dt)
        elif isinstance(v, str):
            group.attrs.create(k, v, dtype=str_dt)
        else:
            group.create_dataset(k, data=v)

def _write_indexed_png(path, class_map, palette_rgb):
    """Write class_map as indexed PNG with palette."""
    if class_map.dtype != np.uint8:
        if class_map.max() > 255:
            raise ValueError("Class map has indices >255, cannot save as indexed PNG.")
        class_map = class_map.astype(np.uint8)

    img = Image.fromarray(class_map, mode="P")

    pal = np.zeros((256, 3), dtype=np.uint8)
    n = min(256, palette_rgb.shape[0])
    pal[:n, :] = palette_rgb[:n, :]
    img.putpalette(pal.reshape(-1).tolist())
    img.save(path, format="PNG")

CLEAN_PRESETS = {
    "Soft":      {"window_pct": 0, "iterations": 1, "min_area": 2},
    "Balanced":  {"window_pct": 2, "iterations": 10, "min_area": 5},
    "Strong":    {"window_pct": 5, "iterations": 15,  "min_area": 10},
}

class IdentificationWidget(QWidget, Ui_IdentificationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Queue structures
        self.job_order: List[str] = []  # only job names in execution order
        self.jobs: Dict[str, ClassificationJob] = {}  # name -> job
        self._cube_load_dialog = None
        self._cube_load_worker = None


        # Table init
        self._init_classification_table(self.tableWidget_classificationList)
        self._init_cleaning_list()

        # Classifiy as thread init
        self.threadpool = QThreadPool()
        self._running_idx: int = -1
        self._current_worker = None
        self._stop_all = False
        self.only_selected = False

        # Combo to select which model's result to show
        self.comboBox_clas_show_model.clear()
        self.comboBox_clas_show_model.currentIndexChanged.connect(self._on_show_model_changed)
        self._refresh_show_model_combo()  # start empty/disabled

        # Remplacer placeholders par ZoomableGraphicsView
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)
        self.viewer_left.enable_rect_selection = True
        self.viewer_right.enable_rect_selection = False # no rectangle selection for right view

        self._last_vnir = None
        self._last_swir = None
        self.cube = None
        self.data = None
        self.wl = None
        self.binary_map = None
        self.saved_rec= None
        self.binary_algo = None
        self.binary_param = None
        self.substrate_label= 'Unknown Substrate'
        self.alpha = self.horizontalSlider_overlay_transparency.value() / 100.0

        self.TRAIN_WL=np.arange(400, 1705, 5)
        self.whole_range = None

        # Connections
        self.pushButton_load.clicked.connect(self.open_load_cube_dialog)
        self.pushButton_reset.clicked.connect(self.reset_all)
        self.pushButton_launch_bin.clicked.connect(self.launch_binarization)
        self.pushButton_reset_bin.clicked.connect(self.reset_binary)
        self.horizontalSlider_overlay_transparency.valueChanged.connect(self.update_alpha)
        self.comboBox_bin_algorith_choice.currentIndexChanged.connect(self.update_bin_defaults)
        self.pushButton_clas_add_ink.clicked.connect(
            lambda: self.add_job(self.comboBox_clas_ink_model.currentText()))
        self.pushButton_clas_classify_substrate.clicked.connect(self.classify_substrate)
        self.pushButton_clas_remove.clicked.connect(self.remove_selected_job)
        self.pushButton_clas_remove_all.clicked.connect(lambda:self.remove_all_jobs())
        self.pushButton_clas_up.clicked.connect(self.move_selected_job_up)
        self.pushButton_clas_down.clicked.connect(self.move_selected_job_down)
        self.pushButton_clas_start.clicked.connect(self.start_queue)
        self.pushButton_clas_start_selected.clicked.connect(self.start_selected_job)
        self.pushButton_clas_stop.clicked.connect(self.stop_queue)
        self.pushButton_clas_reinit.clicked.connect(self.reinit_selected_job)
        self.pushButton_show_all.clicked.connect(self._show_all_results_dialog)
        self.pushButton_save_map.clicked.connect(self.on_click_save_map)
        self.radioButton_overlay_binary.toggled.connect(self.update_overlay)
        self.radioButton_overlay_identification.toggled.connect(self.update_overlay)
        self.pushButton_clean_start_selected.clicked.connect(self._on_click_clean_start_selected)
        self.pushButton_clean_start_all.clicked.connect(self._on_click_clean_start_all)
        self.pushButton_load_map.clicked.connect(self.load_map)

        self.radioButton_clean_show_raw.toggled.connect(self.update_overlay)
        self.radioButton_clean_show_cleaned.toggled.connect(self.update_overlay)
        self.radioButton_clean_show_both.toggled.connect(self.update_rgb_controls)
        self.radioButton_clean_show_both.toggled.connect(self.update_overlay)

        self.comboBox_clean_preset.currentIndexChanged.connect(self.apply_clean_preset)
        self.apply_clean_preset(self.comboBox_clean_preset.currentIndex())

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

        self.radioButton_band_selection_manual.toggled.connect(self.band_viz)
        self.radioButton_band_selection_auto.toggled.connect(self.band_viz)

        self.radioButton_rgb_default.setChecked(True)
        self.update_rgb_controls()

        # --- Boutons de rotation / flip ---
        self.pushButton_rotate.clicked.connect(lambda: self.transform(np.rot90))
        self.pushButton_flip_h.clicked.connect(lambda: self.transform(np.fliplr))
        self.pushButton_flip_v.clicked.connect(lambda: self.transform(np.flipud))
        self.radioButton_overlay_binary.toggled.connect(self.update_overlay)
        self.radioButton_overlay_identification.toggled.connect(self.update_overlay)

        self.update_bin_defaults()
        self.palette_bgr = {
            0: (128, 128, 128),  # Substrate (grey)
            3: (211, 0, 148),  # MGP (purpe)
            2: (0, 255, 255),  # CC (yellow)
            1: (0, 165, 255)  # NCC (orange)
        }

        self.labels={
            0: 'Substrate',
            3:  'MGP' ,
            2: 'CC',
            1: 'NCC'
        }

        self.tabWidget_Process.setCurrentIndex(0)
        self.tabWidget_info_classification.setCurrentIndex(0)
        self.tabWidget_Image.setCurrentIndex(1)

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
        self.binary_map = None
        self.saved_rec = None
        self.binary_algo = None
        self.binary_param = None
        self.substrate_label = 'Unknown Substrate'
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

    def band_viz(self):
        self.radioButton_grayscale.setChecked(True)
        self.tabWidget_Image.setCurrentIndex(0)
        if self.radioButton_band_selection_auto.isChecked():
            best_band=int(self.cube.get_best_band())
            wl_best=int(self.wl[best_band])
            self.horizontalSlider_blue_channel.setValue(wl_best)
        self.update_rgb_controls()

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
            if default :
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
            if len(self.wl)>7:
                sixieme=len(self.wl) // 6
                idx1,idx2,idx3=sixieme,3*sixieme,5*sixieme
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

            try:
                self.cube.normalize_spectral(self.TRAIN_WL, min_wl=400, max_wl=1700, interp_kind="linear", in_place=True)
            except ValueError as e:
                QMessageBox.warning(self, "Spectral error", str(e))
                return

            # Ensure UI buffers are in sync
            self.data = self.cube.data
            self.wl = self.cube.wl
            self.update_rgb_controls()
            self.update_overlay()

    def _apply_loaded_cube_to_ui(self, payload):
        """
        UI-thread: apply cube + refresh views.
        payload = (cube_out, loaded_cube, range)
        """
        cube_out, loaded_cube, cube_range = payload

        # Update last VNIR / SWIR if relevant
        if loaded_cube is not None:
            if cube_range == "VNIR":
                self._last_vnir = loaded_cube
            elif cube_range == "SWIR":
                self._last_swir = loaded_cube

        # Apply final cube
        self.cube = cube_out
        self.data = cube_out.data
        self.wl = cube_out.wl

        # Refresh UI
        self.update_rgb_controls()
        self.show_rgb_image()
        self.update_overlay()

    def _finish_cube_load_ui(self):
        """UI-thread: close dialog and release state."""
        if self._cube_load_dialog is not None:
            try:
                self._cube_load_dialog.close()
            except Exception:
                pass
            self._cube_load_dialog = None
        self._cube_load_worker = None

    def _start_cube_load_worker(self, message: str, filename: str, load_fn):
        """Start LoadCubeWorker with a non-cancellable LoadingDialog (like unmixing)."""
        # Prevent concurrent loads
        if self._cube_load_dialog is not None:
            return

        self._cube_load_dialog = LoadingDialog(
            message=message,
            filename=filename,
            parent=self,
            cancellable=False,  # IMPORTANT: so close() always works
        )
        self._cube_load_dialog.show()

        worker = LoadCubeWorker(load_fn=load_fn)
        self._cube_load_worker = worker

        worker.signals.finished.connect(self._apply_loaded_cube_to_ui)
        worker.signals.error.connect(lambda err: QMessageBox.warning(self, "Error", f"Failed to load cube:\n\n{err}"))
        worker.signals.finished.connect(lambda _: self._finish_cube_load_ui())
        worker.signals.error.connect(lambda _: self._finish_cube_load_ui())
        worker.signals.canceled.connect(lambda: self._finish_cube_load_ui())

        # Use existing pool if you want; otherwise globalInstance is fine too
        self.threadpool.start(worker)

    def load_cube(self, filepath=None, cube=None, cube_info=None, range=None):
        if self.no_reset_jobs_on_new_cube():
            return

        # Resolve filepath if needed (UI thread)
        if cube is None and not filepath:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not filepath:
                return

        # Snapshot inputs for the worker
        vnir_in = None
        swir_in = None
        single_cube_in = None
        filepath_in = filepath
        range_in = range
        cube_info_in = cube_info

        # Decide inputs
        if cube is not None:
            # Cube already provided
            if range_in is None:
                single_cube_in = cube
            elif range_in == "VNIR":
                vnir_in = cube
                swir_in = self._last_swir
            elif range_in == "SWIR":
                swir_in = cube
                vnir_in = self._last_vnir
            else:
                QMessageBox.warning(self, "Error", "Problem with cube range parameter.")
                return
        else:
            # Need to load from filepath
            pass

        def _do_load_build():
            loaded_cube = None

            # 1) Load cube if needed
            if cube is None:
                loaded_cube = Hypercube(filepath=filepath_in, load_init=True)
                if cube_info_in is not None:
                    loaded_cube.metadata = loaded_cube.cube_info.metadata_temp

                if range_in is None:
                    _single = loaded_cube
                    _vnir = None
                    _swir = None
                elif range_in == "VNIR":
                    _vnir = loaded_cube
                    _swir = self._last_swir
                    _single = None
                elif range_in == "SWIR":
                    _swir = loaded_cube
                    _vnir = self._last_vnir
                    _single = None
                else:
                    raise ValueError("Problem with cube range parameter.")
            else:
                # Cube already provided
                if range_in is None:
                    _single = cube
                    _vnir = None
                    _swir = None
                elif range_in == "VNIR":
                    _vnir = cube
                    _swir = self._last_swir
                    _single = None
                elif range_in == "SWIR":
                    _swir = cube
                    _vnir = self._last_vnir
                    _single = None
                else:
                    raise ValueError("Problem with cube range parameter.")

            # 2) Fuse (or passthrough)
            if _single is not None:
                out = _single
            else:
                out = fused_cube(_vnir, _swir)

            # 3) Normalize spectral axis
            out.normalize_spectral(
                self.TRAIN_WL,
                min_wl=400,
                max_wl=1700,
                interp_kind="linear",
                in_place=True
            )

            return out, loaded_cube, range_in

        # UI: update last VNIR/SWIR references *before* worker if cube provided;
        # if loaded from file, we'll update them after load by re-calling load_cube(range=...) if needed.
        # Here, keep it simple: update only for provided cubes; for filepath loads, update in a follow-up if you want.
        if cube is not None:
            if range_in == "VNIR":
                self._last_vnir = cube
            elif range_in == "SWIR":
                self._last_swir = cube

        # Start worker + dialog (like unmixing)
        label = "Loading cube…"
        fn = filepath_in if filepath_in else (range_in or "cube")
        if range_in in ("VNIR", "SWIR"):
            label = f"Loading {range_in} cube…"

        self._start_cube_load_worker(message=label, filename=str(fn), load_fn=_do_load_build)

    def show_rgb_image(self):

        if self.radioButton_clean_show_both.isChecked() and self.radioButton_overlay_identification.isChecked() :
            return

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

    def reset_binary(self):
        try:
            self.binary_map=np.ones((self.data.shape[0],self.data.shape[1]))
            self.show_binary_result()
        except:
            pass

    def launch_binarization(self):
        if self.cube is None:
            QMessageBox.warning(self, "Warning", "Load a cube first.")
            return
        algorithm = self.comboBox_bin_algorith_choice.currentText()
        param = {
            'k': self.doubleSpinBox_bin_k.value(),
            'window': self.spinBox_bin_window_size.value(),
            'padding': self.comboBox_padding_mode.currentText()
        }

        if self.radioButton_band_selection_manual.isChecked():
            val_slid=self.horizontalSlider_blue_channel.value()
            idx = np.argmin(np.abs(val_slid - self.wl))
        else:
            idx=None

        try:
            self.saved_rec = None
            self.binary_map = self.cube.get_binary_from_best_band(algorithm, param, idx)
            rect = self._get_selected_rect()
            if rect is not None:
                y, x, h, w = rect
                self.saved_rec = rect

            ### Uncomment if want to binarize only selected zone

            # rect = self._get_selected_rect()
            # if rect is None:
            #     self.saved_rec=None
            #     self.binary_map = self.cube.get_binary_from_best_band(algorithm, param,idx)
            # else:
            #     y, x, h, w = rect
            #     self.saved_rec = rect
            #     sub_data = self.data[y:y + h, x:x + w, :]
            #     # -> on a besoin d'un petit helper côté Hypercube si tu n'en as pas déjà
            #     #    (ici on réutilise "best_band" mais en local)
            #     sub_cube = Hypercube(data=sub_data, wl=self.wl, cube_info=self.cube.cube_info)
            #     sub_binary = sub_cube.get_binary_from_best_band(algorithm, param,idx)
            #
            #     # Recompose une carte binaire pleine taille, remplie de 0 ailleurs
            #     full_bin = np.zeros(self.data.shape[:2], dtype=sub_binary.dtype)
            #     full_bin[y:y + h, x:x + w] = sub_binary
            #     self.binary_map = full_bin

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Binarization failed: {e}")
            return

        self.binary_param=param
        self.binary_algo=algorithm
        self.radioButton_overlay_binary.setChecked(True)
        self.show_binary_result()
        self.viewer_right.fitImage()
        self._refresh_clean_sources_list()

    def show_binary_result(self,only_images=False):
        if self.binary_map is None or not hasattr(self, "rgb_image"):
            return

        # Palette binaire (BGR) :
        bin_colors = {1: (0, 0, 0), 0: (255, 255, 255)}

        # Image couleur (BGR) de la carte binaire
        mask = (self.binary_map.astype(np.uint8) > 0)
        h, w = mask.shape
        result_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        result_bgr[~mask] = bin_colors[0]
        result_bgr[mask] = bin_colors[1]

        # Affichage de la carte à droite
        self.viewer_right.setImage(self._np2pixmap(result_bgr))
        self._draw_current_rect(surface=False)

        if not hasattr(self, "rgb_image") or self.rgb_image is None:
            self.show_rgb_image()

        # If sizes mismatch (e.g., new cube), rebuild or resize
        if self.rgb_image is None or self.rgb_image.ndim != 3 or self.rgb_image.shape[:2] != result_bgr.shape[:2]:
            # Rebuild from current data; if still mismatch, resize the overlay map
            self.show_rgb_image()
            if self.rgb_image is None or self.rgb_image.shape[:2] != result_bgr.shape[:2]:
                result_bgr = cv2.resize(result_bgr, (self.rgb_image.shape[1], self.rgb_image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

        # Alpha default (in case update_alpha wasn't called yet)
        if not hasattr(self, "alpha") or self.alpha is None:
            self.alpha = 0.5

        # Overlay sur l’image RGB avec alpha (comme la classification)
        overlay = cv2.addWeighted(self.rgb_image, 1 - self.alpha, result_bgr, self.alpha, 0)
        self.viewer_left.setImage(self._np2pixmap(overlay))
        self._draw_current_rect(surface=False)

        if only_images:
            return

        self.update_legend()
        self._set_info_rows()

    def update_overlay(self,only_images=False):
        if self.radioButton_overlay_binary.isChecked():
            self.show_binary_result(only_images)
        elif self.radioButton_overlay_identification.isChecked():
            self.show_classification_result(only_images)

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.update_overlay(only_images=True)

    def _show_all_results_dialog(self):
        # Récupère les jobs ayant une carte brute
        jobs_with_maps = [(name, job) for name, job in self.jobs.items() if getattr(job, "class_map", None) is not None]
        if not jobs_with_maps:
            QMessageBox.information(self, "No results", "Aucun résultat de classification disponible pour l’instant.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("All classification results")
        dlg.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dlg.setSizeGripEnabled(True)
        avail = QGuiApplication.primaryScreen().availableGeometry()
        dlg.resize(int(avail.width() * 0.6), int(avail.height() * 0.6))

        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(12)

        # Affichera-t-on une colonne "Cleaned Map" ?
        any_clean = any(getattr(job, "clean_map", None) is not None for _, job in jobs_with_maps)

        # --- HEADER -------------------------------------------------------------
        header = QWidget()
        hh = QHBoxLayout(header)
        hh.setContentsMargins(8, 0, 8, 0)
        hh.setSpacing(16)

        # Largeur de la colonne des titres (identique à celle des lignes)
        first_col_min_w = 220

        title_hdr = QLabel("<b>Job</b>")
        title_hdr.setTextFormat(Qt.RichText)
        title_hdr.setMinimumWidth(first_col_min_w)
        title_hdr.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_hdr.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        hh.addWidget(title_hdr, 0)

        raw_hdr = QLabel("<b>Raw Map</b>")
        raw_hdr.setTextFormat(Qt.RichText)
        raw_hdr.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        raw_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        hh.addWidget(raw_hdr, 1)

        if any_clean:
            clean_hdr = QLabel("<b>Cleaned Map</b>")
            clean_hdr.setTextFormat(Qt.RichText)
            clean_hdr.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            clean_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            hh.addWidget(clean_hdr, 1)

        vbox.addWidget(header)

        # --- Helper pour coloriser une carte d’indices en RGB -------------------
        def colorize(cm: np.ndarray) -> np.ndarray:
            h, w = cm.shape
            out = np.zeros((h, w, 3), dtype=np.uint8)
            for val, bgr in self.palette_bgr.items():
                out[cm == val] = bgr
            return out

        max_w, max_h = 200, 200

        # --- LIGNES -------------------------------------------------------------
        for name, job in jobs_with_maps:
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(8, 8, 8, 8)
            hl.setSpacing(16)

            # Colonne gauche : titre + méta (aligné à gauche)
            title = QLabel(
                f"<b>{name}</b><br><span style='color:gray'>"
                f"{getattr(job, 'kind', '')}"
                f"</span>"
            )
            title.setTextFormat(Qt.RichText)
            title.setMinimumWidth(first_col_min_w)
            title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            title.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            hl.addWidget(title, 0, Qt.AlignLeft | Qt.AlignTop)

            # Colonne Raw Map : centré horizontalement
            cm_raw = job.class_map
            pix_raw = self._np2pixmap(colorize(cm_raw)).scaled(
                max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            raw_label = QLabel()
            raw_label.setPixmap(pix_raw)
            raw_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            raw_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            hl.addWidget(raw_label, 1, Qt.AlignHCenter | Qt.AlignTop)

            # Colonne Cleaned Map (si dispo) : centré horizontalement
            if getattr(job, "clean_map", None) is not None:
                cm_clean = job.clean_map
                pix_clean = self._np2pixmap(colorize(cm_clean)).scaled(
                    max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                clean_label = QLabel()
                clean_label.setPixmap(pix_clean)
                clean_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
                clean_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                hl.addWidget(clean_label, 1, Qt.AlignHCenter | Qt.AlignTop)
            else:
                if any_clean:
                    # Cellule vide pour garder l’alignement des colonnes
                    spacer = QWidget()
                    spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                    hl.addWidget(spacer, 1)

            vbox.addWidget(row)

            # Séparateur doux entre les lignes
            sep = QLabel("<hr>")
            sep.setTextFormat(Qt.RichText)
            vbox.addWidget(sep)

        scroll.setWidget(container)
        main = QVBoxLayout(dlg)
        main.addWidget(scroll)
        dlg.setLayout(main)
        dlg.exec_()

    def update_bin_defaults(self):
        algo = self.comboBox_bin_algorith_choice.currentText().lower()

        try:
            if algo == "niblack":
                self.doubleSpinBox_bin_k.setRange(-1.0, 1.0)
                self.doubleSpinBox_bin_k.setValue(-0.2)
                self.spinBox_bin_window_size.setValue(3)
                self.comboBox_padding_mode.setCurrentText("replicate")

            elif algo == "otsu":
                self.doubleSpinBox_bin_k.setRange(0.0, 0.0)  # k inutile
                self.doubleSpinBox_bin_k.setValue(0.0)
                self.spinBox_bin_window_size.setValue(3)  # pas utilisé mais on fixe
                self.comboBox_padding_mode.setCurrentText("replicate")

            elif algo == "sauvola":
                self.doubleSpinBox_bin_k.setRange(0.1, 0.9)
                self.doubleSpinBox_bin_k.setValue(0.4)
                self.spinBox_bin_window_size.setValue(3)
                self.comboBox_padding_mode.setCurrentText("replicate")

            elif algo == "wolf":
                self.doubleSpinBox_bin_k.setRange(0.0, 1.0)
                self.doubleSpinBox_bin_k.setValue(0.5)
                self.spinBox_bin_window_size.setValue(3)
                self.comboBox_padding_mode.setCurrentText("reflect")

            elif algo == "bradley":
                self.doubleSpinBox_bin_k.setRange(0.0, 1.0)  # cohérent avec ton code
                self.doubleSpinBox_bin_k.setValue(0.1)
                self.spinBox_bin_window_size.setValue(15)
                self.comboBox_padding_mode.setCurrentText("replicate")

        except Exception as e:
            print(f"[update_bin_defaults] Error : {e}")

    def _np2pixmap(self, img):
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

    def load_classifier(self,path):

        if getattr(sys, 'frozen', False):  # pynstaller case
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        models_dir =  os.path.join(BASE_DIR,"identification/data")

        if "CNN 1D" in path:
            import torch
            from classifier_train import SpectralCNN1D
            input_length = self.data.shape[2]
            num_classes = 3
            self.classifier = SpectralCNN1D(input_length, num_classes)
            self.classifier.load_state_dict(torch.load(os.path.join(models_dir, "model_CNN1D.pth"), map_location="cpu"))

            self.classifier.eval()
            self.classifier_type = "cnn"

        else:
            import joblib

            model=joblib.load(path)
            self.classifier = model['pipeline']
            self.classifier_type = "sklearn"
            self.classifier_wl=model['train_wl']

    def _load_substrate_model(self, model_name, trained_path=None):
        if trained_path:
            self.load_classifier(trained_path)
            return

        if model_name == "Add from disk.":
            path, _ = QFileDialog.getOpenFileName(
                self, "Load substrate model",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "identification/data"),
                "joblib files (*.joblib)"
            )
            if not path:
                raise RuntimeError("No model selected.")
            import joblib
            payload = joblib.load(path)
            self.classifier = payload["pipeline"]
            self.classifier_type = "sklearn"
            self.classifier_wl = payload.get("train_wl", None)
            return

        # Built-in models
        if getattr(sys, 'frozen', False):
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        filename = f"model_{model_name.lower()}_background_10pct.joblib"
        self.load_classifier(os.path.join(BASE_DIR, "identification", "data", filename))

    def _open_busy_dialog(self, message: str):
        if getattr(self, "_busy_dialog", None) is not None:
            return

        self._busy_dialog = LoadingDialog(
            message=message,
            parent=self,
            cancellable=False,  # IMPORTANT : comme unmixing
        )
        self._busy_dialog.show()

    def _close_busy_dialog(self):
        dlg = getattr(self, "_busy_dialog", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
            self._busy_dialog = None

    def classify_substrate(self,trained_path=None):
        model_name=self.comboBox_clas_substrate_model.currentText()
        rect = self._get_selected_rect()

        SUBSTRATE_LABEL_DIC={11 : 'Parchment',
                             12 : 'Cotton-Linen',
                             13 : 'Linen',
                             14 : 'Paper',
                             15 : 'Translucent paper'
        }

        if trained_path:
            try:
                self.load_classifier(trained_path)
            except Exception as e:
                QMessageBox.critical(self, "Model error", f"Could not load substrate model:\n{e}")
                return

        if self.binary_map is None:
            QMessageBox.warning(self, "No binarization map",
                                "Launch binarization first")
            return

        if rect is None:

            sub_map=self.binary_map
            sub_data=self.data

        else:
            y, x, h, w = rect
            self.saved_rec = rect
            sub_data = self.data[y:y + h, x:x + w, :]
            sub_map=self.binary_map[y:y + h, x:x + w]

        mask_sub = (sub_map == 0)
        if mask_sub.sum() == 0:
            QMessageBox.information(self, "No substrate", "No substrate pixels found in this region.")
            return
        X = sub_data[mask_sub, :]  # (N, B)

        try:
            self._load_substrate_model(model_name, trained_path=trained_path)
        except Exception as e:
            QMessageBox.critical(self, "Model error", f"Could not load substrate model:\n{e}")
            return

        # Check wavelength compatibility & propose training if needed ---
        ok, msg = self.features_compatible(getattr(self, "classifier_wl", None))
        if not ok:
            reply = QMessageBox.question(
                self, "Train substrate model?",
                f"{msg}\n\nTrain a new *background (10%)* model for the current spectral range?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

            if model_name not in ['KNN', 'RDF', 'LDA', 'SVM']:
                QMessageBox.warning(self, 'Model can be trained',
                                    'Model can not be trained.\nPlease choose between LDA, KNN, RDF or SVM')
                return

            # Build a filename that embeds the spectral range
            spectral_tag = f"_{int(self.wl[0])}-{int(self.wl[-1])}"
            if getattr(sys, 'frozen', False):
                BASE_DIR = sys._MEIPASS
            else:
                BASE_DIR = os.path.dirname(os.path.dirname(__file__))

            out_dir = os.path.join(BASE_DIR, "identification", "data")
            os.makedirs(out_dir, exist_ok=True)
            savepath = os.path.join(out_dir, f"model_{model_name.lower()}_background_10pct{spectral_tag}.joblib")

            # Launch TrainWorker (same as ink path)
            worker = TrainWorker(
                clf_type=model_name,
                kind="background_10pct",
                savepath=savepath,
                train_wl=self.wl
            )

            self._open_busy_dialog("Training substrate model…")

            def _on_saved(path):
                try:
                    # reload the freshly trained model
                    print('[SUBSTRATE IDENTIFICATION] : Model trained and saved at : ', path)
                    self.classify_substrate(trained_path=path)
                except Exception as e:
                    QMessageBox.critical(self, "Training", f"Model trained but failed to reload:\n{e}")

            worker.signals.saved.connect(_on_saved)
            worker.signals.error.connect(lambda msg: QMessageBox.critical(self, "Training error", msg))

            # Progress handling
            worker.signals.progress.connect(self._on_classif_progress)  # réutilise ta fonction existante

            worker.signals.finished.connect(self._close_busy_dialog)
            worker.signals.error.connect(lambda _: self._close_busy_dialog())

            self.threadpool.start(worker)
            return

        if self.checkBox_classify_substrate_fast.isChecked():
            '''Make the Mid spectra of all pixels of sub_data where sub_map is 0 and classify with loaded model'''
            self._open_busy_dialog("Fast substrate identification…")
            x_mean = X.mean(axis=0, keepdims=True)
            pred = int(self.classifier.predict(x_mean)[0])
            self.substrate_label = SUBSTRATE_LABEL_DIC[pred]
            QMessageBox.information(self, "Substrate (fast)", f'Substrate result of classification : {self.substrate_label}')
        else :
            '''Make the classification of all sub_data where sub_map is 0 and get %of each label classified'''
            try:
                # détermine un chunk raisonnable
                N = X.shape[0]
                target_steps = 50
                min_chunk, max_chunk = 20, 200000
                chunk_size = max(1, min(max_chunk, max(min_chunk, N // target_steps)))

                worker = ClassifyWorker(
                    X, self.classifier, self.classifier_type,
                    chunk_size=chunk_size, emit_partial=False
                )

                def _on_err(msg):
                    QMessageBox.critical(self, "Substrate classification", msg)

                def _on_res(preds: np.ndarray):
                    # stats
                    if preds.size == 0:
                        s = "No predictions."
                    else:
                        import numpy as _np
                        classes = _np.unique(preds)
                        counts = _np.bincount(preds, minlength=classes.max() + 1)
                        tot = preds.size
                        lines = []
                        for c in classes:
                            pct = 100.0 * counts[c] / tot
                            substrate_label = SUBSTRATE_LABEL_DIC[c]
                            lines.append(f"{substrate_label} : {counts[c]} px  ({pct:.1f}%)")
                        s = "\n".join(lines)
                        counts = np.bincount(preds)
                        tot = counts.sum()
                        pred = int(np.argmax(counts))

                        self.substrate_label=SUBSTRATE_LABEL_DIC[pred]

                    QMessageBox.information(self, "Substrate (full)", s)
                    self.update_overlay()

                worker.signals.progress.connect(self._on_classif_progress)
                worker.signals.error.connect(_on_err)
                worker.signals.result.connect(_on_res)

                worker.signals.finished.connect(self._close_busy_dialog)
                worker.signals.error.connect(lambda _: self._close_busy_dialog())

                self._open_busy_dialog("Classifying substrate…")

                self.threadpool.start(worker)


            except Exception as e:
                QMessageBox.critical(self, "Substrate classification", f"Unexpected error:\n{e}")

        self._close_busy_dialog()

        for key in self.jobs:
            self.jobs[key].substrate_name=self.substrate_label
        self.update_overlay()

    def _on_classif_progress(self, value):
        try:
            self.progressBar_classif.setValue(int(value))
        except Exception:
            pass

    def _on_classif_error(self, message):
        QMessageBox.critical(self, "Classification error", message)

    def _on_classif_result(self, preds, mask):
        # Construire la carte de classes sans bloquer le thread GUI trop longtemps
        class_map = np.zeros_like(self.binary_map, dtype=np.uint8)
        class_map[mask] = preds + 1  # 1..N classes, 0 = substrat
        self.class_map = class_map
        self.radioButton_overlay_identification.setChecked(True)
        self.show_classification_result()
        self._refresh_clean_sources_list()

    def _on_classif_finished(self):
        try:
            self.progressBar_classif.setVisible(False)
        except Exception:
            pass
        self.pushButton_clas_ink_launch.setEnabled(True)
        self._current_worker = None

    def update_legend(self):
        from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget, QHBoxLayout
        from PyQt5.QtGui import QPixmap, QColor

        # 1) Obtenir/Créer le layout
        layout = self.frame_legend.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.frame_legend.setLayout(layout)
        else:
            # 2) Vider proprement le layout existant
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

        # 3) Choisir la palette/labels selon le mode
        if self.radioButton_overlay_binary.isChecked():
            colors = {1: (0, 0, 0), 0: (255, 255, 255)}
            if self.substrate_label is None:
                sub_name='Substrate'
            else:
                sub_name=self.substrate_label
            labels = {0: sub_name, 1: 'Ink'}
        else:
            colors = self.palette_bgr
            labels = self.labels

        # 4) Construire les items de légende
        for key in sorted(labels.keys()):
            bgr = colors[key]
            rgb = self.bgr_to_rgb(bgr)

            swatch = QPixmap(20, 20)
            swatch.fill(QColor(*rgb))
            icon = QLabel()
            icon.setPixmap(swatch)

            text = QLabel(labels[key])
            text.setStyleSheet("font-weight: bold;")

            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(2, 2, 2, 2)
            h.addWidget(icon)
            h.addWidget(text)

            layout.addWidget(row)

        # 5) Pousser le contenu vers le haut
        layout.addStretch(1)

    def show_classification_result(self,only_images=False):

        def _colorize(cm: np.ndarray) -> np.ndarray:
            """class_map to RGB using self.palette_bgr."""
            colors = self.palette_bgr
            h, w = cm.shape
            result_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for val, bgr in colors.items():
                result_rgb[cm == val] = bgr
            return result_rgb

        # Récupère le job actuellement sélectionné (ou abandonne s'il n'y en a pas)
        job = getattr(self, "_current_job", None)
        job = job() if callable(job) else self._current_job()
        if not job or job.class_map is None:
            self.update_legend()
            self._set_info_rows()
            if self.data is not None:
                cm_clean= np.zeros((self.data.shape[0],self.data.shape[1]))
            else:
                cm_clean=np.zeros((1,1))
            rgb_right = _colorize(cm_clean)
            self.viewer_right.setImage(self._np2pixmap(rgb_right))
            self.label_viewer_right.setText("RAW map ")
            return

        show_raw = self.radioButton_clean_show_raw.isChecked()
        show_clean =self.radioButton_clean_show_cleaned.isChecked()
        show_both = self.radioButton_clean_show_both.isChecked()

        cm_raw = job.class_map
        if getattr(job, "clean_map", None) is not None:
            cm_clean = job.clean_map
            clean_map_exist=True
        else :
            cm_clean =job.class_map
            clean_map_exist = False

        if show_both:
            # NO Overlay
            rgb_left = _colorize(cm_raw)
            rgb_right = _colorize(cm_clean)
            self.viewer_left.setImage(self._np2pixmap(rgb_left))
            self.viewer_right.setImage(self._np2pixmap(rgb_right))
            # Dessine le rectangle associé au job (contour fin)
            self._draw_current_rect(use_job=True, surface=False)
            self.label_viewer_left.setText("RAW map ")
            if clean_map_exist:
                self.label_viewer_right.setText("CLEANED map ")
                self.label_viewer_right.setStyleSheet("color: black;")

            else:
                self.label_viewer_right.setText("RAW map (No cleaning done yet) ")
                self.label_viewer_right.setStyleSheet("color: red;")

        else:

            self.label_viewer_left.setText("False RGB")

            if show_clean:
                cm = cm_clean
                if clean_map_exist:
                    self.label_viewer_right.setText("CLEANED map ")
                    self.label_viewer_right.setStyleSheet("color: black;")

                else:
                    self.label_viewer_right.setText("RAW map (No cleaning done yet) ")
                    self.label_viewer_right.setStyleSheet("color: red;")

            else:
                cm=cm_raw
                self.label_viewer_right.setText("RAW map ")
                self.label_viewer_right.setStyleSheet("color: black;")

            rgb_map = _colorize(cm)

            self.viewer_right.setImage(self._np2pixmap(rgb_map))

            if hasattr(self, "rgb_image") and self.rgb_image is not None:
                overlay = cv2.addWeighted(self.rgb_image, 1 - self.alpha, rgb_map, self.alpha, 0)
                self.viewer_left.setImage(self._np2pixmap(overlay))
            else:
                self.viewer_left.setImage(self._np2pixmap(rgb_map))

            self._draw_current_rect(use_job=True, surface=False)


        if only_images:
            return

        # Légende + infos (inchangés)
        self.labels[0] = job.substrate_name
        self.update_legend()
        self._set_info_rows()

    def _refresh_show_model_combo(self, select_name: str = None):
        """
        Rebuild comboBox_clas_show_model with ONLY jobs that have a class_map ready.
        If select_name is provided and present, selects it.
        Disables the combo if empty.
        """
        cb = self.comboBox_clas_show_model
        # Bloque les signaux pour éviter un refresh->signal->refresh en boucle
        was_blocked = cb.blockSignals(True)
        cb.clear()

        # Remplir avec les jobs qui ont déjà un résultat
        available = [name for name in self.job_order if (name in self.jobs and self.jobs[name].class_map is not None)]
        for name in available:
            cb.addItem(name, name)  # text=name, userData=name

        # Sélection
        if select_name and select_name in available:
            cb.setCurrentIndex(available.index(select_name))
        elif cb.count() > 0:
            cb.setCurrentIndex(cb.count() - 1)  # dernier calculé par défaut

        # Activer/désactiver
        cb.setEnabled(cb.count() > 0)

        cb.blockSignals(was_blocked)

    def _on_show_model_changed(self, idx: int):
        """Apply the selected model's class_map to the viewers."""
        if idx < 0:
            return
        cb = self.comboBox_clas_show_model
        name = cb.itemData(idx, Qt.UserRole)
        if not name:
            name = cb.currentText()
        job = self.jobs.get(name)
        if not job or job.class_map is None:
            return

        # Affiche la carte du job sélectionné
        self.class_map = job.class_map
        self.radioButton_overlay_identification.setChecked(True)
        self.show_classification_result()

    def _init_classification_table(self,tw):
        tw.setColumnCount(5)
        tw.setHorizontalHeaderLabels(["Model", "Kind", "Status", "Progress", "Duration"])
        tw.setSelectionBehavior(tw.SelectRows)
        tw.setSelectionMode(tw.SingleSelection)
        tw.setEditTriggers(tw.NoEditTriggers)
        tw.verticalHeader().setVisible(False)
        header = tw.horizontalHeader()
        self.tableWidget_classificationList.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Model
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Kind
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Status stays auto-sized
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # Progress stays auto-sized
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Duration

    def _make_progress_bar(self, value: int = 0) -> QProgressBar:
        pb = QProgressBar()
        pb.setRange(0, 100)
        pb.setValue(int(value))
        pb.setTextVisible(True)
        pb.setAlignment(Qt.AlignCenter)
        return pb

    def _refresh_table(self):
        """Rebuild rows from job_order."""
        tw = self.tableWidget_classificationList
        tw.setRowCount(0)
        for row, name in enumerate(self.job_order):
            job = self.jobs[name]
            tw.insertRow(row)
            # Model
            it_model = QTableWidgetItem(job.name)
            it_model.setToolTip(job.name)
            tw.setItem(row, 0, it_model)
            # Kind
            it_kind = QTableWidgetItem(job.kind)
            it_kind.setToolTip(job.kind)
            tw.setItem(row, 1, it_kind)
            # Status
            it_status = QTableWidgetItem(job.status)
            tw.setItem(row, 2, it_status)
            # Progress
            pb = self._make_progress_bar(job.progress)
            tw.setCellWidget(row, 3, pb)
            # Duration
            it_dur = QTableWidgetItem("" if job.duration_s is None else f"{job.duration_s:.1f} s")
            it_dur.setTextAlignment(Qt.AlignCenter)
            tw.setItem(row, 4, it_dur)

    def _selected_row(self) -> int:
        sm = self.tableWidget_classificationList.selectionModel()
        if not sm: return -1
        rows = sm.selectedRows()
        return rows[0].row() if rows else -1

    def _row_to_name(self, row: int) -> Optional[str]:
        if 0 <= row < len(self.job_order):
            return self.job_order[row]
        return None

    def _name_to_row(self, name: str) -> Optional[int]:
        try:
            return self.job_order.index(name)
        except ValueError:
            return None

    def _ensure_unique_name(self, base: str) -> str:

        if base in self.jobs:
            choice = QMessageBox.question(
                self, "Already added",
                "Model already loaded in queue.\nDo you want to add it again ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if choice == QMessageBox.No:
                return None

        name = base
        i = 2
        while name in self.jobs:
            name = f"{base} ({i})"
            i += 1
        return name

    def _train_new_model(self,name):

        match name:
            case 'LDA':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model=LinearDiscriminantAnalysis(solver='svd')
                pass
            case 'KNN':
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import KNeighborsClassifier
                model = make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(
                        n_neighbors=1,
                        metric='cosine',
                        weights='uniform'
                    )
                )
                pass

            case 'SVM':
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC

                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel='rbf',  # Gaussian kernel
                        C=10,  # Box constraint
                        gamma='scale',  # Automatic kernel scale
                        decision_function_shape='ovo'
                    )
                )
                pass

            case 'RDF':
                from sklearn.ensemble import RandomForestClassifier
                n_trees_total = 30
                model = RandomForestClassifier(
                    n_estimators=0,  # Start with 0 tree
                    max_features=None,  # Use all predictors
                    max_leaf_nodes=751266,  # Max number of splits
                    bootstrap=True,
                    warm_start=True,  # Allow incremental training
                    n_jobs=-1,
                )
                pass

            case _:
                QMessageBox.warning(self,'Model can be trained','Model can not be trained.\nPlease choose between LDA, KNN, RDF or SVM')
                return

        return

    def add_job(self, name: str):

        trained=False

        if 'SUB_' in name:
            kind='Substrate'
        else:
            kind='Ink 3 classes'

        clf_type=name
        if getattr(sys, 'frozen', False):  # pynstaller case
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        save_model_folder = os.path.join(BASE_DIR,
                                         "identification/data")

        filename = f"model_{clf_type.lower()}.joblib"
        savepath=os.path.join(save_model_folder, filename)

        if clf_type=="Add from disk...":
            ## dialog to open file
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Load model",
                os.path.join(BASE_DIR,"identification/data"),
                "joblib files (*.joblib)"
            )
            if not path:
                return

            ## check if same number of features with message
            import joblib
            model=joblib.load(path)
            train_wl=model['train_wl']

            print('[Add from disk...] len wl :',len(train_wl))

            ## validate and ask a name (and kind ?)
            compatible, _ = self.features_compatible(train_wl)
            print('[Add from disk...] compatible :', compatible)
            if compatible:
                try:
                    clf_type_temp=model['pipeline'].steps[1][0]
                    print('kind from pipeline : ',clf_type_temp)
                except:
                    try:
                        clf_type_temp = model['pipeline'].__class__.__name__
                        print('kind from raw : ', clf_type_temp)

                    except:
                        clf_type_temp='Unkown'
                        print('kind not found')
                try:
                    clf_type={'kneighborsclassifier':'KNN','RandomForestClassifier':'RDF','LinearDiscriminantAnalysis':'LDA','svc':'SVM'}[clf_type_temp]
                except:
                    clf_type=clf_type_temp

                model_name, ok = QInputDialog.getText(self, 'Classifier loaded', f'Classifier {clf_type} loaded. Enter name :',text=clf_type)
                if ok :
                    if model_name:
                        print("User entered:", model_name)
                    else:
                        model_name='USERS'
                        print("User No entered name. Default name:", model_name)

                    name = model_name
                    savepath=path
                    trained=True

        if not trained :
            if len(self.wl) != len(self.TRAIN_WL):
                trained=False
                reply=QMessageBox.question(self,
                                     'Train new ?',
                                     'Spectral range smaller than pretrained model. \nDo you want to train model first ?',
                                     QMessageBox.Yes | QMessageBox.No)
                if reply==QMessageBox.No:
                    return
                else:
                    if name not in ['KNN','RDF','LDA','SVM']:
                        QMessageBox.warning(self, 'Model can be trained',
                                            'Model can not be trained.\nPlease choose between LDA, KNN, RDF or SVM')
                        return

                    spectral_range_name=f'_{int(self.wl[0])}-{int(self.wl[-1])}'

                    savepath, _ = QFileDialog.getSaveFileName(
                        self,
                        "Choose Model filename",
                        os.path.join(save_model_folder, name+spectral_range_name),
                        "joblib (*.joblib)"
                    )

            else:
                trained=True

        # enforce uniqueness on 'name'
        unique = self._ensure_unique_name(name)

        if unique is None:
            return

        job = ClassificationJob(unique, clf_type, kind)
        if not trained:
            job.status='To Train'
        job.binary_param=self.binary_param
        job.binary_algo=self.binary_algo
        job.spectral_range_used=[self.wl[0],self.wl[-1]]
        job.trained=trained
        job.trained_path=savepath
        job.substrate_name=self.substrate_label

        self.jobs[unique] = job
        self.job_order.append(unique)
        self._refresh_table()
        self._refresh_show_model_combo()

    def remove_all_jobs(self,ask_confirm=True):

        print('[Ask remove confirm : ]',ask_confirm)
        if ask_confirm:
            for job in self.jobs:
                if self.jobs[job].status in ['Done', 'Running']:
                    reply = QMessageBox.question(
                        self, "Reset Jobs ?",
                        f"Some jobs have been done and will be loosed.\n Are you sure to continue ? ",
                        QMessageBox.Yes | QMessageBox.Cancel
                    )
                    if reply == QMessageBox.Cancel:
                        return
                    break

        table = self.tableWidget_classificationList
        for row in range(table.rowCount()):
            self.remove_job(-1, confirm_done=False)

        self.update_overlay()

    def remove_job(self,row,confirm_done = True):
        table = self.tableWidget_classificationList
        table.size()
        if row < 0:
            row=table.rowCount()-1

        # If the first column holds the job name; adjust index if yours differs
        NAME_COL = 0
        item = table.item(row, NAME_COL)
        if item is None:
            return
        name = item.text()

        # Prevent removing the currently running job
        if self._current_worker and 0 <= self._running_idx < len(self.job_order):
            running_name = self.job_order[self._running_idx]
            if name == running_name:
                QMessageBox.warning(self, "Busy", "Job is running. Stop it first.")
                return

        job = self.jobs.get(name)
        if confirm_done and job and job.status == "Done":
            if not self._confirm("Confirm removal",
                                 f"'{name}' is already DONE.\nDo you want to remove it anyway?"):
                return

        # Update data structures by name (not by row index)
        self.jobs.pop(name, None)
        try:
            self.job_order.remove(name)
        except ValueError:
            pass  # already gone / not in the list

        # Keep running index consistent if we removed something before it
        if hasattr(self, "_running_idx") and self._running_idx is not None and self._running_idx > -1:
            # If the removed job was before the running one in job_order, shift left
            # (only matters when a queue is active and indices are aligned to job_order)
            # No-op if queue not running.
            pass  # optional: recompute _running_idx from running_name if needed

        # Remove the row in the UI safely (sorting can reorder view rows)
        sorting = table.isSortingEnabled()
        if sorting:
            table.setSortingEnabled(False)
        table.removeRow(row)
        if sorting:
            table.setSortingEnabled(True)

        # Refresh any dependent UI
        self._refresh_show_model_combo()
        self._refresh_table()
        self.update_overlay()

    def remove_selected_job(self):
        table = self.tableWidget_classificationList
        row = table.currentRow()
        self.remove_job(row)
        self.update_overlay()

    def start_selected_job(self):
        table = self.tableWidget_classificationList
        row = table.currentRow()
        if row < 0 or row >= len(self.job_order):
            return
        name = self.job_order[row]
        job = self.jobs.get(name)
        if not job:
            return

        if job.status == "Done":
            if not self._confirm("Confirm re-run",
                                 f"'{name}' is already DONE.\nDo you want to run it again?"):
                return
            # Reset visible state for a clean re-run
            job.status = "Queued"
            job.progress = 0
            job.duration_s = None
            self._update_row_from_job(name)

        rec = self._get_selected_rect()
        self.saved_rec = rec
        job.rect = rec

        self._running_idx = row
        self._stop_all = False
        self._skip_done_on_run = False
        self.only_selected = True
        self._launch_next_job()

    def move_selected_job_up(self):
        row = self._selected_row()
        if row <= 0: return
        self.job_order[row - 1], self.job_order[row] = self.job_order[row], self.job_order[row - 1]
        self._refresh_table()
        self.tableWidget_classificationList.selectRow(row - 1)

    def move_selected_job_down(self):
        row = self._selected_row()
        if row < 0 or row >= len(self.job_order) - 1: return
        self.job_order[row + 1], self.job_order[row] = self.job_order[row], self.job_order[row + 1]
        self._refresh_table()
        self.tableWidget_classificationList.selectRow(row + 1)

    def start_queue(self):
        if not self.job_order:
            QMessageBox.information(self, "Empty", "No jobs in the queue.")
            return
        if self._current_worker is not None:
            QMessageBox.information(self, "Busy", "A job is already running.")
            return

        self._stop_all = False
        self._skip_done_on_run = False
        self.only_selected=False

        rec=self._get_selected_rect()
        self.saved_rec=rec
        for key in self.jobs:
            self.jobs[key].rect=rec

        """Start the queue of jobs, with confirmation if some are already Done."""
        done_exists = any(self.jobs.get(n) and self.jobs[n].status == "Done" for n in self.job_order)
        if done_exists:
            reply = QMessageBox.question(
                self,
                "Confirm restart",
                "Some jobs are already marked as Done.\n"
                "Do you want to restart them anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            self._skip_done_on_run = (reply == QMessageBox.No)
        else:
            self._skip_done_on_run = False

        self._running_idx = 0
        self._launch_next_job()

    def stop_queue(self):
        """Arrête proprement la file entière : annule le job courant et bloque toute relance."""
        # 1) bloquer les relances
        self._stop_all = True

        # 2) annuler le worker en cours
        if self._current_worker:
            self._current_worker.cancel()
            if 0 <= self._running_idx < len(self.job_order):
                running_name = self.job_order[self._running_idx]
                job = self.jobs.get(running_name)
                if job:
                    if job.status=='Training ...':
                        job.status = "To Train"
                    else :
                        job.status = "Canceled"
                    job.progress = 0
                    job.duration_s = None
                    self._update_row_from_job(running_name)

        # 3) marquer les jobs restants comme annulés (hors courant)
        if self._running_idx >= 0:
            for i in range(self._running_idx + 1, len(self.job_order)):
                name = self.job_order[i]
                job = self.jobs.get(name)
                if job and job.status in ("Queued", "Running"):
                    job.status = "Canceled"
                    job.progress = 0
                    job.duration_s = None
                    self._update_row_from_job(name)
                elif job and job.status in ("To Train", "Training ..."):
                    job.status = "To Train"
                    job.progress = 0

    def _launch_next_job(self):
        """Launch the next job in the queue if available."""
        if self._stop_all:
            self._current_worker = None
            self._running_idx = -1
            return

        if self.only_selected:
            idx_lim=self._running_idx+1
        else:
            idx_lim=len(self.job_order)

        while 0 <= self._running_idx < idx_lim:
            name = self.job_order[self._running_idx]
            job = self.jobs[name]

            # Skip jobs that are already Done if they remain in the queue
            if job.status == "Done":
                if self._skip_done_on_run:
                    self._running_idx += 1
                    continue
                else:
                    # Rerun: reset visible state (do not modify job_order)
                    job.status = "Queued"
                    job.progress = 0
                    job.duration_s = None
                    self._update_row_from_job(name)

            if job.status == "To Train":
                # flip to running (training)
                job.status = "Training..."
                job.progress = 0
                job._t0 = time.time()
                self._update_row_from_job(name)

                # Prepare TrainWorker
                kind_str = job.kind.lower()  # e.g., "ink 3 classes" or "background"
                # pick a canonical "kind" token for your datasets
                train_kind = "3_inks_classes" if "ink" in kind_str else (
                    "background_10pct" if "10pct" in kind_str else "background")
                w = TrainWorker(job.clf_type, train_kind, job.trained_path, self.wl)
                n = job.name
                w.signals.progress.connect(lambda v, n=n: self._on_job_progress(n, v))
                w.signals.error.connect(lambda msg, n=n: self._on_job_error(n, msg))
                w.signals.progress.connect(lambda v, n=n: self._on_job_progress(n, v))

                def _on_saved(path, n=n):
                    j = self.jobs.get(n)
                    if not j: return
                    if self._stop_all or j.status == "Canceled":
                        return
                    j.trained = True
                    j.trained_path = path
                    j.status = "Queued"  # queue it for classification right away
                    j.duration_s = None if j._t0 is None else (time.time() - j._t0)
                    self._update_row_from_job(n)
                    # Immediately proceed to (re)launch on the same index
                    self._launch_next_job()

                w.signals.saved.connect(_on_saved)
                w.signals.finished.connect(lambda n=n: None)  # keep if you need cleanup

                self._current_worker = w
                self.threadpool.start(w)
                return

            # Check preconditions
            if self.binary_map is None or self.data is None:
                QMessageBox.warning(self, "Warning", "Run binarization and load a cube first.")
                job.status = "Error"
                self._update_row_from_job(name)
                self._running_idx += 1
                continue

            # Set job to Running
            job.status = "Running"
            job.progress = 0
            job._t0 = time.time()

            self._update_row_from_job(name)

            # Load classifier and create worker

            self.load_classifier(job.trained_path)

            # --- make sure model grid matches current cube; train if needed
            ok, msg = self.features_compatible(getattr(self, "classifier_wl", None))
            if not ok:
                # ask/auto-train a compatible ink model
                reply = QMessageBox.question(
                    self, "Train ink model?",
                    f"{msg}\n\nTrain a new ink classifier for the current spectral range?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    job.status = "To Train"
                    self._update_row_from_job(name)
                    self._running_idx += 1
                    continue

                # Build a filename that embeds the spectral range
                spectral_tag = f"_{int(self.wl[0])}-{int(self.wl[-1])}"
                if getattr(sys, 'frozen', False):
                    BASE_DIR = sys._MEIPASS
                else:
                    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
                out_dir = os.path.join(BASE_DIR, "identification", "data")
                os.makedirs(out_dir, exist_ok=True)
                savepath = os.path.join(out_dir, f"model_{job.clf_type.lower()}{spectral_tag}.joblib")

                # Launch a TrainWorker for the 3-ink-classes training set
                w = TrainWorker(
                    clf_type=job.clf_type,
                    kind="3classes",
                    savepath=savepath,
                    train_wl=self.wl
                )

                # when saved, update the job and immediately relaunch prediction on the same job
                def _on_saved(path, n=name):
                    j = self.jobs.get(n)
                    if not j: return
                    if self._stop_all or j.status == "Canceled":
                        return
                    j.trained = True
                    j.trained_path = path
                    j.status = "Queued"
                    self._update_row_from_job(n)
                    self._launch_next_job()

                w.signals.saved.connect(_on_saved)
                w.signals.error.connect(lambda m, n=name: self._on_job_error(n, m))
                w.signals.finished.connect(lambda n=name: None)
                self._current_worker = w
                self.threadpool.start(w)
                return

            mask = self.binary_map.astype(bool)

            if job.rect is not None:
                x, y, w, h = job.rect
                roi_mask = np.zeros_like(mask, dtype=bool)
                roi_mask[x:(x+w), y:(y+h)] = True
                mask &= roi_mask

            job._mask_indices = np.flatnonzero(mask.ravel())  # indices plats des pixels True
            job._shape = mask.shape
            job.class_map = np.zeros(mask.shape, dtype=np.uint8)

            spectra = self.data[mask, :]
            N = spectra.shape[0]
            target_steps = 50
            min_chunk = 20
            max_chunk = 20000
            chunk_size = max(1, min(max_chunk, max(min_chunk, N // target_steps)))

            worker = ClassifyWorker(
                spectra, self.classifier, self.classifier_type,
                chunk_size=chunk_size, emit_partial=True
            )

            n = job.name
            worker.signals.progress.connect(lambda v, n=n: self._on_job_progress(n, v))
            worker.signals.result_partial.connect(lambda s, e, pc, n=n: self._on_job_partial(n, s, e, pc))
            worker.signals.result.connect(lambda preds, n=n, m=mask: self._on_job_result(n, preds, m))
            worker.signals.error.connect(lambda msg, n=n: self._on_job_error(n, msg))
            worker.signals.finished.connect(lambda n=n: self._on_job_finished(n))

            self._current_worker = worker
            self.threadpool.start(worker)
            return  # Exit after launching a job

        # If we reached here, no jobs left
        self._current_worker = None
        self._running_idx = -1

    def _update_row_from_job(self, name: str):
        row = self._name_to_row(name)
        if row is None: return
        job = self.jobs[name]
        # Status
        it = self.tableWidget_classificationList.item(row, 2)
        if it: it.setText(job.status)
        # Progress
        pb = self.tableWidget_classificationList.cellWidget(row, 3)
        if isinstance(pb, QProgressBar):
            if job.status in ["Running","Training..."]:
                pb.setVisible(True)
                pb.setValue(int(job.progress))
            elif job.status in ("Canceled", "Error", "Done"):
                if job.status == "Canceled":
                    pb.setValue(0)
            else:
                pb.setValue(int(job.progress))
        # Duration
        dur_item = self.tableWidget_classificationList.item(row, 4)
        if dur_item:
            dur_item.setText("" if job.duration_s is None else f"{job.duration_s:.1f} s")

    def _on_job_progress(self, name: str, value: int):
        job = self.jobs.get(name)
        if not job: return
        job.progress = max(0, min(100, int(value)))
        self._update_row_from_job(name)

    def _on_job_error(self, name: str, message: str):
        job = self.jobs.get(name)
        if not job: return
        # différencier une vraie erreur d’une annulation volontaire
        if "canceled" in message.lower() or "cancelled" in message.lower():
            if job.status=='Training ...':
                job.status = "To Train"
            else:
                job.status = "Canceled"
            job.progress = 0
        else:
            job.status = "Error"
        job.duration_s = None if job._t0 is None else (time.time() - job._t0)
        self._update_row_from_job(name)
        # (optionnel) n’affiche la QMessageBox que pour les vraies erreurs :
        if job.status == "Error":
            QMessageBox.critical(self, f"{name} error", message)

    def _on_job_result(self, name: str, preds: np.ndarray, mask: np.ndarray):
        """Store class_map and render progressively when each job ends."""
        job = self.jobs.get(name)
        if not job: return
        # build class_map for this job
        class_map = np.zeros_like(self.binary_map, dtype=np.uint8)
        class_map[mask] = preds + 1  # shift classes to 1..N (0 substrate)
        job.class_map = class_map

        # show result immediately on viewer
        self.class_map = class_map
        self.radioButton_overlay_identification.setChecked(True)
        self._refresh_show_model_combo(select_name=name)
        self.show_classification_result()

    def _on_job_partial(self, name: str, start: int, end: int, preds_chunk: np.ndarray):
        job = self.jobs.get(name)
        if not job:
            return

        # Sécurité: s'assurer que les structures existent
        if getattr(job, "_mask_indices", None) is None or getattr(job, "_shape", None) is None:
            if self.binary_map is None:
                return
            mask = self.binary_map.astype(bool)
            job._mask_indices = np.flatnonzero(mask.ravel())
            job._shape = mask.shape
            if job.class_map is None:
                job.class_map = np.zeros(job._shape, dtype=np.uint8)

        # Écrire le chunk dans la carte (classes +1 car 0 = substrat)
        flat_idx = job._mask_indices[start:end]
        cm_flat = job.class_map.ravel()
        cm_flat[flat_idx] = preds_chunk.astype(np.uint8) + 1
        job.class_map = cm_flat.reshape(job._shape)

        # S'assurer que le modèle est dispo dans le combo d’affichage
        # (au tout début, ça l’ajoute; ensuite, ça garde la sélection)
        current_cb_idx = self.comboBox_clas_show_model.currentIndex()
        current_name = None
        if current_cb_idx >= 0:
            current_name = (self.comboBox_clas_show_model.itemData(current_cb_idx, Qt.UserRole)
                            or self.comboBox_clas_show_model.currentText())
        self._refresh_show_model_combo(select_name=current_name or name)

        # Si ce modèle est celui affiché, on rafraîchit l’overlay en temps réel
        # (ou si c'est le seul dans la liste)
        selected_idx = self.comboBox_clas_show_model.currentIndex()
        selected_name = None
        if selected_idx >= 0:
            selected_name = (self.comboBox_clas_show_model.itemData(selected_idx, Qt.UserRole)
                             or self.comboBox_clas_show_model.currentText())

        if selected_name == name or self.comboBox_clas_show_model.count() == 1:
            self.class_map = job.class_map
            self.radioButton_overlay_identification.setChecked(True)
            self.show_classification_result()

    def _on_job_finished(self, name: str):
        job = self.jobs.get(name)
        if job:
            if job.status != "Error":
                job.status = "Done" if job.progress == 100 else ("Canceled" if job.progress < 100 else "Done")
            job.duration_s = None if job._t0 is None else (time.time() - job._t0)
            self._update_row_from_job(name)

        # clear worker and advance
        self._current_worker = None

        if self._stop_all:
            self._running_idx = -1
            return

        self._advance_and_launch_next()
        self._refresh_clean_sources_list()

    def _advance_and_launch_next(self):
        if self._stop_all or self.only_selected:
            self._running_idx = -1
            return

        self._running_idx += 1
        if self._running_idx < len(self.job_order):
            self._launch_next_job()
        else:
            self._running_idx = -1  # finished all

    def reinit_selected_job(self):
        table = self.tableWidget_classificationList
        row = table.currentRow()
        if row < 0 or row >= len(self.job_order):
            return
        name = self.job_order[row]
        job = self.jobs[name]

        if job.status == "Done":
            if not self._confirm(
                    "Confirm reinit",
                    f"'{name}' is already DONE.\nThis will discard its result and reset it.\nProceed?"
            ):
                return

        job.reinit()

        self._refresh_table()
        self._refresh_show_model_combo()

    def on_click_save_map(self):
        # 1) Collect available models
        models = self._get_available_models()
        if not models:
            QMessageBox.warning(self, "No models", "No classification models available.")
            return

        # 2) Open dialog
        dft_name=self.cube.metadata.get('source_names')

        try:
            if len(dft_name)==0:
                dft_name='BaseFileName'
            elif len(dft_name)==1:
                dft_name=dft_name[0]
            elif len(dft_name)==2:
                dft_name='-'.join(dft_name)
        except:
            dft_name = 'BaseFileName'

        dlg = SaveClassMapDialog(models, default_base_name=dft_name, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        base = dlg.base_name
        selected_models = dlg.selected_models
        want_h5 = dlg.want_h5
        want_png = dlg.want_png
        want_clean=dlg.want_clean

        # 3) Ask output folder
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        saved_files = []

        # 4) Save each selected model
        for model in selected_models:
            try:
                job = self.jobs[model]
                if want_clean:
                    if job.clean_map is not None:
                        class_map = job.clean_map
                        print(f'[SAVE] No cleaned map for job {job.name}')
                    else:
                        class_map=job.class_map
                else:
                    class_map=job.class_map

            except Exception as e:
                QMessageBox.warning(self, "Missing class map", f"No class map for '{model}':\n{e}")
                continue

            classifier_name, classifier_type = self._get_classifier_meta_for_model(model)

            filename_base = f"{base}_{model}_map"
            labels=self.labels
            labels[0]=job.substrate_name
            palette_rgb = [0] * (256 * 3)  # init black
            for idx, (b, g, r) in self.palette_bgr.items():
                palette_rgb[idx * 3:idx * 3 + 3] = [r, g, b]  # PIL wants RGB
            palette_rgb = np.asarray(palette_rgb, dtype=np.uint8).reshape((-1, 3))

            if isinstance(labels, dict):
                # convertir en liste ordonnée
                max_idx = max(int(k) for k in labels.keys()) if labels else -1
                labels = [labels.get(i, labels.get(i, f"class_{i}")) for i in range(max_idx + 1)]

            dic_binaire=job.binary_param
            dic_binaire['algorithm']=job.binary_algo

            metadata = {
                "classifier_name": classifier_name,
                "classifier_type": classifier_type,
                "class_labels": labels,
                "palette": palette_rgb,
                "wl": self.cube.wl,  # optionnel
                "source_names": self.cube.metadata.get("source_names"),
                "source_files": self.cube.metadata.get("source_files"),
                "rect_crop": job.rect,
                "binary_param": dic_binaire,
                "spectral_range_used": job.spectral_range_used
            }

            if want_clean:
                metadata["clean_param"]=job.clean_param

            if want_h5:
                path_h5 = _ensure_unique_path(out_dir, filename_base, ".h5")
                _write_h5_class_map(path_h5, class_map, metadata)
                saved_files.append(path_h5)

            if want_png:
                path_png = _ensure_unique_path(out_dir, filename_base, ".png")
                _write_indexed_png(path_png, class_map, palette_rgb)
                saved_files.append(path_png)

        if saved_files:
            QMessageBox.information(self, "Done", "Files saved:\n- " + "\n- ".join(saved_files))

    def _get_available_models(self):
        """
        Return all model names present in comboBox_clas_show_model.
        """
        cb = self.comboBox_clas_show_model
        return [cb.itemText(i) for i in range(cb.count())]

    def _get_class_map_for_model(self, model_name: str):
        """
        Fetch the 2D class_map for a model from self.jobs[model_name].clas_map.
        """
        if not hasattr(self, "jobs") or model_name not in self.jobs:
            raise KeyError(f"Model '{model_name}' not found in self.jobs.")
        job = self.jobs[model_name]

        if not hasattr(job, "class_map"):
            raise AttributeError(f"Job for '{model_name}' has no 'clas_map' attribute.")

        cm = np.asarray(job.class_map)
        if cm.ndim != 2:
            raise ValueError(f"Expected a 2D class_map for '{model_name}', got shape {cm.shape}.")
        return cm

    def _get_classifier_meta_for_model(self, model_name: str):
        """
        Return (classifier_name, classifier_type) for HDF5 attributes.
        Pulls from common job attributes; falls back to (model_name, "unknown").
        """
        job = self.jobs.get(model_name) if hasattr(self, "jobs") else None
        if job is not None:
            # Common attribute names you might have on your job:
            #  - name / model_name: human-readable name of the classifier
            #  - backend / framework / kind / model_type: library or family ("sklearn", "pytorch", ...)
            name = getattr(job, "name", None) or getattr(job, "model_name", None) or model_name
            backend = (
                    getattr(job, "backend", None)
                    or getattr(job, "framework", None)
                    or getattr(job, "kind", None)
                    or getattr(job, "model_type", None)
                    or "unknown"
            )
            return str(name), str(backend)
        return model_name, "unknown"

    def _confirm(self, title: str, message: str) -> bool:
        """Return True if user confirms Yes, otherwise False."""
        reply = QMessageBox.question(
            self, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return reply == QMessageBox.Yes

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

    def _current_job(self):
        """Return the job currently selected in comboBox_clas_show_model (or None)."""
        idx = self.comboBox_clas_show_model.currentIndex()
        if idx < 0:
            return None
        name = (self.comboBox_clas_show_model.itemData(idx, Qt.UserRole)
                or self.comboBox_clas_show_model.currentText())
        return self.jobs.get(name)

    def _fmt_rect(self, rect_tuple):
        """(y, x, h, w) → 'y:x:h:w' or '—'."""
        if not rect_tuple:
            return "—"
        y, x, h, w = rect_tuple
        return f"{y}:{x}:{h}:{w}"

    def _set_info_rows(self):

        fl = self.formLayout_Info
        # 1) clear previous content
        if fl is None:
            return
        while fl.count():
            item = fl.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # 2) collect data depending on mode
        is_binary = self.radioButton_overlay_binary.isChecked()
        src_names = []
        try:
            src_names = (self.cube.metadata or {}).get("source_names") or []
        except Exception:
            src_names = []

        if is_binary:
            # No specific job; show binary context + selection rect + sources
            title = QLabel("<b>Binary</b>")
            title.setTextFormat(Qt.RichText)
            fl.addRow(title)
            fl.addRow("Rect :", QLabel(self._fmt_rect(getattr(self, "binary_rec", None))))
            fl.addRow("Sources :", QLabel(", ".join(map(str, src_names)) or "—"))
            fl.addRow("Binary algorithm : ", QLabel(str(self.binary_algo)))
            fl.addRow("Binary parameters : ", QLabel(str(self.binary_param)))
            return

        # Classification mode → use currently displayed job
        job = self._current_job()
        if not job:
            # Nothing selected / no result yet
            title = QLabel("<b>Classification</b>")
            title.setTextFormat(Qt.RichText)
            fl.addRow(title)
            fl.addRow("Sources :", QLabel(", ".join(map(str, src_names)) or "—"))
            return

        # 3) Fill rows for the selected job
        title = QLabel(f"<b>{job.name}</b>")
        title.setTextFormat(Qt.RichText)
        fl.addRow(title)
        fl.addRow("Type :", QLabel(str(job.clf_type)))
        fl.addRow("Kind :",QLabel(str(job.kind)))
        fl.addRow("Rect :", QLabel(self._fmt_rect(getattr(job, "rect", None))))
        fl.addRow("Sources :", QLabel(", ".join(map(str, src_names)) or "—"))
        fl.addRow("Spectral range used :",QLabel(f'{job.spectral_range_used[0]} - {job.spectral_range_used[-1]}'))
        fl.addRow("Binary algorithm : ", QLabel(str(job.binary_algo)))
        fl.addRow("Binary parameters : ", QLabel(str(job.binary_param)))
        fl.addRow("Clean parameters : ", QLabel(str(job.clean_param)))

    def apply_clean_preset(self, _index: int):
        """
        Read the selected preset name and push its values
        into: spinBox_clean_window_size, spinBox_clean_iterations,
        spinBox_clean_min_area.
        """
        name = self.comboBox_clean_preset.currentText().strip()
        cfg = CLEAN_PRESETS.get(name)
        if not cfg:
            return

        # Block signals so we don't trigger other slots while updating
        widgets = [
            self.spinBox_clean_window_size,
            self.spinBox_clean_iterations,
            self.spinBox_clean_min_area,
        ]
        for w in widgets:
            w.blockSignals(True)

        try:
            # Window size (% of min(H,W)) – your spinBox range is [1..10]
            self.spinBox_clean_window_size.setValue(int(cfg["window_pct"]))

            # Iterations
            self.spinBox_clean_iterations.setValue(int(cfg["iterations"]))

            # Min object area (px)
            self.spinBox_clean_min_area.setValue(int(cfg["min_area"]))

        finally:
            for w in widgets:
                w.blockSignals(False)

    def _init_cleaning_list(self):
        """Initialize the model for listView_classificationList_clean."""
        self.clean_list_model = QStandardItemModel(self.listView_classificationList_clean)
        self.listView_classificationList_clean.setModel(self.clean_list_model)

    def _refresh_clean_sources_list(self):
        """
        Rebuild the Cleaning list with:
          - Binary map (if exists)
          - All jobs with status 'Done' and a non-None class_map
        """
        self.clean_list_model.clear()

        # 1) Binary map entry
        if getattr(self, "binary_map", None) is not None:
            item = QStandardItem("Binary map")
            item.setData(("binary", None), Qt.UserRole)
            self.clean_list_model.appendRow(item)

        # 2) DONE jobs
        for name in self.job_order:
            job = self.jobs.get(name)
            if not job:
                continue
            if job.status == "Done" and getattr(job, "class_map", None) is not None:
                item = QStandardItem(name)
                item.setData(("job", name), Qt.UserRole)
                self.clean_list_model.appendRow(item)

    def _get_selected_clean_source(self):
        """Return (class_map, label) from the current selection in the list."""
        idx = self.listView_classificationList_clean.currentIndex()
        if not idx.isValid():
            raise ValueError("Please select a source in the Cleaning list.")
        tag = idx.data(Qt.UserRole)
        if not tag:
            raise ValueError("Internal selection error.")

        kind, name = tag
        if kind == "binary":
            if self.binary_map is None:
                raise ValueError("Binary map not available.")
            return self.binary_map, "Binary"
        elif kind == "job":
            job = self.jobs.get(name)
            if not job or job.class_map is None:
                raise ValueError(f"No class map for '{name}'.")
            return job, name
        else:
            raise ValueError("Unknown selection type.")

    def _iter_all_clean_sources(self):
        """
        Itère sur TOUTES les lignes du ListView et yield (class_map, label).
        Saute les entrées invalides.
        """
        model = self.clean_list_model
        for row in range(model.rowCount()):
            idx = model.index(row, 0)
            tag = idx.data(Qt.UserRole)
            if not tag:
                continue
            kind, name = tag
            if kind == "binary":
                if self.binary_map is not None:
                    yield self.binary_map, "Binary"
            elif kind == "job":
                job = self.jobs.get(name)
                if job is not None and getattr(job, "class_map", None) is not None:
                    yield job, name

    def _on_click_clean_start_selected(self):
        """Handler for pushButton_clean_start_selected."""
        try:
            class_map, src_label = self._get_selected_clean_source()
        except Exception as e:
            QMessageBox.warning(self, "Cleaning", str(e))
            return

        params = self._collect_clean_params()
        self._apply_cleaning_pipeline(class_map, src_label, params)

    def _on_click_clean_start_all(self):
        params = self._collect_clean_params()
        n_total, n_ok, n_err = 0, 0, 0

        for obj, src_label in self._iter_all_clean_sources():
            n_total += 1
            try:
                self._apply_cleaning_pipeline(obj, src_label, params)
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"[Cleaning][{src_label}] ERROR: {e}")

        QMessageBox.information(
            self, "Cleaning (batch)",
            f"Processed: {n_total}\nOK: {n_ok}\nErrors: {n_err}"
        )

    def _collect_clean_params(self):
        """
        Lit les paramètres UI et renvoie un dict prêt pour ta pipeline de cleaning.
        """
        return {
            "window_pct": self.spinBox_clean_window_size.value(),  # %
            "iterations": self.spinBox_clean_iterations.value(),  # int
            "min_area": self.spinBox_clean_min_area.value(),  # px
        }

    def _odd_ksize_from_pct(self, window_pct: int, shape_hw):
        """Convert a percentage of min(H,W) to an odd kernel size >= 3."""
        H, W = shape_hw
        k = max(3, int(round((window_pct / 100.0) * min(H, W))))
        if k % 2 == 0:
            k += 1
        return k

    def _majority_filter_labels(self, cls: np.ndarray, k: int, iterations: int, foreground_only: bool) -> np.ndarray:
        """
        Fast sliding-window majority over labels using per-class counts via cv2.filter2D.
        Only considers classes > 0. Background (0) is kept if foreground_only=True.
        """
        out = cls.copy()
        H, W = out.shape
        # classes (exclude 0)
        classes = np.unique(out)
        classes = classes[classes > 0]
        if classes.size == 0:
            return out

        kernel = np.ones((k, k), dtype=np.uint8)  # uniform window
        for _ in range(int(iterations)):
            # Build per-class count maps
            counts = []
            for c in classes:
                mask = (out == c).astype(np.uint8)
                csum = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
                counts.append(csum.astype(np.int32))
            # Stack counts -> (H, W, C)
            stack = np.stack(counts, axis=-1)  # int32
            # Winner = argmax along classes axis
            winner_idx = np.argmax(stack, axis=-1)  # (H, W), 0..C-1
            winner_labels = classes[winner_idx]  # map back to labels

            # Update only where original was foreground
            fg = out > 0
            out[fg] = winner_labels[fg]

        return out

    def _remove_small_components_per_class(self, cls: np.ndarray, min_area: int) -> np.ndarray:
        """
        Remove connected components smaller than min_area for each class > 0.
        Components removed are set to background (0).
        """
        if min_area <= 0:
            return cls
        out = cls.copy()
        classes = np.unique(out)
        for c in classes:
            if c == 0:
                continue
            mask = (out == c).astype(np.uint8)
            if mask.max() == 0:
                continue
            num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # stats rows: 0 is background of this mask
            for comp_id in range(1, num):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    out[lab == comp_id] = 0
        return out

    def run_cleaning(self, class_map: np.ndarray, params: dict) -> np.ndarray:
        """
        Main entry point:
        - window_pct, iterations,  in_area
        Returns a cleaned class_map with same dtype as input.
        """
        cls = class_map.copy()
        dtype = cls.dtype
        H, W = cls.shape[:2]
        fg_only = True

        min_area = int(params.get("min_area", 0))

        k = self._odd_ksize_from_pct(int(params.get("window_pct", 3)), (H, W))
        iters = int(params.get("iterations", 2))
        cls = self._majority_filter_labels(cls, k=k, iterations=iters, foreground_only=fg_only)

        # Remove small islands per class (>0)
        cls = self._remove_small_components_per_class(cls, min_area=min_area)

        # Keep dtype
        return cls.astype(dtype, copy=False)

    def _apply_cleaning_pipeline(self, obj, label: str, params: dict):
        if label=='Binary':
            class_map=obj
            self.binary_map = self.run_cleaning(class_map, params)
            self.show_binary_result()
        else:
            class_map=obj.class_map
            obj.clean_map=self.run_cleaning(class_map, params)
            obj.clean_param=params
            self.show_classification_result()

    def features_compatible(self,wl_ref, tol_nm: float = 1):

        if wl_ref is not None:
            wl_ref = np.asarray(wl_ref, dtype=float)

            if wl_ref.shape == self.wl.shape and np.allclose(wl_ref, self.wl, atol=tol_nm, rtol=0):
                return True, "Exact wavelength grid match."
            else:
                return False, (f"Wavelength grid mismatch: "
                               f"model bands={wl_ref.size}, cube bands={self.wl.size}")

        return False, "Cannot verify feature compatibility (no train_wl in model)."

    def load_map(self):
        """
        Charge une carte de classification sauvegardée en .h5 avec _write_h5_class_map
        et l'ajoute comme un nouveau job 'Done' dans la file, avec palette + labels
        + métadonnées (rect, params binaires, clean, spectral range…).
        """
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        import os
        import h5py
        import numpy as np

        # --- 1) Choix du fichier ---
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load classification map",
            "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if not path:
            return

        def _decode_str(x):
            if isinstance(x, (bytes, np.bytes_)):
                try:
                    return x.decode("utf-8")
                except Exception:
                    return x.decode(errors="ignore")
            return str(x)

        def _read_group_to_dict(group):
            """Lecture récursive d'un groupe HDF5 -> dict (pour binary_param / clean_param)."""
            out = {}
            # attrs
            for k, v in group.attrs.items():
                out[k] = v
            # datasets / sous-groupes
            for name, item in group.items():
                if isinstance(item, h5py.Dataset):
                    data = item[()]
                    # Décodage éventuel de chaînes
                    if isinstance(data, (bytes, np.bytes_)):
                        data = _decode_str(data)
                    elif isinstance(data, np.ndarray) and data.dtype.kind in ("S", "O", "U"):
                        # tableau de chaînes
                        data = [
                            _decode_str(val)
                            for val in data.ravel().tolist()
                        ]
                    out[name] = data
                elif isinstance(item, h5py.Group):
                    out[name] = _read_group_to_dict(item)
            return out

        try:
            # --- 2) Lecture du fichier HDF5 ---
            with h5py.File(path, "r") as f:
                if "class_map" not in f:
                    raise ValueError("Dataset 'class_map' not found in file.")

                class_map = np.asarray(f["class_map"][()])
                if class_map.ndim != 2:
                    raise ValueError(f"'class_map' must be 2D, got shape {class_map.shape}.")

                meta_grp = f.get("Metadata", None)

                # Valeurs par défaut
                base_name = os.path.splitext(os.path.basename(path))[0]
                clf_name = base_name
                clf_type = "Loaded H5 map"
                labels = None
                palette_bgr = None
                rect = None
                spectral_range_used = None
                binary_param = None
                clean_param = None
                substrate_name = "Unknown Substrate"

                # --- 3) Métadonnées si dispo ---
                if meta_grp is not None:
                    # Nom / type de classif
                    if "classifier_name" in meta_grp.attrs:
                        clf_name = _decode_str(meta_grp.attrs["classifier_name"])
                    if "classifier_type" in meta_grp.attrs:
                        clf_type = _decode_str(meta_grp.attrs["classifier_type"])

                    # Labels des classes
                    if "class_labels" in meta_grp:
                        raw = meta_grp["class_labels"][()]
                        arr = np.array(raw)
                        if arr.ndim == 0:
                            arr = arr.reshape(1)
                        labels = {}
                        for i, v in enumerate(arr.tolist()):
                            labels[i] = _decode_str(v)
                        if labels:
                            substrate_name = labels.get(0, substrate_name)

                    # Palette (enregistrée en RGB, on repasse en BGR)
                    if "palette" in meta_grp:
                        pal = np.asarray(meta_grp["palette"][()], dtype=np.uint8)
                        palette_bgr = {}
                        for i in range(pal.shape[0]):
                            r, g, b = pal[i]
                            palette_bgr[i] = (int(b), int(g), int(r))

                    # Rectangle de crop
                    if "rect_crop" in meta_grp:
                        rc = np.asarray(meta_grp["rect_crop"][()]).ravel()
                        if rc.size >= 4:
                            y, x, h, w = [int(v) for v in rc[:4]]
                            rect = (y, x, h, w)

                    # Spectral range utilisé
                    if "spectral_range_used" in meta_grp:
                        sr = np.asarray(meta_grp["spectral_range_used"][()]).ravel()
                        if sr.size >= 1:
                            spectral_range_used = [float(sr[0]), float(sr[-1])]

                    # Params binaires / cleaning
                    if "binary_param" in meta_grp:
                        binary_param = _read_group_to_dict(meta_grp["binary_param"])
                    if "clean_param" in meta_grp:
                        clean_param = _read_group_to_dict(meta_grp["clean_param"])

                # Fallback du spectral_range si nécessaire
                if spectral_range_used is None:
                    if isinstance(self.wl, np.ndarray) and self.wl.size > 0:
                        spectral_range_used = [float(self.wl[0]), float(self.wl[-1])]
                    else:
                        spectral_range_used = [0.0, 0.0]

                # --- 4) Palette / labels par défaut si manquants ---
                max_cls = int(class_map.max()) if class_map.size else 0

                if labels is None:
                    labels = {i: f"class_{i}" for i in range(max_cls + 1)}

                if palette_bgr is None:
                    # Si une palette existe déjà dans le widget, on la recycle
                    if hasattr(self, "palette_bgr") and self.palette_bgr:
                        palette_bgr = {}
                        for i in range(max_cls + 1):
                            palette_bgr[i] = self.palette_bgr.get(i, (0, 0, 0))
                    else:
                        # Palette simple pseudo-aléatoire mais stable
                        palette_bgr = {0: (128, 128, 128)}
                        for i in range(1, max_cls + 1):
                            palette_bgr[i] = (
                                (37 * i) % 255,
                                (91 * i) % 255,
                                (179 * i) % 255
                            )

        except Exception as e:
            QMessageBox.critical(self, "Load map error", f"Could not load map:\n{e}")
            return

        # --- 5) Intégration dans le widget comme un job 'Done' ---

        # Mise à jour de la palette globale + labels
        self.palette_bgr = palette_bgr
        self.labels = labels
        self.substrate_label = substrate_name

        # Nom unique pour le job
        unique_name = self._ensure_unique_name(clf_name)
        if unique_name is None:
            # L'utilisateur a refusé d'ajouter un doublon
            return

        # Création du job
        job = ClassificationJob(unique_name, clf_type, kind="Loaded H5 map")
        job.status = "Done"
        job.class_map = class_map.astype(np.int32, copy=False)
        job.rect = rect
        job.spectral_range_used = spectral_range_used
        job.binary_param = binary_param
        job.clean_param = clean_param
        if isinstance(binary_param, dict):
            job.binary_algo = binary_param.get("algorithm")
        else:
            job.binary_algo = None
        job.substrate_name = substrate_name

        # Enregistrement dans les structures
        self.jobs[unique_name] = job
        self.job_order.append(unique_name)
        self._refresh_table()
        self._refresh_show_model_combo(select_name=unique_name)
        self._refresh_clean_sources_list()

        # --- 6) Affichage immédiat ---

        # Problème potentiel si la taille de la map != image RGB
        if self.data is not None and class_map.shape != self.data.shape[:2]:
            QMessageBox.warning(
                self, "Size mismatch",
                f"Loaded map shape {class_map.shape} does not match current cube image "
                f"shape {self.data.shape[:2]}.\n"
                "The map will be shown without RGB overlay."
            )

            # On affiche la carte seule (sans overlay) :
            def _colorize(cm: np.ndarray) -> np.ndarray:
                colors = self.palette_bgr
                h, w = cm.shape
                result_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for val, bgr in colors.items():
                    result_rgb[cm == val] = bgr
                return result_rgb

            rgb_map = _colorize(class_map)
            self.viewer_right.setImage(self._np2pixmap(rgb_map))
            self.label_viewer_right.setText("RAW map (loaded)")
            # On laisse la viewer_left telle quelle (RGB actuelle ou vide)
            self.labels[0] = job.substrate_name
            self.update_legend()
            self._set_info_rows()
        else:
            # Cas normal : on utilise le pipeline existant
            self.class_map = class_map
            self.radioButton_overlay_identification.setChecked(True)
            self.show_classification_result()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IdentificationWidget()
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
