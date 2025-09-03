import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QWidget, QApplication, QFileDialog, QSizePolicy, QMessageBox, QSplitter,
                             QDialog,QPushButton,QTableWidgetItem,QHeaderView,QProgressBar,
                             )
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import traceback

from pandas.core.common import count_not_none

from identification.identification_window import Ui_IdentificationWidget
from hypercubes.hypercube import Hypercube
from ground_truth.ground_truth_tool import ZoomableGraphicsView
from identification.load_cube_dialog import Ui_Dialog

# todo : crop cube before binarization/classification
# todo : block slider rgb default
# todo : on job partial to implement
# todo : parfaire le chargement des cubes
# todo : pb avec slier et radiobuttons false color
# todo : pb avec gestion de la table depuis maj passage au rerun
# todo : affichage touts les resultats ensemble
# todo : save clasification
# todo : launch/reinit selected
# todo : check npix pb for chunk size
# todo : legend and update legend

class ClassifySignals(QObject):
    finished = pyqtSignal()                          # fin (toujours émis)
    error = pyqtSignal(str)                          # message d'erreur
    progress = pyqtSignal(int)                       # 0..100
    result = pyqtSignal(np.ndarray)                  # prédictions finales (shape: [N,])
    result_partial = pyqtSignal(int, int, np.ndarray)  # (start, end, preds_chunk)

class ClassifyWorker(QRunnable):
    """
    Exécute une classification sur 'spectra' (np.ndarray [N_pixels, N_bands]) en CHUNKS.
    - classifier: scikit-learn pipeline/estim. OU modèle PyTorch (eval).
    - classifier_type: "sklearn" ou "cnn".
    - chunk_size: nb de pixels par batch (équilibre RAM/vitesse).
    - emit_partial: si True, émet result_partial(start, end, preds_chunk) à chaque chunk.
      -> permet d'afficher la class_map PROGRESSIVEMENT.
    """
    def __init__(self, spectra: np.ndarray, classifier, classifier_type: str,
                 chunk_size: int = 100_000, emit_partial: bool = True):
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

@dataclass
class ClassificationJob:
    name: str                 # unique key shown in table (e.g., "SVM (RBF)")
    clf_type: str             # "knn" | "cnn" | "svm" etc...
    kind: List[str]           # e.g., ["Substrate","Ink 3 classes" ...]
    status: str = "Queued"    # "Queued" | "Running" | "Done" | "Canceled" | "Error"
    progress: int = 0         # 0..100
    duration_s: Optional[float] = None
    class_map: Optional[np.ndarray] = None
    _t0: Optional[float] = field(default=None, repr=False)  # internal start time

def fused_cube(cube1,cube2):
    cubes={}
    if cube1.wl[0]<500 and  cube2.wl[0]>800:
        cubes['VNIR']=cube1
        cubes['SWIR']=cube2
    elif cube1.wl[0]>800 and  cube2.wl[0]<500:
        cubes['VNIR'] = cube2
        cubes['SWIR'] = cube1
    else:
        print('error with cubes range')
        return

    target_ranges = {'VNIR': (400, 950), 'SWIR': (955, 1700)}

    # Vérification couverture
    full_covered = all(
        key in cubes and
        cubes[key].wl[0] <= target_ranges[key][0] and
        cubes[key].wl[-1] >= target_ranges[key][1]
        for key in target_ranges
    )

    if full_covered:
        hyps_cut = {}
        for key in target_ranges:
            wl = cubes[key].wl
            data = cubes[key].data
            start_idx = np.argmin(np.abs(wl - target_ranges[key][0]))
            end_idx = np.argmin(np.abs(wl - target_ranges[key][1]))
            data_cut = data[:, :, start_idx:end_idx + 1]
            wl_cut = wl[start_idx:end_idx + 1]
            hyps_cut[key] = Hypercube(data=data_cut, wl=wl_cut,
                                      cube_info=cubes[key].cube_info)

        data_fused = np.concatenate((hyps_cut['VNIR'].data, hyps_cut['SWIR'].data), axis=2)
        wl_fused = np.concatenate((hyps_cut['VNIR'].wl, hyps_cut['SWIR'].wl))

        return data_fused,wl_fused

class LoadCubeDialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None, wl_step=5):
        super().__init__(parent)
        self.setupUi(self)

        self.cubes = {}  # {"VNIR": cube_obj, "SWIR": cube_obj, ...}
        self.wl_step = wl_step  # pas spectral pour interpolation

        # Connexion des boutons
        self.pushButton_load_cube_1.clicked.connect(lambda: self.load_cube("VNIR"))
        self.pushButton_load_cube_2.clicked.connect(lambda: self.load_cube("SWIR"))
        self.pushButton_valid.clicked.connect(self.validate_cubes)

        self.update_instructions()

    def load_cube(self, key,filepath=None):
        """Charge un cube hyperspectral et le stocke interpolé."""

        # Charger cube brut
        cube = Hypercube()
        cube.open_hyp(default_path=filepath, ask_calib=False)
        filepath=cube.cube_info.filepath

        if cube.data is None or cube.wl is None:
            QMessageBox.warning(self, "Error", "Failed to load cube or missing wavelengths.")
            return

        # Interpolation
        data_interp, wl_interp = cube.get_interpolate_cube(self.wl_step)
        cube.data = data_interp
        cube.wl = wl_interp

        # Stockage
        self.cubes[key] = cube

        # Mise à jour affichage
        if key == "VNIR":
            self.label_filepath_cube_1.setText(filepath)
            self.label_spec_range_cube_1.setText(f"{wl_interp[0]:.0f} - {wl_interp[-1]:.0f} nm")
        elif key == "SWIR":
            self.label_filepath_cube_2.setText(filepath)
            self.label_spec_range_cube_2.setText(f"{wl_interp[0]:.0f} - {wl_interp[-1]:.0f} nm")

        self.update_instructions()

    def update_instructions(self):
        """Affiche l'état actuel de la couverture spectrale."""
        ranges = []
        for key, cube in self.cubes.items():
            ranges.append((cube.wl[0], cube.wl[-1]))

        if not ranges:
            self.label_instructions.setText("No cubes loaded yet.")
            return

        ranges.sort()
        text_ranges = " + ".join([f"{r[0]:.0f}-{r[1]:.0f}nm" for r in ranges])
        self.label_instructions.setText(f"Currently covered: {text_ranges}")

    def validate_cubes(self):
        """Vérifie la couverture et fusionne si possible."""
        target_ranges = {'VNIR': (400, 950), 'SWIR': (955, 1700)}

        # Vérification couverture
        full_covered = all(
            key in self.cubes and
            self.cubes[key].wl[0] <= target_ranges[key][0] and
            self.cubes[key].wl[-1] >= target_ranges[key][1]
            for key in target_ranges
        )

        if full_covered:
            hyps_cut = {}
            for key in target_ranges:
                wl = self.cubes[key].wl
                data = self.cubes[key].data
                start_idx = np.argmin(np.abs(wl - target_ranges[key][0]))
                end_idx = np.argmin(np.abs(wl - target_ranges[key][1]))
                data_cut = data[:, :, start_idx:end_idx + 1]
                wl_cut = wl[start_idx:end_idx + 1]
                hyps_cut[key] = Hypercube(data=data_cut, wl=wl_cut,
                                          cube_info=self.cubes[key].cube_info)

            data_fused = np.concatenate((hyps_cut['VNIR'].data, hyps_cut['SWIR'].data), axis=2)
            wl_fused = np.concatenate((hyps_cut['VNIR'].wl, hyps_cut['SWIR'].wl))

            self.accept()  # ferme la boîte
            self.result_data = (hyps_cut, data_fused, wl_fused)

        else:
            choice = QMessageBox.question(
                self, "Incomplete range",
                "The full range 400–1700nm is not covered.\nDo you want to add a new cube?",
                QMessageBox.Yes | QMessageBox.No
            )
            if choice == QMessageBox.Yes:
                # Ajoute un nouveau bouton pour un cube supplémentaire
                btn = QPushButton(f"Load cube {len(self.cubes) + 1}", self.frame)
                btn.clicked.connect(lambda: self.load_cube(f"extra_{len(self.cubes)+1}"))
                row = len(self.cubes) + 1
                self.gridLayout.addWidget(btn, row, 2, 1, 1)
            else:
                # Retourne les cubes incomplets
                self.accept()
                self.result_data = (self.cubes, None, None)

class IdentificationWidget(QWidget, Ui_IdentificationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Queue structures
        self.job_order: List[str] = []  # only job names in execution order
        self.jobs: Dict[str, ClassificationJob] = {}  # name -> job

        # Table init
        self._init_classification_table()

        # Classifiy as thread init
        self.threadpool = QThreadPool()
        self._running_idx: int = -1
        self._current_worker = None
        self._stop_all = False

        # Combo to select which model's result to show
        self.comboBox_clas_show_model.clear()
        self.comboBox_clas_show_model.currentIndexChanged.connect(self._on_show_model_changed)
        self._refresh_show_model_combo()  # start empty/disabled

        # Remplacer placeholders par ZoomableGraphicsView
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)

        self.cube = None
        self.data = None
        self.wl = None
        self.binary_map = None
        self.alpha = self.horizontalSlider_overlay_transparency.value() / 100.0

        # Connections
        self.pushButton_load.clicked.connect(self.open_load_cube_dialog)
        self.pushButton_launch_bin.clicked.connect(self.launch_binarization)
        self.horizontalSlider_overlay_transparency.valueChanged.connect(self.update_alpha)
        self.comboBox_bin_algorith_choice.currentIndexChanged.connect(self.update_bin_defaults)

        self.pushButton_clas_add_ink.clicked.connect(
            lambda: self.add_job(self.comboBox_clas_ink_model.currentText())        )
        self.pushButton_clas_remove.clicked.connect(self.remove_selected_job)
        self.pushButton_clas_up.clicked.connect(self.move_selected_job_up)
        self.pushButton_clas_down.clicked.connect(self.move_selected_job_down)
        self.pushButton_clas_start.clicked.connect(self.start_queue)
        self.pushButton_clas_stop.clicked.connect(self.stop_queue)

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

        # --- Boutons de rotation / flip ---
        self.pushButton_rotate.clicked.connect(lambda: self.transform(np.rot90))
        self.pushButton_flip_h.clicked.connect(lambda: self.transform(np.fliplr))
        self.pushButton_flip_v.clicked.connect(lambda: self.transform(np.flipud))

        self.radioButton_overlay_binary.toggled.connect(self.update_overlay)
        self.radioButton_overlay_identification.toggled.connect(self.update_overlay)

        self.update_bin_defaults()
        self.palette_bgr = {
            0: (128, 128, 128),  # Substrate (gris)
            3: (211, 0, 148),  # MGP (violet)
            2: (0, 255, 255),  # CC (jaune)
            1: (0, 165, 255)  # NCC (orange)
        }

    def bgr_to_rgb(self, bgr):
        return (bgr[2], bgr[1], bgr[0])

    def open_load_cube_dialog(self):
        dlg = LoadCubeDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            hyps_cut, data_fused, wl_fused = dlg.result_data

            if data_fused is not None and wl_fused is not None:
                # plage complète fusionnée
                self.cube = Hypercube(data=data_fused,wl=wl_fused)
                self.data = data_fused
                self.wl = wl_fused
            else:
                # plage incomplète → on prend le premier cube chargé
                first_key = next(iter(dlg.cubes))
                self.cube = dlg.cubes[first_key]
                self.data = self.cube.data
                self.wl = self.cube.wl

            self.update_rgb_controls()
            self.show_rgb_image()

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
                element.setValue(self.default_rgb_channels()[i])
            elif self.radioButton_grayscale.isChecked():
                element.setEnabled(i == 2)
            else:
                element.setEnabled(True)

        for i, element in enumerate(self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.default_rgb_channels()[i])
            elif self.radioButton_grayscale.isChecked():
                element.setEnabled(i == 2)
            else:
                element.setEnabled(True)

        self.show_rgb_image()

    def default_rgb_channels(self):
        """Renvoie les canaux RGB par défaut selon plage spectrale"""
        if self.wl[-1] < 1100 and self.wl[0] > 350:
            return [610, 540, 435]
        elif self.wl[-1] >= 1100:
            return [1605, 1205, 1005]
        else:
            mid = int(len(self.wl) / 2)
            return [self.wl[0], self.wl[mid], self.wl[-1]]

    def transform(self, trans_type):
        """Rotation / Flip du cube"""
        try:
            self.data = trans_type(self.data)
            self.cube.data=self.data
            self.binary_map=trans_type(self.binary_map)
        except Exception as e:
            print(f"[transform] Failed: {e}")
            return
        self.show_rgb_image()
        self.show_binary_result()

    def load_cube(self,filepath=None,cube=None):
        flag_loaded=False
        if cube is not None:
            try:
                self.cube = cube
                self.data = self.cube.data
                self.wl = self.cube.wl
                flag_loaded=True
            except:
                print('Probleme with cube in parameter')

        if not flag_loaded:
            if not filepath:
                filepath, _ = QFileDialog.getOpenFileName(
                    self, "Open Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
                )
                if not filepath:
                    return
            try:
                self.cube = Hypercube(filepath=filepath, load_init=True)
                self.data = self.cube.data
                self.wl = self.cube.wl
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load cube: {e}")
                return

        self.update_rgb_controls()
        self.show_rgb_image()

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
        rgb = self.data[:, :, idx]
        rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)
        self.rgb_image = rgb
        self.viewer_left.setImage(self._np2pixmap(rgb))

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
        try:
            self.binary_map = self.cube.get_binary_from_best_band(algorithm, param)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Binarization failed: {e}")
            return
        self.radioButton_overlay_binary.setChecked(True)
        self.show_binary_result()

    def show_binary_result(self):
        if self.binary_map is None:
            return
        bin_img = (self.binary_map * 255).astype(np.uint8)
        bin_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        self.viewer_right.setImage(self._np2pixmap(bin_rgb))

    def update_overlay(self):
        if self.radioButton_overlay_binary.isChecked():
            self.show_binary_result()
        elif self.radioButton_overlay_identification.isChecked():
            self.show_classification_result()

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.update_overlay()

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

    def load_classifier(self,model_name):
        models_dir = "identification/data"

        if model_name == "CNN 1D":
            import torch
            from classifier_train import SpectralCNN1D
            print(f'SpectralCNN1D imported')
            input_length = self.data.shape[2]
            num_classes = 3
            self.classifier = SpectralCNN1D(input_length, num_classes)
            self.classifier.load_state_dict(torch.load(os.path.join(models_dir, "model_CNN1D.pth"), map_location="cpu"))
            print(f'SpectralCNN1D loaded')

            self.classifier.eval()
            self.classifier_type = "cnn"
            print(f'CNN 1D classifier initialized')

        else:
            import joblib
            filename = f"model_{model_name.lower()}.joblib"
            self.classifier = joblib.load(os.path.join(models_dir, filename))
            self.classifier_type = "sklearn"

    def classify_inks(self):
        if self.binary_map is None:
            QMessageBox.warning(self, "Warning", "Run binarization first.")
            return

        # Désactiver le bouton pour éviter les doubles clics
        self.pushButton_clas_ink_launch.setEnabled(False)

        mask = self.binary_map.astype(bool)
        spectra = self.data[mask, :]  # [N_pixels, N_bands]

        try:
            self.progressBar_classif.setValue(0)
            self.progressBar_classif.setVisible(True)
        except Exception:
            pass  # si pas de progress bar dans l'UI

        # Créer et lancer le worker
        worker = ClassifyWorker(spectra, self.classifier, self.classifier_type, chunk_size=100_000)
        worker.signals.progress.connect(self._on_classif_progress)
        worker.signals.error.connect(self._on_classif_error)
        worker.signals.result.connect(lambda preds: self._on_classif_result(preds, mask))
        worker.signals.finished.connect(self._on_classif_finished)

        self._current_worker = worker
        self.threadpool.start(worker)

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

    def _on_classif_finished(self):
        try:
            self.progressBar_classif.setVisible(False)
        except Exception:
            pass
        self.pushButton_clas_ink_launch.setEnabled(True)
        self._current_worker = None

    def update_legend(self):
        # Vider le contenu actuel du frame_legend
        for i in reversed(range(self.frame_legend.layout().count())):
            item = self.frame_legend.layout().itemAt(i)
            if item.widget():
                item.widget().deleteLater()

        layout = self.frame_legend.layout()
        if layout is None:
            from PyQt5.QtWidgets import QVBoxLayout
            layout = QVBoxLayout()
            self.frame_legend.setLayout(layout)

        if self.radioButton_overlay_binary.isChecked():
            legend_items = [
                ("Substrate", (128, 128, 128)),
                ("Ink", (255, 0, 0))
            ]
        else:  # Identification
            legend_items = [
                ("Substrate", (128, 128, 128)),
                ("MGP", (221, 0, 148)),
                ("CC", (0, 255, 255)),
                ("NCC", (0, 165, 255))
            ]

        from PyQt5.QtWidgets import QLabel
        from PyQt5.QtGui import QPixmap, QPainter, QColor

        for label_text, bgr in legend_items:
            # Créer une icône carrée colorée
            rgb = self.bgr_to_rgb(bgr)
            pix = QPixmap(20, 20)
            pix.fill(QColor(*rgb))
            icon_label = QLabel()
            icon_label.setPixmap(pix)

            text_label = QLabel(label_text)
            text_label.setStyleSheet("font-weight: bold;")

            from PyQt5.QtWidgets import QHBoxLayout, QWidget
            item_widget = QWidget()
            h_layout = QHBoxLayout()
            h_layout.setContentsMargins(2, 2, 2, 2)
            h_layout.addWidget(icon_label)
            h_layout.addWidget(text_label)
            item_widget.setLayout(h_layout)

            layout.addWidget(item_widget)

        layout.addStretch(1)

    def show_classification_result(self):
        if not hasattr(self, "class_map"):
            return

        # Palette : 0=substrat gris, 1=violet, 2=jaune, 3=orange
        colors = {
            0: (128, 128, 128),  # Substrate (gris)
            3: (211, 0, 148),  # MGP (violet)
            2: (0, 255, 255),  # CC (jaune)
            1: (0, 165, 255)  # NCC (orange)
        }

        h, w = self.class_map.shape
        result_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for val, col in colors.items():
            result_rgb[self.class_map == val] = col

        self.viewer_right.setImage(self._np2pixmap(result_rgb))

        # Overlay sur image originale
        overlay = cv2.addWeighted(self.rgb_image, 1 - self.alpha, result_rgb, self.alpha, 0)
        self.viewer_left.setImage(self._np2pixmap(overlay))

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

    def _init_classification_table(self):
        tw = self.tableWidget_classificationList
        tw.setColumnCount(5)
        tw.setHorizontalHeaderLabels(["Model", "Kind", "Status", "Progress", "Duration"])
        tw.setSelectionBehavior(tw.SelectRows)
        tw.setSelectionMode(tw.SingleSelection)
        tw.setEditTriggers(tw.NoEditTriggers)
        tw.verticalHeader().setVisible(False)
        header = tw.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Model
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Kind
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Status
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Progress
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Duration

    def _kind_summary(self, labels: List[str], max_chars: int = 32) -> str:
        s = "/".join(labels)
        return s if len(s) <= max_chars else s[:max_chars - 1] + "…"

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
            print(f'[REFRESH TABLE] : row : {row}, name : {name}')
            job = self.jobs[name]
            tw.insertRow(row)
            # Model
            it_model = QTableWidgetItem(job.name)
            it_model.setToolTip(job.name)
            tw.setItem(row, 0, it_model)
            # Kind
            kind_text = self._kind_summary(job.kind)
            it_kind = QTableWidgetItem(kind_text)
            it_kind.setToolTip(", ".join(job.kind))
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

    def add_job(self, name: str):
        clf_type=name
        # enforce uniqueness on 'name'
        unique = self._ensure_unique_name(name)

        if unique is None:
            return

        if 'sub' in unique:
            kind=['Substrate']
        else:
            kind = ['Ink 3 classes']

        job = ClassificationJob(unique, clf_type, kind)
        self.jobs[unique] = job
        self.job_order.append(unique)
        self._refresh_table()
        self._refresh_show_model_combo()

    def remove_selected_job(self):
        table = self.tableWidget_classificationList
        row = table.currentRow()
        if row < 0:
            return

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
        print(f'[START QUEUE] : job order {self.job_order}')
        if not self.job_order:
            QMessageBox.information(self, "Empty", "No jobs in the queue.")
            return
        if self._current_worker is not None:
            QMessageBox.information(self, "Busy", "A job is already running.")
            return

        self._stop_all = False
        self._skip_done_on_run = False

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

        self._stop_all = False
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

    def _launch_next_job(self):
        """Launch the next job in the queue if available."""
        if self._stop_all:
            self._current_worker = None
            self._running_idx = -1
            return

        while 0 <= self._running_idx < len(self.job_order):
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
            self.load_classifier(job.clf_type)

            mask = self.binary_map.astype(bool)
            spectra = self.data[mask, :]
            N = spectra.shape[0]
            target_steps = 50
            min_chunk = 50
            max_chunk = 200000
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
            if job.status == "Running":
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
        self.show_classification_result()  # reuse your existing renderer
        self._refresh_show_model_combo(select_name=name)

    def _on_job_partial(self, name: str, start: int, end: int, preds_chunk: np.ndarray):
        # Todo : afficher chunk par chuk l'image
        pass

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

    def _advance_and_launch_next(self):
        if self._stop_all:
            self._running_idx = -1
            return

        self._running_idx += 1
        if self._running_idx < len(self.job_order):
            self._launch_next_job()
        else:
            self._running_idx = -1  # finished all
            # (optionnel) self.label_status.setText("All jobs done")

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
    data,wl=fused_cube(Hypercube(filepath1,load_init=True),Hypercube(filepath2,load_init=True))
    cube=Hypercube(data=data,wl=wl)
    w.load_cube(cube=cube)

    sys.exit(app.exec_())
