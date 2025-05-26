import os
import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox,QInputDialog , QSplitter
from PyQt5.QtCore import Qt, QEvent, QRect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from scipy.spatial import distance as spdist

from hypercubes.hypercube import Hypercube
from registration.register_tool import ZoomableGraphicsView
# Import the compiled UI
from ground_truth.ground_truth_window import Ui_GroundTruthWidget

# Todo : initialisation des default RGB channel comme vizualisation data
# Todo : gestion des zones de sélection pour la classification supervisée
# Todo : gestion de la selection semi-supervisée
# todo : manual correction pixel by pixel (or pixel groups)
# todo : show average spectrum (and std) in graph zone A DISPARU
#todo : finish selection : multiple pixel and lasso
# todo : semi-supervised

def spectral_angle(p, m):
    """
    Compute the Spectral Angle Mapper (SAM) between two spectra p and m.
    """
    cos = np.dot(p, m) / (np.linalg.norm(p) * np.linalg.norm(m))
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)

class GroundTruthWidget(QWidget, Ui_GroundTruthWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up UI from compiled .py
        self.setupUi(self)
        self.selecting_pixels = False

        self.selecting_pixels = False
        self.selection_mask = None
        self.class_colors = {}  # color of each class
        self._cmap = None
        n0 = self.nclass_box.value()
        self._cmap = cm.get_cmap('jet', n0)         # same jet as final segmentation


        # Replace placeholders with custom widgets
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)
        self._promote_canvas('spec_canvas', FigureCanvas)

        # Enable live spectrum tracking
        self.viewer_left.viewport().setMouseTracking(True)
        self.viewer_left.viewport().installEventFilter(self)

        # Promote spec_canvas placeholder to FigureCanvas
        self.spec_canvas_layout = self.spec_canvas.layout() if hasattr(self.spec_canvas, 'layout') else None
        self._init_spectrum_canvas()
        self.splitter.setStretchFactor(1, 1)
        self.spec_canvas.setVisible(False)

        # State variables
        self.cube = None
        self.data = None
        self.wl= None
        self.cls_map = None
        self.samples = {}
        self.alpha = self.horizontalSlider_transparency_GT.value() / 100.0
        self.mode = 'Unsupervised'
        self.hyps_rgb_chan_DEFAULT=[0,0,0] #default rgb channels (in int nm)
        self.hyps_rgb_chan=[0,0,0] #current rgb (in int nm)
        self.class_means = {} #for spectra of classe


        # Connect widget signals
        self.load_btn.clicked.connect(self.load_cube)
        self.run_btn.clicked.connect(self.run)
        self.comboBox_ClassifMode.currentIndexChanged.connect(self.set_mode)
        self.pushButton_class_selection.clicked.connect(self.start_pixel_selection)

        # RGB sliders <-> spinboxes
        self.sliders_rgb = [self.horizontalSlider_red_channel, self.horizontalSlider_green_channel,
                            self.horizontalSlider_blue_channel]
        self.spinBox_rgb = [self.spinBox_red_channel, self.spinBox_green_channel, self.spinBox_blue_channel]

        for sl, sp in zip(self.sliders_rgb,  self.spinBox_rgb):
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(self.show_image)

        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)

        # Transparency slider
        self.horizontalSlider_transparency_GT.valueChanged.connect(self.on_alpha_change)

        # Live spectrum checkbox
        self.live_cb.stateChanged.connect(self.toggle_live)

        self.distance_funcs = {
            'sqeuclidean': spdist.sqeuclidean,
            'cosine': spdist.cosine,
            'correlation': spdist.correlation,
            'canberra': spdist.canberra,
            'spectral_angle': spectral_angle
        }

    def start_pixel_selection(self):

        if len(self.samples)>0 :
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
        self.show_image()

    def modif_sliders(self):
        max_wl = int(self.wl[-1])
        min_wl = int(self.wl[0])
        wl_step = int(self.wl[1] - self.wl[0])

        default=self.radioButton_rgb_default.isChecked()

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
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        for i, element in enumerate(self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

            self.show_image()


    def _init_spectrum_canvas(self):
        placeholder = getattr(self, 'spec_canvas')
        parent = placeholder.parent()
        from PyQt5.QtWidgets import QSplitter

        # Crée le canvas
        self.spec_fig = Figure()
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')

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

        self.spec_canvas.setVisible(False)
        self.comboBox_pixel_selection_mode
    def _handle_selection(self, coords):
        """Prompt for class and store spectra of the given coordinates."""
        cls, ok = QInputDialog.getInt(
            self, "Class", "Class label number:", 0, 0, self.nclass_box.value() - 1
        )
        if not ok:
            return
        # append spectra
        if cls not in self.class_colors:
            # si _cmap n'existe pas encore, on le crée à la volée
            if self._cmap is None:
                from matplotlib import cm
                n = self.nclass_box.value()
                self._cmap = cm.get_cmap('jet', n)
            rgba = self._cmap(cls)
            r, g, b = (int(255 * rgba[i]) for i in range(3))
            self.class_colors[cls] = (b, g, r)

        for x, y in coords:
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                self.samples.setdefault(cls, []).append(self.data[y, x, :])
                self.selection_mask_map[y, x] = cls

        self.show_image()

        # ask to continue
        reply = QMessageBox.question(
            self, "Continue ?", "Continue selecting pixels?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            self.selecting_pixels = False

    def eventFilter(self, source, event):
        mode = self.comboBox_pixel_selection_mode.currentText()

        # 1) Clic souris
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = self.viewer_left.mapToScene(event.pos())
            x0, y0 = int(pos.x()), int(pos.y())
            if mode == 'pixel':
                self._handle_selection([(x0, y0)])
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

        # 2) Mouvement souris → mise à jour du cadre
        if event.type() == QEvent.MouseMove and hasattr(self, 'rubberBand'):
            self.rubberBand.setGeometry(
                QRect(self.origin, event.pos()).normalized()
            )
            return True

        # 3) Relâchement souris → calcul de la sélection
        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'rubberBand'):
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
            self._handle_selection(coords)
            del self.rubberBand
            return True

        # 4) Mouvement souris pour le live spectrum
        if event.type() == QEvent.MouseMove:
            if self.live_cb.isChecked() and self.data is not None:
                pos = self.viewer_left.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    spectrum = self.data[y, x, :]
                    self.spec_ax.clear()
                    self.spec_ax.plot(spectrum, label='Pixel')
                    # éventuellement tracé des class_means et class_stds...
                    self.spec_ax.set_title(f'Spectrum @ ({x},{y})')
                    self.spec_canvas.setVisible(True)
                    self.spec_canvas.draw()
            return False

        return super().eventFilter(source, event)

    def on_alpha_change(self, val):
        self.alpha = val / 100.0
        self.show_image()

    def toggle_live(self, state):
        if not state:
            self.spec_canvas.setVisible(False)

    def load_cube(self,path):
        if path is None :
            path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not path:
                return
        try:
            self.cube = Hypercube(filepath=path, load_init=True)
            self.data = self.cube.data
            self.wl= self.cube.wl
            if self.wl[-1]<1100 and self.wl[0]>350:
                self.hyps_rgb_chan_DEFAULT = [610, 540, 435]
            elif self.wl[-1]>=1100:
                self.hyps_rgb_chan_DEFAULT = [1605, 1205, 1005]
            else:
                mid=int(len(self.wl)/2)
                self.hyps_rgb_chan_DEFAULT = [self.wl[0], self.wl[mid], self.wl[-1]]
            H, W, _ = self.data.shape
            self.selection_mask_map = np.full((H, W), -1, dtype=int) #init mask
            self.modif_sliders()
            self.show_image()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le cube: {e}")

    def set_mode(self):
        self.mode = self.comboBox_ClassifMode.currentText()

        self.nclass_box.setEnabled(self.mode in ['Unsupervised', 'Semi-supervised'])
        self.pushButton_class_selection.setEnabled(self.mode in ['Supervised', 'Semi-supervised'])

        self.spec_canvas.setVisible(False)
        self.show_image()

    def show_image(self):
        if self.data is None:
            return
        H, W, B = self.data.shape
                # Get band indices from spinboxes for RGB
        self.hyps_rgb_chan = [self.spinBox_red_channel.value(),
               self.spinBox_green_channel.value(),
               self.spinBox_blue_channel.value()]

        idx = [np.argmin(np.abs(self.hyps_rgb_chan[j] - self.wl)) for j in range(3)]
        if self.radioButton_grayscale.isChecked():
            idx=[idx[2],idx[2],idx[2]]

        rgb = self.data[:, :, idx]
        rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)

        if self.cls_map is None:
            overlay = rgb.copy()
        else:
            seg8 = (self.cls_map.astype(np.float32) / (self.nclass_box.value() - 1) * 255).astype(np.uint8)
            cmap = cv2.applyColorMap(seg8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(rgb, 1-self.alpha, cmap, self.alpha, 0)
        self.viewer_left.setImage(self._np2pixmap(overlay))

        if self.cls_map is None:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            pix2 = self._np2pixmap(blank)
        else:
            seg8 = (self.cls_map.astype(np.float32) / (self.nclass_box.value() - 1) * 255).astype(np.uint8)
            pix2 = self._np2pixmap(cv2.applyColorMap(seg8, cv2.COLORMAP_JET))
        self.viewer_right.setImage(pix2)

        if self.selection_mask_map is not None:
            mixed = overlay.copy()
            α = self.alpha

            for cls, color in self.class_colors.items():
                mask2d = (self.selection_mask_map == cls)
                if not mask2d.any():
                    continue

                layer = np.zeros_like(overlay)
                layer[:] = color

                blended = cv2.addWeighted(overlay, 1 - α, layer, α, 0)

                mask3 = mask2d[:, :, None]
                mixed = np.where(mask3, blended, mixed)
            self.viewer_left.setImage(self._np2pixmap(mixed))
            return

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

    def compute_distance(self, u, v):
        name = self.comboBox_distance.currentText()
        fn   = self.distance_funcs.get(name, spectral_angle)
        return fn(u, v)


    def run(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Load a cube !")
            return

        flat = self.data.reshape(-1, self.data.shape[2])

        # 1) Unsupervised
        if self.mode == 'Unsupervised':
            from sklearn.cluster import KMeans
            n = self.nclass_box.value()
            kmeans = KMeans(n_clusters=n).fit(flat)
            labels = kmeans.labels_
            # stocke moyennes, écarts et colormap
            self.class_means = {i: kmeans.cluster_centers_[i] for i in range(n)}
            self.class_stds = {i: np.std(flat[labels == i], axis=0) for i in range(n)}
            self._cmap = cm.get_cmap('jet', n)

        # 2) Supervised pur
        elif self.mode == 'Supervised':
            classes = sorted(self.samples.keys())
            if not classes:
                QMessageBox.warning(self, "Warning", "Sélect at least one pixel !")
                return
            # moyennes par classe labellisée
            means = {c: np.mean(np.vstack(self.samples[c]), axis=0) for c in classes}
            # classification SAM
            labels = np.zeros(flat.shape[0], dtype=int)
            for i, pix in enumerate(flat):
                dists = {c: self.compute_distance(pix, mu) for c, mu in means.items()}
                labels[i] = min(dists, key=dists.get)
            # renumérote en 0..len(classes)-1
            cls_to_idx = {c: idx for idx, c in enumerate(classes)}
            labels = np.vectorize(cls_to_idx.get)(labels)
            # stocke pour l’affichage
            self.class_means = means
            self.class_stds = {c: np.std(np.vstack(self.samples[c]), axis=0) for c in classes}
            self._cmap = cm.get_cmap('jet', len(classes))

        # 3) Semi-supervised
        else:
            if not self.samples:
                QMessageBox.warning(self,"Warning", "Sélect at least one pixel !")
                return
            means = {c: np.mean(np.vstack(sp), axis=0) for c, sp in self.samples.items()}
            labels = np.zeros(flat.shape[0], dtype=int)
            for i, pix in enumerate(flat):
                angles = {c: spectral_angle(pix, mu) for c, mu in means.items()}
                labels[i] = min(angles, key=angles.get)
            self.class_means = means
            self.class_stds = {c: np.std(np.vstack(sp), axis=0) for c, sp in self.samples.items()}
            self._cmap = cm.get_cmap('jet', len(self.class_means))

        # Final : reshape et affichage
        H, W = self.data.shape[:2]
        self.cls_map = labels.reshape(H, W)
        self.show_image()

    def _np2pixmap(self, img):
        from PyQt5.QtGui import QImage, QPixmap
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

if __name__=='__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = GroundTruthWidget()
    folder=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test/'
    file_name='00001-SWIR-mock-up.h5'
    filepath=folder+file_name
    w.load_cube(filepath)
    w.show()
    sys.exit(app.exec_())

