import os
import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtCore import Qt, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from hypercubes.hypercube import Hypercube
from registration.register_tool import ZoomableGraphicsView
# Import the compiled UI
from ground_truth.ground_truth_window import Ui_GroundTruthWidget

class GroundTruthWidget(QWidget, Ui_GroundTruthWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up UI from compiled .py
        self.setupUi(self)

        # Replace placeholder viewer_left with ZoomableGraphicsView
        left_placeholder = self.viewer_left
        left_layout = left_placeholder.parent().layout()
        idx_left = left_layout.indexOf(left_placeholder)
        left_layout.removeWidget(left_placeholder)
        left_placeholder.deleteLater()
        self.viewer_left = ZoomableGraphicsView()
        left_layout.insertWidget(idx_left, self.viewer_left)

        # Replace placeholder viewer_right
        right_placeholder = self.viewer_right
        right_layout = right_placeholder.parent().layout()
        idx_right = right_layout.indexOf(right_placeholder)
        right_layout.removeWidget(right_placeholder)
        right_placeholder.deleteLater()
        self.viewer_right = ZoomableGraphicsView()
        right_layout.insertWidget(idx_right, self.viewer_right)

        # Enable live spectrum tracking
        self.viewer_left.viewport().setMouseTracking(True)
        self.viewer_left.viewport().installEventFilter(self)

        # Promote spec_canvas placeholder to FigureCanvas
        self.spec_canvas_layout = self.spec_canvas.layout() if hasattr(self.spec_canvas, 'layout') else None
        self._init_spectrum_canvas()
        self.spec_canvas.setVisible(False)

        # State variables
        self.cube = None
        self.data = None
        self.cls_map = None
        self.samples = {}
        self.alpha = self.horizontalSlider_transparency_GT_3.value() / 100.0
        self.mode = 'unsupervised'

        # Connect widget signals
        self.load_btn.clicked.connect(self.load_cube)
        self.run_btn.clicked.connect(self.run)
        self.unsup_btn.clicked.connect(lambda: self.set_mode('unsupervised'))
        self.ss_btn.clicked.connect(lambda: self.set_mode('semi-supervised'))

        # RGB sliders <-> spinboxes
        rgb_sliders = [self.horizontalSlider_red_channel_3,
                       self.horizontalSlider_green_channel_3,
                       self.horizontalSlider_blue_channel_3]
        rgb_spinboxes = [self.spinBox_red_channel_3,
                         self.spinBox_green_channel_3,
                         self.spinBox_blue_channel_3]
        for sl, sp in zip(rgb_sliders, rgb_spinboxes):
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(self.show_image)

        # Transparency slider
        self.horizontalSlider_transparency_GT_3.valueChanged.connect(self.on_alpha_change)

        # Live spectrum checkbox
        self.live_cb.stateChanged.connect(self.toggle_live)

    def _init_spectrum_canvas(self):
        self.spec_fig = Figure(figsize=(4,2))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')
        if self.spec_canvas_layout:
            self.spec_canvas_layout.addWidget(self.spec_canvas)
        else:
            self.verticalLayout.addWidget(self.spec_canvas)

    def eventFilter(self, source, event):
        if source is self.viewer_left.viewport() and event.type() == QEvent.MouseMove:
            if self.live_cb.isChecked() and self.data is not None:
                pos = self.viewer_left.mapToScene(event.pos())
                x, y = int(pos.x()), int(pos.y())
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    spectrum = self.data[y, x, :]
                    self.spec_ax.clear()
                    self.spec_ax.plot(spectrum)
                    self.spec_ax.set_title(f'Spectrum @ ({x},{y})')
                    self.spec_canvas.setVisible(True)
                    self.spec_canvas.draw()
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
            self.show_image()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le cube: {e}")

    def set_mode(self, mode):
        self.mode = mode
        self.samples.clear()
        self.nclass_box.setEnabled(mode == 'unsupervised')
        self.spec_canvas.setVisible(False)
        self.show_image()

    def show_image(self):
        if self.data is None:
            return
        H, W, B = self.data.shape
                # Get band indices from spinboxes for RGB
        idx = [self.spinBox_red_channel_3.value(),
               self.spinBox_green_channel_3.value(),
               self.spinBox_blue_channel_3.value()]
        # Ensure indices are within valid range
        idx = [min(max(0, i), B-1) for i in idx]
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

    def run(self):
        if self.data is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord un cube !")
            return
        flat = self.data.reshape(-1, self.data.shape[2])
        if self.mode == 'unsupervised':
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=self.nclass_box.value()).fit_predict(flat)
        else:
            if not self.samples:
                QMessageBox.warning(self, "Attention", "Sélectionnez des pixels pour chaque classe !")
                return
            means = {c: np.mean(np.vstack(sp), axis=0) for c, sp in self.samples.items()}
            labels = np.zeros((flat.shape[0],), dtype=int)
            for i, pix in enumerate(flat):
                angles = {c: spectral_angle(pix, mu) for c, mu in means.items()}
                labels[i] = min(angles, key=angles.get)
        self.cls_map = labels.reshape(self.data.shape[0], self.data.shape[1])
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
