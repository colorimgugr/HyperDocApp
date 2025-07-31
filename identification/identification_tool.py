import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QSizePolicy, QMessageBox, QSplitter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from identification_window import Ui_IdentificationWidget
from hypercubes.hypercube import Hypercube
from ground_truth.ground_truth_tool import ZoomableGraphicsView

class IdentificationWidget(QWidget, Ui_IdentificationWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Remplacer placeholders par ZoomableGraphicsView
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)

        self.cube = None
        self.data = None
        self.wl = None
        self.binary_map = None
        self.alpha = self.horizontalSlider_overlay_transparency.value() / 100.0

        # Connections
        self.pushButton_load.clicked.connect(self.load_cube)
        self.pushButton_launch_bin.clicked.connect(self.launch_binarization)
        self.horizontalSlider_overlay_transparency.valueChanged.connect(self.update_alpha)
        self.comboBox_bin_algorith_choice.currentIndexChanged.connect(self.update_bin_defaults)

    # def _replace_placeholder(self, name, widget_cls, **kwargs):
    #     placeholder = getattr(self, name)
    #     parent = placeholder.parent()
    #     if parent.layout() is not None:
    #         layout = parent.layout()
    #         idx = layout.indexOf(placeholder)
    #         layout.removeWidget(placeholder)
    #         placeholder.deleteLater()
    #         widget = widget_cls(**kwargs) if kwargs else widget_cls()
    #         setattr(self, name, widget)
    #         layout.insertWidget(idx, widget)

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

    def load_cube(self):
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
        self.show_rgb_image()

    def show_rgb_image(self):
        if self.data is None:
            return
        # Valeurs RGB par défaut
        if self.wl[-1] < 1100 and self.wl[0] > 350:
            rgb_chan = [610, 540, 435]
        elif self.wl[-1] >= 1100:
            rgb_chan = [1605, 1205, 1005]
        else:
            mid = int(len(self.wl) / 2)
            rgb_chan = [self.wl[0], self.wl[mid], self.wl[-1]]
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
        self.show_binary_result()

    def show_binary_result(self):
        if self.binary_map is None:
            return
        bin_img = (self.binary_map * 255).astype(np.uint8)
        bin_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        self.viewer_right.setImage(self._np2pixmap(bin_rgb))
        self.update_overlay()

    def update_overlay(self):
        if self.rgb_image is None or self.binary_map is None:
            return
        color_mask = np.zeros_like(self.rgb_image)
        color_mask[self.binary_map] = (0, 0, 255)  # rouge
        overlay = cv2.addWeighted(self.rgb_image, 1 - self.alpha, color_mask, self.alpha, 0)
        self.viewer_left.setImage(self._np2pixmap(overlay))

    def update_alpha(self, value):
        self.alpha = value / 100.0
        self.update_overlay()

    def update_bin_defaults(self):
        algo = self.comboBox_bin_algorith_choice.currentText().lower()
        try:
            if algo == "niblack":
                self.doubleSpinBox_bin_k.setValue(-0.2)
            elif algo == "otsu":
                self.doubleSpinBox_bin_k.setValue(0.0)
            elif algo == "sauvola":
                self.doubleSpinBox_bin_k.setValue(0.4)
                self.spinBox_bin_window_size.setValue(3)
                self.comboBox_padding_mode.setCurrentText("replicate")
            elif algo == "wolf":
                self.doubleSpinBox_bin_k.setValue(0.5)
                self.spinBox_bin_window_size.setValue(3)
                self.comboBox_padding_mode.setCurrentText("reflect")
            elif algo == "bradley":
                self.doubleSpinBox_bin_k.setValue(10)
                self.spinBox_bin_window_size.setValue(15)
                self.comboBox_padding_mode.setCurrentText("replicate")
        except Exception:
            pass

    def _np2pixmap(self, img):
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IdentificationWidget()
    w.show()
    sys.exit(app.exec_())
