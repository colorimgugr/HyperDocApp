import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QSpinBox, QLabel, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QInputDialog, QSlider
)
import cv2

from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

from hypercubes.hypercube import Hypercube
from registration.register_tool import ZoomableGraphicsView

def spectral_angle(p, m):
    # Compute Spectral Angle Mapper between spectra
    cos = np.dot(p, m) / (np.linalg.norm(p) * np.linalg.norm(m))
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)

class GroundTruthWidget(QWidget):
    """
    Widget to create ground truth segmentation of a hyperspectral cube,
    with zoomable visualization using ZoomableGraphicsView,
    dual view (original+overlay & segmentation) and transparency slider.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cube = None
        self.data = None
        self.cls_map = None
        self.samples = {}
        self.alpha = 0.4  # default overlay transparency

        # Controls
        self.load_btn = QPushButton("Charger Cube")
        self.load_btn.clicked.connect(self.load_cube)
        self.unsup_btn = QPushButton("Non-supervisée")
        self.unsup_btn.clicked.connect(lambda: self.set_mode('unsupervised'))
        self.ss_btn = QPushButton("Semi-supervisée")
        self.ss_btn.clicked.connect(lambda: self.set_mode('semi-supervised'))
        self.nclass_box = QSpinBox(); self.nclass_box.setRange(2,20); self.nclass_box.setValue(4)
        self.run_btn = QPushButton("Lancer"); self.run_btn.clicked.connect(self.run)

        # Transparency slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self.alpha * 100))
        self.slider.valueChanged.connect(self.on_alpha_change)
        self.slider_label = QLabel(f"Transparence: {int(self.alpha*100)}%")

        top = QHBoxLayout()
        for w in (self.load_btn, self.unsup_btn, self.ss_btn, QLabel("# Classes:"), self.nclass_box, self.run_btn,
                  self.slider_label, self.slider):
            top.addWidget(w)

        # Dual zoomable viewers
        self.viewer_left = ZoomableGraphicsView()
        self.viewer_right = ZoomableGraphicsView()

        view_layout = QHBoxLayout()
        view_layout.addWidget(self.viewer_left)
        view_layout.addWidget(self.viewer_right)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(view_layout)
        self.setLayout(layout)
        self.mode = 'unsupervised'

    def on_alpha_change(self, val):
        self.alpha = val / 100.0
        self.slider_label.setText(f"Transparence: {val}%")
        self.show_image()

    def load_cube(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)")
        if not path: return
        try:
            self.cube = Hypercube(filepath=path, load_init=True)
            self.data = self.cube.data
            self.show_image()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le cube: {e}")

    def set_mode(self, mode):
        self.mode = mode
        self.samples.clear()
        self.nclass_box.setEnabled(mode=='unsupervised')
        self.show_image()

    def show_image(self):
        if self.data is None:
            return
        H,W,B = self.data.shape
        idx = [B//3, B//2, 2*B//3]
        rgb = self.data[:,:,idx]
        rgb = (rgb/np.max(rgb)*255).astype(np.uint8)

        # Prepare overlayed image
        if self.cls_map is None:
            overlay_img = rgb.copy()
        else:
            seg8 = (self.cls_map.astype(np.float32)/(self.nclass_box.value()-1)*255).astype(np.uint8)
            cmap = cv2.applyColorMap(seg8, cv2.COLORMAP_JET)
            overlay_img = cv2.addWeighted(rgb, 1-self.alpha, cmap, self.alpha, 0)
        pix_left = self._np2pixmap(overlay_img)
        self.viewer_left.setImage(pix_left)

        # Prepare segmentation-only view
        if self.cls_map is None:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            pix_right = self._np2pixmap(blank)
        else:
            seg8 = (self.cls_map.astype(np.float32)/(self.nclass_box.value()-1)*255).astype(np.uint8)
            cmap = cv2.applyColorMap(seg8, cv2.COLORMAP_JET)
            pix_right = self._np2pixmap(cmap)
        self.viewer_right.setImage(pix_right)

    def run(self):
        if self.data is None:
            QMessageBox.warning(self, "Attention", "Chargez d'abord un cube !")
            return
        H,W,B = self.data.shape
        flat = self.data.reshape(-1,B)
        if self.mode=='unsupervised':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.nclass_box.value())
            labels = kmeans.fit_predict(flat)
        else:
            if not self.samples:
                QMessageBox.warning(self, "Attention", "Sélectionnez des pixels pour chaque classe !")
                return
            means = {c: np.mean(np.vstack(sp),axis=0) for c,sp in self.samples.items()}
            labels = np.zeros((H*W,),dtype=int)
            for i,px in enumerate(flat):
                angles = {c: spectral_angle(px,mu) for c,mu in means.items()}
                labels[i] = min(angles, key=angles.get)
        self.cls_map = labels.reshape(H,W)
        self.show_image()

    def mousePressEvent(self, event):
        # delegate clicks to left viewer for semi-supervised sample
        if self.mode=='semi-supervised' and self.data is not None and event.button()==Qt.LeftButton:
            pos = self.viewer_left.mapToScene(event.pos())
            x,y = int(pos.x()), int(pos.y())
            if 0<=x<self.data.shape[1] and 0<=y<self.data.shape[0]:
                lab,ok = QInputDialog.getInt(self, "Classe", "Numéro de la classe:",min=0)
                if ok:
                    self.samples.setdefault(lab,[]).append(self.data[y,x,:])
        super().mousePressEvent(event)

    def _np2pixmap(self, img):
        # Convert numpy BGR or RGB array to QPixmap
        from PyQt5.QtGui import QImage, QPixmap
        if img.ndim==2:
            fmt = QImage.Format_Grayscale8
        else:
            fmt = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        ptr = img.data
        qimg = QImage(ptr, w, h, img.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

if __name__=='__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = GroundTruthWidget()
    w.show()
    sys.exit(app.exec_())
