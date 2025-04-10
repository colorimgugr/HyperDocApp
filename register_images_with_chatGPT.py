import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtCore import Qt, QPointF, QRectF
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def np_to_qpixmap(img):
    if len(img.shape) == 2:
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
    else:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def overlay_color_blend(fixed, aligned):
    blended = cv2.merge([
        cv2.normalize(fixed, None, 0, 255, cv2.NORM_MINMAX),
        cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX),
        np.zeros_like(fixed)
    ])
    return blended


def overlay_checkerboard(fixed, aligned, tile_size=20):
    result = np.zeros_like(fixed)
    for y in range(0, fixed.shape[0], tile_size):
        for x in range(0, fixed.shape[1], tile_size):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                result[y:y+tile_size, x:x+tile_size] = fixed[y:y+tile_size, x:x+tile_size]
            else:
                result[y:y+tile_size, x:x+tile_size] = aligned[y:y+tile_size, x:x+tile_size]
    return result

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def setImage(self, pixmap):
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom, zoom)

class RegistrationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Registration")

        self.fixed_img = None
        self.moving_img = None
        self.aligned_img = None
        self.kp1 = None
        self.kp2 = None
        self.show_features = False

        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        image_layout = QHBoxLayout()
        param_layout = QHBoxLayout()

        self.load_fixed_btn = QPushButton("Load Fixed Image")
        self.load_moving_btn = QPushButton("Load Moving Image")
        self.register_btn = QPushButton("Register")

        self.load_fixed_btn.clicked.connect(self.load_fixed)
        self.load_moving_btn.clicked.connect(self.load_moving)
        self.register_btn.clicked.connect(self.register_images)

        btn_layout.addWidget(self.load_fixed_btn)
        btn_layout.addWidget(self.load_moving_btn)
        btn_layout.addWidget(self.register_btn)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems([ "Checkerboard Overlay", "Color Overlay","Aligned Only"])
        self.mode_selector.currentIndexChanged.connect(self.update_display)

        self.method_selector = QComboBox()
        self.method_selector.addItems(["ORB", "AKAZE", "SIFT", "ECC"])

        self.transform_selector = QComboBox()
        self.transform_selector.addItems(["Affine", "Perspective"])

        param_layout.addWidget(QLabel("Method:"))
        param_layout.addWidget(self.method_selector)
        param_layout.addWidget(QLabel("Transformation:"))
        param_layout.addWidget(self.transform_selector)

        self.label_fixed = QLabel("Fixed Image")
        self.label_moving = QLabel("Moving Image")
        self.viewer_aligned = ZoomableGraphicsView()

        self.label_fixed.setAlignment(Qt.AlignCenter)
        self.label_moving.setAlignment(Qt.AlignCenter)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_fixed)
        left_layout.addWidget(self.label_moving)

        image_layout.addLayout(left_layout)
        image_layout.addWidget(self.viewer_aligned, stretch=1)

        main_layout.addLayout(btn_layout)
        main_layout.addLayout(param_layout)
        main_layout.addWidget(self.mode_selector)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

    def load_fixed(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Fixed Image')
        if fname:
            self.fixed_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.label_fixed.setPixmap(np_to_qpixmap(self.fixed_img).scaled(300, 300, Qt.KeepAspectRatio))

    def load_moving(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Moving Image')
        if fname:
            self.moving_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.label_moving.setPixmap(np_to_qpixmap(self.moving_img).scaled(300, 300, Qt.KeepAspectRatio))

    def register_images(self):
        if self.fixed_img is None or self.moving_img is None:
            QMessageBox.warning(self, "Error", "Please load both images first.")
            return

        method = self.method_selector.currentText()

        if method == "ORB":
            self.register_features(cv2.ORB_create(5000))
        elif method == "AKAZE":
            self.register_features(cv2.AKAZE_create())
        elif method == "SIFT":
            self.register_features(cv2.SIFT_create())
        elif method == "ECC":
            self.register_images_ecc()
        else:
            QMessageBox.warning(self, "Error", "Unknown method.")

    def register_features(self, detector):
        kp1, des1 = detector.detectAndCompute(self.fixed_img, None)
        kp2, des2 = detector.detectAndCompute(self.moving_img, None)

        if des1 is None or des2 is None:
            QMessageBox.warning(self, "Error", "Feature detection failed.")
            return

        self.kp1, self.kp2 = kp1, kp2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if detector != cv2.SIFT_create() else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        transform_type = self.transform_selector.currentText()
        if transform_type == "Affine":
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            self.aligned_img = cv2.warpAffine(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
        elif transform_type == "Perspective":
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            self.aligned_img = cv2.warpPerspective(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
        else:
            QMessageBox.warning(self, "Error", "Unsupported transformation.")
            return

        self.update_display()

    def register_images_ecc(self):
        try:
            fixed_f = self.fixed_img.astype(np.float32) / 255
            moving_f = self.moving_img.astype(np.float32) / 255

            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
            cc, warp_matrix = cv2.findTransformECC(fixed_f, moving_f, warp_matrix, warp_mode, criteria)

            self.aligned_img = cv2.warpAffine(self.moving_img, warp_matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "ECC Error", str(e))

    def update_display(self):
        if self.fixed_img is None or self.aligned_img is None:
            return

        display_mode = self.mode_selector.currentText()
        if display_mode == "Color Overlay":
            img = overlay_color_blend(self.fixed_img, self.aligned_img)
        elif display_mode == "Checkerboard Overlay":
            img = overlay_checkerboard(self.fixed_img, self.aligned_img)
        else:
            img = self.aligned_img

        # Display the final aligned image
        self.viewer_aligned.setImage(np_to_qpixmap(img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RegistrationApp()
    window.show()
    sys.exit(app.exec_())
