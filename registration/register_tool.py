# cd C:\Users\Usuario\Documents\GitHub\Hypertool\registration
# python -m PyQt5.uic.pyuic -o registration_window.py registration_window.ui
# pyinstaller --noconsole --exclude-module tensorflow --exclude-module torch --icon="registration_icon.ico"   register_tool.py

import sys
from fileinput import filename
from importlib.metadata import metadata

import numpy as np
import cv2
from IPython.core.display_functions import update_display
from PyQt5.QtWidgets import (
    QApplication, QWidget,QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QComboBox, QDialog,QLineEdit,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QRubberBand,QFormLayout,QDialogButtonBox
)
from PyQt5.QtGui import QPixmap, QImage, QTransform,  QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF, QRect, QPoint,QSize,pyqtSignal,QStandardPaths
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os, uuid, tempfile
import random

# TODO : Manual outling features
# TODO : Clean Cache to to well od close and save as.
# TODO : Trier les save depuis tool et depuis main -> Metadatas a bien reflechir.
# TODO : automatic fill name for saving

from registration.registration_window import*
from hypercubes.hypercube import*

def np_to_qpixmap(img):
    if len(img.shape) == 2:
        try:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        except:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            qimg = QImage(img.tobytes(), img.shape[1], img.shape[0],img.shape[1], QImage.Format_Grayscale8)

    else:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).copy()

def overlay_color_blend(fixed, aligned):
    blended = cv2.merge([
        cv2.normalize(fixed, None, 0, 255, cv2.NORM_MINMAX),
        cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX),
        cv2.normalize(fixed, None, 0, 255, cv2.NORM_MINMAX)
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
        self.setCursor(Qt.OpenHandCursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None  # check if image loaded

        # Rubber band selection
        self.origin = QPoint()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._selecting = False
        self.last_rect_item = None
        self.rect_coords = None

    def setImage(self, pixmap):
        self.clear_rectangle()
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom, zoom)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.pixmap_item:
            self.viewport().setCursor(Qt.CrossCursor)
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self._selecting = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selecting:
            max_w = self.viewport().width() - 1
            max_h = self.viewport().height() - 1
            x = min(max(event.pos().x(), 0), max_w)
            y = min(max(event.pos().y(), 0), max_h)
            rect = QRect(self.origin, QPoint(x, y)).normalized()
            self.rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.pixmap_item and self._selecting:
            self.rubber_band.hide()
            self.viewport().setCursor(Qt.OpenHandCursor)
            self._selecting = False

            # Convert view coords to scene coords
            start_scene = self.mapToScene(self.origin)
            end_scene = self.mapToScene(event.pos())

            # Convert scene coords to pixmap (image) coords
            p1 = self.pixmap_item.mapFromScene(start_scene)
            p2 = self.pixmap_item.mapFromScene(end_scene)

            x1, y1 = int(p1.x()), int(p1.y())
            x2, y2 = int(p2.x()), int(p2.y())

            # Clamp aux bords de l'image
            w, h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Erase previous rectangle
            self.clear_rectangle()

            # Création du rectangle avec coins clampés
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            width, height = x_max - x_min, y_max - y_min

            if width < 2 or height < 2:
                self.rect_coords = None
                self.last_rect_item = None
                return

            self.rect_coords = [x_min, y_min, width, height]
            self.last_rect_item = self.scene().addRect(
                x_min, y_min, width, height, QPen(QColor("red"))
            )
        super().mouseReleaseEvent(event)

    def get_rect_coords(self):
        return self.rect_coords

    def clear_rectangle(self):
        if self.last_rect_item:
            try:
                self.scene().removeItem(self.last_rect_item)
            except:
                None
        self.last_rect_item = None
        self.rect_coords=None

class RegistrationApp(QMainWindow, Ui_MainWindow):
    # TODO: show number of features found

    alignedCubeReady = pyqtSignal(CubeInfoTemp) # send signal to main

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Image Registration")

        self.fixed_cube = Hypercube()
        self.moving_cube = Hypercube()
        self.aligned_cube= Hypercube()
        self.fixed_img = None
        self.moving_img = None
        self.aligned_img = None
        self.kp1 = None # features positions
        self.kp2 = None
        self.show_features = False #show features in small images
        self.matches= None # only %selected matches
        self.matches_all=None #all matches list
        self.parent_aligned_for_minicubes=None # to use as parent_cube for mincubes extraction

        self.manual_feature_modif=False #to see if manual have been made in features selection
        self.selected_zone=[0,0]
        self.selected_rect_coords=None

        self.cube=[self.fixed_cube,self.moving_cube]
        self.img=[self.fixed_img,self.moving_img]
        self.radioButton_one=[self.radioButton_one_ref,self.radioButton_one_mov]
        self.radioButton_whole=[self.radioButton_whole_ref,self.radioButton_whole_mov]
        self.slider_channel=[self.horizontalSlider_ref_channel,self.horizontalSlider_mov_channel]
        self.spinBox_channel=[self.spinBox_ref_channel,self.spinBox_mov_channel]

        self.pushButton_open_ref_hypercube.clicked.connect(self.load_fixed_btn)
        self.pushButton_open_mov_hypercube.clicked.connect(self.load_moving_btn)
        self.pushButton_getFeatures.clicked.connect(self.choose_register_method)
        self.pushButton_register.clicked.connect(self.register_imageAndCube)
        self.checkBox_crop.clicked.connect(self.check_selected_zones)
        self.pushButton_save_cube.clicked.connect(self.open_save_dialog)
        self.pushButton_validRegistration.clicked.connect(self.valid_registration)

        self.pushButton_switch_images.clicked.connect(self.switch_fixe_mov)

        self.overlay_selector.currentIndexChanged.connect(self.update_display)

        self.viewer_aligned = ZoomableGraphicsView()
        self.right_layout.addWidget(self.viewer_aligned, stretch=1)

        self.label_fixed = QLabel("Fixed Image")
        self.left_layout.addWidget(self.label_fixed)
        self.viewer_fixed = ZoomableGraphicsView()
        self.left_layout.addWidget(self.viewer_fixed, stretch=1)
        self.label_moving = QLabel("Moving Image")
        self.left_layout.addWidget(self.label_moving)
        self.viewer_moving = ZoomableGraphicsView()
        self.left_layout.addWidget(self.viewer_moving, stretch=1)
        self.viewer_img=[self.viewer_fixed,self.viewer_moving,self.viewer_aligned]
        self.viewer_label=[self.label_fixed,self.label_moving]

        self.setLayout(self.main_layout)

        self.label_fixed.setAlignment(Qt.AlignCenter)
        self.label_moving.setAlignment(Qt.AlignCenter)

        self.horizontalSlider_ref_channel.setEnabled(False)
        self.horizontalSlider_mov_channel.setEnabled(False)
        self.spinBox_ref_channel.setEnabled(False)
        self.spinBox_mov_channel.setEnabled(False)

        self.horizontalSlider_ref_channel.valueChanged.connect(self.update_images)
        self.horizontalSlider_mov_channel.valueChanged.connect(self.update_images)

        self.radioButton_whole_ref.toggled.connect(self.update_sliders)
        self.radioButton_whole_mov.toggled.connect(self.update_sliders)

        self.spinBox_keypointPerPacket.valueChanged.connect(self.update_keypoints_display)
        self.horizontalSlider_keyPacketToShow.valueChanged.connect(self.update_keypoints_display)
        self.features_slider.valueChanged.connect(self.update_slider_packet)

    def update_sliders(self):
        if self.radioButton_whole_ref.isChecked():
            self.horizontalSlider_ref_channel.setEnabled(False)
            self.spinBox_ref_channel.setEnabled(False)
        else:
            self.horizontalSlider_ref_channel.setEnabled(True)
            self.spinBox_ref_channel.setEnabled(True)

        if self.radioButton_whole_mov.isChecked():
            self.horizontalSlider_mov_channel.setEnabled(False)
            self.spinBox_mov_channel.setEnabled(False)

        else:
            self.horizontalSlider_mov_channel.setEnabled(True)
            self.spinBox_mov_channel.setEnabled(True)

        self.update_images()

    def valid_registration(self):
        # save aligned to temp dir

        name_moving = self.moving_cube.filepath.split('/')[-1].split('.')[0]

        if self.viewer_aligned.get_rect_coords() is not None:
            valid_crop = QMessageBox.question(self, 'Croped zone', "Do you want to valid only croped zone ?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if valid_crop:
                y, x, dy, dx = self.viewer_aligned.get_rect_coords()
                y, x, dy, dx = map(int, (y, x, dy, dx))
                self.aligned_cube.cube_info.metadata_temp['position'] =[y, x, dy, dx]
                self.aligned_cube.cube_info.metadata_temp['parent_cube']=name_moving

        try :
            temp_path = os.path.join(tempfile.gettempdir(), name_moving+"_aligned_cube.h5")
            self.aligned_cube.filepath=temp_path
            self.aligned_cube.cube_info.filepath=temp_path

            self.aligned_cube.save(filepath=temp_path,fmt='HDF5')
            self.alignedCubeReady.emit(self.aligned_cube.cube_info)
            # self.pushButton_validRegistration.setEnabled(False)
        except:
            pass

    def update_images(self):

        for i_mov in [0,1]:
            cube=self.cube[i_mov].data
            wl=self.cube[i_mov].wl
            if cube is not None:
                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = np.argmin(np.abs(self.slider_channel[i_mov].value() - wl))
                img = self.cube_to_img(cube, mode, chan)
                img = (img * 256 / np.max(img)).astype('uint8')

                if i_mov:
                    self.moving_img = img
                else:
                    self.fixed_img = img
                self.img = [self.fixed_img, self.moving_img]

                # self.label_img[i_mov].setPixmap(np_to_qpixmap(img).scaled(300, 300, Qt.KeepAspectRatio))
                self.viewer_img[i_mov].setImage(np_to_qpixmap(img))

    def load_cube(self,i_mov=None,fname=None,switch=False):

        if switch:
            # 1) swap the cubes
            old_fixed_cube = self.fixed_cube
            self.fixed_cube = self.moving_cube
            self.moving_cube = old_fixed_cube
            self.cube = [self.fixed_cube, self.moving_cube]

            # 2) swap the images
            old_fixed_img = self.fixed_img
            self.fixed_img = self.moving_img
            self.moving_img = old_fixed_img
            self.img = [self.fixed_img, self.moving_img]

            # 3) update sliders & views for both fixed (idx=0) and moving (idx=1)
            for idx in range(2):
                cube_data = self.cube[idx].data
                wl=self.cube[idx].wl


                # self.slider_channel[idx].setMaximum(cube_data.shape[2] - 1)
                # self.spinBox_channel[idx].setMaximum(cube_data.shape[2] - 1)

                self.slider_channel[idx].setMaximum(int(np.max(wl)))
                self.slider_channel[idx].setMinimum(int(np.min(wl)))
                self.slider_channel[idx].setSingleStep(int(wl[1] - wl[0]))

                self.spinBox_channel[idx].setMaximum(int(np.max(wl)))
                self.spinBox_channel[idx].setMinimum(int(np.min(wl)))
                self.spinBox_channel[idx].setSingleStep(int(wl[1] - wl[0]))

                # reposition initial channel if needed
                if cube_data.shape[2] == 121:
                    self.slider_channel[idx].setValue(750)
                    self.spinBox_channel[idx].setValue(750)
                elif cube_data.shape[2] == 161:
                    self.slider_channel[idx].setValue(1300)
                    self.spinBox_channel[idx].setValue(1300)

                # regenerate the image slice
                mode = ['one', 'whole'][self.radioButton_whole[idx].isChecked()]
                chan = np.argmin(np.abs(self.slider_channel[idx].value() - wl))
                img = self.cube_to_img(cube_data, mode, chan)
                img = (img * 256 / np.max(img)).astype('uint8')

                # clear previous rectangle and display
                self.viewer_img[idx].clear_rectangle()
                self.viewer_img[idx].setImage(np_to_qpixmap(img))
                suffixe_label=[" (fixed)"," (moving)"][idx]
                self.viewer_label[idx].setText(self.cube[idx].filepath.split('/')[-1]+suffixe_label)

            return  # important : on sort de la méthode après le switch

        else :
            if fname is None:
                fname, _ = QFileDialog.getOpenFileName(self, ['Load Fixed Cube','Load Moving Cube'][i_mov])

            if fname:
                if fname[-3:] in['mat', '.h5']:
                    if i_mov:
                        self.moving_cube.open_hyp(fname, open_dialog=False)
                        cube=self.moving_cube.data
                        wl=self.moving_cube.wl
                    else:
                        self.fixed_cube.open_hyp(fname, open_dialog=False)
                        cube=self.fixed_cube.data
                        wl = self.fixed_cube.wl

                    self.cube = [self.fixed_cube, self.moving_cube]

                    # self.slider_channel[i_mov].setMaximum(cube.shape[2]-1)
                    self.slider_channel[i_mov].setMaximum(int(np.max(wl)))
                    self.slider_channel[i_mov].setMinimum(int(np.min(wl)))
                    self.slider_channel[i_mov].setSingleStep(int(wl[1]-wl[0]))

                    # self.spinBox_channel[i_mov].setMaximum(cube.shape[2] - 1)
                    self.spinBox_channel[i_mov].setMaximum(int(np.max(wl)))
                    self.spinBox_channel[i_mov].setMinimum(int(np.min(wl)))
                    self.spinBox_channel[i_mov].setSingleStep(int(wl[1] - wl[0]))

                    if cube.shape[2]==121:
                        self.slider_channel[i_mov].setValue(750)
                        self.spinBox_channel[i_mov].setValue(750)
                    elif cube.shape[2]==161:
                        self.slider_channel[i_mov].setValue(1300)
                        self.spinBox_channel[i_mov].setValue(1300)

                    mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                    chan = np.argmin(np.abs(self.slider_channel[i_mov].value()-wl))
                    img = self.cube_to_img(cube, mode, chan)
                    img =(img * 256 / np.max(img)).astype('uint8')
                else:
                    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

                self.viewer_img[i_mov].clear_rectangle()

                if i_mov:
                    self.moving_img=img
                else:
                    self.fixed_img = img

                self.img = [self.fixed_img, self.moving_img]

                self.viewer_img[i_mov].setImage(np_to_qpixmap(img))
                suffixe_label = [" (fixed)", " (moving)"][i_mov]
                self.viewer_label[i_mov].setText(self.cube[i_mov].filepath.split('/')[-1] + suffixe_label)

        self.pushButton_register.setEnabled(False)

    def load_fixed_btn(self,fname=None):
        self.load_cube(0)

    def load_moving_btn(self,fname=None):
        self.load_cube(1)

    def cube_to_img(self,cube,mode,chan):
        if mode=='whole':
            return np.mean(cube, axis=2).astype(np.float32)
        elif mode=='one':
            return cube[:,:,chan]

    def choose_register_method(self):
        if self.fixed_img is None or self.moving_img is None:
            QMessageBox.warning(self, "Error", "Please load both images first.")
            return

        method = self.method_selector.currentText()
        if method == "ORB":
            self.get_features(cv2.ORB_create(5000))
        elif method == "AKAZE":
            self.get_features(cv2.AKAZE_create())
        elif method == "SIFT":
            self.get_features(cv2.SIFT_create())
        else:
            QMessageBox.warning(self, "Error", "Unknown method.")

    def get_features(self, detector):

        if not self.checkBox_autorize_modify.isChecked():
            QMessageBox.warning(self,'Not autorized','Check Autorize modifying registered cube. \nDoing this, you will loose actual parent cube for minicube extraction.')
            return

        crop = False
        fixed = self.fixed_img
        moving = self.moving_img
        if self.checkBox_crop.isChecked():
            self.check_selected_zones()
            if self.selected_zone.count(1)==0:
                return
            else:
                crop=True
                if self.selected_zone[0]==1:
                    y, x, dy, dx=self.viewer_img[0].get_rect_coords()
                    fixed = self.fixed_img[x:x + dx, y:y + dy]
                    kp1, des1 = detector.detectAndCompute(fixed, None)
                    for kp in kp1:
                        kp.pt = (kp.pt[0] + y, kp.pt[1] + x)
                if self.selected_zone[1] == 1:
                    y, x, dy, dx = self.viewer_img[1].get_rect_coords()
                    moving = self.moving_img[x:x + dx, y:y + dy]
                    kp2, des2 = detector.detectAndCompute(moving, None)
                    for kp in kp2:
                        kp.pt = (kp.pt[0] + y, kp.pt[1] + x)

        if not crop :
            kp1, des1 = detector.detectAndCompute(fixed, None)
            kp2, des2 = detector.detectAndCompute(moving, None)

        try :
            if des1 is None or des2 is None:
                QMessageBox.warning(self, "Error", "Feature detection failed.")
                return

        except:
            QMessageBox.warning(self, "Error",
                                "Select a rectangle in both cubes or erase all.")
            return


        self.kp1, self.kp2 = kp1, kp2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = matcher.match(des1, des2)
        self.matches_all = sorted(matches, key=lambda x: x.distance)

        self.register_imageAndCube()

    def register_imageAndCube(self):

        if not self.checkBox_autorize_modify.isChecked():
            QMessageBox.warning(self,'Not autorized','Check Autorize modifying registered cube. \nDoing this, you will loose actual parent cube for minicube extraction.')
            return

        # Keep only the top percentage of matches
        keep_percent = self.features_slider.value() / 100
        num_keep = int(len(self.matches_all) * keep_percent)
        self.matches = self.matches_all[:num_keep]

        src_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        transform_type = self.transform_selector.currentText()
        registration_done=False

        if transform_type == "Affine":
            # Need at least 3 point pairs for affine
            if len(self.matches) < 3:
                QMessageBox.warning(self, "Error",
                                    "At least 3 matches are required for an affine transform.")
                return

            try:
                matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            except cv2.error as e:
                QMessageBox.warning(self, "OpenCV Error",
                                    f"Affine estimation failed:\n{e}")
                return

            # Check matrix validity
            if matrix is None or matrix.shape != (2, 3):
                QMessageBox.warning(self, "Error",
                                    "Failed to compute a valid affine matrix.")
                return

            # Ensure correct dtype for warpAffine
            matrix = matrix.astype(np.float32)

            # Warp the moving image
            self.aligned_img = cv2.warpAffine(self.moving_img, matrix,
                                              (self.fixed_img.shape[1], self.fixed_img.shape[0]))

            # Prepare an empty numpy array for the aligned cube
            h, w = self.fixed_img.shape[:2]
            depth = self.moving_cube.data.shape[2]
            aligned_arr = np.zeros((h, w, depth), dtype=np.float32)

            # Convert each slice out of the memoryview and warp it
            for k in range(depth):
                slice_k = np.asarray(self.moving_cube.data[:, :, k])
                aligned_arr[:, :, k] = cv2.warpAffine(slice_k, matrix, (w, h))

            registration_done=True

        elif transform_type == "Perspective":

            # Need at least 4 point pairs for homography

            if len(self.matches) < 4:
                QMessageBox.warning(self, "Registration Error",

                                    "At least 4 matches are required for a homography.\n"

                                    "Please try again with better images.")

                return

            try:

                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

            except cv2.error as e:

                QMessageBox.warning(self, "OpenCV Error",

                                    f"Homography estimation failed:\n{e}")

                return

            if matrix is None or matrix.shape != (3, 3):
                QMessageBox.warning(self, "Error",

                                    "Failed to compute a valid homography matrix.")

                return

            # Warp the moving image

            self.aligned_img = cv2.warpPerspective(self.moving_img, matrix,

                                                   (self.fixed_img.shape[1], self.fixed_img.shape[0]))

            # Prepare an empty numpy array for the aligned cube

            h, w = self.fixed_img.shape[:2]

            depth = self.moving_cube.data.shape[2]

            aligned_arr = np.zeros((h, w, depth), dtype=np.float32)

            # Convert each slice out of the memoryview and warp it

            for k in range(depth):
                slice_k = np.asarray(self.moving_cube.data[:, :, k])

                aligned_arr[:, :, k] = cv2.warpPerspective(slice_k, matrix, (w, h))

            registration_done=True

        else:
            QMessageBox.warning(self, "Error", "Unsupported transformation.")
            return

        if registration_done:
            # Replace aligned_cube with a proper Hypercube
            self.aligned_cube = Hypercube(data=aligned_arr,
                                          wl=self.moving_cube.wl,
                                          metadata=self.moving_cube.metadata)
            self.aligned_cube.cube_info=self.moving_cube.cube_info

        self.update_display()
        self.pushButton_register.setEnabled(True)
        self.pushButton_save_cube.setEnabled(True)
        self.pushButton_validRegistration.setEnabled(True)
        self.parent_aligned_for_minicubes = None

    def choose_register_method_ecc(self):
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
        if self.fixed_img.shape != self.aligned_img.shape:
            QMessageBox.warning(self,"Error", "Get feature and register before.")
            return

        display_mode = self.overlay_selector.currentText()
        img=None
        if display_mode == "Color":
            img = overlay_color_blend(self.fixed_img, self.aligned_img)
        elif display_mode == "Checkboard":
            img = overlay_checkerboard(self.fixed_img, self.aligned_img)
        elif display_mode == "View Matches":
            self.update_keypoints_display()
        elif display_mode == "Only aligned":
            img = self.aligned_img

        # Display the final aligned image
        if img is not None:
            self.viewer_aligned.setImage(np_to_qpixmap(img))

    def update_keypoints_display(self):

        self.update_slider_packet()

        keypoints_per_packet = self.spinBox_keypointPerPacket.value()
        packet_idx = self.horizontalSlider_keyPacketToShow.value()

        start_idx = packet_idx * keypoints_per_packet
        end_idx = start_idx + keypoints_per_packet
        selected_matches = self.matches_all[start_idx:end_idx]

        # Créer une image combinée côte à côte
        fixed_img_vis = cv2.cvtColor(self.fixed_img, cv2.COLOR_GRAY2BGR) if len(
            self.fixed_img.shape) == 2 else self.fixed_img.copy()
        moving_img_vis = cv2.cvtColor(self.moving_img, cv2.COLOR_GRAY2BGR) if len(
            self.moving_img.shape) == 2 else self.moving_img.copy()

        # S'assurer que les deux images ont la même hauteur
        max_height = max(fixed_img_vis.shape[0], moving_img_vis.shape[0])
        fixed_img_vis = cv2.copyMakeBorder(fixed_img_vis, 0, max_height - fixed_img_vis.shape[0], 0, 0,
                                           cv2.BORDER_CONSTANT)
        moving_img_vis = cv2.copyMakeBorder(moving_img_vis, 0, max_height - moving_img_vis.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT)

        combined = np.hstack((fixed_img_vis, moving_img_vis))

        for i, m in enumerate(selected_matches):
            kp1 = self.kp1[m.queryIdx]
            kp2 = self.kp2[m.trainIdx]

            pt1 = tuple(np.round(kp1.pt).astype(int))
            pt2 = tuple(np.round(kp2.pt).astype(int))
            pt2_shifted = (int(pt2[0] + fixed_img_vis.shape[1]), pt2[1])  # Décalage pour image de droite

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(combined, pt1, 5, color, thickness = 2)
            cv2.circle(combined, pt2_shifted, 5, color, thickness = 2)
            cv2.line(combined, pt1, pt2_shifted, color, thickness = 2)
            cv2.putText(combined, str(i), (pt1[0] + 6, pt1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(combined, str(i), (pt2_shifted[0] + 6, pt2_shifted[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

        self.viewer_aligned.setImage(np_to_qpixmap(combined))

    def update_slider_packet(self):
        # update max slider
        n_packet=int(len(self.matches_all)/self.spinBox_keypointPerPacket.value())
        if len(self.matches_all)%self.spinBox_keypointPerPacket.value() !=0:
            n_packet+=1
        self.horizontalSlider_keyPacketToShow.setMaximum(n_packet)
        # update label
        packet_show=self.horizontalSlider_keyPacketToShow.value()
        self.label_packetToShow.setText(f'packet {packet_show +1}/{n_packet}')
        if packet_show>n_packet*self.features_slider.value()/ 100:
            self.label_packetToShow.setStyleSheet(u"color: rgb(255, 0, 0);")
        else:
            self.label_packetToShow.setStyleSheet(u"color: rgb(0, 0, 0);")

    def check_selected_zones(self):
        if self.checkBox_crop.isChecked():
            for i in range(2):
                    self.selected_zone[i]=(self.viewer_img[i].get_rect_coords() is not None)

            n_selected_zone = self.selected_zone.count(1)

            if  n_selected_zone==0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Error")
                msg.setText("No zone selected. Please select a rectangle.")
                msg.exec_()
                self.checkBox_crop.setChecked(False)

                return

    def open_save_dialog(self):
        """Ouvre la dialog SaveWindow, récupère les options et déclenche la sauvegarde."""
        dialogWindow = SaveWindow(self)
        # Affiche en modal : si OK, on récupère les options
        if dialogWindow.exec_() == QDialog.Accepted:
            opts = dialogWindow.get_options()
            self.save_cube_with_options(opts)

    def save_cube_with_options(self, opts):
        """
        Sauvegarde les cubes et images selon le dict opts retourné par SaveWindow.get_options().
        """
        save_path_align=None
        save_path_fixed=None
        save_both = opts['save_both']

        flag_save_aligned=False #to follow if parent cube

        if opts['crop_cube']:

            if self.parent_aligned_for_minicubes is None:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("No parent cube ! ")
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setText(
                    "You choosed to keep only the selected part of the whole cube but you have not saved the whole registered cube first.\nThe minicubes would not have accurate parent cubes. \n Do you want to first save the whole cube aligned ?")

                only_whole_aligned = msg_box.addButton("Yes, Save whole aligned cube first", QMessageBox.ActionRole)
                cropped = msg_box.addButton("No, just minicubes", QMessageBox.ActionRole)
                msg_box.exec()

                if msg_box.clickedButton() == only_whole_aligned:
                    save_both = False
                    opts['crop_cube'] = False
                    flag_save_aligned = True

                if not flag_save_aligned:

                    if self.viewer_aligned.get_rect_coords() is None:
                        QMessageBox.warning(self, 'NO selected zone',
                                            'NO selected zone in the aligned cube.\nSelect first a rectangle with the right click or do not check "Croped cubes" on saving')
                        return

                    if not save_both:

                        # if croped selected but not save both -> ask if sure
                        msg_box = QMessageBox(self)
                        msg_box.setWindowTitle("Only one cube ?")
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText(
                            "You choosed to keep only the selected part of the whole cube.\nAre you sure you do not want to save both croped cubes ?")
                        only_aligned = msg_box.addButton("Yes, just keep croped aligned cube", QMessageBox.ActionRole)
                        save_both = msg_box.addButton("No, save both", QMessageBox.ActionRole)
                        msg_box.exec()

                        if msg_box.clickedButton() == save_both:
                            save_both=True

        mini_fixed_cube = Hypercube(data=self.fixed_cube.data,metadata=self.fixed_cube.cube_info.metadata_temp, wl=self.fixed_cube.wl, cube_info=self.fixed_cube.cube_info)
        mini_align_cube = Hypercube(data=self.moving_cube.data, wl=self.moving_cube.wl,metadata=self.moving_cube.cube_info.metadata_temp,
                                    cube_info=self.moving_cube.cube_info)

        # 1) Choix des noms de fichier
        file_name_align=self.aligned_cube.cube_info.filepath.split('.')[0]
        if '_reg' not in os.path.basename(file_name_align):
            file_name_align+='_reg'

        if opts['crop_cube']:
            file_name_align+='_minicube_'

        save_path_align, _ = QFileDialog.getSaveFileName(self,"ALIGNED cube Save As…",file_name_align)
        mini_align_cube.cube_info.filepath=save_path_align

        if not save_path_align:
            QMessageBox.critical(self,'Abort','No filepath given : Save action aborted')
            return

        if save_both:

            folder_name_fixed=os.path.dirname(save_path_align) #same as before
            name_fixed=os.path.basename(self.fixed_cube.cube_info.filepath).split('.')[0]
            if '_reg' not in name_fixed:
                name_fixed+='_reg'

            if opts['crop_cube']:
                name_fixed += '_minicube_'

            try:
                name_fixed+=os.path.basename(save_path_align).split('_')[-1]
            except:pass

            file_name_fixed=os.path.join(folder_name_fixed,name_fixed)

            save_path_fixed, _ = QFileDialog.getSaveFileName(self,"FIXED cube Save As…",file_name_fixed)
            mini_fixed_cube.cube_info.filepath = save_path_fixed

            if not save_path_fixed:
                QMessageBox.critical(self, 'Abort', 'No filepath given : Save action aborted')
                return

        # Crop
        if opts['crop_cube']:
            #todo : add metadata position, name, parent,cube_info

            if self.viewer_aligned.get_rect_coords() is not None:
                y, x, dy, dx = self.viewer_aligned.get_rect_coords()
                y, x, dy, dx = map(int, (y, x, dy, dx))
                mini_fixed_cube.data = self.fixed_cube.data[x:x + dx, y:y + dy, :]
                mini_align_cube.data = self.aligned_cube.data[x:x + dx, y:y + dy, :]

                fixed_img = self.fixed_img[x:x + dx, y:y + dy]
                aligned_img = self.aligned_img[x:x + dx, y:y + dy]


                ## metadata for croped fixed cube from fixed_cube
                mini_fixed_cube.cube_info.metadata_temp['position'] =[y, x, dy, dx]
                try:
                   mini_fixed_cube.cube_info.metadata_temp['parent_cube']=self.fixed_cube.cube_info.metadata_temp['name']
                except:
                   mini_fixed_cube.cube_info.metadata_temp['parent_cube'] = os.path.basename(self.fixed_cube.cube_info.filepath).split('.')[0]


                ## metadata for croped moving cube from parent_aligned_for_minicubes
                mini_align_cube.cube_info.metadata_temp['position'] = [y, x, dy, dx]
                if self.parent_aligned_for_minicubes is not None:
                    try:
                        mini_align_cube.cube_info.metadata_temp['parent_cube'] = self.parent_aligned_for_minicubes.cube_info.metadata_temp['name']
                    except:
                        mini_align_cube.cube_info.metadata_temp['parent_cube'] = os.path.basename(self.parent_aligned_for_minicubes.cube_info.filepath).split('.')[0]
                else:
                    try:
                        mini_align_cube.cube_info.metadata_temp['parent_cube'] = self.moving_cube.cube_info.metadata_temp['name']
                    except:
                        mini_align_cube.cube_info.metadata_temp['parent_cube'] = os.path.basename(self.moving_cube.cube_info.filepath).split('.')[0]

                ##open window to change name and cube info with previous value

                dialog = QDialog(self)
                dialog.setWindowTitle("Minicubes Specific Metadata")
                layout = QVBoxLayout(dialog)
                form_layout = QFormLayout()

                form_layout.addRow(QLabel("<b><div align='center'>FOR REGISTERED (ALIGNED) CUBE</div></b>"))
                lineedit_reg_name = QLineEdit()
                lineedit_reg_name.setText(os.path.basename(save_path_align).split('.')[0])
                form_layout.addRow(QLabel('name'), lineedit_reg_name)
                lineedit_reg_cubeinfo = QLineEdit()
                try: lineedit_reg_cubeinfo.setText(mini_align_cube.cube_info.metadata_temp['cubeinfo'])
                except: pass
                form_layout.addRow(QLabel("cubeinfo"), lineedit_reg_cubeinfo)

                if save_both:
                    form_layout.addRow(QLabel(""))
                    form_layout.addRow(QLabel("<b><div align='center'>FOR FIXED (REFERENCE) CUBE</div></b>"))
                    lineedit_fix_name = QLineEdit()
                    lineedit_fix_name.setText(os.path.basename(save_path_fixed).split('.')[0])

                    form_layout.addRow(QLabel('name'), lineedit_fix_name)
                    lineedit_fix_cubeinfo = QLineEdit()
                    try:
                        lineedit_fix_cubeinfo.setText(mini_fixed_cube.cube_info.metadata_temp['cubeinfo'])
                    except:
                        pass
                    form_layout.addRow(QLabel("cubeinfo"), lineedit_fix_cubeinfo)

                form_layout.addRow(QLabel(""))

                lineedit_number = QLineEdit()
                try: lineedit_number.setText(os.path.basename(save_path_fixed).split('.')[0].split('_')[-1])
                except:pass
                form_layout.addRow(QLabel("number"), lineedit_number)

                layout.addLayout(form_layout)

                # Buttons
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                layout.addWidget(buttons)

                def on_accept():

                    mini_align_cube.cube_info.metadata_temp['name'] = lineedit_reg_name.text().strip()
                    mini_fixed_cube.cube_info.metadata_temp['name'] = lineedit_fix_name.text().strip()
                    mini_align_cube.cube_info.metadata_temp['cubeinfo'] = lineedit_reg_cubeinfo.text().strip()
                    mini_fixed_cube.cube_info.metadata_temp['cubeinfo'] = lineedit_fix_cubeinfo.text().strip()
                    mini_align_cube.cube_info.metadata_temp['number'] = lineedit_number.text().strip()
                    mini_fixed_cube.cube_info.metadata_temp['number'] = lineedit_number.text().strip()

                    dialog.accept()

                buttons.accepted.connect(on_accept)
                buttons.rejected.connect(dialog.reject)

                dialog.setLayout(layout)
                dialog.exec_()


        else:
            fixed_img = self.fixed_img
            aligned_img = self.aligned_img

        if opts['image_mode_rgb']: ## si rgb on remplace les image de gris precedentes par les false rgb par default
            wl = mini_fixed_cube.wl
            if wl[-1] < 1100 and wl[0] > 350:
                wl_rgb = [610, 540, 435]
            elif wl[-1] >= 1100:
                wl_rgb = [1605, 1205, 1005]
            else:
                mid = int(len(wl) / 2)
                wl_rgb = [wl[0], wl[mid], wl[-1]]

            chan=[np.argmin(np.abs(wl-wl_col)) for wl_col in wl_rgb]
            chan.reverse()
            print(chan)

            fixed_img = (mini_fixed_cube.data[:, :, chan]*255).clip(0,255).astype(np.uint8)

            wl = mini_align_cube.wl
            if wl[-1] < 1100 and wl[0] > 350:
                wl_rgb = [610, 540, 435]
            elif wl[-1] >= 1100:
                wl_rgb = [1605, 1205, 1005]
            else:
                mid = int(len(wl) / 2)
                wl_rgb = [wl[0], wl[mid], wl[-1]]

            chan=[np.argmin(np.abs(wl-wl_col)) for wl_col in wl_rgb]
            chan.reverse()
            print(chan)

            aligned_img = (mini_align_cube.data[:, :, chan]*255).clip(0,255).astype(np.uint8)

        # Image
        if opts['export_images']:
            # on s’assure d’avoir un “.” devant l’extension
            ext = '.' + opts['image_format'].lower().lstrip('.')

            # --- pour l’image alignée ---
            folder = os.path.dirname(save_path_align)
            base = os.path.splitext(os.path.basename(save_path_align))[0]
            save_path_image = os.path.join(folder, base + ext)
            # cv2.imwrite renvoie False si ça a échoué
            if not cv2.imwrite(save_path_image, aligned_img):
                QMessageBox.warning(self, "Save Error",
                                    f"Impossible to save : {save_path_image}")

            # --- si on veut sauvegarder aussi l’image fixe ---
            if save_both:
                folder2 = os.path.dirname(save_path_fixed)
                base2 = os.path.splitext(os.path.basename(save_path_fixed))[0]
                save_path_image2 = os.path.join(folder2, base2 + ext)
                if not cv2.imwrite(save_path_image2, fixed_img):
                    QMessageBox.warning(self, "Save Error",
                                        f"Impossible to save : {save_path_image2}")

        # 4) Export cubes
        fmt = opts['cube_format']
        try :
            mini_align_cube.save(save_path_align,fmt=fmt,meta_from_cube_info=True)
            if flag_save_aligned or not opts['crop_cube']:
                self.parent_aligned_for_minicubes=mini_align_cube
                self.checkBox_autorize_modify.setChecked(False)
            if not save_both:
                QMessageBox.information(self, "Succès", f"Cube saved as {fmt} in :\n{save_path_align}")

        except :
            QMessageBox.warning(self, "Problem", f"Cube NOT SAVED as {fmt} in :\n{save_path_align}")

        if save_both:
            mini_fixed_cube.save(save_path_fixed,fmt=fmt,meta_from_cube_info=True)
            QMessageBox.information(self, "Succès", f"Cubes saved as {fmt} in :\n{save_path_align} \n{save_path_fixed} ")

    def switch_fixe_mov(self):
        self.load_cube(switch=True)

    def clean_cache(self):
        import glob
        cache = tempfile.gettempdir()
        for f in glob.glob(os.path.join(cache, "aligned_*.h5")):
            try:
                os.remove(f)
            except:
                pass

# TODO : clean different load and save function in data_viz and register and openSave

def save_cropped_registered_images(self):
    if self.fixed_img is None or self.aligned_img is None:
        QMessageBox.warning(self, "Erreur", "Les images ne sont pas prêtes.")
        return

    if not hasattr(self.viewer_aligned, 'get_rect_coords'):
        QMessageBox.warning(self, "Erreur", "Aucune sélection trouvée dans viewer_aligned.")
        return

    # Ouvre un QFileDialog pour sélectionner un dossier
    save_dir = QFileDialog.getExistingDirectory(self, "Choisir un dossier de sauvegarde")
    if not save_dir:
        return  # L'utilisateur a annulé

    y, x, dy, dx = self.viewer_aligned.get_rect_coords()
    x, y, dx, dy = int(x), int(y), int(dx), int(dy)

    # Rogner et sauvegarder les cubes
    if hasattr(self, "aligned_cube"):
        aligned_cube_crop = self.aligned_cube[y:y+dy, x:x+dx, :]
        np.save(os.path.join(save_dir, "aligned_cube_crop.npy"), aligned_cube_crop)

    if hasattr(self, "fixed_cube"):
        fixed_cube_crop = self.fixed_cube[y:y+dy, x:x+dx, :]
        np.save(os.path.join(save_dir, "fixed_cube_crop.npy"), fixed_cube_crop)

    QMessageBox.information(self, "Succès", f"Images et cubes rognés sauvegardés dans:\n{save_dir}")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = RegistrationApp()
    window.show()
    window.clean_cache()
    app.setStyle('Fusion')

    folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    path_fixed_cube=folder_cube+'MPD41a_SWIR.mat'
    path_moving_cube=folder_cube+'MPD41a_VNIR.mat'
    window.load_cube(0,path_fixed_cube)
    window.load_cube(1,path_moving_cube)

    sys.exit(app.exec_())
