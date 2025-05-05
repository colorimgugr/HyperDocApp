# cd C:\Users\Usuario\Documents\GitHub\Hypertool\registration
# python -m PyQt5.uic.pyuic -o registration_window.py registration_window.ui

import sys
import numpy as np
import cv2
from IPython.core.display_functions import update_display
from PyQt5.QtWidgets import (
    QApplication, QWidget,QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QComboBox, QDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QRubberBand
)
from PyQt5.QtGui import QPixmap, QImage, QTransform,  QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF, QRect, QPoint,QSize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import random

# TODO : Manual outling features

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
    # TODO: add file title above image
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None  # check if image loaded

        # Rubber band selection
        self.origin = QPoint()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._selecting = False
        self.last_rect_item = None
        self.rect_coords = None

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

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.pixmap_item:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self._selecting = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selecting:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.pixmap_item and self._selecting:
            self.rubber_band.hide()
            self._selecting = False

            # Convert view coords to scene coords
            start_scene = self.mapToScene(self.origin)
            end_scene = self.mapToScene(event.pos())

            # Convert scene coords to pixmap (image) coords
            p1 = self.pixmap_item.mapFromScene(start_scene)
            p2 = self.pixmap_item.mapFromScene(end_scene)

            x1, y1 = int(p1.x()), int(p1.y())
            x2, y2 = int(p2.x()), int(p2.y())


            # Limite aux bords de l'image
            w, h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
            if all(0 <= v < w for v in [x1, x2]) and all(0 <= v < h for v in [y1, y2]):

                # Erase previous rectangle
                self.clear_rectangle()

                # add new rectangle
                # Crée un rectangle normalisé avec les coins (x1, y1) et (x2, y2)
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                width = x_max - x_min
                height = y_max - y_min

                if width<2 or height<2:
                    self.rect_coords=None
                    self.last_rect_item = None
                    return

                self.rect_coords = [x_min, y_min, width, height]
                self.last_rect_item = self.scene().addRect(x_min, y_min, width, height, QPen(QColor("red")))

        super().mouseReleaseEvent(event)

    def get_rect_coords(self):
        return self.rect_coords

    def clear_rectangle(self):
        if self.last_rect_item:
            try:
                self.scene().removeItem(self.last_rect_item)
            except:
                None
        self.rect_coords=None

class RegistrationApp(QMainWindow, Ui_MainWindow):
    # TODO: show number of features found
    # TODO : save cube with option of minicubes (selected zone)
    def __init__(self):
        super().__init__()
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

        self.manual_feature_modif=False #to see if manual have been made in features selection
        self.selected_zone=[0,0]
        self.selected_rect_coords=None

        self.cube=[self.fixed_cube,self.moving_cube]
        self.img=[self.fixed_img,self.moving_img]
        self.radioButton_one=[self.radioButton_one_ref,self.radioButton_one_mov]
        self.radioButton_whole=[self.radioButton_whole_ref,self.radioButton_whole_mov]
        self.slider_channel=[self.horizontalSlider_ref_channel,self.horizontalSlider_mov_channel]
        self.spinBox_channel=[self.spinBox_ref_channel,self.spinBox_mov_channel]

        self.pushButton_open_ref_hypercube.clicked.connect(self.load_fixed)
        self.pushButton_open_mov_hypercube.clicked.connect(self.load_moving)
        self.pushButton_getFeatures.clicked.connect(self.choose_register_method)
        self.pushButton_register.clicked.connect(self.register_imageAndCube)
        self.checkBox_crop.clicked.connect(self.check_selected_zones)
        self.pushButton_save_cube.clicked.connect(self.open_save_dialog)
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

    def update_images(self):

        for i_mov in [0,1]:
            cube=self.cube[i_mov].data
            if cube is not None:
                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = self.slider_channel[i_mov].value()
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
            old_fixed_cube=self.fixed_cube
            self.fixed_cube=self.moving_cube
            self.moving_cube=old_fixed_cube
            self.cube = [self.fixed_cube, self.moving_cube]

            for i_mov in range(2):
                cube=self.cube[i_mov].data
                self.slider_channel[i_mov].setMaximum(cube.shape[2] - 1)
                self.spinBox_channel[i_mov].setMaximum(cube.shape[2] - 1)

                if cube.shape[2] == 121:
                    self.slider_channel[i_mov].setValue(60)
                    self.spinBox_channel[i_mov].setValue(60)
                elif cube.shape[2] == 161:
                    self.slider_channel[i_mov].setValue(10)
                    self.spinBox_channel[i_mov].setValue(10)

                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = self.slider_channel[i_mov].value()
                img = self.cube_to_img(cube, mode, chan)
                img = (img * 256 / np.max(img)).astype('uint8')

                # else:
                #     img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

            self.viewer_img[i_mov].clear_rectangle()
            self.viewer_img[i_mov].clear_rectangle()

            if i_mov:
                self.moving_img = img
            else:
                self.fixed_img = img

            self.img = [self.fixed_img, self.moving_img]

            self.viewer_img[i_mov].setImage(np_to_qpixmap(img))

        else :
            if fname is None:
                fname, _ = QFileDialog.getOpenFileName(self, ['Load Fixed Cube','Load Moving Cube'][i_mov])

            if fname:
                if fname[-3:] in['mat', '.h5']:
                    if i_mov:
                        self.moving_cube.open_hyp(fname, open_dialog=False)
                        cube=self.moving_cube.data
                    else:
                        self.fixed_cube.open_hyp(fname, open_dialog=False)
                        cube=self.fixed_cube.data

                    self.cube = [self.fixed_cube, self.moving_cube]
                    self.slider_channel[i_mov].setMaximum(cube.shape[2]-1)
                    self.spinBox_channel[i_mov].setMaximum(cube.shape[2]-1)

                    if cube.shape[2]==121:
                        self.slider_channel[i_mov].setValue(60)
                        self.spinBox_channel[i_mov].setValue(60)
                    elif cube.shape[2]==161:
                        self.slider_channel[i_mov].setValue(10)
                        self.spinBox_channel[i_mov].setValue(10)

                    mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                    chan = self.slider_channel[i_mov].value()
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

        self.pushButton_register.setEnabled(False)

    def load_fixed(self):
        self.load_cube(0)

    def load_moving(self):
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

        if des1 is None or des2 is None:
            QMessageBox.warning(self, "Error", "Feature detection failed.")
            return

        self.kp1, self.kp2 = kp1, kp2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = matcher.match(des1, des2)
        self.matches_all = sorted(matches, key=lambda x: x.distance)

        self.register_imageAndCube()
        self.pushButton_register.setEnabled(True)
        self.pushButton_save_cube.setEnabled(True)

    def register_imageAndCube(self):

        # keep only %best matches
        keep_percent = self.features_slider.value() / 100
        num_keep = int(len(self.matches_all) * keep_percent)
        self.matches = self.matches_all[:num_keep]

        src_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        transform_type = self.transform_selector.currentText()
        if transform_type == "Affine":
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            self.aligned_img = cv2.warpAffine(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
            self.aligned_cube = np.zeros((self.fixed_cube.data.shape[0],self.fixed_cube.data.shape[1],self.moving_cube.data.shape[2]), dtype=np.float32)
            for k in range(self.moving_cube.data.shape[2]):
                self.aligned_cube[:,:,k] = cv2.warpAffine(self.moving_cube.data[:,:,k], matrix,
                                                  (self.fixed_img.shape[1], self.fixed_img.shape[0]))

        elif transform_type == "Perspective":
            # Check if there are enough matches to compute homography
            if len(self.matches) >= 4:
                matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            else:
                # Show a popup warning if not enough matches are found
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Registration Error")
                msg.setText("Not enough matches to compute homography.\nPlease try again with better images.")
                msg.exec_()
                matrix = None
                return
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            self.aligned_img = cv2.warpPerspective(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
            self.aligned_cube.data = np.zeros((self.fixed_cube.data.shape[0],self.fixed_cube.data.shape[1],self.moving_cube.data.shape[2]), dtype=np.float32)

            for k in range(self.moving_cube.data.shape[2]):
                self.aligned_cube.data[:,:,k] = cv2.warpPerspective(self.moving_cube.data[:,:,k], matrix,
                                                  (self.fixed_img.shape[1], self.fixed_img.shape[0]))
        else:
            QMessageBox.warning(self, "Error", "Unsupported transformation.")
            return

        self.update_display()

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

    def save_cube(self):

        save_both=False

        # Ouvre un QFileDialog pour sélectionner un dossier
        save_dir = QFileDialog.getExistingDirectory(self, "Choisir un dossier de sauvegarde")
        if not save_dir:
            return  # L'utilisateur a annulé

        y, x, dy, dx = self.viewer_aligned.get_rect_coords()
        x, y, dx, dy = int(x), int(y), int(dx), int(dy)

        # Rogner les images
        fixed_crop = self.fixed_img[y:y + dy, x:x + dx]
        aligned_crop = self.aligned_img[y:y + dy, x:x + dx]

        cv2.imwrite(os.path.join(save_dir, "fixed_crop.png"), fixed_crop)
        cv2.imwrite(os.path.join(save_dir, "aligned_crop.png"), aligned_crop)

        # Rogner et sauvegarder les cubes
        if hasattr(self, "aligned_cube"):
            aligned_cube_crop = self.aligned_cube[y:y + dy, x:x + dx, :]
            np.save(os.path.join(save_dir, "aligned_cube_crop.npy"), aligned_cube_crop)

        if hasattr(self, "fixed_cube"):
            fixed_cube_crop = self.fixed_cube[y:y + dy, x:x + dx, :]
            np.save(os.path.join(save_dir, "fixed_cube_crop.npy"), fixed_cube_crop)

        QMessageBox.information(self, "Succès", f"Images et cubes rognés sauvegardés dans:\n{save_dir}")

    def open_save_dialog(self):
        """Ouvre la dialog SaveWindow, récupère les options et déclenche la sauvegarde."""
        dlg = SaveWindow(self)
        # Affiche en modal : si OK, on récupère les options
        if dlg.exec_() == QDialog.Accepted:
            opts = dlg.get_options()
            self.save_cube_with_options(opts)

    def save_cube_with_options(self, opts):
        """
        Sauvegarde les cubes et images selon le dict opts retourné par SaveWindow.get_options().
        """

        save_path_align=None
        save_path_fixed=None
        save_both = opts['save_both']

        # 1) Choix des noms de fichier
        save_path_align, _ = QFileDialog.getSaveFileName(
            parent=None,
            caption="ALIGNED cube Save As…")
        if save_both:
            save_path_fixed, _ = QFileDialog.getSaveFileName(
                parent=None,
                caption="ALIGNED cube Save As…")

        mini_fixed_cube=Hypercube(data=self.fixed_cube.data,wl=self.fixed_cube.wl,metadata=self.fixed_cube.metadata)
        mini_align_cube=Hypercube(data=self.moving_cube.data,wl=self.moving_cube.wl,metadata=self.moving_cube.metadata)

        # Crop
        if opts['crop_cube']:
            y, x, dy, dx = self.viewer_aligned.get_rect_coords()
            y, x, dy, dx = map(int, (y, x, dy, dx))
            mini_fixed_cube.data = self.fixed_cube.data[y:y + dy, x:x + dx, :]
            mini_align_cube.data = self.aligned_cube.data[y:y + dy, x:x + dx, :]

            fixed_img = self.fixed_img[x:x + dx, y:y + dy]
            aligned_img = self.aligned_img[x:x + dx, y:y + dy]
        else:
            fixed_img = self.fixed_img
            aligned_img = self.aligned_img

        # Image
        if opts['export_images']:
            ext = opts['image_format'].lower()
            folder= os.path.dirname(save_path_align)
            name=save_path_align.split('/')[-1].split('.')[0]
            save_path_temp=folder+'/'+name
            fixed_fn = save_path_temp+ext
            folder = os.path.dirname(save_path_fixed)
            name = save_path_fixed.split('/')[-1].split('.')[0]
            save_path_temp = folder + '/' + name
            aligned_fn = save_path_temp+ext
            cv2.imwrite(aligned_fn, aligned_img)
            if save_both:
                cv2.imwrite(fixed_fn, fixed_img)


        # 4) Export cubes
        fmt = opts['cube_format']
        mini_align_cube.save_hyp(save_path_align,fmt=fmt)
        if not save_both:
            QMessageBox.information(self, "Succès", f"Cube saved as {fmt} in :\n{save_path_align}")
        if save_both:
            mini_fixed_cube.save_hyp(save_path_fixed,fmt=fmt)
            QMessageBox.information(self, "Succès", f"Cubes saved as {fmt} in :\n{save_path_align} \n{save_path_fixed} ")


    def switch_fixe_mov(self):
        self.load_cube(switch=True)

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
    app.setStyle('Fusion')

    folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    path_fixed_cube=folder_cube+'MPD41a_SWIR.mat'
    path_moving_cube=folder_cube+'MPD41a_VNIR.mat'
    window.load_cube(0,path_fixed_cube)
    window.load_cube(1,path_moving_cube)

    sys.exit(app.exec_())
