# cd C:\Users\Usuario\Documents\GitHub\Hypertool\registration
# python -m PyQt5.uic.pyuic -o registration_window.py registration_window.ui
# pyinstaller --noconsole --exclude-module tensorflow --exclude-module torch --icon="registration_icon.ico"   register_tool.py

import sys
import os
import re
import warnings

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QSpinBox, QProgressBar,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QComboBox, QDialog, QLineEdit,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand, QFormLayout, QDialogButtonBox
)
from PyQt5.QtGui import QPixmap, QImage, QTransform, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF, QRect, QPoint, QSize, pyqtSignal, QStandardPaths
from PyQt5 import QtCore

warnings.filterwarnings("ignore", category=DeprecationWarning)

from registration.registration_window import *
from registration.save_window_register_tool import *
from hypercubes.hypercube import *
from interface.some_widget_for_interface import *

# TODO : Manual outling features

from registration.registration_window import*
from hypercubes.hypercube import*
from interface.some_widget_for_interface import LoadingDialog

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

def find_paired_cube_path(current_path):
    """
    À partir du chemin d'un cube VNIR ou SWIR, essaie de trouver son homologue.
    Retourne le chemin du fichier si trouvé, sinon None.
    """
    if not current_path:
        return None

    dirname, basename = os.path.split(current_path)

    if "SWIR" in basename:
        alt_name = basename.replace("SWIR", "VNIR")
    elif "VNIR" in basename:
        alt_name = basename.replace("VNIR", "SWIR")
    else:
        return None  # Aucun tag identifiable

    alt_path = os.path.join(dirname, alt_name)
    return alt_path if os.path.exists(alt_path) else None

class RegistrationApp(QMainWindow, Ui_MainWindow):

    cubeLoaded = QtCore.pyqtSignal(Hypercube)
    cube_saved = pyqtSignal(CubeInfoTemp)

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
        self.auto_load_lock = False #to control auto_load and do not have infinite loop

        self.manual_feature_modif=False #to see if manual have been made in features selection
        self.selected_zone=[0,0]
        self.selected_rect_coords=None

        self.cube=[self.fixed_cube,self.moving_cube]
        self.img=[self.fixed_img,self.moving_img]
        self.transforms=["",""] # to follow transforms
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
        self.pushButton_reset.clicked.connect(self.reset_all)

        self.pushButton_switch_images.clicked.connect(self.switch_fixe_mov)

        #transform connect :
        self.pushButton_rotate_mov.clicked.connect(lambda: self.transform(np.rot90,1))
        self.pushButton_flip_h_mov.clicked.connect(lambda: self.transform(np.fliplr,1))
        self.pushButton_flip_v_mov.clicked.connect(lambda: self.transform(np.flipud,1))
        self.pushButton_rotate_fix.clicked.connect(lambda: self.transform(np.rot90,0))
        self.pushButton_flip_h_fix.clicked.connect(lambda: self.transform(np.fliplr,0))
        self.pushButton_flip_v_fix.clicked.connect(lambda: self.transform(np.flipud,0))
        # self.pushButton_reset_transform.clicked.connect(self.undo_all_transforms)

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

        self.viewer_aligned.middleClicked.connect(self.middle_click_on_match) #interaction for features supress
        self.viewer_aligned.moveFeatureStart.connect(self.start_move_feature) # move features
        self.viewer_aligned.moveFeatureUpdate.connect(self.update_move_feature)
        self.viewer_aligned.moveFeatureEnd.connect(self.end_move_feature)

        self.overlay_selector.currentIndexChanged.connect(self._update_viewer_aligned_tooltip)
        self._update_viewer_aligned_tooltip()
        self.overlay_selector.currentIndexChanged.connect(self._update_view_matches_interaction_mode)
        self._update_view_matches_interaction_mode()

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

        self.checkBox_auto_load_complental.setChecked(False)

    def _update_viewer_aligned_tooltip(self):
        if self.overlay_selector.currentText() == "View Matches":
            self.viewer_aligned.setToolTip(
                "Interaction in “View Matches”\n"
                "- Middle-click or CTRL+click near a match point to remove that match (by displayed index).\n"
                "- Drag a match point to manually adjust its position (left or right side) with RIGHT CLICK."
                "- ROI selection is disabled during View Matches mode"
            )
        else:
            self.viewer_aligned.setToolTip("")

    def _update_view_matches_interaction_mode(self):
        is_view_matches = (self.overlay_selector.currentText() == "View Matches")

        # Disable rectangle selection only in View Matches (right-click is used for keypoint move)
        self.viewer_aligned.enable_rect_selection = (not is_view_matches)

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

    def transform(self,trans_type,i_mov):
        data = self.cube[i_mov].data
        try:
            data=trans_type(data)
            if trans_type==np.rot90:
                self.transforms[i_mov]+='r'
            elif trans_type==np.flipud:
                self.transforms[i_mov]+='v'
            elif trans_type==np.fliplr:
                self.transforms[i_mov]+='h'

        except Exception as e:
            print("[transform] Failed on data:", e)
            return

        self.cube[i_mov].data=data
        # regenerate the image slice
        mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
        chan = np.argmin(np.abs(self.slider_channel[i_mov].value() - self.cube[i_mov].wl))
        img = self.cube_to_img(data, mode, chan)
        img = (img * 256 / np.max(img)).astype('uint8')
        self.viewer_img[i_mov].clear_rectangle()
        self.viewer_img[i_mov].setImage(np_to_qpixmap(img))
        suffixe_label = [" (fixed)", " (moving)"][i_mov]
        self.viewer_label[i_mov].setText(self.cube[i_mov].filepath.split('/')[-1] + suffixe_label)

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

    def load_cube_info(self, ci: CubeInfoTemp):
        """
        Used by the manager to inject updated CubeInfoTemp.
        Detects whether the filepath matches the fixed or moving cube.
        """
        if self.fixed_cube and self.fixed_cube.cube_info.filepath == ci.filepath:
            self.fixed_cube.cube_info=ci

        elif self.moving_cube and self.moving_cube.cube_info.filepath == ci.filepath:
            self.moving_cube.cube_info=ci

        else:
            print(f"[Warning] CubeInfo path does not match fixed or moving cube: {ci.filepath}")

    def load_cube(self,i_mov=None,filepath=None,cube_info=None,switch=False,cube=None):
        if cube is None :
            print('[Regisration : cube in argument is None]')
        else :
            print('[Regisration : cube received ]')

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

            return

        else :

            if filepath is None:
                try :
                    filepath=cube_info.filepath
                except:
                    pass
                    try :
                        filepath = cube.cube_info.filepath
                    except:
                        filepath, _ = QFileDialog.getOpenFileName(self, ['Load Fixed Cube','Load Moving Cube'][i_mov])

            if not filepath:
                return

            which_cube=['FIXED','MOVING'][i_mov]
            message_progress=  "[Register Tool] Loading "+which_cube+" cube..."
            loading = LoadingDialog(message_progress, filename=filepath, parent=self)
            loading.show()
            QApplication.processEvents()

            if filepath[-3:] in['mat', '.h5','hdr']:
                if i_mov:
                    if cube is None:
                        print('[Regisration : Moving cube is none]')
                        self.moving_cube.open_hyp(filepath, cube_info=cube_info,open_dialog=False)
                    else:
                        self.moving_cube = cube

                    data=self.moving_cube.data
                    wl=self.moving_cube.wl
                    self.cubeLoaded.emit(self.moving_cube)  # Notify the manager

                else:
                    if cube is None:
                        print('[Regisration : Moving cube is none]')
                        self.fixed_cube.open_hyp(filepath,cube_info=cube_info, open_dialog=False)
                    else:
                        self.fixed_cube=cube

                    data=self.fixed_cube.data
                    wl = self.fixed_cube.wl
                    self.cubeLoaded.emit(self.fixed_cube)  # Notify the manager

                # Auto-load paired cube if not already loaded
                paired_path=None
                if not self.auto_load_lock:
                    paired_path = find_paired_cube_path(filepath)

                    if paired_path:
                        load_fixe_auto = True
                    else:
                        print(f"[REG] Aucun cube équivalent trouvé pour : {filepath}")

                self.cube = [self.fixed_cube, self.moving_cube]

                # self.slider_channel[i_mov].setMaximum(cube.shape[2]-1)
                self.slider_channel[i_mov].setMaximum(int(np.max(wl)))
                self.slider_channel[i_mov].setMinimum(int(np.min(wl)))
                self.slider_channel[i_mov].setSingleStep(int(wl[1]-wl[0]))

                # self.spinBox_channel[i_mov].setMaximum(cube.shape[2] - 1)
                self.spinBox_channel[i_mov].setMaximum(int(np.max(wl)))
                self.spinBox_channel[i_mov].setMinimum(int(np.min(wl)))
                self.spinBox_channel[i_mov].setSingleStep(int(wl[1] - wl[0]))

                if data.shape[2]==121:
                    self.slider_channel[i_mov].setValue(750)
                    self.spinBox_channel[i_mov].setValue(750)
                elif data.shape[2]==161:
                    self.slider_channel[i_mov].setValue(1300)
                    self.spinBox_channel[i_mov].setValue(1300)

                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = np.argmin(np.abs(self.slider_channel[i_mov].value()-wl))
                img = self.cube_to_img(data, mode, chan)
                img =(img * 256 / np.max(img)).astype('uint8')
            else :
                # try :
                #     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                # except:
                #     loading.close()
                #     return
                QMessageBox.critical(self,"Invalid file","This file can not be opened by this tool.")
                loading.close()
                return

            self.viewer_img[i_mov].clear_rectangle()

            if i_mov:
                self.moving_img=img
            else:
                self.fixed_img = img

            self.img = [self.fixed_img, self.moving_img]

            self.viewer_img[i_mov].setImage(np_to_qpixmap(img))
            suffixe_label = [" (fixed)", " (moving)"][i_mov]
            self.viewer_label[i_mov].setText(self.cube[i_mov].filepath.split('/')[-1] + suffixe_label)

            loading.close()

        if not self.auto_load_lock and paired_path is not None and self.checkBox_auto_load_complental.isChecked():
            self.auto_load_lock = True
            self.load_cube(i_mov=1 - i_mov, filepath=paired_path)
            self.auto_load_lock = False

        self.pushButton_register.setEnabled(False)

    def load_fixed_btn(self):
        self.load_cube(0)

    def load_moving_btn(self):
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

        if self.matches_all is None:
            return

        self.update_slider_packet()

        keypoints_per_packet = self.spinBox_keypointPerPacket.value()
        packet_idx = self.horizontalSlider_keyPacketToShow.value()

        start_idx = packet_idx * keypoints_per_packet
        end_idx = start_idx + keypoints_per_packet
        selected_matches = self.matches_all[start_idx:end_idx]
        self.match_display_to_global_index = {
            i: start_idx + i for i in range(len(selected_matches))
        } #to keep corespondace between displayed number and number in the feature list

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

        def index_to_color(idx):
            # Une fonction simple pour faire un "arc-en-ciel" déterministe
            r = (37 * idx) % 256
            g = (97 * idx) % 256
            b = (173 * idx) % 256
            return (r, g, b)

        for i, m in enumerate(selected_matches):
            kp1 = self.kp1[m.queryIdx]
            kp2 = self.kp2[m.trainIdx]

            pt1 = tuple(np.round(kp1.pt).astype(int))
            pt2 = tuple(np.round(kp2.pt).astype(int))
            pt2_shifted = (int(pt2[0] + fixed_img_vis.shape[1]), pt2[1])  # Décalage pour image de droite

            color = index_to_color(start_idx + i)
            cv2.circle(combined, pt1, 5, color, thickness = 2)
            cv2.circle(combined, pt2_shifted, 5, color, thickness = 2)
            cv2.line(combined, pt1, pt2_shifted, color, thickness = 2)
            cv2.putText(combined, str(i), (pt1[0] + 6, pt1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(combined, str(i), (pt2_shifted[0] + 6, pt2_shifted[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

        self.currently_displayed_matches = selected_matches
        self.match_display_to_global_index = {
            i: start_idx + i for i in range(len(selected_matches))
        }

        self.viewer_aligned.setImage(np_to_qpixmap(combined))

    def update_slider_packet(self):
        # update max slider
        if self.matches_all is None:
            return

        n_features=len(self.matches_all)
        n_features_per_packet=self.spinBox_keypointPerPacket.value()
        n_packet=int(n_features/n_features_per_packet)
        # if n_features%n_features_per_packet !=0:
        #     n_packet+=1
        self.horizontalSlider_keyPacketToShow.setMaximum(n_packet)
        # update label
        packet_show=self.horizontalSlider_keyPacketToShow.value()
        feat_start=packet_show*n_features_per_packet
        feat_stop=(packet_show+1)*n_features_per_packet-1
        if feat_stop>=n_features:feat_stop=n_features
        self.label_packetToShow.setText(f"features {feat_start} to {feat_stop} / {n_features}")
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

    def _add_suffix_once(self,stem: str, suffix: str) -> str:
        """Ajoute suffix uniquement s'il n'est pas déjà en fin de nom (avant extension)."""
        return stem if re.search(re.escape(suffix) + r"$", stem) else (stem + suffix)

    def _default_out_path(self,src_path: str, *, role: str, crop: bool) -> str:
        """
        role: 'aligned' -> _reg ; 'fixed' -> _fix
        crop: ajoute _minicube
        """
        folder = os.path.dirname(src_path) if src_path else ""
        stem = os.path.splitext(os.path.basename(src_path or "cube"))[0]

        # Nettoyage léger : si des anciens suffixes trainent en fin de nom, on les retire
        stem = re.sub(r"(_reg(_minicube)?|_fix(_minicube)?)$", "", stem)

        suffix = "_reg" if role == "aligned" else "_fix"
        stem = self._add_suffix_once(stem, suffix)

        if crop:
            stem = self._add_suffix_once(stem, "_minicube")

        return os.path.join(folder, stem)

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

            if not save_both and not flag_save_aligned:

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
                elif msg_box.clickedButton() == only_aligned:
                    save_both=False

        print(f"save both : {save_both}")

        mini_fixed_cube = Hypercube(data=self.fixed_cube.data,metadata=self.fixed_cube.cube_info.metadata_temp, wl=self.fixed_cube.wl, cube_info=self.fixed_cube.cube_info)
        mini_align_cube = Hypercube(data=self.moving_cube.data, wl=self.moving_cube.wl,metadata=self.moving_cube.cube_info.metadata_temp,
                                    cube_info=self.moving_cube.cube_info)

        # 1) Choix des noms de fichier
        src_aligned = getattr(self.aligned_cube.cube_info, "filepath", None) or getattr(self.moving_cube.cube_info,
                                                                                        "filepath", None)
        default_align = self._default_out_path(src_aligned, role="aligned", crop=opts["crop_cube"])
        save_path_align, _ = QFileDialog.getSaveFileName(self, "ALIGNED cube Save As…", default_align)
        if not save_path_align:
            QMessageBox.critical(self, "Abort", "No filepath given : Save action aborted")
            return

        # --- fixed cube (si save_both) ---
        if save_both:
            src_fixed = getattr(self.fixed_cube.cube_info, "filepath", None)
            default_fixed = self._default_out_path(src_fixed, role="fixed", crop=opts["crop_cube"])
            save_path_fixed, _ = QFileDialog.getSaveFileName(self, "FIXED cube Save As…", default_fixed)
            if not save_path_fixed:
                QMessageBox.critical(self, "Abort", "No filepath given : Save action aborted")
                return

        # Crop
        if opts['crop_cube']:

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

                lineedit_reg_parent = QLineEdit()
                lineedit_reg_parent.setText(mini_align_cube.cube_info.metadata_temp['parent_cube'])
                form_layout.addRow(QLabel("parent_cube"), lineedit_reg_parent)

                if save_both:
                    form_layout.addRow(QLabel(""))
                    form_layout.addRow(QLabel("<b><div align='center'>FOR FIXED (REFERENCE) CUBE</div></b>"))
                    lineedit_fix_name = QLineEdit()
                    lineedit_fix_name.setText(os.path.basename(save_path_fixed).split('.')[0])

                    form_layout.addRow(QLabel('name'), lineedit_fix_name)

                    lineedit_fix_parent = QLineEdit()
                    lineedit_fix_parent.setText(mini_fixed_cube.cube_info.metadata_temp['parent_cube'])
                    form_layout.addRow(QLabel("parent_cube"), lineedit_fix_parent)

                form_layout.addRow(QLabel(""))

                if save_both:
                    form_layout.addRow(QLabel("<b><div align='center'>FOR BOTH CUBES</div></b>"))

                lineedit_cubeinfo = QLineEdit()
                try:
                    lineedit_cubeinfo.setText(mini_align_cube.cube_info.metadata_temp['cubeinfo'])
                except:
                    pass
                form_layout.addRow(QLabel("cubeinfo"), lineedit_cubeinfo)

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
                    mini_align_cube.cube_info.metadata_temp['cubeinfo'] = lineedit_cubeinfo.text().strip()
                    mini_align_cube.cube_info.metadata_temp['number'] = lineedit_number.text().strip()
                    mini_align_cube.cube_info.metadata_temp['parent_cube'] = lineedit_reg_parent.text().strip()
                    if save_both:
                        mini_fixed_cube.cube_info.metadata_temp['name'] = lineedit_fix_name.text().strip()
                        mini_fixed_cube.cube_info.metadata_temp['cubeinfo'] = lineedit_cubeinfo.text().strip()
                        mini_fixed_cube.cube_info.metadata_temp['number'] = lineedit_number.text().strip()
                        mini_fixed_cube.cube_info.metadata_temp['parent_cube'] = lineedit_fix_parent.text().strip()

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
            if len(save_path_align.split('.'))==1:
                if fmt == "MATLAB":
                    ext = '.mat'
                elif fmt == "HDF5":
                    ext = '.h5'
                elif fmt == "ENVI":
                    ext = '.hdr'
                else :
                    ext = '.h5'
                save_path_align+=ext

            mini_align_cube.cube_info.filepath=save_path_align
            self.cube_saved.emit(mini_align_cube.cube_info)
            print(f"mini_align_cube save {mini_align_cube.cube_info.filepath}")
            if flag_save_aligned or not opts['crop_cube']:
                self.parent_aligned_for_minicubes=mini_align_cube
                self.checkBox_autorize_modify.setChecked(False)
            if not save_both:
                QMessageBox.information(self, "Succès", f"Cube saved as {fmt} in :\n{save_path_align}")

        except :
            QMessageBox.warning(self, "Problem", f"Cube NOT SAVED as {fmt} in :\n{save_path_align}")

        if save_both:
            mini_fixed_cube.save(save_path_fixed,fmt=fmt,meta_from_cube_info=True)
            if len(save_path_fixed.split('.'))==1:
                if fmt == "MATLAB":
                    ext = '.mat'
                elif fmt == "HDF5":
                    ext = '.h5'
                elif fmt == "ENVI":
                    ext = '.hdr'
                else :
                    ext = '.h5'
                save_path_fixed+=ext
            mini_fixed_cube.cube_info.filepath=save_path_fixed
            self.cube_saved.emit(mini_fixed_cube.cube_info)
            QMessageBox.information(self, "Succès", f"Cubes saved as {fmt} in :\n{save_path_align} \n{save_path_fixed} ")

    def switch_fixe_mov(self):
        self.load_cube(switch=True)

    def middle_click_on_match(self, scene_pos):
        if self.overlay_selector.currentText() != "View Matches":
            return

        clicked_x, clicked_y = scene_pos.x(), scene_pos.y()
        img_width = self.fixed_img.shape[1]
        min_dist = 15
        closest_display_idx = None

        for i, match in enumerate(self.currently_displayed_matches):
            pt1 = np.array(self.kp1[match.queryIdx].pt)
            pt2 = np.array(self.kp2[match.trainIdx].pt) + np.array([img_width, 0])
            dist1 = np.linalg.norm(pt1 - np.array([clicked_x, clicked_y]))
            dist2 = np.linalg.norm(pt2 - np.array([clicked_x, clicked_y]))

            if dist1 < min_dist or dist2 < min_dist:
                closest_display_idx = i
                break

        if closest_display_idx is not None:
            self.dialog_remove_match(closest_display_idx)

    def dialog_remove_match(self, match_idx):
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove Match")
        layout = QVBoxLayout(dialog)

        label = QLabel("Supress feature #")
        spinbox = QSpinBox()
        spinbox.setRange(0, len(self.matches_all) - 1)
        spinbox.setValue(match_idx)

        layout.addWidget(label)
        layout.addWidget(spinbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_accept():
            display_idx = spinbox.value()
            real_idx = self.match_display_to_global_index.get(display_idx, None)
            if real_idx is not None:
                del self.matches_all[real_idx]
                self.update_keypoints_display()
                self.checkBox_autorize_modify.setChecked(False)
            dialog.accept()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)

        dialog.exec_()

    def start_move_feature(self, scene_pos):
        if self.overlay_selector.currentText() != "View Matches":
            return

        clicked_x, clicked_y = scene_pos.x(), scene_pos.y()
        img_width = self.fixed_img.shape[1]
        min_dist = 15

        for i, match in enumerate(self.currently_displayed_matches):
            pt1 = np.array(self.kp1[match.queryIdx].pt)
            pt2 = np.array(self.kp2[match.trainIdx].pt) + np.array([img_width, 0])
            if np.linalg.norm(pt1 - [clicked_x, clicked_y]) < min_dist:
                self.viewer_aligned.editing_match = (i, "left")
                return
            elif np.linalg.norm(pt2 - [clicked_x, clicked_y]) < min_dist:
                self.viewer_aligned.editing_match = (i, "right")
                return

    def update_move_feature(self, scene_pos):
        if not self.viewer_aligned.editing_match:
            return

        idx_display, side = self.viewer_aligned.editing_match
        real_idx = self.match_display_to_global_index.get(idx_display, None)
        if real_idx is None:
            return

        match = self.matches_all[real_idx]

        if side == "left":
            self.kp1[match.queryIdx].pt = (scene_pos.x(), scene_pos.y())
        elif side == "right":
            self.kp2[match.trainIdx].pt = (scene_pos.x() - self.fixed_img.shape[1], scene_pos.y())

        self.update_keypoints_display()

    def end_move_feature(self):
        self.checkBox_autorize_modify.setChecked(False)

    def reset_all(self):
        
        ans=QMessageBox.warning(self,'Reset All','If you reset you will loose all the work you have done here.\n \nAre you sure you want to reset this tool ? ', QMessageBox.Yes|QMessageBox.Cancel)
        if ans==QMessageBox.Cancel:
            return
        
        # reinit cube and images
        self.fixed_cube = Hypercube()
        self.moving_cube = Hypercube()
        self.aligned_cube = Hypercube()
        self.fixed_img = None
        self.moving_img = None
        self.aligned_img = None
        self.img = [None, None]

        # clean features and variables
        self.kp1 = None
        self.kp2 = None
        self.matches = None
        self.matches_all = None
        self.currently_displayed_matches = []
        self.match_display_to_global_index = {}
        self.manual_feature_modif = False
        self.parent_aligned_for_minicubes = None

        # Reset sliders
        for slider, spin in zip(self.slider_channel + self.spinBox_channel,
                                self.slider_channel + self.spinBox_channel):
            slider.setValue(0)
            slider.setEnabled(False)
            spin.setValue(0)
            spin.setEnabled(False)

        self.horizontalSlider_keyPacketToShow.setValue(0)
        self.features_slider.setValue(50)

        # Reset rectangles
        for viewer in self.viewer_img:
            viewer.clear_rectangle()

        # Reset images & labels
        for viewer in self.viewer_img:
            viewer.setImage(QPixmap())
        for label in self.viewer_label:
            label.setText("")

        # Reset aligned viewer
        self.viewer_aligned.setImage(QPixmap())

        # Buttons disabled les boutons
        self.pushButton_register.setEnabled(False)
        self.pushButton_save_cube.setEnabled(False)

        # checkbox to intial state
        self.checkBox_crop.setChecked(False)
        self.checkBox_autorize_modify.setChecked(True)

class SaveWindow(QDialog, Ui_Save_Window):
    """Dialog to configure saving options."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.pushButton_save_cube_final.clicked.connect(self.accept)
        self.pushButton_Cancel.clicked.connect(self.reject)

        # ----------------------------
        # Tooltips (Save dialog)
        # ----------------------------
        self.comboBox_cube_format.setToolTip(
            "Cube format\n"
            "Choose the output cube format: MATLAB / HDF5 / ENVI (as provided by your Hypercube save backend)."
        )

        self.radioButton_both_cube_save.setToolTip(
            "Save both cubes\n"
            "If enabled, saves both:\n"
            "- the aligned (registered) cube, and\n"
            "- the fixed (reference) cube\n"
            "into the same folder (separate filenames)."
        )

        self.checkBox_minicube_save.setToolTip(
            "Save as minicube (crop)\n"
            "If enabled, saves only the rectangle selected in the aligned viewer (minicube).\n"
            "If no parent aligned cube exists yet, the tool may ask you to save the whole aligned cube first "
            "for proper parent tracking."
        )

        self.checkBox_export_images.setToolTip(
            "Export preview images\n"
            "If enabled, saves an image file (same base name as the saved cube path)."
        )

        self.comboBox_image_format.setToolTip(
            "Image format\n"
            "Select the exported image format (used only if “Export preview images” is enabled)."
        )

        self.radioButton_RGB_save_image.setToolTip(
            "Export as false RGB\n"
            "If enabled, preview images are exported as false RGB using default wavelength triplets (VNIR/SWIR heuristics).\n"
            "If disabled, exports the current grayscale images."
        )

        self.pushButton_save_cube_final.setToolTip(
            "Save\n"
            "Validate options and start export."
        )

        self.pushButton_Cancel.setToolTip(
            "Cancel\n"
            "Close this dialog without saving."
        )

        self.pushButton_save_cube_final.clicked.connect(self.accept)
        self.pushButton_Cancel.clicked.connect(self.reject)

    def closeEvent(self, event):
        self.reject()
        super().closeEvent(event)

    def get_options(self):
        opts = {
            'cube_format':   self.comboBox_cube_format.currentText(),
            'save_both':     self.radioButton_both_cube_save.isChecked(),
            'crop_cube':     self.checkBox_minicube_save.isChecked(),
            'export_images': self.checkBox_export_images.isChecked(),
        }
        if opts['export_images']:
            opts['image_format']   = self.comboBox_image_format.currentText()
            opts['image_mode_rgb'] = self.radioButton_RGB_save_image.isChecked()
        else:
            opts['image_format']   = None
            opts['image_mode_rgb'] = False
        return opts

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = RegistrationApp()
    window.show()
    app.setStyle('Fusion')

    # folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    # path_fixed_cube=folder_cube+'MPD41a_SWIR.mat'
    # path_moving_cube=folder_cube+'MPD41a_VNIR.mat'
    #
    # hc_fix=Hypercube(path_fixed_cube,load_init=True)
    # hc_mov=Hypercube(path_moving_cube,load_init=True)
    #
    # window.load_cube(0,cube=hc_fix)
    # window.load_cube(1,cube=hc_mov)

    sys.exit(app.exec_())