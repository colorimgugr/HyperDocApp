import sys
import os
import numpy as np
import cv2

from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget, QFileDialog, QVBoxLayout,QDialog,QDialogButtonBox, QFormLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QRectF,pyqtSignal

from minicube.extract_minicube_window import Ui_Form
from hypercubes.hypercube import Hypercube,CubeInfoTemp
from interface.some_widget_for_interface import ZoomableGraphicsView

## bloc non important warning
import warnings
warnings.filterwarnings("ignore", message="Parameters with non-lowercase names")


class MiniCubeTool(QWidget, Ui_Form):

    cube_saved = pyqtSignal(CubeInfoTemp)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cube = None
        self.data = None
        self.wl = None
        self.hyps_rgb_chan_DEFAULT = [0, 0, 0]
        self.range=None

        self.cubes_path=None # to change quickly cubes

        # Inject viewer
        self.viewer = ZoomableGraphicsView()
        layout = self.frame_image.layout()
        if layout:
            layout.addWidget(self.viewer)
        else:
            layout = QVBoxLayout(self.frame_image)
            layout.addWidget(self.viewer)

        # Connect load and previous next
        self.pushButton_open_hypercube.clicked.connect(self.load_cube)
        self.pushButton_next_cube.clicked.connect(lambda: self.change_hyp_quick(-1))
        self.pushButton_prev_cube.clicked.connect(lambda: self.change_hyp_quick(+1))

        # Connect RGB controls
        for slider, spin in zip(
            [self.horizontalSlider_red_channel, self.horizontalSlider_green_channel, self.horizontalSlider_blue_channel],
            [self.spinBox_red_channel, self.spinBox_green_channel, self.spinBox_blue_channel]
        ):
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(self.update_image)
            spin.valueChanged.connect(self.update_image)

        # color mode radiobuttons connect
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)

        # transform connect
        self.pushButton_rotate.clicked.connect(lambda: self.transform(np.rot90))
        self.pushButton_flip_h.clicked.connect(lambda: self.transform(np.fliplr))
        self.pushButton_flip_v.clicked.connect(lambda: self.transform(np.flipud))

        #save
        self.pushButton_save_minicube.clicked.connect(self.save_selected_minicube)

    def change_hyp_quick(self, prev_next):

        if not self.cubes_path:
            return

        last_hyp_path = self.cubes_path
        init_dir_hyp = os.path.dirname(last_hyp_path)
        file_init = os.path.basename(last_hyp_path)
        last_num = file_init.split('-')[0]
        if "." in last_num:
            return

        files = sorted(os.listdir(init_dir_hyp))
        index_init = files.index(file_init)

        file_new = files[(index_init + prev_next) % len(files)]
        i = prev_next
        allowed_ext = ('.h5', '.mat', '.hdr')
        while (last_num in file_new or not file_new.lower().endswith(allowed_ext)):
            i += prev_next
            file_new = files[(index_init + i) % len(files)]

        file_hyp = init_dir_hyp + '/' + file_new

        try:
            self.load_cube(filepath=file_hyp)
        except:
            pass

    def load_cube(self,filepath=None,cube_info=None,cube=None):

        if not cube:
            if not filepath:
                if cube_info:
                    try:
                        filepath=cube_info.filepath
                    except:
                        pass
                filepath, _ = QFileDialog.getOpenFileName(self, "Open Cube", "", "Hypercube (*.h5 *.mat *.hdr)")
                if not filepath :
                    return


            self.cube = Hypercube(filepath=filepath, load_init=True,cube_info=cube_info)

        else:
            self.cube=cube

        self.data = self.cube.data
        self.wl = self.cube.wl
        self.cubes_path=filepath

        # Choix des canaux par défaut
        if self.wl[-1] < 1100 and not self.wl[0] > 435:
            self.hyps_rgb_chan_DEFAULT = [610, 540, 435]
            self.range='-VNIR-'
        elif self.wl[-1] >= 1100 and not self.wl[0]> 1005:
            self.hyps_rgb_chan_DEFAULT = [1605, 1205, 1005]
            self.range = '-SWIR-'
        else:
            mid = int(len(self.wl) / 2)
            self.hyps_rgb_chan_DEFAULT = [int(self.wl[0]), int(self.wl[mid]), int(self.wl[-1])]
            self.range = '-'

        self.modif_sliders(force_defaults=True)
        self.label_path_cube.setText(filepath)

    def transform(self, trans_type):

        try:
            self.data = trans_type(self.data)

        except Exception as e:
            print("[transform] Failed on data:", e)
            return

        for attr in ['cls_map', 'selection_mask_map']:
            try:
                arr = getattr(self, attr, None)
                if arr is not None:
                    setattr(self, attr, trans_type(arr))
            except Exception as e:
                print(f"[transform] Failed on {attr}:", e)

        self.update_image()

    def modif_sliders(self,force_defaults=False):
        if self.wl is None:
            return

        min_wl = int(self.wl[0])
        max_wl = int(self.wl[-1])
        step = int(self.wl[1] - self.wl[0])

        default = self.radioButton_rgb_default.isChecked()

        labels = [self.label_red_channel, self.label_green_channel, self.label_blue_channel]
        if self.radioButton_grayscale.isChecked():
            labels[0].setText("")
            labels[1].setText("")
            labels[2].setText("Gray")
        else:
            labels[0].setText("Red")
            labels[1].setText("Green")
            labels[2].setText("Blue")

        spinboxes = [self.spinBox_red_channel, self.spinBox_green_channel, self.spinBox_blue_channel]
        sliders = [self.horizontalSlider_red_channel, self.horizontalSlider_green_channel, self.horizontalSlider_blue_channel]

        for i in range(3):
            sliders[i].setMinimum(min_wl)
            sliders[i].setMaximum(max_wl)
            sliders[i].setSingleStep(step)

            spinboxes[i].setMinimum(min_wl)
            spinboxes[i].setMaximum(max_wl)
            spinboxes[i].setSingleStep(step)

            if default or force_defaults:
                value = self.hyps_rgb_chan_DEFAULT[i]
            else:
                value = spinboxes[i].value()

            sliders[i].setValue(value)
            spinboxes[i].setValue(value)

            enabled = (
                (not default and not self.radioButton_grayscale.isChecked()) or
                (self.radioButton_grayscale.isChecked() and i == 2)
            )

            sliders[i].setEnabled(enabled)
            spinboxes[i].setEnabled(enabled)

        self.update_image()

    def _np2pixmap(self, img):
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

    def update_image(self):
        if self.data is None or self.wl is None:
            return

        wavelengths = [
            self.spinBox_red_channel.value(),
            self.spinBox_green_channel.value(),
            self.spinBox_blue_channel.value()
        ]
        idx = [np.argmin(np.abs(self.wl - wl)) for wl in wavelengths]

        if self.radioButton_grayscale.isChecked():
            idx = [idx[2]] * 3

        rgb = self.data[:, :, idx]
        if np.max(rgb) > 0:
            rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)
        else:
            rgb = np.zeros_like(rgb, dtype=np.uint8)

        pixmap = self._np2pixmap(rgb)
        self.viewer.setImage(pixmap)

        coords = self.viewer.get_rect_coords()
        if coords is not None:
            x, y, w, h = coords
            rect_f = QRectF(x, y, w, h)
            self.viewer.add_selection_overlay(rect_f,surface=False)

    def show_metadata_dialog(self, minicube):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Minicube Metadata")
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        name_edit = QLineEdit()
        parent_edit = QLineEdit()
        cubeinfo_edit = QLineEdit()
        number_edit = QLineEdit()

        # Prédéfini avec valeurs actuelles
        parent_edit.setText(minicube.cube_info.metadata_temp.get('parent_cube', ''))
        cubeinfo_edit.setText(minicube.cube_info.metadata_temp.get('cubeinfo', ''))
        number_edit.setText(minicube.cube_info.metadata_temp.get('number', ''))

        form_layout.addRow("Name:", name_edit)
        form_layout.addRow("Parent Cube:", parent_edit)
        form_layout.addRow("Cube Info:", cubeinfo_edit)
        form_layout.addRow("Number:", number_edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_accept():
            minicube.cube_info.metadata_temp['name'] = name_edit.text().strip()
            minicube.cube_info.metadata_temp['parent_cube'] = parent_edit.text().strip()
            minicube.cube_info.metadata_temp['cubeinfo'] = cubeinfo_edit.text().strip()
            minicube.cube_info.metadata_temp['number'] = number_edit.text().strip()
            dialog.accept()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)

        dialog.setLayout(layout)
        return dialog.exec_() == QDialog.Accepted

    def save_selected_minicube(self):
        if self.data is None:
            QMessageBox.warning(self, "No cube", "No cube loaded.")
            return

        rect = self.viewer.get_rect_coords()
        if rect is None:
            QMessageBox.warning(self, "No selection", "Please select a region first.")
            return

        y, x, dy, dx = map(int, rect)
        minicube_data = self.data[x:x + dx, y:y + dy, :]

        minicube = Hypercube(
            data=minicube_data,
            wl=self.wl,
            metadata=self.cube.cube_info.metadata_temp,
            cube_info=self.cube.cube_info
        )

        minicube.cube_info.metadata_temp['position'] = [y, x, dy, dx]
        try:
            name=minicube.cube_info.metadata_temp['parent_cube']
        except:
            name = os.path.basename(self.cube.cube_info.filepath)

        minicube.cube_info.metadata_temp['parent_cube'] = name

        self.show_metadata_dialog(minicube)

        # ask path
        default_folder = os.path.dirname(self.cube.cube_info.filepath)
        number=minicube.cube_info.metadata_temp['number']
        if not number:
            number='00'
        default_name=number+self.range
        default_path=os.path.join(default_folder,default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save minicube", default_path, "HDF5 (*.h5);;MATLAB (*.mat);;ENVI (*.hdr)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == '.mat':
            fmt = 'MATLAB'
        elif ext == '.hdr':
            fmt = 'ENVI'
        else:
            fmt = 'HDF5'

        minicube.save(path, fmt=fmt,meta_from_cube_info=True)
        QMessageBox.information(self, "Saved", f"Minicube saved to:\n{path}")
        minicube.cube_info.filepath=path
        self.cube_saved.emit(minicube.cube_info)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MiniCubeTool()
    window.show()

    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\Samples\minicubes/'
    fname = '00189-VNIR-mock-up.h5'
    import os
    filepath_main = os.path.join(folder, fname)

    window.load_cube(filepath_main)

    sys.exit(app.exec_())
