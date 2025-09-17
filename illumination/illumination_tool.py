import sys
import csv
import numpy as np
import cv2

## GUI
from PyQt5 import QtCore
from PyQt5.QtGui    import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtWidgets import ( QSplitter,
    QApplication,QSizePolicy, QGraphicsScene, QGraphicsPixmapItem,QRubberBand,QWidget, QFileDialog, QMessageBox,QInputDialog , QSplitter,QGraphicsView,QLabel,
)
from PyQt5.QtCore import Qt, QEvent, QRect, QRectF, QPoint, QSize

from PyQt5.QtWidgets import (QApplication, QSizePolicy, QSplitter,QTableWidgetItem,QHeaderView,QProgressBar,
                            QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QDialogButtonBox, QCheckBox, QScrollArea, QWidget, QFileDialog, QMessageBox
                             )

from illumination.illumination_window import Ui_IlluminationWidget
from hypercubes.hypercube import Hypercube,CubeInfoTemp
from interface.some_widget_for_interface import ZoomableGraphicsView,LoadingDialog
from scipy.interpolate import interp1d, PchipInterpolator

class IlluminationWidget(QWidget, Ui_IlluminationWidget):
    cubeLoaded = QtCore.pyqtSignal(Hypercube)
    cube_saved = QtCore.pyqtSignal(CubeInfoTemp)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.illuminants_dict = {}
        self.load_illuminants("Illuminants.csv")

        self.CMFs = np.array([
            [0.000232, 6.97e-06, 0.001086],
            [0.000415, 1.24e-05, 0.001946],
            [0.000742, 2.20e-05, 0.003486],
            [0.001368, 3.90e-05, 0.00645],
            [0.002236, 6.40e-05, 0.01055],
            [0.004243, 1.20e-04, 0.02005],
            [0.00765, 2.17e-04, 0.03621],
            [0.01431, 3.96e-04, 0.06785],
            [0.02319, 6.40e-04, 0.1102],
            [0.04351, 0.00121, 0.2074],
            [0.07763, 0.00218, 0.3713],
            [0.13438, 0.004, 0.6456],
            [0.21477, 0.0073, 1.03905],
            [0.2839, 0.0116, 1.3856],
            [0.3285, 0.01684, 1.62296],
            [0.34828, 0.023, 1.74706],
            [0.34806, 0.0298, 1.7826],
            [0.3362, 0.038, 1.77211],
            [0.3187, 0.048, 1.7441],
            [0.2908, 0.06, 1.6692],
            [0.2511, 0.0739, 1.5281],
            [0.19536, 0.09098, 1.28764],
            [0.1421, 0.1126, 1.0419],
            [0.09564, 0.13902, 0.81295],
            [0.05795, 0.1693, 0.6162],
            [0.03201, 0.20802, 0.46518],
            [0.0147, 0.2586, 0.3533],
            [0.0049, 0.323, 0.272],
            [0.0024, 0.4073, 0.2123],
            [0.0093, 0.503, 0.1582],
            [0.0291, 0.6082, 0.1117],
            [0.06327, 0.71, 0.07825],
            [0.1096, 0.7932, 0.05725],
            [0.1655, 0.862, 0.04216],
            [0.22575, 0.91485, 0.02984],
            [0.2904, 0.954, 0.0203],
            [0.3597, 0.9803, 0.0134],
            [0.43345, 0.99495, 0.00875],
            [0.51205, 1.0, 0.00575],
            [0.5945, 0.995, 0.0039],
            [0.6784, 0.9786, 0.00275],
            [0.7621, 0.952, 0.0021],
            [0.8425, 0.9154, 0.0018],
            [0.9163, 0.87, 0.00165],
            [0.9786, 0.8163, 0.0014],
            [1.0263, 0.757, 0.0011],
            [1.0567, 0.6949, 0.001],
            [1.0622, 0.631, 0.0008],
            [1.0456, 0.5668, 0.0006],
            [1.0026, 0.503, 0.00034],
            [0.9384, 0.4412, 0.00024],
            [0.85445, 0.381, 0.00019],
            [0.7514, 0.321, 0.0001],
            [0.6424, 0.265, 0.00005],
            [0.5419, 0.217, 0.00003],
            [0.4479, 0.175, 0.00002],
            [0.3608, 0.1382, 0.00001],
            [0.2835, 0.107, 0],
            [0.2187, 0.0816, 0],
            [0.1649, 0.061, 0],
            [0.1212, 0.04458, 0],
            [0.0874, 0.032, 0],
            [0.0636, 0.0232, 0],
            [0.04677, 0.017, 0],
            [0.0329, 0.01192, 0],
            [0.0227, 0.00821, 0],
            [0.01584, 0.005723, 0],
            [0.011359, 0.004102, 0],
            [0.008111, 0.002929, 0],
            [0.00579, 0.002091, 0],
            [0.004106, 0.001484, 0],
            [0.002899, 0.001047, 0],
            [0.002049, 0.00074, 0],
            [0.00144, 0.00052, 0],
            [0.001, 0.000361, 0],
            [0.00069, 0.000249, 0],
            [0.000476, 0.000172, 0],
            [0.000332, 0.00012, 0],
            [0.000235, 8.48e-05, 0],
            [0.000166, 6.00e-05, 0],
            [0.000117, 4.24e-05, 0],
            [8.31e-05, 3.00e-05, 0],
            [5.87e-05, 2.12e-05, 0],
            [4.15e-05, 1.50e-05, 0],
            [2.94e-05, 1.06e-05, 0],
            [2.07e-05, 7.47e-06, 0],
            [1.46e-05, 5.26e-06, 0],
            [1.03e-05, 3.70e-06, 0],
            [7.22e-06, 2.61e-06, 0],
            [5.09e-06, 1.84e-06, 0],
            [3.58e-06, 1.29e-06, 0],
            [2.52e-06, 9.11e-07, 0],
            [1.78e-06, 6.42e-07, 0],
            [1.25e-06, 4.52e-07, 0],
        ])

        self.comboBox_Illuminant.addItems(self.illuminants_dict.keys())
        self.comboBox_Illuminant.setCurrentText("D65")

        self.comboBox_Illuminant.currentIndexChanged.connect(self.update_image)
        self.doubleSpinBox_Gamma.valueChanged.connect(self.update_image)
        self.doubleSpinBox_D.valueChanged.connect(self.update_image)

        # Connect widget signals
        self.load_btn.clicked.connect(lambda : self.on_load_cube())

        # Layout and label inside frame_viewer
        self._replace_placeholder('frame_viewer', ZoomableGraphicsView)

        self.pushButton_save.clicked.connect(self.save_image)

        self.cube = None
        self.data = None
        self.wl =None

        self.image_rgb=None

        self.comparison_windows = []
        self.pushButton_add.clicked.connect(self.open_comparison_window)

    def load_illuminants(self, filename):
        try:
            with open(filename, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    name = row["Name"].strip()
                    value_str = row.get("Values", "").strip()
                    if value_str:
                        value = [float(v) for v in value_str.split(',')]
                    else:
                        value = None
                    self.illuminants_dict[name] = value
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado. Diccionario vacío.")
            self.illuminants_dict = {}

    def on_load_cube(self):
        self.load_cube()
        if self.cube is None:
            return
        if self.wl[0] > 400 or self.wl[-1] < 780:
            QMessageBox.warning(self, "Error", "The reflectance range must include 400–780 nm")
            self.cube = None
            self.image_rgb = None
            return
        self.rgb_from_illuminant()
        self.show_image()

    def show_image(self):
        self.frame_viewer.setImage(self.np_to_qpixmap())
        pass

    def update_image(self):
        if self.cube is None:
            return
        self.rgb_from_illuminant()
        self.show_image()

    def rgb_from_illuminant(self):
        # Que acabe la imagen en astype(np.uint8)
        selected_illuminant = self.comboBox_Illuminant.currentText()
        illuminant_spectra = self.illuminants_dict.get(selected_illuminant, [])
        gamma = self.doubleSpinBox_Gamma.value()
        D = self.doubleSpinBox_D.value()
        working_range = np.arange(400, 781, 5)

        height, width, channels = self.data.shape
        reflectance_cube_reshaped = self.data.reshape(-1, channels)
        interp_func = PchipInterpolator(self.wl, reflectance_cube_reshaped.T, extrapolate=False)
        reflectance_cube_interpolated = interp_func(working_range).T
        cube_vis = reflectance_cube_interpolated.reshape(height, width, len(working_range))

        # Interpolación de CMFs e iluminantes
        CMFs_interp = np.zeros((len(working_range), 3))
        for i in range(3):
            f = PchipInterpolator(np.arange(360, 830, 5), self.CMFs[:, i], extrapolate=False)
            CMFs_interp[:, i] = f(working_range)
        illuminant_interp = PchipInterpolator(np.arange(400, 781), illuminant_spectra, extrapolate=False)(working_range)

        sRGB = self.spectral_to_sRGB(cube_vis, gamma, illuminant_interp, CMFs_interp)

        self.image_rgb = np.clip(sRGB * 255, 0, 255).astype(np.uint8)
        self.image_rgb = self.image_rgb[...,::-1] # Pasar de RGB a BGR

        results_ciecam = {}
        XYZ_flat, XYZw = self.calculate_XYZs(cube_vis, illuminant_interp, CMFs_interp)
        results_ciecam = self.XYZtoCiecam_v2(XYZ_flat, XYZw, D)

        adapted_images = {}
        XYZ_adapted = results_ciecam
        XYZ_adapted = XYZ_adapted.reshape(height, width, 3)

        adapted_image = self.XYZ2sRGB(XYZ_adapted, gamma)

        self.image_rgb = np.clip(adapted_image * 255, 0, 255).astype(np.uint8)
        self.image_rgb = self.image_rgb[..., ::-1]  # Pasar de RGB a BGR

        #self.image_rgb = (self.data[:, :, (10, 50, 90)] * 255).astype(np.uint8)
        pass

    def XYZ2sRGB(self, XYZ, gamma=2.2):
        XYZ = np.maximum(XYZ, 0)
        Y = XYZ[:, :, 1]
        XYZ_normalized = XYZ / np.max(Y)

        hei, wid, _ = XYZ_normalized.shape
        XYZ_flat = XYZ_normalized.reshape(-1, 3)

        M = np.array([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0414],
                      [0.0557, -0.2040, 1.0570]])
        RGB = XYZ_flat @ M.T

        RGB = np.clip(RGB, 0, 1)
        sRGB = RGB ** (1 / gamma)
        sRGB = sRGB.reshape(hei, wid, 3)

        return sRGB

    def spectral_to_sRGB(self, im, gamma, illum, xyz):
        # im: [H, W, n_wl], illum: [n_wl], xyz: [n_wl,3]
        im = im / np.max(im)  # normalizar
        radiances = im * illum[None, None, :]  # aplicar iluminante
        XYZ = np.tensordot(radiances, xyz, axes=([2], [0]))  # integración sobre espectro
        sRGB = self.XYZ2sRGB(XYZ, gamma)
        return sRGB

    def calculate_XYZs(self, reflectance_cube, illuminant, cmf):
        _, _, bands = reflectance_cube.shape
        k = 100 / np.sum(illuminant * cmf[:, 1])
        XYZ_illum = np.sum(illuminant[:, None] * cmf, axis=0)
        XYZw = XYZ_illum * k

        reflectance_flat = reflectance_cube.reshape(-1, bands)
        XYZ_flat = k * reflectance_flat @ np.diag(illuminant) @ cmf
        return XYZ_flat, XYZw

    def XYZtoCiecam_v2(self, XYZ, XYZw, D):
        XwYwZw = XYZw
        Mcat = np.array([[0.7328, 0.4296, -0.1624],
                         [-0.7036, 1.6975, 0.0061],
                         [0.0030, 0.0136, 0.9834]])

        RwGwBw = Mcat @ XwYwZw
        RGB = (Mcat @ XYZ.T)

        RcGcBc = (XwYwZw[1] * D / RwGwBw + 1 - D)[:, None] * RGB
        XYZprima = np.linalg.inv(Mcat) @ RcGcBc
        return XYZprima.T

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

    def load_cube(self,cube_info=None,filepath=None,cube=None):

        if cube is None :

            if cube_info is not None:
                if filepath is None:
                    try:
                        if cube_info.filepath is not None:
                            filepath=cube_info.filepath
                    except:
                        pass
                else :
                    if filepath !=cube_info.filepath :
                        QMessageBox.warning(self, "Warning", "Path  is different from the filepath of cubeInfo")
                        return

            if not filepath :
                print('Ask path for cube')
                filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
                )
                if not filepath:
                    return

            message_progress = "[Illumination Tool] Loading cube..."
            loading = LoadingDialog(message_progress, filename=filepath, parent=self)
            loading.show()
            QApplication.processEvents()

            try :
                cube = Hypercube(filepath=filepath, cube_info= cube_info,load_init=True)
            except:
                QMessageBox.information(self,"Problem at loading","Impossible to load this cube. Please check format.")
                loading.close()
                return

            loading.close()

            self.cubeLoaded.emit(cube)  # Notify the manager

        self.cube = cube
        self.data = self.cube.data
        self.wl = self.cube.wl

        if self.wl[-1] < 1100 and self.wl[0] > 350:
            self.hyps_rgb_chan_DEFAULT = [610, 540, 435]
        elif self.wl[-1] >= 1100:
            self.hyps_rgb_chan_DEFAULT = [1605, 1205, 1005]
        else:
            mid = int(len(self.wl) / 2)
            self.hyps_rgb_chan_DEFAULT = [self.wl[0], self.wl[mid], self.wl[-1]]

    def np_to_qpixmap(self):
        img=self.image_rgb
        if len(img.shape) == 2:
            try:
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
            except:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                qimg = QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)

        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0],
                          QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).copy()

    def save_image(self):
        if not hasattr(self, "image_rgb") or self.image_rgb is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )

        if file_path:
            img = self.image_rgb[..., ::-1]
            if img.dtype != np.uint8:
                img = (255 * np.clip(img, 0, 1)).astype(np.uint8)

            height, width, channels = img.shape
            if channels == 3:
                qimg = QImage(img.tobytes(), width, height, width * 3, QImage.Format_RGB888)
            elif channels == 4:
                qimg = QImage(img.tobytes(), width, height, width * 4, QImage.Format_RGBA8888)
            else:
                QMessageBox.warning(self, "Error", "Unsupported image format.")
                return

            if not qimg.save(file_path):
                QMessageBox.warning(self, "Failed to save image.")

    def open_comparison_window(self):
        if self.cube is None:
            return

        # Crear una nueva instancia de IlluminationWidget
        new_window = IlluminationWidget()

        # Pasar el mismo cubo y datos
        new_window.cube = self.cube
        new_window.image_rgb = self.image_rgb
        new_window.data = self.data
        new_window.wl = self.wl

        # Generar la imagen inicial con los parámetros actuales
        new_window.rgb_from_illuminant()
        new_window.show()
        new_window.show_image()

        # Guardar referencia para que no se cierre automáticamente
        self.comparison_windows.append(new_window)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IlluminationWidget()
    w.show()
    #filepath=r'C:\Users\Usuario\Documents\Test/01644-VNIR-genealogies.h5'
    #w.load_cube(filepath=filepath)
    #image_rgb_temp = w.cube.get_rgb_image([10, 50, 90])
    #w.image_rgb=(image_rgb_temp*255).astype(np.uint8)
    #w.show_image()

    sys.exit(app.exec_())