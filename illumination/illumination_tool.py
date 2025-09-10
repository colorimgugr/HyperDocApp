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

class IlluminationWidget(QWidget, Ui_IlluminationWidget):
    cubeLoaded = QtCore.pyqtSignal(Hypercube)
    cube_saved = QtCore.pyqtSignal(CubeInfoTemp)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.illuminants_dict = {}
        self.load_illuminants("illumination/Illuminants.csv")

        self.comboBox_Illuminant.addItems(self.illuminants_dict.keys())
        self.comboBox_Illuminant.setCurrentText("D65")

        # Connect widget signals
        self.load_btn.clicked.connect(lambda : self.on_load_cube())

        # Layout and label inside frame_viewer
        self._replace_placeholder('frame_viewer', ZoomableGraphicsView)


        self.cube = None
        self.data = None
        self.wl =None

        self.image_rgb=None

    def load_illuminants(self, filename):
        try:
            with open(filename, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    name = row["Name"].strip()
                    value_str = row.get("Value", "").strip()
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
        self.image_rgb=self.cube.get_rgb_image([10,50,90])
        self.show_image()

    def show_image(self):
        self.frame_viewer.setImage(self.np_to_qpixmap())
        pass

    def rgb_from_illuminant(self):
        # Que acabe en astype(np.uint8)
        pass


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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IlluminationWidget()
    w.show()
    filepath=r'C:\Users\Usuario\Documents\Test/01644-VNIR-genealogies.h5'
    w.load_cube(filepath=filepath)
    image_rgb_temp = w.cube.get_rgb_image([10, 50, 90])
    w.image_rgb=(image_rgb_temp*255).astype(np.uint8)
    w.show_image()

    sys.exit(app.exec_())