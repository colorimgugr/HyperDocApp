import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QSlider, QFileDialog, QHBoxLayout
)
from PIL import Image

from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os
from Hyperdoc_GUI_design import*

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.hyps = [Hypercube(),Hypercube()] # create an hypercube
        self.GT=GT()

        self.canvas_image = Canvas_Image() # Canvas graphique
        self.verticalLayout_image.addWidget(self.canvas_image)

        # Connect pushButtons du canvas load image
        self.pushButton_open_hypercube.clicked.connect(self.open_hypercubes_and_GT)
        self.pushButton_save_image.clicked.connect(self.save_image)
        self.pushButton_save_image.setEnabled(False)

        # Connect elements du canvas GT image
        self.sliders_rgb=[self.horizontalSlider_red_channel,self.horizontalSlider_green_channel,self.horizontalSlider_blue_channel]
        self.spinBox_rgb=[self.spinBox_red_channel,self.spinBox_green_channel,self.spinBox_blue_channel]

        for element in self.sliders_rgb:
            element.valueChanged.connect(self.update_plot)

        for element in self.spinBox_rgb:
            element.valueChanged.connect(self.update_plot)

    def open_hypercubes_and_GT(self):
        """ open .h5 et initialise les sliders. """

        default_dir = os.getcwd()+"/HyperdocApp/hypercubes"
        filepath, _ = QFileDialog.getOpenFileName(self, "Ouvrir un hypercube", default_dir, "Fichiers HDF5 (*.h5)")

        path_VNIR = filepath
        path_SWIR = filepath
        path_GT = filepath

        if filepath:
            if 'VNIR' in filepath:
                path_SWIR = filepath.replace("VNIR", "SWIR")
            elif 'SWIR' in filepath:
                path_VNIR = filepath.replace("SWIR", "VNIR")
            else:
                print('File no valid')

        path_GT= path_VNIR[:-3] + '_GT.png'
        path_GT=path_GT.replace("-VNIR", "")

        self.hyps[0].load_hypercube(path_VNIR)
        self.hyps[1].load_hypercube(path_SWIR)
        self.GT=GT.load_image(path_GT)


    def modif_slider(self,hyp):
            # Ajuste les sliders aux indices disponibles
            max_wl=self.hyp.wl[-1]
            min_wl=self.hyp.wl[0]
            wl_step=int(self.hyp.wl[1]-self.hyp.wl[0])

            if len(self.hyp.wl)==121:
                default_RGB_channels=[450,550,650]
            elif len(self.hyp.wl)==161:
                default_RGB_channels = [900, 1300, 1700]

            for i,element in enumerate (self.sliders_rgb):
                element.setMinimum(min_wl)
                element.setMaximum(max_wl)
                element.setSingleStep(wl_step)
                element.setValue(default_RGB_channels[i])

            for i,element in enumerate (self.spinBox_rgb):
                element.setMinimum(min_wl)
                element.setMaximum(max_wl)
                element.setValue(default_RGB_channels[i])
                element.setSingleStep(wl_step)

            self.update_plot()
            self.pushButton_save_image.setEnabled(True)

    def update_plot(self):
        """ Met à jour l’image affichée en fonction des sliders. """
        if self.hyp.data is None:
            return

        indices = [np.argmin(np.abs(slider.value()-self.hyp.wl)) for slider in self.sliders_rgb]
        rgb_image = self.hyp.get_rgb_image(indices)

        self.canvas_image.update_image(rgb_image)

    def save_image(self):
        """ Sauvegarde l'image affichée. """
        self.canvas_image.save_image()

class Hypercube:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.wl = None
        self.metadata = None

        if  self.filepath:
            self.load_hypercube( self.filepath)

    def load_hypercube(self, filepath):
        """ Charge un fichier .h5 et extrait l'hypercube et evalue les longueurs d'onde. """
        with h5py.File(filepath, 'r') as f:
            self.data = np.array(f['DataCube']).T
            self.metadata = {attr: f.attrs[attr] for attr in f.attrs}
            self.wl = self.metadata['wl']


    def get_rgb_image(self, indices):
        if self.data is None:
            return None
        return self.data[:, :, indices]

class Canvas_Image(FigureCanvas):
    def __init__(self):
        fig, self.ax = plt.subplots()
        super().__init__(fig)

    def update_image(self, rgb_image):
        """ Met à jour l'affichage avec une nouvelle image RGB. """
        if rgb_image is not None:
            self.ax.clear()
            self.ax.imshow(rgb_image)
            self.ax.axis("off")
            self.draw()

    def save_image(self):
        """ Sauvegarde l’image affichée sous forme de fichier. """
        filepath, _ = QFileDialog.getSaveFileName(
            None, "Sauvegarder l'image", "", "Images PNG (*.png);;Images JPEG (*.jpg)"
        )
        if filepath:
            self.figure.savefig(filepath, dpi=300)
            print(f"Image sauvegardée sous {filepath}")

class GT:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data=None
        self.image=None
        self.metadata=None
        self.cmap=None

    def load_image(self,filepath):
        self.image=np.array(Image.open(filepath))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())