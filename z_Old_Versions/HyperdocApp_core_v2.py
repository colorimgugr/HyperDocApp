# cd C:\Users\Usuario\Documents\GitHub\Hyperspectral_Yannick\HyperdocApp
# python -m PyQt5.uic.pyuic -o Hyperdoc_GUI_design.py Hyperdoc_GUI_design.ui

import sys
from tkinter import Radiobutton

import h5py
import numpy as np          
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QSlider, QFileDialog, QHBoxLayout
)
from PIL import Image

from PyQt5.QtCore import Qt,QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.text
import os
from Hyperdoc_GUI_design import*

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.hyps = [Hypercube(),Hypercube()] # create hypercubes objects[VNIR,SWIR]
        self.hyps_rgb_chan_DEFAULT=[[650, 550, 450],[1700, 1300, 900]]   # defaults channels for hypercubes images
        self.hyps_rgb_chan=self.hyps_rgb_chan_DEFAULT.copy()  # channels for hypercubes images, initialiser aux DEFAULT

        self.GT=GroundTruth() # create GT object
        self.image_loaded=[False,False,False]

        self.canvas_image = Canvas_Image() # Canvas pour les images
        self.verticalLayout_image.addWidget(self.canvas_image)

        self.canvas_spectra = Canvas_Spectra() # Canvas pour les spectres
        self.verticalLayout_spectra.addWidget(self.canvas_spectra)

        # Connect pushButtons du canvas load image
        self.pushButton_open_hypercube.clicked.connect(self.open_hypercubes_and_GT)
        self.pushButton_save_image.clicked.connect(self.save_image)
        self.pushButton_save_image.setEnabled(False)

        # Connect elements du canvas GT image
        self.sliders_rgb=[self.horizontalSlider_red_channel,self.horizontalSlider_green_channel,self.horizontalSlider_blue_channel]
        self.spinBox_rgb=[self.spinBox_red_channel,self.spinBox_green_channel,self.spinBox_blue_channel]

        # Timer pour débouncer les changements de slider
        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(300)  # Délai de 300 ms
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_image_from_element)

        for element in self.sliders_rgb:
            element.valueChanged.connect(self.start_debounce_timer)

        for element in self.spinBox_rgb:
            element.valueChanged.connect(self.start_debounce_timer)

        # event connection
        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)
        self.radioButton_SWIR.toggled.connect(self.modif_sliders)
        self.radioButton_VNIR.toggled.connect(self.modif_sliders)
        self.horizontalSlider_transparency_GT.valueChanged.connect(self.start_debounce_timer)

    def start_debounce_timer(self):
        """Déclenche le timer pour mettre à jour l'image après un délai."""
        if not self.debounce_timer.isActive():
            self.debounce_timer.start()

    def update_image_from_element(self):
        """Met à jour l'image après le délai."""
        self.modif_channels()
        if self.image_loaded[2]:
            transparency = self.horizontalSlider_transparency_GT.value() / 100.0  # Convertir en valeur entre 0 et 1
            self.canvas_image.set_gt_transparency(transparency)

    def open_hypercubes_and_GT(self):
        """ open .h5 et initialise les sliders. """

        default_dir = os.getcwd()+"/HyperdocApp/hypercubes"
        filepath, _ = QFileDialog.getOpenFileName(self, "Ouvrir un hypercube", default_dir, "Fichiers HDF5 (*.h5)")

        path_VNIR = filepath
        path_SWIR = filepath
        path_GT = filepath

        if 'VNIR' in filepath:
            path_SWIR = filepath.replace("VNIR", "SWIR")
        elif 'SWIR' in filepath:
            path_VNIR = filepath.replace("SWIR", "VNIR")
        else:
            print('File no valid')

        path_GT= path_VNIR[:-3] + '_GT.png'
        path_GT=path_GT.replace("-VNIR", "")

        try:
            self.hyps[0].load_hypercube(path_VNIR)
            self.image_loaded[0] = True
        except:
            print('No VNIR')
            self.image_loaded[0] = False


        try:
            self.hyps[1].load_hypercube(path_SWIR)
            self.image_loaded[1] = True
        except:
            print('No SWIR')
            self.image_loaded[1] = False

        try:
            self.GT.load_image(path_GT)
            self.image_loaded[2]=True

        except:
            print('No GT')
            self.image_loaded[2] = False

        self.pushButton_save_image.setEnabled(True)
        self.modif_sliders(default=True)
        self.horizontalSlider_transparency_GT.setValue(0)
        self.update_image(load=True,image_loaded=self.image_loaded)
        self.update_spectra(load=True)

    def modif_channels(self):

        hyp_active=self.radioButton_SWIR.isChecked()    # 0 (VNIR) ou 1 (SWIR)
        if self.hyps[hyp_active].wl is not None:
            self.hyps_rgb_chan[hyp_active] = [slider.value() for slider in self.sliders_rgb]

        self.update_image(index=hyp_active)

    def modif_sliders(self,default=False):

        hyp_active=self.radioButton_SWIR.isChecked()    # 0 (VNIR) ou 1 (SWIR)
        hyp=self.hyps[hyp_active]
        # Ajuste les sliders aux indices disponibles
        max_wl=int(hyp.wl[-1])
        min_wl=int(hyp.wl[0])
        wl_step=int(hyp.wl[1]-hyp.wl[0])

        if self.radioButton_grayscale.isChecked():
            self.label_red_channel.setText('')
            self.label_green_channel.setText('')
            self.label_blue_channel.setText('Gray')
        else:
            self.label_red_channel.setText('Red')
            self.label_green_channel.setText('Green')
            self.label_blue_channel.setText('Blue')

        for i,element in enumerate (self.sliders_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[hyp_active][i])
            else:
                element.setValue(self.hyps_rgb_chan[hyp_active][i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i==2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        for i,element in enumerate (self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[hyp_active][i])
            else:
                element.setValue(self.hyps_rgb_chan[hyp_active][i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i==2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

            self.update_image()

        try:
            self.update_image(index=0)
            self.update_image(index=1)
        except: return

    def update_image(self,load=False,index=None,image_loaded=[True,True,True]):
        """ Met à jour l’image affichée en fonction des sliders. """

        if load:

            rgb_images = []
            for i, hyp in enumerate(self.hyps):
                if image_loaded[i]:
                    if self.radioButton_grayscale.isChecked():
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[i][2] - hyp.wl)) for j in range(3)]
                    else:
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[i][j] - hyp.wl)) for j in range(3)]

                    rgb_images.append(hyp.get_rgb_image(channels_index))
                else:
                    rgb_images.append(None)

            if image_loaded[2]:
                rgb_images.append(self.GT.image)
            else:
                rgb_images.append(None)

            self.canvas_image.load_image(rgb_images)

        else:
            if index==1 or index ==0:
                if image_loaded[index]:
                    hyp=self.hyps[index]
                    if self.radioButton_grayscale.isChecked():
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[index][2] - hyp.wl)) for j in range(3)]
                    else:
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[index][j] - hyp.wl)) for j in range(3)]
                    rgb_image=hyp.get_rgb_image(channels_index)
                    self.canvas_image.update_image(rgb_image,index)

    def update_spectra(self,load=False,std=False):
        """ Met à jour l’image affichée en fonction des sliders. """
        if load:
            hyp=self.hyps[0]
            GT_index=[int(i) for i in hyp.metadata['GTLabels'][0]]
            GT_material=[i for i in hyp.metadata['GTLabels'][1]]
            GT_colors=hyp.metadata['GT_cmap'][:,GT_index]

            wls=[hyp.metadata['wl'] for hyp in self.hyps]
            spectra_mean=[hyp.metadata['spectra_mean'] for hyp in self.hyps]
            spectra_std=[hyp.metadata['spectra_std'] for hyp in self.hyps]

            self.canvas_spectra.load_spectra(wls,spectra_mean,spectra_std,GT_material,GT_colors)


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
        self.fig=Figure()
        gs = GridSpec(2, 2, figure=self.fig)
        self.ax0=self.fig.add_subplot(gs[0, 0]) #VNIR
        self.ax1 = self.fig.add_subplot(gs[1, 0]) #SWIR
        self.ax2=self.fig.add_subplot(gs[:,1]) #GT
        self.axs=[self.ax0,self.ax1,self.ax2]
        for ax in self.axs: ax.set_axis_off()
        self.images=[] #pour les 3 images de bases
        self.gt_overlays = []  # Pour stocker l'image GT en superposition

        super().__init__(self.fig)

    def load_image(self,rgb_images):
        for ax in self.axs: ax.clear()
        for i,rgb_image in enumerate(rgb_images):
            if rgb_image is not None:
                self.axs[i].set_axis_off()
                im=self.axs[i].imshow(rgb_image)
                self.images.append(im)

                # Superposition GT sur VNIR et SWIR
                if i < 2 and rgb_images[2] is not None:  # VNIR (i=0) et SWIR (i=1)
                    gt_overlay = self.axs[i].imshow(rgb_images[2], alpha=0)
                    self.gt_overlays.append(gt_overlay)

        self.ax0.set_title('VNIR')
        self.ax1.set_title('SWIR')
        self.ax2.set_title('Ground Truth')
        self.draw()

    def set_gt_transparency(self, transparency):
        """Met à jour la transparence de la GT en superposition sur VNIR et SWIR."""
        for gt_overlay in self.gt_overlays:
            gt_overlay.set_alpha(transparency)
        self.draw()  # Met à jour le canvas

    def update_image(self, rgb_image,index):
        """ Met à jour l'affichage avec une nouvelle image RGB. """
        if self.images:
            self.images[index].set_data(rgb_image)
            self.draw()
                
    def save_image(self):
        """ Sauvegarde l’image affichée sous forme de fichier. """
        filepath, _ = QFileDialog.getSaveFileName(
            None, "Sauvegarder l'image", "", "Images PNG (*.png);;Images JPEG (*.jpg)"
        )
        if filepath:
            self.figure.savefig(filepath, dpi=300)
            print(f"Image sauvegardée sous {filepath}")

class Canvas_Spectra(FigureCanvas):
    def __init__(self):
        self.fig =Figure()  # Crée une figure
        self.ax = self.fig.add_subplot(111)  # Ajoute des axes à la figure
        super().__init__(self.fig)
        self.colors=[]
        self.material=[]
        self.spectra_mean = []  # pour les 6 spectres moyens de bases
        self.spectra_std = []  # pour les 6 std des spectres de bases
        self.wl=[]
        self.n_spectra=len(self.spectra_mean)
        self.leg=None
        self.lines_list=[[]]
        self.map_legend_to_ax=[]

    def load_spectra(self,wls, spectra_mean, spectra_std, GT_material, GT_colors):
        # for i, spectrum in enumerate(spectra):
        self.ax.clear()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Reflectance")
        self.ax.set_title("Average reflectance spectra of Ground Truth material")
        self.wl=wls
        self.spectra_mean= spectra_mean
        self.spectra_std = spectra_std
        self.colors=GT_colors.T
        self.material=GT_material
        n_mat=len(self.material)

        for i in range(n_mat):
            self.ax.plot(self.wl[0],spectra_mean[0][i],color=tuple(self.colors[i]),label=self.material[i])
            self.ax.plot(self.wl[1],spectra_mean[1][i],color=tuple(self.colors[i]))

        self.ax.grid()

    # <editor-fold desc="Interactive legend">
        self.n_spectra = n_mat
        self.lines_list = self.ax.get_lines()
        self.lines_list = np.split(np.array(self.lines_list), self.n_spectra)

        self.leg = self.ax.legend(loc='center left', title='Materials', draggable=True)

        self.leg.get_title().set_picker(True)
        self.map_legend_to_ax = {}
        pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.
        i = 0
        for legend_line in self.leg.get_lines():
            legend_line.set_picker(pickradius)  # Enable picking on the legend line.
            group_num = i
            i += 1
            self.map_legend_to_ax[legend_line] = group_num

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        self.draw()

    def on_pick(self, event):
        obj_picked = event.artist

        if obj_picked not in self.map_legend_to_ax:
            return

        line_group = self.map_legend_to_ax[obj_picked]
        visible = not self.lines_list[line_group][0].get_visible()

        for ax_line in self.lines_list[line_group]:
            ax_line.set_visible(visible)

        obj_picked.set_alpha(1.0 if visible else 0.2)

        self.fig.canvas.draw()  # Mise à jour de l'affichage

       # </editor-fold>


    def update_spectra(self, rgb_image, index):
        """ Met à jour l'affichage avec une nouvelle image RGB. """
        ## check if std checked
        ## add an interactive legend
        ## show 6 spectra with color og GT
        ## show experimental spectra of the pixel
        ## title

        self.draw()

    def save_spectra(self):
        """ Sauvegarde des spectres sous forme de fichier txt ou csv. """
        filepath, _ = QFileDialog.getSaveFileName(
            None, "Sauvegarder les spectres", "", "Txt (*.txt);; CSV (*.csv)"
        )
        # if filepath:
        #     self.figure.savefig(filepath, dpi=300)
        #     print(f"Image sauvegardée sous {filepath}")


class GroundTruth:
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