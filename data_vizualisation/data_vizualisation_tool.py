# cd C:\Users\Usuario\Documents\GitHub\Hypertool\data_vizualisation
# python -m PyQt5.uic.pyuic -o Hyperdoc_GUI_design.py Hyperdoc_GUI_design.ui
# G:\Mi unidad\CIMLab\Proyectos y OTRI\Hyperdoc\Datos\Database_samples_paper\minicubes
# pyinstaller --noconsole --onefile --icon="hyperdoc_logo_transparente.ico" --add-data "Hyperdoc_logo_transparente_CIMLab.png:." HyperdocApp_core_v3.5.1.py
# G:\Mi unidad\CIMLab\Proyectos y OTRI\Hyperdoc\Datos\Database_samples\HYPERDOC Database\Samples
# --exclude-module tensorflow --exclude-module torch

import sys
import traceback
import os

# math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# GUI
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QScrollArea,QDialog,QFormLayout,
    QSlider, QFileDialog, QHBoxLayout,QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,QMessageBox
)
from PyQt5.QtCore import Qt,QTimer,QEvent
from PyQt5.QtGui import QPixmap,QPainter,QIcon,QFont

# images
import h5py
from PIL import Image

# graphs
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.text
from matplotlib.pyplot import fill_between

# intern
from data_vizualisation.data_vizualisation_window import*
from hypercubes.hypercube import*

class Data_Viz_Window(QWidget,Ui_DataVizualisation):

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.hyps = [Hypercube(),Hypercube()] # create hypercubes objects[VNIR,SWIR] two maximum
        self.folder_GT=None
        self.GTexist = True
        self.spec_range=['VNIR','SWIR'] # ['UVIS','VNIR','SWIR']
        self.folder_app=os.path.dirname(__file__)
        self.cubes_path=os.path.dirname(__file__)

        self.hyps_rgb_chan_DEFAULT={'UVIS':[550,450,350],'VNIR':[610, 540, 435],'SWIR':[1605, 1205, 1005]}   # defaults channels for hypercubes images
        self.hyps_rgb_chan=[self.hyps_rgb_chan_DEFAULT[self.spec_range[0]],self.hyps_rgb_chan_DEFAULT[self.spec_range[1]]]  # channels for hypercubes images, initialiser aux DEFAULT

        self.GT=GroundTruth() # create GT object
        self.image_loaded=[False,False,False]

        self.canvas_image = Canvas_Image(self) # Canvas pour les images
        self.verticalLayout_image.addWidget(self.canvas_image)

        self.canvas_spectra = Canvas_Spectra() # Canvas pour les spectres
        self.verticalLayout_spectra.addWidget(self.canvas_spectra)

        # Connect pushButtons du canvas load image
        self.pushButton_open_hypercube.clicked.connect(self.open_hypercubes_and_GT)
        self.pushButton_save_image.clicked.connect(self.save_image)

        # Connect to window Metadata
        self.pushButton_see_all_metadata.clicked.connect(self.show_all_metadata)

        # group active hyp radiobuttons :
        self.radioButtons_ActiveHyp= [self.radioButton_VNIR, self.radioButton_SWIR, self.radioButton_UVIS]
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

        # connect combobox
        self.comboBox_metadata.currentIndexChanged.connect(self.update_metadata_label)

        # event connection
        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)
        self.radioButton_SWIR.clicked.connect(self.modif_sliders)
        self.radioButton_VNIR.clicked.connect(self.modif_sliders)
        self.radioButton_SWIR.clicked.connect(self.update_metadata_label)
        self.radioButton_VNIR.clicked.connect(self.update_metadata_label)
        self.horizontalSlider_transparency_GT.valueChanged.connect(self.start_debounce_timer)
        self.pushButton_save_spectra.clicked.connect(self.save_spectra)
        self.checkBox_std.clicked.connect(lambda:self.update_spectra (load=False))
        self.pushButton_next_cube.clicked.connect(self.next_cube)
        self.pushButton_prev_cube.clicked.connect(self.prev_cube)

        for elem in [self.pushButton_next_cube,self.pushButton_prev_cube,self.pushButton_save_image,self.horizontalSlider_red_channel,self.horizontalSlider_green_channel,self.horizontalSlider_blue_channel,self.spinBox_red_channel,self.spinBox_green_channel,self.spinBox_blue_channel,self.checkBox_std,self.pushButton_save_spectra,self.horizontalSlider_transparency_GT,self.radioButton_VNIR,self.radioButton_rgb_user,self.radioButton_rgb_default,self.radioButton_SWIR,self.radioButton_grayscale]:
            elem.setEnabled(False)
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))


        # load lookup table for VNIR - SWIR correspodance
        if getattr(sys, 'frozen', False):  # pynstaller case
            BASE_DIR = sys._MEIPASS
        else:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        table_path = os.path.join(BASE_DIR,
                                         "data_vizualisation/Spatially registered minicubes equivalence.csv")

        self.minicube_association_table = pd.read_csv(table_path)

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

    def prev_cube(self):
        self.change_hyp_quick(-1)

    def next_cube(self):
        self.change_hyp_quick(+1)

    def change_hyp_quick(self,prev_next):


        last_hyp_path = self.cubes_path
        init_dir_hyp = '/'.join(last_hyp_path.split('/')[:-1])
        file_init = last_hyp_path.split('/')[-1]
        last_num=file_init.split('-')[0]
        if "." in last_num:
            return

        files = sorted(os.listdir(init_dir_hyp))
        index_init = files.index(file_init)

        file_new = files[(index_init + prev_next) % len(files)]
        i = prev_next
        while (last_num in file_new or '.h5' not in file_new):
            i += prev_next
            file_new = files[(index_init + i) % len(files)]

        file_hyp = init_dir_hyp + '/' + file_new

        try:
            self.open_hypercubes_and_GT(filepath=file_hyp)
        except:
            pass

    def open_hypercubes_and_GT(self,filepath=None):
        """ load cube and look for complemtal cube and also GT  """

        # set 3 filepath to 0
        path_VNIR = None
        path_SWIR = None
        path_UV = None
        file_GT = None
        self.image_loaded=[0,0,0] # hyp 0, hyp 1 and GT

        if filepath:quick_change=True # with arrows
        else : quick_change=False

        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Open hypercube", self.cubes_path)

        if not filepath:
            return

        self.cubes_path=filepath # update default folder

        cube=Hypercube(filepath,load_init=True) # load hypercube

        # Test if VNIR SWIR or UV range (other)
        if 'VNIR' in filepath or (cube.wl[-1] < 1100 and cube.wl[0] > 350):
            path_VNIR = filepath
            if 'VNIR' in filepath:
                path_SWIR = filepath.replace("VNIR", "SWIR")
                if not os.path.exists(path_SWIR):
                    # use table
                    base_name=os.path.basename(filepath).split('.')[0]
                    matching_rows = self.minicube_association_table[self.minicube_association_table['VNIR'] == base_name]

                    if not matching_rows.empty:
                        cube_asoc = matching_rows['SWIR'].iloc[0]
                        path_SWIR = filepath.replace(base_name, cube_asoc)
                    else:
                        path_SWIR = None  # ou lève une exception ou gère autrement
                        cube_asoc = self.minicube_association_table.loc[self.minicube_association_table['VNIR'] == base_name]['SWIR'][0]
                        path_SWIR=filepath.replace(base_name,cube_asoc)

        elif 'SWIR' in filepath or cube.wl[-1] >= 1100:
            path_SWIR = filepath
            if 'SWIR' in filepath:
                path_VNIR = filepath.replace("SWIR", "VNIR")
                if not os.path.exists(path_VNIR):
                    # use table
                    base_name=os.path.basename(filepath).split('.')[0]

                    matching_rows = self.minicube_association_table[self.minicube_association_table['SWIR'] == base_name]

                    if not matching_rows.empty:
                        cube_asoc = matching_rows['VNIR'].iloc[0]
                        path_VNIR = filepath.replace(base_name, cube_asoc)
                    else:
                        path_VNIR = None  # ou lève une exception ou gère autrement
                        cube_asoc = \
                        self.minicube_association_table.loc[self.minicube_association_table['SWIR'] == base_name]['VNIR'][0]
                        path_VNIR = filepath.replace(base_name, cube_asoc)
        else:
            path_UV=filepath

        # load hypercubes
        if path_VNIR is not None:

            try:
                self.hyps[0].open_hyp(default_path=path_VNIR,open_dialog=False,show_exception=False)

                if self.hyps[0].data is None:
                    self.image_loaded[0] = False

                else:
                    self.image_loaded[0] = True
                    self.spec_range[0] = 'VNIR'
            except:
                self.image_loaded[0] = False

        elif path_UV is not None:
            try:
                self.hyps[0].open_hyp(path_UV,open_dialog=False,show_exception=False)
                if self.hyps[0].data is None :
                    self.image_loaded[0] = False
                else :
                    self.image_loaded[0] = True
                    self.spec_range[0] = 'UVIS'
            except:
                self.image_loaded[0] = False
                return

        if path_SWIR is not None:

            try:
                self.hyps[1].open_hyp(path_SWIR,open_dialog=False,show_exception=False)
                if self.hyps[1].data is None :
                    self.image_loaded[1] = False
                else :
                    self.image_loaded[1] = True
                    self.spec_range[1] = 'SWIR'
            except:
                self.image_loaded[1] = False

        # construct GT name from filename adding _GT

        if self.image_loaded[0]:
            if path_VNIR is not None:
                file_GT=(path_VNIR.split('.')[0] + '_GT.png').split('/')[-1]
            elif path_UV is not None :
                file_GT = (path_UV.split('.')[0] + '_GT.png').split('/')[-1]
        elif self.image_loaded[1]:
            file_GT=(path_SWIR.split('.')[0] + '_GT.png').split('/')[-1]

        # load GT using previous folder_name saved with file_GT
        if self.folder_GT is not None :
            path_GT = self.folder_GT + '/' + file_GT
            try:
                self.GT.load_image(path_GT)
                self.image_loaded[2] = True
            except:
                pass

        # try using same folder as opened cube
        if not self.image_loaded[2]:
            self.folder_GT = os.path.dirname(filepath)
            path_GT =  os.path.join(self.folder_GT,file_GT)
            try :
                self.GT.load_image(path_GT)
                self.image_loaded[2] = True
            except:
                pass

        # try using parent folder/GT of opened cube
        if not self.image_loaded[2]:
            self.folder_GT = os.path.join(os.path.dirname(os.path.dirname(filepath)),'GT')
            path_GT = os.path.join(self.folder_GT, file_GT)
            try:
                self.GT.load_image(path_GT)
                self.image_loaded[2] = True
            except:
                pass

        # try using parent folder of opened cube
        if not self.image_loaded[2]:
            self.folder_GT = os.path.dirname(os.path.dirname(filepath))
            path_GT =  os.path.join(self.folder_GT,file_GT)
            try:
                self.GT.load_image(path_GT)
                self.image_loaded[2] = True
            except:
                # try by asking to user to choose file
                 if not quick_change:
                    qm=QMessageBox()
                    ans=qm.question(self, 'No GT found', "No Ground Truth found for this minicube.\nDo you want to open the Ground Truth manually ?", qm.Yes | qm.No)
                    if ans==qm.Yes:
                        filepath, _ = QFileDialog.getOpenFileName(self, "Open Ground Truth PNG image",path_VNIR,
                                                                  "PNG (*.png)")
                        try:
                            path_GT = filepath

                            self.GT.load_image(path_GT)
                            self.image_loaded[2] = True
                            self.folder_GT=os.path.dirname(filepath)

                        except:
                            msg = QMessageBox()
                            msg.setIcon(QMessageBox.Icon.Warning)
                            msg.setText("No Ground Truth found.")
                            msg.setWindowTitle("GT not found")
                            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                            msg.exec()

        ## Update widgets

        # get last cube selected  :
        i_act=0
        for radio in self.radioButtons_ActiveHyp:
            if radio.isChecked():
                break
            i_act+=1

        for elem in [self.pushButton_next_cube,self.pushButton_prev_cube,self.pushButton_save_image,self.horizontalSlider_red_channel,self.horizontalSlider_green_channel,self.horizontalSlider_blue_channel,self.spinBox_red_channel,self.spinBox_green_channel,self.spinBox_blue_channel,self.checkBox_std,self.pushButton_save_spectra,self.horizontalSlider_transparency_GT,self.radioButton_VNIR,self.radioButton_rgb_user,self.radioButton_rgb_default,self.radioButton_SWIR,self.radioButton_grayscale]:
            elem.setEnabled(True)

        for radio in self.radioButtons_ActiveHyp:
            radio.setAutoExclusive(False)
            radio.setEnabled(False)
            radio.setChecked(False)

        if self.image_loaded[0]:
            if self.spec_range[0]=='VNIR':
                self.radioButton_VNIR.setEnabled(True)
                if i_act==0 or not self.image_loaded[1] or i_act==2:
                    self.radioButton_VNIR.setChecked(True)

            elif self.spec_range[0] == 'UVIS':
                self.radioButton_UVIS.setEnabled(True)
                if i_act==2 or not self.image_loaded[1]:
                    self.radioButton_UVIS.setChecked(True)

        if  self.image_loaded[1]:
            self.radioButton_SWIR.setEnabled(True)
            if i_act == 1 or not self.image_loaded[0]:
                self.radioButton_SWIR.setChecked(True)

        if not self.image_loaded[2]:
            self.horizontalSlider_transparency_GT.setEnabled(False)

        for radio in self.radioButtons_ActiveHyp:
            radio.setAutoExclusive(True)

        self.modif_sliders(default=True)
        self.horizontalSlider_transparency_GT.setValue(0)
        self.update_image(load=True)
        self.update_combo_meta(init=True)

        hyp = self.hyps[self.radioButton_SWIR.isChecked()]
        try:
            len(hyp.metadata['GTLabels'])
            self.GTexist = True
        except:
            self.GTexist=False

        if self.GTexist:
            self.update_spectra(load=True)
            self.checkBox_std.setEnabled(True)
            self.pushButton_save_spectra.setEnabled(True)
        else:
            self.canvas_spectra.clear_spectra()
            self.checkBox_std.setEnabled(False)
            self.pushButton_save_spectra.setEnabled(False)

        message=''
        if self.image_loaded[0] :
            if self.spec_range[0]=='VNIR':
                message+='VNIR found \n'
            elif self.spec_range[0]=='UVIS':
                message+='UV-VIS found \n'
        if self.image_loaded[1]:message+='SWIR found \n'
        if self.image_loaded[2]:message+='GT found'
        else : message+='GT NOT FOUND'

        self.label_general_message.setText(message)

    def update_combo_meta(self,init=False):
        hyp = self.hyps[self.radioButton_SWIR.isChecked()]

        last_key = self.comboBox_metadata.currentText()
        if last_key=='': last_key='cubeinfo'

        if init:
            self.comboBox_metadata.clear()

        if hyp.metadata is not None:
            for key in hyp.metadata.keys():
                if key not in ['wl','GT_cmap','spectra_mean','spectra_std']:
                    if key in ['GTLabels','pixels_averaged']:
                        try:
                            len(hyp.metadata[key])
                            self.comboBox_metadata.addItem(f"{key}")
                        except:
                            a=0

                    else:
                        self.comboBox_metadata.addItem(f"{key}")
                        if key==last_key:
                            self.comboBox_metadata.setCurrentText(key)

            self.update_metadata_label()

    def update_metadata_label(self):
        key = self.comboBox_metadata.currentText()
        if key=='':
            key='cubeinfo'
        hyp = self.hyps[self.radioButton_SWIR.isChecked()]
        raw = hyp.metadata[key]
        match key:
            case 'GTLabels':
                if len(raw.shape)==2:
                    st=f'GT indexes : <b>{(' , ').join(raw[0])}</b>  <br>  GT names : <b>{(' , ').join(raw[1])}</b>'
                elif len(raw.shape)==1:
                    st=f'GT indexes : <b>{(raw[0])}</b>  <br>  GT names : <b>{raw[1]}</b>'

            case 'aged':
                st=f'The sample has been aged ? <br> <b>{raw}</b>'

            case 'bands':
                st=f'The camera have <b>{raw[0]}</b> spectral bands.'

            case 'date':
                if len(raw)>1:info=raw
                else: info=raw[0]
                st=f'Date of the sample : <b>{info}</b>'

            case 'device':
                st=f'Capture made with the device : <br> <b>{raw}</b>'

            case 'illumination':
                st=f'Lamp used for the capture : <br> <b>{raw}</b>'

            case 'name':
                st=f'Name of the minicube : <br> <b>{raw}</b>'

            case 'number':
                st = f'Number of the minicube : <br> <b>{raw}</b>'

            case 'parent_cube':
                st = f'Parent cube of the minicube : <br> <b>{raw}</b>'

            case 'pixels_averaged':
                st = f'The number of pixels used for the <b>{len(raw)}</b> mean spectra of the GT materials are : <br> <b>{(' , ').join([str(r) for r in raw])}</b> '

            case 'reference_white':
                st = f'The reference white used for reflectance measurement is : <br> <b>{raw}</b>'

            case 'restored':
                st = f'The sample has been restored ?  <br> <b> {['NO','YES'][raw[0]]}</b>'

            case 'stage':
                st = f'The capture was made with a  <b>{raw}</b> stage'

            case 'reference_white':
                st = f'The reference white used for reflectance measurement is : <br> <b>{raw}</b>'

            case 'substrate':
                st = f'The substrate of the sample is : <br> <b>{raw}</b>'

            case 'texp':
                st = f'The exposure time set for the capture was <b>{raw[0]:.2f}</b> ms.'

            case 'height':
                st = f'The height of the minicube <b>{raw[0]}</b> pixels.'

            case 'width':
                st = f'The width of the minicube <b>{raw[0]}</b> pixels.'

            case 'position':
                st = f'The (x,y) coordinate of the upper right pixel of the minicube in the parent cube is : <br> <b>({raw[0]},{raw[1]})</b>'

            case 'range':
                val=['UV','VNIR : 400 - 1000 nm','SWIR : 900 - 1700 nm'][list(raw).index(1)]
                st = f'The range of the capture is : <br> <b>{val}</b>'

            case _:
                st=f'<b>{hyp.metadata[key]}</b>'

        self.label_metadata.setText(st)

    def show_all_metadata(self):

        hyp = self.hyps[self.radioButton_SWIR.isChecked()]

        if hyp.metadata is None:
            QMessageBox.information(self, "No Metadata", "No metadata available.")
            return

            # Window of dialog kind
        dialog = QDialog(self)
        dialog.setWindowTitle("All Metadata")
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_layout = QFormLayout(inner)

        # add rows to the form_layout
        for key, val in hyp.metadata.items():
            # Ignore entries too long
            if key in ['spectra_mean', 'spectra_std', 'GT_cmap', 'wl']:
                continue

            try:
                if isinstance(val, list) or isinstance(val, np.ndarray):
                    val_str = ', '.join(str(v) for v in val)
                elif isinstance(val, dict):
                    val_str = str(val)
                else:
                    val_str = str(val)
            except Exception as e:
                val_str = f"<unable to display: {e}>"

            form_layout.addRow(f"{key}:", QLabel(val_str))

        #add scroll widget in dialog eindow layout
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        # Close push button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.setLayout(layout)
        dialog.resize(500, 600)
        dialog.exec()

    def modif_channels(self):

        hyp_active=self.radioButton_SWIR.isChecked()    # 0 (VNIR) ou 1 (SWIR)
        if self.hyps[hyp_active].wl is not None:
            self.hyps_rgb_chan[hyp_active] = [slider.value() for slider in self.sliders_rgb]

        for elem in self.spinBox_rgb:
            val=elem.value()
            wl=self.hyps[hyp_active].wl
            if val not in wl:
                index_good=np.abs(val-wl).argmin()
                elem.setValue(wl[index_good])

        self.update_image(index=hyp_active)

    def modif_sliders(self,default=False):

        hyp_active=self.radioButton_SWIR.isChecked()    # 0 (VNIR ou UV) ou 1 (SWIR)
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
                element.setValue(self.hyps_rgb_chan_DEFAULT[self.spec_range[hyp_active]][i])
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
                element.setValue(self.hyps_rgb_chan_DEFAULT[self.spec_range[hyp_active]][i])
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

    def update_image(self,index=None,load=False,UV=False):
        """ Met à jour l’image affichée en fonction des sliders. """

        if load:
            self.canvas_image.create_axis(self.image_loaded)
            rgb_images = []
            title=''
            for i, hyp in enumerate(self.hyps):
                if self.image_loaded[i]:
                    if self.radioButton_grayscale.isChecked():
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[i][2] - hyp.wl)) for j in range(3)]
                    else:
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[i][j] - hyp.wl)) for j in range(3)]
                    rgb_image= hyp.get_rgb_image(channels_index)
                    rgb_image/=np.max(rgb_image)
                    rgb_images.append(rgb_image)
                    if type(hyp.metadata['number']) is str and type(hyp.metadata['parent_cube']) is str :
                        title=f'{hyp.metadata['number']} - {hyp.metadata['parent_cube']}'
                    else :
                        title=hyp.cube_info.metadata_temp['name']
                else:
                    rgb_images.append(None)

            if self.image_loaded[2]:
                rgb_images.append(self.GT.image)
            else:
                rgb_images.append(None)

            self.canvas_image.load_image(rgb_images,title=title,UV=UV)

        else:
            if index==1 or index ==0:
                if self.image_loaded[index]:
                    hyp=self.hyps[index]
                    if self.radioButton_grayscale.isChecked():
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[index][2] - hyp.wl)) for j in range(3)]
                    else:
                        channels_index = [np.argmin(np.abs(self.hyps_rgb_chan[index][j] - hyp.wl)) for j in range(3)]
                    rgb_image=hyp.get_rgb_image(channels_index)
                    self.canvas_image.update_image(rgb_image/np.max(rgb_image),index)

    def update_spectra(self,load=False):

        std = self.checkBox_std.isChecked()

        if load :
            hyp = self.hyps[self.radioButton_SWIR.isChecked()]
            try :
                len(hyp.metadata['spectra_mean'])
            except:
                return

            hyps_loaded=[self.hyps[i] for i in [0,1] if self.image_loaded[i]]
            wls = [hyp.wl for hyp in hyps_loaded]

            try :
                if len(hyp.metadata['GTLabels'].shape)==2:
                    GT_index = [[int(i),int(i)-1][int(i)==255] for i in hyp.metadata['GTLabels'][0]]
                    GT_material = [i for i in hyp.metadata['GTLabels'][1]]
                    GT_colors = hyp.metadata['GT_cmap'][:, GT_index]
                    spectra_mean = [hyp.metadata['spectra_mean'] for hyp in hyps_loaded]
                    spectra_std = [hyp.metadata['spectra_std'] for hyp in hyps_loaded]
                elif len(hyp.metadata['GTLabels'].shape)==1:
                    GT_index=int(hyp.metadata['GTLabels'][0])
                    if GT_index==255:GT_index=254
                    GT_material = [hyp.metadata['GTLabels'][1]]
                    GT_colors = np.array([hyp.metadata['GT_cmap'][:, GT_index]]).T
                    spectra_mean =[[hyp.metadata['spectra_mean'] for hyp in hyps_loaded]]
                    spectra_std = [[hyp.metadata['spectra_std'] for hyp in hyps_loaded]]
            except:
                pass

            try:
                self.canvas_spectra.load_spectra(wls,spectra_mean,spectra_std,GT_material,GT_colors,std,self.image_loaded)
            except:
                self.canvas_spectra.clear_spectra()
        else:
            self.canvas_spectra.update_spectra(std)

    def save_image(self):
        """ save current images """
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                None, "Sauvegarder l'image", "", "Images PNG (*.png);;Images JPEG (*.jpg)"
            )
            self.canvas_image.save_image(filepath)
            self.label_general_message.setText(f'Images saved as : \n {filepath}')
        except:
            self.label_general_message.setText(f'Saving images FAILED')

    def save_spectra(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Enregistrer le fichier", "",
                                                   "CSV (*.csv);;TXT (*.txt);;PNG (*.png);;JPG (*.jpg);;SVG (*.svg)")
        try:

            if file_name:  # Si l'utilisateur a choisi un fichier
                term = file_name[-4:]

                if term in ['.png','.jpg','.svg']:
                    self.canvas_spectra.save_spectra(file_name)
                    self.label_general_message.setText(f'Spectra image saved as : {file_name.split('/')[-1]}')

                else:
                    delimiter = '\t'
                    end_name=['_VNIR.','_SWIR.']

                    for i in [0,1]:
                        if self.image_loaded[i]:
                            hyp=self.hyps[i]
                            if hyp.metadata['spectra_mean'] is not None:
                                data=np.concatenate((np.expand_dims(hyp.metadata['wl'],0),hyp.metadata['spectra_mean']),axis=0)
                                header='wl'

                                for l in hyp.metadata['GTLabels'][1]:
                                    header += delimiter
                                    header += l

                                np.savetxt(file_name.replace('.',end_name[i]), data.T, header=header, comments='', delimiter='\t')

                    self.label_general_message.setText(f'Spectra values saved as : {file_name.split('/')[-1]}')

        except:
            self.label_general_message.setText('Saving spectra FAILED')

class Canvas_Image(FigureCanvas):
    def __init__(self,parent_logic):
        self.fig=Figure(facecolor=(1, 1, 1, 0.1))
        super().__init__(self.fig)
        self.logic = parent_logic # to make a reference to Data_Viz class
        self.gs = GridSpec(2, 2, figure=self.fig)
        self.ax0 = self.fig.add_subplot(self.gs[0, 0])  # VNIR
        self.ax1 = self.fig.add_subplot(self.gs[1, 0])  # SWIR
        self.ax2 = self.fig.add_subplot(self.gs[:, 1])  # GT
        self.axs = [self.ax0, self.ax1, self.ax2]
        for ax in self.axs: ax.set_axis_off()
        self.images=[] #pour les 3 images de bases
        self.gt_overlays = []  # Pour stocker l'image GT en superposition

        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.85)

        #add interaction
        self.drag_start = None
        self.drag_ax = None
        self.left_button_down = False

        self.mpl_connect('figure_leave_event', self.on_figure_leave) # will be use to cancel drag
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

    def on_figure_leave(self, event):
        self.left_button_down = False
        self.drag_start = None
        self.drag_ax = None

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return

        # get wich image
        index = self.axs.index(ax)
        if self.images[index] is None:
            return

        # Zoom factor
        base_scale = 1.2
        scale = base_scale if event.step < 0 else 1 / base_scale

        #get center of homotethie from ouse
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        xdata,ydata = event.xdata,event.ydata

        # calculate new lims of image
        new_xlim = [xdata + (x - xdata) * scale for x in xlim]
        new_ylim = [ydata + (y - ydata) * scale for y in ylim]

        # to check if image not smaller than original
        img_shape = self.images[index].get_array().shape
        max_width = 1.1*img_shape[1]
        max_height = 1.1*img_shape[0]
        if (new_xlim[1] - new_xlim[0]) > max_width or (new_ylim[0] - new_ylim[1]) > max_height:
            return

        #Apply to all axes
        for i, a in enumerate(self.axs):
            if self.images[i] is not None:
                img_shape = self.images[i].get_array().shape
                a.set_xlim(*self.clamp_xlim(new_xlim, img_shape[1]))
                a.set_ylim(*self.clamp_ylim(new_ylim, img_shape[0]))
        self.draw()

    def clamp_xlim(self, xlim, width):
        x0, x1 = xlim
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if x1 > width:
            x0 -= (x1 - width)
            x1 = width
        return x0, x1

    def clamp_ylim(self, ylim, height):
        y0, y1 = ylim
        if y1 < 0:
            y0 -= y1
            y1 = 0
        if y0 > height:
            y1 -= (y0 - height)
            y0 = height
        return y0, y1

    def on_press(self, event):
        if event.button == 1 and event.inaxes:
            self.left_button_down = True
            self.drag_start = (event.xdata, event.ydata)
            self.drag_ax = event.inaxes

    def on_release(self, event):
        if event.button == 1:
            self.left_button_down = False
            self.drag_start = None
            self.drag_ax = None

    def on_mouse_move(self,event):
        # check if on VNIR or SWIR, if not, erase

        if event.inaxes is None:
            return

        # Pan if in drag_mode

        if self.left_button_down and event.inaxes == self.drag_ax and self.drag_start is not None:
            dx = self.drag_start[0] - event.xdata
            dy = self.drag_start[1] - event.ydata

            xlim = self.drag_ax.get_xlim()
            ylim = self.drag_ax.get_ylim()

            new_xlim = (xlim[0] + dx, xlim[1] + dx)
            new_ylim = (ylim[0] + dy, ylim[1] + dy)

            #for all axes
            for i, ax in enumerate(self.axs):
                if self.images[i] is not None:
                    img_shape = self.images[i].get_array().shape
                    ax.set_xlim(*self.clamp_xlim(new_xlim, img_shape[1]))
                    ax.set_ylim(*self.clamp_ylim(new_ylim, img_shape[0]))

            self.draw()
            return

        # if no button is pressed -> live spectrum

        if not self.logic.checkBox_live_spectrum.isChecked():
            return

        if event.inaxes not in self.axs[:2]:
            self.logic.canvas_spectra.update_live_spectra()
            return

        index = self.axs.index(event.inaxes)

        if self.images[index] is None:
            return

        try:
            x, y = int(event.xdata), int(event.ydata)

            if x < 0 or y < 0:
                return
            hyp = self.logic.hyps[index]
            if hyp.data is None:
                return

            if y >= hyp.data.shape[0] or x >= hyp.data.shape[1]:
                return

            spectrum = hyp.data[y, x, :]
            wavelength = hyp.wl

            self.logic.canvas_spectra.update_live_spectra(spectrum, wavelength)

        except Exception as e:
            print(f"[on_mouse_move] error: {e}")
            return

    def create_axis(self,images=[False,False,False]):
        self.fig.clear()
        if images==[False,False,False]:
            self.ax0 = self.fig.add_subplot(self.gs[0, 0])  # VNIR
            self.ax1 = self.fig.add_subplot(self.gs[1, 0])  # SWIR
            self.ax2 = self.fig.add_subplot(self.gs[:, 1])  # GT
        elif images==[True,False,False]:
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[0, 0])
            self.ax0 = self.fig.add_subplot(self.gs[:, :]) # VNIR

        elif images==[False, True, False]:
            self.ax0 = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[0, 0])
            self.ax1 = self.fig.add_subplot(self.gs[:, :])  # SWIR

        elif images==[True, True, False]:
            self.ax2 = self.fig.add_subplot(self.gs[0, 0])
            self.ax0 = self.fig.add_subplot(self.gs[0, :])  # VNIR
            self.ax1 = self.fig.add_subplot(self.gs[1, :])  # SWIR


        elif images==[True, False, True]:
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])
            self.ax0 = self.fig.add_subplot(self.gs[:, 0])  # VNIR
            self.ax2 = self.fig.add_subplot(self.gs[:, 1])  # GT

        elif images==[False, True, True]:
            self.ax0 = self.fig.add_subplot(self.gs[0, 0])
            self.ax1 = self.fig.add_subplot(self.gs[:, 0])  # SWIR
            self.ax2 = self.fig.add_subplot(self.gs[:, 1])  # GT

        elif images==[True, True, True]:
            self.ax0 = self.fig.add_subplot(self.gs[0, 0])
            self.ax1 = self.fig.add_subplot(self.gs[1, 0])  # SWIR
            self.ax2 = self.fig.add_subplot(self.gs[:, 1])  # GT

        self.axs = [self.ax0, self.ax1, self.ax2]
        for ax in self.axs: ax.set_axis_off()

    def load_image(self,rgb_images,title=None,UV=False):

        self.images=[]

        for i,rgb_image in enumerate(rgb_images):
            self.axs[i].clear()
            self.axs[i].set_axis_off()

            if rgb_image is not None:

                im=self.axs[i].imshow(rgb_image)

                self.images.append(im)

                if not UV :
                    self.axs[i].set_title(['VNIR','SWIR','Ground Truth'][i])
                else :
                    self.axs[i].set_title(['UVIS', 'SWIR', 'Ground Truth'][i])
                # Superposition GT sur VNIR et SWIR

                if i < 2 and rgb_images[2] is not None:  # VNIR (i=0) et SWIR (i=1)
                    gt_overlay = self.axs[i].imshow(rgb_images[2], alpha=0)
                    self.gt_overlays.append(gt_overlay)

            else:
                self.images.append(None)

        self.fig.suptitle(title)
        self.fig.tight_layout()

        ## same size and centered for GT if two hypercubes images
        bbox0 = self.axs[0].get_position()
        bbox1 = self.axs[1].get_position()
        height = (bbox0.height + bbox1.height) / 2
        width = min(bbox0.width, bbox1.width)
        new_bbox2 = [bbox1.x1 + 0.02, bbox0.y0, width, height]  # [x0, y0, width, height]
        y_center = (bbox0.y0 + bbox1.y1) / 2  # centre vertical entre les deux
        y0 = y_center - height / 2
        x0 = bbox1.x1 + 0.02  # marge à droite
        new_bbox2 = [x0, y0, width, height]
        self.axs[2].set_position(new_bbox2)

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
                
    def save_image(self,filepath):
        """ Sauvegarde l’image affichée sous forme de fichier. """

        if filepath:
            self.figure.savefig(filepath, dpi=300)

class Canvas_Spectra(FigureCanvas):
    def __init__(self):
        self.fig =Figure(facecolor=(1, 1, 1, 0.1))  # Crée une figure
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)  # Ajoute des axes à la figure
        self.ax.set_facecolor((0.4,0.4,0.4,1))
        self.ax.set_axis_off()
        self.colors=[]
        self.material=[]
        self.spectra_mean = []  # pour les 6 spectres moyens de bases
        self.spectra_std = []  # pour les 6 std des spectres de bases
        self.wl=[]
        self.n_spectra=len(self.spectra_mean)
        self.leg=None
        self.lines_list=[]
        self.fill_between_list = []
        self.map_legend_to_ax=[]
        self.std_visible=None

    def clear_spectra(self):
        self.ax.clear()
        self.ax.set_facecolor((0.4, 0.4, 0.4, 1))
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Reflectance")
        self.ax.set_title("Live spectrum (no Ground Truth)")
        self.ax.grid(True)
        self.draw()

    def load_spectra(self,wls, spectra_mean, spectra_std, GT_material, GT_colors,std=False,image_loaded=[False,False,False]):
        # for i, spectrum in enumerate(spectra):
        self.ax.clear()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Reflectance")
        self.ax.set_title("Average reflectance spectra of Ground Truth material")

        self.wl=wls
        self.spectra_mean= spectra_mean
        self.spectra_std = spectra_std
        maxR=1
        for spec in spectra_mean:
            if np.max(spec)>maxR:maxR=np.max(spec)
        self.ax.set_ylim((0, 0.05+maxR))
        self.colors=GT_colors.T
        self.material=GT_material
        self.lines_list = []
        self.fill_between_list = []
        self.std_visible=std

        if isinstance(self.material , str):
            n_mat=1
        else :
            n_mat=len(self.material)

        for i in range(n_mat):
            label_i = self.cut_long_string(self.material[i], 25)

            if image_loaded[0]:
                std_VNIR=self.ax.fill_between(self.wl[0],spectra_mean[0][i] + 1*spectra_std[0][i],
                                spectra_mean[0][i] - 1*spectra_std[0][i],
                                color=tuple(self.colors[i]), alpha=.6, linewidth=0)
                self.fill_between_list.append(std_VNIR)

                if not std:
                    std_VNIR.set_visible(False)


                line_VNIR, = self.ax.plot(self.wl[0], spectra_mean[0][i], color=tuple(self.colors[i]),
                                          label=label_i)
                self.lines_list.append(line_VNIR)

            if image_loaded[1]:
                j=0
                if image_loaded[0]:
                    j=1

                std_SWIR=self.ax.fill_between(self.wl[j], spectra_mean[j][i] + 1 * spectra_std[j][i],
                                     spectra_mean[j][i] - 1 * spectra_std[j][i],
                                     color=tuple(self.colors[i]), alpha=.6, linewidth=0)

                self.fill_between_list.append(std_SWIR)

                if not std:
                    std_SWIR.set_visible(False)

                line_SWIR,=self.ax.plot(self.wl[j],spectra_mean[j][i],color=tuple(self.colors[i]),label=[label_i,None][j])
                self.lines_list.append(line_SWIR)

        self.ax.grid()

    # <editor-fold desc="Interactive legend">
        self.n_spectra = n_mat

        self.lines_list = np.split(np.array(self.lines_list), self.n_spectra)
        self.fill_between_list = np.split(np.array(self.fill_between_list), self.n_spectra)

        self.leg = self.ax.legend(loc='lower right', title='Materials', title_fontproperties={'weight':'bold'}, draggable=True)
        self.leg.get_frame().set_facecolor((1,1,1,0.01))

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

        self.fig.tight_layout()
        self.draw()

    def on_pick(self, event):
        obj_picked = event.artist

        if obj_picked not in self.map_legend_to_ax:
            return

        line_group = self.map_legend_to_ax[obj_picked]
        visible = not self.lines_list[line_group][0].get_visible()

        for ax_line in self.lines_list[line_group]:
            ax_line.set_visible(visible)

        for fill_between in self.fill_between_list[line_group]:
            if self.std_visible:
                fill_between.set_visible(visible)

        obj_picked.set_alpha(1.0 if visible else 0.2)

        self.fig.canvas.draw()  # Mise à jour de l'affichage

    def save_spectra(self,file_name):
        self.fig.savefig(file_name)

    def update_spectra(self,std):

        self.std_visible=std

        for i,line in enumerate(self.lines_list):
            if line[0].get_visible():
                for fill in self.fill_between_list[i]:
                    fill.set_visible(std)

        self.draw()

    def update_live_spectra(self, spectrum=None, wavelength=None):

        if spectrum is None or wavelength is None:
            if hasattr(self, "live_line"):
                if self.live_line is not None:
                    try:
                        self.live_line.remove()
                    except Exception as e:
                        print(f"[update_live_spectra] Could not remove live_line: {e}")
                    self.live_line = None
                    self.draw()
            return

        if not hasattr(self, "live_line") or self.live_line is None :
            self.live_line, = self.ax.plot(wavelength, spectrum, color='blue', linestyle='-', label='_nolegend_')

        else:
            self.live_line.set_data(wavelength, spectrum)

        self.ax.relim()
        self.ax.autoscale_view()
        maxR = 1
        if np.max(spectrum) > maxR: maxR = np.max(spectrum)
        self.ax.set_ylim((0, 0.05 + maxR))
        self.draw()

    def cut_long_string(self,text=None,len_max=20):

        if len(text)>len_max:
            text_start=text[:len_max-1]
            index_cut=[idx for idx, char in enumerate(text_start) if char == ' ']
            if len(index_cut)!=0:
                index_cut=index_cut[-1]
                text_start = text[:index_cut]
                text_end = text[index_cut + 1:]
            else:
                index_cut=len_max-1
                text_start = text[:index_cut]+'-'
                text_end = text[index_cut:]

            text_split=[text_start]
            while len(text_end)>len_max:
                text=text_end
                text_start=text[:len_max-1]
                index_cut = [idx for idx, char in enumerate(text_start) if char == ' ']
                if len(index_cut) != 0:
                    index_cut = index_cut[-1]
                    text_start = text[:index_cut]
                    text_end = text[index_cut + 1:]
                else:
                    index_cut = len_max - 1
                    text_start = text[:index_cut]
                    text_end = text[index_cut:]

                text_split.append(text_start)

            text_split.append(text_end)
            text=('\n').join(text_split)

        return text

class GroundTruth:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data=None
        self.image=None
        self.metadata=None
        self.cmap=None

    def load_image(self,filepath):
        self.image=np.array((Image.open(filepath)).convert('RGB'))

def excepthook(exc_type, exc_value, exc_traceback):
    """Capture les exceptions et les affiche dans une boîte de dialogue."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText("Une erreur est survenue :")
    msg_box.setInformativeText(error_msg)
    msg_box.exec_()

def update_font(_app,width=None,_font="Segoe UI",):
    global window

    if not width:
        screen = _app.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()

    else:
        screen_width=width

    if screen_width < 1280:
        font_size = 7
    elif screen_width < 1920:
        font_size = 8
    else:
        font_size = 9

    _app.setFont(QFont(_font, font_size))
    plt.rcParams.update({"font.size": font_size + 3, "font.family": _font})

def check_resolution_change():
    """ Vérifie si la résolution a changé et met à jour la police si besoin """
    global last_width  # On garde la dernière largeur connue
    screen = app.screenAt(window.geometry().center())
    current_width = screen.size().width()

    if current_width != last_width:
        update_font(app,current_width)
        last_width = current_width

if __name__ == "__main__":

    sys.excepthook = excepthook
    app = QApplication(sys.argv)

    window = Data_Viz_Window()
    window.showMaximized()

    update_font(app)
    app.setStyle('Fusion')

    # Timer for screen resolution check
    last_width = app.primaryScreen().size().width()
    timer = QTimer()
    timer.timeout.connect(check_resolution_change)
    timer.start(500)  # Vérifie toutes les 500 ms

    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    file_name = '00189-VNIR-mock-up.h5'
    filepath = folder + file_name
    window.open_hypercubes_and_GT(filepath=filepath)

    sys.exit(app.exec_())

