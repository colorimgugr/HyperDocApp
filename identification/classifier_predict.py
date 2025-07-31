
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import sys

import numpy as np
from scipy.interpolate import interp1d
import scipy.io

from hypercubes.hypercube import*

import joblib
import torch
from torchvision import models
import torch.nn as nn

## fonctions and class
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super().__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Création du canvas Matplotlib
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Exemple : image aléatoire
        img = np.random.rand(100, 100, 3)
        self.canvas.ax.imshow(img)
        self.canvas.ax.set_title("Image VNIR/SWIR")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Matplotlib dans Qt")
        self.resize(800, 600)


def interpolate_cube(cube, _wl_step, interp_kind='linear'):

    interp_func = interp1d(
        cube.wl,
        cube.data,
        kind=interp_kind,
        axis=2,
        bounds_error=False,
        fill_value=(cube.data[:, :, 0], cube.data[:, :, -1])
    )

    wl_i = int(cube.wl[0] / wl_step) * wl_step
    wl_e = int(cube.wl[-1] / wl_step + 0.5) * wl_step
    wl_interp = np.arange(wl_i, wl_e + wl_step, wl_step)

    data_cube_interpolated = interp_func(wl_interp)

    return data_cube_interpolated,wl_interp


# load training data and train model
## Here 261 features : spectrum = concatenate (VNIR [450:950 include] , SWIR[955:1700 included])

wl_step=5 # data with each 5 nm wl

## load data to predict
folder=r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\identification'
files={'VNIR':os.path.join(folder,'01378-VNIR-royal.h5'),'SWIR':os.path.join(folder,'01500-SWIR-royal.h5')}
hyps={}

for key in files:
    hyps[key]=Hypercube(filepath=files[key],load_init=True)
    hyp=hyps[key]
    hyp.data,hyp.wl=interpolate_cube(hyp,wl_step)
print('[DATA] loaded')

## concatenate cube

if not (hyps['VNIR'].data.shape[0]==hyps['SWIR'].data.shape[0] and hyps['VNIR'].data.shape[1]==hyps['SWIR'].data.shape[1]):
    print('Cube not same shape. You must registered them together first')

H,W=hyps['VNIR'].data.shape[0],hyps['VNIR'].data.shape[1]

wl_limits_concatenate={'VNIR':(400,950),'SWIR':(955,1700)}
hyps_cut={}
for key in hyps:
    index_limits=(np.argmin(np.abs(hyps[key].wl-wl_limits_concatenate[key][0])),np.argmin(np.abs(hyps[key].wl-wl_limits_concatenate[key][1])))
    data_cut=hyps[key].data[:,:,index_limits[0]:index_limits[1]+1]
    wl_cut=hyps[key].wl[index_limits[0]:index_limits[1]+1]
    hyps_cut[key]=Hypercube(data=data_cut,wl=wl_cut,cube_info=hyps[key].cube_info)



# todo : if spectral range camera user < spectral range data set -> cut dataset before training model. If model already trained...ignore features


## check data size

if not (hyps_cut['SWIR'].data.shape[2]+hyps_cut['VNIR'].data.shape[2])==X_train.shape[1]:
    print('Spectral information of not same shape. Please check spectral range')

data_fused=np.concatenate((hyps_cut['VNIR'].data,hyps_cut['SWIR'].data),axis=2).reshape(-1,X_train.shape[1])
wl_fused=np.concatenate((hyps_cut['VNIR'].wl,hyps_cut['SWIR'].wl))

img_raw=hyps['VNIR'].data[:,:,[50,30,10]]

