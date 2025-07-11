# cd C:\Users\Usuario\Documents\GitHub\Hypertool\ground_truth
# python -m PyQt5.uic.pyuic -o registration_window.py registration_window.ui
# pyinstaller  --exclude-module tensorflow --exclude-module torch --icon="GT_icon.ico" --add-data "ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv;ground_truth"  ground_truth_tool.py
# opts   --noconsole --onefile

# Gnl
import os
import sys

## GUI
from PyQt5 import QtCore
from PyQt5.QtGui    import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import ( QSplitter,
    QApplication,QSizePolicy, QGraphicsScene, QGraphicsPixmapItem,QRubberBand,QWidget, QFileDialog, QMessageBox,QInputDialog , QSplitter,QGraphicsView,QLabel,
)
from PyQt5.QtCore import Qt, QEvent, QRect, QRectF, QPoint, QSize

# Graphs
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib import colormaps
from matplotlib.path import Path

# Maths
import numpy as np
import cv2
from scipy.spatial import distance as spdist

# Intern
from hypercubes.hypercube import Hypercube,CubeInfoTemp
from ground_truth.GT_table_viz import LabelWidget
from ground_truth.ground_truth_window import Ui_GroundTruthWidget
from interface.some_widget_for_interface import LoadingDialog


# todo : give GT labels names and number for RGB code ? -> save GT in new dataset of file + png
# todo : link to cube_info (read and fill)
# todo : actualize GT_cmp if label added OR load from default GT_table
# todo : check if cube already hace a GT map done (looking at GT labels for example)
# todo : upload GT_cmap from csv ?

## GT colors
GT_cmap=np.array([[0.        , 1.        , 0.24313725, 0.22745098, 0.37254902,
        0.26666667, 0.05882353, 0.31764706, 0.36078431, 0.34509804,
        0.5372549 , 0.93333333, 0.9254902 , 0.90196078, 0.74901961,
        0.7372549 , 0.6745098 , 1.        , 0.88627451, 0.14901961,
        0.11764706, 0.2745098 , 0.        , 0.34901961, 0.03529412,
        0.2       , 0.46666667, 1.        , 0.50196078, 0.14117647,
        0.        , 0.07843137, 0.50588235, 0.71372549, 0.83921569,
        0.14509804, 0.        , 0.77647059, 0.99215686, 0.99215686,
        0.16078431, 0.83137255, 0.70588235, 0.63137255, 0.9254902 ,
        0.98431373, 0.90196078, 0.63137255, 0.61568627, 0.31764706,
        0.92156863, 0.90980392, 0.97254902, 0.96470588, 0.83529412,
        0.79215686, 0.43529412, 0.79607843, 0.70588235, 0.76078431,
        0.94117647, 0.23529412, 0.        , 0.20784314, 0.92156863,
        0.69411765, 0.6745098 , 0.63529412, 0.92156863, 0.83137255,
        0.68627451, 0.60784314, 0.6745098 , 0.52941176, 0.45098039,
        0.29411765, 0.31372549, 0.78431373, 0.76470588, 0.83137255,
        0.83137255, 0.45098039, 1.        , 0.29411765, 0.1372549 ,
        0.07843137, 0.12941176, 0.88235294, 0.77254902, 0.74901961,
        0.71372549, 0.88235294, 0.70980392, 0.6       , 0.6745098 ,
        0.47843137, 0.25490196, 0.54901961, 0.35294118, 0.29411765,
        0.75686275, 0.83137255, 0.63529412, 0.90588235, 0.50980392,
        0.71764706, 0.90588235, 0.68627451, 0.60784314, 0.92156863,
        0.97647059, 0.86666667, 0.33333333, 0.84313725, 0.14509804,
        0.06666667, 0.2627451 , 0.98431373, 0.90588235, 0.61568627,
        0.15686275, 0.45098039, 0.1372549 , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.49019608, 0.09803922,
        0.8627451 , 0.74509804, 0.58823529, 0.39215686, 0.19607843],
       [0.        , 1.        , 0.14901961, 0.1254902 , 0.24705882,
        0.16470588, 0.05882353, 0.14509804, 0.24705882, 0.2745098 ,
        0.54901961, 0.85098039, 0.83921569, 0.9372549 , 0.85882353,
        0.74509804, 0.78823529, 0.83921569, 0.8       , 0.21176471,
        0.1372549 , 0.11764706, 0.25490196, 0.4745098 , 0.54117647,
        0.39607843, 0.29411765, 0.56078431, 0.10196078, 0.58431373,
        0.40392157, 0.21176471, 0.77647059, 0.32941176, 0.4745098 ,
        0.17647059, 0.34901961, 0.17254902, 0.50980392, 0.49411765,
        0.70196078, 0.80392157, 0.75686275, 0.20392157, 0.74117647,
        0.85098039, 0.8745098 , 0.38823529, 0.44313725, 0.36078431,
        0.16470588, 0.84705882, 0.70588235, 0.76470588, 0.70196078,
        0.7372549 , 0.21176471, 0.70196078, 0.54509804, 0.64705882,
        0.94117647, 0.78431373, 0.2745098 , 0.09411765, 0.92156863,
        0.64705882, 0.25490196, 0.29411765, 0.56078431, 0.71764706,
        0.60784314, 0.67843137, 0.75686275, 0.49803922, 0.41176471,
        0.34117647, 0.8627451 , 0.7254902 , 0.83529412, 0.95294118,
        0.41176471, 0.5372549 , 0.71764706, 0.7254902 , 0.62745098,
        0.35294118, 0.17254902, 0.96078431, 0.68627451, 0.16470588,
        0.21568627, 0.63921569, 0.55294118, 0.52156863, 0.75294118,
        0.64705882, 0.29411765, 0.41176471, 0.21568627, 0.78431373,
        0.67843137, 0.90980392, 0.80392157, 0.32156863, 0.37254902,
        0.52156863, 0.7254902 , 0.25490196, 0.68627451, 0.84313725,
        0.85098039, 0.45098039, 0.82352941, 0.41176471, 0.38823529,
        0.54509804, 0.78039216, 0.77254902, 0.69411765, 0.43921569,
        0.30980392, 0.66666667, 0.22352941, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.49019608, 0.09803922,
        0.8627451 , 0.74509804, 0.58823529, 0.39215686, 0.19607843],
       [0.        , 1.        , 0.25490196, 0.14509804, 0.14117647,
        0.11764706, 0.05882353, 0.12941176, 0.25882353, 0.14509804,
        0.56470588, 0.78039216, 0.7254902 , 0.80392157, 0.78039216,
        0.56862745, 0.80392157, 0.79215686, 0.81960784, 0.15686275,
        0.15686275, 0.2       , 0.54901961, 0.63921569, 0.79607843,
        0.56862745, 0.26666667, 0.49019608, 0.25098039, 0.74509804,
        0.6       , 0.28627451, 0.88235294, 0.14117647, 0.32156863,
        0.48627451, 0.25098039, 0.1254902 , 0.21176471, 0.        ,
        0.82352941, 0.8       , 0.        , 0.19607843, 0.        ,
        0.38431373, 0.78823529, 0.37254902, 0.1254902 , 0.58039216,
        0.07843137, 0.70980392, 0.        , 0.34117647, 0.45098039,
        0.62352941, 0.1372549 , 0.44705882, 0.56470588, 0.80784314,
        0.78431373, 0.82352941, 0.39215686, 0.64705882, 0.82352941,
        0.60784314, 0.22352941, 0.26666667, 0.07843137, 0.55294118,
        0.05882353, 0.1372549 , 0.17647059, 0.02745098, 0.        ,
        0.29411765, 0.90196078, 0.25490196, 0.29411765, 0.33333333,
        0.41960784, 0.45098039, 0.23529412, 0.74117647, 0.66666667,
        0.50980392, 0.73333333, 0.7254902 , 0.49019608, 0.26666667,
        0.16862745, 0.19607843, 0.39215686, 0.16862745, 0.19607843,
        0.09411765, 0.11764706, 0.16862745, 0.21568627, 0.82352941,
        0.3254902 , 0.35294118, 0.25098039, 0.42352941, 0.37254902,
        0.35294118, 0.50980392, 0.34117647, 0.16470588, 0.11372549,
        0.25490196, 0.19607843, 0.14509804, 0.49803922, 0.20784314,
        0.3372549 , 0.18039216, 0.01960784, 0.09803922, 0.17647059,
        0.06666667, 0.06666667, 0.35686275, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.49019608, 0.09803922,
        0.8627451 , 0.74509804, 0.58823529, 0.39215686, 0.19607843]])

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.OpenHandCursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None  # check if image loaded

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

class GroundTruthWidget(QWidget, Ui_GroundTruthWidget):
    cubeLoaded = QtCore.pyqtSignal(str)
    cube_saved = QtCore.pyqtSignal(CubeInfoTemp)

    def __init__(self, parent=None,cubeInfo=None):
        super().__init__(parent)
        # Set up UI from compiled .py
        self.setupUi(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) ##all creen of possible

        self.selecting_pixels = False # mode selection ref activated
        self._pixel_selecting = False  # for manual pixel selection for dragging mode
        self.erase_selection = False # erase mode on or off
        self._pixel_coords = []  # collected  (x,y) during dragging
        self._preview_mask = None # temp mask during dragging pixel selection
        self.class_info = {}         #dictionnary of lists :  {key:[label GT, name GT,(R,G,B)]}
        self.class_colors ={}  # color of each class
        n0 = self.nclass_box.value()

        # Replace placeholders with custom widgets
        self._replace_placeholder('viewer_left', ZoomableGraphicsView)
        self._replace_placeholder('viewer_right', ZoomableGraphicsView)
        self._promote_canvas('spec_canvas', FigureCanvas)

        self.viewer_left.viewport().installEventFilter(self)
        self.viewer_right.viewport().installEventFilter(self)

        # Enable live spectrum tracking
        self.viewer_left.viewport().setCursor(Qt.CrossCursor) # curseur croix
        self.viewer_left.viewport().setMouseTracking(True)
        self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)

        # Promote spec_canvas placeholder to FigureCanvas
        self.spec_canvas_layout = self.spec_canvas.layout() if hasattr(self.spec_canvas, 'layout') else None
        self.init_spectrum_canvas()
        self.spec_canvas.setVisible(True)
        self.show_selection=True
        self.live_spectra_update=True

        # State variables
        self.cube = None
        self.data = None
        self.wl= None
        self.current_cube_info=None
        self.cls_map = None
        self.samples = {} # to save pixels spectra samples for GT
        self.sample_coords = {c: set() for c in self.samples.keys()} # to remember coord of pixel samples
        self.alpha = self.horizontalSlider_transparency_GT.value() / 100.0
        self.mode = self.comboBox_ClassifMode.currentText()
        self.hyps_rgb_chan_DEFAULT=[0,0,0] #default rgb channels (in int nm)
        self.hyps_rgb_chan=[0,0,0] #current rgb (in int nm)
        self.class_means = {} #for spectra of classe
        self.class_ncount={} #for npixels classified
        self.selected_bands=[]
        self.selected_span_patch=[]

        # Connect widget signals
        self.load_btn.clicked.connect(self.load_cube)
        self.run_btn.clicked.connect(self.run)
        self.comboBox_ClassifMode.currentIndexChanged.connect(self.set_mode)
        self.pushButton_class_selection.toggled.connect(self.on_toggle_selection)
        self.pushButton_erase_selected_pix.toggled.connect(self.on_toggle_erase)
        self.checkBox_see_selection_overlay.toggled.connect(self.toggle_show_selection)
        self.pushButton_merge.clicked.connect(self.merge_selec_GT)
        self.pushButton_class_name_assign.clicked.connect(self.open_label_table)
        self.pushButton_band_selection.toggled.connect(self.band_selection)
        self.pushButton_keep_GT.clicked.connect(self.keep_GT)
        self.pushButton_reset.clicked.connect(self.reset_all)

        # RGB sliders <-> spinboxes
        self.sliders_rgb = [self.horizontalSlider_red_channel, self.horizontalSlider_green_channel,
                            self.horizontalSlider_blue_channel]
        self.spinBox_rgb = [self.spinBox_red_channel, self.spinBox_green_channel, self.spinBox_blue_channel]

        for sl, sp in zip(self.sliders_rgb,  self.spinBox_rgb):
            sl.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(sl.setValue)
            sl.valueChanged.connect(self.show_image)

        self.radioButton_rgb_user.toggled.connect(self.modif_sliders)
        self.radioButton_rgb_default.toggled.connect(self.modif_sliders)
        self.radioButton_grayscale.toggled.connect(self.modif_sliders)

        # Transparency slider
        self.horizontalSlider_transparency_GT.valueChanged.connect(self.on_alpha_change)

        # Live spectrum checkbox
        self.live_cb.stateChanged.connect(self.toggle_live)

        self.distance_funcs = {
            'sqeuclidean': spdist.sqeuclidean,
            'cosine': spdist.cosine,
            'correlation': spdist.correlation,
            'canberra': spdist.canberra,
        }

        # init stretch of each layout in QSplitters
        self.splitter.setStretchFactor(0, 1) #init stretch of images and spectra
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 600])
        self.splitter.setHandleWidth(2)

        self.splitter_2.setStretchFactor(0,4) #init stretch of image hyp and GT
        self.splitter_2.setStretchFactor(1, 1)

        # style poignée QSplitter
        self.splitter.setHandleWidth(2)
        self.splitter_2.setHandleWidth(4)
        self.splitter.setStyleSheet("""QSplitter::handle {background-color: darkgray;}""")
        self.splitter_2.setStyleSheet("""QSplitter::handle {background-color: darkgray;}""")

        self.set_mode()

    def eventFilter(self, source, event):
        mode = self.comboBox_pixel_selection_mode.currentText()

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            return False      ## to dont block drag

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton and (self.selecting_pixels or self.erase_selection):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            print('Clicked OK')
            pos = self.viewer_left.mapToScene(event.pos())
            x0, y0 = int(pos.x()), int(pos.y())
            if mode == 'pixel':
                # on commence la collecte
                self._pixel_selecting = True
                self._pixel_coords = [(x0, y0)]
                return True
            elif mode == 'rectangle':
                # début du drag
                from PyQt5.QtWidgets import QRubberBand
                self.origin = event.pos()
                self.rubberBand = QRubberBand(QRubberBand.Rectangle,
                                              self.viewer_left.viewport())
                self.rubberBand.setGeometry(self.origin.x(),
                                            self.origin.y(), 1, 1)
                self.rubberBand.show()
                return True
            elif mode == 'ellipse':
                from PyQt5.QtWidgets import QGraphicsEllipseItem
                from PyQt5.QtGui import QPen

                self.origin = event.pos()
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                self.ellipse_item = QGraphicsEllipseItem()
                self.ellipse_item.setPen(pen)
                self.ellipse_item.setBrush(Qt.transparent)
                self.viewer_left.scene().addItem(self.ellipse_item)
                return True

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            if not self.selecting_pixels:
                self.live_spectra_update=not self.live_spectra_update

        # 2) Mouvement souris → mise à jour de la selection en cours
        if event.type() == QEvent.MouseMove and self._pixel_selecting and mode == 'pixel':
            if not (self.selecting_pixels or self.erase_selection):
                return False
            pos = self.viewer_left.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            if (x, y) not in self._pixel_coords:
                self._pixel_coords.append((x, y))
            if self._preview_mask is None:
                H, W = self.data.shape[:2]
                self._preview_mask = np.zeros((H, W), dtype=bool)

            self._preview_mask[y, x] = True
            self.show_image(preview=True)

            return True

        if event.type() == QEvent.MouseMove and hasattr(self, 'rubberBand'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            self.rubberBand.setGeometry(
                QRect(self.origin, event.pos()).normalized()
            )
            return True

        if event.type() == QEvent.MouseMove and mode == 'ellipse' and hasattr(self, 'ellipse_item'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            sc_orig = self.viewer_left.mapToScene(self.origin)
            sc_now = self.viewer_left.mapToScene(event.pos())
            x0, y0 = sc_orig.x(), sc_orig.y()
            x1, y1 = sc_now.x(), sc_now.y()
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.ellipse_item.setRect(rect)
            return True

        # 3) Relâchement souris → calcul de la sélection

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton and mode == 'pixel' and self._pixel_selecting :
            if not (self.selecting_pixels or self.erase_selection):
                return False
            print('realeased OK')
            # get pixels
            coords = self._pixel_coords.copy()
            #  Si au moins 3 points, propose de fermer le cheminif min 3 points, propose contour
            if len(coords) >= 3:
                reply = QMessageBox.question(
                    self, "Close Path?",
                    "You have selected multiple pixels.\n"
                    "Do you want to close the path and include all pixels inside the contour?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    pts = np.array(coords)
                    poly = Path(pts)
                    x0, y0 = pts[:, 0].min().astype(int), pts[:, 1].min().astype(int)
                    x1, y1 = pts[:, 0].max().astype(int), pts[:, 1].max().astype(int)
                    filled = list(coords)
                    for yy in range(y0, y1 + 1):
                        for xx in range(x0, x1 + 1):
                            if poly.contains_point((xx, yy)):
                                filled.append((xx, yy))

                    # to avoid dobbles
                    seen = set()
                    coords = []
                    for p in filled:
                        if p not in seen:
                            seen.add(p)
                            coords.append(p)

            if self.erase_selection:
                self._handle_erasure(coords)
            else :
                self._handle_selection(coords) # close selection

            # ready to new selection
            self._pixel_selecting = False
            self._erase_selecting = False
            self._preview_mask = None
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'rubberBand'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            rect = self.rubberBand.geometry()
            self.rubberBand.hide()
            # coins en coordonnées image
            tl = self.viewer_left.mapToScene(rect.topLeft())
            br = self.viewer_left.mapToScene(rect.bottomRight())
            x0, y0 = int(tl.x()), int(tl.y())
            x1, y1 = int(br.x()), int(br.y())
            # liste de tous les pixels dans le rectangle
            coords = [
                (xx, yy)
                for yy in range(max(0, min(y0, y1)), min(self.data.shape[0], max(y0, y1) + 1))
                for xx in range(max(0, min(x0, x1)), min(self.data.shape[1], max(x0, x1) + 1))
            ]

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection

            del self.rubberBand
            return True

        if event.type() == QEvent.MouseButtonRelease and hasattr(self, 'ellipse_item'):
            if not (self.selecting_pixels or self.erase_selection):
                return False
            rect = self.ellipse_item.rect()
            self.viewer_left.scene().removeItem(self.ellipse_item)
            del self.ellipse_item

            cx, cy = rect.center().x(), rect.center().y()
            rx, ry = rect.width() / 2, rect.height() / 2
            x0, x1 = int(rect.left()), int(rect.right())
            y0, y1 = int(rect.top()), int(rect.bottom())

            coords = []
            for yy in range(max(0, y0), min(self.data.shape[0], y1 + 1)):
                for xx in range(max(0, x0), min(self.data.shape[1], x1 + 1)):
                    if ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1:
                        coords.append((xx, yy))

            if self.erase_selection:
                self._handle_erasure(coords)
            else:
                self._handle_selection(coords)  # close selection
            return True

        # 4) Mouvement souris pour le live spectrum
        if source is self.viewer_left.viewport() and event.type() == QEvent.MouseMove and self.live_spectra_update:
            if self.live_cb.isChecked() and self.data is not None:
                pos = self.viewer_left.mapToScene(event.pos())
                x,y=int(pos.x()),int(pos.y())
                H, W = self.data.shape[0], self.data.shape[1]
                if 0 <= x < W and 0 <= y < H:
                    self.update_spectra(x, y)

            return True

        # return super().eventFilter(source, event)
        return False

    def keep_GT(self):
        # On envoie dans metadata : map de GT, class_counts (pixels_averaged), 'GT_cmap','spectra_mean','spectra_std'

        if self.cls_map is None:
            QMessageBox.warning(self, "Warning", "Nothig to keep. Launch segmentation first")
            return

        self.cube.cube_info.metadata_temp['pixels_averaged']=list(self.class_ncount.values())
        GT_name=[]
        GT_num=[]
        for key in self.class_info:
            GT_num.append(str(self.class_info[key][0]))
            GT_name.append(self.class_info[key][1])

        self.cube.cube_info.metadata_temp['GTLabels']=[GT_num,GT_name]
        self.cube.cube_info.metadata_temp['GT_cmap']=GT_cmap

        self.cube.cube_info.metadata_temp['spectra_mean']=list(self.class_means.values())
        self.cube.cube_info.metadata_temp['spectra_std']=list(self.class_stds.values())

        if  hasattr(self, 'GT_image'):
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save segmentation image",
                "",
                "PNG (*.png)"
            )
            if not path:
                return

            cv2.imwrite(path, self.GT_image)

            reply = QMessageBox.question(
                self, "Erase selection?",
                "Ground truth image saved in :\n{path} \n \n Do you also want to save updates metadata of the cube ?",

                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.cube.metadata=self.cube.cube_info.metadata_temp
                filepath=self.cube.cube_info.filepath
                ext = os.path.splitext(filepath)[1].lower()

                if ext ==  ".mat":
                    fmt='MATLAB'
                elif ext == ".h5":
                    fmt='HDF5'
                elif ext == ".hdf5":
                    fmt = 'HDF5'
                elif ext == '.hdr':
                    fmt='ENVI'
                else :
                    fmt = 'HDF5'

                self.cube.save(filepath,fmt=fmt)
                self.cube_saved.emit(self.cube.cube_info)

        else :
            print('No Overlay')

    def open_label_table(self):

        if getattr(sys, 'frozen', False): # pynstaller case
            BASE_DIR = sys._MEIPASS
        else :
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        csv_path = os.path.join(BASE_DIR, "ground_truth/Materials labels and palette assignation - Materials_labels_palette.csv")
        # csv_path = 'Materials labels and palette assignation - Materials_labels_palette.csv'
        self.class_win = LabelWidget(csv_path,self.class_info)
        self.class_win.resize(1000, 600)
        self.class_win.class_info_updated.connect(self.on_class_info_updated) # connect to signal from LabelWidget
        self.class_win.show()

    def on_class_info_updated(self, class_info):

        self.class_info=class_info

        for c, info in class_info.items():
            if c in self.class_colors:
                self.class_colors[c] = (info[2][2],info[2][1],info[2][0])

        self.show_image()
        self.update_legend()

    def start_pixel_selection(self):

        self.show_selection=True
        self.pushButton_class_selection.setText("Stop Selection")
        self.pushButton_erase_selected_pix.setChecked(False)
        self.live_spectra_update=False # to bloc tracking

        if len(self.samples)>0 :
            reply = QMessageBox.question(
                self, "Erase selection?",
                "Do you want to erase previous selection?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                H, W = self.selection_mask_map.shape
                # Réinitialise à -1 (aucune classe)
                self.selection_mask_map[:] = -1
                self.samples.clear()

        self.selecting_pixels = True
        # self.viewer_left.setDragMode(QGraphicsView.NoDrag)
        self.viewer_left.setCursor(Qt.CrossCursor)
        self.viewer_left.viewport().setCursor(Qt.CrossCursor)
        self.show_image()

    def toggle_show_selection(self):

        self.show_selection = self.checkBox_see_selection_overlay.isChecked()
        self.show_image()

    def stop_pixel_selection(self):

        self.selecting_pixels = False

        # ready to select
        self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)
        self.viewer_left.setCursor(Qt.ArrowCursor)
        self.viewer_left.viewport().setCursor(Qt.ArrowCursor)

        # remet le bouton à l'état initial
        self.pushButton_class_selection.setText("Start Selection")
        self.pushButton_class_selection.setChecked(False)

        # efface tout preview en cours
        self.selecting_pixels = False

        # enfin, on affiche l'image normale (sans preview ni sélection en cours)
        self.show_image()

    def on_toggle_erase(self, checked):
        self.erase_selection = checked

        if checked:
            self._pixel_selecting=False
            self.stop_pixel_selection()

            self.show_selection = True

            self.pushButton_erase_selected_pix.setText("Stop Erasing")
            self.pushButton_class_selection.setChecked(False)
            # self.viewer_left.setDragMode(QGraphicsView.NoDrag)
            self.viewer_left.setCursor(Qt.CrossCursor)

        else:
            self.pushButton_erase_selected_pix.setText("Erase Pixels")
            self.viewer_left.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewer_left.unsetCursor()

    def on_toggle_selection(self, checked: bool):

        if checked:
            self.erase_selection=False
            self.start_pixel_selection()
            self.update_legend()

        else:
            # fin du mode sélection
            self.stop_pixel_selection()

    def merge_selec_GT(self):
        """
        Fusionne les annotations manuelles (selection_mask_map)
        dans la carte de segmentation algorithmique (cls_map),
        puis met à jour les prototypes (moyennes et écart-types).
        """

        if self.cls_map is None:
            QMessageBox.warning(
                self, "Warning",
                "No segmentation done : Perform a segmentation and try again if needed."
            )
            return


        mask = (self.selection_mask_map >= 0) ##mask of manual selected

        if not mask.any():
            QMessageBox.information(
                self, "Info",
                "Not selected pixel to mergi with segmented result"
            )
            return

        self.checkBox_enable_segment.setChecked(False) # secure selection by disabled segmentation

        self.cls_map[mask] = self.selection_mask_map[mask] # assign manual selecte class in segmented image

        # update class prototypes
        unique_labels = np.unique(self.cls_map)
        new_means = {}
        new_stds = {}
        for c in unique_labels:
            # collecte les coordonnées dont cls_map == c
            ys, xs = np.where(self.cls_map == c)
            # construit un tableau (N_c x B) de leurs spectres
            spectra = np.stack([self.data[y, x, :] for x, y in zip(xs, ys)], axis=0)
            # moyenne et écart-type
            new_means[c] = np.mean(spectra, axis=0)
            new_stds[c] = np.std(spectra, axis=0)

        self.class_means = new_means
        self.class_stds = new_stds

        n = len(unique_labels)

        self.prune_unused_classes()
        self.show_image()
        self.update_counts()
        self.update_legend()

    def modif_sliders(self):
        max_wl = int(self.wl[-1])
        min_wl = int(self.wl[0])
        wl_step = int(self.wl[1] - self.wl[0])

        default=self.radioButton_rgb_default.isChecked()

        if self.radioButton_grayscale.isChecked():
            self.label_red_channel.setText('')
            self.label_green_channel.setText('')
            self.label_blue_channel.setText('Gray')
        else:
            self.label_red_channel.setText('Red')
            self.label_green_channel.setText('Green')
            self.label_blue_channel.setText('Blue')

        for i, element in enumerate(self.sliders_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        for i, element in enumerate(self.spinBox_rgb):
            element.setMinimum(min_wl)
            element.setMaximum(max_wl)
            element.setSingleStep(wl_step)
            if default:
                element.setValue(self.hyps_rgb_chan_DEFAULT[i])
            else:
                element.setValue(self.hyps_rgb_chan[i])
            if self.radioButton_rgb_default.isChecked():
                element.setEnabled(False)
            elif self.radioButton_grayscale.isChecked():
                if i == 2:
                    element.setEnabled(True)
                else:
                    element.setEnabled(False)
            else:
                element.setEnabled(True)

        self.show_image()

    def init_spectrum_canvas(self):
        placeholder = getattr(self, 'spec_canvas')
        parent = placeholder.parent()

        # Crée le canvas
        self.spec_fig = Figure(facecolor=(1, 1, 1, 0.1))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_facecolor((0.7,0.7,0.7,1))
        self.spec_ax.set_title('Spectra')
        self.spec_ax.grid()


        self.span_selector = SpanSelector(
            ax=self.spec_ax,  # votre axe “Spectrum”
            onselect=self._on_bandselect,  # callback
            direction="horizontal",  # sélection horizontale
            useblit=True,  # activer le “blitting”
            minspan=1.0,  # au moins 1 unité sur l’axe λ
            props=dict(alpha=0.3, facecolor='tab:blue')
        )

        self.span_selector.set_active(False)

        # Remplace dans le splitter ou dans le layout
        if isinstance(parent, QSplitter):
            idx = parent.indexOf(placeholder)
            placeholder.deleteLater()
            parent.insertWidget(idx, self.spec_canvas)
        elif parent.layout() is not None:
            layout = parent.layout()
            idx = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            layout.insertWidget(idx, self.spec_canvas)
        else:
            placeholder.deleteLater()
            self.verticalLayout.addWidget(self.spec_canvas)

        self.spec_canvas.setVisible(True)

    def _on_bandselect(self, lambda_min, lambda_max):
        """
        Callback  SpanSelector
        """

        if self._band_action is None:
            return

        # 1) S’assurer que lambda_min < lambda_max
        if lambda_min > lambda_max:
            lambda_min, lambda_max = lambda_max, lambda_min

        # 2) Conversion en indices d’onde
        idx_min = int(np.argmin(np.abs(self.wl - lambda_min)))
        idx_max = int(np.argmin(np.abs(self.wl - lambda_max)))

        # 3) update self.selected_bands
        if self._band_action == 'add':
            for idx in range(idx_min,idx_max+1):
                if idx not in self.selected_bands:
                    self.selected_bands.append(idx)

            print(f"Selected band : [{idx_min} → {idx_max}] "
                  f"({self.wl[idx_min]:.1f} → {self.wl[idx_max]:.1f} nm)")

            patch=self.spec_ax.axvspan(
                lambda_min, lambda_max,
                alpha=0.2, color='tab:blue'
            )

            self.selected_span_patch.append(patch)

        elif self._band_action == 'del':

            for idx in range(idx_min,idx_max+1):
                if idx in self.selected_bands:
                    self.selected_bands.remove(idx)

            self.selected_bands=sorted(self.selected_bands)

            for patch in self.selected_span_patch:  # reset all patch
                patch.remove()
                self.selected_span_patch = []

            bands={}
            i_band=0
            for i in range(len(self.selected_bands)-1): # get bands from index
                if (self.selected_bands[i+1] -self.selected_bands[i]) ==1:
                    try:
                        bands[i_band].append(self.selected_bands[i])
                    except:
                        bands[i_band]=[self.selected_bands[i]]
                else:
                    try:
                        bands[i_band].append(self.selected_bands[i])
                    except:
                        bands[i_band]=[self.selected_bands[i]]

                    i_band+=1


            # recreate patches
            for i_band in bands:
                lambda_min, lambda_max=self.wl[bands[i_band][0]],self.wl[bands[i_band][-1]]

                patch = self.spec_ax.axvspan(
                    lambda_min, lambda_max,
                    alpha=0.2, color='tab:blue'
                )

                self.selected_span_patch.append(patch)

        self.spec_canvas.draw_idle()

    def _handle_selection(self, coords):
        """Prompt for class and store spectra of the given coordinates."""
        max_cls = self.nclass_box.value() - 1
        labels = [str(i) for i in range(max_cls + 1)]

        # 2) Ouvrir un QInputDialog.getItem() au lieu de getInt()
        #    - on force l’édition à se faire via la liste déroulante
        cls_str, ok = QInputDialog.getItem(
            self,
            "Class",
            "Choose class label:",
            labels,
            0,  # index initial (par défaut on sélectionne “0”)
            False  # False = l’utilisateur ne peut pas taper autre chose que la liste
        )

        if not ok:
            return

        cls = int(cls_str)
        if cls not in self.class_colors:
            self._assign_initial_colors(cls)
        else:
            print({self.class_colors[cls]})

        # append spectra

        for x, y in coords:
            if not (0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]):
                continue

            # A) s’il appartenait déjà à une autre classe, on l’enlève
            old = self.selection_mask_map[y, x]
            if old >= 0 and old != cls:
                # retirer coord de sample_coords[old] et de samples[old]
                if (x, y) in self.sample_coords.get(old, set()):
                    self.sample_coords[old].remove((x, y))
                # reconstruire la liste des spectres pour old
                self.samples[old] = [
                    self.data[yy, xx, :]
                    for (xx, yy) in self.sample_coords.get(old, ())
                ]

            # B) on (ré)assigne le pixel à la classe cls
            self.selection_mask_map[y, x] = cls
            # ajouter dans sample_coords et samples si pas déjà présent
            if (x, y) not in self.sample_coords.setdefault(cls, set()):
                self.sample_coords.setdefault(cls, set()).add((x, y))
                self.samples.setdefault(cls, []).append(self.data[y, x, :])

            # 3) rafraîchir l’affichage
        self.show_image()
        self.update_legend()

    def _handle_erasure(self, coords):

        for x, y in coords:
            cls = self.selection_mask_map[y, x]
            if cls >= 0:
                # enlève du mask
                self.selection_mask_map[y, x] = -1
                # enlève des sets et listes
                if (x, y) in self.sample_coords.get(cls, set()):
                    self.sample_coords[cls].remove((x, y))
                # reconstruit self.samples[cls]
                self.samples[cls] = [
                    self.data[yy, xx, :]
                    for (xx, yy) in self.sample_coords.get(cls, [])
                ]
                if len(self.sample_coords.get(cls, [])) == 0:
                    # on supprime tous les attributs relatifs à cette classe
                    self.sample_coords.pop(cls, None)
                    self.samples.pop(cls, None)
                    self.class_colors.pop(cls, None)
                    self.class_means.pop(cls, None)
                    self.class_stds.pop(cls, None)

        self.prune_unused_classes()
        self.show_image()
        self.update_legend()

    def update_spectra(self,x=None,y=None):
        self.spec_ax.clear()
        x_graph = self.wl

        if x is not None and y is not None:
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                spectrum = self.data[y, x, :]
                # Spectre du pixel
                self.spec_ax.plot(x_graph, spectrum, label='Pixel')
                self.spec_ax.grid(color='black')

        # Spectres GT moyens ± std
        if self.checkBox_seeGTspectra.isChecked() and hasattr(self, 'class_means'):
            for c, mu in self.class_means.items():
                std = self.class_stds[c]
                b, g, r = self.class_colors[c]
                col = (r/255.0, g/255.0, b/255.0)
                self.spec_ax.fill_between(
                    x_graph, mu - std, mu + std,
                    color=col, alpha=0.3, linewidth=0
                )
                self.spec_ax.plot(
                    x_graph, mu, '--',
                    color=col, label=f"Class {c}"
                )
            if self.spec_ax.get_legend_handles_labels()[1]:
                self.spec_ax.legend(loc='upper right', fontsize='small')
            self.spec_ax.set_title(f"Spectra")
            self.spec_canvas.setVisible(True)

        for patch in self.selected_span_patch:
            # patch est un PolyCollection produit par axvspan()
            # On le remet dans l’axe courant :
            self.spec_ax.add_patch(patch)

            # 4) On rafraîchit le canvas
        self.spec_canvas.draw_idle()

    def on_alpha_change(self, val):
        self.alpha = val / 100.0
        self.show_image()

    def toggle_live(self, state):
        if not state:
            self.spec_canvas.setVisible(False)
        else:
            self.update_spectra()
            self.live_spectra_update=True

    def load_cube(self,cube_info=None,path=None):

        if self.cls_map is not None : # if work done, stop to permit saving before continue.
            reply = QMessageBox.question(
                self, "Erase previous selection ?",
                "Do you want to erase previous cube work ?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
               return

        if cube_info is not None:
            if path is None:
                try:
                    if cube_info.filepath is not None:
                        path=cube_info.filepath
                except:
                    pass
            else :
                if path !=cube_info.filepath :
                    QMessageBox.warning(self, "Warning", "Path  is different from the filepath of cubeInfo")
                    return

        if not path :
            print('Ask path for cube')
            path, _ = QFileDialog.getOpenFileName(
            self, "Open Hypercube", "", "Hypercube files (*.mat *.h5 *.hdr)"
            )
            if not path:
                return

        message_progress = "[Ground Truth Tool] Loading cube..."
        loading = LoadingDialog(message_progress, filename=path, parent=self)
        loading.show()
        QApplication.processEvents()

        try :
            cube = Hypercube(filepath=path, load_init=True)
        except:
            QMessageBox.information(self,"Problem at loading","Impossible to load this cube. Please check format.")
            loading.close()
            return

        loading.close()

        # todo : check if GT already done in the file

        if cube_info is None:
            if "GTLabels" in cube.metadata.keys():
                if len(cube.metadata["GTLabels"][0]) != 0:
                    reply = QMessageBox.question(
                        self, "Erase previous Ground Truth ?",
                        "Ground truth labels has been found in the file. \n Are you sure that you want to make a new Ground Truth for this cube ?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return

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

        self.reset_state()
        self.modif_sliders()
        self.show_image(path) # Notify the manager

    def load_cube_info(self, ci: CubeInfoTemp):
        self.cube.cube_info = ci

    def reset_state(self):
        """
        Réinitialise tous les états liés au cube courant pour repartir d'un état vierge.
        """
        # 1. Segmentation algorithmique
        self.cls_map = None
        # 2. Sélection manuelle
        if self.data is not None:
            H, W = self.data.shape[:2]
            self.selection_mask_map = np.full((H, W), -1, dtype=int)
        # 3. Samples et prototypes
        self.samples = {}
        self.sample_coords = {}
        self.class_means = {}
        self.class_stds = {}
        self.class_colors = {}
        # 4. Masques de preview/erase
        self._preview_mask = None
        if hasattr(self, '_erase_mask'):
            self._erase_mask = None
        # 5. UI
        # Masquer le canvas de spectres
        self.spec_canvas.setVisible(False)
        self.selecting_pixels = False
        self.erase_selection = False
        self.pushButton_class_selection.setChecked(False)
        self.pushButton_erase_selected_pix.setChecked(False)
        self.show_image()
        self.update_legend()
        # Remettre le slider de threshold à 100%
        if hasattr(self, 'horizontalSlider_threshold'):
            self.horizontalSlider_threshold.setValue(100)

    def set_mode(self):
        self.mode = self.comboBox_ClassifMode.currentText()
        if self.mode == 'Supervised':
            self.stackedWidget.setCurrentWidget(self.page_distance)
            self.label_metric.setText('Spectral distance')
        elif self.mode == 'Unsupervised':
            self.stackedWidget.setCurrentWidget(self.page_normalized)
            self.label_metric.setText('Spectral Normalization')


        self.show_image()

    def show_image(self, preview=False):
        if self.data is None:
            return

        #hyp image
        H, W, B = self.data.shape
                # Get band indices from spinboxes for RGB
        self.hyps_rgb_chan = [self.spinBox_red_channel.value(),
               self.spinBox_green_channel.value(),
               self.spinBox_blue_channel.value()]

        idx = [np.argmin(np.abs(self.hyps_rgb_chan[j] - self.wl)) for j in range(3)]
        if self.radioButton_grayscale.isChecked():
            idx=[idx[2],idx[2],idx[2]]

        rgb = self.data[:, :, idx]
        rgb = (rgb / np.max(rgb) * 255).astype(np.uint8)

        # overlay of GT

        if self.cls_map is None:
            overlay = rgb.copy()

        else:

            # 1) Construire seg_color (H x W x 3) en BGR

            H, W = self.cls_map.shape

            seg_color = np.zeros((H, W, 3), dtype=np.uint8)

            for cls, (b, g, r) in self.class_colors.items():
                mask = (self.cls_map == cls)

                # On applique la couleur b,g,r à tous les pixels de cette classe

                seg_color[mask] = [b, g, r]

            # 2) Si vous avez défini une classe “other” (indice = n_classes),

            #    et que vous n’avez pas de couleur pour elle, vous pouvez la mettre en grisé, ex.:

            other_idx = set(np.unique(self.cls_map)) - set(self.class_colors.keys())

            for cls in other_idx:
                gray = 128

                mask = (self.cls_map == cls)

                seg_color[mask] = [gray, gray, gray]

            # 3) Faire l’overlay final avec la transparence

            overlay = cv2.addWeighted(rgb, 1 - self.alpha, seg_color, self.alpha, 0)

        if self.cls_map is None:
            blank = np.zeros((H, W, 3), dtype=np.uint8)
            pix2 = self._np2pixmap(blank)
        else:
            # On recycle seg_color calculé plus haut :
            # s’il n’est pas dans une variable, recalculer de la même façon :
            seg_color2 = np.zeros((H, W, 3), dtype=np.uint8)
            for cls, (b, g, r) in self.class_colors.items():
                mask = (self.cls_map == cls)
                seg_color2[mask] = [b, g, r]
            other_idx = set(np.unique(self.cls_map)) - set(self.class_colors.keys())
            for cls in other_idx:
                gray = 128
                mask = (self.cls_map == cls)
                seg_color2[mask] = [gray, gray, gray]

            pix2 = self._np2pixmap(seg_color2)
            self.GT_image = seg_color2

        self.viewer_right.setImage(pix2)

        # overlay of selection blended to GT overlay

        current = overlay.copy()
        if self.selection_mask_map is not None and self.show_selection:
            mixed = overlay.copy()
            α = 0.7
            for cls, color in self.class_colors.items():
                mask2d = (self.selection_mask_map == cls)
                if not mask2d.any():
                    continue

                layer = np.zeros_like(overlay)
                layer[:] = color

                blended = cv2.addWeighted(overlay, 1 - α, layer, α, 0)

                mask3 = mask2d[:, :, None]
                current = np.where(mask3, blended, current)

        self.current_composite = current

        if preview and self._preview_mask is not None:
            base = self.current_composite
            layer = np.zeros_like(base)
            layer[..., :] = 0,0,255  # BGR = (0,0,0)
            mixed = cv2.addWeighted(base, 1-0.1, layer, 0.1, 0)
            mask3 = self._preview_mask[:, :, None]
            result = np.where(mask3, mixed, base)
            self.viewer_left.setImage(self._np2pixmap(result))
            return

        self.viewer_left.setImage(self._np2pixmap(self.current_composite))

    def update_legend(self):

        if len(self.class_ncount)!=0:

            for i in reversed(range(self.frame_legend.layout().count())):
                w = self.frame_legend.layout().itemAt(i).widget()
                self.frame_legend.layout().removeWidget(w)
                w.deleteLater()

            for c in sorted(self.class_colors):
                b, g, r = self.class_colors[c]
                txt=str(c)
                if self.class_ncount is not None :
                    txt+='-'+str(self.class_ncount[c])+'px'

                lbl = QLabel(txt)
                # lbl.setFixedSize(30, 20)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(
                    f"background-color: rgb({r},{g},{b});"
                    "color: white;"
                    "border-radius: 3px;"
                    "font-weight: bold;"
                )
                self.frame_legend.layout().addWidget(lbl)

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

    def _promote_canvas(self, name, canvas_cls):
        placeholder = getattr(self, name)
        parent = placeholder.parent()
        from PyQt5.QtWidgets import QSplitter

        # Crée le nouveau canvas
        canvas = canvas_cls()
        # Supprime l’ancien placeholder
        placeholder.deleteLater()

        if isinstance(parent, QSplitter):
            # cas splitter : insère au même emplacement
            idx = parent.indexOf(placeholder)
            parent.insertWidget(idx, canvas)
        else:
            # cas layout classique
            layout = parent.layout() or self.verticalLayout
            layout.addWidget(canvas)

        # Conserve refs pour live spectrum
        self.spec_canvas = canvas
        self.spec_fig = getattr(canvas, 'figure', None) or Figure()
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title('Spectrum')
        canvas.setVisible(False)

    def compute_distance(self, u, v):
        name = self.comboBox_distance.currentText()
        fn   = self.distance_funcs.get(name)
        return fn(u, v)

    def run(self):

        self.stop_pixel_selection()

        if not self.checkBox_enable_segment.isChecked():
            QMessageBox.warning(self, "Warning", "Enable segmentation with checkbox !")
            return

        if self.data is None:
            QMessageBox.warning(self, "Warning", "Load a cube !")
            return

        samples_seg={}
        if len(self.selected_bands) > 0:
            bandes = self.selected_bands
            bandes_sorted = sorted(bandes)
            data_seg = self.data[:, :, bandes_sorted]

        else:
            bandes_sorted=list(range(len(self.wl)))
            data_seg=self.data

        H, W, B_sel = data_seg.shape
        flat = data_seg.reshape(-1, B_sel)

        # 1) Unsupervised
        if self.mode == 'Unsupervised':
            if self.comboBox_normalized.currentText()=='Normalize':
                max_flat=np.max(flat,axis=1,keepdims=True)
                flat/=max_flat

            from sklearn.cluster import KMeans
            n = self.nclass_box.value()
            kmeans = KMeans(n_clusters=n).fit(flat)
            labels = kmeans.labels_

            # Final : reshape et affichage
            H, W = self.data.shape[:2]
            self.cls_map = labels.reshape(H, W)

            # stocke moyennes, écarts et colormap
            full_means = {}
            full_stds = {}
            for c in range(n):
                mask_c = (self.cls_map == c)  # True pour tous les pixels de la classe c
                pixels_spectre_complet = self.data[mask_c]  # shape = (N_pixels_classe, F)
                if pixels_spectre_complet.size == 0:
                    full_means[c] = np.zeros(self.data.shape[2])
                    full_stds[c] = np.zeros(self.data.shape[2])
                else:
                    full_means[c] = pixels_spectre_complet.mean(axis=0)
                    full_stds[c] = pixels_spectre_complet.std(axis=0)
            self.class_means = full_means
            self.class_stds = full_stds

        elif self.mode == 'Supervised':
            # 1) Récupère les prototypes des classes labellisées
            classes = sorted(self.samples.keys())
            if not classes:
                QMessageBox.warning(self, "Warning", "Choose Unsupervided classification or select references pixels and try again !")
                return

            means = {}
            for c in classes:
                full_spectra_c = np.vstack(self.samples[c])  # shape = (N_pixels_de_c, F)
                truncated_spectra_c = full_spectra_c[:, bandes_sorted]  # shape = (N_pixels_de_c, B_sel)
                means[c] = truncated_spectra_c.mean(axis=0)

            thr_pct = self.slider_class_thr.value()
            thr_frac = thr_pct / 100.0  # 0.0–1.0

            other_label = len(classes)
            labels = np.full(flat.shape[0], other_label, dtype=int)

            for i, pix in enumerate(flat):
                dists = np.array([self.compute_distance(pix, means[c])
                                  for c in classes])
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                if thr_pct == 100:
                    labels[i] = classes[min_idx]
                else:
                    # on normalise la distance entre 0 et 1 sur cet exemple
                    max_dist = dists.max() if dists.max() > 0 else 1.0
                    norm_dist = min_dist / max_dist
                    if norm_dist <= thr_frac:
                        labels[i] = classes[min_idx]

            full_means = {}
            full_stds = {}

            for c in range(len(classes)):
                mask_c = (self.cls_map == c)  # True pour tous les pixels de la classe c
                pixels_spectre_complet = self.data[mask_c]  # shape = (N_pixels_classe, F)
                if pixels_spectre_complet.size == 0:
                    full_means[c] = np.zeros(self.data.shape[2])
                    full_stds[c] = np.zeros(self.data.shape[2])
                else:
                    full_means[c] = pixels_spectre_complet.mean(axis=0)
                    full_stds[c] = pixels_spectre_complet.std(axis=0)
            self.class_means = full_means
            self.class_stds = full_stds

            # 5) reshape et préparation de l’affichage
            H, W = self.data.shape[:2]
            self.cls_map = labels.reshape(H, W)

            # on prend K = nombre de classes + 1 for “other”
            n_colors = other_label + 1

        self.prune_unused_classes()
        self._assign_initial_colors()
        self.show_image()
        self.update_counts()
        self.update_legend()
        self.update_spectra()

    def update_counts(self):
        labels, counts = np.unique(self.cls_map, return_counts=True)
        for cls, cnt in zip(labels, counts):
            self.class_ncount[cls]=cnt

    def _np2pixmap(self, img):
        from PyQt5.QtGui import QImage, QPixmap
        if img.ndim == 2:
            fmt = QImage.Format_Grayscale8
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format_RGB888
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], fmt)
        return QPixmap.fromImage(qimg).copy()

    def _assign_initial_colors(self,c=None):

        if c is not None :
            unique_labels=[c]
        elif self.cls_map is not None:
            unique_labels = np.unique(self.cls_map)
        else:
            return

        cmap = colormaps.get_cmap('tab10')

        for cls in unique_labels:
            if cls not in self.class_colors:
                # cmap renvoie un tuple RGBA avec floats 0..1
                r_f, g_f, b_f, _ = cmap(cls)
                # on convertit en entiers 0..255
                r, g, b = int(255 * r_f), int(255 * g_f), int(255 * b_f)
                # MAIS OpenCV attend BGR, donc on stocke (b,g,r)
                self.class_colors[cls] = (b, g, r)
                if cls not in self.class_info:
                    self.class_info[cls] = [None,None,(0,0,0)]
                self.class_info[cls][2]=(r,g,b)

    def prune_unused_classes(self):
        """
        Supprime de self.class_colors et self.class_info
        tous les labels qui ne figurent plus dans self.cls_map.
        """
        if self.cls_map is None:
            return

        labels_in_map = set(np.unique(self.cls_map))
        for d in (self.class_colors, self.class_info):
            for cls in list(d.keys()):
                if cls not in labels_in_map:
                    del d[cls]

    def band_selection(self,checked):
        if checked:

            try:
                msg = QMessageBox(self)
                msg.setWindowTitle("Bands selection")
                msg.setText("Add or suppress bands ")
                add_button = msg.addButton("Add", QMessageBox.AcceptRole)
                remove_button = msg.addButton("Suppress", QMessageBox.AcceptRole)
                reset_button=msg.addButton("Clear all bands", QMessageBox.AcceptRole)
                cancel_button = msg.addButton(QMessageBox.Cancel)
                msg.setDefaultButton(add_button)
                msg.exec_()

                if msg.clickedButton() == add_button:
                    self._band_action = 'add'
                elif msg.clickedButton() == remove_button:
                    self._band_action = 'del'
                elif msg.clickedButton() == reset_button:
                    print('reset')
                    self._band_action = None
                    self.selected_bands = []

                    for patch in self.selected_span_patch:  # reset patch
                        patch.remove()
                        self.selected_span_patch = []

                    self.pushButton_band_selection.setChecked(False)
                    self.spec_canvas.draw_idle()
                    return

                else:
                    self.span_selector.set_active(False)
                    self.pushButton_band_selection.setChecked(False)
                    return

                self.span_selector.set_active(True)
                self.pushButton_band_selection.setText('STOP SELECTION')
            except:
                QMessageBox.warning(
                    self, "Warning",
                    "No band selection choice"
                )
                self.pushButton_band_selection.setChecked(False)

                return

        else:
            self.span_selector.set_active(False)
            self.pushButton_band_selection.setText('Band selection')

    def reset_all(self):
        # Check if sure ?
        ans = QMessageBox.warning(self, 'Reset All',
                                  'If you reset you will loose all the work you have done here.\n \nAre you sure you want to reset this tool ? ',
                                  QMessageBox.Yes | QMessageBox.Cancel)
        if ans == QMessageBox.Cancel:
            return

        # Core data
        self.cube = None
        self.data = None
        self.wl = None
        self.current_cube_info = None
        self.cls_map = None
        self.GT_image = None

        # Selection and class data
        self.selection_mask_map = None
        self.samples.clear()
        self.sample_coords.clear()
        self.class_means.clear()
        self.class_stds.clear()
        self.class_colors.clear()
        self.class_info.clear()
        self.class_ncount.clear()

        # Band selections and spectrum patches
        for patch in self.selected_span_patch:
            try:
                patch.remove()
            except:
                pass
        self.selected_span_patch.clear()
        self.selected_bands.clear()
        self._band_action = None

        # Temporary masks
        self._preview_mask = None
        if hasattr(self, '_erase_mask'):
            self._erase_mask = None

        # Selection state
        self.selecting_pixels = False
        self.erase_selection = False
        self._pixel_selecting = False
        self._pixel_coords = []
        self.alpha = self.horizontalSlider_transparency_GT.value() / 100.0
        self.live_spectra_update = True

        # Spectrum plot UI
        self.spec_ax.clear()
        self.spec_canvas.draw_idle()
        self.spec_canvas.setVisible(False)
        self.pushButton_class_selection.setChecked(False)
        self.pushButton_erase_selected_pix.setChecked(False)
        self.checkBox_see_selection_overlay.setChecked(True)
        self.checkBox_seeGTspectra.setChecked(True)
        self.live_cb.setChecked(True)

        # Clear legend layout
        while self.frame_legend.layout().count():
            item = self.frame_legend.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset RGB sliders
        self.radioButton_rgb_default.setChecked(True)

        # Clear images
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        self.viewer_left.setImage(self._np2pixmap(blank))
        self.viewer_right.setImage(self._np2pixmap(blank))


if __name__=='__main__':

    app = QApplication(sys.argv)
    w = GroundTruthWidget()
    # folder=r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    # file_name='00278-SWIR-mock-up.h5'
    # filepath=folder+file_name
    # w.load_cube(path=filepath)
    w.show()
    sys.exit(app.exec_())

