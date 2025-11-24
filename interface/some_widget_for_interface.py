from PyQt5.QtWidgets import (QProgressBar,QLabel,QDialog, QVBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand,QGraphicsRectItem
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor
from PyQt5.QtCore import QRect, QPoint, Qt, QSize,pyqtSignal, QPointF,QRectF

import cv2
import os
import numpy as np

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

class ZoomableGraphicsView(QGraphicsView):
    middleClicked = pyqtSignal(QPointF) # suppres features
    moveFeatureStart = pyqtSignal(QPointF) # movefeatures
    moveFeatureUpdate = pyqtSignal(QPointF)
    moveFeatureEnd = pyqtSignal()
    selectionChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCursor(Qt.OpenHandCursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.pixmap_item = None  # check if image loaded

        # Rubber band selection
        self.enable_rect_selection = True
        self.origin = QPoint()
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        self._selecting = False
        self.last_rect_item = None
        self.rect_coords = None

        # if overlay
        self._selection_overlay = None

        # Moving point
        self.editing_match = None  # (match_index, side: "left"/"right")
        self.setMouseTracking(True)  # enable mouseMoveEvent without press

    def setImage(self, pixmap):
        # self.clear_rectangle()
        self.scene().clear()
        self._selection_overlay = None
        self.last_rect_item = None
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))

    def fitImage(self, scale_factor=0.8):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.scale(scale_factor, scale_factor)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom, zoom)

    def mousePressEvent(self, event):
        if (event.button() == Qt.RightButton and self.pixmap_item and self.enable_rect_selection): # rectangle selection
            self.viewport().setCursor(Qt.CrossCursor)
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self._selecting = True

        elif event.button() == Qt.MiddleButton and self.pixmap_item: # feature supress
            scene_pos = self.mapToScene(event.pos())
            self.middleClicked.emit(scene_pos)

        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier and self.pixmap_item: # feature supress
            scene_pos = self.mapToScene(event.pos())
            self.middleClicked.emit(scene_pos)
            event.accept()
            return

        elif event.button() == Qt.LeftButton and event.modifiers() == Qt.AltModifier and self.pixmap_item:
            scene_pos = self.mapToScene(event.pos())
            self.moveFeatureStart.emit(scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selecting:
            max_w = self.viewport().width() - 1
            max_h = self.viewport().height() - 1
            x = min(max(event.pos().x(), 0), max_w)
            y = min(max(event.pos().y(), 0), max_h)
            rect = QRect(self.origin, QPoint(x, y)).normalized()
            self.rubber_band.setGeometry(rect)

        elif self.editing_match is not None:
            scene_pos = self.mapToScene(event.pos())
            self.moveFeatureUpdate.emit(scene_pos)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.pixmap_item and self._selecting:
            self.rubber_band.hide()
            self.viewport().setCursor(Qt.OpenHandCursor)
            self._selecting = False

            # Convert view coords to scene coords
            start_scene = self.mapToScene(self.origin)
            end_scene = self.mapToScene(event.pos())

            # Convert scene coords to pixmap (image) coords
            p1 = self.pixmap_item.mapFromScene(start_scene)
            p2 = self.pixmap_item.mapFromScene(end_scene)

            x1, y1 = int(p1.x()), int(p1.y())
            x2, y2 = int(p2.x()), int(p2.y())

            # Clamp aux bords de l'image
            w, h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Erase previous rectangle
            self.clear_selection_overlay()
            self.clear_rectangle()

            # Création du rectangle avec coins clampés
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            width, height = x_max - x_min, y_max - y_min

            if width < 2 or height < 2:
                self.rect_coords = None
                self.last_rect_item = None
                return

            self.rect_coords = [x_min, y_min, width, height]
            self.last_rect_item = self.scene().addRect(
                x_min, y_min, width, height, QPen(QColor("red"))
            )

        if self.editing_match is not None:
            self.moveFeatureEnd.emit()
            self.editing_match = None

        self.selectionChanged.emit()
        super().mouseReleaseEvent(event)

    def get_rect_coords(self):
        return self.rect_coords

    def clear_rectangle(self):
        if self.last_rect_item:
            try:
                self.scene().removeItem(self.last_rect_item)
            except:
                None
        self.last_rect_item = None
        self.rect_coords=None

    # def add_selection_overlay(self, rect: QRectF):
    #     # Remove previous overlay if any
    #     if hasattr(self, '_selection_overlay') and self._selection_overlay is not None:
    #         self.scene().removeItem(self._selection_overlay)
    #
    #     overlay = QGraphicsRectItem(rect)
    #     overlay.setBrush(QColor(0, 255, 0, 80) ) # Green with transparency
    #     overlay.setPen(QPen(Qt.green, 2, Qt.DashLine))
    #     overlay.setZValue(10)  # Ensure it's above the image
    #     self.scene().addItem(overlay)
    #     self._selection_overlay = overlay

    def add_selection_overlay(self, rect: QRectF,surface=True):
        # Remove previous overlay if any
        if hasattr(self, '_selection_overlay') and self._selection_overlay is not None:
            try:
                self.scene().removeItem(self._selection_overlay)
            except RuntimeError:
                pass  # item déjà supprimé côté C++
            self._selection_overlay = None

        overlay = QGraphicsRectItem(rect)
        if surface:
            overlay.setBrush(QColor(0, 255, 0, 80))  # Green with transparency
        overlay.setPen(QPen(Qt.red, 1, Qt.DashLine))
        overlay.setZValue(10)  # On top
        self.scene().addItem(overlay)
        self._selection_overlay = overlay

    def clear_selection_overlay(self):
        if hasattr(self, '_selection_overlay') and self._selection_overlay is not None:
            self.scene().removeItem(self._selection_overlay)
            self._selection_overlay = None


class LoadingDialog(QDialog):
    def __init__(self, message="Loading...", filename=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        if filename:
            label_file = QLabel(f"<i>{os.path.basename(filename)}</i>")
            label_file.setAlignment(Qt.AlignCenter)
            layout.addWidget(label_file)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # mode indéterminé
        layout.addWidget(self.progress)

        self.setFixedWidth(350)


