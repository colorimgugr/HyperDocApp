import re

from hypercubes.hypercube import*
from metadata.metadata_dock import*

from PyQt5.QtWidgets import (
    QApplication,  QPushButton, QLabel, QVBoxLayout, QWidget, QScrollArea,QDialog,QFormLayout,
     QMessageBox
)

# TODO : ici ou dans hypercube - generation automatique des metadata.
# TODO : ajouter outils de generation de la valeur de la metadata (wl, bands,height,name,parent_cube,position,width)
# TODO : propose edit all Metadata : type formulaire with add_meta possibility
# TODO : save data

class MetadataTool(QWidget, Ui_Metadata_tool):

    def __init__(self,cube_info:CubeInfoTemp=CubeInfoTemp(), parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cube_info=cube_info
        self.meta_load=cube_info.metadata_temp # to keep in memory in order to reset

        # connect combobox
        self.comboBox_metadata.currentIndexChanged.connect(self.update_metadata_label)
        self.update_combo_meta(init=True)

        # stacked param init
        self.stacked_metadata.setCurrentIndex(0)
        self.textEdit_metadata.setReadOnly(True)

        #connect to edit
        self.checkBox_edit.toggled.connect(self._toggle_edit_metadata)

        # connect buttons
        self.pushButton_save.pressed.connect(self.keep_metadata)
        self.pushButton_cancel.pressed.connect(self.reset_metadata)
        self.pushButton_reset_one.pressed.connect(self.reset_metadatum)
        self.toolButton_up.clicked.connect(lambda : self.step_combo(-1))
        self.toolButton_down.clicked.connect(lambda : self.step_combo(+1))
        self.pushButton_see_all_metadata.clicked.connect(self.show_all_metadata)

    def step_combo(self, delta: int):
        combo = self.comboBox_metadata
        i = combo.currentIndex() + delta
        if 0 <= i < combo.count():
            combo.setCurrentIndex(i)
        # self.update_combo_meta()

    def update_combo_meta(self,init=False):

        last_key = self.comboBox_metadata.currentText()
        if last_key=='': last_key='cubeinfo'

        if init:
            self.comboBox_metadata.clear()

        if self.cube_info.metadata_temp :
            for key in self.cube_info.metadata_temp.keys():
                if key not in ['wl','GT_cmap','spectra_mean','spectra_std']:
                    if key in ['GTLabels','pixels_averaged']:
                        try:
                            len(self.cube_info.metadata_temp[key])
                            self.comboBox_metadata.addItem(f"{key}")
                        except:
                            pass

                    else:
                        self.comboBox_metadata.addItem(f"{key}")
                        if key==last_key:
                            self.comboBox_metadata.setCurrentText(key)

            self.update_metadata_label()

    def set_cube_info(self,cube:CubeInfoTemp):
        self.cube_info = cube
        self.meta_load = cube.metadata_temp.copy()
        self.label_file_name_meta.setText(os.path.basename(self.cube_info.filepath))
        self.update_combo_meta(init=True)

    def reset_metadatum(self):
        key = self.comboBox_metadata.currentText()
        self.cube_info.metadata_temp[key] = self.meta_load[key]
        self.update_metadata_label()

    def reset_metadata(self):
        self.cube_info.metadata_temp=self.meta_load
        self.update_metadata_label()

    def update_metadata_label(self):
        self.textEdit_metadata.setStyleSheet("QTextEdit  { color: black; }")
        key = self.comboBox_metadata.currentText()
        if key=='':
            key='cubeinfo'
        raw = self.cube_info.metadata_temp[key]
        match key:
            case 'GTLabels' | 'gtlabels':
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
                st=f'<b>{self.cube_info.metadata_temp[key]}</b>'
        self.label_metadata.setText(st)

        # edit text
        if key in ['GTLabels','gtlabels','bands','height','name','pixels_averaged','position','width']:
            self.textEdit_metadata.setReadOnly(True)
            self.textEdit_metadata.setStyleSheet("QTextEdit  { color: red; }")

        try:
            disp=raw
            self.textEdit_metadata.setText(disp)
        except :
            disp = str(raw)
            try:
                self.textEdit_metadata.setText(disp)

            except:
                self.textEdit_metadata.setText(repr(type(raw)))

        #print on console to debugg/analyse
        # if isinstance(raw, np.ndarray):
        #     # par exemple : forme et dtype
        #     type_print = f"array shape={raw.shape}, dtype={raw.dtype}"
        # else:
        #     type_print = repr(type(raw))
        # print(type_print)

    def keep_metadata(self):
        key = self.comboBox_metadata.currentText()
        raw_in=self.textEdit_metadata.toPlainText()
        meta_init = self.cube_info.metadata_temp[key]
        meta_valid=False

        if isinstance(meta_init,str):
            self.cube_info.metadata_temp[key]=raw_in
            meta_valid=True

        elif isinstance(meta_init,np.ndarray):
            if len(meta_init.shape)==1:
                raw_in=raw_in.replace('[','')
                raw_in=raw_in.replace(']','')
                raw_in = raw_in.replace('\n', '')
                list_temp=raw_in.split(' ')
                try :
                    self.cube_info.metadata_temp[key] = np.array(list_temp,dtype=meta_init.dtype)
                    meta_valid = True
                except :
                    try:
                        self.cube_info.metadata_temp[key] = np.array(list_temp, dtype=meta_init.dtype)
                        meta_valid = True
                    except:
                        meta_valid = False

            elif len(meta_init.shape) == 2:
                rows=re.findall(r"\[([^\]]+)\]", raw_in)
                list_temp=[]
                for r in rows:
                    elem=re.findall(r"'([^']*)'", r)
                    list_temp.append(elem)

                self.cube_info.metadata_temp[key] = np.array(list_temp, dtype=meta_init.dtype)
                meta_valid=True

        if not meta_valid:
            QMessageBox.warning(self,
                                "Warnings",  # titre de la fenêtre
                                "Metadata has not been keeped. Check structure."  # texte du message
                                )
            self.textEdit_metadata.setStyleSheet("QTextEdit  { color: red; }")

        else:
            self.textEdit_metadata.setStyleSheet("QTextEdit  { color: green; }")

    def _toggle_edit_metadata(self, editable: bool):
        """
        Basculer entre mode lecture (QLabel) et mode édition (QTextEdit).
        """
        self.stacked_metadata.setCurrentIndex(1 if editable else 0)
        self.textEdit_metadata.setReadOnly(not editable)
        self.pushButton_save.setEnabled(editable)
        self.textEdit_metadata.setStyleSheet("QTextEdit  { color: black; }")
        self.update_metadata_label()

    def show_all_metadata(self):


        if self.cube_info.metadata_temp is None:
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
        for key, val in self.cube_info.metadata_temp.items():
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


if __name__ == '__main__':
    folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    sample = '00189-VNIR-mock-up.h5'
    filepath = os.path.join(folder, sample)

    cube = Hypercube(filepath=filepath, load_init=True)

    app = QApplication(sys.argv)

    meta_window = MetadataTool()
    meta_window.set_cube_info(cube.cube_info)
    meta_window.update_combo_meta(init=True)
    meta_window.show()

    sys.exit(app.exec_())
