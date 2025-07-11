import re
import copy
import os

from hypercubes.hypercube import*
from metadata.metadata_dock import*

from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,  QPushButton, QLabel, QVBoxLayout, QWidget, QScrollArea,QDialog,QFormLayout,
     QMessageBox,QFileDialog,QDialogButtonBox,QHBoxLayout,QCheckBox,QLineEdit,QComboBox
)

class MetadataTool(QWidget, Ui_Metadata_tool):

    metadataChanged = pyqtSignal(object)
    cubeLoaded = QtCore.pyqtSignal(str)

    def __init__(self,cube_info:CubeInfoTemp=CubeInfoTemp(), parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cube_info = cube_info if cube_info is not None else CubeInfoTemp()
        self.meta_load = self.cube_info.metadata_temp.copy()
        self.hidden_meta=['wl','GT_cmap','spectra_mean','spectra_std','RGB']
        self.not_editable=['GTLabels','gtlabels','bands','height','pixels_averaged','position','width']

        # connect combobox
        self.comboBox_metadata.currentIndexChanged.connect(self.update_metadata_label)
        self.update_combo_meta(init=True)

        # stacked param init
        self.stacked_metadata.setCurrentIndex(0)
        self.textEdit_metadata.setReadOnly(True)

        #connect checkbox to edit
        self.checkBox_edit.toggled.connect(self.toggle_edit_metadata)

        # connect buttons
        self.pushButton_save.pressed.connect(self.keep_metadatum)
        self.pushButton_cancel.pressed.connect(self.reset_metadata)
        self.pushButton_reset_one.pressed.connect(self.reset_metadatum)
        self.toolButton_up.clicked.connect(lambda : self.step_combo(-1))
        self.toolButton_down.clicked.connect(lambda : self.step_combo(+1))
        self.pushButton_see_all_metadata.clicked.connect(self.show_all_metadata)
        self.pushButton_generateMeta.clicked.connect(self.generate_metadata)
        self.pushButton_add.clicked.connect(self.add_remove_metadatum)
        self.pushButton_valid_all_changes.clicked.connect(self.emit_updated_cube)
        self.pushButton_load_cube.clicked.connect(self.open_cube)

    def ask_if_modif(self,index):
        discard=True
        combo = self.comboBox_metadata
        key = combo.itemText(index)

        raw_in = self.textEdit_metadata.toPlainText().strip()
        actual_raw = self.get_raw_display(key).strip()

        if raw_in != actual_raw:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Metadatum modified")
            msg_box.setText("Do you want to <b>discard</b> the modification of the metadatum?")
            discard_button = msg_box.addButton("Discard change in metadatum", QMessageBox.AcceptRole)
            cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.exec()

            if msg_box.clickedButton() == cancel_button:
                discard=False

        return discard

    def step_combo(self, delta: int):
        combo = self.comboBox_metadata
        i = combo.currentIndex()

        # check if modif in edit Mode
        if self.checkBox_edit.isChecked():
            if not self.ask_if_modif(i):
                return

        # Go to the new key
        new_index = i + delta
        if 0 <= new_index < combo.count():
            combo.setCurrentIndex(new_index)

    def get_raw_display(self, key: str) -> str:
        """Used to make str for different type of metadata for the edit widget"""
        raw = self.cube_info.metadata_temp.get(key, "")

        # Méthode d’affichage comme dans update_metadata_label()
        if isinstance(raw, str):
            return raw

        elif isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                return ' '.join(str(v) for v in raw)
            elif raw.ndim == 2:
                lines = []
                for row in raw:
                    if isinstance(row, np.ndarray):
                        lines.append("[ " + ' '.join(f"'{v}'" for v in row) + " ]")
                    else:
                        lines.append(str(row))
                return '\n'.join(lines)

        elif isinstance(raw, list):
            return ' '.join(str(v) for v in raw)

        elif raw is None:
            return ""

        else:
            return str(raw)

    def update_combo_meta(self,init=False):

        last_key = self.comboBox_metadata.currentText()
        if last_key=='': last_key='cubeinfo'

        if init:
            self.comboBox_metadata.clear()

        if self.cube_info.metadata_temp :
            for key in self.cube_info.metadata_temp.keys():
                if key not in self.hidden_meta:
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

    def set_cube_info(self,cubeInfo:CubeInfoTemp):
        self.cube_info = copy.deepcopy(cubeInfo)
        self.meta_load = copy.deepcopy(cubeInfo.metadata_temp)
        self.label_file_name_meta.setText(os.path.basename(self.cube_info.filepath))
        self.update_combo_meta(init=True)

    def reset_metadatum(self):
        key = self.comboBox_metadata.currentText()
        self.cube_info.metadata_temp[key] = self.meta_load[key]
        self.update_metadata_label()

    def generate_metadata(self):
        """to propose to copy or create metadata"""
        # todo : add dialog to propose copy from other cube

        # check if already have metadata
        if len(self.cube_info.metadata_temp)!=0:
            ans = QMessageBox.question(
                self,
                "Metadata already exists",
                "Some of existing metadata will be overwritten. Continue ?",
                QMessageBox.Yes | QMessageBox.Cancel
            )

            if ans != QMessageBox.Yes:
                return

        #ask if create from nothig or copy from other cube

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Generate or copy ?")
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(
            "How do you want to generate your metadata set : \n -> from scratch ? \n -> from another cube ? ")
        scratch_button = msg_box.addButton("From scratch ", QMessageBox.ActionRole)
        copy_button = msg_box.addButton("From another cube", QMessageBox.ActionRole)
        msg_box.exec()

        if msg_box.clickedButton() == scratch_button:
            self.create_metadata_set()
        elif msg_box.clickedButton() == copy_button:
            self.copy_from_other_cube()

    def create_metadata_set(self):
        """ to propose to create metada """

        if self.cube_info is None or self.cube_info.filepath is None:
            QMessageBox.warning(self, "No Cube Loaded", "You must load a cube before creating metadata.")
            return

        data_shape = self.cube_info.data_shape  # (H, W, B)
        if data_shape is None or len(data_shape) != 3:
            QMessageBox.warning(self, "Invalid Data", "Cube data shape is invalid.")
            return

        metadata_template = {
            "aged": [0],
            "bands": [data_shape[2]],
            "cubeinfo": "",
            "date": "",
            "device": "",
            "height": [data_shape[0]],
            "illumination": "",
            "name": "",
            "number": "",
            "parent_cube": "",
            "reference_white": "",
            "restored": [0],
            "stage": "",
            "substrate": "",
            "texp": [0.0],
            "width": [data_shape[1]]
        }

        dialog = QDialog(self)
        dialog.setWindowTitle("Create Metadata")
        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        widgets = {}  # key: (checkbox, input)

        for key, default in metadata_template.items():
            hbox = QHBoxLayout()

            lineedit = QLineEdit()
            if isinstance(default, list):
                lineedit.setText(" ".join(str(v) for v in default))
            else:
                lineedit.setText(str(default))
            hbox.addWidget(lineedit)

            checkbox = QCheckBox()
            checkbox.setChecked(True)
            hbox.addWidget(checkbox)

            form_layout.addRow(QLabel(key), hbox)
            widgets[key] = (checkbox, lineedit)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_accept():
            for key, (chk, inp) in widgets.items():
                if chk.isChecked():
                    text = inp.text().strip()
                    # if text == "":
                    #     continue
                    if key in ["bands", "height", "width", "restored", "range", "texp"]:
                        try:
                            values = [float(v) if "." in v else int(v) for v in text.split()]
                            self.cube_info.metadata_temp[key] = np.array(values)
                        except Exception as e:
                            print(f"[CreateMetadata] Failed to convert {key}: {e}")
                            self.cube_info.metadata_temp[key] = text
                    else:
                        self.cube_info.metadata_temp[key] = text

            dialog.accept()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)

        dialog.setLayout(layout)
        dialog.exec_()

        # Mise à jour de l'interface
        self.update_combo_meta(init=True)

    def copy_from_other_cube(self):
        """Open a form to copy metadata from another cube with checkbox selection."""

        dialog = QDialog(self)
        dialog.setWindowTitle("Copy Metadata from Another Cube")
        dialog.setModal(True)
        main_layout = QVBoxLayout(dialog)

        # Button to load another cube
        load_button = QPushButton("Load cube to copy metadata")
        main_layout.addWidget(load_button)

        # Scrollable area for metadata fields
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QFormLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        widgets = {}  # Dictionary: key -> (checkbox, lineedit)
        metadata = {}  # Will hold the loaded metadata

        def load_cube():
            """Open file dialog and load metadata from another cube."""
            nonlocal metadata  # to reuse in accept_copy
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Cube", "", "Cube files (*.h5 *.hdr *.mat);;All Files (*)"
            )
            if not path:
                return
            try:
                new_cube = Hypercube(filepath=path, load_init=True)
                metadata = new_cube.cube_info.metadata_temp.copy()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load cube:\n{e}")
                return

            # Clear previous form fields
            for i in reversed(range(scroll_layout.count())):
                item = scroll_layout.itemAt(i)
                if item.widget():
                    item.widget().deleteLater()
            widgets.clear()

            # Add each metadata field
            for key, val in metadata.items():
                hbox = QHBoxLayout()
                lineedit = QLineEdit()
                if isinstance(val, (list, np.ndarray)):
                    lineedit.setText(" ".join(str(x) for x in np.array(val).flatten()))
                else:
                    lineedit.setText(str(val))

                if key in self.not_editable:
                    lineedit.setReadOnly(True)
                    lineedit.setStyleSheet("color: red;")

                hbox.addWidget(lineedit)

                checkbox = QCheckBox()
                checkbox.setChecked(True)
                hbox.addWidget(checkbox)

                widgets[key] = (checkbox, lineedit)
                scroll_layout.addRow(QLabel(key), hbox)

        load_button.clicked.connect(load_cube)

        # Dialog validation buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        main_layout.addWidget(buttons)

        def accept_copy():
            """Apply selected metadata to current cube."""
            for key, (chk, line) in widgets.items():
                if chk.isChecked():
                    if key in self.not_editable:
                        # Direct copy from the source metadata
                        self.cube_info.metadata_temp[key] = metadata[key]
                    else:
                        text = line.text().strip()
                        if text == "":
                            continue
                        try:
                            values = [float(x) if '.' in x else int(x) for x in text.split()]
                            self.cube_info.metadata_temp[key] = np.array(values) if len(values) > 1 else values[0]
                        except:
                            self.cube_info.metadata_temp[key] = text
            dialog.accept()

        buttons.accepted.connect(accept_copy)
        buttons.rejected.connect(dialog.reject)

        dialog.resize(700, 600)
        if dialog.exec_() == QDialog.Accepted:
            self.update_combo_meta(init=True)

    def add_remove_metadatum(self):
        """Open a dialog to add or remove a metadatum from metadata_temp"""

        dialog = QDialog(self)
        dialog.setWindowTitle("Add or Remove Metadata")
        layout = QVBoxLayout(dialog)

        # --- ADD PART ---
        add_group = QVBoxLayout()
        add_label = QLabel("<b>Add a metadatum</b>")
        form_add = QFormLayout()

        key_input = QLineEdit()
        value_input = QLineEdit()

        type_combo = QComboBox()
        type_combo.addItems(["str", "int", "float", "list", "ndarray"])

        add_button = QPushButton("Add")

        form_add.addRow("Name:", key_input)
        form_add.addRow("Value:", value_input)
        form_add.addRow("Type:", type_combo)
        form_add.addRow("", add_button)

        add_group.addWidget(add_label)
        add_group.addLayout(form_add)
        layout.addLayout(add_group)

        # --- REMOVE PART ---
        remove_group = QVBoxLayout()
        remove_label = QLabel("<b>Remove a metadatum</b>")
        form_remove = QFormLayout()

        remove_combo = QComboBox()
        remove_combo.addItems(list(self.cube_info.metadata_temp.keys()))
        remove_button = QPushButton("Remove")

        form_remove.addRow("Select key:", remove_combo)
        form_remove.addRow("", remove_button)

        remove_group.addWidget(remove_label)
        remove_group.addLayout(form_remove)
        layout.addLayout(remove_group)

        # --- CALLBACKS ---

        def add_metadata():
            key = key_input.text().strip()
            raw_val = value_input.text().strip()
            dtype = type_combo.currentText()

            if not key:
                QMessageBox.warning(dialog, "Error", "Key cannot be empty.")
                return

            try:
                if dtype == "str":
                    val = raw_val
                elif dtype == "int":
                    val = int(raw_val)
                elif dtype == "float":
                    val = float(raw_val)
                elif dtype == "list":
                    val = eval(raw_val)  # ex: "[1, 2, 3]"
                    if not isinstance(val, list):
                        raise ValueError
                elif dtype == "ndarray":
                    val = np.array(eval(raw_val))
                else:
                    raise ValueError("Unknown type")
            except Exception:
                QMessageBox.critical(dialog, "Error", f"Invalid value for type {dtype}.")
                return

            self.cube_info.metadata_temp[key] = val
            QMessageBox.information(dialog, "Success", f"Metadatum '{key}' added.")
            self.update_combo_meta(init=True)
            dialog.accept()

        def remove_metadata():
            key = remove_combo.currentText()
            if not key:
                return
            reply = QMessageBox.question(
                dialog,
                "Confirm",
                f"Do you really want to remove '{key}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.cube_info.metadata_temp.pop(key, None)
                QMessageBox.information(dialog, "Removed", f"Metadatum '{key}' removed.")
                self.update_combo_meta(init=True)
                dialog.accept()

        add_button.clicked.connect(add_metadata)
        remove_button.clicked.connect(remove_metadata)

        dialog.setLayout(layout)
        dialog.exec()

    def reset_metadata(self):
        """ all values back to initial copy named meta_load """
        self.cube_info.metadata_temp=self.meta_load
        self.update_metadata_label()

    def update_metadata_label(self):
        self.textEdit_metadata.setStyleSheet("QTextEdit  { color: black; }")
        key = self.comboBox_metadata.currentText()
        if key=='':
            try:
                key='cubeinfo'
                raw = self.cube_info.metadata_temp[key]
            except:
                pass

        raw = self.cube_info.metadata_temp[key]

        try :
            if key == 'GTLabels' | 'gtlabels':
                if len(raw.shape)==2:
                    st=f"GT indexes : <b>{(' , ').join(raw[0])}</b>  <br>  GT names : <b>{(' , ').join(raw[1])}</b>"
                elif len(raw.shape)==1:
                    st=f"GT indexes : <b>{(raw[0])}</b>  <br>  GT names : <b>{raw[1]}</b>"

            elif key == 'aged':
                st=f"The sample has been aged ? <br> <b>{raw}</b>"

            elif key == 'bands':
                st=f"The camera have <b>{raw[0]}</b> spectral bands."

            elif key == 'date':
                if len(raw)>1:info=raw
                else: info=raw[0]
                st=f"Date of the sample : <b>{info}</b>"

            elif key == 'device':
                st=f"Capture made with the device : <br> <b>{raw}</b>"

            elif key == 'illumination':
                st=f"Lamp used for the capture : <br> <b>{raw}</b>"

            elif key == 'name':
                st=f"Name of the minicube : <br> <b>{raw}</b>"

            elif key == 'number':
                st = f"Number of the minicube : <br> <b>{raw}</b>"

            elif key == 'parent_cube':
                st = f"Parent cube of the minicube : <br> <b>{raw}</b>"

            elif key == 'pixels_averaged':
                st = f"The number of pixels used for the <b>{len(raw)}</b> mean spectra of the GT materials are : <br> <b>{(' , ').join([str(r) for r in raw])}</b> "

            elif key == 'reference_white':
                st = f"The reference white used for reflectance measurement is : <br> <b>{raw}</b>"

            elif key == 'restored':
                    st = f"The sample has been restored ?  <br> <b> {['NO','YES'][raw[0]]}</b>"

            elif key == 'stage':
                st = f"The capture was made with a  <b>{raw}</b> stage"

            elif key == 'reference_white':
                st = f"The reference white used for reflectance measurement is : <br> <b>{raw}</b>"

            elif key == 'substrate':
                st = f"The substrate of the sample is : <br> <b>{raw}</b>"

            elif key == 'texp':
                st = f"The exposure time set for the capture was <b>{raw[0]:.2f}</b> ms."

            elif key == 'height':
                st = f"The height of the minicube <b>{raw[0]}</b> pixels."

            elif key == 'width':
                st = f"The width of the minicube <b>{raw[0]}</b> pixels."

            elif key == 'position':
                st = f"The (x,y) coordinate of the upper right pixel of the minicube in the parent cube is : <br> <b>({raw[0]},{raw[1]})</b>"

            elif key == 'range':
                val=['UV','VNIR : 400 - 1000 nm','SWIR : 900 - 1700 nm'][list(raw).index(1)]
                st = f"The range of the capture is : <br> <b>{val}</b>"

            else :
                st=f"<b>{self.cube_info.metadata_temp[key]}</b>"
        except:
            st=f"<b> !!! PROBLEM WITH METADATUM FORMAT !!! <br> <br> Here the raw :  <br> <br> {raw} </b>"

        self.label_metadata.setText(st)

        # edit text
        if key in self.not_editable:
            self.textEdit_metadata.setReadOnly(True)
            self.textEdit_metadata.setStyleSheet("QTextEdit  { color: red; }")
        else:
            self.textEdit_metadata.setReadOnly(False)

        disp = self.get_raw_display(key)
        self.textEdit_metadata.setText(disp)

    def keep_metadatum(self):
        """permit to keep the metadatum change"""
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
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Type problem")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("The type is not the same as initial metadatum type.\nDo you want to <b>force type of metadatum to string </b> or cancel modification ?")
            force_button = msg_box.addButton("Force type to string ? ", QMessageBox.AcceptRole)
            cancel_button = msg_box.addButton("Cancel modification", QMessageBox.RejectRole)
            msg_box.exec()

            if msg_box.clickedButton() == force_button:
                self.cube_info.metadata_temp[key]=raw_in
                self.textEdit_metadata.setStyleSheet("QTextEdit  { color: green; }")

            else :
                self.textEdit_metadata.setStyleSheet("QTextEdit  { color: red; }")

        else:
            self.textEdit_metadata.setStyleSheet("QTextEdit  { color: green; }")

    def toggle_edit_metadata(self):
        """
        Switch between read (QLabel) and edit (QTextEdit).
        """
        editable = self.checkBox_edit.isChecked()

        try:
            key = self.comboBox_metadata.currentText()
            self.cube_info.metadata_temp[key]
        except:
            print(self.cube_info)
            if self.checkBox_edit.isChecked():
                if self.cube_info.filepath is None :
                    QMessageBox.warning(self,'Problem with metadata','No metadata to edit.\nSend a cube to this tool first.')
                    return
                else:
                    qm=QMessageBox
                    ans=qm.question(self,'No metadata in the cube','Do you want to create metadata for this cube ?',qm.Yes | qm.No)
                    if ans==qm.Yes:
                        QMessageBox.information(self,'OK','Good for you...I will implement it')
                self.checkBox_edit.setChecked(False)
                return

        if not editable:
            if not self.ask_if_modif(self.comboBox_metadata.currentIndex()):
                self.checkBox_edit.blockSignals(True)
                self.checkBox_edit.setChecked(not editable)
                self.checkBox_edit.blockSignals(False)
                return

        self.stacked_metadata.setCurrentIndex(1 if editable else 0)
        self.textEdit_metadata.setReadOnly(not editable)
        self.pushButton_save.setEnabled(editable)
        self.textEdit_metadata.setStyleSheet("QTextEdit  { color: black; }")
        self.update_metadata_label()

    def show_all_metadata(self):
        """Open pop-up showing all metadata in scrollable form, with toggle to reveal hidden entries."""

        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QScrollArea, QWidget, QFormLayout,
            QLabel, QPushButton, QCheckBox
        )
        from PyQt5.QtCore import Qt

        if self.cube_info.metadata_temp is None:
            QMessageBox.information(self, "No Metadata", "No metadata available.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("All Metadata")
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_layout = QFormLayout(inner)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        check_hidden = QCheckBox("Show full values for large metadata")
        layout.addWidget(check_hidden)

        entries = {}  # key -> (full_str, QLabel)

        for key, val in self.cube_info.metadata_temp.items():
            try:
                if isinstance(val, (list, np.ndarray)):
                    val_str = ', '.join(str(v) for v in val)
                elif isinstance(val, dict):
                    val_str = str(val)
                else:
                    val_str = str(val)
            except Exception as e:
                val_str = f"<unable to display: {e}>"

            label_val = QLabel("HIDDEN" if key in self.hidden_meta else val_str)
            label_val.setWordWrap(True)
            form_layout.addRow(f"{key}:", label_val)

            entries[key] = (val_str, label_val)

        def toggle_visibility(state):
            for key in self.hidden_meta:
                if key in entries:
                    full_str, label = entries[key]
                    label.setText(full_str if state == Qt.Checked else "HIDDEN")

        check_hidden.stateChanged.connect(toggle_visibility)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.setLayout(layout)
        dialog.resize(600, 600)
        dialog.exec()

    def on_metadata_updated(self, updated_ci: CubeInfoTemp): #
        """when signal from hypercube Manager is received"""
        print('[MetadataTool] : MODIF CUBE INFOR SIGNAL RECEIVED')
        if self.cube_info and self.cube_info == updated_ci:
            self.set_cube_info(updated_ci)
            self.update_metadata_label()
            print(f"[MetadataTool] Metadata updated externally for {updated_ci.filepath}")

    def emit_updated_cube(self):
        """to send changes in cubeInfo to HypercubeManager"""
        qm=QMessageBox
        ans=qm.warning(self,"Sure ?","This action will send the modification to the other tools.\nTo reset changes you will need to upload again the cube in the app.\nValid changes ?", qm.Yes|qm.No)
        if ans==qm.Yes:
            self.metadataChanged.emit(self.cube_info)
            print(f"[MetadataTool] Metadata updated and signal emitted for {self.cube_info.filepath}")

        else:
            return

    def open_cube(self,path=None):

        if not path:
            path,_=QFileDialog.getOpenFileName(self,"Open cube for Metadata tool")
            if not path:
                return

        cube = Hypercube(filepath=path, load_init=True)
        self.set_cube_info(cube.cube_info)
        self.update_combo_meta(init=True)
        self.cubeLoaded.emit(path)  # Notify the manager

    def load_cube_info(self, ci: CubeInfoTemp):
        # Called by the manager to inject the latest version of the cube info
        self.set_cube_info(ci)

if __name__ == '__main__':
    filepath = None

    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Samples\minicubes/'
    # sample = '00189-VNIR-mock-up.h5'
    folder=r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database\Other/'
    # sample= 'reg_test.h5'
    sample=  'MPD41a_VNIR.mat'

    filepath = os.path.join(folder, sample)

    app = QApplication(sys.argv)

    meta_window = MetadataTool()

    while not filepath:
        filepath,_=QFileDialog.getOpenFileName(
            None,
            "Select Hypercube",
            "",
            "All files (*.*)"
        )
        print(filepath)

    cube = Hypercube(filepath=filepath, load_init=True)
    meta_window.set_cube_info(cube.cube_info)
    meta_window.update_combo_meta(init=True)

    meta_window.show()

    sys.exit(app.exec_())
