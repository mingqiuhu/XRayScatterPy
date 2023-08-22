import os

from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore

class connect_file_handler(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()

    def connect(self):
        self.ui.pbutton_open.clicked.connect(lambda *_: self.open_directory())
        self.ui.actionSave_As.triggered.connect(lambda *_: self.save_as())
        self.ui.actionSet_UI_Preferences.triggered.connect(lambda *_: self.set_ui_preferences()) # may need to refresh entire ui
        self.ui.actionUse_Default_Preferences.triggered.connect(lambda *_: self.use_default_preferences()) # may need to refresh entire ui
        self.ui.actionSave_current_preferences_as.triggered.connect(lambda *_: self.save_current_preferences_as())
        self.ui.treeview_datalist.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.lineedit_find.editingFinished.connect(lambda *_: self.ui.refresh_ui('treeview_datalist'))
        self.ui.pbutton_left_save.clicked.connect(lambda *_: self.save_image(pos='left'))
        self.ui.pbutton_right_save.clicked.connect(lambda *_: self.save_image(pos='right'))

    def open_directory(self):
        default_dir = "/Users/mingqiuhu/Coding/ui development/XRayScatterPy/data/tiff_files_calibration_q"
        if not os.path.exists(default_dir):
            default_dir = None
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", default_dir)
        if folder_path:
            self.ui.set_ui_params(folder_path=folder_path)
            self.ui.refresh_ui('treeview_datalist_reload', 'graphicsview_left', 'graphicsview_right')

    def save_as(self):
        print('save_as')

    def set_ui_preferences(self):
        print('set_ui_preferences')

    def use_default_preferences(self):
        print('use_default_preferences')

    def save_current_preferences_as(self):
        print('save_current_preferences_as')

    def save_image(self, pos=None):
        print(pos)
