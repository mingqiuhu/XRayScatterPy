from xray_scatter_py.ui_mainwindow import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup

class ui_mainwindow_functionalized(Ui_MainWindow):
    def __init__(self, MainWindow):
        Ui_MainWindow.__init__(self)
        self.setupUi(MainWindow)

        self.__ui_params = {}

        self.params_dict_list = None
        self.image_array = None
        self.detx0 = None
        self.params_dict_list = None
        self.image_array = None 
        self.qx_array, self.qy_array, self.qz_array = None, None, None
        self.sr_array = None
        self.image_array_rel = None
        self.image_array_abs = None

        self.connect_buttons()
        self.connect_boxes()
        self.connect_actions()

    def set_ui_params(self, **kwargs):
        self.__ui_params.update(kwargs)

    # connect_buttons
    def connect_buttons(self):
        self.pbutton_open.clicked.connect(lambda *_: self.open_directory())
        self.pbutton_previous.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-1))
        self.pbutton_previous.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_previous10.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-10))
        self.pbutton_previous10.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_previous100.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-100))
        self.pbutton_previous100.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_next.clicked.connect(lambda *_: self.previous_next_data(diff_idx=1))
        self.pbutton_next.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_next10.clicked.connect(lambda *_: self.previous_next_data(diff_idx=10))
        self.pbutton_next10.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_next100.clicked.connect(lambda *_: self.previous_next_data(diff_idx=100))
        self.pbutton_next100.clicked.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.pbutton_left_save.clicked.connect(lambda *_: self.save_image(pos='left'))
        self.pbutton_right_save.clicked.connect(lambda *_: self.save_image(pos='right'))
        self.pbutton_left_draw.clicked.connect(lambda *_: self.draw_1d_area(pos='left'))
        self.pbutton_right_draw.clicked.connect(lambda *_: self.draw_1d_area(pos='right'))
        self.pbutton_left_show.clicked.connect(lambda *_: self.show_1d(pos='left'))
        self.pbutton_right_show.clicked.connect(lambda *_: self.show_1d(pos='right'))

    def open_directory(self):
        print('open_directory')

    def previous_next_data(self, diff_idx=0):
        print(diff_idx)

    def save_image(self, pos=None):
        print(pos)

    def draw_1d_area(self, pos=None):
        print(pos)

    def show_1d(self, pos=None):
        print(pos)

    # connect_boxes
    def connect_boxes(self):
        self.lineedit_find.editingFinished.connect(lambda *_: self.filter_data_list())
        self.lineedit_find.editingFinished.connect(lambda *_: self.refresh_ui('treeview_datalist'))
        self.lineedit_cbar_min.editingFinished.connect(lambda *_: self.refresh_ui('hslider_cbar_min', 'graphicsview_left', 'graphicsview_right'))
        self.lineedit_cbar_max.editingFinished.connect(lambda *_: self.refresh_ui('hslider_cbar_max', 'graphicsview_left', 'graphicsview_right'))
        self.hslider_cbar_min.valueChanged.connect(lambda *_: self.refresh_ui('lineedit_cbar_min', 'graphicsview_left', 'graphicsview_right'))
        self.hslider_cbar_max.valueChanged.connect(lambda *_: self.refresh_ui('lineedit_cbar_max', 'graphicsview_left', 'graphicsview_right'))
        # self.cbox_monitor multithreading
        self.cbox_log.stateChanged.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right', 'lineedit_cbar_min', 'lineedit_cbar_max', 'hslider_cbar_min', 'hslider_cbar_max'))

        self.rbutton_group = QButtonGroup(self.centralwidget)
        self.rbutton_group.addButton(self.rbutton_orig)
        self.rbutton_group.addButton(self.rbutton_polar)
        self.rbutton_group.addButton(self.rbutton_gi_perp_para)
        self.rbutton_group.addButton(self.rbutton_gi_qz_qy)
        self.rbutton_group.buttonClicked.connect(lambda *_: self.refresh_ui(
            'graphicsview_right',
            'label_right_min1',
            'label_right_max1',
            'label_right_min2',
            'label_right_max2',
            'lineedit_right_min1',
            'lineedit_right_max1',
            'lineedit_right_min2',
            'lineedit_right_max2',
        ))

        self.lineedit_left_min1.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_left'))
        self.lineedit_left_max1.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_left'))
        self.lineedit_left_min2.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_left'))
        self.lineedit_left_max2.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_left'))
        self.lineedit_right_min1.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_right'))
        self.lineedit_right_max1.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_right'))
        self.lineedit_right_min2.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_right'))
        self.lineedit_right_max2.editingFinished.connect(lambda *_: self.refresh_ui('graphicsview_right'))

    def filter_data_list(self):
        print('filter_data_list')

    # connect_actions
    def connect_actions(self):
        # File
        self.actionSave_As.triggered.connect(lambda *_: self.save_as())
        self.actionSet_UI_Preferences.triggered.connect(lambda *_: self.set_ui_preferences()) # may need to refresh entire ui
        self.actionUse_Default_Preferences.triggered.connect(lambda *_: self.use_default_preferences()) # may need to refresh entire ui
        self.actionSave_current_preferences_as.triggered.connect(lambda *_: self.save_current_preferences_as())
        # View
        self.actionGrid.isCheckable = True
        self.actionGrid.triggered.connect(lambda *_: self.set_ui_params(grid=self.actionGrid.isChecked()))
        self.actionGrid.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionZoom.triggered.connect(lambda *_: self.zoom())
        self.actionZoom.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionRotate.triggered.connect(lambda *_: self.rotate())
        self.actionRotate.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionImage_History.triggered.connect(lambda *_: self.image_history())
        # Calibrations
        self.actionLoad.triggered.connect(lambda *_: self.load())
        self.actionLoad.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionSave_current.triggered.connect(lambda *_: self.save_current())
        self.actionManually_set.triggered.connect(lambda *_: self.manually_set())
        self.actionManually_set.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionq_calibration.triggered.connect(lambda *_: self.q_calibration())
        self.actionq_calibration.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionintensity_calibration.triggered.connect(lambda *_: self.intensity_calibration())
        self.actionintensity_calibration.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.actionauto_correct_center.triggered.connect(lambda *_: self.auto_correct_center())
        self.actionauto_correct_center.triggered.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))
        # Processing
        self.actionReflectivity.triggered.connect(lambda *_: self.reflectivity())

    def save_as(self):
        print('save_as')

    def set_ui_preferences(self):
        print('set_ui_preferences')

    def use_default_preferences(self):
        print('use_default_preferences')

    def save_current_preferences_as(self):
        print('save_current_preferences_as')

    def zoom(self):
        print('zoom')
        self.set_ui_params(zoom=True, zoom_min=0.5, zoom_max=2.0) # the numbers should come from user selection

    def rotate(self):
        print('rotate')
        self.set_ui_params(rotate=True, rotate_angle=90) # the numbers should come from user selection

    def image_history(self):
        print('image_history')

    def load(self):
        print('load')

    def save_current(self):
        print('save_current')

    def manually_set(self):
        print('manually_set')

    def q_calibration(self):
        print('q_calibration')

    def intensity_calibration(self):
        print('intensity_calibration')

    def auto_correct_center(self):
        print('auto_correct_center')

    def reflectivity(self):
        # this function may need to be imported from another module
        print('reflectivity')

    # refresh_ui
    def refresh_ui(self, *args):

        def refresh_treeview_datalist():
            print('refresh_treeview_datalist')
        def refresh_graphicsview_left():
            print('refresh_graphicsview_left')
        def refresh_graphicsview_right():
            print('refresh_graphicsview_right')
        def refresh_lineedit_cbar_min():
            print('refresh_lineedit_cbar_min')
        def refresh_lineedit_cbar_max():
            print('refresh_lineedit_cbar_max')
        def refresh_hslider_cbar_min():
            print('refresh_hslider_cbar_min')
        def refresh_hslider_cbar_max():
            print('refresh_hslider_cbar_max')
        def refresh_label_right_min1():
            print('refresh_label_right_min1')
        def refresh_label_right_max1():
            print('refresh_label_right_max1')
        def refresh_label_right_min2():
            print('refresh_label_right_min2')
        def refresh_label_right_max2():
            print('refresh_label_right_max2')
        def refresh_lineedit_right_min1():
            print('refresh_lineedit_right_min1')
        def refresh_lineedit_right_max1():
            print('refresh_lineedit_right_max1')
        def refresh_lineedit_right_min2():
            print('refresh_lineedit_right_min2')
        def refresh_lineedit_right_max2():
            print('refresh_lineedit_right_max2')

        refresh_dict = {
            'treeview_datalist': refresh_treeview_datalist,
            'graphicsview_left': refresh_graphicsview_left,
            'graphicsview_right': refresh_graphicsview_right,
            'lineedit_cbar_min': refresh_lineedit_cbar_min,
            'lineedit_cbar_max': refresh_lineedit_cbar_max,
            'hslider_cbar_min': refresh_hslider_cbar_min,
            'hslider_cbar_max': refresh_hslider_cbar_max,
            'label_right_min1': refresh_label_right_min1,
            'label_right_max1': refresh_label_right_max1,
            'label_right_min2': refresh_label_right_min2,
            'label_right_max2': refresh_label_right_max2,
            'lineedit_right_min1': refresh_lineedit_right_min1,
            'lineedit_right_max1': refresh_lineedit_right_max1,
            'lineedit_right_min2': refresh_lineedit_right_min2,
            'lineedit_right_max2': refresh_lineedit_right_max2,
        }

        [refresh_dict[arg]() for arg in args]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = ui_mainwindow_functionalized(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())