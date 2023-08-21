import sys


from PyQt5.QtWidgets import QApplication, QMainWindow


from xray_scatter_py.ui.ui_mainwindow import Ui_MainWindow
from xray_scatter_py.ui.ui_reflectivity import connect_reflectivity
from xray_scatter_py.ui.ui_calibration import connect_calibration
from xray_scatter_py.ui.ui_view import connect_view
from xray_scatter_py.ui.ui_file_handler import connect_file_handler
from xray_scatter_py.ui.ui_refresher import connect_refresher
from xray_scatter_py.ui.ui_browse import connect_browse
from xray_scatter_py.ui.ui_1d import connect_1d

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

        self.connect_signal_slots()

    def set_ui_params(self, **kwargs):
        self.__ui_params.update(kwargs)

    def connect_signal_slots(self):
        connect_file_handler(self)
        connect_refresher(self)
        connect_browse(self)
        connect_view(self)
        connect_calibration(self)
        connect_1d(self)
        connect_reflectivity(self)

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