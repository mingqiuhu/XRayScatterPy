import sys
import os
import datetime
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItemIterator, QTreeWidgetItem
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

from xray_scatter_py.ui.ui_mainwindow import Ui_MainWindow
from xray_scatter_py.ui.ui_reflectivity import connect_reflectivity
from xray_scatter_py.ui.ui_calibration import connect_calibration
from xray_scatter_py.ui.ui_view import connect_view
from xray_scatter_py.ui.ui_file_handler import connect_file_handler
from xray_scatter_py.ui.ui_refresher import connect_refresher
from xray_scatter_py.ui.ui_browse import connect_browse
from xray_scatter_py.ui.ui_1d import connect_1d

from xray_scatter_py import data_plotting, utils, calibration

HLABEL_DICT = {
    'q': r'$q\ \mathrm{(Å^{-1})}$',
    'qx': r'$q_\mathrm{x}\ \mathrm{(Å^{-1})}$',
    'qy': r'$q_\mathrm{y}\ \mathrm{(Å^{-1})}$',
    'qz': r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$',
    'qz0': r'$q_\mathrm{z,0}\ \mathrm{(Å^{-1})}$',
    'q_parallel': r'$q_\Vert\ \mathrm{(Å^{-1})}$',
    'theta sample': r'${\theta}_\mathrm{sample}\ (°)$',
    'kzi': r'$k_\mathrm{z,i}\ \mathrm{(Å^{-1})}$',
    'alpha i': r'$\alpha_\mathrm{i}\ (°)$',
}
VLABEL_DICT = {
    'qz': r'$q_\mathrm{z}\ \mathrm{(Å^{-1})}$',
    'azimuth': r'$\mathrm{azimuth}\ (°)$',
    'abs': r'$I\ \mathrm{(cm^{-1}sr^{-1})}$',
    'a.u.': r'$I\ \mathrm{(a.u.)}$',
    'reflectivity': r'$R$',
    'total': r'$I_\mathrm{total}\ \mathrm{(a.u.)}$',
    'spillover': r'$I_\mathrm{spill\ over}\ \mathrm{(a.u.)}$',
    'depth': r'$z_\mathrm{1/e}\ \mathrm{(Å)}$',
    'kzf': r'$k_\mathrm{z,f}\ \mathrm{(Å^{-1})}$',
    'q_vertical': r'$q_\mathrm{⊥}\ \mathrm{(Å^{-1})}$'
}


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
        print('kwargs', kwargs)
        self.__ui_params.update(kwargs)
        print('ui params', self.__ui_params)

    def connect_signal_slots(self):
        connect_file_handler(self)
        connect_refresher(self)
        connect_browse(self)
        connect_view(self)
        connect_calibration(self)
        connect_1d(self)
        connect_reflectivity(self)

    def refresh_ui(self, *args):

        def refresh_treeview_datalist(reload=False):
            if reload:
                model = QStandardItemModel()
                model.setHorizontalHeaderLabels(["No", "Date Modified"])
                self.treeview_datalist.setModel(model)
                self.treeview_datalist.dir_path = self.__ui_params['folder_path']

                tiff_files = [f for f in os.listdir(self.__ui_params['folder_path']) if f.endswith('.tiff')]
                tiff_files = sorted(tiff_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
                grad_files = [f for f in os.listdir(self.__ui_params['folder_path']) if f.endswith('.grad')]
                grad_files = sorted(grad_files)

                for f in tiff_files:
                    full_path = os.path.join(self.__ui_params['folder_path'], f)
                    date_modified = os.path.getmtime(full_path)
                    number = ''.join(filter(str.isdigit, f))
                    number_item = QStandardItem(number)
                    number_item.setFlags(number_item.flags() & ~Qt.ItemIsEditable)

                    date_modified_item = QStandardItem(datetime.datetime.fromtimestamp(date_modified).strftime('%Y-%m-%d %H:%M:%S'))
                    date_modified_item.setFlags(date_modified_item.flags() & ~Qt.ItemIsEditable)

                    model.appendRow([number_item, date_modified_item])

                for f in grad_files:
                    full_path = os.path.join(self.__ui_params['folder_path'], f)
                    date_modified = os.path.getmtime(full_path)

                    name_item = QStandardItem(f)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)

                    date_modified_item = QStandardItem(datetime.datetime.fromtimestamp(date_modified).strftime('%Y-%m-%d %H:%M:%S'))
                    date_modified_item.setFlags(date_modified_item.flags() & ~Qt.ItemIsEditable)

                    model.appendRow([name_item, date_modified_item])

                self.treeview_datalist.resizeColumnToContents(0)
                self.treeview_datalist.resizeColumnToContents(1)

                print('relaoded_treeview_datalist')

            regex_input = self.lineedit_find.text()
            full_regex = ".*" + regex_input + ".*"

            reg_exp = QtCore.QRegExp(full_regex, QtCore.Qt.CaseInsensitive)

            # Getting the proxy model for filtering
            proxy_model = QtCore.QSortFilterProxyModel()
            try:
                proxy_model.setSourceModel(self.treeview_datalist.model().sourceModel())
            except:
                proxy_model.setSourceModel(self.treeview_datalist.model())
            proxy_model.setFilterRegExp(reg_exp)

            # Applying the filter on the first column
            proxy_model.setFilterKeyColumn(0)

            # Setting the filtered model to the treeview
            self.treeview_datalist.setModel(proxy_model)
            self.treeview_datalist.update()

            self.treeview_datalist.selectionModel().selectionChanged.connect(lambda *_: self.refresh_ui('graphicsview_left', 'graphicsview_right'))


            print('filtered_data_list based on regex:', regex_input)           

        def refresh_graphicsview_left():
            selected_path = self.treeview_datalist.get_latest_selected_path()
            if selected_path and selected_path.lower().endswith('.tiff'):
                selected_dir = os.path.dirname(selected_path)
                selected_no = int(''.join(filter(str.isdigit, os.path.basename(selected_path))))

                DETX0 = 100.4

                params_dict_list, image_array = utils.read_multiimage(
                    selected_dir, selected_no)
                theta_array, azimuth_array = calibration.get_angle(
                    DETX0, params_dict_list, image_array)
                qx_array, qy_array, qz_array = calibration.get_q(
                    DETX0, params_dict_list, image_array)
                sr_array = calibration.get_sr(DETX0, params_dict_list, theta_array)
                image_array_rel = calibration.get_rel_intensity(
                    params_dict_list, image_array, sr_array)
                try:
                    image_array_abs = calibration.get_abs_intensity(
                        params_dict_list, image_array_rel)
                    self.graphicsview_left.update_figure(
                        qy_array[0], qz_array[0], image_array_abs[0],
                        hlabel=HLABEL_DICT['qy'], vlabel=VLABEL_DICT['qz'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                        if_log=self.cbox_log.isChecked()
                    )
                except:
                    self.graphicsview_left.update_figure(
                        qy_array[0], qz_array[0], image_array_rel[0],
                        hlabel='qy (A-1)', vlabel='qz (A-1)', clabel='Intensity (a.u.)',
                        if_log=self.cbox_log.isChecked()
                    )
            print("latest selected path", self.treeview_datalist.get_latest_selected_path())
            print("all selected paths", self.treeview_datalist.get_selected_paths())
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
            'treeview_datalist_reload': lambda: refresh_treeview_datalist(reload=True),
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