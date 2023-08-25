import sys
import os
import datetime
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItemIterator, QTreeWidgetItem
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QCoreApplication
from xray_scatter_py.ui.ui_mainwindow import Ui_MainWindow, CustomTreeView
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
    'y': r'$y\ \mathrm{(mm)}$',
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
    'q_vertical': r'$q_\mathrm{⊥}\ \mathrm{(Å^{-1})}$',
    'z': r'$z\ \mathrm{(mm)}$',
}


class ui_mainwindow_functionalized(Ui_MainWindow):
    def __init__(self, MainWindow):
        Ui_MainWindow.__init__(self)
        self.setupUi(MainWindow)

        self.__ui_params = {}
        self.loaded_data = {}
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

            self.treeview_datalist.selectionModel().selectionChanged.connect(lambda *_: self.refresh_ui('graphicsview_left_reload', 'graphicsview_right'))
            print('filtered_data_list based on regex:', regex_input)   

        def refresh_graphicsview_left(reload=False):
            if reload:
                print('reloading graphicsview_left')
                self.loaded_data = {}
                selected_path = self.treeview_datalist.get_latest_selected_path()
                if selected_path and selected_path.lower().endswith('.tiff'):
                    selected_dir = os.path.dirname(selected_path)
                    selected_no = int(''.join(filter(str.isdigit, os.path.basename(selected_path))))

                    DETX0 = 100.4

                    self.loaded_data['params_dict_list'], self.loaded_data['image_array'] = utils.read_multiimage(
                        selected_dir, selected_no)
                    self.loaded_data['theta_array'], self.loaded_data['azimuth_array'] = calibration.get_angle(
                        DETX0, self.loaded_data['params_dict_list'], self.loaded_data['image_array'])
                    self.loaded_data['qx_array'], self.loaded_data['qy_array'], self.loaded_data['qz_array'] = calibration.get_q(
                        DETX0, self.loaded_data['params_dict_list'], self.loaded_data['image_array'])
                    self.loaded_data['sr_array'] = calibration.get_sr(DETX0, self.loaded_data['params_dict_list'], self.loaded_data['theta_array'])
                    self.loaded_data['image_array_rel'] = calibration.get_rel_intensity(
                        self.loaded_data['params_dict_list'], self.loaded_data['image_array'], self.loaded_data['sr_array'])
            try:
                self.loaded_data['image_array_abs'] = calibration.get_abs_intensity(
                    self.loaded_data['params_dict_list'], self.loaded_data['image_array'])
                self.graphicsview_left.update_figure(
                    self.loaded_data['qy_array'][0], self.loaded_data['qz_array'][0], self.loaded_data['image_array_abs'][0],
                    hlabel=HLABEL_DICT['qy'], vlabel=VLABEL_DICT['qz'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                    if_log=self.cbox_log.isChecked()
                )
            except:
                if 'image_array_rel' in self.loaded_data.keys():
                    self.graphicsview_left.update_figure(
                        self.loaded_data['qy_array'][0], self.loaded_data['qz_array'][0], self.loaded_data['image_array_rel'][0],
                        hlabel=HLABEL_DICT['qy'], vlabel=VLABEL_DICT['qz'], clabel=r'$Intensity\ \mathrm{(a.u.)}$',
                        if_log=self.cbox_log.isChecked()
                    )

            print('refresh_graphicsview_left')

        def refresh_graphicsview_right():
            
            DETX0 = 100.4
            def plt_orig():
                self.label_right_min1.setText(QCoreApplication.translate("MainWindow", u"r min", None))
                self.label_right_max1.setText(QCoreApplication.translate("MainWindow", u"r max", None))
                self.label_right_min2.setText(QCoreApplication.translate("MainWindow", u"azimuth start", None))
                self.label_right_max2.setText(QCoreApplication.translate("MainWindow", u"azimuth end", None))
                self.graphicsview_right.type = 'orig'
                if 'x_array' not in self.loaded_data.keys():
                    try:
                        self.loaded_data['x_array'], self.loaded_data['y_array'], self.loaded_data['z_array'] = calibration.get_mm(
                            DETX0,
                            self.loaded_data['params_dict_list'],
                            self.loaded_data['image_array']
                        )
                    except:
                        return
                
                if 'image_array_abs' in self.loaded_data.keys():
                    self.graphicsview_right.update_figure(
                        self.loaded_data['y_array'][0], self.loaded_data['z_array'][0], self.loaded_data['image_array_abs'][0],
                        hlabel=HLABEL_DICT['y'], vlabel=VLABEL_DICT['z'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                        if_log=self.cbox_log.isChecked()
                    )
                else:
                    self.graphicsview_right.update_figure(
                        self.loaded_data['y_array'][0], self.loaded_data['z_array'][0], self.loaded_data['image_array_rel'][0],
                        hlabel=HLABEL_DICT['y'], vlabel=VLABEL_DICT['z'], clabel=r'$Intensity\ \mathrm{(a.u.)}$',
                        if_log=self.cbox_log.isChecked()
                    )

            def plt_polar():
                self.label_right_min1.setText(QCoreApplication.translate("MainWindow", u"q min", None))
                self.label_right_max1.setText(QCoreApplication.translate("MainWindow", u"q max", None))
                self.label_right_min2.setText(QCoreApplication.translate("MainWindow", u"azimuth min", None))
                self.label_right_max2.setText(QCoreApplication.translate("MainWindow", u"azimuth max", None))

                self.graphicsview_right.type = 'polar'
                if 'image_array_abs' in self.loaded_data.keys():
                    self.graphicsview_right.update_figure_polar(
                        self.loaded_data['azimuth_array'][0],
                        self.loaded_data['qx_array'][0],
                        self.loaded_data['qy_array'][0],
                        self.loaded_data['qz_array'][0],
                        self.loaded_data['image_array_abs'][0],
                        self.loaded_data['params_dict_list'][0],
                        hlabel=HLABEL_DICT['q'], vlabel=VLABEL_DICT['azimuth'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                        if_log=self.cbox_log.isChecked()
                    )
                elif 'image_array_rel' in self.loaded_data.keys():
                    self.graphicsview_right.update_figure_polar(
                        self.loaded_data['azimuth_array'][0],
                        self.loaded_data['qx_array'][0],
                        self.loaded_data['qy_array'][0],
                        self.loaded_data['qz_array'][0],
                        self.loaded_data['image_array_rel'][0],
                        self.loaded_data['params_dict_list'][0],
                        hlabel=HLABEL_DICT['q'], vlabel=VLABEL_DICT['azimuth'], clabel=r'$Intensity\ \mathrm{(a.u.)}$',
                        if_log=self.cbox_log.isChecked()
                    )
                print('plt_polar')
            def plt_gi_qz_qy():
                self.label_right_min1.setText(QCoreApplication.translate("MainWindow", u"qy min", None))
                self.label_right_max1.setText(QCoreApplication.translate("MainWindow", u"qy max", None))
                self.label_right_min2.setText(QCoreApplication.translate("MainWindow", u"qz min", None))
                self.label_right_max2.setText(QCoreApplication.translate("MainWindow", u"qz max", None))

                self.graphicsview_right.type = 'gi_qz_qy'
                if 'qx_array_gi' not in self.loaded_data.keys():
                    try:
                        self.loaded_data['qx_array_gi'], self.loaded_data['qy_array_gi'], self.loaded_data['qz_array_gi'] = calibration.get_q_gi(
                            self.loaded_data['qx_array'], self.loaded_data['qy_array'], self.loaded_data['qz_array'], self.loaded_data['params_dict_list']
                        )
                    except:
                        return
                if 'image_array_abs' in self.loaded_data.keys():
                    self.graphicsview_right.update_figure(
                        self.loaded_data['qy_array_gi'][0], self.loaded_data['qz_array_gi'][0], self.loaded_data['image_array_abs'][0],
                        hlabel=HLABEL_DICT['qy'], vlabel=VLABEL_DICT['qz'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                        if_log=self.cbox_log.isChecked()
                    )
                else:
                    self.graphicsview_right.update_figure(
                        self.loaded_data['qy_array_gi'][0], self.loaded_data['qz_array_gi'][0], self.loaded_data['image_array_rel'][0],
                        hlabel=HLABEL_DICT['qy'], vlabel=VLABEL_DICT['qz'], clabel=r'$Intensity\ \mathrm{(a.u.)}$',
                        if_log=self.cbox_log.isChecked()
                    )
                print('plt_gi_qz_qy')
            def plt_gi_perp_para():
                self.label_right_min1.setText(QCoreApplication.translate("MainWindow", u"q parallel min", None))
                self.label_right_max1.setText(QCoreApplication.translate("MainWindow", u"q parallel max", None))
                self.label_right_min2.setText(QCoreApplication.translate("MainWindow", u"q perpendicular min", None))
                self.label_right_max2.setText(QCoreApplication.translate("MainWindow", u"q perpendicular max", None))

                self.graphicsview_right.type = 'gi_perp_para'
                if 'qx_array_gi' not in self.loaded_data.keys():
                    try:
                        self.loaded_data['qx_array_gi'], self.loaded_data['qy_array_gi'], self.loaded_data['qz_array_gi'] = calibration.get_q_gi(
                            self.loaded_data['qx_array'], self.loaded_data['qy_array'], self.loaded_data['qz_array'], self.loaded_data['params_dict_list']
                        )
                    except:
                        return
                if 'image_array_abs' in self.loaded_data.keys():
                    self.graphicsview_right.update_figure_gi(
                        self.loaded_data['qx_array_gi'][0],
                        self.loaded_data['qy_array_gi'][0],
                        self.loaded_data['qz_array_gi'][0],
                        self.loaded_data['image_array_abs'][0],
                        hlabel=HLABEL_DICT['q_parallel'], vlabel=VLABEL_DICT['q_vertical'], clabel=r'$Intensity\ \mathrm{(cm^{-1}sr^{-1})}$',
                        if_log=self.cbox_log.isChecked()
                    )
                else:
                    self.graphicsview_right.update_figure_gi(
                        self.loaded_data['qx_array_gi'][0],
                        self.loaded_data['qy_array_gi'][0],
                        self.loaded_data['qz_array_gi'][0],
                        self.loaded_data['image_array_rel'][0],
                        hlabel=HLABEL_DICT['q_parallel'], vlabel=VLABEL_DICT['q_vertical'], clabel=r'$Intensity\ \mathrm{(a.u.)}$',
                        if_log=self.cbox_log.isChecked()
                    )
                print('plt_gi_perp_para')

            refresh_dict = {
                'rbutton_orig': plt_orig,
                'rbutton_polar': plt_polar,
                'rbutton_gi_qz_qy': plt_gi_qz_qy,
                'rbutton_gi_perp_para': plt_gi_perp_para

            }
            if self.rbutton_group.checkedButton():
                refresh_dict[self.rbutton_group.checkedButton().objectName()]()
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
            'graphicsview_left_reload': lambda: refresh_graphicsview_left(reload=True),
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

        print('\nrefreshing ui: -----------------------------------')
        [refresh_dict[arg]() for arg in args]
        print("latest selected path", self.treeview_datalist.get_latest_selected_path())
        print("all selected paths", self.treeview_datalist.get_selected_paths())   
        print('---------------------------------------------------\n')     
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = ui_mainwindow_functionalized(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())