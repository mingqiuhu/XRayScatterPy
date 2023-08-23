import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
import numpy as np
import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        # self.plot_set()
        fig = Figure()
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    @staticmethod
    def plot_set() -> None:
        """Update the global matplotlib settings with following parameters for
        higher figure resolution, larger tick labels, and thicker lines.

        Args:
            - None

        Returns:
            - None
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        matplotlib.rcParams['figure.dpi'] = 120
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        matplotlib.rcParams['font.size'] = 18
        matplotlib.rcParams['legend.fontsize'] = 18
        matplotlib.rcParams['axes.linewidth'] = 2
        matplotlib.rcParams['xtick.major.width'] = 3
        matplotlib.rcParams['xtick.minor.width'] = 3
        matplotlib.rcParams['ytick.major.width'] = 3
        matplotlib.rcParams['ytick.minor.width'] = 3

    def update_figure(self, h_mesh, v_mesh, c_mesh, **kwargs):
        prev_fig_size = self.figure.get_size_inches()
        self.figure = Figure(figsize=prev_fig_size)
        self.axes = self.figure.add_subplot(111)

        hticks = kwargs.get('hticks', None)
        vticks = kwargs.get('vticks', None)
        cticks = kwargs.get('cticks', None)
        hlabel = kwargs.get('hlabel', None)
        vlabel = kwargs.get('vlabel', None)
        clabel = kwargs.get('clabel', None)
        if_log = kwargs.get('if_log', False)

        c_max = min(np.max(c_mesh), kwargs.get('c_max', np.inf))
        c_min = max(np.min(c_mesh[c_mesh > 0]), kwargs.get('c_min', -np.inf))
        norm = matplotlib.colors.LogNorm(c_min, c_max) if if_log else matplotlib.colors.Normalize(c_min, c_max)
        mesh = self.axes.pcolormesh(h_mesh, v_mesh, c_mesh, cmap='jet', linewidths=3, norm=norm, shading='nearest')
        if hticks:
            self.axes.set_xticks(hticks)
        if vticks:
            self.axes.set_yticks(vticks)
        if hlabel:
            self.axes.set_xlabel(hlabel)
        if vlabel:
            self.axes.set_ylabel(vlabel)

        if kwargs.get('crop', False):
            self.axes.set_xlim(
                h_mesh[np.where(-v_mesh == np.max(-v_mesh))],
                h_mesh[np.where(-v_mesh == np.min(-v_mesh))])

            self.axes.set_ylim(
                v_mesh[np.where(-h_mesh == np.min(-h_mesh))],
                v_mesh[np.where(-h_mesh == np.max(-h_mesh))])
                
        self.axes.set_aspect('equal')
        cbar = plt.colorbar(mesh, ax=self.axes)
        if cticks:
            cbar.set_ticks(cticks)
        if clabel:
            cbar.set_label(clabel)
        self.draw()

    def update_patch(self, patch):
        pass


class CustomTreeView(QTreeView):

    def __init__(self, parent=None, dir_path=None):
        super(CustomTreeView, self).__init__(parent)
        self.setSelectionMode(QTreeView.ExtendedSelection)
        self.dir_path = dir_path
        self.latest_selected_item = None

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection):
        super(CustomTreeView, self).selectionChanged(selected, deselected)
        print(selected.indexes())
        if selected.indexes() and len(selected.indexes()) >= 2:
            source_model = self.model().sourceModel()
            source_index = self.model().mapToSource(selected.indexes()[-2])
            self.latest_selected_item = source_model.itemFromIndex(source_index)
        if deselected.indexes() and len(self.selectionModel().selectedIndexes()) == 2:
            curr_index = self.selectionModel().selectedIndexes()[-2]
            self.latest_selected_item = self.model().sourceModel().itemFromIndex(self.model().mapToSource(curr_index))

    def get_selected_paths(self):
        paths = []
        indexes = self.selectionModel().selectedIndexes()
        source_model = self.model().sourceModel()
        # We want every other index, starting from the first one
        for i in range(0, len(indexes), 2):
            index = indexes[i]
            source_index = self.model().mapToSource(index)
            file_name = source_model.itemFromIndex(source_index).text()

            if len(file_name) == 7 and file_name.isdigit():
                file_name = "latest_" + file_name + "_craw.tiff"
            
            file_name = os.path.join(self.dir_path, file_name)
            paths.append(file_name)
        return paths

    def get_latest_selected_path(self):
        if self.latest_selected_item:
            file_name = self.latest_selected_item.text()
            if len(file_name) == 7 and file_name.isdigit():
                file_name = "latest_" + file_name + "_craw.tiff"
            return os.path.join(self.dir_path, file_name)
        return None
    
    def select_previous_next(self, diff_idx=0):
        model = self.model()
        indexes = self.selectionModel().selectedIndexes()
        
        # If there's no current selection, exit
        if not indexes:
            return

        # Get the current row
        current_row = indexes[-2].row()
        
        # Get the next index in the first column
        next_row = current_row + diff_idx        
        # If the next index is valid, select it
        if model.index(next_row, 0).isValid():
            self.selectionModel().clearSelection()
            self.selectionModel().select(model.index(next_row, 0), QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            self.scrollTo(model.index(next_row, 0), QAbstractItemView.PositionAtCenter)
            return True




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.showMaximized()
        #MainWindow.resize(1211, 715)
        self.actionmat_file = QAction(MainWindow)
        self.actionmat_file.setObjectName(u"actionmat_file")
        self.actiontiff_file = QAction(MainWindow)
        self.actiontiff_file.setObjectName(u"actiontiff_file")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionSet_UI_Preferences = QAction(MainWindow)
        self.actionSet_UI_Preferences.setObjectName(u"actionSet_UI_Preferences")
        self.actionUse_Default_Preferences = QAction(MainWindow)
        self.actionUse_Default_Preferences.setObjectName(u"actionUse_Default_Preferences")
        self.actionGrid = QAction(MainWindow)
        self.actionGrid.setObjectName(u"actionGrid")
        self.actionZoom = QAction(MainWindow)
        self.actionZoom.setObjectName(u"actionZoom")
        self.actionImage_History = QAction(MainWindow)
        self.actionImage_History.setObjectName(u"actionImage_History")
        self.actionRotate = QAction(MainWindow)
        self.actionRotate.setObjectName(u"actionRotate")
        self.actionLoad = QAction(MainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        self.actionSave_current = QAction(MainWindow)
        self.actionSave_current.setObjectName(u"actionSave_current")
        self.actionManually_set = QAction(MainWindow)
        self.actionManually_set.setObjectName(u"actionManually_set")
        self.actionq_calibration = QAction(MainWindow)
        self.actionq_calibration.setObjectName(u"actionq_calibration")
        self.actionintensity_calibration = QAction(MainWindow)
        self.actionintensity_calibration.setObjectName(u"actionintensity_calibration")
        self.actionSave_current_preferences_as = QAction(MainWindow)
        self.actionSave_current_preferences_as.setObjectName(u"actionSave_current_preferences_as")
        self.actionauto_correct_center = QAction(MainWindow)
        self.actionauto_correct_center.setObjectName(u"actionauto_correct_center")
        self.actionReflectivity = QAction(MainWindow)
        self.actionReflectivity.setObjectName(u"actionReflectivity")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.vlayout_data_list = QVBoxLayout()
        self.vlayout_data_list.setObjectName(u"vlayout_data_list")
        self.pbutton_open = QPushButton(self.centralwidget)
        self.pbutton_open.setObjectName(u"pbutton_open")

        self.vlayout_data_list.addWidget(self.pbutton_open)

        self.treeview_datalist = CustomTreeView(self.centralwidget)
        self.treeview_datalist.setObjectName(u"treeview_datalist")

        self.vlayout_data_list.addWidget(self.treeview_datalist)

        self.hlayout_find = QHBoxLayout()
        self.hlayout_find.setObjectName(u"hlayout_find")
        self.label_find = QLabel(self.centralwidget)
        self.label_find.setObjectName(u"label_find")

        self.hlayout_find.addWidget(self.label_find)

        self.lineedit_find = QLineEdit(self.centralwidget)
        self.lineedit_find.setObjectName(u"lineedit_find")

        self.hlayout_find.addWidget(self.lineedit_find)


        self.vlayout_data_list.addLayout(self.hlayout_find)


        self.horizontalLayout_2.addLayout(self.vlayout_data_list)

        self.vlayout_images = QVBoxLayout()
        self.vlayout_images.setObjectName(u"vlayout_images")
        self.glayout_left_image = QGridLayout()
        self.glayout_left_image.setObjectName(u"glayout_left_image")
        self.label_left_image = QLabel(self.centralwidget)
        self.label_left_image.setObjectName(u"label_left_image")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_left_image.setFont(font)
        self.label_left_image.setAlignment(Qt.AlignCenter)

        self.glayout_left_image.addWidget(self.label_left_image, 0, 0, 1, 1)

        self.label_right_image = QLabel(self.centralwidget)
        self.label_right_image.setObjectName(u"label_right_image")
        self.label_right_image.setFont(font)
        self.label_right_image.setAlignment(Qt.AlignCenter)

        self.glayout_left_image.addWidget(self.label_right_image, 0, 1, 1, 1)

        self.vlayout_left_settings = QVBoxLayout()
        self.vlayout_left_settings.setObjectName(u"vlayout_left_settings")
        self.glayout_cbar = QGridLayout()
        self.glayout_cbar.setObjectName(u"glayout_cbar")
        self.label_cbar_min = QLabel(self.centralwidget)
        self.label_cbar_min.setObjectName(u"label_cbar_min")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_cbar_min.sizePolicy().hasHeightForWidth())
        self.label_cbar_min.setSizePolicy(sizePolicy)

        self.glayout_cbar.addWidget(self.label_cbar_min, 0, 0, 1, 1)

        self.lineedit_cbar_min = QLineEdit(self.centralwidget)
        self.lineedit_cbar_min.setObjectName(u"lineedit_cbar_min")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineedit_cbar_min.sizePolicy().hasHeightForWidth())
        self.lineedit_cbar_min.setSizePolicy(sizePolicy1)

        self.glayout_cbar.addWidget(self.lineedit_cbar_min, 0, 1, 1, 1)

        self.hslider_cbar_min = QSlider(self.centralwidget)
        self.hslider_cbar_min.setObjectName(u"hslider_cbar_min")
        self.hslider_cbar_min.setOrientation(Qt.Horizontal)

        self.glayout_cbar.addWidget(self.hslider_cbar_min, 0, 2, 1, 1)

        self.label_cbar_max = QLabel(self.centralwidget)
        self.label_cbar_max.setObjectName(u"label_cbar_max")
        sizePolicy.setHeightForWidth(self.label_cbar_max.sizePolicy().hasHeightForWidth())
        self.label_cbar_max.setSizePolicy(sizePolicy)

        self.glayout_cbar.addWidget(self.label_cbar_max, 1, 0, 1, 1)

        self.lineedit_cbar_max = QLineEdit(self.centralwidget)
        self.lineedit_cbar_max.setObjectName(u"lineedit_cbar_max")
        sizePolicy1.setHeightForWidth(self.lineedit_cbar_max.sizePolicy().hasHeightForWidth())
        self.lineedit_cbar_max.setSizePolicy(sizePolicy1)

        self.glayout_cbar.addWidget(self.lineedit_cbar_max, 1, 1, 1, 1)

        self.hslider_cbar_max = QSlider(self.centralwidget)
        self.hslider_cbar_max.setObjectName(u"hslider_cbar_max")
        self.hslider_cbar_max.setOrientation(Qt.Horizontal)

        self.glayout_cbar.addWidget(self.hslider_cbar_max, 1, 2, 1, 1)


        self.vlayout_left_settings.addLayout(self.glayout_cbar)

        self.hlayout_monitor_log = QHBoxLayout()
        self.hlayout_monitor_log.setObjectName(u"hlayout_monitor_log")
        self.cbox_monitor = QCheckBox(self.centralwidget)
        self.cbox_monitor.setObjectName(u"cbox_monitor")

        self.hlayout_monitor_log.addWidget(self.cbox_monitor)

        self.cbox_log = QCheckBox(self.centralwidget)
        self.cbox_log.setObjectName(u"cbox_log")

        self.hlayout_monitor_log.addWidget(self.cbox_log)

        self.hspacer_monitor_log = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.hlayout_monitor_log.addItem(self.hspacer_monitor_log)


        self.vlayout_left_settings.addLayout(self.hlayout_monitor_log)

        self.hlayout_browse = QHBoxLayout()
        self.hlayout_browse.setObjectName(u"hlayout_browse")
        self.pbutton_previous100 = QPushButton(self.centralwidget)
        self.pbutton_previous100.setObjectName(u"pbutton_previous100")

        self.hlayout_browse.addWidget(self.pbutton_previous100)

        self.pbutton_previous5 = QPushButton(self.centralwidget)
        self.pbutton_previous5.setObjectName(u"pbutton_previous5")

        self.hlayout_browse.addWidget(self.pbutton_previous5)

        self.pbutton_previous = QPushButton(self.centralwidget)
        self.pbutton_previous.setObjectName(u"pbutton_previous")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.pbutton_previous.sizePolicy().hasHeightForWidth())
        self.pbutton_previous.setSizePolicy(sizePolicy2)

        self.hlayout_browse.addWidget(self.pbutton_previous)

        self.hspacer_browse = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.hlayout_browse.addItem(self.hspacer_browse)

        self.pbutton_next = QPushButton(self.centralwidget)
        self.pbutton_next.setObjectName(u"pbutton_next")

        self.hlayout_browse.addWidget(self.pbutton_next)

        self.pbutton_next5 = QPushButton(self.centralwidget)
        self.pbutton_next5.setObjectName(u"pbutton_next5")

        self.hlayout_browse.addWidget(self.pbutton_next5)

        self.pbutton_next100 = QPushButton(self.centralwidget)
        self.pbutton_next100.setObjectName(u"pbutton_next100")

        self.hlayout_browse.addWidget(self.pbutton_next100)


        self.vlayout_left_settings.addLayout(self.hlayout_browse)


        self.glayout_left_image.addLayout(self.vlayout_left_settings, 1, 0, 1, 1)

        self.vlayout_right_options = QVBoxLayout()
        self.vlayout_right_options.setObjectName(u"vlayout_right_options")
        self.rbutton_orig = QRadioButton(self.centralwidget)
        self.rbutton_orig.setObjectName(u"rbutton_orig")

        self.vlayout_right_options.addWidget(self.rbutton_orig)

        self.rbutton_polar = QRadioButton(self.centralwidget)
        self.rbutton_polar.setObjectName(u"rbutton_polar")

        self.vlayout_right_options.addWidget(self.rbutton_polar)

        self.rbutton_gi_qz_qy = QRadioButton(self.centralwidget)
        self.rbutton_gi_qz_qy.setObjectName(u"rbutton_gi_qz_qy")

        self.vlayout_right_options.addWidget(self.rbutton_gi_qz_qy)

        self.rbutton_gi_perp_para = QRadioButton(self.centralwidget)
        self.rbutton_gi_perp_para.setObjectName(u"rbutton_gi_perp_para")

        self.vlayout_right_options.addWidget(self.rbutton_gi_perp_para)

        self.vspacer_right_options = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.vlayout_right_options.addItem(self.vspacer_right_options)


        self.glayout_left_image.addLayout(self.vlayout_right_options, 1, 1, 1, 1)

        self.hlayout_left_save = QHBoxLayout()
        self.hlayout_left_save.setObjectName(u"hlayout_left_save")
        self.pbutton_left_save = QPushButton(self.centralwidget)
        self.pbutton_left_save.setObjectName(u"pbutton_left_save")

        self.hlayout_left_save.addWidget(self.pbutton_left_save)

        self.hspacer_left_save = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.hlayout_left_save.addItem(self.hspacer_left_save)


        self.glayout_left_image.addLayout(self.hlayout_left_save, 2, 0, 1, 1)

        self.hlayout_right_save = QHBoxLayout()
        self.hlayout_right_save.setObjectName(u"hlayout_right_save")
        self.pbutton_right_save = QPushButton(self.centralwidget)
        self.pbutton_right_save.setObjectName(u"pbutton_right_save")

        self.hlayout_right_save.addWidget(self.pbutton_right_save)

        self.hspacer_right_save = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.hlayout_right_save.addItem(self.hspacer_right_save)


        self.glayout_left_image.addLayout(self.hlayout_right_save, 2, 1, 1, 1)

        self.graphicsview_left = MplCanvas(self.centralwidget)
        self.graphicsview_left.setObjectName(u"graphicsview_left")
        # self.graphicsview_left.setMinimumSize(QSize(400, 400))
        # self.graphicsview_left.setMaximumSize(QSize(400, 400))

        self.glayout_left_image.addWidget(self.graphicsview_left, 3, 0, 1, 1)

        self.graphicsview_right = MplCanvas(self.centralwidget)
        self.graphicsview_right.setObjectName(u"graphicsview_right")
        # self.graphicsview_right.setMinimumSize(QSize(400, 400))
        # self.graphicsview_right.setMaximumSize(QSize(400, 400))

        self.glayout_left_image.addWidget(self.graphicsview_right, 3, 1, 1, 1)

        self.glayout_left_1d = QGridLayout()
        self.glayout_left_1d.setObjectName(u"glayout_left_1d")
        self.pbutton_left_draw = QPushButton(self.centralwidget)
        self.pbutton_left_draw.setObjectName(u"pbutton_left_draw")

        self.glayout_left_1d.addWidget(self.pbutton_left_draw, 0, 0, 1, 1)

        self.lebel_left_min1 = QLabel(self.centralwidget)
        self.lebel_left_min1.setObjectName(u"lebel_left_min1")

        self.glayout_left_1d.addWidget(self.lebel_left_min1, 0, 1, 1, 1)

        self.lineedit_left_min1 = QLineEdit(self.centralwidget)
        self.lineedit_left_min1.setObjectName(u"lineedit_left_min1")

        self.glayout_left_1d.addWidget(self.lineedit_left_min1, 0, 2, 1, 1)

        self.label_left_min2 = QLabel(self.centralwidget)
        self.label_left_min2.setObjectName(u"label_left_min2")

        self.glayout_left_1d.addWidget(self.label_left_min2, 0, 3, 1, 1)

        self.lineedit_left_min2 = QLineEdit(self.centralwidget)
        self.lineedit_left_min2.setObjectName(u"lineedit_left_min2")

        self.glayout_left_1d.addWidget(self.lineedit_left_min2, 0, 4, 1, 1)

        self.pbutton_left_show = QPushButton(self.centralwidget)
        self.pbutton_left_show.setObjectName(u"pbutton_left_show")

        self.glayout_left_1d.addWidget(self.pbutton_left_show, 1, 0, 1, 1)

        self.label_left_max1 = QLabel(self.centralwidget)
        self.label_left_max1.setObjectName(u"label_left_max1")

        self.glayout_left_1d.addWidget(self.label_left_max1, 1, 1, 1, 1)

        self.lineedit_left_max1 = QLineEdit(self.centralwidget)
        self.lineedit_left_max1.setObjectName(u"lineedit_left_max1")

        self.glayout_left_1d.addWidget(self.lineedit_left_max1, 1, 2, 1, 1)

        self.label_left_max2 = QLabel(self.centralwidget)
        self.label_left_max2.setObjectName(u"label_left_max2")

        self.glayout_left_1d.addWidget(self.label_left_max2, 1, 3, 1, 1)

        self.lineedit_left_max2 = QLineEdit(self.centralwidget)
        self.lineedit_left_max2.setObjectName(u"lineedit_left_max2")

        self.glayout_left_1d.addWidget(self.lineedit_left_max2, 1, 4, 1, 1)


        self.glayout_left_image.addLayout(self.glayout_left_1d, 4, 0, 1, 1)

        self.glayout_right_1d = QGridLayout()
        self.glayout_right_1d.setObjectName(u"glayout_right_1d")
        self.pbutton_right_draw = QPushButton(self.centralwidget)
        self.pbutton_right_draw.setObjectName(u"pbutton_right_draw")

        self.glayout_right_1d.addWidget(self.pbutton_right_draw, 0, 0, 1, 1)

        self.label_right_min1 = QLabel(self.centralwidget)
        self.label_right_min1.setObjectName(u"label_right_min1")

        self.glayout_right_1d.addWidget(self.label_right_min1, 0, 1, 1, 1)

        self.lineedit_right_min1 = QLineEdit(self.centralwidget)
        self.lineedit_right_min1.setObjectName(u"lineedit_right_min1")

        self.glayout_right_1d.addWidget(self.lineedit_right_min1, 0, 2, 1, 1)

        self.label_right_min2 = QLabel(self.centralwidget)
        self.label_right_min2.setObjectName(u"label_right_min2")

        self.glayout_right_1d.addWidget(self.label_right_min2, 0, 3, 1, 1)

        self.lineedit_right_min2 = QLineEdit(self.centralwidget)
        self.lineedit_right_min2.setObjectName(u"lineedit_right_min2")

        self.glayout_right_1d.addWidget(self.lineedit_right_min2, 0, 4, 1, 1)

        self.pbutton_right_show = QPushButton(self.centralwidget)
        self.pbutton_right_show.setObjectName(u"pbutton_right_show")

        self.glayout_right_1d.addWidget(self.pbutton_right_show, 1, 0, 1, 1)

        self.label_right_max1 = QLabel(self.centralwidget)
        self.label_right_max1.setObjectName(u"label_right_max1")

        self.glayout_right_1d.addWidget(self.label_right_max1, 1, 1, 1, 1)

        self.lineedit_right_max1 = QLineEdit(self.centralwidget)
        self.lineedit_right_max1.setObjectName(u"lineedit_right_max1")

        self.glayout_right_1d.addWidget(self.lineedit_right_max1, 1, 2, 1, 1)

        self.label_right_max2 = QLabel(self.centralwidget)
        self.label_right_max2.setObjectName(u"label_right_max2")

        self.glayout_right_1d.addWidget(self.label_right_max2, 1, 3, 1, 1)

        self.lineedit_right_max2 = QLineEdit(self.centralwidget)
        self.lineedit_right_max2.setObjectName(u"lineedit_right_max2")

        self.glayout_right_1d.addWidget(self.lineedit_right_max2, 1, 4, 1, 1)


        self.glayout_left_image.addLayout(self.glayout_right_1d, 4, 1, 1, 1)


        self.vlayout_images.addLayout(self.glayout_left_image)

        self.vspacer_images = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.vlayout_images.addItem(self.vspacer_images)


        self.horizontalLayout_2.addLayout(self.vlayout_images)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1211, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuUI_Preferences = QMenu(self.menuFile)
        self.menuUI_Preferences.setObjectName(u"menuUI_Preferences")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuCalibrations = QMenu(self.menubar)
        self.menuCalibrations.setObjectName(u"menuCalibrations")
        self.menuAuto = QMenu(self.menuCalibrations)
        self.menuAuto.setObjectName(u"menuAuto")
        self.menuProcessing = QMenu(self.menubar)
        self.menuProcessing.setObjectName(u"menuProcessing")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuCalibrations.menuAction())
        self.menubar.addAction(self.menuProcessing.menuAction())
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.menuUI_Preferences.menuAction())
        self.menuUI_Preferences.addAction(self.actionSet_UI_Preferences)
        self.menuUI_Preferences.addAction(self.actionUse_Default_Preferences)
        self.menuUI_Preferences.addAction(self.actionSave_current_preferences_as)
        self.menuView.addAction(self.actionGrid)
        self.menuView.addAction(self.actionZoom)
        self.menuView.addAction(self.actionRotate)
        self.menuView.addAction(self.actionImage_History)
        self.menuCalibrations.addAction(self.actionLoad)
        self.menuCalibrations.addAction(self.actionSave_current)
        self.menuCalibrations.addAction(self.actionManually_set)
        self.menuCalibrations.addAction(self.menuAuto.menuAction())
        self.menuAuto.addAction(self.actionq_calibration)
        self.menuAuto.addAction(self.actionintensity_calibration)
        self.menuAuto.addAction(self.actionauto_correct_center)
        self.menuProcessing.addAction(self.actionReflectivity)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionmat_file.setText(QCoreApplication.translate("MainWindow", u"mat-file", None))
        self.actiontiff_file.setText(QCoreApplication.translate("MainWindow", u"tiff-file", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save as", None))
        self.actionSet_UI_Preferences.setText(QCoreApplication.translate("MainWindow", u"Set UI preferences", None))
        self.actionUse_Default_Preferences.setText(QCoreApplication.translate("MainWindow", u"Use default preferences", None))
        self.actionGrid.setText(QCoreApplication.translate("MainWindow", u"Grid", None))
        self.actionZoom.setText(QCoreApplication.translate("MainWindow", u"Zoom", None))
        self.actionImage_History.setText(QCoreApplication.translate("MainWindow", u"Image history", None))
        self.actionRotate.setText(QCoreApplication.translate("MainWindow", u"Rotate", None))
        self.actionLoad.setText(QCoreApplication.translate("MainWindow", u"Load last", None))
        self.actionSave_current.setText(QCoreApplication.translate("MainWindow", u"Save current", None))
        self.actionManually_set.setText(QCoreApplication.translate("MainWindow", u"Manually set", None))
        self.actionq_calibration.setText(QCoreApplication.translate("MainWindow", u"q-calibration", None))
        self.actionintensity_calibration.setText(QCoreApplication.translate("MainWindow", u"intensity-calibration", None))
        self.actionSave_current_preferences_as.setText(QCoreApplication.translate("MainWindow", u"Save current preferences as", None))
        self.actionauto_correct_center.setText(QCoreApplication.translate("MainWindow", u"auto-correct center", None))
        self.actionReflectivity.setText(QCoreApplication.translate("MainWindow", u"Reflectivity", None))
        self.pbutton_open.setText(QCoreApplication.translate("MainWindow", u"Open dir", None))
        self.label_find.setText(QCoreApplication.translate("MainWindow", u"Find:", None))
        self.label_left_image.setText(QCoreApplication.translate("MainWindow", u"2D image in qz - qy coordinate", None))
        self.label_right_image.setText(QCoreApplication.translate("MainWindow", u"Other coordinate options for 2D image", None))
        self.label_cbar_min.setText(QCoreApplication.translate("MainWindow", u"Colorbar min:", None))
        self.label_cbar_max.setText(QCoreApplication.translate("MainWindow", u"Colorbar max:", None))
        self.cbox_monitor.setText(QCoreApplication.translate("MainWindow", u"Monitoring?", None))
        self.cbox_log.setText(QCoreApplication.translate("MainWindow", u"Log colorscale?", None))
        self.pbutton_previous100.setText(QCoreApplication.translate("MainWindow", u"<<< 100", None))
        self.pbutton_previous5.setText(QCoreApplication.translate("MainWindow", u"<< 5", None))
        self.pbutton_previous.setText(QCoreApplication.translate("MainWindow", u"< 1", None))
        self.pbutton_next.setText(QCoreApplication.translate("MainWindow", u"1 >", None))
        self.pbutton_next5.setText(QCoreApplication.translate("MainWindow", u"5 >>", None))
        self.pbutton_next100.setText(QCoreApplication.translate("MainWindow", u"100 >>>", None))
        self.rbutton_orig.setText(QCoreApplication.translate("MainWindow", u"Original image: z - y", None))
        self.rbutton_polar.setText(QCoreApplication.translate("MainWindow", u"Polar transformation: azimuth - q", None))
        self.rbutton_gi_qz_qy.setText(QCoreApplication.translate("MainWindow", u"GI coordinate: qz - qy", None))
        self.rbutton_gi_perp_para.setText(QCoreApplication.translate("MainWindow", u"GI coordinate: q\u22a5 - q\u2225", None))
        self.pbutton_left_save.setText(QCoreApplication.translate("MainWindow", u"Save Image", None))
        self.pbutton_right_save.setText(QCoreApplication.translate("MainWindow", u"Save Image", None))
        self.pbutton_left_draw.setText(QCoreApplication.translate("MainWindow", u"Draw 1D Average Area", None))
        self.lebel_left_min1.setText(QCoreApplication.translate("MainWindow", u"ymin", None))
        self.label_left_min2.setText(QCoreApplication.translate("MainWindow", u"zmin", None))
        self.pbutton_left_show.setText(QCoreApplication.translate("MainWindow", u"Show 1D Average", None))
        self.label_left_max1.setText(QCoreApplication.translate("MainWindow", u"ymax", None))
        self.label_left_max2.setText(QCoreApplication.translate("MainWindow", u"zmax", None))
        self.pbutton_right_draw.setText(QCoreApplication.translate("MainWindow", u"Draw 1D Average Area", None))
        self.label_right_min1.setText(QCoreApplication.translate("MainWindow", u"ymin", None))
        self.label_right_min2.setText(QCoreApplication.translate("MainWindow", u"zmin", None))
        self.pbutton_right_show.setText(QCoreApplication.translate("MainWindow", u"Show 1D Average", None))
        self.label_right_max1.setText(QCoreApplication.translate("MainWindow", u"ymax", None))
        self.label_right_max2.setText(QCoreApplication.translate("MainWindow", u"zmax", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuUI_Preferences.setTitle(QCoreApplication.translate("MainWindow", u"UI preferences", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", u"View", None))
        self.menuCalibrations.setTitle(QCoreApplication.translate("MainWindow", u"Calibrations", None))
        self.menuAuto.setTitle(QCoreApplication.translate("MainWindow", u"Auto", None))
        self.menuProcessing.setTitle(QCoreApplication.translate("MainWindow", u"Processing", None))
    # retranslateUi