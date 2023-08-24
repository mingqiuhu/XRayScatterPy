from PyQt5.QtWidgets import QButtonGroup

class connect_refresher(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()
    
    def connect(self):
        self.ui.lineedit_cbar_min.editingFinished.connect(lambda *_: self.ui.refresh_ui('hslider_cbar_min', 'graphicsview_left', 'graphicsview_right'))
        self.ui.lineedit_cbar_max.editingFinished.connect(lambda *_: self.ui.refresh_ui('hslider_cbar_max', 'graphicsview_left', 'graphicsview_right'))
        self.ui.hslider_cbar_min.valueChanged.connect(lambda *_: self.ui.refresh_ui('lineedit_cbar_min', 'graphicsview_left', 'graphicsview_right'))
        self.ui.hslider_cbar_max.valueChanged.connect(lambda *_: self.ui.refresh_ui('lineedit_cbar_max', 'graphicsview_left', 'graphicsview_right'))
        # self.cbox_monitor multithreading
        self.ui.cbox_log.stateChanged.connect(lambda *_: self.ui.refresh_ui('lineedit_cbar_min', 'lineedit_cbar_max', 'hslider_cbar_min', 'hslider_cbar_max', 'graphicsview_left', 'graphicsview_right'))
        
        self.ui.rbutton_group = QButtonGroup(self.ui.centralwidget)
        self.ui.rbutton_group.addButton(self.ui.rbutton_orig)
        self.ui.rbutton_group.addButton(self.ui.rbutton_polar)
        self.ui.rbutton_group.addButton(self.ui.rbutton_gi_perp_para)
        self.ui.rbutton_group.addButton(self.ui.rbutton_gi_qz_qy)
        self.ui.rbutton_group.buttonClicked.connect(lambda *_: self.ui.refresh_ui(
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
        self.ui.lineedit_left_min1.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_left'))
        self.ui.lineedit_left_max1.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_left'))
        self.ui.lineedit_left_min2.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_left'))
        self.ui.lineedit_left_max2.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_left'))
        self.ui.lineedit_right_min1.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_right'))
        self.ui.lineedit_right_max1.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_right'))
        self.ui.lineedit_right_min2.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_right'))
        self.ui.lineedit_right_max2.editingFinished.connect(lambda *_: self.ui.refresh_ui('graphicsview_right'))
