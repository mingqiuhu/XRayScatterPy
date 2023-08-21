class connect_calibration(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()
    
    def connect(self):
        self.ui.actionLoad.triggered.connect(lambda *_: self.load())
        self.ui.actionLoad.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionSave_current.triggered.connect(lambda *_: self.save_current())
        self.ui.actionManually_set.triggered.connect(lambda *_: self.manually_set())
        self.ui.actionManually_set.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionq_calibration.triggered.connect(lambda *_: self.q_calibration())
        self.ui.actionq_calibration.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionintensity_calibration.triggered.connect(lambda *_: self.intensity_calibration())
        self.ui.actionintensity_calibration.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionauto_correct_center.triggered.connect(lambda *_: self.auto_correct_center())
        self.ui.actionauto_correct_center.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))


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
