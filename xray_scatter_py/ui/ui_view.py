class connect_view(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()
    
    def connect(self):

        self.ui.actionGrid.isCheckable = True
        self.ui.actionGrid.triggered.connect(lambda *_: self.ui.set_ui_params(grid=self.ui.actionGrid.isChecked()))
        self.ui.actionGrid.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionZoom.triggered.connect(lambda *_: self.zoom())
        self.ui.actionZoom.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionRotate.triggered.connect(lambda *_: self.rotate())
        self.ui.actionRotate.triggered.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.actionImage_History.triggered.connect(lambda *_: self.image_history())

    def zoom(self):
        print('zoom')
        self.ui.set_ui_params(zoom=True, zoom_min=0.5, zoom_max=2.0) # the numbers should come from user selection

    def rotate(self):
        print('rotate')
        self.ui.set_ui_params(rotate=True, rotate_angle=90) # the numbers should come from user selection

    def image_history(self):
        print('image_history')
