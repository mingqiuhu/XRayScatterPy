class connect_browse(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()
    
    def connect(self):
        self.ui.pbutton_previous.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-1))
        self.ui.pbutton_previous.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.pbutton_previous5.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-5))
        self.ui.pbutton_previous5.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.pbutton_previous100.clicked.connect(lambda *_: self.previous_next_data(diff_idx=-100))
        self.ui.pbutton_previous100.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.pbutton_next.clicked.connect(lambda *_: self.previous_next_data(diff_idx=1))
        self.ui.pbutton_next.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.pbutton_next5.clicked.connect(lambda *_: self.previous_next_data(diff_idx=5))
        self.ui.pbutton_next5.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))
        self.ui.pbutton_next100.clicked.connect(lambda *_: self.previous_next_data(diff_idx=100))
        self.ui.pbutton_next100.clicked.connect(lambda *_: self.ui.refresh_ui('graphicsview_left', 'graphicsview_right'))

    def previous_next_data(self, diff_idx=0):
        print(diff_idx)
