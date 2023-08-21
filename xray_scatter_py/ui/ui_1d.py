class connect_1d(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()

    def connect(self):
        self.ui.pbutton_left_draw.clicked.connect(lambda *_: self.draw_1d_area(pos='left'))
        self.ui.pbutton_right_draw.clicked.connect(lambda *_: self.draw_1d_area(pos='right'))
        self.ui.pbutton_left_show.clicked.connect(lambda *_: self.show_1d(pos='left'))
        self.ui.pbutton_right_show.clicked.connect(lambda *_: self.show_1d(pos='right'))    
   
    def draw_1d_area(self, pos=None):
        print(pos)

    def show_1d(self, pos=None):
        print(pos)
