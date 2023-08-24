class connect_browse(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()
    
    def connect(self):
        self.ui.pbutton_previous.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=-1))
        self.ui.pbutton_previous5.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=-5))
        self.ui.pbutton_previous100.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=-100))
        self.ui.pbutton_next.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=1))
        self.ui.pbutton_next5.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=5))
        self.ui.pbutton_next100.clicked.connect(lambda *_: self.ui.treeview_datalist.select_previous_next(diff_idx=100))

    def previous_next_data(self, diff_idx=0):
        print(diff_idx)
