class connect_reflectivity(object):
    def __init__(self, ui):
        self.ui = ui
        self.connect()

    def connect(self):
        self.ui.actionReflectivity.triggered.connect(lambda *_: self.reflectivity())

    def reflectivity(self):
        print('reflectivity')