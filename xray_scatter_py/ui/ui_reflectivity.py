class ui_reflectivity():
    def __init__(self, ui):
        self.ui = ui

    def connect(self):
        self.ui.actionReflectivity.triggered.connect(lambda *_: self.reflectivity())

    def reflectivity(self):
        print('reflectivity')