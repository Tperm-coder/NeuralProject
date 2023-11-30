from gui import MyGUI
from PyQt5.QtWidgets import *
from PyQt5 import uic


#constants
UI_PATH = "./gui.ui"


def main():
    # initialize GUI
    app = QApplication([])
    window = MyGUI(UI_PATH)
    app.exec_()


# initial function
if __name__ == '__main__':
    main()