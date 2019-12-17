# from PyQt5 import QtWidgets
# from PyQt5 import uic
# from PyQt5.QtGui import QIcon, QPixmap, QClipboard
# from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QHBoxLayout, QDialog, QBoxLayout, QMainWindow
# from PyQt5.QtWebEngineWidgets import QWebEnginePage
# from PyQt5.QtWebEngineWidgets import QWebEngineView
# from PyQt5.QtCore import QUrl, QDir, QMimeData
from mainwindow import MainWindow
from PyQt5 import QtWidgets


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())