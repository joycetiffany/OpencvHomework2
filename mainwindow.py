# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
from matplotlib import pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(436, 292)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btn1_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn1_1.setObjectName("pushButton")
        self.btn1_1.clicked.connect(pushButton_pushedbtn1_1)
        self.btn1_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn1_1.setObjectName("btn1_1")
        self.verticalLayout_3.addWidget(self.btn1_1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btn2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.btn2_1.setObjectName("pushButton")
        self.btn2_1.clicked.connect(pushButton_pushedbtn2_1)
        self.btn2_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn2_1.setObjectName("btn2_1")
        self.verticalLayout_4.addWidget(self.btn2_1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.btn3_1 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn3_1.setObjectName("pushButton")
        self.btn3_1.clicked.connect(pushButton_pushedbtn3_1)
        self.btn3_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn3_1.setObjectName("btn3_1")
        self.verticalLayout_5.addWidget(self.btn3_1)
        self.btn3_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.btn3_2.setObjectName("pushButton")
        self.btn3_2.clicked.connect(pushButton_pushedbtn3_2)
        self.btn3_2.setMinimumSize(QtCore.QSize(0, 50))
        self.btn3_2.setObjectName("btn3_2")
        self.verticalLayout_5.addWidget(self.btn3_2)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.btn4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.btn4_1.setObjectName("pushButton")
        self.btn4_1.clicked.connect(pushButton_pushedbtn4_1)
        self.btn4_1.setMinimumSize(QtCore.QSize(0, 50))
        self.btn4_1.setObjectName("btn4_1")
        self.verticalLayout_6.addWidget(self.btn4_1)
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Stereo"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Display"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Background Subtraction"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Background Subtraction"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Feature Tracking"))
        self.btn3_1.setText(_translate("MainWindow", "3.1 Preprocessing"))
        self.btn3_2.setText(_translate("MainWindow", "3.2 Video Tracking"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Augmented Reality"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 Augmented Reality"))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent=parent)
        self.setupUi(self)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_F5:
            self.close()


def pushButton_pushedbtn1_1(self):
    imgR = cv2.imread('imR.png',1)
    imgL = cv2.imread('imL.png',1)
    window_size = 9
    min_disp = 13
    num_disp = 109 - min_disp
    blockSize = 14
    uniquenessRatio = 0
    speckleRange = 3
    speckleWindowSize = 0
    disp12MaxDiff = 28
    P1 = 8 * 3 * window_size ** 2
    P2 = 32 * 3 * window_size ** 2
    # stereo = cv2.StereoBM_create(64, 9)
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=blockSize,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 8.0
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)
    # disparity = stereo.compute(imgL,imgR)
    # cv2.imshow('gray', disparity)
    cv2.waitKey()
    cv2.destroyAllWindows()

def pushButton_pushedbtn2_1(self):
    print("btn2_1")

def pushButton_pushedbtn3_1(self):
    print("btn3_1")

def pushButton_pushedbtn3_2(self):
    print("btn3_2")

def pushButton_pushedbtn4_1(self):
    print("btn4_1")



