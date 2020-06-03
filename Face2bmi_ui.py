# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Face2bmi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Face2bmi(object):
    def setupUi(self, Face2bmi):
        Face2bmi.setObjectName("Face2bmi")
        Face2bmi.resize(1024, 759)
        self.graphicsView = QtWidgets.QLabel(Face2bmi)
        self.graphicsView.setGeometry(QtCore.QRect(70, 120, 600, 600))
        self.graphicsView.setObjectName("graphicsView")
        self.groupBox = QtWidgets.QGroupBox(Face2bmi)
        self.groupBox.setGeometry(QtCore.QRect(710, 280, 140, 291))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.camera = QtWidgets.QPushButton(self.groupBox)
        self.camera.setObjectName("camera")
        self.verticalLayout.addWidget(self.camera)
        self.stop = QtWidgets.QPushButton(self.groupBox)
        self.stop.setObjectName("stop")
        self.verticalLayout.addWidget(self.stop)
        self.saveCam = QtWidgets.QPushButton(self.groupBox)
        self.saveCam.setObjectName("saveCam")
        self.verticalLayout.addWidget(self.saveCam)
        self.groupBox_2 = QtWidgets.QGroupBox(Face2bmi)
        self.groupBox_2.setGeometry(QtCore.QRect(710, 120, 140, 141))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.bmi_Browser = QtWidgets.QTextBrowser(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(19)
        self.bmi_Browser.setFont(font)
        self.bmi_Browser.setObjectName("bmi_Browser")
        self.verticalLayout_2.addWidget(self.bmi_Browser)
        self.groupBox_3 = QtWidgets.QGroupBox(Face2bmi)
        self.groupBox_3.setGeometry(QtCore.QRect(870, 280, 140, 370))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.folder = QtWidgets.QPushButton(self.groupBox_3)
        self.folder.setObjectName("folder")
        self.verticalLayout_3.addWidget(self.folder)
        self.next = QtWidgets.QPushButton(self.groupBox_3)
        self.next.setObjectName("next")
        self.verticalLayout_3.addWidget(self.next)
        self.prev = QtWidgets.QPushButton(self.groupBox_3)
        self.prev.setObjectName("prev")
        self.verticalLayout_3.addWidget(self.prev)
        self.predict_all = QtWidgets.QPushButton(self.groupBox_3)
        self.predict_all.setObjectName("predict_all")
        self.verticalLayout_3.addWidget(self.predict_all)
        self.predict = QtWidgets.QPushButton(Face2bmi)
        self.predict.setGeometry(QtCore.QRect(885, 130, 110, 110))
        self.predict.setObjectName("predict")
        self.groupBox_4 = QtWidgets.QGroupBox(Face2bmi)
        self.groupBox_4.setGeometry(QtCore.QRect(70, 20, 930, 73))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.dir_name = QtWidgets.QLabel(self.groupBox_4)
        self.dir_name.setObjectName("dir_name")
        self.verticalLayout_4.addWidget(self.dir_name)
        self.pic_name = QtWidgets.QLabel(self.groupBox_4)
        self.pic_name.setObjectName("pic_name")
        self.verticalLayout_4.addWidget(self.pic_name)
        self.normalized = QtWidgets.QLabel(Face2bmi)
        self.normalized.setGeometry(QtCore.QRect(710, 580, 140, 140))
        self.normalized.setText("")
        self.normalized.setObjectName("normalized")

        self.retranslateUi(Face2bmi)
        QtCore.QMetaObject.connectSlotsByName(Face2bmi)

    def retranslateUi(self, Face2bmi):
        _translate = QtCore.QCoreApplication.translate
        Face2bmi.setWindowTitle(_translate("Face2bmi", "Face2bmi"))
        self.camera.setText(_translate("Face2bmi", "Open Camera"))
        self.stop.setText(_translate("Face2bmi", "Stop"))
        self.saveCam.setText(_translate("Face2bmi", "Save"))
        self.label.setText(_translate("Face2bmi", "BMI"))
        self.folder.setText(_translate("Face2bmi", "Open Files"))
        self.next.setText(_translate("Face2bmi", "Next"))
        self.prev.setText(_translate("Face2bmi", "Prev"))
        self.predict_all.setText(_translate("Face2bmi", "Predict All"))
        self.predict.setText(_translate("Face2bmi", "Predict"))
        self.dir_name.setText(_translate("Face2bmi", "Directory"))
        self.pic_name.setText(_translate("Face2bmi", "File Name"))
