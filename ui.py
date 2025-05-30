# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(783, 784)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 760, 760))
        self.frame.setMinimumSize(QtCore.QSize(760, 760))
        self.frame.setStyleSheet("#frame{\n"
                                 "background-color: rgb(255, 255, 255);\n"
                                 "}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(100, 20, 571, 541))
        self.frame_2.setMinimumSize(QtCore.QSize(541, 511))
        self.frame_2.setStyleSheet("#frame_2{\n"
                                   "border: 1px dashed black;\n"
                                   "}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setGeometry(QtCore.QRect(1, 1, 669, 48))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label = QtWidgets.QLabel(self.frame_4)
        self.label.setGeometry(QtCore.QRect(0, 0, 571, 41))
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("font: 16pt \"隶书\";")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setGeometry(QtCore.QRect(1, 49, 571, 71))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")

        # 添加模型选择下拉框
        self.modelComboBox = QtWidgets.QComboBox(self.frame_5)
        self.modelComboBox.setGeometry(QtCore.QRect(20, 30, 120, 28))
        self.modelComboBox.setObjectName("modelComboBox")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")

        # 添加分隔线
        self.line = QtWidgets.QFrame(self.frame_5)
        self.line.setGeometry(QtCore.QRect(0, 0, 591, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        # 添加按钮
        self.pushButton = QtWidgets.QPushButton(self.frame_5)
        self.pushButton.setGeometry(QtCore.QRect(150, 30, 93, 28))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 30, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 30, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")

        # 添加新按钮 - 模型性能
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_4.setGeometry(QtCore.QRect(450, 30, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")

        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setGeometry(QtCore.QRect(19, 119, 531, 421))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.frame_7 = QtWidgets.QFrame(self.frame_6)
        self.frame_7.setGeometry(QtCore.QRect(80, 10, 400, 400))
        self.frame_7.setMinimumSize(QtCore.QSize(400, 400))
        self.frame_7.setStyleSheet("#frame_7{\n"
                                   "border: 1px dashed black;\n"
                                   "}")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.label_5 = QtWidgets.QLabel(self.frame_7)
        self.label_5.setGeometry(QtCore.QRect(0, 0, 400, 400))
        self.label_5.setMinimumSize(QtCore.QSize(400, 400))
        self.label_5.setMaximumSize(QtCore.QSize(400, 400))
        self.label_5.setStyleSheet("font: 18pt \"方正姚体\";")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(100, 570, 571, 171))
        self.frame_3.setStyleSheet("#frame_3{\n"
                                   "border: 1px dashed black;\n"
                                   "}")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.label_2 = QtWidgets.QLabel(self.frame_3)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 131, 16))
        self.label_2.setStyleSheet("font: 12pt \"隶书\";\n"
                                   "")
        self.label_2.setObjectName("label_2")
        self.frame_8 = QtWidgets.QFrame(self.frame_3)
        self.frame_8.setGeometry(QtCore.QRect(20, 39, 531, 111))
        self.frame_8.setStyleSheet("")
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.label_3 = QtWidgets.QLabel(self.frame_8)
        self.label_3.setGeometry(QtCore.QRect(30, 10, 471, 41))
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame_8)
        self.label_4.setGeometry(QtCore.QRect(30, 60, 471, 41))
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.pushButton_3.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "人脸表情识别操作平台"))
        self.modelComboBox.setItemText(0, _translate("MainWindow", "Vision Transformer"))
        self.modelComboBox.setItemText(1, _translate("MainWindow", "CNN"))
        self.modelComboBox.setItemText(2, _translate("MainWindow", "ResNet18"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "关闭"))
        self.pushButton_4.setText(_translate("MainWindow", "模型性能"))  # 设置新按钮文字
        self.label_5.setText(_translate("MainWindow", "识别窗口"))
        self.label_2.setText(_translate("MainWindow", "识别结果为："))
