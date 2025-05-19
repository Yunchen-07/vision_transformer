import sys
import os
import shutil
import json
import warnings

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from model_vision_transformer import vit_base_patch16_224_in21k

# 导入CNN和ResNet18模型定义
from cnn_train import EmotionCNN
from resnet18_train import ResNet18

from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ui_1 import Ui_MainWindow

# 禁用警告信息
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # 设置设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 获取类别字典
        self.class_indict = self.get_class_dict()
        self.image_path = None

        # 初始化并加载所有模型
        self.models = {
            "Vision Transformer": self.load_vit_model(),
            "CNN": self.load_cnn_model(),
            "ResNet18": self.load_resnet18_model()
        }

        # 设置默认模型为ViT
        self.current_model = self.models["Vision Transformer"]

        # 设置UI
        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 连接信号和槽
        self.pushButton.clicked.connect(self.change_img)
        self.pushButton_2.clicked.connect(self.predict_img)
        self.modelComboBox.currentTextChanged.connect(self.change_model)


    def get_class_dict(self):
        """获取类别映射字典"""
        json_path = 'static/json/class_indices.json'
        assert os.path.exists(json_path), f"文件不存在: '{json_path}'"
        with open(json_path, "r", encoding="utf-8") as f:
            class_indict = json.load(f)
        return class_indict

    def load_vit_model(self):
        """加载Vision Transformer模型"""
        try:
            model = vit_base_patch16_224_in21k(num_classes=len(self.class_indict),
                                               has_logits=False).to(self.device)
            model.load_state_dict(torch.load("models/vision_transformer.pth",
                                             map_location=self.device))
            print("ViT模型加载成功")
            return model
        except Exception as e:
            print(f"ViT模型加载失败: {str(e)}")
            return None

    def load_cnn_model(self):
        """加载CNN模型"""
        try:
            model = EmotionCNN(num_classes=len(self.class_indict)).to(self.device)
            checkpoint = torch.load("CNN_result/best_model.pth",
                                    map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("CNN模型加载成功")
            return model
        except Exception as e:
            print(f"CNN模型加载失败: {str(e)}")
            return None

    def load_resnet18_model(self):
        """加载ResNet18模型"""
        try:
            model = ResNet18(num_classes=len(self.class_indict)).to(self.device)
            checkpoint = torch.load("ResNet18_results/best_model.pth",
                                    map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("ResNet18模型加载成功")
            return model
        except Exception as e:
            print(f"ResNet18模型加载失败: {str(e)}")
            return None

    def change_model(self, model_name):
        """切换预测模型"""
        if self.models[model_name] is not None:
            self.current_model = self.models[model_name]
            print(f"切换到模型: {model_name}")
        else:
            print(f"模型 {model_name} 未能正确加载")
        # 清空之前的预测结果
        self.label_3.setText("")
        self.label_4.setText("")

    def change_img(self):
        """上传并显示图片"""
        openfile_name = QFileDialog.getOpenFileName(self, '选择图片', '',
                                                    'Image files(*.jpg *.png *.jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            return

        try:
            # 保存并处理图片
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)

            # 读取并调整图片大小
            img_init = cv2.imread(target_image_name)
            h, w, c = img_init.shape
            scale = 300 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)

            # 调整图片大小为模型输入尺寸
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)

            # 显示图片
            self.label_5.setScaledContents(True)
            self.label_5.setPixmap(QPixmap("images/show.png"))
            self.label_3.setText('')
            self.label_4.setText('')

        except Exception as e:
            print(f"图片处理出错: {str(e)}")
            self.label_3.setText("图片处理出错")
            self.label_4.setText(str(e))

    def predict_img(self):
        """使用当前选择的模型进行预测"""
        if self.current_model is None:
            self.label_3.setText("错误")
            self.label_4.setText("当前模型未正确加载")
            return

        # 图像预处理
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        try:
            # 加载和转换图像
            img = Image.open('images/target.png')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            # 执行预测
            self.current_model.eval()
            with torch.no_grad():
                output = torch.squeeze(self.current_model(img.to(self.device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # 显示预测结果
            result_label = f"类别: {self.class_indict[str(predict_cla)]}"
            result_prob = f"置信度: {float(predict[predict_cla]):.4f}"

            self.label_3.setText(result_label)
            self.label_4.setText(result_prob)

        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            self.label_3.setText("预测出错")
            self.label_4.setText(str(e))

    def run_test(self, image_path, model_name="Vision Transformer"):
        """执行完整测试流程并返回结果"""
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"测试图像不存在: {image_path}"}

        try:
            # 设置当前模型
            if self.models[model_name] is None:
                return {"status": "error", "message": f"模型 {model_name} 未正确加载"}
            self.current_model = self.models[model_name]

            # 处理图像
            target_image_name = "images/tmp_up." + image_path.split(".")[-1]
            shutil.copy(image_path, target_image_name)

            img_init = cv2.imread(target_image_name)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)

            # 预测
            data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            img = Image.open('images/target.png')
            img = data_transform(img).unsqueeze(0)

            self.current_model.eval()
            with torch.no_grad():
                output = torch.squeeze(self.current_model(img.to(self.device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # 返回结果
            return {
                "status": "success",
                "predicted_class": self.class_indict[str(predict_cla)],
                "confidence": float(predict[predict_cla]),
                "all_probs": {self.class_indict[str(i)]: float(predict[i])
                              for i in range(len(predict))}
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyWindow()
    my_window.show()
    sys.exit(app.exec_())
