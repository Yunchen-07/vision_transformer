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

from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ui import Ui_MainWindow

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_indict = self.get_class_dict()
        self.image_path = None
        # 加载模型
        model_path = "models/vision_transformer.pth"  # 模型权重路径
        self.model = vit_base_patch16_224_in21k(num_classes=len(self.class_indict), has_logits=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # 类别
        self.class_names = self.class_indict
        self.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.pushButton.clicked.connect(self.change_img)
        self.pushButton_2.clicked.connect(self.predict_img)

    def get_class_dict(self):
        """获取类别"""
        json_path = 'static/json/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            class_indict = json.load(f)
        return class_indict

    # 上传并显示图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            img_init = cv2.imread(target_image_name)  # 打开图片
            h, w, c = img_init.shape
            scale = 300 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到300的高，方便界面显示
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
            cv2.imwrite('images/target.png', img_init)
            self.label_5.setScaledContents(True)
            self.label_5.setPixmap(QPixmap("images/show.png"))
            self.label_3.setText('')
            self.label_4.setText('')

    # 预测图片
    def predict_img(self):
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        img = Image.open(r'images/target.png')
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        self.model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        result_label = "class: {}".format(self.class_indict[str(predict_cla)])
        result_prob = "prob: {}".format(predict[predict_cla].numpy())
        self.label_3.setText(result_label)  # 在界面上做显示
        self.label_4.setText(result_prob)


    def run_test(self, image_path):
        """执行完整测试流程并返回结果"""
        # 上传图像
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"测试图像不存在: {image_path}"}

        # 模拟图像上传
        target_image_name = "images/tmp_up." + image_path.split(".")[-1]
        shutil.copy(image_path, target_image_name)

        # 图像预处理（复用现有方法）
        img_init = cv2.imread(target_image_name)
        h, w, c = img_init.shape
        scale = 300 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)

        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)

        # 执行预测
        try:
            data_transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            img = Image.open(r'images/target.png')
            img = data_transform(img).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                output = torch.squeeze(self.model(img)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            result = {
                "status": "success",
                "predicted_class": self.class_indict[str(predict_cla)],
                "confidence": float(predict[predict_cla].numpy()),
                "all_probs": {self.class_indict[str(i)]: float(predict[i].numpy())
                              for i in range(len(predict))}
            }
            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_window = MyWindow()
    my_window.show()
    sys.exit(app.exec_())
