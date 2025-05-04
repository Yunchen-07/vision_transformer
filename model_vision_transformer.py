import os
import sys
import json
from time import *
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from torchvision import transforms, datasets
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

import netron
import torch.onnx

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


class Logger(object):
    """控制台输出记录到文件."""

    def __init__(self, file_name, stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


class MainProcess:
    def __init__(self, train_path, test_path, model_name):
        self.train_path = train_path
        self.test_path = test_path
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def data_load(self):
        """加载数据"""
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
        train_dataset = datasets.ImageFolder(root=self.train_path,
                                             transform=data_transform["train"])
        train_num = len(train_dataset)
        class_dict1 = train_dataset.class_to_idx
        class_dict2 = dict((val, key) for key, val in class_dict1.items())
        class_names = list(class_dict2.values())
        json_str = json.dumps(class_dict2, indent=4)
        with open('static/json/class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        batch_size = 32
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw,
                                                   pin_memory=True)

        validate_dataset = datasets.ImageFolder(root=self.test_path,
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      pin_memory=True)
        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))

        return train_loader, validate_loader, class_names, train_num, val_num

    def model_load(self, num_classes):
        """加载模型"""
        return vit_base_patch16_224_in21k(num_classes=num_classes, has_logits=False).to(self.device)

    @staticmethod
    def show_loss_acc(train_loss_history, train_acc_history,
                      test_loss_history, test_acc_history):
        """展示训练过程的曲线"""
        # 按照上下结构将图画输出
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(test_acc_history, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(test_loss_history, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('results/vision_transformer_results.png', dpi=100)

    @staticmethod
    def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
        """热力图"""
        # 这里是创建一个画布
        fig, ax = plt.subplots(figsize=(10, 10))
        # cmap https://blog.csdn.net/ztf312/article/details/102474190
        im = ax.imshow(harvest, cmap="OrRd")
        # 这里是修改标签
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(y_labels)))
        ax.set_yticks(np.arange(len(x_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(y_labels)
        ax.set_yticklabels(x_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # 添加每个热力块的具体数值
        # Loop over data dimensions and create text annotations.
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, round(harvest[i, j], 2),
                               ha="center", va="center", color="black")
        ax.set_xlabel("Predict label")
        ax.set_ylabel("Actual label")
        ax.set_title(title)
        fig.tight_layout()
        plt.colorbar(im)
        plt.savefig(save_name, dpi=100)
        # plt.show()

    def heatmaps(self, model, validate_loader, class_names):
        """生成热力图"""
        print("正在生成热力图...")
        # 加载模型权重
        weights_path = self.model_name
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))

        # 对模型分开进行推理
        test_real_labels = []
        test_pre_labels = []
        model.eval()
        with torch.no_grad():  # 张量的计算过程中无需计算梯度
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(self.device))
                predict_y = torch.max(outputs, dim=1)[1]  # 每行最大值的索引
                test_real_labels.extend(val_labels.numpy())
                test_pre_labels.extend(predict_y.cpu().numpy())

        class_names_length = len(class_names)
        heat_maps = np.zeros((class_names_length, class_names_length))
        for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
            heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1
        heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
        heat_maps_float = heat_maps / heat_maps_sum
        print(heat_maps_float)
        self.show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names,
                           harvest=heat_maps_float, save_name="results/vision_transformer_heatmap.png")

        return test_real_labels, test_pre_labels

    def calculate_confusion_matrix(self, test_real_labels, test_pre_labels, class_names):
        """计算混淆矩阵"""
        print("正在生混淆矩阵...")
        report_str = classification_report(test_real_labels, test_pre_labels, target_names=class_names)
        print(report_str)
        report_dict = classification_report(test_real_labels, test_pre_labels, target_names=class_names,
                                            output_dict=True)
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        # 准确率
        plt.subplot(2, 2, 1)
        accuracy_value = [report_dict['accuracy']]
        bar1 = plt.bar(['all class'], accuracy_value, width=0.02)
        plt.bar_label(bar1, fmt='%.2f', label_type='edge')
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=90)
        # 精确率
        plt.subplot(2, 2, 2)
        precision_value = [report_dict[i]['precision'] for i in class_names]
        bar2 = plt.bar(class_names, precision_value)
        plt.bar_label(bar2, fmt='%.2f', label_type='edge')
        plt.title("Precision of each class")
        plt.ylabel("Precision")
        plt.xticks(rotation=90)
        # 召回率
        plt.subplot(2, 2, 3)
        recall_value = [report_dict[i]['recall'] for i in class_names]
        bar3 = plt.bar(class_names, recall_value)
        plt.bar_label(bar3, fmt='%.2f', label_type='edge')
        plt.title("Recall of each class")
        plt.ylabel("Recall")
        plt.xticks(rotation=90)
        # F1-score
        plt.subplot(2, 2, 4)
        f1_score_value = [report_dict[i]['f1-score'] for i in class_names]
        bar4 = plt.bar(class_names, f1_score_value)
        plt.bar_label(bar4, fmt='%.2f', label_type='edge')
        plt.title("F1-score of each class")
        plt.ylabel("F1-score")
        plt.xticks(rotation=90)
        plt.savefig('results/vision_transformer_confusion_matrix.png', dpi=100)

    def main(self, epochs):
        # 记录训练过程
        log_file_name = './results/vision_transformer训练和验证过程.txt'
        # 记录正常的 print 信息
        sys.stdout = Logger(log_file_name)

        # 开始训练，记录开始时间
        begin_time = time()
        # 加载数据
        train_loader, validate_loader, class_names, train_num, val_num = self.data_load()
        print("class_names: ", class_names)
        train_steps = len(train_loader)
        val_steps = len(validate_loader)
        # 加载模型
        model = self.model_load(len(class_names))  # 创建模型

        # 加载预训练权重
        model_weight_path = "models/vit_base_patch16_224_in21k_pre.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=self.device)

        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

        freeze_layers = False  # 是否冻结网络层
        if freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        # 将模型放入GPU中
        model.to(self.device)
        # 定义损失函数
        loss_function = nn.CrossEntropyLoss()
        # 定义优化器
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
        # 定义学习率下降策略
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        train_loss_history, train_acc_history = [], []
        test_loss_history, test_acc_history = [], []
        best_acc = 0.0

        for epoch in range(0, epochs):
            # 下面是模型训练
            model.train()
            running_loss = 0.0
            train_acc = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            # 进来一个batch的数据，计算一次梯度，更新一次网络
            for step, data in enumerate(train_bar):
                images, labels = data  # 获取图像及对应的真实标签
                outputs = model(images.to(self.device))  # 得到预测的标签
                train_loss = loss_function(outputs, labels.to(self.device))  # 计算损失
                train_loss.backward()  # 反向传播，计算当前梯度
                optimizer.step()  # 根据梯度更新网络参数
                optimizer.zero_grad()  # 清空过往梯度

                # print statistics
                running_loss += train_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]  # 每行最大值的索引
                # torch.eq()进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                train_acc += torch.eq(predict_y, labels.to(self.device)).sum().item()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         train_loss)
            scheduler.step()  # 更新学习率
            # 下面是模型验证
            model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
            val_acc = 0.0  # accumulate accurate number / epoch
            testing_loss = 0.0
            with torch.no_grad():  # 张量的计算过程中无需计算梯度
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(self.device))

                    val_loss = loss_function(outputs, val_labels.to(self.device))  # 计算损失
                    testing_loss += val_loss.item()

                    predict_y = torch.max(outputs, dim=1)[1]  # 每行最大值的索引
                    # torch.eq()进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
                    val_acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()

            train_loss = running_loss / train_steps
            train_accurate = train_acc / train_num
            test_loss = testing_loss / val_steps
            val_accurate = val_acc / val_num

            train_loss_history.append(train_loss)
            train_acc_history.append(train_accurate)
            test_loss_history.append(test_loss)
            test_acc_history.append(val_accurate)

            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, train_loss, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), self.model_name)

        # 记录结束时间
        end_time = time()
        run_time = end_time - begin_time
        print('该循环程序运行时间：', run_time, "s")
        # 数据保存
        str_train_loss_history = ','.join([str(i) for i in train_loss_history])
        str_train_acc_history = ','.join([str(i) for i in train_acc_history])
        str_test_loss_history = ','.join([str(i) for i in test_loss_history])
        str_test_acc_history = ','.join([str(i) for i in test_acc_history])
        with open('results/vision_transformer_loss_acc.txt', 'w') as f:
            f.write("train_loss_history: " + str_train_loss_history + "\n")
            f.write("train_acc_history: " + str_train_acc_history + "\n")
            f.write("test_loss_history: " + str_test_loss_history + "\n")
            f.write("test_acc_history: " + str_test_acc_history + "\n")

        # 绘制模型训练过程图
        self.show_loss_acc(train_loss_history, train_acc_history,
                           test_loss_history, test_acc_history)
        # 画热力图
        test_real_labels, test_pre_labels = self.heatmaps(model, validate_loader, class_names)
        # 计算混淆矩阵
        self.calculate_confusion_matrix(test_real_labels, test_pre_labels, class_names)


if __name__ == '__main__':
    # todo 加载数据集， 修改数据集的路径
    train_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\train"
    test_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\test"

    model_name0 = r"models/vision_transformer.pth"

    cnn = MainProcess(train_dir, test_dir, model_name0)
    # 主程序入口，更改训练轮数
    cnn.main(epochs=50)
