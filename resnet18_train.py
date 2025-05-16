import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

# 设置中文字体和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出维度不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


# 定义ResNet18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重
        self._initialize_weights()

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 数据加载函数
def data_loader(train_dir, test_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes


# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch + 1} (Training)")

    for _, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_acc = 100. * correct / total
        current_loss = running_loss / (loop.n + 1)

        loop.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# 验证函数
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    loop = tqdm(enumerate(val_loader), total=len(val_loader),
                desc=f"Epoch {epoch + 1} (Validation)")

    with torch.no_grad():
        for _, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_acc = 100. * correct / total
            current_loss = running_loss / (loop.n + 1)

            loop.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


# 绘制训练过程图
def plot_training_process(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, val_accs, 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_process.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names, save_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# 绘制性能指标
def plot_performance_metrics(report, class_names, save_dir):
    metrics_df = pd.DataFrame(columns=['Class', 'Precision', 'Recall', 'F1-score'])

    for i, class_name in enumerate(class_names):
        metrics_df.loc[i] = [
            class_name,
            report[class_name]['precision'],
            report[class_name]['recall'],
            report[class_name]['f1-score']
        ]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, metrics_df['Precision'], width,
            label='Precision', color='blue', alpha=0.7)
    plt.bar(x, metrics_df['Recall'], width,
            label='Recall', color='green', alpha=0.7)
    plt.bar(x + width, metrics_df['F1-score'], width,
            label='F1-score', color='red', alpha=0.7)

    plt.xlabel('类别')
    plt.ylabel('得分')
    plt.title('各类别性能指标')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'),
                      index=False, encoding='utf-8')


# 保存训练历史
def save_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_Accuracy': train_accs,
        'Val_Accuracy': val_accs
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    torch.save(history, os.path.join(save_dir, 'training_history.pth'))


if __name__ == "__main__":
    try:
        # 创建保存目录
        save_dir = f"resnet18_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        # 数据集路径
        train_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\train"
        test_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\test"

        # 加载数据
        train_loader, test_loader, class_names = data_loader(train_dir, test_dir, batch_size=32)
        print("类别名称: ", class_names)

        # 初始化模型、损失函数和优化器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        model = ResNet18(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1, patience=3)

        # 训练参数
        num_epochs = 50
        best_accuracy = 0.0
        best_model_path = os.path.join(save_dir, "best_model.pth")

        # 记录训练历史
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # 早停参数
        patience = 7
        patience_counter = 0

        # 在训练循环中修改这部分代码
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = train(model, train_loader, criterion,
                                          optimizer, device, epoch)
            # 验证阶段
            val_loss, val_acc, all_preds, all_labels = validate(model, test_loader,
                                                                criterion, device, epoch)

            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # 如果学习率发生变化，打印信息
            if new_lr != current_lr:
                print(f'\n学习率已更新: {current_lr:.6f} -> {new_lr:.6f}')

            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"\nEpoch {epoch + 1}/{num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Train Acc = {train_acc * 100:.2f}%, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Acc = {val_acc * 100:.2f}%, "
                f"Learning Rate = {new_lr:.6f}"
            )

            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': best_accuracy,
                    'class_names': class_names
                }, best_model_path)

                print(f"保存最佳模型 (准确率: {best_accuracy * 100:.2f}%)")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                print(f"\n{patience}个epoch未改善，停止训练")
                break

            # 绘制当前训练过程图
            plot_training_process(train_losses, val_losses, train_accs,
                                  val_accs, save_dir)

        # 保存训练历史
        save_training_history(train_losses, val_losses, train_accs, val_accs, save_dir)

        # 加载最佳模型进行最终评估
        print("\n加载最佳模型进行评估...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 最终评估
        final_loss, final_acc, final_preds, final_labels = validate(model,
                                                                    test_loader,
                                                                    criterion,
                                                                    device, -1)

        # 计算并保存混淆矩阵
        cm = confusion_matrix(final_labels, final_preds)
        plot_confusion_matrix(cm, class_names, save_dir)

        # 计算并保存分类报告
        report = classification_report(final_labels, final_preds,
                                       target_names=class_names,
                                       output_dict=True)

        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w',
                  encoding='utf-8') as f:
            f.write("分类报告:\n")
            f.write(classification_report(final_labels, final_preds,
                                          target_names=class_names))

        # 绘制并保存性能指标
        plot_performance_metrics(report, class_names, save_dir)

        # 打印最终结果
        print("\n训练完成!")
        print(f"最佳模型保存在: {save_dir}")
        print(f"最佳验证准确率: {best_accuracy * 100:.2f}%")
        print(f"最终测试集准确率: {final_acc * 100:.2f}%")
        print("\n各类别性能指标:")
        for class_name in class_names:
            metrics = report[class_name]
            print(f"{class_name}:")
            print(f"  准确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1-score']:.4f}")

    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()