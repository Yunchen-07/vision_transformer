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


# 定义CNN模型
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 数据加载函数
def data_loader(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 计算统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 计算当前准确率和平均损失
        current_acc = 100. * correct / total
        current_loss = running_loss / (loop.n + 1)

        # 更新进度条信息
        loop.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# 修改验证函数以显示实时准确率
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

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 保存预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 计算当前准确率和平均损失
            current_acc = 100. * correct / total
            current_loss = running_loss / (loop.n + 1)

            # 更新进度条信息
            loop.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# 绘制训练过程图
def plot_training_process(train_losses, val_losses, accuracies, save_dir):
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
    plt.plot(epochs, accuracies, 'g-', label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_process.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
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

    # 绘制性能指标柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, metrics_df['Precision'], width, label='Precision', color='blue', alpha=0.7)
    plt.bar(x, metrics_df['Recall'], width, label='Recall', color='green', alpha=0.7)
    plt.bar(x + width, metrics_df['F1-score'], width, label='F1-score', color='red', alpha=0.7)

    plt.xlabel('类别')
    plt.ylabel('得分')
    plt.title('各类别性能指标')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存性能指标到CSV
    metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False, encoding='utf-8')


# 保存训练历史
def save_training_history(train_losses, val_losses, accuracies, save_dir):
    # 保存为CSV
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Val_Accuracy': accuracies
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # 保存为PTH
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    torch.save(history, os.path.join(save_dir, 'training_history.pth'))


# 主函数
if __name__ == "__main__":
    try:
        # 创建保存目录
        save_dir = f"model_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        # 数据集路径
        train_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\train"
        test_dir = r"D:\BaiduNetdiskEntpriseDownload\facial_emotion_datasets\facial_emotion_datasets\test"

        # 加载数据
        train_loader, test_loader, class_names = data_loader(train_dir, test_dir)
        print("类别名称: ", class_names)

        # 初始化模型、损失函数和优化器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        model = EmotionCNN(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练参数
        num_epochs = 20
        best_accuracy = 0.0
        best_model_path = os.path.join(save_dir, "best_model.pth")
        train_losses = []
        val_losses = []
        accuracies = []

        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
            # 验证阶段
            val_loss, val_acc, all_preds, all_labels = validate(model, test_loader, criterion, device, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(val_acc)

            print(
                f"\nEpoch {epoch + 1}/{num_epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Train Acc = {train_acc * 100:.2f}%, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val Acc = {val_acc * 100:.2f}%"
            )

            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                model_info = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_accuracy,
                    'class_names': class_names,
                    'input_size': (224, 224),
                    'normalize_mean': [0.5, 0.5, 0.5],
                    'normalize_std': [0.5, 0.5, 0.5]
                }
                torch.save(model_info, best_model_path)
                print(f"最佳模型已保存 (准确率: {best_accuracy * 100:.2f}%)")

        # 保存训练历史并绘制训练过程图
        save_training_history(train_losses, val_losses, accuracies, save_dir)
        plot_training_process(train_losses, val_losses, accuracies, save_dir)

        # 加载最佳模型并进行最终评估
        print("\n加载最佳模型进行评估...")
        model_info = torch.load(best_model_path)
        model.load_state_dict(model_info['model_state_dict'])

        # 在测试集上进行最终评估
        final_loss, final_accuracy, final_preds, final_labels = validate(model, test_loader, criterion, device, -1)

        # 计算并保存混淆矩阵
        cm = confusion_matrix(final_labels, final_preds)
        plot_confusion_matrix(cm, class_names, save_dir)

        # 计算并保存分类报告
        report = classification_report(final_labels, final_preds,
                                       target_names=class_names,
                                       output_dict=True)

        # 保存分类报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("分类报告:\n")
            f.write(classification_report(final_labels, final_preds, target_names=class_names))

        # 绘制并保存性能指标
        plot_performance_metrics(report, class_names, save_dir)

        # 打印最终结果
        print("\n训练完成!")
        print(f"最佳模型保存在: {save_dir}")
        print(f"最佳验证准确率: {model_info['accuracy']:.4f}")
        print(f"最终测试集准确率: {final_accuracy:.4f}")
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