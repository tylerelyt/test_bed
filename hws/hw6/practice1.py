#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet 二分类器实现
使用预训练的 ResNet 模型构建二分类器，判断图像是否属于指定类别

功能：
1. 图像数据收集和预处理
2. ResNet 模型构建（冻结主干网络或微调）
3. 二分类训练
4. 模型评估和预测
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import requests
from pathlib import Path
import json

class ImageClassifier:
    """ResNet 二分类器类"""
    
    def __init__(self, model_name='resnet50', num_classes=2, freeze_backbone=True):
        """
        初始化分类器
        
        Args:
            model_name: 预训练模型名称 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            num_classes: 分类数量（二分类为2）
            freeze_backbone: 是否冻结主干网络
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # 构建模型
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 数据预处理
        self.transform = self._get_transforms()
        
        print(f"模型初始化完成")
        print(f"   设备: {self.device}")
        print(f"   模型: {model_name}")
        print(f"   冻结主干: {freeze_backbone}")
    
    def _build_model(self):
        """构建 ResNet 模型"""
        # 获取预训练模型
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif self.model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif self.model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
        
        # 冻结主干网络（可选）
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # 替换最后的全连接层
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def _get_transforms(self):
        """获取数据预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_transforms_train(self):
        """获取训练数据预处理转换（包含数据增强）"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

class CustomImageDataset(Dataset):
    """自定义图像数据集"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            transform: 数据预处理转换
            is_train: 是否为训练模式
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 获取图像文件列表
        self.images = []
        self.labels = []
        
        # 正类图像（标签为1）
        positive_dir = os.path.join(data_dir, 'positive')
        if os.path.exists(positive_dir):
            for img_name in os.listdir(positive_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(positive_dir, img_name))
                    self.labels.append(1)
        
        # 负类图像（标签为0）
        negative_dir = os.path.join(data_dir, 'negative')
        if os.path.exists(negative_dir):
            for img_name in os.listdir(negative_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(negative_dir, img_name))
                    self.labels.append(0)
        
        print(f"数据集加载完成: {len(self.images)} 张图像")
        print(f"   正类: {sum(self.labels)} 张")
        print(f"   负类: {len(self.labels) - sum(self.labels)} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"图像加载失败 {img_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用预处理
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageCollector:
    """图像收集器 - 从网络收集图像数据"""
    
    def __init__(self, save_dir="collected_images"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "negative"), exist_ok=True)
    
    def download_image(self, url, filename, category):
        """下载单张图像"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            filepath = os.path.join(self.save_dir, category, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"❌ 下载失败 {url}: {e}")
            return False
    
    def collect_ultraman_images(self, num_images=15):
        """收集奥特曼相关图像（正类）"""
        print("开始收集奥特曼图像...")
        
        # 奥特曼相关的搜索关键词
        keywords = [
            "ultraman", "奥特曼", "ウルトラマン", "ultra man",
            "ultraman zero", "ultraman taro", "ultraman ace"
        ]
        
        # 这里可以集成图像搜索API，暂时使用示例URL
        # 实际使用时可以调用百度图片、Google图片等API
        sample_urls = [
            "https://example.com/ultraman1.jpg",  # 示例URL
            "https://example.com/ultraman2.jpg",
        ]
        
        success_count = 0
        for i in range(num_images):
            # 实际实现中，这里应该调用图像搜索API
            # 暂时跳过，用户需要手动收集图像
            pass
        
        print(f"奥特曼图像收集完成: {success_count} 张")
        return success_count
    
    def collect_negative_images(self, num_images=15):
        """收集负类图像（非奥特曼）"""
        print("开始收集负类图像...")
        
        # 负类图像类型
        categories = ["animals", "landscapes", "cars", "buildings", "food"]
        
        success_count = 0
        for i in range(num_images):
            # 实际实现中，这里应该调用图像搜索API
            # 暂时跳过，用户需要手动收集图像
            pass
        
        print(f"负类图像收集完成: {success_count} 张")
        return success_count

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4):
        """设置训练参数"""
        # 只训练需要梯度的参数
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        print(f"训练设置完成")
        print(f"   学习率: {learning_rate}")
        print(f"   权重衰减: {weight_decay}")
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="训练中"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="验证中"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(dataloader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=20):
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, predictions, labels = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 学习率调整
            if self.scheduler:
                self.scheduler.step()
            
            # 打印结果
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 绘制训练曲线
        self._plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    def _plot_training_curves(self, train_losses, train_accs, val_losses, val_accs):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses, label='训练损失', color='blue')
        ax1.plot(val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_accs, label='训练准确率', color='blue')
        ax2.plot(val_accs, label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="评估中"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 打印结果
        print(f"\n模型评估结果")
        print("-" * 40)
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def _plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['负类', '正类'], 
                    yticklabels=['负类', '正类'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path, transform):
        """预测单张图像"""
        self.model.eval()
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"图像加载失败: {e}")
            return None
        
        # 预处理
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probability[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probability[0].cpu().numpy()
        }

def main():
    """主函数 - 演示完整流程"""
    print("ResNet 二分类器演示")
    print("=" * 50)
    
    # 1. 图像收集（需要手动收集或使用API）
    print("\n第一步：图像数据收集")
    print("请手动收集图像数据，或使用图像搜索API")
    print("目录结构：")
    print("  collected_images/")
    print("  ├── positive/     # 正类图像（奥特曼）")
    print("  └── negative/     # 负类图像（非奥特曼）")
    
    # 检查数据目录
    data_dir = "collected_images"
    if not os.path.exists(data_dir):
        print(f"\n数据目录不存在: {data_dir}")
        print("请先收集图像数据")
        return
    
    # 2. 创建数据集
    print(f"\n第二步：创建数据集")
    classifier = ImageClassifier(model_name='resnet50', freeze_backbone=True)
    
    # 创建训练和验证数据集
    train_dataset = CustomImageDataset(data_dir, classifier._get_transforms_train(), is_train=True)
    val_dataset = CustomImageDataset(data_dir, classifier._get_transforms(), is_train=False)
    
    if len(train_dataset) == 0:
        print("没有找到图像数据，请检查数据目录")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # 3. 训练模型
    print(f"\n第三步：训练模型")
    trainer = ModelTrainer(classifier.model, classifier.device)
    trainer.setup_training(learning_rate=0.001)
    
    # 开始训练
    training_results = trainer.train(train_loader, val_loader, num_epochs=15)
    
    # 4. 评估模型
    print(f"\n第四步：评估模型")
    evaluator = ModelEvaluator(classifier.model, classifier.device)
    evaluation_results = evaluator.evaluate(val_loader)
    
    # 5. 单张图像预测示例
    print(f"\n第五步：单张图像预测")
    if len(train_dataset) > 0:
        # 使用第一张图像作为示例
        sample_image_path = train_dataset.images[0]
        prediction = evaluator.predict_single_image(sample_image_path, classifier.transform)
        
        if prediction:
            class_names = ['负类', '正类']
            predicted_class = class_names[prediction['predicted_class']]
            confidence = prediction['confidence']
            
            print(f"示例图像: {os.path.basename(sample_image_path)}")
            print(f"预测结果: {predicted_class}")
            print(f"置信度: {confidence:.4f}")
    
    print(f"\n演示完成！")
    print(f"最佳验证准确率: {training_results['best_val_acc']:.2f}%")

if __name__ == "__main__":
    # 检查依赖
    try:
        import torch
        import torchvision
        print(f"PyTorch {torch.__version__} 可用")
        print(f"TorchVision {torchvision.__version__} 可用")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install torch torchvision")
        exit(1)
    
    # 运行主程序
    main()
