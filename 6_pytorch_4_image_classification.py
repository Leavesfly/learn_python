# pytorch_4_image_classification.py - 图像分类实战
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("=== PyTorch 教程 4：图像分类实战 ===\n")

# 1. 数据预处理和加载
print("1. 数据预处理和加载")
print("-" * 30)

# 定义图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 由于可能无法下载 CIFAR-10，我们创建模拟数据
print("创建模拟 CIFAR-10 数据...")

# 模拟 CIFAR-10 数据
class MockCIFAR10:
    def __init__(self, n_samples=1000):
        self.data = torch.randn(n_samples, 3, 32, 32)
        self.targets = torch.randint(0, 10, (n_samples,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 创建训练和测试数据
train_dataset = MockCIFAR10(n_samples=2000)
test_dataset = MockCIFAR10(n_samples=500)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"训练数据: {len(train_dataset)} 样本")
print(f"测试数据: {len(test_dataset)} 样本")
print(f"类别数: {len(classes)}")

# 2. 定义 CNN 模型
print("\n2. 定义 CNN 模型")
print("-" * 30)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNN()
print(net)

# 检查是否有 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
net.to(device)

# 3. 定义损失函数和优化器
print("\n3. 定义损失函数和优化器")
print("-" * 40)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("损失函数: CrossEntropyLoss")
print("优化器: SGD with momentum=0.9, lr=0.001")

# 4. 训练模型
print("\n4. 训练模型")
print("-" * 20)

n_epochs = 20
train_losses = []
train_accuracies = []

for epoch in range(n_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    net.train()  # 设置为训练模式
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 20 == 19:  # 每 20 个 batch 打印一次
            print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 20:.3f}')
            running_loss = 0.0
    
    # 计算这个 epoch 的准确率
    epoch_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(trainloader)
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f'Epoch {epoch + 1} - Accuracy: {epoch_accuracy:.2f}%')

print('训练完成!')

# 5. 测试模型
print("\n5. 测试模型")
print("-" * 20)

net.eval()  # 设置为评估模式
correct = 0
total = 0

# 不计算梯度
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'测试集准确率: {test_accuracy:.2f}%')

# 6. 分类别准确率
print("\n6. 分类别准确率")
print("-" * 30)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

net.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {accuracy:.2f}%')

# 7. 模型改进技巧
print("\n7. 模型改进技巧")
print("-" * 30)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 使用批归一化和更深的网络
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

print("改进的 CNN 模型:")
print("- 添加批归一化层")
print("- 增加网络深度")
print("- 使用 Dropout 防止过拟合")
print("- 更多的卷积核")

# 8. 学习率调度
print("\n8. 学习率调度")
print("-" * 30)

improved_net = ImprovedCNN().to(device)
optimizer = optim.Adam(improved_net.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("使用 StepLR 调度器：每 7 个 epoch 学习率乘以 0.1")

# 9. 保存最佳模型
print("\n9. 保存最佳模型")
print("-" * 30)

best_accuracy = 0.0
best_model_state = None

# 这里可以在训练循环中实现最佳模型保存
torch.save(net.state_dict(), 'cnn_cifar10.pth')
print("模型已保存为 'cnn_cifar10.pth'")

print("\n=== 图像分类实战教程 4 完成 ===")

# 10. 使用预训练模型
print("\n10. 预训练模型示例（概念）")
print("-" * 40)

print("使用预训练模型的步骤：")
print("1. 加载预训练模型（如 ResNet, VGG 等）")
print("2. 冻结预训练层的参数")
print("3. 替换最后的分类层")
print("4. 只训练新的分类层")
print("5. 可选：解冻部分层进行微调")
