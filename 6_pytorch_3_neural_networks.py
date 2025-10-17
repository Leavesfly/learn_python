# pytorch_3_neural_networks.py - PyTorch 神经网络构建
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

print("=== PyTorch 教程 3：神经网络构建 ===\n")

# 1. 定义神经网络
print("1. 定义神经网络")
print("-" * 30)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道1，输出通道6，卷积核3x3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 是经过卷积和池化后的图像尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 池化层使用 2x2 的窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了 batch 维度之外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 打印网络参数
params = list(net.parameters())
print(f"\n参数数量: {len(params)}")
print(f"conv1 权重大小: {params[0].size()}")

# 2. 前向传播测试
print("\n2. 前向传播测试")
print("-" * 30)

input_tensor = torch.randn(1, 1, 32, 32)  # batch_size=1, channels=1, height=32, width=32
out = net(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {out.shape}")
print(f"输出: {out}")

# 清零梯度缓存，然后反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 3. 损失函数
print("\n3. 损失函数")
print("-" * 30)

output = net(input_tensor)
target = torch.randn(10)  # 模拟真实标签
target = target.view(1, -1)  # 使其与输出形状相同

criterion = nn.MSELoss()
loss = criterion(output, target)
print(f"MSE 损失: {loss}")

print(f"loss.grad_fn: {loss.grad_fn}")  # MSELoss
print(f"loss.grad_fn.next_functions[0][0]: {loss.grad_fn.next_functions[0][0]}")  # Linear
print(f"loss.grad_fn.next_functions[0][0].next_functions[0][0]: {loss.grad_fn.next_functions[0][0].next_functions[0][0]}")  # ReLU

# 4. 反向传播
print("\n4. 反向传播")
print("-" * 30)

net.zero_grad()  # 清零所有参数的梯度缓存

print('conv1.bias.grad 反向传播前:')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad 反向传播后:')
print(net.conv1.bias.grad)

# 5. 更新权重
print("\n5. 更新权重")
print("-" * 30)

# 简单的权重更新规则（随机梯度下降）
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 使用 torch.optim 包进行权重更新
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练循环中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input_tensor)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 进行权重更新

print("权重更新完成")

# 6. 完整的训练示例 - 简单的分类任务
print("\n6. 完整的训练示例")
print("-" * 40)

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 生成模拟数据
torch.manual_seed(42)
n_samples = 1000
input_size = 20
hidden_size = 64
num_classes = 3

X = torch.randn(n_samples, input_size)
y = torch.randint(0, num_classes, (n_samples,))

# 创建数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
n_epochs = 50
for epoch in range(n_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    if (epoch + 1) % 10 == 0:
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 7. 模型保存和加载
print("\n7. 模型保存和加载")
print("-" * 30)

# 保存整个模型
torch.save(model, 'model.pth')

# 只保存模型参数（推荐）
torch.save(model.state_dict(), 'model_params.pth')

# 加载模型参数
model2 = SimpleNet(input_size, hidden_size, num_classes)
model2.load_state_dict(torch.load('model_params.pth'))
model2.eval()  # 设置为评估模式

print("模型保存和加载完成")

print("\n=== 神经网络教程 3 完成 ===")