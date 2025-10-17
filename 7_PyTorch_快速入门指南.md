# PyTorch 快速入门指南

## 📚 学习路线图

本系列文档旨在帮助你快速掌握 PyTorch 深度学习框架。建议按照以下顺序学习：

```
7_PyTorch_快速入门指南.md (本文档)
    ↓
7_PyTorch_架构与核心原理.md
    ↓
7_张量操作详解.md
    ↓
7_自动微分机制.md
    ↓
7_神经网络构建.md
    ↓
7_训练与优化技巧.md
    ↓
7_实战案例_图像分类.md
```

---

## 🎯 学习目标

完成本系列学习后，你将能够：

- ✅ 理解 PyTorch 的核心架构和设计理念
- ✅ 熟练使用张量（Tensor）进行各种操作
- ✅ 掌握自动梯度（Autograd）机制
- ✅ 构建自己的神经网络模型
- ✅ 训练和优化深度学习模型
- ✅ 完成实际的图像分类项目

---

## 🔧 环境准备

### 1. 安装 PyTorch

访问 [PyTorch 官网](https://pytorch.org/) 选择适合你系统的版本：

```bash
# CPU 版本（快速开始）
pip install torch torchvision torchaudio

# GPU 版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 验证安装

```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 创建一个简单的张量
x = torch.tensor([1, 2, 3])
print(f"张量: {x}")
```

### 3. 推荐的学习环境

- **IDE**: VSCode, PyCharm, Jupyter Notebook
- **Python 版本**: 3.8+
- **建议配置**: 8GB+ RAM，有 GPU 更佳

---

## 📖 快速示例

### Hello PyTorch！

```python
import torch
import torch.nn as nn

# 1. 创建数据
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 定义模型
model = nn.Linear(1, 1)  # 线性模型: y = wx + b

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(100):
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 5. 测试
with torch.no_grad():
    test_x = torch.tensor([[5.0]])
    test_y = model(test_x)
    print(f'输入: 5.0, 预测: {test_y.item():.4f}')
```

---

## 🎓 核心概念预览

### 1. 张量（Tensor）

PyTorch 的基本数据结构，类似于 NumPy 的数组，但支持 GPU 加速：

```python
# 创建张量的多种方式
a = torch.tensor([1, 2, 3])
b = torch.zeros(3, 4)
c = torch.randn(2, 3)
```

### 2. 自动微分（Autograd）

自动计算梯度，是深度学习的核心：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # 自动计算梯度
print(x.grad)  # 输出: tensor([4.])
```

### 3. 神经网络模块（nn.Module）

构建网络的基础类：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)
```

---

## 🗂️ 文档内容概览

### [7_PyTorch_架构与核心原理.md](./7_PyTorch_架构与核心原理.md)
- PyTorch 的整体架构
- 核心组件详解
- 动态计算图原理
- 与其他框架的对比

### [7_张量操作详解.md](./7_张量操作详解.md)
- 张量的创建和属性
- 索引、切片、变形
- 数学运算
- 与 NumPy 的互操作

### [7_自动微分机制.md](./7_自动微分机制.md)
- 自动梯度的工作原理
- 计算图的构建和反向传播
- 梯度管理技巧
- 实战：手写梯度下降

### [7_神经网络构建.md](./7_神经网络构建.md)
- nn.Module 详解
- 常用层介绍
- 自定义层和模型
- 模型的保存与加载

### [7_训练与优化技巧.md](./7_训练与优化技巧.md)
- 完整的训练流程
- 优化器选择
- 学习率调度
- 正则化技术

### [7_实战案例_图像分类.md](./7_实战案例_图像分类.md)
- 数据准备和预处理
- CNN 模型构建
- 训练和评估
- 模型优化技巧

---

## 💡 学习建议

### 对于初学者

1. **循序渐进**：按照文档顺序学习，不要跳过基础部分
2. **动手实践**：每个示例都要自己运行一遍
3. **理解原理**：不要只记住 API，要理解背后的原理
4. **做笔记**：记录重要的概念和容易犯的错误

### 对于有经验的开发者

1. **重点关注**：PyTorch 的动态计算图和 Autograd 机制
2. **对比学习**：如果熟悉 TensorFlow，注意两者的差异
3. **深入源码**：理解关键组件的实现
4. **实战项目**：尽快开始自己的项目

---

## 📚 推荐资源

### 官方资源
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 论坛](https://discuss.pytorch.org/)

### 学习资源
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch 官方示例](https://github.com/pytorch/examples)
- [Papers with Code](https://paperswithcode.com/) - 论文和代码实现

### 实战项目
- MNIST 手写数字识别
- CIFAR-10 图像分类
- 迁移学习
- GANs 生成对抗网络

---

## 🎯 学习检查清单

在学习每个主题后，检查是否达到以下目标：

- [ ] **张量操作**
  - [ ] 能够创建各种类型的张量
  - [ ] 熟练使用索引和切片
  - [ ] 理解张量的内存模型

- [ ] **自动微分**
  - [ ] 理解计算图的概念
  - [ ] 能够手动进行梯度检查
  - [ ] 知道何时使用 `no_grad()`

- [ ] **神经网络**
  - [ ] 能够定义自己的模型
  - [ ] 理解前向传播和反向传播
  - [ ] 熟悉常用层的使用

- [ ] **训练流程**
  - [ ] 理解完整的训练循环
  - [ ] 能够选择合适的优化器
  - [ ] 掌握基本的调试技巧

---

## ⚠️ 常见问题

### Q1: PyTorch vs TensorFlow？
**A**: PyTorch 更适合研究和快速原型开发（动态图），TensorFlow 在生产部署上更成熟（静态图优化）。目前 TensorFlow 2.x 也支持动态图（Eager Execution）。

### Q2: 需要 GPU 吗？
**A**: 学习阶段不是必须的，但训练大模型时 GPU 能显著加速。可以使用 Google Colab 免费获得 GPU 资源。

### Q3: 如何调试 PyTorch 代码？
**A**: 
- 使用 `print()` 打印张量形状
- 使用 `pdb` 或 IDE 断点
- 检查梯度是否正常：`torch.autograd.gradcheck()`
- 使用 TensorBoard 可视化

### Q4: 模型训练很慢？
**A**:
- 使用 GPU（`.to('cuda')`）
- 增加 batch size
- 使用 DataLoader 的多进程（`num_workers`）
- 使用混合精度训练（`torch.cuda.amp`）

---

## 🚀 开始学习

准备好了吗？让我们从 [PyTorch 架构与核心原理](./7_PyTorch_架构与核心原理.md) 开始！

记住：**实践是最好的老师**。不要只是阅读，要动手写代码！

---

*最后更新: 2025-10-17*
