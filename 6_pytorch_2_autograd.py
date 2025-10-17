# pytorch_2_autograd.py - PyTorch 自动梯度系统
import torch

print("=== PyTorch 教程 2：自动梯度系统 ===\n")

# 1. 自动梯度基础
print("1. 自动梯度基础")
print("-" * 30)

# 创建需要梯度的张量
x = torch.ones(2, 2, requires_grad=True)
print(f"输入张量 x: \n{x}")

# 执行操作
y = x + 2
print(f"y = x + 2: \n{y}")
print(f"y.grad_fn: {y.grad_fn}")  # 创建 y 的函数

z = y * y * 3
out = z.mean()
print(f"z = y * y * 3: \n{z}")
print(f"out = z.mean(): {out}")

# 计算梯度
out.backward()
print(f"x 的梯度: \n{x.grad}")

# 2. 梯度计算示例
print("\n2. 梯度计算示例")
print("-" * 30)

x = torch.randn(3, requires_grad=True)
y = x * 2

# 如果输出不是标量，需要传入 gradient 参数
while y.data.norm() < 1000:
    y = y * 2

print(f"y: {y}")
print(f"y 的数据范数: {y.data.norm()}")

# 为非标量输出计算梯度
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(f"x 的梯度: {x.grad}")

# 3. 停止梯度跟踪
print("\n3. 停止梯度跟踪")
print("-" * 30)

x = torch.randn(3, requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")
print(f"(x ** 2).requires_grad: {(x ** 2).requires_grad}")

# 使用 .detach() 停止梯度跟踪
with torch.no_grad():
    print(f"在 no_grad 块中 (x ** 2).requires_grad: {(x ** 2).requires_grad}")

# 或者使用 .detach()
y = x.detach()
print(f"x.detach().requires_grad: {y.requires_grad}")

# 4. 线性回归示例
print("\n4. 线性回归梯度下降示例")
print("-" * 40)

# 生成模拟数据
torch.manual_seed(42)
n_samples, n_features = 100, 1
true_w, true_b = 2.5, 1.3

# 生成训练数据
X = torch.randn(n_samples, n_features)
y = true_w * X.squeeze() + true_b + 0.1 * torch.randn(n_samples)

print(f"数据形状 - X: {X.shape}, y: {y.shape}")

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"初始参数 - w: {w.item():.4f}, b: {b.item():.4f}")

# 训练循环
learning_rate = 0.01
n_epochs = 1000

for epoch in range(n_epochs):
    # 前向传播
    y_pred = X.squeeze() * w + b
    
    # 计算损失（均方误差）
    loss = ((y_pred - y) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 参数更新
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\n训练完成!")
print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"学习参数: w = {w.item():.4f}, b = {b.item():.4f}")

# 5. 梯度累积
print("\n5. 梯度累积示例")
print("-" * 30)

x = torch.tensor([1.0], requires_grad=True)

# 第一次前向和反向传播
y1 = x ** 2
y1.backward()
print(f"第一次梯度: {x.grad}")

# 第二次前向和反向传播（梯度会累积）
y2 = x ** 3
y2.backward()
print(f"累积梯度: {x.grad}")

# 清零梯度
x.grad.zero_()
print(f"清零后梯度: {x.grad}")

print("\n=== 自动梯度教程 2 完成 ===")