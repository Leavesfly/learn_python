# pytorch_1_basics.py - PyTorch 基础：张量操作
import torch
import numpy as np

print("=== PyTorch 基础教程 1：张量操作 ===\n")

# 1. 创建张量
print("1. 创建张量")
print("-" * 30)

# 从 Python 列表创建
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"从列表创建张量: \n{x_data}")

# 从 NumPy 数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"从 NumPy 数组创建张量: \n{x_np}")

# 创建特殊张量
x_ones = torch.ones_like(x_data)  # 保持 x_data 的形状
print(f"全 1 张量: \n{x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖数据类型
print(f"随机张量: \n{x_rand}")

# 指定形状创建张量
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"随机张量 (2x3): \n{rand_tensor}")
print(f"全 1 张量 (2x3): \n{ones_tensor}")
print(f"全 0 张量 (2x3): \n{zeros_tensor}")

# 2. 张量属性
print("\n2. 张量属性")
print("-" * 30)

tensor = torch.rand(3, 4)
print(f"张量形状: {tensor.shape}")
print(f"数据类型: {tensor.dtype}")
print(f"存储设备: {tensor.device}")

# 3. 张量操作
print("\n3. 张量操作")
print("-" * 30)

# 索引和切片
tensor = torch.ones(4, 4)
print(f"原始张量: \n{tensor}")
print(f"第一行: {tensor[0]}")
print(f"第一列: {tensor[:, 0]}")
print(f"最后一列: {tensor[..., -1]}")

# 修改张量
tensor[:, 1] = 0
print(f"修改第二列后: \n{tensor}")

# 连接张量
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"在维度1上连接张量: \n{t1}")

# 算术操作
y1 = tensor @ tensor.T  # 矩阵乘法
y2 = tensor.matmul(tensor.T)  # 矩阵乘法的另一种写法
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  # 将结果写入现有张量

print(f"矩阵乘法结果形状: {y1.shape}")

# 元素级操作
z1 = tensor * tensor  # 元素级乘法
z2 = tensor.mul(tensor)  # 元素级乘法的另一种写法
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)  # 将结果写入现有张量

print(f"元素级乘法: \n{z1}")

# 单元素张量聚合
agg = tensor.sum()
agg_item = agg.item()  # 将单元素张量转换为 Python 数值
print(f"所有元素的和: {agg_item}, 类型: {type(agg_item)}")

# 就地操作
print(f"原张量: \n{tensor}")
tensor.add_(5)  # 就地操作，会修改原张量
print(f"就地加 5 后: \n{tensor}")

# 4. 与 NumPy 的互操作
print("\n4. 与 NumPy 的互操作")
print("-" * 30)

# 张量转 NumPy
t = torch.ones(5)
print(f"PyTorch 张量: {t}")
n = t.numpy()
print(f"NumPy 数组: {n}")

# 修改张量会同时修改 NumPy 数组
t.add_(1)
print(f"修改张量后: {t}")
print(f"NumPy 数组也被修改: {n}")

# NumPy 转张量
n = np.ones(5)
t = torch.from_numpy(n)
print(f"NumPy 数组: {n}")
print(f"PyTorch 张量: {t}")

# 修改 NumPy 数组会同时修改张量
np.add(n, 1, out=n)
print(f"修改 NumPy 数组后: {n}")
print(f"张量也被修改: {t}")

print("\n=== 基础教程 1 完成 ===")