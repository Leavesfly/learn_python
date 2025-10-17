# 项目环境搭建指南 🚀

本指南将帮助你快速搭建AI智能体技术学习项目的开发环境。

## 📋 前置要求

在开始之前，请确保你的系统已安装：

- **Python 3.8 或更高版本** 
  - 查看版本：`python --version` 或 `python3 --version`
  - 下载地址：https://www.python.org/downloads/

- **pip 包管理器**（通常随Python安装）
  - 查看版本：`pip --version` 或 `pip3 --version`

- **Git**（可选，用于版本控制）
  - 查看版本：`git --version`
  - 下载地址：https://git-scm.com/

## 🔧 环境搭建步骤

### 方式一：使用虚拟环境（推荐）

#### 1. 创建虚拟环境

```bash
# 在项目目录下执行
python -m venv venv
```

#### 2. 激活虚拟环境

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

> 💡 提示：激活后，命令行提示符前会显示 `(venv)`

#### 3. 升级pip（可选但推荐）

```bash
pip install --upgrade pip
```

#### 4. 安装项目依赖

**基础安装：**
```bash
pip install -r requirements.txt
```

**开发模式安装（推荐）：**
```bash
# 以可编辑模式安装项目
pip install -e .

# 或安装包含开发工具
pip install -e ".[dev]"

# 或安装包含Jupyter支持
pip install -e ".[jupyter]"

# 或安装所有可选依赖
pip install -e ".[dev,jupyter]"
```

### 方式二：直接安装（不使用虚拟环境）

```bash
# 直接安装依赖
pip install -r requirements.txt

# 或使用开发模式
pip install -e ".[dev,jupyter]"
```

> ⚠️ 警告：不使用虚拟环境可能导致包冲突，不推荐

## ✅ 验证安装

### 1. 检查核心依赖

```bash
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
```

### 2. 运行测试示例

```bash
# 运行Python快速入门
python 1_quickstart.py

# 运行PyTorch基础教程
python 6_pytorch_1_basics.py

# 运行强化学习示例
python 12_rl_0.py
```

### 3. 运行单元测试

```bash
# 如果安装了pytest
pytest

# 运行特定测试
pytest 19_test_rag.py
```

## 📦 使用Makefile（macOS/Linux）

如果你的系统支持`make`命令，可以使用更简便的方式：

```bash
# 查看所有可用命令
make help

# 安装依赖
make install-dev

# 清理临时文件
make clean

# 运行测试
make test

# 格式化代码
make format

# 代码风格检查
make lint

# 完整检查流程
make all
```

## 🎯 开始学习

安装完成后，建议按照以下顺序学习：

1. **Python基础**（如果你来自Java背景）
   ```bash
   python 1_INDEX_LEARNING_GUIDE.py
   ```

2. **PyTorch入门**
   ```bash
   python 6_pytorch_1_basics.py
   ```

3. **强化学习基础**
   ```bash
   python 12_rl_0.py
   ```

4. **探索高级主题**
   - 查看 README.md 了解完整学习路径
   - 阅读各模块的 *_README*.md 文件

## 🛠️ 开发工具配置

### VS Code 推荐扩展

- Python (Microsoft)
- Pylance
- Python Docstring Generator
- Black Formatter
- Jupyter

### PyCharm 配置

1. 打开项目
2. 配置Python解释器：`Settings → Project → Python Interpreter`
3. 选择虚拟环境中的Python解释器

### Jupyter Notebook

如果安装了Jupyter支持：

```bash
# 启动Jupyter Notebook
jupyter notebook

# 或使用JupyterLab
jupyter lab
```

## ❓ 常见问题

### Q1: PyTorch安装失败

**解决方案：**
访问 https://pytorch.org/ 获取适合你系统的安装命令。

对于CPU版本：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Q2: ImportError: No module named 'xxx'

**解决方案：**
确保虚拟环境已激活，然后重新安装依赖：
```bash
pip install -r requirements.txt
```

### Q3: 权限错误 (Permission denied)

**解决方案：**
- 使用虚拟环境（推荐）
- 或在安装命令后添加 `--user` 参数
  ```bash
  pip install --user -r requirements.txt
  ```

### Q4: Windows PowerShell 无法激活虚拟环境

**解决方案：**
以管理员身份运行PowerShell，执行：
```powershell
Set-ExecutionPolicy RemoteSigned
```

### Q5: matplotlib中文显示乱码

**解决方案：**
参考项目中的配置或在代码中添加：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
# 或
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
```

## 📚 更多资源

- **项目主文档**: [README.md](README.md)
- **Python基础**: [1_README_JAVA_TO_PYTHON.md](1_README_JAVA_TO_PYTHON.md)
- **PyTorch教程**: [7_PyTorch_快速入门指南.md](7_PyTorch_快速入门指南.md)
- **各模块README**: 查看以 `*_README*.md` 命名的文件

## 💡 最佳实践

1. **始终使用虚拟环境** - 避免依赖冲突
2. **定期更新依赖** - `pip list --outdated` 查看可更新的包
3. **保持代码整洁** - 使用 `black` 格式化代码
4. **编写测试** - 确保代码质量
5. **阅读文档** - 每个模块都有详细说明

## 🎉 准备就绪！

环境搭建完成后，你就可以开始AI智能体技术的学习之旅了！

如有任何问题，欢迎：
- 查阅项目文档
- 提交Issue
- 参与讨论

**祝学习愉快！Happy Coding! 🚀**
