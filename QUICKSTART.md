# 快速开始指南 ⚡

这是最快速的项目上手指南，3分钟即可开始学习！

## 🚀 一键开始

### 方法1：自动化脚本（最简单）

```bash
# 下载或进入项目目录
cd learn_python

# 运行自动初始化脚本
python init_project.py
```

脚本会自动：
- ✅ 检查Python版本
- ✅ 检查pip工具
- ✅ 安装所有依赖
- ✅ 验证安装结果
- ✅ 运行示例程序

### 方法2：手动安装（传统方式）

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python 1_quickstart.py
```

## 📖 开始学习

### 如果你是Java程序员

从这里开始：

```bash
# 1. 查看学习指南
python 1_INDEX_LEARNING_GUIDE.py

# 2. 学习基础语法对比
python 1_basic_syntax_comparison.py

# 3. 学习类型系统
python 1_type_system_comparison.py

# 4. 学习面向对象
python 1_oop_comparison.py
```

### 如果你想学习AI技术

直接跳到感兴趣的主题：

```bash
# PyTorch基础
python 6_pytorch_1_basics.py

# 强化学习
python 12_rl_0.py

# RAG系统
python 19_rag_quick_demo.py

# AI编程助手
python 23_ai_coding_cursor.py
```

## 🎯 5分钟体验核心功能

### 1. Python基础（1分钟）
```bash
python 1_quickstart.py
```

### 2. PyTorch入门（1分钟）
```bash
python 6_pytorch_1_basics.py
```

### 3. 强化学习（2分钟）
```bash
python 12_rl_0.py
```

### 4. 智能体演示（1分钟）
```bash
python 17_simple_react_demo.py
```

## 📚 完整学习路径

查看详细的学习路径：

1. **完整文档**: [README.md](README.md)
2. **环境搭建**: [PROJECT_SETUP_GUIDE.md](PROJECT_SETUP_GUIDE.md)
3. **项目结构**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🛠️ 常用命令（Makefile）

如果系统支持`make`：

```bash
make help          # 查看所有命令
make install-dev   # 安装开发依赖
make test          # 运行测试
make format        # 格式化代码
```

## ❓ 遇到问题？

### 安装失败
```bash
# 升级pip
pip install --upgrade pip

# 重新安装
pip install -r requirements.txt
```

### PyTorch安装慢
访问 https://pytorch.org/ 选择合适的版本

### 其他问题
查看 [PROJECT_SETUP_GUIDE.md](PROJECT_SETUP_GUIDE.md) 的常见问题部分

## 🎉 准备就绪！

环境搭建完成后，选择你感兴趣的模块开始学习吧！

**祝学习愉快！** 🚀
