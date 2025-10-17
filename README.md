# AI智能体技术学习项目 🤖

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个系统性的AI智能体技术教学与实践项目，从基础到高级，涵盖强化学习、大模型、多智能体系统、RAG等前沿技术。

## 📚 项目概述

本项目是为Java程序员学习Python和AI智能体技术而设计的完整教程，通过递进式的代码示例和详细的文档说明，帮助学习者构建从基础到高级的知识体系。

### 核心技术模块

- **🔄 强化学习 (RL)**: Q-Learning、DQN等算法实现
- **🧠 混合专家模型 (MoE)**: Mixture of Experts架构
- **🤖 大模型模拟**: GPT系列、DeepSeek、Qwen3模型
- **👥 多智能体系统**: 协作、竞争、通信机制
- **🔧 智能体模式**: ReAct、自进化、工具使用
- **📖 RAG系统**: 检索增强生成、向量数据库
- **💻 AI编程助手**: Cursor风格的智能编程系统
- **🔬 深度研究**: DeepResearch智能体
- **🏗️ MCP架构**: 模块化认知管道设计
- **🦾 具身智能**: 环境感知、决策执行系统

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

1. **克隆项目**（如果使用Git）
```bash
git clone <repository-url>
cd learn_python
```

2. **创建虚拟环境**（推荐）
```bash
python -m venv venv

# macOS/Linux激活
source venv/bin/activate

# Windows激活
venv\Scripts\activate
```

3. **安装依赖**
```bash
# 安装基础依赖
pip install -r requirements.txt

# 或使用现代化安装方式
pip install -e .

# 安装开发工具（可选）
pip install -e ".[dev]"

# 安装Jupyter支持（可选）
pip install -e ".[jupyter]"
```

### 验证安装

```bash
# 运行第一个示例
python 1_quickstart.py

# 运行强化学习示例
python 12_rl_0.py

# 运行PyTorch基础教程
python 6_pytorch_1_basics.py
```

## 📖 学习路径

### 阶段1：Python基础（Java转Python）

从`1_`开头的文件开始，系统学习Python语法和特性：

- [`1_INDEX_LEARNING_GUIDE.py`](1_INDEX_LEARNING_GUIDE.py) - 学习指南
- [`1_basic_syntax_comparison.py`](1_basic_syntax_comparison.py) - 基础语法对比
- [`1_type_system_comparison.py`](1_type_system_comparison.py) - 类型系统对比
- [`1_oop_comparison.py`](1_oop_comparison.py) - 面向对象编程
- [`1_collections_comparison.py`](1_collections_comparison.py) - 集合类型
- [`1_exception_handling.py`](1_exception_handling.py) - 异常处理
- [`1_stdlib_comparison.py`](1_stdlib_comparison.py) - 标准库

### 阶段2：PyTorch深度学习基础

从`6_`和`7_`开头的文件，学习PyTorch框架：

- [`6_pytorch_1_basics.py`](6_pytorch_1_basics.py) - 张量操作
- [`6_pytorch_2_autograd.py`](6_pytorch_2_autograd.py) - 自动微分
- [`6_pytorch_3_neural_networks.py`](6_pytorch_3_neural_networks.py) - 神经网络构建
- [`6_pytorch_4_image_classification.py`](6_pytorch_4_image_classification.py) - 图像分类实战

### 阶段3：强化学习

从`12_rl_`系列文件学习强化学习算法：

- [`12_rl_0.py`](12_rl_0.py) - RL基础概念
- [`12_rl_1.py`](12_rl_1.py) - Q-Learning算法
- [`12_rl_2.py`](12_rl_2.py) - SARSA算法
- [`12_rl_3.py`](12_rl_3.py) - DQN深度强化学习

### 阶段4：大模型与智能体

学习GPT、MoE、多智能体等高级主题：

- [`13_moe_implementation.py`](13_moe_implementation.py) - 混合专家模型
- [`14_gpt*.py`](14_gpt1.py) - GPT系列模型实现
- [`15_multi_agent_system.py`](15_multi_agent_system.py) - 多智能体系统
- [`17_agent_patterns.py`](17_agent_patterns.py) - ReAct模式
- [`18_self_evolving_agent.py`](18_self_evolving_agent.py) - 自进化智能体

### 阶段5：应用系统

实战级的AI应用系统：

- [`19_rag_vector_demo.py`](19_rag_vector_demo.py) - RAG检索增强生成
- [`22_openmanus_agent_system.py`](22_openmanus_agent_system.py) - OpenManus系统
- [`23_ai_coding_cursor.py`](23_ai_coding_cursor.py) - AI编程助手
- [`24_deep_research_agent.py`](24_deep_research_agent.py) - 深度研究智能体
- [`25_qwen3_model.py`](25_qwen3_model.py) - 通义千问集成

### 阶段6：高级架构

学习MCP架构和具身智能：

- [`26_mcp_core.py`](26_mcp_core.py) - MCP核心实现
- [`27_embodied_robot_demo.py`](27_embodied_robot_demo.py) - 具身智能机器人

## 📁 项目结构

```
learn_python/
├── 1_*.py                    # Python基础教程（Java对比）
├── 6_pytorch_*.py            # PyTorch深度学习教程
├── 12_rl_*.py               # 强化学习系列
├── 13_moe_implementation.py  # 混合专家模型
├── 14_*.py                  # GPT系列和DeepSeek
├── 15_multi_agent_*.py      # 多智能体系统
├── 16_llm_agent_advanced.py # 高级LLM智能体
├── 17_*_react*.py           # ReAct模式智能体
├── 18_*_evolving*.py        # 自进化智能体
├── 19_rag_*.py              # RAG检索增强生成
├── 22_openmanus_*.py        # OpenManus系统
├── 23_ai_coding_cursor.py   # AI编程助手
├── 24_deep_research_*.py    # 深度研究智能体
├── 25_qwen3_*.py            # 通义千问Qwen3
├── 26_mcp_*.py              # MCP架构
├── 27_embodied_*.py         # 具身智能
├── requirements.txt          # 项目依赖
├── setup.py                 # 安装配置（传统）
├── pyproject.toml           # 项目配置（现代）
└── README.md                # 本文件
```

## 🛠️ 开发工具

### 代码格式化

```bash
# 使用Black格式化代码
black .

# 检查代码风格
flake8 .
```

### 类型检查

```bash
# 使用mypy进行类型检查
mypy *.py
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest 19_test_rag.py

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

## 📝 文档说明

每个模块都配有详细的README文档：

- `*_README*.md` - 各模块的详细说明文档
- `*_SUMMARY.md` - 项目总结和进度报告
- `*_INDEX.md` - 索引和快速导航

## 🤝 贡献指南

欢迎贡献代码和改进建议！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

**山泽**

- 项目专注于AI智能体技术教学
- 适合Java背景的开发者学习Python和AI技术

## 🙏 致谢

感谢所有为本项目提供灵感和帮助的开源项目和社区。

## 📮 联系方式

如有问题或建议，欢迎：

- 提交 Issue
- 发送 Pull Request
- 联系作者（请在setup.py中更新邮箱）

---

**Happy Learning! 祝学习愉快！** 🎉
