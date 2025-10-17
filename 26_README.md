# Agent MCP (Model Context Protocol) 完整实现

<div align="center">

**🚀 为 AI Agent 提供标准化的外部资源和工具访问能力 🚀**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-4%2F4%20Passing-brightgreen.svg)](26_test_mcp.py)

</div>

---

## 📖 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [使用示例](#使用示例)
- [文档](#文档)
- [测试](#测试)
- [贡献](#贡献)

## 🎯 简介

**MCP (Model Context Protocol)** 是由 Anthropic 推出的开放标准协议，旨在标准化 AI 应用与外部数据源、工具之间的集成方式。

本项目提供了 MCP 协议的**完整 Python 实现**，包括：

- ✅ 完整的协议实现（基于 JSON-RPC 2.0）
- ✅ 易用的 Server 和 Client SDK
- ✅ 丰富的示例应用
- ✅ 详尽的文档和教程
- ✅ 完善的测试用例

### 为什么选择 MCP？

| 传统方法 | MCP 方法 |
|---------|---------|
| ❌ 每个工具独立集成 | ✅ 统一协议接口 |
| ❌ 维护成本高 | ✅ 一次实现，处处复用 |
| ❌ 紧耦合 | ✅ 松耦合架构 |
| ❌ 难以扩展 | ✅ 插件式扩展 |

## ⚡ 快速开始

### 1. 运行演示

```bash
# 进入项目目录
cd /Users/yefei.yf/Qoder/learn_python

# 运行完整演示
python 26_mcp_demo.py
```

### 2. 运行测试

```bash
# 运行所有测试
python 26_test_mcp.py
```

### 3. 简单示例

```python
# 导入 MCP 核心模块
mcp = __import__('26_mcp_core')

# 创建 Server
server = mcp.MCPServer(name="My Server", version="1.0.0")

# 注册一个工具
def greet(name: str) -> str:
    return f"你好, {name}!"

server.register_tool(mcp.Tool(
    name="greet",
    description="问候工具",
    category=mcp.ToolCategory.CUSTOM,
    input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
    function=greet
))

# 创建 Client 并连接
client = mcp.MCPClient()
client.connect("myserver", server)

# 调用工具
result = client.call_tool("myserver", "greet", {"name": "世界"})
print(result)  # {"content": "你好, 世界!"}
```

## 🌟 核心特性

### 三大核心组件

#### 1. 📦 Resource（资源）
提供只读的上下文数据

```python
resource = Resource(
    uri="file:///docs/readme.md",
    name="README",
    resource_type=ResourceType.FILE,
    description="项目说明文档"
)
```

#### 2. 🔧 Tool（工具）
可调用的可执行功能

```python
tool = Tool(
    name="calculator",
    description="数学计算工具",
    category=ToolCategory.COMPUTATION,
    function=calculator_func
)
```

#### 3. 📝 Prompt（提示词）
可复用的提示词模板

```python
prompt = Prompt(
    name="code_review",
    template="请审查以下 {language} 代码：\n{code}",
    arguments=[...]
)
```

### 核心功能

- ✅ **标准化协议**：基于 JSON-RPC 2.0
- ✅ **灵活架构**：Server-Client 解耦
- ✅ **易于扩展**：插件式工具注册
- ✅ **完善文档**：详细的使用指南
- ✅ **实际应用**：文件系统、数据分析示例
- ✅ **Agent 集成**：与 AI Agent 无缝集成

## 📁 项目结构

```
26_mcp_*/
├── 核心代码
│   ├── 26_mcp_core.py          # MCP 核心实现 (512 行)
│   └── 26_mcp_demo.py          # 完整应用演示 (650 行)
│
├── 文档
│   ├── 26_README.md            # 项目主文档（本文件）
│   ├── 26_README_MCP.md        # MCP 概览介绍
│   ├── 26_QUICKSTART.md        # 快速入门指南
│   ├── 26_MCP_SUMMARY.md       # 完整技术总结
│   ├── 26_INDEX.md             # 学习路径索引
│   ├── 26_architecture.md      # 架构可视化
│   └── 26_PROJECT_COMPLETE.md  # 项目完成总结
│
└── 测试
    └── 26_test_mcp.py          # 功能测试脚本
```

### 文件说明

| 文件 | 类型 | 说明 |
|------|------|------|
| `26_mcp_core.py` | 代码 | MCP 核心实现，包含所有基础类 |
| `26_mcp_demo.py` | 代码 | 完整演示，包含实际应用场景 |
| `26_test_mcp.py` | 测试 | 自动化测试脚本 |
| `26_README_MCP.md` | 文档 | MCP 概念和架构说明 |
| `26_QUICKSTART.md` | 文档 | 5分钟快速上手教程 |
| `26_MCP_SUMMARY.md` | 文档 | 深入的技术总结 |

## 💻 使用示例

### 示例 1: 创建简单的 MCP Server

```python
from mcp_core_26 import MCPServer, Resource, Tool, ToolCategory, ResourceType

class FileSystemServer(MCPServer):
    def __init__(self):
        super().__init__(name="FileSystem", version="1.0.0")
        
        # 注册资源
        self.register_resource(Resource(
            uri="file:///docs/readme.md",
            name="README",
            resource_type=ResourceType.FILE
        ))
        
        # 注册工具
        def search(keyword: str):
            return f"搜索结果: {keyword}"
        
        self.register_tool(Tool(
            name="search",
            description="搜索工具",
            category=ToolCategory.SEARCH,
            function=search
        ))
```

### 示例 2: AI Agent 集成

```python
class MCPEnabledAgent:
    def __init__(self, name: str):
        self.name = name
        self.client = MCPClient()
    
    def connect_to_server(self, server_name: str, server: MCPServer):
        self.client.connect(server_name, server)
    
    def process_query(self, query: str) -> str:
        # 使用 MCP 工具处理查询
        if "搜索" in query:
            result = self.client.call_tool(
                "filesystem",
                "search",
                {"keyword": query}
            )
            return result["content"]
        
        return "无法处理该查询"
```

## 📚 文档

### 入门文档

1. **[MCP 概览](26_README_MCP.md)** - 了解 MCP 是什么
2. **[快速开始](26_QUICKSTART.md)** - 5分钟快速上手
3. **[学习索引](26_INDEX.md)** - 完整的学习路径

### 深入学习

4. **[技术总结](26_MCP_SUMMARY.md)** - 深入的技术细节
5. **[架构设计](26_architecture.md)** - 可视化架构图
6. **[项目总结](26_PROJECT_COMPLETE.md)** - 项目完成情况

### 代码示例

- `26_mcp_demo.py` - 完整的应用示例
- `26_test_mcp.py` - 测试用例参考

## 🧪 测试

本项目包含完整的测试套件：

```bash
# 运行所有测试
python 26_test_mcp.py
```

### 测试覆盖

- ✅ 核心组件测试（Resource, Tool, Prompt）
- ✅ MCP Server 功能测试
- ✅ MCP Client 功能测试
- ✅ JSON-RPC 协议测试

### 测试结果

```
✅ 通过  核心组件
✅ 通过  MCP Server
✅ 通过  JSON-RPC
✅ 通过  MCP Client

总计: 4/4 测试通过
```

## 🎓 学习路径

### 初学者（1天）

1. 阅读 `26_README_MCP.md` 了解基础概念
2. 运行 `python 26_mcp_demo.py` 体验功能
3. 阅读 `26_QUICKSTART.md` 学习使用方法

### 进阶（3天）

1. 深入学习 `26_mcp_core.py` 核心实现
2. 研究 `26_mcp_demo.py` 实际应用
3. 阅读 `26_MCP_SUMMARY.md` 技术总结

### 精通（1周）

1. 创建自定义 MCP Server
2. 集成到实际 AI Agent 项目
3. 性能优化和安全加固

## 🚀 实际应用场景

### 1. 知识库问答
- Resources: 文档库、Wiki
- Tools: 语义搜索、摘要生成
- Prompts: QA 模板

### 2. 代码助手
- Resources: 代码仓库、API 文档
- Tools: 代码分析、测试生成
- Prompts: 代码审查模板

### 3. 数据分析
- Resources: 数据库、数据集
- Tools: 统计计算、可视化
- Prompts: 分析报告模板

## 🤝 贡献

欢迎贡献代码、文档或建议！

### 如何贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关链接

- [MCP 官方网站](https://modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [Anthropic 博客](https://www.anthropic.com/news/model-context-protocol)

## 📞 联系方式

- 项目维护者：基于 Anthropic MCP 规范实现
- 问题反馈：通过 Issues 提交

## 🙏 致谢

- Anthropic 团队提出 MCP 协议
- Python 开源社区
- 所有贡献者和使用者

---

<div align="center">

**开始你的 MCP 之旅！** 🎉

[查看文档](26_INDEX.md) · [快速开始](26_QUICKSTART.md) · [运行演示](26_mcp_demo.py)

</div>
