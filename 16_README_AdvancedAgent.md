# 高级LLM Agent系统

## 概述

这是一个从零构建的基于LLM的智能Agent系统，实现了完整的上下文工程功能，包括长短期记忆管理、RAG（检索增强生成）、工具调用等高级特性。

## 🌟 核心特性

### 1. 多层记忆系统
- **工作记忆（Working Memory）**: 短期记忆，容量有限（默认10条），用于存储当前对话中的临时信息
- **情节记忆（Episodic Memory）**: 存储特定事件和经历，带有时间戳和重要性评分
- **语义记忆（Semantic Memory）**: 存储一般知识和概念，用于长期知识保持

### 2. RAG检索增强生成系统
- **文档索引**: 支持添加和索引文档内容
- **语义检索**: 基于TF-IDF的简单嵌入模型进行相似度检索
- **上下文生成**: 自动为查询生成相关的上下文信息

### 3. 工具调用系统
- **工具注册**: 动态注册和管理工具函数
- **智能解析**: 从文本中解析工具调用指令
- **结果集成**: 将工具执行结果集成到响应中

### 4. 上下文工程
- **智能压缩**: 自动压缩对话历史以适应上下文长度限制
- **优先级管理**: 优先保留重要信息（系统指令、当前查询等）
- **多源融合**: 整合记忆、RAG、工具信息生成完整上下文

## 🏗️ 系统架构

```
AdvancedAgent
├── MemoryManager        # 记忆管理系统
│   ├── WorkingMemory    # 工作记忆
│   ├── EpisodicMemory   # 情节记忆
│   └── SemanticMemory   # 语义记忆
├── RAGSystem           # 检索增强生成
│   ├── DocumentStore   # 文档存储
│   ├── EmbeddingModel  # 嵌入模型
│   └── Retriever       # 检索器
├── ToolRegistry        # 工具注册表
│   ├── ToolManager     # 工具管理
│   └── CallParser      # 调用解析
└── ContextEngine       # 上下文引擎
    ├── HistoryManager  # 历史管理
    ├── Compressor      # 内容压缩
    └── Builder         # 上下文构建
```

## 🚀 快速开始

### 基本使用

```python
from 16_llm_agent_advanced import AdvancedAgent

# 创建Agent实例
agent = AdvancedAgent(
    name="我的助手",
    system_prompt="你是一个智能助手，可以帮助用户解决各种问题。"
)

# 添加知识
agent.add_knowledge("Python是一种编程语言", "python_info")

# 与Agent对话
response = agent.process_message("什么是Python？")
print(response)
```

### 添加自定义工具

```python
def my_custom_tool(param1: str, param2: int) -> dict:
    """自定义工具函数"""
    return {"result": f"处理了 {param1} 和 {param2}"}

# 注册工具
agent.register_tool(
    name="my_tool",
    func=my_custom_tool,
    description="这是我的自定义工具",
    parameters={
        "param1": {"type": "string", "description": "字符串参数"},
        "param2": {"type": "integer", "description": "整数参数"}
    }
)
```

## 🔧 内置工具

### 1. 计算器工具
```python
# 使用方式：在对话中说 "计算 10 + 5"
agent.process_message("帮我计算 15 * 3")
```

### 2. 时间工具
```python
# 使用方式：在对话中询问时间
agent.process_message("现在几点了？")
```

### 3. 笔记工具
```python
# 创建笔记
agent.process_message('创建笔记 "学习Python基础"')

# 查看笔记
agent.process_message("查看笔记")
```

## 📊 记忆管理

### 记忆类型和用途

| 记忆类型 | 容量限制 | 持久性 | 用途 |
|---------|---------|--------|------|
| 工作记忆 | 10条 | 临时 | 当前对话上下文 |
| 情节记忆 | 无限制 | 永久 | 重要事件和经历 |
| 语义记忆 | 无限制 | 永久 | 知识和概念 |

### 记忆检索算法

1. **关键词匹配**: 基于词汇重叠度计算相关性
2. **重要性排序**: 按记忆重要性和访问频率排序
3. **访问统计**: 自动更新记忆访问次数和时间

## 🔍 RAG系统详解

### 文档处理流程

1. **文档添加**: 将文档内容添加到系统
2. **嵌入计算**: 使用TF-IDF计算文档嵌入向量
3. **索引构建**: 建立文档索引以支持快速检索
4. **相似度计算**: 基于余弦相似度进行文档匹配

### 检索策略

- **Top-K检索**: 返回相似度最高的K个文档
- **阈值过滤**: 可设置相似度阈值过滤低质量结果
- **上下文长度控制**: 自动截断过长内容以适应上下文限制

## ⚙️ 上下文工程

### 上下文构建流程

1. **系统提示**: 设置Agent基本行为和能力
2. **工具信息**: 列出可用工具及其描述
3. **相关记忆**: 检索与当前查询相关的记忆
4. **RAG上下文**: 从知识库检索相关文档
5. **对话历史**: 压缩后的最近对话记录
6. **当前查询**: 用户的当前问题

### 压缩策略

- **历史压缩**: 保留最近6条消息，长内容截断到200字符
- **优先级保留**: 优先保留系统指令和当前查询
- **智能截断**: 按重要性截断中间内容，保持上下文连贯性

## 🎯 工具调用机制

### 工具调用格式

在对话中使用特殊格式触发工具调用：
```
[tool:工具名(参数1="值1", 参数2=值2)]
```

### 支持的参数类型

- **字符串**: 用引号包围
- **数字**: 整数或浮点数
- **布尔值**: true/false

### 错误处理

- 工具不存在时返回错误信息
- 参数解析失败时提供详细错误描述
- 工具执行异常时捕获并报告错误

## 📈 性能优化

### 记忆管理优化

- **定期整合**: 每10轮对话自动整合重要记忆
- **访问统计**: 跟踪记忆访问频率，优化检索性能
- **重要性衰减**: 随时间降低记忆重要性

### 上下文优化

- **长度控制**: 智能截断超长上下文
- **信息密度**: 优先保留信息密度高的内容
- **缓存机制**: 缓存频繁访问的上下文片段

## 🔧 配置选项

### Agent配置

```python
agent = AdvancedAgent(
    name="助手名称",
    system_prompt="系统提示词",
    max_context_length=4000  # 最大上下文长度
)
```

### 记忆配置

```python
# 自定义记忆数据库
memory_manager = MemoryManager(db_path="memory.db")

# 设置工作记忆容量
memory_manager.working_memory = deque(maxlen=20)
```

### RAG配置

```python
# 自定义嵌入维度
rag_system = RAGSystem(embedding_dim=256)
```

## 🧪 运行演示

### 系统组件演示

```bash
python 16_llm_agent_advanced.py
# 选择选项 2
```

### 交互式Agent演示

```bash
python 16_llm_agent_advanced.py
# 选择选项 1
```

## 📋 示例对话

```
👤 你: 什么是人工智能？

🤖 高级助手: 根据我的知识，人工智能（AI）是由人类开发的智能系统，能够执行通常需要人类智能的任务。它是一个广泛的概念，包含了机器学习、深度学习等多个分支技术。

👤 你: 计算 25 * 4

🤖 高级助手: 我来帮你计算。[tool:calculator(operation="multiply", a=25, b=4)]

工具 calculator 执行结果: {'result': 100.0, 'expression': '25.0 multiply 4.0 = 100.0'}

👤 你: 创建笔记 "AI学习计划"

🤖 高级助手: 我来为你创建笔记。[tool:note(action="create", content="AI学习计划")]

工具 note 执行结果: {'message': '已创建笔记 1', 'note': {'id': '1', 'content': 'AI学习计划', 'created_at': '2024-01-01T12:00:00.000000'}}
```

## 🔮 扩展性

### 添加新的记忆类型

```python
class CustomMemory:
    def __init__(self):
        self.custom_memories = []
    
    def add_custom_memory(self, content, metadata):
        # 自定义记忆逻辑
        pass
```

### 集成外部LLM

```python
def call_external_llm(context: str) -> str:
    """集成OpenAI GPT或其他LLM API"""
    # 实现外部LLM调用
    pass

# 替换Agent中的_simulate_llm_response方法
agent._simulate_llm_response = call_external_llm
```

### 添加新的工具类型

```python
def web_search_tool(query: str) -> dict:
    """网络搜索工具"""
    # 实现网络搜索功能
    return {"results": "搜索结果"}

agent.register_tool("web_search", web_search_tool, "网络搜索")
```

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者！

---

*这个Agent系统演示了现代LLM应用中的关键技术，包括记忆管理、知识检索、工具集成等。它为构建更智能的AI助手提供了一个完整的框架。*