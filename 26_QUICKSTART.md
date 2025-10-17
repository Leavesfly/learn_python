# MCP (Model Context Protocol) 快速开始

## 🚀 5分钟快速入门

### 步骤 1: 运行演示程序

```bash
cd /Users/yefei.yf/Qoder/learn_python
python 26_mcp_demo.py
```

选择菜单选项：
- `1` - 查看基础 MCP 功能
- `2` - 体验数据分析场景
- `3` - 观察 AI Agent 如何使用 MCP
- `4` - 交互式对话体验
- `5` - 运行所有演示

### 步骤 2: 理解核心概念

#### Resource（资源）- 数据提供者
```python
# 文件资源
resource = Resource(
    uri="file:///docs/readme.md",
    name="README",
    resource_type=ResourceType.FILE,
    description="项目说明文档"
)
```

#### Tool（工具）- 可执行功能
```python
# 搜索工具
tool = Tool(
    name="search_files",
    description="搜索文件内容",
    category=ToolCategory.SEARCH,
    function=search_function
)
```

#### Prompt（提示词）- 模板复用
```python
# 代码审查模板
prompt = Prompt(
    name="code_review",
    template="请审查以下代码：\n{code}\n...",
    arguments=[{"name": "code", "type": "string"}]
)
```

### 步骤 3: 创建你的第一个 MCP Server

```python
from 26_mcp_core import MCPServer, Resource, Tool, ToolCategory, ResourceType

class MyFirstServer(MCPServer):
    def __init__(self):
        super().__init__(name="My First MCP Server", version="1.0.0")
        
        # 注册一个简单的资源
        self.register_resource(Resource(
            uri="hello://world",
            name="Hello World",
            resource_type=ResourceType.MEMORY,
            description="我的第一个资源"
        ))
        self.set_resource_content("hello://world", "Hello, MCP!")
        
        # 注册一个简单的工具
        def greet(name: str) -> str:
            return f"你好, {name}！欢迎使用 MCP！"
        
        self.register_tool(Tool(
            name="greet",
            description="问候工具",
            category=ToolCategory.CUSTOM,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            },
            function=greet
        ))

# 使用你的 Server
server = MyFirstServer()
```

### 步骤 4: 创建客户端并使用

```python
from 26_mcp_core import MCPClient

# 创建客户端
client = MCPClient()
client.connect("myserver", server)

# 读取资源
content = client.read_resource("myserver", "hello://world")
print(content)  # {"uri": "hello://world", "content": "Hello, MCP!"}

# 调用工具
result = client.call_tool("myserver", "greet", {"name": "张三"})
print(result)  # {"content": "你好, 张三！欢迎使用 MCP!"}
```

## 📚 实际应用示例

### 示例 1: 文件系统助手

```python
# 已实现在 26_mcp_demo.py 中
server = FileSystemMCPServer()
client = MCPClient()
client.connect("fs", server)

# 搜索文件
result = client.call_tool("fs", "search_files", {"keyword": "MCP"})
print(f"找到 {result['content']['found']} 个匹配文件")

# 列出目录
result = client.call_tool("fs", "list_directory", {"path": "/docs"})
print(f"目录包含 {result['content']['count']} 个文件")
```

### 示例 2: 数据分析助手

```python
# 已实现在 26_mcp_demo.py 中
server = DataAnalysisMCPServer()
client = MCPClient()
client.connect("data", server)

# 查询数据
users = client.call_tool("data", "query_data", {
    "data_uri": "db://users",
    "filter_field": "city",
    "filter_value": "北京"
})

# 统计分析
stats = client.call_tool("data", "calculate_statistics", {
    "data_uri": "db://sales",
    "field": "amount"
})
print(f"平均销售额: {stats['content']['average']}")
```

### 示例 3: AI Agent 集成

```python
# 已实现在 26_mcp_demo.py 中
agent = MCPEnabledAgent("智能助手")
agent.connect_to_server("filesystem", FileSystemMCPServer())
agent.connect_to_server("dataanalysis", DataAnalysisMCPServer())

# Agent 自动选择合适的工具和资源
response = agent.process_query("搜索包含 API 的文档")
print(response)

response = agent.process_query("分析销售数据")
print(response)
```

## 🎯 常见场景

### 场景 1: 知识库问答

```python
class KnowledgeBaseMCPServer(MCPServer):
    def __init__(self):
        super().__init__(name="Knowledge Base", version="1.0.0")
        
        # 注册文档资源
        for doc in documents:
            self.register_resource(Resource(
                uri=f"kb://doc/{doc.id}",
                name=doc.title,
                resource_type=ResourceType.DOCUMENT
            ))
        
        # 注册语义搜索工具
        def semantic_search(query: str, top_k: int = 5):
            # 实现向量搜索
            return search_results
        
        self.register_tool(Tool(
            name="semantic_search",
            description="语义搜索文档",
            function=semantic_search
        ))
```

### 场景 2: 代码助手

```python
class CodeAssistantMCPServer(MCPServer):
    def __init__(self):
        super().__init__(name="Code Assistant", version="1.0.0")
        
        # 代码文件资源
        for file in code_files:
            self.register_resource(Resource(
                uri=f"file:///{file.path}",
                name=file.name,
                resource_type=ResourceType.FILE,
                mime_type="text/x-python"
            ))
        
        # 代码分析工具
        def analyze_code(code: str, language: str):
            # 实现代码分析
            return analysis_result
        
        self.register_tool(Tool(
            name="analyze_code",
            description="分析代码质量",
            function=analyze_code
        ))
```

### 场景 3: 数据可视化

```python
class VisualizationMCPServer(MCPServer):
    def __init__(self):
        super().__init__(name="Visualization", version="1.0.0")
        
        # 图表生成工具
        def generate_chart(data: list, chart_type: str):
            # 生成图表
            return chart_data
        
        self.register_tool(Tool(
            name="generate_chart",
            description="生成数据可视化图表",
            function=generate_chart
        ))
```

## 🔧 调试技巧

### 1. 查看服务器能力

```python
# 获取服务器信息
info = server.get_server_info()
print(info)

# 列出所有资源
resources = server.list_resources()
for res in resources:
    print(f"Resource: {res['name']} - {res['uri']}")

# 列出所有工具
tools = server.list_tools()
for tool in tools:
    print(f"Tool: {tool['name']} - {tool['description']}")
```

### 2. 测试工具调用

```python
from 26_mcp_core import ToolCall

# 手动创建工具调用
tool_call = ToolCall(
    name="greet",
    arguments={"name": "测试"}
)

# 执行并检查结果
result = server.call_tool(tool_call)
print(f"Success: {not result.is_error}")
print(f"Result: {result.content}")
print(f"Time: {result.execution_time:.3f}s")
```

### 3. 启用详细日志

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 所有 MCP 操作都会输出详细日志
```

## 📖 深入学习

### 推荐阅读顺序

1. **基础理解**
   - 📄 `26_README_MCP.md` - MCP 介绍
   - 🏗️ 理解三大组件（Resource, Tool, Prompt）

2. **核心实现**
   - 💻 `26_mcp_core.py` - 核心代码
   - 🔍 研究 MCPServer 和 MCPClient 实现

3. **实践应用**
   - 🎯 `26_mcp_demo.py` - 运行演示
   - 🛠️ 修改示例，创建自己的 Server

4. **进阶内容**
   - 📊 `26_MCP_SUMMARY.md` - 完整总结
   - 🚀 性能优化、安全控制

### 下一步建议

1. **动手实践**
   - 运行所有演示代码
   - 修改参数观察变化
   - 添加新的资源和工具

2. **实际项目**
   - 为你的项目创建 MCP Server
   - 集成到现有的 AI Agent
   - 测试性能和稳定性

3. **社区参与**
   - 查看官方文档
   - 探索开源 MCP Servers
   - 分享你的实现

## ⚠️ 常见问题

### Q1: 为什么导入 26_mcp_core 失败？
**A**: Python 模块名不能以数字开头。解决方案：
```python
# 方案1: 使用 __import__
mcp_core = __import__('26_mcp_core')

# 方案2: 重命名文件
# 26_mcp_core.py -> mcp_core.py
# 然后: from mcp_core import ...
```

### Q2: 如何添加自定义工具？
**A**: 
```python
def my_custom_tool(arg1: str, arg2: int) -> dict:
    # 你的逻辑
    return {"result": "..."}

server.register_tool(Tool(
    name="my_tool",
    description="自定义工具",
    category=ToolCategory.CUSTOM,
    input_schema={...},
    function=my_custom_tool
))
```

### Q3: 如何处理大文件资源？
**A**: 使用流式传输或分块加载：
```python
def _load_resource_content(self, uri: str):
    # 分块读取大文件
    with open(uri, 'r') as f:
        return f.read(1024 * 1024)  # 1MB chunks
```

### Q4: 如何实现权限控制？
**A**:
```python
class SecureMCPServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.permissions = {}
    
    def call_tool(self, tool_call: ToolCall):
        # 检查权限
        if not self.has_permission(tool_call.name):
            return ToolResult(
                call_id=tool_call.id,
                content=None,
                is_error=True,
                error_message="权限不足"
            )
        return super().call_tool(tool_call)
```

## 🎉 恭喜！

你已经掌握了 MCP 的基础知识！现在可以：

✅ 理解 MCP 的核心概念
✅ 创建自己的 MCP Server
✅ 使用 MCP Client 访问资源和工具
✅ 将 MCP 集成到 AI Agent

**继续探索**：
- 运行 `python 26_mcp_demo.py` 体验完整功能
- 查看 `26_MCP_SUMMARY.md` 深入学习
- 实现你自己的 MCP 应用场景

祝你在 MCP 的世界里玩得开心！🚀
