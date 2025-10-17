# MCP (Model Context Protocol) 完整总结

## 📌 核心概念

### 什么是 MCP？

**Model Context Protocol (MCP)** 是由 Anthropic 推出的开放标准协议，旨在解决 AI 应用与外部数据源、工具之间的集成问题。

### 核心价值

1. **标准化接口**：统一的协议规范，避免重复开发
2. **松耦合架构**：AI 应用与数据源解耦，易于维护
3. **可扩展性**：轻松添加新的资源和工具
4. **互操作性**：不同系统间的无缝集成
5. **安全性**：集中式权限控制和审计

## 🏗️ 架构设计

```
┌──────────────────────────────────────────┐
│         AI Application (Client)           │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │       MCP Client SDK                 │ │
│  │  - 资源访问                          │ │
│  │  - 工具调用                          │ │
│  │  - 提示词管理                        │ │
│  └─────────────────────────────────────┘ │
└────────────────┬──────────────────────────┘
                 │
                 │ JSON-RPC 2.0 Protocol
                 │ (HTTP / WebSocket / Stdio)
                 │
┌────────────────┴──────────────────────────┐
│          MCP Server Framework             │
│                                           │
│  ┌───────────┬──────────┬──────────────┐ │
│  │ Resources │  Tools   │   Prompts    │ │
│  │  (只读)   │ (可执行) │ (可复用模板) │ │
│  └───────────┴──────────┴──────────────┘ │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │    External Data & Services         │ │
│  │  - 文件系统                         │ │
│  │  - 数据库                           │ │
│  │  - API 服务                         │ │
│  │  - 搜索引擎                         │ │
│  │  - 第三方工具                       │ │
│  └─────────────────────────────────────┘ │
└───────────────────────────────────────────┘
```

## 🎯 三大核心组件

### 1. Resources（资源）

**定义**：提供上下文数据的只读接口

**特点**：
- 只读访问
- 结构化数据
- URI 标识
- MIME 类型支持

**典型用途**：
```python
# 文件资源
file:///path/to/document.pdf

# 数据库资源
db://users/table/records

# API 资源
api://weather/current/beijing

# 内存资源
memory://conversation/history
```

**适用场景**：
- 📄 文档库访问
- 🗄️ 数据库查询
- 🌐 API 数据获取
- 💾 缓存数据读取

### 2. Tools（工具）

**定义**：Agent 可调用的可执行功能

**特点**：
- 参数化输入
- 返回结果
- JSON Schema 定义
- 异步执行支持

**典型示例**：
```python
{
  "name": "search_files",
  "description": "在文件中搜索关键词",
  "category": "search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "keyword": {"type": "string"},
      "path": {"type": "string"}
    },
    "required": ["keyword"]
  }
}
```

**工具分类**：
- 🔢 计算工具（calculator, statistics）
- 🔍 搜索工具（search, query）
- 📊 数据访问（database, api）
- 🖥️ 系统工具（file_ops, command）
- 🔧 自定义工具

### 3. Prompts（提示词模板）

**定义**：可复用的参数化提示词片段

**特点**：
- 参数化
- 可组合
- 版本化
- 领域特定

**示例**：
```python
{
  "name": "code_review",
  "description": "代码审查提示词",
  "template": """
请审查以下 {language} 代码：

```{language}
{code}
```

关注点：
1. 代码质量
2. 性能问题
3. 安全隐患
4. 最佳实践

请提供详细的审查意见。
  """,
  "arguments": [
    {"name": "language", "type": "string", "required": true},
    {"name": "code", "type": "string", "required": true}
  ]
}
```

## 🔄 工作流程

### 典型交互流程

```
1. 连接阶段
   Client ─────> Server: 连接请求
   Client <───── Server: 服务器信息（资源、工具、提示词列表）

2. 发现阶段
   Client ─────> Server: resources/list
   Client <───── Server: 返回可用资源列表
   
   Client ─────> Server: tools/list
   Client <───── Server: 返回可用工具列表

3. 使用阶段
   Client ─────> Server: resources/read (uri)
   Client <───── Server: 返回资源内容
   
   Client ─────> Server: tools/call (name, args)
   Client <───── Server: 返回执行结果

4. Agent 处理
   Agent 分析用户请求
     ↓
   选择合适的资源和工具
     ↓
   通过 MCP Client 调用
     ↓
   整合结果返回用户
```

## 📦 协议规范

### JSON-RPC 2.0 请求

```json
{
  "jsonrpc": "2.0",
  "id": "req-12345",
  "method": "tools/call",
  "params": {
    "name": "search_files",
    "arguments": {
      "keyword": "MCP",
      "path": "/docs"
    }
  }
}
```

### JSON-RPC 2.0 响应

```json
{
  "jsonrpc": "2.0",
  "id": "req-12345",
  "result": {
    "callId": "call-67890",
    "content": {
      "found": 5,
      "results": [...]
    },
    "isError": false,
    "executionTime": 0.15
  }
}
```

### 错误处理

```json
{
  "jsonrpc": "2.0",
  "id": "req-12345",
  "error": {
    "code": -32601,
    "message": "方法不存在",
    "data": {"method": "invalid/method"}
  }
}
```

## 🚀 实际应用场景

### 1. 代码辅助系统

```
MCP Resources:
  - file:///project/**/*.py (代码文件)
  - git://commits/recent (Git 历史)
  - docs://api/reference (API 文档)

MCP Tools:
  - analyze_code (代码分析)
  - run_tests (运行测试)
  - lint_check (代码检查)
  - git_diff (查看差异)

使用流程:
1. Agent 读取代码文件 (Resource)
2. 调用代码分析工具 (Tool)
3. 使用代码审查模板 (Prompt)
4. 生成审查报告
```

### 2. 数据分析系统

```
MCP Resources:
  - db://sales/records (销售数据)
  - db://users/profiles (用户数据)
  - api://market/trends (市场趋势)

MCP Tools:
  - query_database (查询数据)
  - calculate_stats (统计计算)
  - generate_chart (生成图表)
  - export_report (导出报告)

使用流程:
1. Agent 从数据库读取数据 (Resource)
2. 调用统计工具分析 (Tool)
3. 使用分析报告模板 (Prompt)
4. 生成分析结果
```

### 3. 文档助手系统

```
MCP Resources:
  - file:///docs/**/*.md (文档文件)
  - memory://conversation/history (对话历史)
  - api://wiki/search (知识库)

MCP Tools:
  - search_docs (搜索文档)
  - summarize (内容总结)
  - translate (翻译)
  - update_index (更新索引)

使用流程:
1. Agent 搜索相关文档 (Tool)
2. 读取文档内容 (Resource)
3. 使用总结模板 (Prompt)
4. 返回精炼答案
```

## 💡 最佳实践

### 设计原则

1. **单一职责**：每个 Server 专注特定领域
2. **资源粒度**：合理划分资源，避免过大或过小
3. **工具设计**：输入输出清晰，职责明确
4. **错误处理**：完善的错误信息和恢复机制
5. **性能优化**：缓存、批量处理、异步执行

### 安全考虑

1. **权限控制**：
   - 资源访问权限
   - 工具执行权限
   - 敏感数据脱敏

2. **输入验证**：
   - JSON Schema 验证
   - 参数范围检查
   - SQL 注入防护

3. **审计日志**：
   - 记录所有操作
   - 敏感操作追踪
   - 异常行为监控

### 性能优化

1. **资源缓存**：
```python
class CachedMCPServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_ttl = 300  # 5分钟
    
    def get_resource(self, uri):
        if uri in self.cache:
            cached_time, content = self.cache[uri]
            if time.time() - cached_time < self.cache_ttl:
                return content
        
        content = super().get_resource(uri)
        self.cache[uri] = (time.time(), content)
        return content
```

2. **批量操作**：
```python
# 批量读取资源
def batch_read_resources(self, uris: List[str]):
    return [self.get_resource(uri) for uri in uris]

# 批量调用工具
def batch_call_tools(self, tool_calls: List[ToolCall]):
    return [self.call_tool(tc) for tc in tool_calls]
```

3. **异步处理**：
```python
import asyncio

async def async_call_tool(self, tool_call: ToolCall):
    # 异步执行工具
    result = await asyncio.to_thread(
        self.tools[tool_call.name]["function"],
        **tool_call.arguments
    )
    return result
```

## 🔧 开发指南

### 创建自定义 MCP Server

```python
from 26_mcp_core import MCPServer, Resource, Tool, Prompt

class MyCustomServer(MCPServer):
    def __init__(self):
        super().__init__(name="My Server", version="1.0.0")
        self._setup()
    
    def _setup(self):
        # 注册资源
        self.register_resource(Resource(
            uri="custom://data",
            name="Custom Data",
            resource_type=ResourceType.CUSTOM,
            description="My custom data source"
        ))
        
        # 注册工具
        self.register_tool(Tool(
            name="my_tool",
            description="My custom tool",
            category=ToolCategory.CUSTOM,
            input_schema={...},
            function=self.my_tool_function
        ))
        
        # 注册提示词
        self.register_prompt(Prompt(
            name="my_prompt",
            description="My custom prompt",
            template="..."
        ))
    
    def my_tool_function(self, **kwargs):
        # 工具实现
        return {"result": "..."}
```

### 集成到 Agent

```python
from 26_mcp_core import MCPClient

class MyAgent:
    def __init__(self):
        self.mcp_client = MCPClient()
    
    def connect_servers(self):
        # 连接多个 MCP Server
        self.mcp_client.connect("custom", MyCustomServer())
        self.mcp_client.connect("files", FileSystemServer())
    
    def process_query(self, query: str):
        # 1. 发现可用资源和工具
        tools = self.mcp_client.list_tools("custom")
        
        # 2. 分析查询意图
        # 3. 选择合适的工具
        # 4. 调用工具
        result = self.mcp_client.call_tool(
            "custom", "my_tool", {"arg": "value"}
        )
        
        # 5. 整合结果返回
        return result
```

## 📊 性能指标

### 基准测试

```python
import time

def benchmark_mcp():
    server = FileSystemMCPServer()
    client = MCPClient()
    client.connect("fs", server)
    
    # 测试资源读取
    start = time.time()
    for _ in range(1000):
        client.read_resource("fs", "file:///test.txt")
    print(f"Resource read: {time.time() - start:.2f}s")
    
    # 测试工具调用
    start = time.time()
    for _ in range(1000):
        client.call_tool("fs", "search", {"keyword": "test"})
    print(f"Tool call: {time.time() - start:.2f}s")
```

### 优化建议

1. **连接池**：复用 MCP 连接
2. **请求合并**：批量请求减少往返
3. **并发处理**：异步并发执行
4. **智能缓存**：LRU/LFU 缓存策略
5. **流式传输**：大数据分块传输

## 🌐 生态系统

### 官方工具

- **MCP Inspector**：调试和测试工具
- **MCP CLI**：命令行管理工具
- **MCP SDK**：多语言 SDK

### 社区资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [示例仓库](https://github.com/modelcontextprotocol/servers)

### 常用 MCP Servers

1. **文件系统 Server**：访问本地文件
2. **数据库 Server**：连接 SQL/NoSQL
3. **Git Server**：版本控制集成
4. **浏览器 Server**：Web 自动化
5. **搜索 Server**：全文搜索引擎

## 🎓 学习路径

### 入门阶段
1. 理解 MCP 基本概念
2. 学习 JSON-RPC 协议
3. 运行示例代码
4. 创建简单的 Server

### 进阶阶段
1. 设计复杂的资源结构
2. 实现高级工具功能
3. 优化性能和安全
4. 集成到实际项目

### 高级阶段
1. 开发自定义传输协议
2. 实现分布式 MCP Server
3. 构建 MCP 生态系统
4. 贡献开源社区

## 📝 总结

MCP 为 AI Agent 与外部世界的交互提供了标准化、可扩展的解决方案。通过统一的协议接口，开发者可以：

✅ **快速集成**：无需为每个数据源编写适配器
✅ **灵活扩展**：轻松添加新的资源和工具
✅ **可移植性**：Agent 可以无缝切换不同的 MCP Server
✅ **安全可控**：集中式权限管理和审计
✅ **生态共建**：共享和复用 MCP Server

MCP 正在成为 AI Agent 开发的事实标准，值得深入学习和应用！

---

**相关文件**：
- `26_README_MCP.md` - MCP 介绍文档
- `26_mcp_core.py` - MCP 核心实现
- `26_mcp_demo.py` - 完整演示代码

**下一步**：运行 `python 26_mcp_demo.py` 开始体验！
