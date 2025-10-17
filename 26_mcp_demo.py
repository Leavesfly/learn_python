# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 完整演示
====================================

演示如何使用 MCP 构建 AI Agent 系统
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

# 说明：由于 Python 模块名不能以数字开头，这里直接复制核心类
# 实际使用时，应将 26_mcp_core.py 重命名为 mcp_core.py

# 临时解决方案：从 26_mcp_core 导入
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入，如果失败则使用替代方案
try:
    # 使用 __import__ 导入以数字开头的模块
    mcp_core_module = __import__('26_mcp_core')
    MCPServer = mcp_core_module.MCPServer
    MCPClient = mcp_core_module.MCPClient
    Resource = mcp_core_module.Resource
    ResourceType = mcp_core_module.ResourceType
    ResourceContent = mcp_core_module.ResourceContent
    Tool = mcp_core_module.Tool
    ToolCategory = mcp_core_module.ToolCategory
    ToolCall = mcp_core_module.ToolCall
    Prompt = mcp_core_module.Prompt
    create_json_schema = mcp_core_module.create_json_schema
except Exception as e:
    print(f"警告：无法导入 26_mcp_core 模块: {e}")
    print("请确保 26_mcp_core.py 文件存在于同一目录")
    print("或者将其重命名为 mcp_core.py 后使用 'from mcp_core import ...'")
    sys.exit(1)


# ============================================================================
# 示例 1: 创建文件系统 MCP Server
# ============================================================================

class FileSystemMCPServer(MCPServer):
    """文件系统 MCP Server 示例"""
    
    def __init__(self):
        super().__init__(name="FileSystem Server", version="1.0.0")
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()
    
    def _setup_resources(self):
        """设置文件系统资源"""
        # 模拟文件资源
        files = {
            "file:///docs/readme.md": "# 项目说明\n\n这是一个使用 MCP 的示例项目。",
            "file:///docs/api.md": "# API 文档\n\n## 端点\n- GET /api/data",
            "file:///config/settings.json": '{"debug": true, "port": 8080}'
        }
        
        for uri, content in files.items():
            filename = uri.split("/")[-1]
            resource = Resource(
                uri=uri,
                name=filename,
                resource_type=ResourceType.FILE,
                description=f"文件: {filename}",
                mime_type="text/plain" if uri.endswith(".md") else "application/json"
            )
            self.register_resource(resource)
            self.set_resource_content(uri, content)
    
    def _setup_tools(self):
        """设置文件系统工具"""
        
        def search_files(keyword: str) -> Dict[str, Any]:
            """搜索文件内容"""
            results = []
            for uri, content in self._resource_content_cache.items():
                if keyword.lower() in str(content).lower():
                    results.append({
                        "uri": uri,
                        "preview": str(content)[:100] + "..."
                    })
            return {"found": len(results), "results": results}
        
        def list_directory(path: str = "/") -> Dict[str, Any]:
            """列出目录内容"""
            files = [uri for uri in self.resources.keys() if uri.startswith(f"file://{path}")]
            return {"path": path, "files": files, "count": len(files)}
        
        # 注册搜索工具
        self.register_tool(Tool(
            name="search_files",
            description="在文件中搜索关键词",
            category=ToolCategory.SEARCH,
            input_schema=create_json_schema({
                "keyword": {"type": "string", "description": "搜索关键词"}
            }, required=["keyword"]),
            function=search_files
        ))
        
        # 注册列目录工具
        self.register_tool(Tool(
            name="list_directory",
            description="列出目录中的文件",
            category=ToolCategory.DATA_ACCESS,
            input_schema=create_json_schema({
                "path": {"type": "string", "description": "目录路径", "default": "/"}
            }),
            function=list_directory
        ))
    
    def _setup_prompts(self):
        """设置提示词模板"""
        
        # 文件分析提示词
        self.register_prompt(Prompt(
            name="analyze_file",
            description="分析文件内容的提示词模板",
            template="""请分析以下文件内容：

文件：{filename}
内容：
{content}

请提供：
1. 文件类型和格式
2. 主要内容摘要
3. 关键信息提取
4. 建议的改进点""",
            arguments=[
                {"name": "filename", "type": "string", "required": True},
                {"name": "content", "type": "string", "required": True}
            ]
        ))


# ============================================================================
# 示例 2: 创建数据分析 MCP Server
# ============================================================================

class DataAnalysisMCPServer(MCPServer):
    """数据分析 MCP Server 示例"""
    
    def __init__(self):
        super().__init__(name="Data Analysis Server", version="1.0.0")
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()
    
    def _setup_resources(self):
        """设置数据资源"""
        # 模拟数据库数据
        user_data = {
            "database": "users",
            "records": [
                {"id": 1, "name": "Alice", "age": 25, "city": "北京"},
                {"id": 2, "name": "Bob", "age": 30, "city": "上海"},
                {"id": 3, "name": "Charlie", "age": 28, "city": "深圳"}
            ]
        }
        
        sales_data = {
            "database": "sales",
            "records": [
                {"product": "笔记本", "amount": 5000, "date": "2024-01-15"},
                {"product": "手机", "amount": 3000, "date": "2024-01-16"},
                {"product": "平板", "amount": 2000, "date": "2024-01-17"}
            ]
        }
        
        # 注册资源
        user_resource = Resource(
            uri="db://users",
            name="用户数据",
            resource_type=ResourceType.DATABASE,
            description="用户信息数据库",
            mime_type="application/json"
        )
        self.register_resource(user_resource)
        self.set_resource_content("db://users", user_data)
        
        sales_resource = Resource(
            uri="db://sales",
            name="销售数据",
            resource_type=ResourceType.DATABASE,
            description="销售记录数据库",
            mime_type="application/json"
        )
        self.register_resource(sales_resource)
        self.set_resource_content("db://sales", sales_data)
    
    def _setup_tools(self):
        """设置数据分析工具"""
        
        def calculate_statistics(data_uri: str, field: str) -> Dict[str, Any]:
            """计算统计数据"""
            resource_content = self.get_resource(data_uri)
            if not resource_content:
                return {"error": "数据源不存在"}
            
            data = resource_content.content
            records = data.get("records", [])
            
            values = [record.get(field) for record in records if field in record]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if not numeric_values:
                return {"error": f"字段 {field} 不包含数值数据"}
            
            return {
                "field": field,
                "count": len(numeric_values),
                "sum": sum(numeric_values),
                "average": sum(numeric_values) / len(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values)
            }
        
        def query_data(data_uri: str, filter_field: Optional[str] = None, 
                      filter_value: Optional[Any] = None) -> Dict[str, Any]:
            """查询数据"""
            resource_content = self.get_resource(data_uri)
            if not resource_content:
                return {"error": "数据源不存在"}
            
            data = resource_content.content
            records = data.get("records", [])
            
            if filter_field and filter_value is not None:
                filtered = [r for r in records if r.get(filter_field) == filter_value]
            else:
                filtered = records
            
            return {
                "total": len(records),
                "filtered": len(filtered),
                "results": filtered
            }
        
        # 注册统计工具
        self.register_tool(Tool(
            name="calculate_statistics",
            description="计算数据的统计信息（总和、平均值、最大值、最小值）",
            category=ToolCategory.COMPUTATION,
            input_schema=create_json_schema({
                "data_uri": {"type": "string", "description": "数据源URI"},
                "field": {"type": "string", "description": "要统计的字段名"}
            }, required=["data_uri", "field"]),
            function=calculate_statistics
        ))
        
        # 注册查询工具
        self.register_tool(Tool(
            name="query_data",
            description="查询和过滤数据",
            category=ToolCategory.DATA_ACCESS,
            input_schema=create_json_schema({
                "data_uri": {"type": "string", "description": "数据源URI"},
                "filter_field": {"type": "string", "description": "过滤字段"},
                "filter_value": {"description": "过滤值"}
            }, required=["data_uri"]),
            function=query_data
        ))
    
    def _setup_prompts(self):
        """设置提示词模板"""
        
        self.register_prompt(Prompt(
            name="data_analysis_report",
            description="数据分析报告模板",
            template="""# 数据分析报告

## 数据源
{data_source}

## 统计结果
{statistics}

## 分析结论
请基于以上数据提供：
1. 数据分布特征
2. 异常值识别
3. 趋势分析
4. 业务建议""",
            arguments=[
                {"name": "data_source", "type": "string", "required": True},
                {"name": "statistics", "type": "string", "required": True}
            ]
        ))


# ============================================================================
# 示例 3: AI Agent 使用 MCP
# ============================================================================

class MCPEnabledAgent:
    """支持 MCP 的 AI Agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.client = MCPClient(client_id=name.lower().replace(" ", "_"))
        self.conversation_history = []
        
        print(f"🤖 Agent '{name}' 已创建，支持 MCP 协议")
    
    def connect_to_server(self, server_name: str, server: MCPServer):
        """连接到 MCP Server"""
        self.client.connect(server_name, server)
        print(f"✅ Agent 已连接到 '{server_name}' 服务器")
    
    def discover_capabilities(self):
        """发现所有连接服务器的能力"""
        capabilities = {}
        
        for server_name in self.client.list_servers():
            capabilities[server_name] = {
                "resources": self.client.list_resources(server_name),
                "tools": self.client.list_tools(server_name),
                "prompts": self.client.list_prompts(server_name)
            }
        
        return capabilities
    
    def process_query(self, query: str) -> str:
        """处理用户查询"""
        self.conversation_history.append({"role": "user", "content": query})
        
        # 简单的意图识别
        if "搜索" in query or "查找" in query:
            response = self._handle_search_query(query)
        elif "统计" in query or "分析" in query or "计算" in query:
            response = self._handle_analysis_query(query)
        elif "读取" in query or "查看" in query or "显示" in query:
            response = self._handle_read_query(query)
        else:
            response = self._handle_general_query(query)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def _handle_search_query(self, query: str) -> str:
        """处理搜索查询"""
        # 提取关键词（简单实现）
        keywords = query.replace("搜索", "").replace("查找", "").strip()
        
        # 调用文件系统服务器的搜索工具
        try:
            result = self.client.call_tool(
                "filesystem",
                "search_files",
                {"keyword": keywords}
            )
            
            if "isError" in result and result["isError"]:
                return f"搜索失败: {result.get('errorMessage')}"
            
            content = result.get("content", {})
            found = content.get("found", 0)
            
            if found == 0:
                return f"未找到包含 '{keywords}' 的文件"
            
            results = content.get("results", [])
            response = f"找到 {found} 个匹配的文件：\n\n"
            for r in results[:3]:  # 只显示前3个
                response += f"📄 {r['uri']}\n   {r['preview']}\n\n"
            
            return response
        except Exception as e:
            return f"搜索出错: {str(e)}"
    
    def _handle_analysis_query(self, query: str) -> str:
        """处理分析查询"""
        # 简单示例：分析销售数据
        try:
            result = self.client.call_tool(
                "dataanalysis",
                "calculate_statistics",
                {"data_uri": "db://sales", "field": "amount"}
            )
            
            if "isError" in result and result["isError"]:
                return f"分析失败: {result.get('errorMessage')}"
            
            stats = result.get("content", {})
            
            response = f"""📊 销售数据统计分析：

- 记录数量：{stats.get('count')}
- 总销售额：¥{stats.get('sum')}
- 平均销售额：¥{stats.get('average', 0):.2f}
- 最高销售额：¥{stats.get('max')}
- 最低销售额：¥{stats.get('min')}
"""
            return response
        except Exception as e:
            return f"分析出错: {str(e)}"
    
    def _handle_read_query(self, query: str) -> str:
        """处理读取查询"""
        # 列出资源
        try:
            resources = self.client.list_resources("filesystem")
            
            if not resources:
                return "没有可用的资源"
            
            response = "📚 可用资源：\n\n"
            for resource in resources[:5]:
                response += f"- {resource['name']}: {resource['description']}\n"
            
            return response
        except Exception as e:
            return f"读取出错: {str(e)}"
    
    def _handle_general_query(self, query: str) -> str:
        """处理通用查询"""
        capabilities = self.discover_capabilities()
        
        total_resources = sum(len(cap['resources']) for cap in capabilities.values())
        total_tools = sum(len(cap['tools']) for cap in capabilities.values())
        
        return f"""我是 {self.name}，通过 MCP 协议连接到了多个服务器。

当前能力：
- 📦 可访问 {total_resources} 个资源
- 🔧 可使用 {total_tools} 个工具
- 🌐 连接到 {len(capabilities)} 个服务器

你可以让我：
- 搜索文件内容
- 分析数据统计
- 读取资源信息

请告诉我你需要什么帮助！"""


# ============================================================================
# 主演示函数
# ============================================================================

def demo_basic_mcp():
    """基础 MCP 演示"""
    print("\n" + "="*70)
    print("📋 演示 1: MCP 基础功能")
    print("="*70)
    
    # 创建服务器
    server = FileSystemMCPServer()
    
    # 创建客户端
    client = MCPClient()
    client.connect("filesystem", server)
    
    # 列出资源
    print("\n📦 可用资源：")
    resources = client.list_resources("filesystem")
    for res in resources:
        print(f"  - {res['name']}: {res['description']}")
    
    # 读取资源
    print("\n📖 读取资源内容：")
    content = client.read_resource("filesystem", "file:///docs/readme.md")
    if content:
        print(f"  URI: {content['uri']}")
        print(f"  内容: {content['content'][:100]}...")
    
    # 列出工具
    print("\n🔧 可用工具：")
    tools = client.list_tools("filesystem")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # 调用工具
    print("\n🔍 调用搜索工具：")
    result = client.call_tool("filesystem", "search_files", {"keyword": "API"})
    print(f"  结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 获取提示词
    print("\n📝 获取提示词模板：")
    prompt = client.get_prompt(
        "filesystem",
        "analyze_file",
        filename="readme.md",
        content="示例文档内容"
    )
    if prompt:
        print(f"  {prompt[:200]}...")


def demo_data_analysis():
    """数据分析演示"""
    print("\n" + "="*70)
    print("📊 演示 2: 数据分析 MCP Server")
    print("="*70)
    
    # 创建数据分析服务器
    server = DataAnalysisMCPServer()
    
    # 创建客户端
    client = MCPClient()
    client.connect("dataanalysis", server)
    
    # 查询数据
    print("\n🔍 查询用户数据：")
    result = client.call_tool("dataanalysis", "query_data", {
        "data_uri": "db://users",
        "filter_field": "city",
        "filter_value": "北京"
    })
    print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 统计分析
    print("\n📈 销售额统计分析：")
    result = client.call_tool("dataanalysis", "calculate_statistics", {
        "data_uri": "db://sales",
        "field": "amount"
    })
    print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")


def demo_agent_with_mcp():
    """AI Agent 使用 MCP 演示"""
    print("\n" + "="*70)
    print("🤖 演示 3: AI Agent 使用 MCP")
    print("="*70)
    
    # 创建服务器
    fs_server = FileSystemMCPServer()
    da_server = DataAnalysisMCPServer()
    
    # 创建 Agent
    agent = MCPEnabledAgent("智能助手")
    agent.connect_to_server("filesystem", fs_server)
    agent.connect_to_server("dataanalysis", da_server)
    
    # 发现能力
    print("\n🔍 发现 Agent 能力...")
    capabilities = agent.discover_capabilities()
    print(f"  连接的服务器: {list(capabilities.keys())}")
    
    # 测试查询
    test_queries = [
        "你好，介绍一下自己",
        "搜索 API 相关的文档",
        "统计销售数据",
        "查看所有可用资源"
    ]
    
    for query in test_queries:
        print(f"\n👤 用户: {query}")
        response = agent.process_query(query)
        print(f"🤖 助手: {response}")
        print("-" * 70)


def interactive_demo():
    """交互式演示"""
    print("\n" + "="*70)
    print("💬 演示 4: 交互式 MCP Agent")
    print("="*70)
    
    # 创建完整的 MCP 生态系统
    fs_server = FileSystemMCPServer()
    da_server = DataAnalysisMCPServer()
    
    agent = MCPEnabledAgent("MCP 智能助手")
    agent.connect_to_server("filesystem", fs_server)
    agent.connect_to_server("dataanalysis", da_server)
    
    print("\n✅ MCP Agent 已准备就绪！")
    print("\n可用命令：")
    print("  - 搜索 <关键词>")
    print("  - 统计分析")
    print("  - 查看资源")
    print("  - 能力展示")
    print("  - quit 退出")
    print("\n" + "="*70)
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n👋 再见！")
                break
            
            if user_input == "能力展示":
                caps = agent.discover_capabilities()
                print("\n📊 当前连接的 MCP 服务器：")
                for server_name, cap in caps.items():
                    print(f"\n  🌐 {server_name}:")
                    print(f"    - 资源: {len(cap['resources'])} 个")
                    print(f"    - 工具: {len(cap['tools'])} 个")
                    print(f"    - 提示词: {len(cap['prompts'])} 个")
                continue
            
            response = agent.process_query(user_input)
            print(f"\n🤖 助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 MCP (Model Context Protocol) 完整演示")
    print("="*70)
    print("""
MCP 是一个标准化协议，用于连接 AI 应用与外部资源、工具。

本演示包含：
1. 基础 MCP 功能（资源、工具、提示词）
2. 数据分析 MCP Server
3. AI Agent 使用 MCP
4. 交互式演示

选择演示模式：
1 - 基础功能演示
2 - 数据分析演示
3 - Agent 使用演示
4 - 交互式演示
5 - 全部演示
0 - 退出
    """)
    
    while True:
        try:
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用 MCP 演示系统！")
                break
            elif choice == '1':
                demo_basic_mcp()
            elif choice == '2':
                demo_data_analysis()
            elif choice == '3':
                demo_agent_with_mcp()
            elif choice == '4':
                interactive_demo()
            elif choice == '5':
                demo_basic_mcp()
                demo_data_analysis()
                demo_agent_with_mcp()
            else:
                print("❌ 无效选择，请输入 0-5")
        
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
