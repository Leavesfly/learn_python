# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 核心实现
=====================================

实现 MCP 协议的核心组件：
1. Resource（资源）：提供上下文数据
2. Tool（工具）：可调用的功能
3. Prompt（提示词模板）：可复用的提示词
4. Server（服务器）：提供 MCP 服务
5. Client（客户端）：消费 MCP 服务

基于 JSON-RPC 2.0 协议
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


# ============================================================================
# 核心数据结构
# ============================================================================

class ResourceType(Enum):
    """资源类型枚举"""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"
    DOCUMENT = "document"


class ToolCategory(Enum):
    """工具类别枚举"""
    COMPUTATION = "computation"
    SEARCH = "search"
    DATA_ACCESS = "data_access"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class Resource:
    """MCP 资源定义"""
    uri: str  # 资源唯一标识符，如 file:///path/to/file
    name: str
    resource_type: ResourceType
    description: str = ""
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "uri": self.uri,
            "name": self.name,
            "type": self.resource_type.value,
            "description": self.description,
            "mimeType": self.mime_type,
            "metadata": self.metadata
        }


@dataclass
class ResourceContent:
    """资源内容"""
    uri: str
    content: Any
    mime_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "content": self.content,
            "mimeType": self.mime_type
        }


@dataclass
class Tool:
    """MCP 工具定义"""
    name: str
    description: str
    category: ToolCategory
    input_schema: Dict[str, Any]  # JSON Schema 格式
    function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（不包含函数引用）"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "inputSchema": self.input_schema,
            "metadata": self.metadata
        }


@dataclass
class ToolCall:
    """工具调用请求"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments
        }


@dataclass
class ToolResult:
    """工具调用结果"""
    call_id: str
    content: Any
    is_error: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "callId": self.call_id,
            "content": self.content,
            "isError": self.is_error,
            "errorMessage": self.error_message,
            "executionTime": self.execution_time
        }


@dataclass
class Prompt:
    """MCP 提示词模板"""
    name: str
    description: str
    template: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """渲染提示词"""
        return self.template.format(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
            "metadata": self.metadata
        }


@dataclass
class MCPRequest:
    """MCP 请求（基于 JSON-RPC 2.0）"""
    jsonrpc: str = "2.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", str(uuid.uuid4())),
            method=data.get("method", ""),
            params=data.get("params", {})
        )


@dataclass
class MCPResponse:
    """MCP 响应（基于 JSON-RPC 2.0）"""
    jsonrpc: str = "2.0"
    id: str = ""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id
        }
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response


# ============================================================================
# MCP Server 实现
# ============================================================================

class MCPServer:
    """MCP 服务器 - 提供资源、工具和提示词"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources: Dict[str, Resource] = {}
        self.tools: Dict[str, Tool] = {}
        self.prompts: Dict[str, Prompt] = {}
        
        # 资源内容缓存
        self._resource_content_cache: Dict[str, Any] = {}
        
        print(f"✅ MCP Server '{name}' v{version} 初始化完成")
    
    # ========== Resource 管理 ==========
    
    def register_resource(self, resource: Resource):
        """注册资源"""
        self.resources[resource.uri] = resource
        print(f"📦 资源已注册: {resource.name} ({resource.uri})")
    
    def set_resource_content(self, uri: str, content: Any):
        """设置资源内容"""
        self._resource_content_cache[uri] = content
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有资源"""
        return [resource.to_dict() for resource in self.resources.values()]
    
    def get_resource(self, uri: str) -> Optional[ResourceContent]:
        """获取资源内容"""
        if uri not in self.resources:
            return None
        
        resource = self.resources[uri]
        content = self._resource_content_cache.get(uri)
        
        if content is None:
            # 如果没有缓存，尝试动态加载
            content = self._load_resource_content(uri)
        
        return ResourceContent(
            uri=uri,
            content=content,
            mime_type=resource.mime_type
        )
    
    def _load_resource_content(self, uri: str) -> Any:
        """动态加载资源内容（子类可重写）"""
        return f"资源 {uri} 的内容"
    
    # ========== Tool 管理 ==========
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        print(f"🔧 工具已注册: {tool.name} - {tool.description}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """调用工具"""
        import time
        start_time = time.time()
        
        if tool_call.name not in self.tools:
            return ToolResult(
                call_id=tool_call.id,
                content=None,
                is_error=True,
                error_message=f"工具 '{tool_call.name}' 不存在"
            )
        
        tool = self.tools[tool_call.name]
        
        if tool.function is None:
            return ToolResult(
                call_id=tool_call.id,
                content=None,
                is_error=True,
                error_message=f"工具 '{tool_call.name}' 没有关联函数"
            )
        
        try:
            result = tool.function(**tool_call.arguments)
            execution_time = time.time() - start_time
            
            return ToolResult(
                call_id=tool_call.id,
                content=result,
                is_error=False,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                call_id=tool_call.id,
                content=None,
                is_error=True,
                error_message=str(e),
                execution_time=execution_time
            )
    
    # ========== Prompt 管理 ==========
    
    def register_prompt(self, prompt: Prompt):
        """注册提示词模板"""
        self.prompts[prompt.name] = prompt
        print(f"📝 提示词已注册: {prompt.name} - {prompt.description}")
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有提示词模板"""
        return [prompt.to_dict() for prompt in self.prompts.values()]
    
    def get_prompt(self, name: str, **kwargs) -> Optional[str]:
        """获取并渲染提示词"""
        if name not in self.prompts:
            return None
        
        prompt = self.prompts[name]
        return prompt.render(**kwargs)
    
    # ========== RPC 处理 ==========
    
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理 MCP 请求"""
        method = request.method
        params = request.params
        
        try:
            if method == "resources/list":
                result = self.list_resources()
            elif method == "resources/read":
                uri = params.get("uri", "")
                resource_content = self.get_resource(uri)
                result = resource_content.to_dict() if resource_content else None
            elif method == "tools/list":
                result = self.list_tools()
            elif method == "tools/call":
                tool_call = ToolCall(
                    id=params.get("id", str(uuid.uuid4())),
                    name=params.get("name", ""),
                    arguments=params.get("arguments", {})
                )
                tool_result = self.call_tool(tool_call)
                result = tool_result.to_dict()
            elif method == "prompts/list":
                result = self.list_prompts()
            elif method == "prompts/get":
                name = params.get("name", "")
                args = params.get("arguments", {})
                result = {"prompt": self.get_prompt(name, **args)}
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"方法不存在: {method}"}
                )
            
            return MCPResponse(id=request.id, result=result)
        
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": f"内部错误: {str(e)}"}
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {
                "resources": len(self.resources),
                "tools": len(self.tools),
                "prompts": len(self.prompts)
            }
        }


# ============================================================================
# MCP Client 实现
# ============================================================================

class MCPClient:
    """MCP 客户端 - 连接并使用 MCP Server"""
    
    def __init__(self, client_id: Optional[str] = None):
        self.client_id = client_id or str(uuid.uuid4())[:8]
        self.connected_servers: Dict[str, MCPServer] = {}
        print(f"🔌 MCP Client {self.client_id} 已创建")
    
    def connect(self, server_name: str, server: MCPServer):
        """连接到 MCP Server"""
        self.connected_servers[server_name] = server
        print(f"✅ 已连接到服务器: {server_name}")
    
    def disconnect(self, server_name: str):
        """断开连接"""
        if server_name in self.connected_servers:
            del self.connected_servers[server_name]
            print(f"❌ 已断开连接: {server_name}")
    
    def list_servers(self) -> List[str]:
        """列出已连接的服务器"""
        return list(self.connected_servers.keys())
    
    def _send_request(self, server_name: str, request: MCPRequest) -> MCPResponse:
        """发送请求到服务器"""
        if server_name not in self.connected_servers:
            return MCPResponse(
                id=request.id,
                error={"code": -32000, "message": f"未连接到服务器: {server_name}"}
            )
        
        server = self.connected_servers[server_name]
        return server.handle_request(request)
    
    # ========== Resource 操作 ==========
    
    def list_resources(self, server_name: str) -> List[Dict[str, Any]]:
        """列出服务器的资源"""
        request = MCPRequest(method="resources/list")
        response = self._send_request(server_name, request)
        return response.result if response.result else []
    
    def read_resource(self, server_name: str, uri: str) -> Optional[Dict[str, Any]]:
        """读取资源内容"""
        request = MCPRequest(method="resources/read", params={"uri": uri})
        response = self._send_request(server_name, request)
        return response.result
    
    # ========== Tool 操作 ==========
    
    def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """列出服务器的工具"""
        request = MCPRequest(method="tools/list")
        response = self._send_request(server_name, request)
        return response.result if response.result else []
    
    def call_tool(self, server_name: str, tool_name: str, 
                  arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用工具"""
        request = MCPRequest(
            method="tools/call",
            params={
                "id": str(uuid.uuid4()),
                "name": tool_name,
                "arguments": arguments or {}
            }
        )
        response = self._send_request(server_name, request)
        return response.result if response.result else {}
    
    # ========== Prompt 操作 ==========
    
    def list_prompts(self, server_name: str) -> List[Dict[str, Any]]:
        """列出服务器的提示词模板"""
        request = MCPRequest(method="prompts/list")
        response = self._send_request(server_name, request)
        return response.result if response.result else []
    
    def get_prompt(self, server_name: str, prompt_name: str, 
                   **kwargs) -> Optional[str]:
        """获取提示词"""
        request = MCPRequest(
            method="prompts/get",
            params={"name": prompt_name, "arguments": kwargs}
        )
        response = self._send_request(server_name, request)
        
        if response.result and "prompt" in response.result:
            return response.result["prompt"]
        return None


# ============================================================================
# 工具函数
# ============================================================================

def create_json_schema(properties: Dict[str, Dict[str, Any]], 
                       required: Optional[List[str]] = None) -> Dict[str, Any]:
    """创建 JSON Schema"""
    return {
        "type": "object",
        "properties": properties,
        "required": required or []
    }


if __name__ == "__main__":
    print("MCP 核心模块已加载")
    print("\n支持的功能：")
    print("- Resource: 资源管理（文件、数据库、API等）")
    print("- Tool: 工具调用（计算、搜索、数据访问等）")
    print("- Prompt: 提示词模板管理")
    print("- Server: MCP 服务器实现")
    print("- Client: MCP 客户端实现")
    print("- JSON-RPC 2.0 协议支持")
