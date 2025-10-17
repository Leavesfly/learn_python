# -*- coding: utf-8 -*-
"""
OpenManus Agent系统模拟实现 - 核心系统
=====================================

实现OpenManus Agent系统的四大核心特征：
1. 双执行机制（直接Agent模式和Flow编排模式）
2. 分层架构（BaseAgent→ReActAgent→ToolCallAgent→Manus）
3. 计划驱动任务分解
4. 动态工具调用

Author: 山泽
Date: 2025-10-03
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import re
import logging


# ============================================================================
# 核心枚举和数据结构
# ============================================================================

class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class ExecutionMode(Enum):
    """执行模式枚举"""
    DIRECT_AGENT = "direct_agent"       # 直接Agent模式
    FLOW_ORCHESTRATION = "flow"         # Flow编排模式


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Message:
    """消息结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    role: str = "user"  # user, assistant, system, tool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """任务结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCall:
    """工具调用结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0


# ============================================================================
# 工具注册表
# ============================================================================

class ToolRegistry:
    """工具注册表 - 管理所有可用工具"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_built_in_tools()
    
    def register_tool(self, name: str, func: Callable, description: str):
        """注册工具"""
        self.tools[name] = {"function": func, "description": description}
        logging.info(f"工具已注册: {name}")
    
    def call_tool(self, tool_call: ToolCall) -> ToolCall:
        """执行工具调用"""
        start_time = time.time()
        
        if tool_call.name not in self.tools:
            tool_call.error = f"工具 '{tool_call.name}' 不存在"
            return tool_call
        
        try:
            func = self.tools[tool_call.name]["function"]
            result = func(**tool_call.arguments)
            tool_call.result = result
        except Exception as e:
            tool_call.error = str(e)
        
        tool_call.execution_time = time.time() - start_time
        return tool_call
    
    def list_tools(self) -> List[Dict[str, str]]:
        """列出所有工具"""
        return [{"name": name, "description": info["description"]} for name, info in self.tools.items()]
    
    def _register_built_in_tools(self):
        """注册内置工具"""
        
        def calculator(expression: str) -> Dict[str, Any]:
            """计算器工具"""
            try:
                allowed_chars = set('0123456789+-*/()., ')
                if not all(c in allowed_chars for c in expression):
                    return {"error": "表达式包含非法字符"}
                result = eval(expression)
                return {"result": result, "expression": f"{expression} = {result}"}
            except Exception as e:
                return {"error": str(e)}
        
        def get_time() -> Dict[str, Any]:
            """时间工具"""
            now = datetime.now()
            return {
                "current_time": now.strftime("%Y年%m月%d日 %H:%M:%S"),
                "timestamp": now.timestamp()
            }
        
        def text_analyzer(text: str) -> Dict[str, Any]:
            """文本分析工具"""
            return {
                "length": len(text),
                "words": len(text.split()),
                "chinese_chars": len([c for c in text if '\u4e00' <= c <= '\u9fff']),
                "has_question": '？' in text or '?' in text
            }
        
        self.register_tool("calculator", calculator, "数学计算工具")
        self.register_tool("get_time", get_time, "获取当前时间")
        self.register_tool("text_analyzer", text_analyzer, "文本分析工具")


# ============================================================================
# 分层架构实现
# ============================================================================

class BaseAgent(ABC):
    """BaseAgent - OpenManus系统的基础层"""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = "BaseAgent"):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.state = AgentState.IDLE
        self.messages: List[Message] = []
        self.session_id = str(uuid.uuid4())[:8]
        
        # 配置日志
        self.logger = logging.getLogger(f"{name}_{self.agent_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - {name} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def add_message(self, message: Message):
        """添加消息"""
        self.messages.append(message)
    
    def set_state(self, new_state: AgentState):
        """设置状态"""
        old_state = self.state
        self.state = new_state
        self.logger.info(f"状态变更: {old_state.value} -> {new_state.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "message_count": len(self.messages)
        }
    
    @abstractmethod
    def process_message(self, message: Message) -> Message:
        """处理消息 - 子类必须实现"""
        pass


class ReActAgent(BaseAgent):
    """ReActAgent - 推理与行动结合层"""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = "ReActAgent"):
        super().__init__(agent_id, name)
        self.tool_registry = ToolRegistry()
        self.max_iterations = 3
        self.thought_history = []
    
    def _think(self, query: str) -> str:
        """思考步骤"""
        self.set_state(AgentState.THINKING)
        
        if any(op in query for op in ['+', '-', '*', '/', '计算', '算']):
            thought = "这是一个数学问题，我需要使用计算器工具"
        elif any(word in query for word in ['时间', '几点', '现在']):
            thought = "用户询问时间，我需要使用时间工具"
        elif any(word in query for word in ['分析', '统计']):
            thought = "这需要文本分析，我应该使用文本分析工具"
        else:
            thought = "我需要分析这个问题并决定如何回答"
        
        self.thought_history.append(thought)
        return thought
    
    def _act(self, thought: str, query: str) -> Optional[ToolCall]:
        """行动步骤"""
        self.set_state(AgentState.EXECUTING)
        
        if "计算器" in thought:
            expression = self._extract_math_expression(query)
            if expression:
                return ToolCall(name="calculator", arguments={"expression": expression})
        elif "时间工具" in thought:
            return ToolCall(name="get_time", arguments={})
        elif "文本分析" in thought:
            return ToolCall(name="text_analyzer", arguments={"text": query})
        
        return None
    
    def _extract_math_expression(self, text: str) -> Optional[str]:
        """提取数学表达式"""
        pattern = r'[\d+\-*/().\s]+'
        matches = re.findall(pattern, text)
        return max(matches, key=len).strip() if matches else None
    
    def process_message(self, message: Message) -> Message:
        """ReAct主处理流程"""
        self.add_message(message)
        query = message.content
        
        for iteration in range(self.max_iterations):
            thought = self._think(query)
            action = self._act(thought, query)
            
            if action:
                executed_action = self.tool_registry.call_tool(action)
                if executed_action.result and not executed_action.error:
                    response_content = f"基于我的分析：{thought}\n\n执行结果：{executed_action.result}"
                    break
                else:
                    response_content = f"工具执行失败：{executed_action.error}"
                    break
            else:
                response_content = f"基于我的思考：{thought}"
                break
        else:
            response_content = "抱歉，无法在限定步骤内完成"
        
        response = Message(content=response_content, role="assistant")
        self.add_message(response)
        self.set_state(AgentState.COMPLETED)
        return response


class ToolCallAgent(ReActAgent):
    """ToolCallAgent - 增强工具调用和管理层"""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = "ToolCallAgent"):
        super().__init__(agent_id, name)
        self.tool_call_history: List[ToolCall] = []
    
    def register_custom_tool(self, name: str, func: Callable, description: str):
        """注册自定义工具"""
        self.tool_registry.register_tool(name, func, description)
        self.logger.info(f"自定义工具已注册: {name}")
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """获取工具使用统计"""
        total_calls = len(self.tool_call_history)
        successful_calls = len([tc for tc in self.tool_call_history if tc.result and not tc.error])
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0
        }


class Manus(ToolCallAgent):
    """Manus - OpenManus系统的核心层"""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = "Manus"):
        super().__init__(agent_id, name)
        
        # 执行模式管理
        self.execution_mode = ExecutionMode.DIRECT_AGENT
        self.flow_definitions: Dict[str, Dict[str, Any]] = {}
        
        # 任务管理
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        
        # 计划系统
        self.planning_enabled = True
        
        self.logger.info("Manus核心系统初始化完成")
    
    def set_execution_mode(self, mode: ExecutionMode):
        """设置执行模式"""
        self.execution_mode = mode
        self.logger.info(f"执行模式已切换到: {mode.value}")
    
    def process_message(self, message: Message) -> Message:
        """根据执行模式处理消息"""
        self.add_message(message)
        
        if self.execution_mode == ExecutionMode.DIRECT_AGENT:
            return self._direct_agent_processing(message)
        elif self.execution_mode == ExecutionMode.FLOW_ORCHESTRATION:
            return self._flow_orchestration_processing(message)
        
        return Message(content="不支持的执行模式", role="assistant")
    
    def _direct_agent_processing(self, message: Message) -> Message:
        """直接Agent模式处理"""
        self.logger.info("使用直接Agent模式处理")
        
        if self.planning_enabled and self._is_complex_query(message.content):
            return self._planning_driven_processing(message)
        else:
            return super().process_message(message)
    
    def _flow_orchestration_processing(self, message: Message) -> Message:
        """Flow编排模式处理"""
        self.logger.info("使用Flow编排模式处理")
        
        flow_id = self._select_flow_for_query(message.content)
        if flow_id in self.flow_definitions:
            return self._execute_flow(flow_id, message)
        else:
            self.logger.warning("未找到合适的Flow，回退到直接Agent模式")
            return self._direct_agent_processing(message)
    
    def _is_complex_query(self, query: str) -> bool:
        """判断是否为复杂查询"""
        complex_indicators = ["分析", "研究", "计划", "步骤", "详细", "完整"]
        return any(indicator in query for indicator in complex_indicators)
    
    def _planning_driven_processing(self, message: Message) -> Message:
        """计划驱动的处理流程"""
        self.set_state(AgentState.PLANNING)
        
        # 任务分解
        tasks = self._decompose_task(message.content)
        
        # 执行任务
        results = []
        for task in tasks:
            result = self._execute_task(task)
            results.append(result)
            task.status = TaskStatus.COMPLETED if result else TaskStatus.FAILED
        
        # 结果合成
        final_result = self._synthesize_results(results, message.content)
        
        response = Message(
            content=final_result,
            role="assistant",
            metadata={"execution_mode": "planning_driven", "tasks_count": len(tasks)}
        )
        
        self.add_message(response)
        self.set_state(AgentState.COMPLETED)
        return response
    
    def _decompose_task(self, query: str) -> List[Task]:
        """任务分解"""
        tasks = []
        
        if "计算" in query and "分析" in query:
            tasks = [
                Task(name="数据提取", description="从查询中提取计算数据"),
                Task(name="数学计算", description="执行数学计算"),
                Task(name="结果分析", description="分析计算结果")
            ]
        elif "研究" in query or "学习" in query:
            tasks = [
                Task(name="信息收集", description="收集相关信息"),
                Task(name="内容分析", description="分析收集的内容"),
                Task(name="总结整理", description="整理和总结信息")
            ]
        else:
            tasks = [
                Task(name="问题分析", description="分析用户问题"),
                Task(name="解决方案", description="制定解决方案"),
                Task(name="结果输出", description="输出最终结果")
            ]
        
        return tasks
    
    def _execute_task(self, task: Task) -> Optional[Any]:
        """执行单个任务"""
        task.status = TaskStatus.IN_PROGRESS
        
        if "计算" in task.description:
            return {"type": "calculation", "status": "completed"}
        elif "时间" in task.description:
            tool_call = ToolCall(name="get_time", arguments={})
            result = self.tool_registry.call_tool(tool_call)
            return result.result
        else:
            return {"type": "general", "status": "completed"}
    
    def _synthesize_results(self, results: List[Any], original_query: str) -> str:
        """合成最终结果"""
        synthesis = f"基于任务分解，我为'{original_query}'提供了完整解决方案：\n\n"
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict) and "current_time" in result:
                synthesis += f"{i}. 时间信息：{result['current_time']}\n"
            else:
                synthesis += f"{i}. 任务完成：{result}\n"
        
        return synthesis
    
    def register_flow(self, flow_id: str, flow_definition: Dict[str, Any]):
        """注册Flow定义"""
        self.flow_definitions[flow_id] = flow_definition
        self.logger.info(f"Flow已注册: {flow_id}")
    
    def _select_flow_for_query(self, query: str) -> str:
        """为查询选择合适的Flow"""
        if "计算" in query:
            return "calculation_flow"
        elif "时间" in query:
            return "time_flow"
        return "default_flow"
    
    def _execute_flow(self, flow_id: str, message: Message) -> Message:
        """执行Flow"""
        self.logger.info(f"执行Flow: {flow_id}")
        
        flow_def = self.flow_definitions[flow_id]
        nodes = flow_def.get("nodes", [])
        
        final_result = f"Flow '{flow_id}' 执行完成，包含 {len(nodes)} 个节点"
        
        return Message(
            content=final_result,
            role="assistant",
            metadata={"flow_id": flow_id, "nodes_count": len(nodes)}
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "agent_info": self.get_status(),
            "execution_mode": self.execution_mode.value,
            "tool_stats": self.get_tool_stats(),
            "task_queue_size": len(self.task_queue),
            "flow_definitions": len(self.flow_definitions),
            "planning_enabled": self.planning_enabled
        }


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("OpenManus Agent系统核心模块已加载")
    print("支持的特性：")
    print("- 分层架构：BaseAgent → ReActAgent → ToolCallAgent → Manus")
    print("- 双执行机制：直接Agent模式 & Flow编排模式")
    print("- 计划驱动任务分解")
    print("- 动态工具调用")