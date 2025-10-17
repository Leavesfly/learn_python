# -*- coding: utf-8 -*-
"""
智能Agent系统Demo
包含多种类型的Agent：任务执行Agent、对话Agent、工具使用Agent等
"""

import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class Message:
    """消息类"""
    sender: str
    content: str
    timestamp: datetime
    message_type: str = "text"


class Tool(ABC):
    """工具基类"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具功能"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass


class CalculatorTool(Tool):
    """计算器工具"""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "执行基本数学运算：加法、减法、乘法、除法"
    
    def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """执行计算操作"""
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"success": False, "error": "除零错误"}
                result = a / b
            else:
                return {"success": False, "error": f"不支持的操作: {operation}"}
            
            return {
                "success": True,
                "result": result,
                "operation": f"{a} {operation} {b} = {result}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class WeatherTool(Tool):
    """天气查询工具（模拟）"""
    
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return "查询指定城市的天气信息"
    
    def execute(self, city: str) -> Dict[str, Any]:
        """模拟天气查询"""
        # 模拟天气数据
        weather_conditions = ["晴天", "多云", "阴天", "小雨", "中雨"]
        temperatures = list(range(15, 35))
        
        return {
            "success": True,
            "city": city,
            "weather": random.choice(weather_conditions),
            "temperature": random.choice(temperatures),
            "humidity": random.randint(30, 80),
            "timestamp": datetime.now().isoformat()
        }


class TodoTool(Tool):
    """待办事项管理工具"""
    
    def __init__(self):
        self.todos: List[Dict] = []
        self.next_id = 1
    
    @property
    def name(self) -> str:
        return "todo"
    
    @property
    def description(self) -> str:
        return "管理待办事项：添加、查看、完成、删除任务"
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """执行待办事项操作"""
        try:
            if action == "add":
                task = kwargs.get("task", "")
                if not task:
                    return {"success": False, "error": "任务内容不能为空"}
                
                todo_item = {
                    "id": self.next_id,
                    "task": task,
                    "completed": False,
                    "created_at": datetime.now().isoformat()
                }
                self.todos.append(todo_item)
                self.next_id += 1
                
                return {
                    "success": True,
                    "message": f"已添加任务: {task}",
                    "todo": todo_item
                }
            
            elif action == "list":
                return {
                    "success": True,
                    "todos": self.todos,
                    "count": len(self.todos)
                }
            
            elif action == "complete":
                task_id = kwargs.get("id")
                for todo in self.todos:
                    if todo["id"] == task_id:
                        todo["completed"] = True
                        todo["completed_at"] = datetime.now().isoformat()
                        return {
                            "success": True,
                            "message": f"任务 {task_id} 已完成"
                        }
                
                return {"success": False, "error": f"未找到任务 ID: {task_id}"}
            
            elif action == "delete":
                task_id = kwargs.get("id")
                for i, todo in enumerate(self.todos):
                    if todo["id"] == task_id:
                        deleted_todo = self.todos.pop(i)
                        return {
                            "success": True,
                            "message": f"已删除任务: {deleted_todo['task']}"
                        }
                
                return {"success": False, "error": f"未找到任务 ID: {task_id}"}
            
            else:
                return {"success": False, "error": f"不支持的操作: {action}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Message] = []
    
    def add_tool(self, tool: Tool) -> None:
        """添加工具"""
        self.tools[tool.name] = tool
    
    def log_message(self, sender: str, content: str, message_type: str = "text") -> None:
        """记录消息"""
        message = Message(
            sender=sender,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type
        )
        self.conversation_history.append(message)
    
    @abstractmethod
    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        pass
    
    def get_available_tools(self) -> str:
        """获取可用工具列表"""
        if not self.tools:
            return "当前没有可用工具"
        
        tool_info = []
        for tool in self.tools.values():
            tool_info.append(f"- {tool.name}: {tool.description}")
        
        return "可用工具:\n" + "\n".join(tool_info)


class TaskAgent(BaseAgent):
    """任务执行Agent"""
    
    def __init__(self, name: str = "TaskAgent"):
        super().__init__(name, "任务执行助手")
        # 添加默认工具
        self.add_tool(CalculatorTool())
        self.add_tool(WeatherTool())
        self.add_tool(TodoTool())
    
    def process_input(self, user_input: str) -> str:
        """处理用户输入并执行相应任务"""
        self.log_message("user", user_input)
        
        user_input = user_input.lower().strip()
        
        # 解析用户意图
        if "计算" in user_input or "算" in user_input:
            response = self._handle_calculation(user_input)
        elif "天气" in user_input:
            response = self._handle_weather_query(user_input)
        elif "待办" in user_input or "任务" in user_input or "todo" in user_input:
            response = self._handle_todo_operation(user_input)
        elif "工具" in user_input or "帮助" in user_input:
            response = self.get_available_tools()
        else:
            response = self._generate_general_response(user_input)
        
        self.log_message(self.name, response)
        return response
    
    def _handle_calculation(self, user_input: str) -> str:
        """处理计算请求"""
        # 简单的计算解析（实际应用中会更复杂）
        try:
            if "+" in user_input:
                parts = user_input.split("+")
                if len(parts) == 2:
                    a = float(parts[0].strip().split()[-1])
                    b = float(parts[1].strip().split()[0])
                    result = self.tools["calculator"].execute("add", a, b)
                    return f"计算结果: {result['operation']}"
            elif "*" in user_input:
                parts = user_input.split("*")
                if len(parts) == 2:
                    a = float(parts[0].strip().split()[-1])
                    b = float(parts[1].strip().split()[0])
                    result = self.tools["calculator"].execute("multiply", a, b)
                    return f"计算结果: {result['operation']}"
        except:
            pass
        
        return "请提供明确的计算表达式，例如: '计算 10 + 5' 或 '3 * 7'"
    
    def _handle_weather_query(self, user_input: str) -> str:
        """处理天气查询"""
        # 简单提取城市名称
        cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
        city = "北京"  # 默认城市
        
        for c in cities:
            if c in user_input:
                city = c
                break
        
        result = self.tools["weather"].execute(city)
        if result["success"]:
            return (f"{city}的天气情况:\n"
                   f"天气: {result['weather']}\n"
                   f"温度: {result['temperature']}°C\n"
                   f"湿度: {result['humidity']}%")
        else:
            return "天气查询失败"
    
    def _handle_todo_operation(self, user_input: str) -> str:
        """处理待办事项操作"""
        todo_tool = self.tools["todo"]
        
        if "添加" in user_input or "新增" in user_input:
            # 提取任务内容
            task = user_input.replace("添加", "").replace("新增", "").replace("任务", "").replace("待办", "").strip()
            if task:
                result = todo_tool.execute("add", task=task)
                return result.get("message", "任务添加失败")
            else:
                return "请提供要添加的任务内容，例如: '添加任务 学习Python'"
        
        elif "查看" in user_input or "列表" in user_input:
            result = todo_tool.execute("list")
            if result["success"] and result["todos"]:
                todo_list = []
                for todo in result["todos"]:
                    status = "✓" if todo["completed"] else "○"
                    todo_list.append(f"{status} [{todo['id']}] {todo['task']}")
                return "待办事项列表:\n" + "\n".join(todo_list)
            else:
                return "当前没有待办事项"
        
        elif "完成" in user_input:
            # 提取任务ID
            words = user_input.split()
            for word in words:
                if word.isdigit():
                    task_id = int(word)
                    result = todo_tool.execute("complete", id=task_id)
                    return result.get("message", "任务完成操作失败")
            return "请提供要完成的任务ID，例如: '完成任务 1'"
        
        else:
            return ("待办事项操作:\n"
                   "- 添加任务: '添加任务 [任务内容]'\n"
                   "- 查看列表: '查看待办'\n"
                   "- 完成任务: '完成任务 [任务ID]'")
    
    def _generate_general_response(self, user_input: str) -> str:
        """生成通用回复"""
        responses = [
            f"你好！我是{self.name}，一个{self.role}。",
            "我可以帮你进行计算、查询天气、管理待办事项等。",
            "请告诉我你需要什么帮助，或者输入'帮助'查看可用功能。"
        ]
        return random.choice(responses)


class ConversationAgent(BaseAgent):
    """对话Agent"""
    
    def __init__(self, name: str = "ConversationAgent"):
        super().__init__(name, "对话助手")
        self.context: Dict[str, Any] = {}
    
    def process_input(self, user_input: str) -> str:
        """处理对话输入"""
        self.log_message("user", user_input)
        
        # 更新上下文
        self.context["last_input"] = user_input
        self.context["input_count"] = self.context.get("input_count", 0) + 1
        
        # 生成回复
        response = self._generate_contextual_response(user_input)
        
        self.log_message(self.name, response)
        return response
    
    def _generate_contextual_response(self, user_input: str) -> str:
        """根据上下文生成回复"""
        user_input = user_input.lower().strip()
        
        # 问候处理
        greetings = ["你好", "hi", "hello", "早上好", "晚上好"]
        if any(greeting in user_input for greeting in greetings):
            return f"你好！我是{self.name}，很高兴与你交流！有什么可以帮助你的吗？"
        
        # 情感处理
        if "开心" in user_input or "高兴" in user_input:
            return "真为你高兴！开心的事情总是让人心情愉悦。"
        elif "难过" in user_input or "伤心" in user_input:
            return "我理解你的感受。有时候倾诉会让人感觉好一些。"
        elif "紧张" in user_input or "焦虑" in user_input:
            return "深呼吸，放松一下。很多事情没有想象中那么严重。"
        
        # 询问相关
        if "你是谁" in user_input or "自我介绍" in user_input:
            return f"我是{self.name}，一个{self.role}。我可以与你进行自然对话，倾听你的想法。"
        elif "你能做什么" in user_input:
            return "我可以与你聊天，倾听你的想法，提供情感支持，或者只是陪你度过一段时光。"
        
        # 默认回复
        contextual_responses = [
            "这很有趣，请继续告诉我更多。",
            "我明白了，这确实值得思考。",
            "感谢你与我分享这些想法。",
            "你的观点很独特，我很欣赏。",
            "这让我想到了很多东西。"
        ]
        
        return random.choice(contextual_responses)


class AgentOrchestrator:
    """Agent协调器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.current_agent: Optional[BaseAgent] = None
    
    def register_agent(self, agent: BaseAgent) -> None:
        """注册Agent"""
        self.agents[agent.name] = agent
        if not self.current_agent:
            self.current_agent = agent
    
    def switch_agent(self, agent_name: str) -> bool:
        """切换当前Agent"""
        if agent_name in self.agents:
            self.current_agent = self.agents[agent_name]
            return True
        return False
    
    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        if not self.current_agent:
            return "没有可用的Agent"
        
        # 检查是否需要切换Agent
        user_input_lower = user_input.lower()
        if "切换到任务助手" in user_input_lower or "taskagent" in user_input_lower:
            if self.switch_agent("TaskAgent"):
                return "已切换到任务助手，我可以帮你执行各种任务。"
        elif "切换到对话助手" in user_input_lower or "conversationagent" in user_input_lower:
            if self.switch_agent("ConversationAgent"):
                return "已切换到对话助手，我们可以愉快地聊天。"
        elif "显示agents" in user_input_lower or "agent列表" in user_input_lower:
            agent_list = []
            for name, agent in self.agents.items():
                current_mark = "(*)" if agent == self.current_agent else ""
                agent_list.append(f"- {name} {current_mark}: {agent.role}")
            return "可用的Agent:\n" + "\n".join(agent_list) + "\n\n输入'切换到 [Agent名称]'来切换Agent"
        
        # 使用当前Agent处理输入
        return self.current_agent.process_input(user_input)
    
    def get_current_agent_info(self) -> str:
        """获取当前Agent信息"""
        if self.current_agent:
            return f"当前Agent: {self.current_agent.name} ({self.current_agent.role})"
        return "没有激活的Agent"


def demo_agent_system():
    """演示Agent系统"""
    print("=" * 50)
    print("🤖 智能Agent系统Demo")
    print("=" * 50)
    
    # 创建Agent协调器
    orchestrator = AgentOrchestrator()
    
    # 创建并注册Agent
    task_agent = TaskAgent()
    conversation_agent = ConversationAgent()
    
    orchestrator.register_agent(task_agent)
    orchestrator.register_agent(conversation_agent)
    
    print(f"\n{orchestrator.get_current_agent_info()}")
    print("\n输入 'help' 获取帮助，输入 'quit' 退出")
    print("-" * 50)
    
    # 交互循环
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n👋 再见！感谢使用Agent系统！")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'help':
                print("""
💡 帮助信息:
- 输入任何消息与当前Agent交互
- '显示agents' - 查看所有可用的Agent
- '切换到任务助手' - 切换到TaskAgent
- '切换到对话助手' - 切换到ConversationAgent
- 'quit' - 退出程序

🔧 TaskAgent功能:
- 计算: '计算 10 + 5', '3 * 7'
- 天气: '查询北京天气'
- 待办: '添加任务 学习Python', '查看待办', '完成任务 1'

💬 ConversationAgent功能:
- 自然对话和情感交流
                """)
                continue
            
            # 处理用户输入
            response = orchestrator.process_input(user_input)
            print(f"\n🤖 {orchestrator.current_agent.name}: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def demo_individual_agents():
    """演示单个Agent"""
    print("\n" + "=" * 50)
    print("🧪 单个Agent测试")
    print("=" * 50)
    
    # 测试TaskAgent
    print("\n📋 TaskAgent测试:")
    task_agent = TaskAgent()
    
    test_inputs = [
        "计算 15 + 25",
        "查询上海天气",
        "添加任务 完成项目文档",
        "查看待办",
        "完成任务 1",
        "帮助"
    ]
    
    for test_input in test_inputs:
        print(f"输入: {test_input}")
        response = task_agent.process_input(test_input)
        print(f"回复: {response}\n")
    
    # 测试ConversationAgent
    print("\n💬 ConversationAgent测试:")
    conv_agent = ConversationAgent()
    
    conv_inputs = [
        "你好",
        "我今天很开心",
        "你能做什么",
        "我有点紧张"
    ]
    
    for conv_input in conv_inputs:
        print(f"输入: {conv_input}")
        response = conv_agent.process_input(conv_input)
        print(f"回复: {response}\n")


if __name__ == "__main__":
    print("🚀 启动Agent系统Demo...")
    
    # 演示单个Agent
    demo_individual_agents()
    
    # 演示完整系统
    demo_agent_system()