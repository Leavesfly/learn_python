# -*- coding: utf-8 -*-
"""
Multi-Agent系统快速演示版本
展示核心功能：Agent通信、任务分配、协作机制
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AgentType(Enum):
    """Agent类型"""
    ANALYST = "分析师"
    RESEARCHER = "研究员" 
    COORDINATOR = "协调员"
    EXECUTOR = "执行员"


@dataclass
class Message:
    """消息类"""
    sender: str
    receiver: str
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Agent:
    """简化的Agent类"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, system):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = f"{agent_type.value}-{agent_id[-4:]}"
        self.system = system
        self.messages = []
        self.tasks_completed = 0
        self.conversation_count = {}  # 记录与每个Agent的对话次数
        self.max_conversations = 2    # 最大对话轮次
        
    def receive_message(self, message: Message):
        """接收消息"""
        self.messages.append(message)
        print(f"📨 {self.name} 收到消息: {message.content}")
        
        # 防止无限对话循环
        if message.content != "系统广播" and not message.content.startswith("团队任务"):
            # 检查对话次数
            sender_id = message.sender
            if sender_id not in self.conversation_count:
                self.conversation_count[sender_id] = 0
                
            if self.conversation_count[sender_id] < self.max_conversations:
                self.conversation_count[sender_id] += 1
                reply = self._generate_reply(message.content)
                self.send_message(message.sender, reply)
    
    def send_message(self, receiver_id: str, content: str):
        """发送消息"""
        message = Message(self.agent_id, receiver_id, content)
        self.system.deliver_message(message)
        print(f"📤 {self.name} 发送消息给 {receiver_id}: {content}")
    
    def _generate_reply(self, original_message: str) -> str:
        """生成回复"""
        replies = {
            AgentType.ANALYST: [
                "根据我的分析，这个问题需要深入的数据研究。",
                "我建议我们先收集更多的数据点来验证假设。",
                "从分析角度看，这个方向很有潜力。"
            ],
            AgentType.RESEARCHER: [
                "这个话题很有趣！我会查找相关的研究文献。",
                "基于现有研究，我认为我们应该关注以下几个方面...",
                "我可以提供一些学术界的最新发现。"
            ],
            AgentType.COORDINATOR: [
                "我来协调一下，让我们分工合作完成这个任务。",
                "根据大家的专长，我建议这样分配工作...",
                "让我们制定一个详细的执行计划。"
            ],
            AgentType.EXECUTOR: [
                "收到！我会立即开始执行这个任务。",
                "任务执行中，预计30分钟内完成。",
                "已完成任务，结果如下..."
            ]
        }
        
        type_replies = replies.get(self.agent_type, ["我明白了。"])
        reply_index = len(self.messages) % len(type_replies)
        return type_replies[reply_index]
    
    def execute_task(self, task_description: str):
        """执行任务"""
        print(f"⚙️ {self.name} 开始执行任务: {task_description}")
        self.tasks_completed += 1
        
        # 模拟任务执行时间
        import time
        time.sleep(0.5)
        
        result = f"任务'{task_description}'已完成，结果符合预期。"
        print(f"✅ {self.name} 完成任务: {result}")
        return result


class MultiAgentSystem:
    """多Agent系统"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.teams: Dict[str, List[str]] = {}
        self.message_history: List[Message] = []
        
    def add_agent(self, agent_type: AgentType) -> str:
        """添加Agent"""
        agent_id = f"{agent_type.name.lower()}_{len(self.agents)+1}"
        agent = Agent(agent_id, agent_type, self)
        self.agents[agent_id] = agent
        print(f"➕ 添加Agent: {agent.name} (ID: {agent_id})")
        return agent_id
    
    def create_team(self, team_name: str, agent_ids: List[str]):
        """创建团队"""
        self.teams[team_name] = agent_ids
        print(f"👥 创建团队 '{team_name}': {[self.agents[aid].name for aid in agent_ids]}")
    
    def deliver_message(self, message: Message):
        """传递消息"""
        self.message_history.append(message)
        if message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
        elif message.receiver == "broadcast":
            # 广播消息
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    agent.receive_message(message)
    
    def broadcast_message(self, sender_id: str, content: str):
        """广播消息"""
        message = Message(sender_id, "broadcast", content)
        self.deliver_message(message)
        print(f"📢 {self.agents[sender_id].name} 广播消息: {content}")
    
    def assign_task_to_team(self, team_name: str, task_description: str):
        """给团队分配任务"""
        if team_name not in self.teams:
            print(f"❌ 团队 '{team_name}' 不存在")
            return
            
        team_agents = self.teams[team_name]
        print(f"\n📋 给团队 '{team_name}' 分配任务: {task_description}")
        
        # 协调员分配子任务
        coordinator = None
        for agent_id in team_agents:
            if self.agents[agent_id].agent_type == AgentType.COORDINATOR:
                coordinator = self.agents[agent_id]
                break
        
        if coordinator:
            coordinator.send_message("broadcast", f"团队任务: {task_description}")
            
            # 给每个成员分配具体子任务
            subtasks = [
                "数据收集和初步分析",
                "深入研究和文献调研", 
                "方案设计和实施",
                "结果验证和质量检查"
            ]
            
            for i, agent_id in enumerate(team_agents):
                if agent_id != coordinator.agent_id and i < len(subtasks):
                    agent = self.agents[agent_id]
                    agent.execute_task(subtasks[i])
        else:
            # 没有协调员，直接分配给第一个Agent
            first_agent = self.agents[team_agents[0]]
            first_agent.execute_task(task_description)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "agents_count": len(self.agents),
            "teams_count": len(self.teams),
            "messages_count": len(self.message_history),
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "type": agent.agent_type.value,
                    "tasks_completed": agent.tasks_completed,
                    "messages_received": len(agent.messages)
                }
                for agent_id, agent in self.agents.items()
            }
        }


def demo_basic_communication():
    """演示基本通信"""
    print("=" * 60)
    print("🤖 Multi-Agent系统 - 基本通信演示")
    print("=" * 60)
    
    system = MultiAgentSystem()
    
    # 创建不同类型的Agent
    analyst_id = system.add_agent(AgentType.ANALYST)
    researcher_id = system.add_agent(AgentType.RESEARCHER)
    
    print("\n💬 Agent间点对点通信:")
    analyst = system.agents[analyst_id]
    analyst.send_message(researcher_id, "你好，我们合作分析一个项目如何？")
    
    print("\n📊 通信统计:")
    for agent_id, agent in system.agents.items():
        print(f"- {agent.name}: 收到 {len(agent.messages)} 条消息")


def demo_team_collaboration():
    """演示团队协作"""
    print("\n" + "=" * 60)
    print("👥 Multi-Agent系统 - 团队协作演示")
    print("=" * 60)
    
    system = MultiAgentSystem()
    
    # 创建完整团队
    coord_id = system.add_agent(AgentType.COORDINATOR)
    analyst_id = system.add_agent(AgentType.ANALYST)
    researcher_id = system.add_agent(AgentType.RESEARCHER)
    executor_id = system.add_agent(AgentType.EXECUTOR)
    
    # 创建团队
    team_agents = [coord_id, analyst_id, researcher_id, executor_id]
    system.create_team("AI研发团队", team_agents)
    
    print("\n📢 团队广播通信:")
    system.broadcast_message(coord_id, "欢迎大家加入AI研发团队！")
    
    print("\n📋 团队任务执行:")
    system.assign_task_to_team("AI研发团队", "开发智能客服系统")
    
    print("\n📈 最终统计:")
    status = system.get_system_status()
    print(f"- 系统中共有 {status['agents_count']} 个Agent")
    print(f"- 创建了 {status['teams_count']} 个团队")
    print(f"- 总共产生 {status['messages_count']} 条消息")
    
    print("\n👤 各Agent表现:")
    for agent_info in status['agents'].values():
        print(f"- {agent_info['name']}: 完成 {agent_info['tasks_completed']} 个任务, "
              f"收到 {agent_info['messages_received']} 条消息")


def demo_complex_scenario():
    """演示复杂协作场景"""
    print("\n" + "=" * 60)
    print("🚀 Multi-Agent系统 - 复杂协作场景")
    print("=" * 60)
    
    system = MultiAgentSystem()
    
    # 创建多个团队
    # 团队1: 研发团队
    coord1 = system.add_agent(AgentType.COORDINATOR)
    analyst1 = system.add_agent(AgentType.ANALYST)
    researcher1 = system.add_agent(AgentType.RESEARCHER)
    
    # 团队2: 执行团队
    coord2 = system.add_agent(AgentType.COORDINATOR)
    executor1 = system.add_agent(AgentType.EXECUTOR)
    executor2 = system.add_agent(AgentType.EXECUTOR)
    
    system.create_team("研发团队", [coord1, analyst1, researcher1])
    system.create_team("执行团队", [coord2, executor1, executor2])
    
    print("\n🔄 跨团队协作:")
    # 研发团队完成设计
    system.assign_task_to_team("研发团队", "设计新产品架构")
    
    # 团队间通信
    system.agents[coord1].send_message(coord2, "研发完成，请开始执行阶段")
    
    # 执行团队开始工作
    system.assign_task_to_team("执行团队", "实施新产品开发")
    
    print(f"\n🎯 协作成果:")
    status = system.get_system_status()
    total_tasks = sum(agent['tasks_completed'] for agent in status['agents'].values())
    total_messages = status['messages_count']
    
    print(f"- 两个团队协作完成了 {total_tasks} 个任务")
    print(f"- 团队间产生了 {total_messages} 次通信交互")
    print(f"- 实现了高效的跨团队协作机制")


def main():
    """主函数"""
    print("🌟 从零构建的基于LLM的Multi-Agent系统")
    print("展示了完整的多智能体架构和协作机制\n")
    
    # 运行各种演示
    demo_basic_communication()
    demo_team_collaboration()
    demo_complex_scenario()
    
    print("\n" + "=" * 60)
    print("🎉 所有演示完成！")
    print("\n💡 这个Multi-Agent系统的特点:")
    print("• ✨ 从零构建，无外部依赖")
    print("• ✨ 支持多种Agent类型和角色")
    print("• ✨ 灵活的消息通信机制")
    print("• ✨ 团队协作和任务分配")
    print("• ✨ 可扩展的架构设计")
    print("• ✨ 实时状态监控和统计")
    
    print("\n🔧 可以进一步扩展:")
    print("• 集成真实的LLM API")
    print("• 添加更多专业Agent类型")
    print("• 实现持久化存储")
    print("• 添加Web界面管理")
    print("• 集成外部工具和API")


if __name__ == "__main__":
    main()