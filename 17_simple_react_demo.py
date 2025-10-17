# -*- coding: utf-8 -*-
"""
ReAct Agent 简单演示
"""

import re
import random


class SimpleReActAgent:
    """简化的ReAct Agent演示"""
    
    def __init__(self):
        self.steps = []
        self.max_steps = 5
        
    def think(self, query, context=""):
        """思考步骤"""
        if "计算" in query or any(op in query for op in ['+', '-', '*', '/']):
            return "这是一个数学问题，我需要进行计算"
        elif "什么是" in query or "介绍" in query:
            return "这是一个知识问答问题，我需要提供准确信息"
        else:
            return f"我需要仔细分析这个问题：{query}"
    
    def parse_action(self, thought):
        """从思考中解析出行动"""
        if "计算" in thought:
            return "calculate"
        elif "知识" in thought:
            return "search_knowledge"
        else:
            return "analyze"
    
    def execute_action(self, action, query):
        """执行行动"""
        if action == "calculate":
            # 简单的数学计算
            numbers = re.findall(r'\d+', query)
            if len(numbers) >= 2:
                a, b = int(numbers[0]), int(numbers[1])
                if '+' in query:
                    return f"{a} + {b} = {a + b}"
                elif '*' in query:
                    return f"{a} * {b} = {a * b}"
                elif '-' in query:
                    return f"{a} - {b} = {a - b}"
                elif '/' in query and b != 0:
                    return f"{a} / {b} = {a / b}"
            return "计算结果：无法解析数学表达式"
        
        elif action == "search_knowledge":
            knowledge_base = {
                "python": "Python是一种高级编程语言，简洁易学，广泛用于数据科学和AI开发",
                "ai": "人工智能(AI)是让机器能够模拟人类智能的技术",
                "机器学习": "机器学习是AI的一个分支，让计算机从数据中学习模式"
            }
            
            for key in knowledge_base:
                if key in query.lower():
                    return f"知识查询结果：{knowledge_base[key]}"
            
            return "知识查询结果：未找到相关信息"
        
        else:
            return f"分析结果：'{query}' 是一个需要进一步处理的问题"
    
    def is_complete(self, observation):
        """判断是否完成"""
        return "结果" in observation or "=" in observation
    
    def process(self, query):
        """主处理流程"""
        print(f"\n🔍 处理查询: {query}")
        print("=" * 50)
        
        self.steps = []
        context = ""
        
        for step_num in range(1, self.max_steps + 1):
            print(f"\n📝 第{step_num}步:")
            
            # 1. 思考
            thought = self.think(query, context)
            print(f"💭 思考: {thought}")
            self.steps.append(f"思考: {thought}")
            
            # 2. 行动
            action = self.parse_action(thought)
            print(f"🎯 行动: {action}")
            self.steps.append(f"行动: {action}")
            
            # 3. 执行并观察
            observation = self.execute_action(action, query)
            print(f"👀 观察: {observation}")
            self.steps.append(f"观察: {observation}")
            
            context += f" {observation}"
            
            # 4. 判断是否完成
            if self.is_complete(observation):
                final_answer = f"基于我的分析和行动，{observation}"
                print(f"\n✅ 最终答案: {final_answer}")
                return final_answer
        
        return "抱歉，我无法在限定步骤内完成这个任务"


def demo_react_agent():
    """演示ReAct Agent"""
    print("🤖 ReAct Agent 演示")
    print("=" * 60)
    
    agent = SimpleReActAgent()
    
    test_queries = [
        "计算 15 * 8",
        "什么是Python",
        "介绍机器学习",
        "25 + 17 等于多少"
    ]
    
    for query in test_queries:
        result = agent.process(query)
        
        print(f"\n📋 执行步骤总结:")
        for i, step in enumerate(agent.steps, 1):
            print(f"  {i}. {step}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    demo_react_agent()