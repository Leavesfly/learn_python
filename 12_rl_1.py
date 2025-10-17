"""
强化学习入门教程 - 基础理论与概念
作者: Qoder AI Assistant
日期: 2025-09-16

本文件包含强化学习的基本概念、术语和理论基础
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Optional

class RLBasics:
    """
    强化学习基础概念演示类
    """
    
    def __init__(self):
        self.episodes_data = []
        
    def explain_concepts(self):
        """
        强化学习核心概念解释
        """
        concepts = {
            "智能体 (Agent)": "做决策的实体，如游戏中的玩家、机器人等",
            "环境 (Environment)": "智能体所处的世界，提供状态和奖励",
            "状态 (State)": "描述环境当前情况的信息",
            "动作 (Action)": "智能体可以执行的操作",
            "奖励 (Reward)": "环境对智能体动作的反馈信号",
            "策略 (Policy)": "智能体选择动作的规则，π(a|s)",
            "价值函数 (Value Function)": "评估状态或状态-动作对的好坏",
            "Q函数": "状态-动作价值函数，Q(s,a)",
            "探索vs利用": "探索新动作 vs 利用已知最优动作的权衡"
        }
        
        print("🎯 强化学习核心概念")
        print("=" * 50)
        for concept, explanation in concepts.items():
            print(f"📌 {concept}: {explanation}")
        print()
        
    def rl_process_flow(self):
        """
        强化学习过程流程图（文字版）
        """
        flow = """
        🔄 强化学习交互流程：
        
        1. 智能体观察当前状态 s_t
        2. 根据策略π选择动作 a_t
        3. 执行动作，环境给出新状态 s_{t+1} 和奖励 r_t
        4. 智能体更新策略/价值函数
        5. 重复步骤1-4
        
        目标：最大化累积奖励 G_t = Σ γ^k * r_{t+k+1}
        其中 γ 是折扣因子 (0 ≤ γ ≤ 1)
        """
        print(flow)

class SimpleGridWorld:
    """
    简单网格世界环境 - 用于演示强化学习基本概念
    """
    
    def __init__(self, size: int = 4):
        self.size = size
        self.reset()
        self.goal_state = (size-1, size-1)  # 右下角为目标
        self.obstacles = [(1, 1), (2, 2)]   # 障碍物位置
        
    def reset(self) -> Tuple[int, int]:
        """重置环境，返回初始状态"""
        self.agent_pos = (0, 0)  # 左上角开始
        return self.agent_pos
    
    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """获取当前状态下的有效动作"""
        valid_actions = []
        x, y = state
        
        # 0: 上, 1: 下, 2: 左, 3: 右
        if x > 0: valid_actions.append(0)  # 上
        if x < self.size - 1: valid_actions.append(1)  # 下
        if y > 0: valid_actions.append(2)  # 左
        if y < self.size - 1: valid_actions.append(3)  # 右
        
        return valid_actions
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作，返回新状态、奖励和是否结束
        """
        x, y = self.agent_pos
        
        # 根据动作移动
        if action == 0 and x > 0:  # 上
            x -= 1
        elif action == 1 and x < self.size - 1:  # 下
            x += 1
        elif action == 2 and y > 0:  # 左
            y -= 1
        elif action == 3 and y < self.size - 1:  # 右
            y += 1
        
        new_pos = (x, y)
        
        # 检查是否撞到障碍物
        if new_pos in self.obstacles:
            new_pos = self.agent_pos  # 不移动
            reward = -1.0
        else:
            self.agent_pos = new_pos
            # 奖励设计
            if new_pos == self.goal_state:
                reward = 10.0  # 到达目标
            else:
                reward = -0.1  # 每步小惩罚，鼓励快速到达目标
        
        done = (new_pos == self.goal_state)
        return new_pos, reward, done
    
    def render(self):
        """可视化当前状态"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = '█'
        
        # 标记目标
        grid[self.goal_state[0]][self.goal_state[1]] = 'G'
        
        # 标记智能体
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print("\n当前网格世界状态:")
        print("A: 智能体, G: 目标, █: 障碍物, .: 空地")
        for row in grid:
            print(' '.join(row))
        print()

def demo_random_policy():
    """
    演示随机策略在网格世界中的表现
    """
    print("🎮 随机策略演示")
    print("=" * 30)
    
    env = SimpleGridWorld()
    total_episodes = 5
    
    for episode in range(total_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n第 {episode + 1} 回合:")
        env.render()
        
        while steps < 50:  # 最大步数限制
            valid_actions = env.get_valid_actions(state)
            action = random.choice(valid_actions)
            
            action_names = ['上', '下', '左', '右']
            print(f"步骤 {steps + 1}: 执行动作 '{action_names[action]}'")
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            
            if done:
                print(f"🎉 成功到达目标! 总奖励: {total_reward:.1f}, 步数: {steps}")
                break
        
        if not done:
            print(f"❌ 未能到达目标，总奖励: {total_reward:.1f}")

def explain_value_functions():
    """
    解释价值函数的概念
    """
    explanation = """
    📊 价值函数详解
    
    1. 状态价值函数 V(s):
       - 定义: 从状态s开始，遵循策略π的期望累积奖励
       - 公式: V^π(s) = E[G_t | S_t = s]
       - 含义: 告诉我们在某个状态"有多好"
    
    2. 动作价值函数 Q(s,a):
       - 定义: 在状态s执行动作a，然后遵循策略π的期望累积奖励
       - 公式: Q^π(s,a) = E[G_t | S_t = s, A_t = a]
       - 含义: 告诉我们在某个状态执行某个动作"有多好"
    
    3. Bellman方程:
       - V(s) = Σ π(a|s) * Σ P(s'|s,a) * [R(s,a,s') + γ*V(s')]
       - Q(s,a) = Σ P(s'|s,a) * [R(s,a,s') + γ*Σ π(a'|s')*Q(s',a')]
    
    4. 最优价值函数:
       - V*(s) = max_π V^π(s)
       - Q*(s,a) = max_π Q^π(s,a)
    """
    print(explanation)

def main():
    """
    强化学习基础教程主函数
    """
    print("🚀 欢迎来到强化学习入门教程!")
    print("=" * 50)
    
    # 基础概念介绍
    rl_basics = RLBasics()
    rl_basics.explain_concepts()
    rl_basics.rl_process_flow()
    
    # 价值函数解释
    explain_value_functions()
    
    # 随机策略演示
    demo_random_policy()
    
    print("\n📚 学习建议:")
    print("1. 理解智能体与环境的交互过程")
    print("2. 掌握奖励设计的重要性")
    print("3. 理解探索与利用的权衡")
    print("4. 学习价值函数的概念")
    print("5. 继续学习具体算法 (Q-Learning, DQN等)")

if __name__ == "__main__":
    main()