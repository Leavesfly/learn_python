"""
Deep Q-Network (DQN) 实现
使用神经网络近似Q函数的深度强化学习算法

DQN的核心创新:
1. 使用深度神经网络近似Q函数
2. 经验回放 (Experience Replay)
3. 目标网络 (Target Network)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# 定义经验元组
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """
    Deep Q-Network 神经网络架构
    """
    
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        初始化DQN网络
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间大小
            hidden_size: 隐藏层大小
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, capacity):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """从缓冲区采样一批经验"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states =