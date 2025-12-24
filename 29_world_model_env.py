"""
世界模型环境模拟器

包含两个环境：
1. SimpleGridWorld - 简单的网格世界
2. CarRacing (简化版) - 模拟赛车环境
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw


# ==================== GridWorld 环境 ====================
class SimpleGridWorld:
    """
    简单的网格世界环境
    
    - 状态: 64x64 RGB图像
    - 动作: 上(0), 下(1), 左(2), 右(3)
    - 目标: 智能体(蓝色)到达目标(绿色)，避开障碍物(红色)
    """
    
    def __init__(self, grid_size: int = 8, image_size: int = 64):
        self.grid_size = grid_size
        self.image_size = image_size
        self.cell_size = image_size // grid_size
        
        # 动作空间
        self.action_space = 4
        self.actions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机初始化位置
        self.agent_pos = [np.random.randint(0, self.grid_size), 
                          np.random.randint(0, self.grid_size)]
        
        # 目标位置
        while True:
            self.goal_pos = [np.random.randint(0, self.grid_size),
                            np.random.randint(0, self.grid_size)]
            if self.goal_pos != self.agent_pos:
                break
        
        # 障碍物
        self.obstacles = []
        num_obstacles = np.random.randint(2, 5)
        for _ in range(num_obstacles):
            while True:
                obs_pos = [np.random.randint(0, self.grid_size),
                          np.random.randint(0, self.grid_size)]
                if obs_pos != self.agent_pos and obs_pos != self.goal_pos:
                    self.obstacles.append(obs_pos)
                    break
        
        self.steps = 0
        self.max_steps = 100
        
        return self._render()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 移动智能体
        delta = self.actions[action]
        new_pos = [self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]]
        
        # 边界检查
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            # 障碍物检查
            if new_pos not in self.obstacles:
                self.agent_pos = new_pos
        
        # 计算奖励
        reward = -0.01  # 时间惩罚
        done = False
        
        # 到达目标
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        
        # 碰到障碍物
        if self.agent_pos in self.obstacles:
            reward = -1.0
            done = True
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        obs = self._render()
        info = {'steps': self.steps}
        
        return obs, reward, done, info
    
    def _render(self) -> np.ndarray:
        """渲染环境为RGB图像"""
        # 创建白色背景
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制网格线
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            draw.line([(pos, 0), (pos, self.image_size)], fill='lightgray', width=1)
            draw.line([(0, pos), (self.image_size, pos)], fill='lightgray', width=1)
        
        # 绘制障碍物 (红色)
        for obs_pos in self.obstacles:
            x, y = obs_pos[1] * self.cell_size, obs_pos[0] * self.cell_size
            draw.rectangle(
                [x + 2, y + 2, x + self.cell_size - 2, y + self.cell_size - 2],
                fill='red'
            )
        
        # 绘制目标 (绿色)
        x, y = self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size
        draw.rectangle(
            [x + 2, y + 2, x + self.cell_size - 2, y + self.cell_size - 2],
            fill='green'
        )
        
        # 绘制智能体 (蓝色圆形)
        x, y = self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size
        center_x, center_y = x + self.cell_size // 2, y + self.cell_size // 2
        radius = self.cell_size // 3
        draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            fill='blue'
        )
        
        # 转换为numpy数组并归一化
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 转换为 [C, H, W] 格式
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array


# ==================== 数据收集器 ====================
class DataCollector:
    """收集环境交互数据用于训练世界模型"""
    
    def __init__(self, env, device: str = "cpu"):
        self.env = env
        self.device = device
    
    def collect_random_episodes(
        self,
        num_episodes: int = 100,
        max_steps: int = 50
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        收集随机策略的轨迹数据
        
        Returns:
            observations: 所有观察的张量 [N, 3, 64, 64]
            sequences: 序列数据列表
        """
        print(f"\n收集 {num_episodes} 个随机轨迹...")
        
        all_observations = []
        sequences = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            for step in range(max_steps):
                # 随机动作
                action = np.random.randint(0, self.env.action_space)
                
                # 记录数据
                episode_data['observations'].append(obs)
                episode_data['actions'].append(action)
                
                all_observations.append(obs)
                
                # 环境交互
                obs, reward, done, _ = self.env.step(action)
                
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                
                if done:
                    break
            
            # 转换为张量
            sequence = {
                'observations': torch.FloatTensor(
                    np.array(episode_data['observations'])
                ).to(self.device),
                'actions': torch.FloatTensor(
                    self._one_hot_actions(episode_data['actions'])
                ).to(self.device),
                'rewards': torch.FloatTensor(episode_data['rewards']),
                'dones': torch.FloatTensor(episode_data['dones'])
            }
            
            sequences.append(sequence)
            
            if (ep + 1) % 20 == 0:
                print(f"  收集进度: {ep + 1}/{num_episodes}")
        
        # 合并所有观察
        observations = torch.FloatTensor(np.array(all_observations)).to(self.device)
        
        print(f"✓ 收集完成: {len(all_observations)} 个观察, {len(sequences)} 个序列")
        
        return observations, sequences
    
    def _one_hot_actions(self, actions: list) -> np.ndarray:
        """将动作转换为one-hot编码"""
        one_hot = np.zeros((len(actions), self.env.action_space))
        for i, action in enumerate(actions):
            one_hot[i, action] = 1.0
        return one_hot


# ==================== 简化的赛车环境 ====================
class SimpleCarRacing:
    """
    简化的赛车环境 (灵感来自 CarRacing-v0)
    
    小车在赛道上行驶，目标是沿着赛道前进
    """
    
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self.action_space = 4  # 左转, 右转, 加速, 刹车
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.car_x = self.image_size // 2
        self.car_y = self.image_size - 10
        self.car_angle = 0  # 向上
        self.speed = 0
        self.steps = 0
        self.max_steps = 200
        
        # 生成简单赛道
        self.track_center = self.image_size // 2
        self.track_width = 20
        
        return self._render()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 动作: 0=左转, 1=右转, 2=加速, 3=刹车
        if action == 0:  # 左转
            self.car_angle -= 5
        elif action == 1:  # 右转
            self.car_angle += 5
        elif action == 2:  # 加速
            self.speed = min(self.speed + 0.5, 5)
        elif action == 3:  # 刹车
            self.speed = max(self.speed - 0.5, 0)
        
        # 更新位置
        rad = np.radians(self.car_angle)
        self.car_x += self.speed * np.sin(rad)
        self.car_y -= self.speed * np.cos(rad)
        
        # 边界处理
        self.car_x = np.clip(self.car_x, 0, self.image_size)
        self.car_y = np.clip(self.car_y, 0, self.image_size)
        
        # 计算奖励
        reward = 0.0
        done = False
        
        # 在赛道上行驶
        if abs(self.car_x - self.track_center) < self.track_width:
            reward = self.speed * 0.1  # 速度奖励
        else:
            reward = -1.0  # 偏离赛道
            done = True
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        obs = self._render()
        info = {'steps': self.steps, 'speed': self.speed}
        
        return obs, reward, done, info
    
    def _render(self) -> np.ndarray:
        """渲染环境"""
        img = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        draw = ImageDraw.Draw(img)
        
        # 绘制赛道
        track_left = self.track_center - self.track_width
        track_right = self.track_center + self.track_width
        draw.rectangle(
            [track_left, 0, track_right, self.image_size],
            fill='darkgray'
        )
        
        # 赛道边线
        draw.line([(track_left, 0), (track_left, self.image_size)], fill='white', width=2)
        draw.line([(track_right, 0), (track_right, self.image_size)], fill='white', width=2)
        
        # 绘制小车 (红色矩形)
        car_size = 4
        draw.rectangle(
            [self.car_x - car_size, self.car_y - car_size,
             self.car_x + car_size, self.car_y + car_size],
            fill='red'
        )
        
        # 转换为数组
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array


if __name__ == "__main__":
    print("=" * 60)
    print("世界模型环境模拟器")
    print("=" * 60)
    
    # 测试 GridWorld
    print("\n1. 测试 SimpleGridWorld:")
    env = SimpleGridWorld()
    obs = env.reset()
    print(f"   观察空间: {obs.shape}")
    print(f"   动作空间: {env.action_space}")
    
    for i in range(5):
        action = np.random.randint(0, env.action_space)
        obs, reward, done, info = env.step(action)
        print(f"   步骤 {i+1}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
        if done:
            break
    
    # 测试数据收集
    print("\n2. 测试 DataCollector:")
    collector = DataCollector(env)
    observations, sequences = collector.collect_random_episodes(num_episodes=5, max_steps=10)
    print(f"   收集的观察: {observations.shape}")
    print(f"   收集的序列: {len(sequences)} 个")
    
    print("\n✓ 环境测试完成!")
