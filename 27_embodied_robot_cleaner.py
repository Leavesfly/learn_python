"""
具身智能扫地机器人系统 (Embodied Intelligence Robot Cleaner)

这是一个完整的具身智能系统实现，展示了：
1. 环境感知 (Perception): 多传感器融合
2. 决策制定 (Decision Making): 端到端深度强化学习
3. 动作执行 (Action Execution): 运动控制和清扫动作
4. 学习适应 (Learning & Adaptation): 在线学习和策略优化

具身智能的核心概念：
- 感知-行动循环 (Perception-Action Loop)
- 端到端学习 (End-to-End Learning)
- 环境交互 (Environment Interaction)
- 体验式学习 (Experiential Learning)
"""

import numpy as np
import random
import json
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass, asdict
import time


# ==================== 环境模拟模块 ====================

@dataclass
class GridCell:
    """网格单元"""
    x: int
    y: int
    has_obstacle: bool = False  # 是否有障碍物
    dust_level: float = 0.0  # 灰尘程度 (0-1)
    visited_count: int = 0  # 访问次数
    
    def __hash__(self):
        return hash((self.x, self.y))


class RoomEnvironment:
    """房间环境模拟器"""
    
    def __init__(self, width: int = 10, height: int = 10, obstacle_ratio: float = 0.15):
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = []
        self.total_dust = 0.0
        self.cleaned_dust = 0.0
        
        # 初始化网格
        self._initialize_grid()
        # 添加障碍物
        self._add_obstacles(obstacle_ratio)
        # 分布灰尘
        self._distribute_dust()
        
    def _initialize_grid(self):
        """初始化网格"""
        self.grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(GridCell(x=x, y=y))
            self.grid.append(row)
    
    def _add_obstacles(self, ratio: float):
        """添加障碍物"""
        num_obstacles = int(self.width * self.height * ratio)
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        
        # 排除起始位置
        positions = [(x, y) for x, y in positions if not (x == 0 and y == 0)]
        
        obstacle_positions = random.sample(positions, min(num_obstacles, len(positions)))
        for x, y in obstacle_positions:
            self.grid[y][x].has_obstacle = True
    
    def _distribute_dust(self):
        """分布灰尘"""
        self.total_dust = 0.0
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x].has_obstacle:
                    # 灰尘分布不均匀，模拟真实环境
                    dust = random.uniform(0.3, 1.0)
                    self.grid[y][x].dust_level = dust
                    self.total_dust += dust
    
    def get_cell(self, x: int, y: int) -> Optional[GridCell]:
        """获取单元格"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效（在边界内且无障碍）"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return not self.grid[y][x].has_obstacle
    
    def clean_cell(self, x: int, y: int, efficiency: float = 0.8) -> float:
        """清扫单元格，返回清理的灰尘量"""
        cell = self.get_cell(x, y)
        if cell and not cell.has_obstacle:
            cleaned = cell.dust_level * efficiency
            cell.dust_level = max(0, cell.dust_level - cleaned)
            self.cleaned_dust += cleaned
            return cleaned
        return 0.0
    
    def get_cleanliness_ratio(self) -> float:
        """获取清洁度比例"""
        if self.total_dust == 0:
            return 1.0
        return self.cleaned_dust / self.total_dust
    
    def visualize(self, robot_pos: Tuple[int, int] = None) -> str:
        """可视化环境"""
        lines = []
        lines.append("=" * (self.width * 4 + 1))
        for y in range(self.height):
            line = "|"
            for x in range(self.width):
                cell = self.grid[y][x]
                if robot_pos and robot_pos == (x, y):
                    line += " R |"  # 机器人位置
                elif cell.has_obstacle:
                    line += " # |"  # 障碍物
                elif cell.dust_level > 0.7:
                    line += " ··|"  # 高灰尘
                elif cell.dust_level > 0.3:
                    line += " · |"  # 中等灰尘
                elif cell.dust_level > 0.05:
                    line += " . |"  # 低灰尘
                else:
                    line += "   |"  # 干净
            lines.append(line)
            lines.append("-" * (self.width * 4 + 1))
        return "\n".join(lines)


# ==================== 感知模块 ====================

@dataclass
class SensorData:
    """传感器数据"""
    lidar_readings: List[float]  # 激光雷达读数（8个方向）
    dust_sensor: float  # 当前位置灰尘传感器
    position: Tuple[int, int]  # 位置信息
    battery_level: float  # 电池电量
    local_map: List[List[float]]  # 局部地图（5x5区域）


class PerceptionSystem:
    """感知系统 - 多传感器融合"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        self.sensor_range = 2  # 传感器范围
        
        # 8个方向的激光雷达
        self.lidar_directions = [
            (0, -1),   # 北
            (1, -1),   # 东北
            (1, 0),    # 东
            (1, 1),    # 东南
            (0, 1),    # 南
            (-1, 1),   # 西南
            (-1, 0),   # 西
            (-1, -1),  # 西北
        ]
    
    def sense(self, robot_x: int, robot_y: int, battery: float) -> SensorData:
        """执行传感器感知"""
        # 激光雷达读数 - 检测障碍物距离
        lidar_readings = self._get_lidar_readings(robot_x, robot_y)
        
        # 灰尘传感器
        cell = self.environment.get_cell(robot_x, robot_y)
        dust_sensor = cell.dust_level if cell else 0.0
        
        # 局部地图构建
        local_map = self._build_local_map(robot_x, robot_y)
        
        return SensorData(
            lidar_readings=lidar_readings,
            dust_sensor=dust_sensor,
            position=(robot_x, robot_y),
            battery_level=battery,
            local_map=local_map
        )
    
    def _get_lidar_readings(self, x: int, y: int) -> List[float]:
        """获取8个方向的激光雷达读数"""
        readings = []
        max_range = 5
        
        for dx, dy in self.lidar_directions:
            distance = 0
            for step in range(1, max_range + 1):
                check_x = x + dx * step
                check_y = y + dy * step
                
                # 超出边界
                if not (0 <= check_x < self.environment.width and 
                        0 <= check_y < self.environment.height):
                    distance = step
                    break
                
                # 检测到障碍物
                cell = self.environment.get_cell(check_x, check_y)
                if cell and cell.has_obstacle:
                    distance = step
                    break
            
            # 归一化到 [0, 1]
            normalized_distance = distance / max_range if distance > 0 else 1.0
            readings.append(normalized_distance)
        
        return readings
    
    def _build_local_map(self, x: int, y: int) -> List[List[float]]:
        """构建局部地图（5x5区域）"""
        local_map = []
        for dy in range(-2, 3):
            row = []
            for dx in range(-2, 3):
                map_x, map_y = x + dx, y + dy
                cell = self.environment.get_cell(map_x, map_y)
                
                if cell is None:
                    row.append(-1.0)  # 边界外
                elif cell.has_obstacle:
                    row.append(-0.5)  # 障碍物
                else:
                    row.append(cell.dust_level)  # 灰尘水平
            local_map.append(row)
        
        return local_map
    
    def encode_state(self, sensor_data: SensorData) -> np.ndarray:
        """将传感器数据编码为神经网络输入"""
        state_vector = []
        
        # 1. 激光雷达数据 (8维)
        state_vector.extend(sensor_data.lidar_readings)
        
        # 2. 当前位置灰尘传感器 (1维)
        state_vector.append(sensor_data.dust_sensor)
        
        # 3. 归一化位置信息 (2维)
        state_vector.append(sensor_data.position[0] / self.environment.width)
        state_vector.append(sensor_data.position[1] / self.environment.height)
        
        # 4. 电池电量 (1维)
        state_vector.append(sensor_data.battery_level)
        
        # 5. 局部地图展平 (25维)
        for row in sensor_data.local_map:
            state_vector.extend(row)
        
        return np.array(state_vector, dtype=np.float32)


# ==================== 决策模块 ====================

class NeuralNetwork:
    """简单的神经网络（用于策略网络）"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 第一层
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        # 第二层
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)  # ReLU
        # 输出层
        output = np.dot(h2, self.W3) + self.b3
        return output
    
    def copy_from(self, other: 'NeuralNetwork'):
        """复制权重"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class DQNAgent:
    """深度Q网络智能体 - 端到端学习决策"""
    
    # 动作空间定义
    ACTIONS = {
        0: "MOVE_NORTH",     # 向北移动
        1: "MOVE_EAST",      # 向东移动
        2: "MOVE_SOUTH",     # 向南移动
        3: "MOVE_WEST",      # 向西移动
        4: "CLEAN",          # 清扫当前位置
        5: "ROTATE_SCAN",    # 旋转扫描周围环境
    }
    
    ACTION_VECTORS = {
        0: (0, -1),   # 北
        1: (1, 0),    # 东
        2: (0, 1),    # 南
        3: (-1, 0),   # 西
    }
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = len(self.ACTIONS)
        
        # Q网络和目标Q网络
        self.q_network = NeuralNetwork(state_dim, hidden_dim, self.action_dim)
        self.target_network = NeuralNetwork(state_dim, hidden_dim, self.action_dim)
        self.target_network.copy_from(self.q_network)
        
        # 超参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # 训练统计
        self.training_steps = 0
        self.update_target_freq = 100  # 目标网络更新频率
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（ε-贪婪策略）"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # 贪婪策略
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            # 计算目标Q值
            if done:
                target_q = reward
            else:
                next_q_values = self.target_network.forward(next_state)
                target_q = reward + self.gamma * np.max(next_q_values)
            
            # 当前Q值
            current_q_values = self.q_network.forward(state)
            target_q_values = current_q_values.copy()
            target_q_values[action] = target_q
            
            # 简化的梯度下降（完整实现需要反向传播）
            loss = self._update_network(state, target_q_values)
            total_loss += loss
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 更新目标网络
        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        return total_loss / self.batch_size
    
    def _update_network(self, state: np.ndarray, target_q: np.ndarray) -> float:
        """更新网络权重（简化版梯度下降）"""
        # 前向传播
        current_q = self.q_network.forward(state)
        
        # 计算损失
        loss = np.mean((current_q - target_q) ** 2)
        
        # 简化的权重更新（实际应使用完整的反向传播）
        gradient = 2 * (current_q - target_q) / len(current_q)
        
        # 更新输出层权重
        h2 = np.maximum(0, np.dot(
            np.maximum(0, np.dot(state, self.q_network.W1) + self.q_network.b1),
            self.q_network.W2
        ) + self.q_network.b2)
        
        self.q_network.W3 -= self.learning_rate * np.outer(h2, gradient)
        self.q_network.b3 -= self.learning_rate * gradient
        
        return loss


# ==================== 动作执行模块 ====================

class RobotActuator:
    """机器人执行器 - 动作执行系统"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        self.cleaning_efficiency = 0.8  # 清洁效率
        
    def execute_move(self, current_pos: Tuple[int, int], action_id: int) -> Tuple[Tuple[int, int], bool]:
        """执行移动动作"""
        if action_id not in DQNAgent.ACTION_VECTORS:
            return current_pos, False
        
        dx, dy = DQNAgent.ACTION_VECTORS[action_id]
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        
        # 检查新位置是否有效
        if self.environment.is_valid_position(new_x, new_y):
            return (new_x, new_y), True
        
        return current_pos, False  # 撞墙或障碍物
    
    def execute_clean(self, position: Tuple[int, int]) -> float:
        """执行清扫动作"""
        x, y = position
        cleaned_amount = self.environment.clean_cell(x, y, self.cleaning_efficiency)
        return cleaned_amount
    
    def execute_scan(self, position: Tuple[int, int]) -> Dict:
        """执行扫描动作"""
        # 模拟360度扫描，更新周围环境信息
        scan_data = {
            "position": position,
            "surroundings": []
        }
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = position[0] + dx, position[1] + dy
                cell = self.environment.get_cell(x, y)
                if cell:
                    scan_data["surroundings"].append({
                        "pos": (x, y),
                        "dust": cell.dust_level,
                        "obstacle": cell.has_obstacle
                    })
        
        return scan_data


# ==================== 完整的具身智能体 ====================

@dataclass
class RobotState:
    """机器人状态"""
    position: Tuple[int, int]
    battery: float
    total_cleaned: float
    steps: int
    collisions: int


class EmbodiedRobotCleaner:
    """具身智能扫地机器人 - 完整系统"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        
        # 感知系统
        self.perception = PerceptionSystem(environment)
        
        # 决策系统（DQN智能体）
        state_dim = 37  # 8(lidar) + 1(dust) + 2(pos) + 1(battery) + 25(local_map)
        self.agent = DQNAgent(state_dim=state_dim, hidden_dim=128)
        
        # 执行系统
        self.actuator = RobotActuator(environment)
        
        # 机器人状态
        self.state = RobotState(
            position=(0, 0),
            battery=100.0,
            total_cleaned=0.0,
            steps=0,
            collisions=0
        )
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cleanliness = []
    
    def reset(self):
        """重置机器人状态"""
        self.state = RobotState(
            position=(0, 0),
            battery=100.0,
            total_cleaned=0.0,
            steps=0,
            collisions=0
        )
        # 重新分布灰尘
        self.environment._distribute_dust()
        self.environment.cleaned_dust = 0.0
    
    def step(self, training: bool = True) -> Tuple[float, bool]:
        """执行一步感知-决策-执行循环"""
        # 1. 感知环境
        sensor_data = self.perception.sense(
            self.state.position[0],
            self.state.position[1],
            self.state.battery / 100.0
        )
        
        # 2. 编码状态
        state_vector = self.perception.encode_state(sensor_data)
        
        # 3. 决策（选择动作）
        action = self.agent.select_action(state_vector, training=training)
        
        # 4. 执行动作
        reward, done = self._execute_action(action)
        
        # 5. 感知新状态
        new_sensor_data = self.perception.sense(
            self.state.position[0],
            self.state.position[1],
            self.state.battery / 100.0
        )
        new_state_vector = self.perception.encode_state(new_sensor_data)
        
        # 6. 存储经验（用于训练）
        if training:
            self.agent.store_experience(
                state_vector, action, reward, new_state_vector, done
            )
        
        return reward, done
    
    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """执行动作并计算奖励"""
        reward = 0.0
        done = False
        
        self.state.steps += 1
        
        # 移动动作
        if action in [0, 1, 2, 3]:
            new_pos, success = self.actuator.execute_move(self.state.position, action)
            
            if success:
                self.state.position = new_pos
                reward += 0.1  # 成功移动奖励
                
                # 访问新区域奖励
                cell = self.environment.get_cell(new_pos[0], new_pos[1])
                if cell:
                    cell.visited_count += 1
                    if cell.visited_count == 1:
                        reward += 0.5  # 首次访问奖励
            else:
                self.state.collisions += 1
                reward -= 1.0  # 碰撞惩罚
        
        # 清扫动作
        elif action == 4:
            cleaned = self.actuator.execute_clean(self.state.position)
            self.state.total_cleaned += cleaned
            reward += cleaned * 10.0  # 清扫奖励与清理量成正比
            
            if cleaned < 0.01:
                reward -= 0.5  # 清扫已经干净的地方有惩罚
        
        # 扫描动作
        elif action == 5:
            self.actuator.execute_scan(self.state.position)
            reward += 0.05  # 小奖励鼓励探索
        
        # 电池消耗
        battery_cost = 0.1
        self.state.battery -= battery_cost
        
        # 检查结束条件
        if self.state.battery <= 0:
            reward -= 10.0  # 电池耗尽惩罚
            done = True
        elif self.state.steps >= 500:  # 最大步数
            done = True
        elif self.environment.get_cleanliness_ratio() > 0.95:  # 清洁度达标
            reward += 50.0  # 完成任务大奖励
            done = True
        
        # 时间惩罚（鼓励快速完成）
        reward -= 0.01
        
        return reward, done
    
    def train_episode(self) -> Dict:
        """训练一个回合"""
        self.reset()
        total_reward = 0.0
        losses = []
        
        while True:
            reward, done = self.step(training=True)
            total_reward += reward
            
            # 训练网络
            if self.state.steps % 4 == 0:  # 每4步训练一次
                loss = self.agent.train()
                if loss > 0:
                    losses.append(loss)
            
            if done:
                break
        
        # 记录统计信息
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(self.state.steps)
        self.episode_cleanliness.append(self.environment.get_cleanliness_ratio())
        
        return {
            "reward": total_reward,
            "steps": self.state.steps,
            "cleanliness": self.environment.get_cleanliness_ratio(),
            "cleaned_amount": self.state.total_cleaned,
            "collisions": self.state.collisions,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "epsilon": self.agent.epsilon
        }
    
    def evaluate(self, verbose: bool = False) -> Dict:
        """评估当前策略"""
        self.reset()
        total_reward = 0.0
        
        if verbose:
            print("=== 评估开始 ===")
            print(self.environment.visualize(self.state.position))
        
        while True:
            reward, done = self.step(training=False)
            total_reward += reward
            
            if verbose and self.state.steps % 20 == 0:
                print(f"\n步数: {self.state.steps}, 位置: {self.state.position}, "
                      f"电池: {self.state.battery:.1f}%, "
                      f"清洁度: {self.environment.get_cleanliness_ratio():.2%}")
                print(self.environment.visualize(self.state.position))
            
            if done:
                break
        
        if verbose:
            print("\n=== 评估结束 ===")
            print(f"总步数: {self.state.steps}")
            print(f"总奖励: {total_reward:.2f}")
            print(f"清洁度: {self.environment.get_cleanliness_ratio():.2%}")
            print(f"碰撞次数: {self.state.collisions}")
        
        return {
            "reward": total_reward,
            "steps": self.state.steps,
            "cleanliness": self.environment.get_cleanliness_ratio(),
            "cleaned_amount": self.state.total_cleaned,
            "collisions": self.state.collisions
        }


# ==================== 训练与可视化 ====================

def train_embodied_robot(episodes: int = 200, eval_interval: int = 20):
    """训练具身智能机器人"""
    print("=" * 60)
    print("具身智能扫地机器人训练系统")
    print("=" * 60)
    
    # 创建环境和机器人
    env = RoomEnvironment(width=10, height=10, obstacle_ratio=0.15)
    robot = EmbodiedRobotCleaner(env)
    
    print(f"\n环境设置:")
    print(f"  - 房间大小: {env.width}x{env.height}")
    print(f"  - 障碍物数量: {sum(1 for row in env.grid for cell in row if cell.has_obstacle)}")
    print(f"  - 总灰尘量: {env.total_dust:.2f}")
    print(f"\n开始训练 {episodes} 个回合...\n")
    
    training_history = []
    best_cleanliness = 0.0
    
    for episode in range(1, episodes + 1):
        # 训练一个回合
        stats = robot.train_episode()
        training_history.append(stats)
        
        # 更新最佳成绩
        if stats["cleanliness"] > best_cleanliness:
            best_cleanliness = stats["cleanliness"]
        
        # 打印进度
        if episode % 10 == 0:
            recent_stats = training_history[-10:]
            avg_reward = np.mean([s["reward"] for s in recent_stats])
            avg_clean = np.mean([s["cleanliness"] for s in recent_stats])
            avg_steps = np.mean([s["steps"] for s in recent_stats])
            
            print(f"Episode {episode:3d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Clean: {avg_clean:.2%} | "
                  f"Steps: {avg_steps:.0f} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Best: {best_cleanliness:.2%}")
        
        # 定期评估
        if episode % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"回合 {episode} 评估:")
            eval_stats = robot.evaluate(verbose=True)
            print(f"{'='*60}\n")
    
    # 保存训练历史
    with open("/Users/yefei.yf/Qoder/learn_python/embodied_robot_training.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳清洁度: {best_cleanliness:.2%}")
    print(f"训练历史已保存到: embodied_robot_training.json")
    print(f"{'='*60}")
    
    return robot, training_history


def visualize_training_results(history: List[Dict]):
    """可视化训练结果"""
    print("\n" + "=" * 60)
    print("训练结果分析")
    print("=" * 60)
    
    episodes = len(history)
    
    # 计算移动平均
    window = 10
    rewards = [h["reward"] for h in history]
    cleanliness = [h["cleanliness"] for h in history]
    steps = [h["steps"] for h in history]
    
    if episodes >= window:
        avg_rewards = []
        avg_clean = []
        avg_steps = []
        
        for i in range(episodes - window + 1):
            avg_rewards.append(np.mean(rewards[i:i+window]))
            avg_clean.append(np.mean(cleanliness[i:i+window]))
            avg_steps.append(np.mean(steps[i:i+window]))
        
        print(f"\n移动平均 (窗口={window}):")
        print(f"  前{window}回合 -> 后{window}回合")
        print(f"  奖励: {avg_rewards[0]:.2f} -> {avg_rewards[-1]:.2f} "
              f"(提升: {avg_rewards[-1]-avg_rewards[0]:+.2f})")
        print(f"  清洁度: {avg_clean[0]:.2%} -> {avg_clean[-1]:.2%} "
              f"(提升: {avg_clean[-1]-avg_clean[0]:+.2%})")
        print(f"  步数: {avg_steps[0]:.0f} -> {avg_steps[-1]:.0f} "
              f"(变化: {avg_steps[-1]-avg_steps[0]:+.0f})")
    
    # 整体统计
    print(f"\n整体统计:")
    print(f"  平均奖励: {np.mean(rewards):.2f} (std: {np.std(rewards):.2f})")
    print(f"  平均清洁度: {np.mean(cleanliness):.2%} (std: {np.std(cleanliness):.2%})")
    print(f"  平均步数: {np.mean(steps):.0f} (std: {np.std(steps):.0f})")
    print(f"  最高清洁度: {max(cleanliness):.2%}")
    print(f"  最低碰撞: {min(h['collisions'] for h in history)}")


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    np.random.seed(42)
    random.seed(42)
    
    # 训练机器人
    robot, history = train_embodied_robot(episodes=200, eval_interval=50)
    
    # 可视化结果
    visualize_training_results(history)
    
    # 最终演示
    print("\n" + "=" * 60)
    print("最终策略演示")
    print("=" * 60)
    robot.evaluate(verbose=True)
