"""
具身智能扫地机器人系统 - 简化演示版
(不依赖numpy，使用纯Python实现)

展示具身智能的核心概念：
1. 环境感知 (Perception)
2. 决策制定 (Decision Making) 
3. 动作执行 (Action Execution)
4. 学习适应 (Learning & Adaptation)
"""

import random
import json
import math
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass, asdict


# ==================== 环境模拟模块 ====================

@dataclass
class GridCell:
    """网格单元"""
    x: int
    y: int
    has_obstacle: bool = False
    dust_level: float = 0.0
    visited_count: int = 0


class RoomEnvironment:
    """房间环境模拟器"""
    
    def __init__(self, width: int = 10, height: int = 10, obstacle_ratio: float = 0.15):
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = []
        self.total_dust = 0.0
        self.cleaned_dust = 0.0
        
        self._initialize_grid()
        self._add_obstacles(obstacle_ratio)
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
                    dust = random.uniform(0.3, 1.0)
                    self.grid[y][x].dust_level = dust
                    self.total_dust += dust
    
    def get_cell(self, x: int, y: int) -> Optional[GridCell]:
        """获取单元格"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return not self.grid[y][x].has_obstacle
    
    def clean_cell(self, x: int, y: int, efficiency: float = 0.8) -> float:
        """清扫单元格"""
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
    
    def visualize(self, robot_pos: Optional[Tuple[int, int]] = None) -> str:
        """可视化环境"""
        lines = []
        lines.append("=" * (self.width * 4 + 1))
        for y in range(self.height):
            line = "|"
            for x in range(self.width):
                cell = self.grid[y][x]
                if robot_pos and robot_pos == (x, y):
                    line += " R |"
                elif cell.has_obstacle:
                    line += " # |"
                elif cell.dust_level > 0.7:
                    line += " ··|"
                elif cell.dust_level > 0.3:
                    line += " · |"
                elif cell.dust_level > 0.05:
                    line += " . |"
                else:
                    line += "   |"
            lines.append(line)
            lines.append("-" * (self.width * 4 + 1))
        return "\n".join(lines)


# ==================== 感知模块 ====================

@dataclass
class SensorData:
    """传感器数据"""
    lidar_readings: List[float]
    dust_sensor: float
    position: Tuple[int, int]
    battery_level: float
    local_map: List[List[float]]


class PerceptionSystem:
    """感知系统 - 多传感器融合"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        self.lidar_directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]
    
    def sense(self, robot_x: int, robot_y: int, battery: float) -> SensorData:
        """执行传感器感知"""
        lidar_readings = self._get_lidar_readings(robot_x, robot_y)
        
        cell = self.environment.get_cell(robot_x, robot_y)
        dust_sensor = cell.dust_level if cell else 0.0
        
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
                
                if not (0 <= check_x < self.environment.width and 
                        0 <= check_y < self.environment.height):
                    distance = step
                    break
                
                cell = self.environment.get_cell(check_x, check_y)
                if cell and cell.has_obstacle:
                    distance = step
                    break
            
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
                    row.append(-1.0)
                elif cell.has_obstacle:
                    row.append(-0.5)
                else:
                    row.append(cell.dust_level)
            local_map.append(row)
        
        return local_map
    
    def encode_state(self, sensor_data: SensorData) -> List[float]:
        """将传感器数据编码为状态向量"""
        state_vector = []
        
        # 激光雷达数据 (8维)
        state_vector.extend(sensor_data.lidar_readings)
        
        # 当前位置灰尘传感器 (1维)
        state_vector.append(sensor_data.dust_sensor)
        
        # 归一化位置信息 (2维)
        state_vector.append(sensor_data.position[0] / self.environment.width)
        state_vector.append(sensor_data.position[1] / self.environment.height)
        
        # 电池电量 (1维)
        state_vector.append(sensor_data.battery_level)
        
        # 局部地图展平 (25维)
        for row in sensor_data.local_map:
            state_vector.extend(row)
        
        return state_vector


# ==================== 决策模块 ====================

class SimpleQTable:
    """简化的Q表（基于状态离散化）"""
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.q_table = {}  # 状态 -> Q值列表
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
    
    def _discretize_state(self, state: List[float]) -> tuple:
        """将连续状态离散化为元组（用作字典键）"""
        # 简化：只使用关键特征
        pos_x = int(state[9] * 10)  # 归一化位置x
        pos_y = int(state[10] * 10)  # 归一化位置y
        dust = int(state[8] * 5)  # 当前灰尘水平
        battery = int(state[11] * 4)  # 电池水平
        
        # 前方障碍物检测（4个主方向）
        obstacles = tuple([1 if state[i] < 0.3 else 0 for i in [0, 2, 4, 6]])
        
        return (pos_x, pos_y, dust, battery, obstacles)
    
    def get_q_values(self, state: List[float]) -> List[float]:
        """获取Q值"""
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions
        return self.q_table[state_key]
    
    def select_action(self, state: List[float], training: bool = True) -> int:
        """选择动作（ε-贪婪策略）"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        q_values = self.get_q_values(state)
        return q_values.index(max(q_values))
    
    def update(self, state: List[float], action: int, reward: float, 
               next_state: List[float], done: bool):
        """更新Q值"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.num_actions
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning更新
        self.q_table[state_key][action] = (
            current_q + self.learning_rate * (target_q - current_q)
        )
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class SimpleAgent:
    """简化的Q-Learning智能体"""
    
    ACTIONS = {
        0: "MOVE_NORTH",
        1: "MOVE_EAST",
        2: "MOVE_SOUTH",
        3: "MOVE_WEST",
        4: "CLEAN",
        5: "ROTATE_SCAN",
    }
    
    ACTION_VECTORS = {
        0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0),
    }
    
    def __init__(self):
        self.q_table = SimpleQTable(num_actions=len(self.ACTIONS))
    
    def select_action(self, state: List[float], training: bool = True) -> int:
        return self.q_table.select_action(state, training)
    
    def learn(self, state, action, reward, next_state, done):
        self.q_table.update(state, action, reward, next_state, done)


# ==================== 动作执行模块 ====================

class RobotActuator:
    """机器人执行器"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        self.cleaning_efficiency = 0.8
        
    def execute_move(self, current_pos: Tuple[int, int], action_id: int) -> Tuple[Tuple[int, int], bool]:
        """执行移动动作"""
        if action_id not in SimpleAgent.ACTION_VECTORS:
            return current_pos, False
        
        dx, dy = SimpleAgent.ACTION_VECTORS[action_id]
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        
        if self.environment.is_valid_position(new_x, new_y):
            return (new_x, new_y), True
        
        return current_pos, False
    
    def execute_clean(self, position: Tuple[int, int]) -> float:
        """执行清扫动作"""
        x, y = position
        cleaned_amount = self.environment.clean_cell(x, y, self.cleaning_efficiency)
        return cleaned_amount


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
    """具身智能扫地机器人"""
    
    def __init__(self, environment: RoomEnvironment):
        self.environment = environment
        self.perception = PerceptionSystem(environment)
        self.agent = SimpleAgent()
        self.actuator = RobotActuator(environment)
        
        self.state = RobotState(
            position=(0, 0),
            battery=100.0,
            total_cleaned=0.0,
            steps=0,
            collisions=0
        )
        
        self.episode_rewards = []
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
        
        # 3. 决策
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
        
        # 6. 学习
        if training:
            self.agent.learn(state_vector, action, reward, new_state_vector, done)
        
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
                reward += 0.1
                
                cell = self.environment.get_cell(new_pos[0], new_pos[1])
                if cell:
                    cell.visited_count += 1
                    if cell.visited_count == 1:
                        reward += 0.5
            else:
                self.state.collisions += 1
                reward -= 1.0
        
        # 清扫动作
        elif action == 4:
            cleaned = self.actuator.execute_clean(self.state.position)
            self.state.total_cleaned += cleaned
            reward += cleaned * 10.0
            
            if cleaned < 0.01:
                reward -= 0.5
        
        # 扫描动作
        elif action == 5:
            reward += 0.05
        
        # 电池消耗
        self.state.battery -= 0.1
        
        # 检查结束条件
        if self.state.battery <= 0:
            reward -= 10.0
            done = True
        elif self.state.steps >= 500:
            done = True
        elif self.environment.get_cleanliness_ratio() > 0.95:
            reward += 50.0
            done = True
        
        reward -= 0.01  # 时间惩罚
        
        return reward, done
    
    def train_episode(self) -> Dict:
        """训练一个回合"""
        self.reset()
        total_reward = 0.0
        
        while True:
            reward, done = self.step(training=True)
            total_reward += reward
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.episode_cleanliness.append(self.environment.get_cleanliness_ratio())
        
        return {
            "reward": total_reward,
            "steps": self.state.steps,
            "cleanliness": self.environment.get_cleanliness_ratio(),
            "cleaned_amount": self.state.total_cleaned,
            "collisions": self.state.collisions,
            "epsilon": self.agent.q_table.epsilon,
            "q_table_size": len(self.agent.q_table.q_table)
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
            
            if verbose and self.state.steps % 50 == 0:
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

def calculate_stats(values: List[float], window: int = 10) -> Dict:
    """计算统计数据"""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    
    result = {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values)
    }
    
    # 移动平均
    if len(values) >= window:
        recent = values[-window:]
        result["recent_mean"] = sum(recent) / len(recent)
    
    return result


def train_embodied_robot(episodes: int = 150, eval_interval: int = 30):
    """训练具身智能机器人"""
    print("=" * 60)
    print("具身智能扫地机器人训练系统")
    print("=" * 60)
    
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
        stats = robot.train_episode()
        training_history.append(stats)
        
        if stats["cleanliness"] > best_cleanliness:
            best_cleanliness = stats["cleanliness"]
        
        # 打印进度
        if episode % 10 == 0:
            recent = training_history[-10:]
            avg_reward = sum(s["reward"] for s in recent) / len(recent)
            avg_clean = sum(s["cleanliness"] for s in recent) / len(recent)
            avg_steps = sum(s["steps"] for s in recent) / len(recent)
            
            print(f"Episode {episode:3d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Clean: {avg_clean:.2%} | "
                  f"Steps: {avg_steps:.0f} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Q表: {stats['q_table_size']:4d} | "
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
    
    # 训练结果分析
    rewards = [h["reward"] for h in training_history]
    cleanliness = [h["cleanliness"] for h in training_history]
    
    reward_stats = calculate_stats(rewards)
    clean_stats = calculate_stats(cleanliness)
    
    print(f"\n训练结果分析:")
    print(f"  奖励: 均值={reward_stats['mean']:.2f}, "
          f"最大={reward_stats['max']:.2f}, 最小={reward_stats['min']:.2f}")
    print(f"  清洁度: 均值={clean_stats['mean']:.2%}, "
          f"最大={clean_stats['max']:.2%}")
    
    if len(training_history) >= 20:
        first_10 = training_history[:10]
        last_10 = training_history[-10:]
        
        first_clean = sum(h["cleanliness"] for h in first_10) / 10
        last_clean = sum(h["cleanliness"] for h in last_10) / 10
        
        print(f"\n学习进步:")
        print(f"  前10回合平均清洁度: {first_clean:.2%}")
        print(f"  后10回合平均清洁度: {last_clean:.2%}")
        print(f"  提升: {(last_clean - first_clean):.2%}")
    
    print(f"\n训练历史已保存到: embodied_robot_training.json")
    print(f"{'='*60}")
    
    return robot, training_history


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    
    # 训练机器人
    robot, history = train_embodied_robot(episodes=150, eval_interval=50)
    
    # 最终演示
    print("\n" + "=" * 60)
    print("最终策略演示")
    print("=" * 60)
    robot.evaluate(verbose=True)
