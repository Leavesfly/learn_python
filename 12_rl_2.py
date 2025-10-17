"""
Q-Learning算法实现
经典的表格式强化学习算法

Q-Learning是一种off-policy的时序差分学习算法
核心思想：通过迭代更新Q表来学习最优动作价值函数
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pickle

class QLearningAgent:
    """
    Q-Learning智能体实现
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Q-Learning智能体
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon: ε-贪婪策略的探索率
            epsilon_decay: ε衰减率
            epsilon_min: ε的最小值
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # 记录训练过程
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_history': []
        }
    
    def get_state_key(self, state):
        """将状态转换为Q表的键"""
        if isinstance(state, tuple):
            return state
        elif isinstance(state, np.ndarray):
            return tuple(state)
        else:
            return state
    
    def choose_action(self, state, valid_actions=None):
        """
        ε-贪婪策略选择动作
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表
            
        Returns:
            选择的动作
        """
        state_key = self.get_state_key(state)
        
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        # ε-贪婪策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state_key]
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning更新规则
        
        Q(s,a) ← Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            # 终止状态，下一个状态的价值为0
            target_q = reward
        else:
            # 选择下一个状态的最大Q值
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * next_max_q
        
        # Q-Learning更新公式
        self.q_table[state_key][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """衰减ε值"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """保存Q表"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filepath: str):
        """加载Q表"""
        with open(filepath, 'rb') as f:
            loaded_q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            self.q_table.update(loaded_q_table)

class FrozenLakeEnvironment:
    """
    冰湖环境 - 经典强化学习问题
    
    4x4网格，智能体需要从起点(0,0)到达目标(3,3)
    地面有洞，掉进去游戏结束
    """
    
    def __init__(self, size=4, hole_prob=0.1):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # 随机生成洞的位置
        self.holes = set()
        for i in range(size):
            for j in range(size):
                if (i, j) != self.start and (i, j) != self.goal:
                    if random.random() < hole_prob:
                        self.holes.add((i, j))
        
        # 确保至少有一条路径可达
        if len(self.holes) > size * size * 0.3:
            self.holes = set(list(self.holes)[:int(size * size * 0.3)])
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.agent_pos = self.start
        return self.agent_pos
    
    def get_valid_actions(self, state):
        """获取有效动作"""
        valid_actions = []
        x, y = state
        
        # 0: 上, 1: 下, 2: 左, 3: 右
        if x > 0: valid_actions.append(0)
        if x < self.size - 1: valid_actions.append(1)
        if y > 0: valid_actions.append(2)
        if y < self.size - 1: valid_actions.append(3)
        
        return valid_actions
    
    def step(self, action):
        """执行动作"""
        x, y = self.agent_pos
        
        # 移动
        if action == 0 and x > 0:  # 上
            x -= 1
        elif action == 1 and x < self.size - 1:  # 下
            x += 1
        elif action == 2 and y > 0:  # 左
            y -= 1
        elif action == 3 and y < self.size - 1:  # 右
            y += 1
        
        self.agent_pos = (x, y)
        
        # 计算奖励
        if self.agent_pos in self.holes:
            reward = -10.0
            done = True
        elif self.agent_pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # 每步小惩罚
            done = False
        
        return self.agent_pos, reward, done
    
    def render(self):
        """可视化环境"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # 标记洞
        for hole in self.holes:
            grid[hole[0]][hole[1]] = 'H'
        
        # 标记目标
        grid[self.goal[0]][self.goal[1]] = 'G'
        
        # 标记智能体
        if self.agent_pos not in self.holes:
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        print("\n冰湖环境:")
        print("A: 智能体, G: 目标, H: 洞, .: 安全区域")
        for row in grid:
            print(' '.join(row))
        print()

def train_q_learning(episodes=1000, render_interval=200):
    """
    训练Q-Learning智能体
    """
    print("🎯 开始Q-Learning训练")
    print("=" * 40)
    
    # 创建环境和智能体
    env = FrozenLakeEnvironment()
    agent = QLearningAgent(
        state_size=16,  # 4x4网格
        action_size=4,   # 上下左右
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("环境设置:")
    env.render()
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:  # 最大步数限制
            # 选择动作
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 更新Q表
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 记录训练数据
        agent.training_history['episode_rewards'].append(total_reward)
        agent.training_history['episode_steps'].append(steps)
        agent.training_history['epsilon_history'].append(agent.epsilon)
        
        # 衰减ε
        agent.decay_epsilon()
        
        # 定期输出训练信息
        if episode % render_interval == 0:
            avg_reward = np.mean(agent.training_history['episode_rewards'][-100:])
            print(f"回合 {episode}: 平均奖励={avg_reward:.2f}, ε={agent.epsilon:.3f}")
    
    return agent, env

def test_trained_agent(agent, env, num_tests=5):
    """
    测试训练好的智能体
    """
    print("\n🧪 测试训练好的智能体")
    print("=" * 30)
    
    # 暂时关闭探索
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    success_count = 0
    
    for test in range(num_tests):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n测试 {test + 1}:")
        env.render()
        
        while steps < 50:
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            action_names = ['上', '下', '左', '右']
            print(f"步骤 {steps + 1}: {action_names[action]}")
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            
            if done:
                if reward > 0:
                    print(f"✅ 成功到达目标! 奖励: {total_reward:.1f}, 步数: {steps}")
                    success_count += 1
                else:
                    print(f"❌ 掉进洞里! 奖励: {total_reward:.1f}")
                break
        
        if not done:
            print(f"⏰ 超时未完成")
    
    # 恢复原始ε值
    agent.epsilon = original_epsilon
    
    success_rate = success_count / num_tests * 100
    print(f"\n📊 成功率: {success_rate:.1f}% ({success_count}/{num_tests})")

def visualize_training_progress(agent):
    """
    可视化训练过程
    """
    print("\n📈 生成训练过程图表...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 回合奖励
    episodes = range(len(agent.training_history['episode_rewards']))
    ax1.plot(episodes, agent.training_history['episode_rewards'], alpha=0.6)
    ax1.set_title('每回合奖励')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('奖励')
    ax1.grid(True)
    
    # 移动平均奖励
    window = 50
    if len(agent.training_history['episode_rewards']) >= window:
        moving_avg = []
        for i in range(window-1, len(agent.training_history['episode_rewards'])):
            avg = np.mean(agent.training_history['episode_rewards'][i-window+1:i+1])
            moving_avg.append(avg)
        
        ax2.plot(range(window-1, len(agent.training_history['episode_rewards'])), moving_avg)
        ax2.set_title(f'{window}回合移动平均奖励')
        ax2.set_xlabel('回合')
        ax2.set_ylabel('平均奖励')
        ax2.grid(True)
    
    # ε值变化
    ax3.plot(episodes, agent.training_history['epsilon_history'])
    ax3.set_title('探索率(ε)变化')
    ax3.set_xlabel('回合')
    ax3.set_ylabel('ε值')
    ax3.grid(True)
    
    # 回合步数
    ax4.plot(episodes, agent.training_history['episode_steps'], alpha=0.6)
    ax4.set_title('每回合步数')
    ax4.set_xlabel('回合')
    ax4.set_ylabel('步数')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('q_learning_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图表已保存为 q_learning_training_progress.png")

def analyze_q_table(agent, env):
    """
    分析学习到的Q表
    """
    print("\n🔍 Q表分析")
    print("=" * 20)
    
    print("学习到的状态-动作价值:")
    action_names = ['上', '下', '左', '右']
    
    # 显示部分Q值
    states_to_show = [(0, 0), (0, 1), (1, 0), (3, 3)]
    for state in states_to_show:
        if state in agent.q_table:
            q_values = agent.q_table[state]
            print(f"\n状态 {state}:")
            for action, q_val in enumerate(q_values):
                print(f"  {action_names[action]}: {q_val:.3f}")
            best_action = np.argmax(q_values)
            print(f"  最佳动作: {action_names[best_action]}")

def main():
    """
    Q-Learning主程序
    """
    print("🚀 Q-Learning算法演示")
    print("=" * 50)
    
    # 训练智能体
    agent, env = train_q_learning(episodes=1000, render_interval=200)
    
    # 测试智能体
    test_trained_agent(agent, env, num_tests=5)
    
    # 分析Q表
    analyze_q_table(agent, env)
    
    # 可视化训练过程
    try:
        visualize_training_progress(agent)
    except Exception as e:
        print(f"可视化时出错: {e}")
        print("请确保安装了matplotlib: pip install matplotlib")
    
    # 保存模型
    agent.save_model('q_learning_model.pkl')
    print("\n💾 模型已保存为 q_learning_model.pkl")
    
    print("\n🎓 Q-Learning学习要点:")
    print("1. Q-Learning是off-policy算法，不需要遵循当前策略")
    print("2. 通过Bellman方程迭代更新Q值")
    print("3. ε-贪婪策略平衡探索和利用")
    print("4. 适用于状态空间较小的问题")
    print("5. 收敛到最优策略的理论保证")

if __name__ == "__main__":
    main()