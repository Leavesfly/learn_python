"""
多臂老虎机算法详解与实现
Multi-Armed Bandit (MAB) Algorithms

多臂老虎机是强化学习中的经典问题，专注于探索与利用的权衡。
虽然是最简单的强化学习场景（只有一个状态），但包含了强化学习的核心思想。

本文件包含：
1. 多臂老虎机问题定义
2. 贪心算法
3. ε-贪心算法
4. Upper Confidence Bound (UCB) 算法
5. Thompson Sampling 算法
6. 算法比较与分析
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import math
from abc import ABC, abstractmethod
from scipy import stats

class BanditEnvironment:
    """
    多臂老虎机环境
    每个老虎机都有不同的奖励分布
    """
    
    def __init__(self, n_arms: int, reward_type: str = "gaussian"):
        """
        初始化多臂老虎机环境
        
        Args:
            n_arms: 老虎机数量
            reward_type: 奖励类型 ("gaussian", "bernoulli")
        """
        self.n_arms = n_arms
        self.reward_type = reward_type
        
        if reward_type == "gaussian":
            # 高斯分布奖励：每个臂有不同的均值，方差为1
            self.true_means = np.random.normal(0, 1, n_arms)
            self.optimal_arm = np.argmax(self.true_means)
        elif reward_type == "bernoulli":
            # 伯努利分布奖励：每个臂有不同的成功概率
            self.true_probs = np.random.beta(2, 2, n_arms)
            self.optimal_arm = np.argmax(self.true_probs)
        
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.arm_counts = np.zeros(self.n_arms)
        self.total_pulls = 0
        self.regret_history = []
        
    def pull_arm(self, arm: int) -> float:
        """
        拉动指定的老虎机臂
        
        Args:
            arm: 老虎机臂的索引
            
        Returns:
            获得的奖励
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"无效的臂索引: {arm}")
        
        self.arm_counts[arm] += 1
        self.total_pulls += 1
        
        if self.reward_type == "gaussian":
            reward = np.random.normal(self.true_means[arm], 1)
            # 计算瞬时遗憾（最优臂的奖励 - 当前选择的期望奖励）
            instant_regret = self.true_means[self.optimal_arm] - self.true_means[arm]
        elif self.reward_type == "bernoulli":
            reward = np.random.binomial(1, self.true_probs[arm])
            instant_regret = self.true_probs[self.optimal_arm] - self.true_probs[arm]
        
        self.regret_history.append(instant_regret)
        return reward
    
    def get_cumulative_regret(self) -> float:
        """获取累积遗憾"""
        return np.sum(self.regret_history)
    
    def get_optimal_arm_ratio(self) -> float:
        """获取选择最优臂的比例"""
        if self.total_pulls == 0:
            return 0.0
        return self.arm_counts[self.optimal_arm] / self.total_pulls
    
    def display_info(self):
        """显示环境信息"""
        print(f"🎰 多臂老虎机环境信息")
        print(f"老虎机数量: {self.n_arms}")
        print(f"奖励类型: {self.reward_type}")
        
        if self.reward_type == "gaussian":
            print("每个臂的真实均值:")
            for i, mean in enumerate(self.true_means):
                marker = "⭐" if i == self.optimal_arm else "  "
                print(f"  臂 {i}: {mean:.3f} {marker}")
        elif self.reward_type == "bernoulli":
            print("每个臂的成功概率:")
            for i, prob in enumerate(self.true_probs):
                marker = "⭐" if i == self.optimal_arm else "  "
                print(f"  臂 {i}: {prob:.3f} {marker}")
        
        print(f"最优臂: {self.optimal_arm}")

class BanditAgent(ABC):
    """
    多臂老虎机智能体抽象基类
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.reset()
    
    def reset(self):
        """重置智能体状态"""
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)
        self.total_reward = 0.0
        self.time_step = 0
    
    @abstractmethod
    def select_arm(self) -> int:
        """选择要拉动的臂"""
        pass
    
    def update(self, arm: int, reward: float):
        """更新智能体状态"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_reward += reward
        self.time_step += 1
    
    def get_estimated_values(self) -> np.ndarray:
        """获取每个臂的估计价值"""
        with np.errstate(divide='ignore', invalid='ignore'):
            estimated_values = self.arm_rewards / self.arm_counts
            # 将未拉动过的臂的估计值设为0
            estimated_values[self.arm_counts == 0] = 0
        return estimated_values

class GreedyAgent(BanditAgent):
    """
    贪心算法智能体
    总是选择当前估计价值最高的臂
    """
    
    def __init__(self, n_arms: int, initial_value: float = 0.0):
        super().__init__(n_arms)
        self.initial_value = initial_value
        
    def select_arm(self) -> int:
        if self.time_step < self.n_arms:
            # 初始阶段：每个臂至少尝试一次
            return self.time_step
        
        # 选择估计价值最高的臂
        estimated_values = self.get_estimated_values()
        # 对于未尝试过的臂，使用初始值
        estimated_values[self.arm_counts == 0] = self.initial_value
        return np.argmax(estimated_values)

class EpsilonGreedyAgent(BanditAgent):
    """
    ε-贪心算法智能体
    以ε的概率探索，以(1-ε)的概率利用
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: bool = False):
        super().__init__(n_arms)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay = decay
        
    def select_arm(self) -> int:
        if self.decay:
            # 随时间衰减ε
            self.epsilon = self.initial_epsilon / (1 + self.time_step * 0.001)
        
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.randint(0, self.n_arms - 1)
        else:
            # 利用：选择当前最佳
            estimated_values = self.get_estimated_values()
            # 未尝试过的臂给予较高的初始估计
            estimated_values[self.arm_counts == 0] = float('inf')
            return np.argmax(estimated_values)

class UCBAgent(BanditAgent):
    """
    Upper Confidence Bound (UCB) 算法智能体
    基于置信区间上界来平衡探索与利用
    """
    
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms)
        self.c = c  # 置信度参数
        
    def select_arm(self) -> int:
        if self.time_step < self.n_arms:
            # 初始阶段：每个臂至少尝试一次
            return self.time_step
        
        # 计算UCB值
        ucb_values = np.zeros(self.n_arms)
        estimated_values = self.get_estimated_values()
        
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                # UCB公式: Q̂(a) + c * sqrt(ln(t) / N(a))
                confidence_bonus = self.c * math.sqrt(
                    math.log(self.time_step) / self.arm_counts[arm]
                )
                ucb_values[arm] = estimated_values[arm] + confidence_bonus
        
        return np.argmax(ucb_values)

class ThompsonSamplingAgent(BanditAgent):
    """
    Thompson Sampling 算法智能体
    基于贝叶斯推断的概率匹配方法
    """
    
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        super().__init__(n_arms)
        # Beta分布的先验参数
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)
        
    def select_arm(self) -> int:
        # 从每个臂的Beta后验分布中采样
        sampled_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            sampled_values[arm] = np.random.beta(self.alpha[arm], self.beta[arm])
        
        return np.argmax(sampled_values)
    
    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        
        # 更新Beta分布参数
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

def run_experiment(env: BanditEnvironment, 
                  agent: BanditAgent, 
                  n_steps: int = 1000,
                  verbose: bool = False) -> dict:
    """
    运行多臂老虎机实验
    
    Args:
        env: 环境
        agent: 智能体
        n_steps: 实验步数
        verbose: 是否输出详细信息
        
    Returns:
        实验结果字典
    """
    env.reset_stats()
    agent.reset()
    
    rewards_history = []
    regret_history = []
    arm_selection_history = []
    
    for step in range(n_steps):
        # 智能体选择臂
        selected_arm = agent.select_arm()
        
        # 环境给出奖励
        reward = env.pull_arm(selected_arm)
        
        # 智能体更新
        agent.update(selected_arm, reward)
        
        # 记录历史
        rewards_history.append(reward)
        regret_history.append(env.regret_history[-1])
        arm_selection_history.append(selected_arm)
        
        if verbose and (step + 1) % 100 == 0:
            print(f"步骤 {step + 1}: 选择臂 {selected_arm}, 奖励 {reward:.3f}")
    
    results = {
        'total_reward': agent.total_reward,
        'cumulative_regret': env.get_cumulative_regret(),
        'optimal_arm_ratio': env.get_optimal_arm_ratio(),
        'rewards_history': rewards_history,
        'regret_history': regret_history,
        'arm_selection_history': arm_selection_history,
        'estimated_values': agent.get_estimated_values(),
        'arm_counts': agent.arm_counts.copy()
    }
    
    return results

def compare_algorithms():
    """
    比较不同多臂老虎机算法的性能
    """
    print("🔬 多臂老虎机算法性能比较")
    print("=" * 50)
    
    # 创建环境
    env = BanditEnvironment(n_arms=10, reward_type="gaussian")
    env.display_info()
    
    # 创建不同的智能体
    agents = {
        "贪心": GreedyAgent(env.n_arms),
        "ε-贪心(0.1)": EpsilonGreedyAgent(env.n_arms, epsilon=0.1),
        "ε-贪心衰减": EpsilonGreedyAgent(env.n_arms, epsilon=0.3, decay=True),
        "UCB": UCBAgent(env.n_arms, c=2.0),
        "Thompson Sampling": ThompsonSamplingAgent(env.n_arms)
    }
    
    n_steps = 2000
    n_runs = 10  # 多次运行求平均
    
    print(f"\n🎯 实验设置: {n_steps} 步, {n_runs} 次运行")
    print("-" * 50)
    
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"正在测试 {agent_name}...")
        
        cumulative_regrets = []
        total_rewards = []
        optimal_ratios = []
        
        for run in range(n_runs):
            result = run_experiment(env, agent, n_steps)
            cumulative_regrets.append(result['cumulative_regret'])
            total_rewards.append(result['total_reward'])
            optimal_ratios.append(result['optimal_arm_ratio'])
        
        results[agent_name] = {
            'avg_cumulative_regret': np.mean(cumulative_regrets),
            'std_cumulative_regret': np.std(cumulative_regrets),
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'avg_optimal_ratio': np.mean(optimal_ratios),
            'std_optimal_ratio': np.std(optimal_ratios)
        }
    
    # 显示结果
    print("\n📊 算法性能比较结果:")
    print("-" * 80)
    print(f"{'算法':<15} {'累积遗憾':<15} {'总奖励':<15} {'最优臂比例':<15}")
    print("-" * 80)
    
    for agent_name, result in results.items():
        regret_str = f"{result['avg_cumulative_regret']:.1f}±{result['std_cumulative_regret']:.1f}"
        reward_str = f"{result['avg_total_reward']:.1f}±{result['std_total_reward']:.1f}"
        ratio_str = f"{result['avg_optimal_ratio']:.3f}±{result['std_optimal_ratio']:.3f}"
        
        print(f"{agent_name:<15} {regret_str:<15} {reward_str:<15} {ratio_str:<15}")
    
    return results, env

def visualize_single_run():
    """
    可视化单次运行的详细过程
    """
    print("\n🎨 单次运行可视化演示")
    print("=" * 30)
    
    # 创建环境
    env = BanditEnvironment(n_arms=5, reward_type="gaussian")
    
    # 创建智能体
    agents = {
        "ε-贪心": EpsilonGreedyAgent(env.n_arms, epsilon=0.1),
        "UCB": UCBAgent(env.n_arms, c=2.0),
        "Thompson Sampling": ThompsonSamplingAgent(env.n_arms)
    }
    
    n_steps = 1000
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: 累积遗憾
    plt.subplot(2, 2, 1)
    for agent_name, agent in agents.items():
        result = run_experiment(env, agent, n_steps)
        cumulative_regret = np.cumsum(result['regret_history'])
        plt.plot(cumulative_regret, label=agent_name, alpha=0.8)
    
    plt.xlabel('时间步')
    plt.ylabel('累积遗憾')
    plt.title('累积遗憾对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 平均奖励
    plt.subplot(2, 2, 2)
    window_size = 50
    
    for agent_name, agent in agents.items():
        result = run_experiment(env, agent, n_steps)
        rewards = result['rewards_history']
        # 计算移动平均
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(rewards[start_idx:i+1]))
        
        plt.plot(moving_avg, label=agent_name, alpha=0.8)
    
    plt.xlabel('时间步')
    plt.ylabel('平均奖励')
    plt.title(f'平均奖励对比 (窗口大小: {window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 臂选择分布
    plt.subplot(2, 2, 3)
    agent = UCBAgent(env.n_arms, c=2.0)
    result = run_experiment(env, agent, n_steps)
    
    plt.bar(range(env.n_arms), result['arm_counts'], alpha=0.7)
    plt.xlabel('老虎机臂')
    plt.ylabel('选择次数')
    plt.title('UCB算法的臂选择分布')
    
    # 标记最优臂
    plt.axvline(x=env.optimal_arm, color='red', linestyle='--', 
                label=f'最优臂 {env.optimal_arm}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 估计值 vs 真实值
    plt.subplot(2, 2, 4)
    true_values = env.true_means
    estimated_values = result['estimated_values']
    
    x = range(env.n_arms)
    plt.bar([i - 0.2 for i in x], true_values, width=0.4, 
            label='真实均值', alpha=0.7)
    plt.bar([i + 0.2 for i in x], estimated_values, width=0.4, 
            label='估计均值', alpha=0.7)
    
    plt.xlabel('老虎机臂')
    plt.ylabel('奖励均值')
    plt.title('真实值 vs 估计值比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandit_algorithms_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为 bandit_algorithms_comparison.png")

def interactive_demo():
    """
    交互式演示
    """
    print("\n🎮 交互式多臂老虎机演示")
    print("=" * 35)
    
    # 创建简单的3臂老虎机
    env = BanditEnvironment(n_arms=3, reward_type="bernoulli")
    env.display_info()
    
    print("\n你可以选择拉动哪个臂 (0, 1, 2)，或输入 'q' 退出")
    print("我们来看看你能否找到最优的老虎机！")
    
    total_reward = 0
    total_pulls = 0
    
    while True:
        choice = input(f"\n第 {total_pulls + 1} 次选择 (0/1/2/q): ").strip().lower()
        
        if choice == 'q':
            break
        
        try:
            arm = int(choice)
            if arm not in [0, 1, 2]:
                print("请输入 0, 1, 2 中的一个数字")
                continue
        except ValueError:
            print("请输入有效的数字或 'q'")
            continue
        
        # 拉动老虎机臂
        reward = env.pull_arm(arm)
        total_reward += reward
        total_pulls += 1
        
        if reward > 0:
            print(f"🎉 恭喜！你获得了奖励！")
        else:
            print(f"😞 很遗憾，这次没有奖励")
        
        print(f"当前总奖励: {total_reward}, 总尝试次数: {total_pulls}")
        print(f"各臂的选择次数: {env.arm_counts}")
        
        if total_pulls >= 20:
            optimal_ratio = env.get_optimal_arm_ratio()
            print(f"\n📊 你选择最优臂的比例: {optimal_ratio:.2%}")
            if optimal_ratio > 0.6:
                print("🏆 表现很好！你很快找到了最优策略！")
            elif optimal_ratio > 0.4:
                print("👍 表现不错，继续保持！")
            else:
                print("🤔 还有改进空间，多探索一下不同的选择！")

def explain_bandit_theory():
    """
    解释多臂老虎机理论
    """
    theory = """
    📚 多臂老虎机理论详解
    
    🎯 问题定义:
    - 有 K 个老虎机臂，每个臂有未知的奖励分布
    - 每次只能选择一个臂，获得相应的奖励
    - 目标：最大化长期累积奖励
    
    🔄 核心挑战 - 探索与利用权衡:
    - 探索 (Exploration): 尝试新的臂来获取信息
    - 利用 (Exploitation): 选择当前认为最好的臂
    - 过度探索：浪费时间在次优臂上
    - 过度利用：可能错过真正的最优臂
    
    📊 评估指标:
    1. 累积遗憾 (Cumulative Regret):
       R(T) = Σ[μ* - μ(a_t)]
       其中 μ* 是最优臂的期望奖励
    
    2. 简单遗憾 (Simple Regret):
       最终推荐臂与最优臂的期望奖励差
    
    🧮 主要算法:
    
    1. ε-贪心算法:
       - 以 ε 概率随机探索
       - 以 (1-ε) 概率选择当前最佳
       - 简单但有效
    
    2. UCB算法:
       - 基于置信区间上界
       - 公式: Q̂(a) + c√(ln(t)/N(a))
       - 理论保证: O(√(K*ln(T)/T)) 遗憾界
    
    3. Thompson Sampling:
       - 贝叶斯方法
       - 从后验分布中采样
       - 优雅地平衡探索与利用
    
    🏆 理论结果:
    - 最优遗憾下界: Ω(√(K*T*ln(T)))
    - UCB和Thompson Sampling都接近最优
    - 对于特定问题，可以达到问题相关的界
    """
    print(theory)

def main():
    """
    多臂老虎机教程主函数
    """
    print("🎰 多臂老虎机算法完整教程")
    print("=" * 50)
    
    # 理论解释
    explain_bandit_theory()
    
    # 算法比较
    results, env = compare_algorithms()
    
    # 可视化演示
    try:
        visualize_single_run()
    except Exception as e:
        print(f"可视化时出错: {e}")
        print("请确保安装了matplotlib: pip install matplotlib")
    
    # 交互式演示
    choice = input("\n是否要进行交互式演示？(y/n): ").strip().lower()
    if choice == 'y':
        interactive_demo()
    
    print("\n🎓 多臂老虎机学习总结:")
    print("1. 多臂老虎机是强化学习的基础问题")
    print("2. 核心在于探索与利用的权衡")
    print("3. ε-贪心简单有效，UCB有理论保证")
    print("4. Thompson Sampling优雅地处理不确定性")
    print("5. 为更复杂的强化学习问题奠定基础")
    
    print("\n📈 进阶学习建议:")
    print("- 上下文老虎机 (Contextual Bandits)")
    print("- 对抗性老虎机 (Adversarial Bandits)")
    print("- 线性老虎机 (Linear Bandits)")
    print("- 神经网络老虎机 (Neural Bandits)")

if __name__ == "__main__":
    main()