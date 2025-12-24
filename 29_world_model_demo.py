"""
世界模型 (World Model) 完整演示

展示从数据收集、模型训练到梦境预测的完整流程

运行方式:
    python 29_world_model_demo.py

功能:
1. 在 GridWorld 环境中收集随机轨迹数据
2. 训练 VQ-VAE 学习视觉表征
3. 训练 MDN-RNN 学习状态转移动态
4. 训练 Controller 学习决策策略
5. 在潜在空间中"做梦"并可视化
6. 对比真实环境和想象环境的差异
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

# 导入世界模型组件
try:
    from world_model_core_29 import WorldModel, WorldModelConfig
    from world_model_env_29 import SimpleGridWorld, DataCollector
except ImportError:
    # 兼容直接导入
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    
    # 动态导入 29_world_model_core
    spec_core = importlib.util.spec_from_file_location(
        "wm_core", 
        os.path.join(os.path.dirname(__file__), "29_world_model_core.py")
    )
    wm_core = importlib.util.module_from_spec(spec_core)
    spec_core.loader.exec_module(wm_core)
    WorldModel = wm_core.WorldModel
    WorldModelConfig = wm_core.WorldModelConfig
    
    # 动态导入 29_world_model_env
    spec_env = importlib.util.spec_from_file_location(
        "wm_env",
        os.path.join(os.path.dirname(__file__), "29_world_model_env.py")
    )
    wm_env = importlib.util.module_from_spec(spec_env)
    spec_env.loader.exec_module(wm_env)
    SimpleGridWorld = wm_env.SimpleGridWorld
    DataCollector = wm_env.DataCollector


# ==================== 可视化工具 ====================
class Visualizer:
    """世界模型可视化工具"""
    
    @staticmethod
    def plot_training_curves(vae_losses, rnn_losses, controller_rewards, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # VAE损失
        axes[0].plot(vae_losses, color='blue', linewidth=2)
        axes[0].set_title('VQ-VAE 训练损失', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # RNN损失
        axes[1].plot(rnn_losses, color='green', linewidth=2)
        axes[1].set_title('MDN-RNN 训练损失', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # Controller奖励
        axes[2].plot(controller_rewards, color='red', alpha=0.3)
        # 绘制移动平均
        window = 10
        if len(controller_rewards) >= window:
            moving_avg = np.convolve(
                controller_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            axes[2].plot(range(window-1, len(controller_rewards)), moving_avg, 
                        color='red', linewidth=2, label=f'MA({window})')
            axes[2].legend()
        
        axes[2].set_title('Controller 训练奖励', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Reward')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 训练曲线已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_reconstruction(original, reconstructed, save_path=None):
        """可视化重构结果"""
        num_samples = min(8, original.shape[0])
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # 原始图像
            img_orig = original[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(img_orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('原始观察', fontweight='bold')
            
            # 重构图像
            img_recon = reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(img_recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('VAE重构', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 重构对比已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_dream_sequence(dream_data, save_path=None):
        """可视化梦境序列"""
        observations = dream_data['observations']
        rewards = dream_data['rewards']
        
        num_frames = min(12, len(observations))
        cols = 6
        rows = (num_frames + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        gs = GridSpec(rows, cols, figure=fig)
        
        for i in range(num_frames):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            
            img = observations[i].squeeze().numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis('off')
            
            if i < len(rewards):
                title = f'步骤 {i}\nR={rewards[i]:.3f}'
            else:
                title = f'步骤 {i}'
            
            ax.set_title(title, fontsize=9)
        
        plt.suptitle('世界模型梦境序列 (在潜在空间中展开)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 梦境序列已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def compare_real_vs_dream(real_obs, dream_obs, save_path=None):
        """对比真实环境和梦境"""
        num_steps = min(8, len(real_obs), len(dream_obs))
        
        fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2, 4))
        
        for i in range(num_steps):
            # 真实环境
            img_real = real_obs[i].transpose(1, 2, 0)
            axes[0, i].imshow(img_real)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('真实环境', fontweight='bold')
            else:
                axes[0, i].set_title(f't={i}')
            
            # 梦境
            img_dream = dream_obs[i].squeeze().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(img_dream)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('梦境预测', fontweight='bold')
            else:
                axes[1, i].set_title(f't={i}')
        
        plt.suptitle('真实环境 vs 世界模型梦境对比', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 对比图已保存到: {save_path}")
        
        plt.show()


# ==================== 主演示流程 ====================
def main():
    print("=" * 70)
    print(" " * 20 + "世界模型 (World Model) 演示")
    print("=" * 70)
    print("\n基于论文: 'World Models' (Ha & Schmidhuber, 2018)")
    print("实现环境: SimpleGridWorld")
    print()
    
    # 配置
    config = WorldModelConfig(
        image_size=64,
        latent_dim=32,
        num_embeddings=512,
        hidden_size=256,
        action_dim=4,
        batch_size=32,
        learning_rate=1e-3
    )
    
    print(f"设备: {config.device}")
    print(f"潜在维度: {config.latent_dim}")
    print(f"RNN隐藏层: {config.hidden_size}")
    print()
    
    # 创建环境
    env = SimpleGridWorld(grid_size=8, image_size=64)
    
    # 创建世界模型
    world_model = WorldModel(config)
    
    # 创建结果目录
    results_dir = "world_model_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ==================== 数据收集 ====================
    print("\n" + "=" * 70)
    print("步骤 1: 数据收集")
    print("=" * 70)
    
    collector = DataCollector(env, device=config.device)
    observations, sequences = collector.collect_random_episodes(
        num_episodes=100,
        max_steps=50
    )
    
    print(f"\n收集到:")
    print(f"  - 观察数量: {observations.shape[0]}")
    print(f"  - 序列数量: {len(sequences)}")
    
    # ==================== 训练 VQ-VAE ====================
    vae_losses = world_model.train_vae(observations, epochs=10)
    
    # 可视化重构结果
    print("\n生成重构对比图...")
    world_model.vae.eval()
    with torch.no_grad():
        sample_obs = observations[:8]
        reconstructed, _, _ = world_model.vae(sample_obs)
    
    Visualizer.plot_reconstruction(
        sample_obs,
        reconstructed,
        save_path=f"{results_dir}/reconstruction_{timestamp}.png"
    )
    
    # ==================== 训练 MDN-RNN ====================
    rnn_losses = world_model.train_rnn(sequences, epochs=10)
    
    # ==================== 训练 Controller ====================
    controller_rewards = world_model.train_controller(env, episodes=50)
    
    # ==================== 绘制训练曲线 ====================
    print("\n" + "=" * 70)
    print("生成训练曲线...")
    print("=" * 70)
    
    Visualizer.plot_training_curves(
        vae_losses,
        rnn_losses,
        controller_rewards,
        save_path=f"{results_dir}/training_curves_{timestamp}.png"
    )
    
    # ==================== 梦境生成 ====================
    print("\n" + "=" * 70)
    print("步骤 4: 在潜在空间中做梦")
    print("=" * 70)
    
    # 收集一个真实轨迹
    real_trajectory = []
    obs = env.reset()
    real_trajectory.append(obs)
    
    dream_actions = []
    for _ in range(15):
        action = np.random.randint(0, 4)
        dream_actions.append(action)
        obs, _, done, _ = env.step(action)
        real_trajectory.append(obs)
        if done:
            break
    
    # 在梦境中展开
    initial_obs = torch.FloatTensor(real_trajectory[0]).unsqueeze(0).to(config.device)
    actions_tensor = torch.FloatTensor(
        [[1 if i == a else 0 for i in range(4)] for a in dream_actions]
    ).to(config.device)
    
    dream_data = world_model.dream(initial_obs, actions_tensor)
    
    print(f"\n梦境生成完成:")
    print(f"  - 梦境长度: {len(dream_data['observations'])}")
    print(f"  - 真实轨迹长度: {len(real_trajectory)}")
    
    # 可视化梦境
    Visualizer.plot_dream_sequence(
        dream_data,
        save_path=f"{results_dir}/dream_sequence_{timestamp}.png"
    )
    
    # 对比真实和梦境
    print("\n生成真实 vs 梦境对比图...")
    Visualizer.compare_real_vs_dream(
        real_trajectory,
        dream_data['observations'],
        save_path=f"{results_dir}/real_vs_dream_{timestamp}.png"
    )
    
    # ==================== 保存模型 ====================
    model_path = f"{results_dir}/world_model_{timestamp}.pt"
    world_model.save(model_path)
    
    # ==================== 性能评估 ====================
    print("\n" + "=" * 70)
    print("步骤 5: 性能评估")
    print("=" * 70)
    
    # 测试控制器
    test_rewards = []
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0
        hidden = None
        
        for step in range(50):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(config.device)
            z = world_model.vae.get_latent(obs_tensor)
            
            if hidden is None:
                h = torch.zeros(1, config.hidden_size).to(config.device)
            else:
                h = hidden[0].squeeze(0)
            
            action = world_model.controller.get_action(z, h, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # 更新RNN状态
            action_onehot = torch.zeros(1, 1, 4).to(config.device)
            action_onehot[0, 0, action] = 1
            _, hidden = world_model.rnn(z.unsqueeze(1), action_onehot, hidden)
            
            if done:
                break
        
        test_rewards.append(episode_reward)
    
    print(f"\n测试结果 (10个回合):")
    print(f"  平均奖励: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  最高奖励: {np.max(test_rewards):.2f}")
    print(f"  最低奖励: {np.min(test_rewards):.2f}")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    print(f"\n所有结果已保存到: {results_dir}/")
    print("\n生成的文件:")
    print(f"  1. reconstruction_{timestamp}.png - VAE重构对比")
    print(f"  2. training_curves_{timestamp}.png - 训练曲线")
    print(f"  3. dream_sequence_{timestamp}.png - 梦境序列")
    print(f"  4. real_vs_dream_{timestamp}.png - 真实vs梦境对比")
    print(f"  5. world_model_{timestamp}.pt - 训练好的模型")
    
    print("\n" + "=" * 70)
    print("世界模型核心概念:")
    print("=" * 70)
    print("1. VQ-VAE: 将高维观察压缩到紧凑的潜在表征")
    print("2. MDN-RNN: 在潜在空间中学习环境动态")
    print("3. Controller: 基于潜在表征做出决策")
    print("4. Dream: 完全在想象中展开轨迹，无需真实环境")
    print("\n优势: 样本效率高、可解释性强、支持规划")
    print("=" * 70)


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行演示
    main()
