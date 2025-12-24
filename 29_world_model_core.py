"""
世界模型 (World Model) 核心架构实现

基于 "World Models" (Ha & Schmidhuber, 2018) 论文的简化实现
结合项目现有的 PyTorch 和强化学习技术栈

核心组件：
1. VQ-VAE (Vector Quantized Variational AutoEncoder) - 表征学习
2. MDN-RNN (Mixture Density Network RNN) - 序列预测
3. Controller - 决策控制器
4. Environment - 简化的 GridWorld 环境

架构流程：
观察 -> VQ-VAE编码 -> 潜在表征 -> MDN-RNN预测 -> Controller决策 -> 动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json
from collections import deque


# ==================== 配置类 ====================
@dataclass
class WorldModelConfig:
    """世界模型配置"""
    # VQ-VAE 配置
    image_size: int = 64  # 输入图像大小
    latent_dim: int = 32  # 潜在空间维度
    num_embeddings: int = 512  # VQ码本大小
    commitment_cost: float = 0.25  # VQ承诺损失系数
    
    # MDN-RNN 配置
    hidden_size: int = 256  # RNN隐藏层大小
    num_mixtures: int = 5  # 混合高斯数量
    sequence_length: int = 32  # 序列长度
    
    # Controller 配置
    action_dim: int = 4  # 动作空间维度 (上下左右)
    controller_hidden: int = 128
    
    # 训练配置
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== VQ-VAE 组件 ====================
class VectorQuantizer(nn.Module):
    """向量量化层 - VQ-VAE的核心组件"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本 (Codebook)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            inputs: [B, C, H, W] 编码器输出
        Returns:
            quantized: 量化后的张量
            vq_loss: VQ损失
            encoding_indices: 编码索引
        """
        # 转换形状: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # 展平: [B*H*W, C]
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        
        # 找到最近的码本向量
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 量化
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(input_shape)
        
        # VQ损失: ||sg[z] - e||^2 + β||z - sg[e]||^2
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 恢复形状: [B, H, W, C] -> [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """VQ-VAE 模型 - 用于学习环境的潜在表征"""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # 编码器: [B, 3, 64, 64] -> [B, latent_dim, 8, 8]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, config.latent_dim, 4, stride=2, padding=1),  # [B, latent_dim, 8, 8]
        )
        
        # 向量量化
        self.vq_layer = VectorQuantizer(
            config.num_embeddings,
            config.latent_dim,
            config.commitment_cost
        )
        
        # 解码器: [B, latent_dim, 8, 8] -> [B, 3, 64, 64]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, 64, 4, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [B, 3, 64, 64]
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码观察到潜在空间"""
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq_layer(z)
        return z_q, vq_loss
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整的前向传播"""
        z = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(z_q)
        
        # 重构损失
        recon_loss = F.mse_loss(x_recon, x)
        
        return x_recon, recon_loss, vq_loss
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """获取潜在表征 [B, latent_dim*8*8]"""
        with torch.no_grad():
            z = self.encoder(x)
            z_q, _, _ = self.vq_layer(z)
            return z_q.view(z_q.size(0), -1)


# ==================== MDN-RNN 组件 ====================
class MDNRNN(nn.Module):
    """
    混合密度网络 RNN - 用于预测未来状态
    
    输入: 当前潜在状态 z_t 和动作 a_t
    输出: 下一状态 z_{t+1} 的概率分布 (混合高斯)
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_mixtures = config.num_mixtures
        
        # 计算潜在表征维度
        latent_size = config.latent_dim * 8 * 8  # 展平后的大小
        
        # LSTM核心
        self.lstm = nn.LSTM(
            input_size=latent_size + config.action_dim,
            hidden_size=config.hidden_size,
            batch_first=True
        )
        
        # MDN输出头: 预测混合高斯参数
        # 对于每个混合分量: π (权重), μ (均值), σ (标准差)
        self.mdn_pi = nn.Linear(config.hidden_size, config.num_mixtures)  # 混合权重
        self.mdn_mu = nn.Linear(config.hidden_size, config.num_mixtures * latent_size)  # 均值
        self.mdn_sigma = nn.Linear(config.hidden_size, config.num_mixtures * latent_size)  # 标准差
        
        # 奖励预测头
        self.reward_head = nn.Linear(config.hidden_size, 1)
        
        # 终止状态预测头
        self.done_head = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        Args:
            z: 潜在状态 [B, T, latent_size]
            action: 动作 [B, T, action_dim]
            hidden: LSTM隐藏状态
        Returns:
            mdn_params: 混合高斯参数字典
            hidden: 新的LSTM隐藏状态
        """
        # 拼接输入
        x = torch.cat([z, action], dim=-1)  # [B, T, latent_size + action_dim]
        
        # LSTM前向
        lstm_out, hidden = self.lstm(x, hidden)  # [B, T, hidden_size]
        
        # MDN参数
        pi = F.softmax(self.mdn_pi(lstm_out), dim=-1)  # [B, T, num_mixtures]
        mu = self.mdn_mu(lstm_out)  # [B, T, num_mixtures * latent_size]
        sigma = F.softplus(self.mdn_sigma(lstm_out)) + 1e-6  # [B, T, num_mixtures * latent_size]
        
        # 奖励和终止预测
        reward = self.reward_head(lstm_out)  # [B, T, 1]
        done = torch.sigmoid(self.done_head(lstm_out))  # [B, T, 1]
        
        mdn_params = {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'reward': reward,
            'done': done
        }
        
        return mdn_params, hidden
    
    def mdn_loss(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算MDN负对数似然损失"""
        batch_size, seq_len, _ = target.shape
        latent_size = target.size(-1)
        
        # 重塑参数
        mu = mu.view(batch_size, seq_len, self.num_mixtures, latent_size)
        sigma = sigma.view(batch_size, seq_len, self.num_mixtures, latent_size)
        target = target.unsqueeze(2)  # [B, T, 1, latent_size]
        
        # 计算每个混合分量的概率密度
        normal_dist = Normal(mu, sigma)
        log_prob = normal_dist.log_prob(target).sum(dim=-1)  # [B, T, num_mixtures]
        
        # 加权求和
        log_pi = torch.log(pi + 1e-8)
        log_mix_prob = log_pi + log_prob
        
        # 对数-和-指数技巧 (log-sum-exp trick)
        loss = -torch.logsumexp(log_mix_prob, dim=-1).mean()
        
        return loss
    
    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """从MDN中采样下一状态"""
        batch_size = pi.size(0)
        latent_size = mu.size(-1) // self.num_mixtures
        
        # 重塑参数
        mu = mu.view(batch_size, self.num_mixtures, latent_size)
        sigma = sigma.view(batch_size, self.num_mixtures, latent_size)
        
        # 采样混合分量
        mixture_idx = Categorical(pi).sample()  # [B]
        
        # 提取对应的 μ 和 σ
        mu_selected = mu[torch.arange(batch_size), mixture_idx]  # [B, latent_size]
        sigma_selected = sigma[torch.arange(batch_size), mixture_idx]
        
        # 从高斯分布采样
        z_next = Normal(mu_selected, sigma_selected).sample()
        
        return z_next


# ==================== Controller 组件 ====================
class Controller(nn.Module):
    """
    控制器 - 基于潜在状态和RNN隐藏状态做决策
    
    这是一个简单的前馈网络，可以用进化策略(ES)或强化学习训练
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        latent_size = config.latent_dim * 8 * 8
        input_size = latent_size + config.hidden_size  # z + h
        
        self.network = nn.Sequential(
            nn.Linear(input_size, config.controller_hidden),
            nn.Tanh(),
            nn.Linear(config.controller_hidden, config.action_dim)
        )
    
    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            z: 潜在状态 [B, latent_size]
            h: RNN隐藏状态 [B, hidden_size]
        Returns:
            action_logits: 动作logits [B, action_dim]
        """
        x = torch.cat([z, h], dim=-1)
        return self.network(x)
    
    def get_action(self, z: torch.Tensor, h: torch.Tensor, deterministic: bool = False) -> int:
        """获取动作"""
        with torch.no_grad():
            logits = self.forward(z, h)
            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                return Categorical(probs).sample().item()


# ==================== 完整世界模型 ====================
class WorldModel(nn.Module):
    """完整的世界模型架构"""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # 三大组件
        self.vae = VQVAE(config)
        self.rnn = MDNRNN(config)
        self.controller = Controller(config)
        
        self.to(config.device)
    
    def train_vae(
        self,
        observations: torch.Tensor,
        epochs: int = 10
    ) -> List[float]:
        """训练VAE编码器"""
        optimizer = optim.Adam(self.vae.parameters(), lr=self.config.learning_rate)
        losses = []
        
        print("=" * 60)
        print("阶段 1: 训练 VQ-VAE 表征学习模型")
        print("=" * 60)
        
        for epoch in range(epochs):
            x_recon, recon_loss, vq_loss = self.vae(observations)
            total_loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"总损失: {total_loss.item():.6f}, "
                      f"重构: {recon_loss.item():.6f}, "
                      f"VQ: {vq_loss.item():.6f}")
        
        return losses
    
    def train_rnn(
        self,
        sequences: List[Dict],
        epochs: int = 10
    ) -> List[float]:
        """训练RNN预测模型"""
        optimizer = optim.Adam(self.rnn.parameters(), lr=self.config.learning_rate)
        losses = []
        
        print("\n" + "=" * 60)
        print("阶段 2: 训练 MDN-RNN 序列预测模型")
        print("=" * 60)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for seq_data in sequences:
                obs = seq_data['observations']  # [T, 3, 64, 64]
                actions = seq_data['actions']  # [T, 4]
                rewards = seq_data['rewards']  # [T]
                dones = seq_data['dones']  # [T]
                
                # 获取潜在表征
                with torch.no_grad():
                    z = self.vae.get_latent(obs).unsqueeze(0)  # [1, T, latent_size]
                
                # 准备动作
                actions_tensor = actions.unsqueeze(0).to(self.config.device)  # [1, T, 4]
                
                # RNN前向
                mdn_params, _ = self.rnn(z[:, :-1], actions_tensor[:, :-1])
                
                # MDN损失 (预测下一状态)
                mdn_loss = self.rnn.mdn_loss(
                    mdn_params['pi'],
                    mdn_params['mu'],
                    mdn_params['sigma'],
                    z[:, 1:]
                )
                
                # 奖励损失
                reward_loss = F.mse_loss(
                    mdn_params['reward'].squeeze(),
                    rewards[:-1].unsqueeze(0).to(self.config.device)
                )
                
                # 终止状态损失
                done_loss = F.binary_cross_entropy(
                    mdn_params['done'].squeeze(),
                    dones[:-1].unsqueeze(0).float().to(self.config.device)
                )
                
                # 总损失
                loss = mdn_loss + 0.1 * reward_loss + 0.1 * done_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(sequences)
            losses.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.6f}")
        
        return losses
    
    def train_controller(
        self,
        env,
        episodes: int = 50
    ) -> List[float]:
        """训练控制器 (使用简单的策略梯度)"""
        optimizer = optim.Adam(self.controller.parameters(), lr=1e-3)
        episode_rewards = []
        
        print("\n" + "=" * 60)
        print("阶段 3: 训练 Controller 决策模型")
        print("=" * 60)
        
        for episode in range(episodes):
            obs = env.reset()
            hidden = None
            episode_reward = 0
            log_probs = []
            rewards = []
            
            for step in range(50):  # 最大步数
                # 转换观察
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                
                # 获取潜在状态
                with torch.no_grad():
                    z = self.vae.get_latent(obs_tensor)
                
                # 获取RNN隐藏状态
                if hidden is None:
                    h = torch.zeros(1, self.config.hidden_size).to(self.config.device)
                else:
                    h = hidden[0].squeeze(0)
                
                # 获取动作
                action_logits = self.controller(z, h)
                action_dist = Categorical(logits=action_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # 环境交互
                next_obs, reward, done, _ = env.step(action.item())
                
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward
                
                # 更新RNN状态
                action_onehot = F.one_hot(action, self.config.action_dim).float().unsqueeze(0).unsqueeze(0)
                _, hidden = self.rnn(z.unsqueeze(1), action_onehot, hidden)
                
                obs = next_obs
                
                if done:
                    break
            
            # 策略梯度更新
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns).to(self.config.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}/{episodes} - "
                      f"平均奖励: {avg_reward:.2f}, "
                      f"最新奖励: {episode_reward:.2f}")
        
        return episode_rewards
    
    def dream(self, initial_obs: torch.Tensor, actions: torch.Tensor) -> Dict:
        """
        在潜在空间中"做梦" - 完全在想象中展开轨迹
        
        Args:
            initial_obs: 初始观察 [1, 3, 64, 64]
            actions: 动作序列 [T, action_dim]
        Returns:
            梦境数据字典
        """
        self.eval()
        with torch.no_grad():
            # 初始编码
            z = self.vae.get_latent(initial_obs)  # [1, latent_size]
            
            dream_data = {
                'latents': [z.cpu()],
                'observations': [initial_obs.cpu()],
                'rewards': [],
                'dones': []
            }
            
            hidden = None
            
            for action in actions:
                # RNN预测
                action_input = action.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
                z_input = z.unsqueeze(1)  # [1, 1, latent_size]
                
                mdn_params, hidden = self.rnn(z_input, action_input, hidden)
                
                # 采样下一状态
                z_next = self.rnn.sample(
                    mdn_params['pi'][:, 0],
                    mdn_params['mu'][:, 0],
                    mdn_params['sigma'][:, 0]
                )
                
                # 解码观察
                z_next_2d = z_next.view(1, self.config.latent_dim, 8, 8)
                obs_next = self.vae.decode(z_next_2d)
                
                # 记录
                dream_data['latents'].append(z_next.cpu())
                dream_data['observations'].append(obs_next.cpu())
                dream_data['rewards'].append(mdn_params['reward'].item())
                dream_data['dones'].append(mdn_params['done'].item())
                
                z = z_next
            
            return dream_data
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'vae': self.vae.state_dict(),
            'rnn': self.rnn.state_dict(),
            'controller': self.controller.state_dict(),
            'config': self.config
        }, path)
        print(f"\n✓ 模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.vae.load_state_dict(checkpoint['vae'])
        self.rnn.load_state_dict(checkpoint['rnn'])
        self.controller.load_state_dict(checkpoint['controller'])
        print(f"✓ 模型已从 {path} 加载")


if __name__ == "__main__":
    print("=" * 60)
    print("世界模型 (World Model) 核心架构")
    print("=" * 60)
    print("\n本模块包含:")
    print("  1. VQ-VAE - 视觉表征学习")
    print("  2. MDN-RNN - 序列预测模型") 
    print("  3. Controller - 决策控制器")
    print("  4. WorldModel - 完整集成架构")
    print("\n请运行 29_world_model_demo.py 查看完整演示")
