"""
LLM的MoE（Mixture of Experts）机制简单实现

MoE是一种神经网络架构，它使用多个专家网络和一个门控网络来处理输入。
门控网络决定每个输入应该由哪些专家处理，从而实现参数的高效利用。

主要组件：
1. 专家网络（Expert Networks）：多个并行的前馈网络
2. 门控网络（Gating Network）：决定专家权重的网络
3. Top-K路由：只激活K个最相关的专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Expert(nn.Module):
    """单个专家网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, expert_id):
        super(Expert, self).__init__()
        self.expert_id = expert_id
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """专家网络前向传播"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """门控网络 - 决定专家权重"""
    
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        # 门控网络通常比较简单
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        """计算每个专家的权重"""
        # 计算门控分数
        gate_scores = self.gate(x)  # [batch_size, num_experts]
        
        # 使用softmax得到概率分布
        gate_weights = F.softmax(gate_scores, dim=-1)
        
        return gate_weights, gate_scores


class TopKGating(nn.Module):
    """Top-K门控 - 只激活前K个专家"""
    
    def __init__(self, input_dim, num_experts, top_k=2, capacity_factor=1.0):
        super(TopKGating, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 添加噪声以增强专家多样性
        self.noise_std = 0.1
        
    def forward(self, x):
        """Top-K门控前向传播"""
        batch_size = x.size(0)
        
        # 计算门控分数
        gate_scores = self.gate(x)
        
        # 添加训练时的噪声
        if self.training:
            noise = torch.randn_like(gate_scores) * self.noise_std
            gate_scores = gate_scores + noise
        
        # 计算top-k专家
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # 对top-k分数进行softmax
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        
        # 创建完整的权重张量
        weights = torch.zeros_like(gate_scores)
        weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return weights, top_k_indices, top_k_weights


class MoELayer(nn.Module):
    """MoE层的完整实现"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, top_k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, i) 
            for i in range(num_experts)
        ])
        
        # Top-K门控网络
        self.gating = TopKGating(input_dim, num_experts, top_k)
        
    def forward(self, x):
        """MoE层前向传播"""
        batch_size, seq_len, input_dim = x.shape
        
        # 重塑输入以便处理
        x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        
        # 获取门控权重
        gate_weights, top_k_indices, top_k_weights = self.gating(x_flat)
        
        # 初始化输出
        output = torch.zeros(x_flat.size(0), self.output_dim, device=x.device)
        
        # 只处理被选中的专家
        for i in range(self.num_experts):
            # 找到应该由专家i处理的样本
            expert_mask = (top_k_indices == i).any(dim=-1)
            
            if expert_mask.any():
                # 获取专家权重
                expert_weights = gate_weights[:, i][expert_mask]
                
                # 获取输入数据
                expert_input = x_flat[expert_mask]
                
                # 专家处理
                expert_output = self.experts[i](expert_input)
                
                # 加权累加到输出
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
        
        # 重塑回原始形状
        output = output.view(batch_size, seq_len, self.output_dim)
        
        # 返回输出和一些统计信息
        expert_usage = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
        
        return output, {
            'gate_weights': gate_weights,
            'expert_usage': expert_usage,
            'top_k_indices': top_k_indices
        }


class SimpleMoETransformer(nn.Module):
    """集成MoE的简单Transformer块"""
    
    def __init__(self, d_model=512, num_experts=8, top_k=2, ff_dim=2048):
        super(SimpleMoETransformer, self).__init__()
        self.d_model = d_model
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 简化的自注意力（这里用线性层代替演示）
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=8)
        
        # MoE前馈网络
        self.moe_layer = MoELayer(
            input_dim=d_model,
            hidden_dim=ff_dim,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=top_k
        )
    
    def forward(self, x):
        """Transformer块前向传播"""
        # 自注意力 + 残差连接
        attn_out, _ = self.self_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        x = x + attn_out.transpose(0, 1)
        x = self.norm1(x)
        
        # MoE前馈网络 + 残差连接
        moe_out, moe_stats = self.moe_layer(x)
        x = x + moe_out
        x = self.norm2(x)
        
        return x, moe_stats


def demonstrate_moe():
    """演示MoE机制"""
    print("=== LLM MoE机制演示 ===\n")
    
    # 设置参数
    batch_size = 4
    seq_len = 10
    d_model = 256
    num_experts = 6
    top_k = 2
    
    # 创建模型
    moe_transformer = SimpleMoETransformer(
        d_model=d_model, 
        num_experts=num_experts, 
        top_k=top_k
    )
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output, moe_stats = moe_transformer(x)
    print(f"输出形状: {output.shape}")
    
    # 分析专家使用情况
    expert_usage = moe_stats['expert_usage']
    print(f"\n专家使用统计:")
    for i, usage in enumerate(expert_usage):
        print(f"专家 {i}: 被使用 {usage.item()} 次")
    
    # 计算专家负载平衡
    total_usage = expert_usage.sum().item()
    avg_usage = total_usage / num_experts
    load_balance = 1.0 - (expert_usage.float().std() / avg_usage)
    print(f"\n负载平衡度: {load_balance:.3f} (1.0为完美平衡)")
    
    # 显示门控权重分布
    gate_weights = moe_stats['gate_weights']
    print(f"\n门控权重统计:")
    print(f"平均权重: {gate_weights.mean(dim=0)}")
    print(f"权重标准差: {gate_weights.std(dim=0)}")
    
    return moe_transformer, moe_stats


def visualize_expert_selection():
    """可视化专家选择过程"""
    print("\n=== 专家选择可视化 ===")
    
    # 创建简单的MoE层
    moe = MoELayer(input_dim=64, hidden_dim=128, output_dim=64, num_experts=4, top_k=2)
    
    # 创建不同类型的输入
    inputs = [
        torch.randn(1, 5, 64),          # 随机输入
        torch.ones(1, 5, 64) * 0.5,     # 恒定输入
        torch.randn(1, 5, 64) * 2,      # 高方差输入
    ]
    
    input_names = ["随机输入", "恒定输入", "高方差输入"]
    
    for i, (inp, name) in enumerate(zip(inputs, input_names)):
        print(f"\n{name}:")
        output, stats = moe(inp)
        
        # 显示每个时间步选择的专家
        top_k_indices = stats['top_k_indices'].view(-1, 2)  # [seq_len, top_k]
        
        for t in range(top_k_indices.size(0)):
            experts = top_k_indices[t].tolist()
            print(f"  时间步 {t}: 选择专家 {experts}")


def moe_training_example():
    """MoE训练示例"""
    print("\n=== MoE训练示例 ===")
    
    # 创建模型和优化器
    model = SimpleMoETransformer(d_model=128, num_experts=4, top_k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据
    batch_size, seq_len, d_model = 8, 16, 128
    
    print("开始训练...")
    for epoch in range(5):
        # 生成随机数据和目标
        x = torch.randn(batch_size, seq_len, d_model)
        target = torch.randn(batch_size, seq_len, d_model)
        
        # 前向传播
        output, moe_stats = model(x)
        
        # 计算损失
        mse_loss = F.mse_loss(output, target)
        
        # 添加负载平衡损失（鼓励专家均匀使用）
        expert_usage = moe_stats['expert_usage'].float()
        balance_loss = expert_usage.var() * 0.01  # 权重可调
        
        total_loss = mse_loss + balance_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 打印训练信息
        if epoch % 2 == 0:
            print(f"轮次 {epoch}: MSE损失={mse_loss:.4f}, 平衡损失={balance_loss:.4f}")
            print(f"  专家使用: {expert_usage.tolist()}")


if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 演示MoE机制
    model, stats = demonstrate_moe()
    
    # 可视化专家选择
    visualize_expert_selection()
    
    # 训练示例
    moe_training_example()
    
    print("\n=== MoE机制总结 ===")
    print("1. MoE通过多个专家网络实现参数的条件激活")
    print("2. 门控网络决定每个输入由哪些专家处理")
    print("3. Top-K路由只激活最相关的K个专家，提高效率")
    print("4. 负载平衡是MoE训练的重要考虑因素")
    print("5. MoE可以显著增加模型容量而不成比例增加计算成本")