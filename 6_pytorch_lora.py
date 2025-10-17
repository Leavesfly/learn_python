import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层的实现
    
    将原始的线性变换 W 分解为 W + BA，其中：
    - W 是预训练的权重矩阵（冻结）
    - B 和 A 是可训练的低秩矩阵
    - rank 控制低秩矩阵的秩
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        
        # 原始的线性层（将被冻结）
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA 的低秩矩阵
        # A 矩阵: (rank, in_features)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # B 矩阵: (out_features, rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # 缩放因子
        self.scaling = alpha / rank
        
        # 初始化权重
        self.reset_parameters()
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        
    def reset_parameters(self):
        """初始化LoRA参数"""
        # 使用Kaiming初始化A矩阵
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B矩阵初始化为0，确保训练开始时LoRA部分输出为0
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 原始线性变换
        result = self.linear(x)
        
        # LoRA 部分：x @ A^T @ B^T
        # 先计算 x @ A^T
        lora_result = x @ self.lora_A.T
        lora_result = self.dropout(lora_result)
        # 再计算结果 @ B^T
        lora_result = lora_result @ self.lora_B.T
        
        # 应用缩放并添加到原始结果
        result = result + lora_result * self.scaling
        
        return result
    
    def merge_lora_weights(self):
        """将LoRA权重合并到原始权重中"""
        if self.merge_weights:
            # 计算 B @ A
            merged_weight = self.lora_B @ self.lora_A
            # 添加到原始权重
            self.linear.weight.data += merged_weight * self.scaling
            
    def separate_lora_weights(self):
        """分离LoRA权重（如果之前已合并）"""
        if self.merge_weights:
            merged_weight = self.lora_B @ self.lora_A
            self.linear.weight.data -= merged_weight * self.scaling


class SimpleTransformerWithLoRA(nn.Module):
    """
    使用LoRA的简单Transformer模型示例
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        lora_rank: int = 16,
        lora_alpha: float = 32.0
    ):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayerWithLoRA(d_model, num_heads, lora_rank, lora_alpha)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_proj = LoRALayer(d_model, vocab_size, rank=lora_rank, alpha=lora_alpha)
        
    def forward(self, x):
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
            
        # 输出投影
        return self.output_proj(x)


class TransformerLayerWithLoRA(nn.Module):
    """使用LoRA的Transformer层"""
    
    def __init__(self, d_model: int, num_heads: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 多头注意力的Query、Key、Value投影（使用LoRA）
        self.q_proj = LoRALayer(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.k_proj = LoRALayer(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.v_proj = LoRALayer(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        self.out_proj = LoRALayer(d_model, d_model, rank=lora_rank, alpha=lora_alpha)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            LoRALayer(d_model, d_model * 4, rank=lora_rank, alpha=lora_alpha),
            nn.ReLU(),
            LoRALayer(d_model * 4, d_model, rank=lora_rank, alpha=lora_alpha)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 多头自注意力
        residual = x
        x = self.norm1(x)
        
        # 简化的注意力计算
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 多头注意力
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.num_heads
        
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # 注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        attn_output = self.out_proj(attn_output)
        
        x = residual + attn_output
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params/total_params:.2%}")
    
    return total_params, trainable_params


def lora_training_demo():
    """LoRA训练演示"""
    print("=" * 50)
    print("LoRA 微调演示")
    print("=" * 50)
    
    # 模型参数
    vocab_size = 10000
    d_model = 512
    seq_len = 128
    batch_size = 8
    
    # 创建模型
    model = SimpleTransformerWithLoRA(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        lora_rank=16,  # 低秩维度
        lora_alpha=32.0  # LoRA缩放因子
    )
    
    print("模型结构:")
    print(model)
    print("\n")
    
    # 统计参数
    count_parameters(model)
    print("\n")
    
    # 创建示例数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 优化器（只优化LoRA参数）
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(input_ids)
        
        # 计算损失
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("LoRA 核心优势:")
    print("1. 大幅减少可训练参数数量")
    print("2. 保持预训练模型的知识")
    print("3. 快速适应新任务")
    print("4. 存储效率高（只需保存LoRA权重）")
    print("=" * 50)


def compare_lora_vs_full_tuning():
    """比较LoRA和全量微调的参数数量"""
    print("\n" + "=" * 50)
    print("LoRA vs 全量微调参数对比")
    print("=" * 50)
    
    # 创建原始模型（假设全量微调）
    class FullModel(nn.Module):
        def __init__(self, vocab_size, d_model=512, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    # 全量微调模型
    full_model = FullModel(10000)
    full_total, full_trainable = count_parameters(full_model)
    
    print("\n全量微调模型:")
    print(f"可训练参数: {full_trainable:,}")
    
    # LoRA模型
    lora_model = SimpleTransformerWithLoRA(10000, lora_rank=16)
    lora_total, lora_trainable = count_parameters(lora_model)
    
    print("\nLoRA模型:")
    print(f"可训练参数: {lora_trainable:,}")
    
    print(f"\n参数减少比例: {(1 - lora_trainable/full_trainable):.2%}")
    print(f"LoRA效率提升: {full_trainable/lora_trainable:.1f}x")


if __name__ == "__main__":
    # 运行LoRA演示
    lora_training_demo()
    
    # 参数对比
    compare_lora_vs_full_tuning()
    
    print("\n" + "=" * 50)
    print("LoRA实现要点:")
    print("1. 冻结预训练权重，只训练低秩矩阵A和B")
    print("2. rank参数控制低秩维度，越小参数越少")
    print("3. alpha参数控制LoRA的影响强度")
    print("4. 可以选择性地对不同层应用LoRA")
    print("5. 训练完成后可以将LoRA权重合并到原模型")
    print("=" * 50)