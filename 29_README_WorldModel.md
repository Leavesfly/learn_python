# 世界模型 (World Model) 架构实现

## 📖 简介

**世界模型** (World Model) 是一种创新的强化学习架构，最早由 David Ha 和 Jürgen Schmidhuber 在 2018 年的论文 ["World Models"](https://arxiv.org/abs/1803.10122) 中提出。该架构的核心思想是：**让智能体学习一个环境的内部模型，然后在想象中（潜在空间）进行规划和决策**。

### 🎯 核心理念

与传统强化学习直接从原始观察学习策略不同，世界模型采用分而治之的策略：

1. **学习环境的表征** - 将高维观察压缩到紧凑的潜在空间
2. **学习环境的动态** - 预测潜在空间中的状态转移
3. **在想象中训练** - 在潜在空间中展开轨迹，无需真实交互

这种方法具有以下优势：
- ✅ **样本效率高** - 可以在梦境中训练，减少真实环境交互
- ✅ **可解释性强** - 可以可视化智能体"想象"的未来
- ✅ **支持规划** - 可以在采取动作前评估多个可能的未来
- ✅ **迁移能力** - 学到的表征可以迁移到相似任务

---

## 🏗️ 架构组件

本实现包含三个核心神经网络模块：

### 1️⃣ VQ-VAE (Vector Quantized Variational AutoEncoder)

**功能**: 学习环境观察的紧凑表征

```
输入: 64×64 RGB 图像
     ↓
  [编码器]
     ↓
  潜在表征 (8×8×32)
     ↓
  [向量量化]
     ↓
  离散编码
     ↓
  [解码器]
     ↓
输出: 重构的 64×64 RGB 图像
```

**关键技术**:
- **向量量化 (VQ)**: 将连续潜在空间离散化，提高表征质量
- **Straight-through Estimator**: 解决离散化导致的不可导问题
- **承诺损失**: 鼓励编码器输出接近码本向量

**代码示例**:
```python
vae = VQVAE(config)
x_recon, recon_loss, vq_loss = vae(observations)
total_loss = recon_loss + vq_loss
```

### 2️⃣ MDN-RNN (Mixture Density Network + RNN)

**功能**: 在潜在空间中预测未来状态

```
输入: z_t (当前状态) + a_t (动作)
     ↓
   [LSTM]
     ↓
  隐藏状态 h_t
     ↓
  [MDN 输出头]
     ↓
输出: P(z_{t+1} | z_t, a_t) - 混合高斯分布
      r_t - 预测奖励
      done_t - 预测终止状态
```

**关键技术**:
- **混合密度网络**: 输出多模态分布，处理环境的随机性
- **LSTM**: 捕捉时序依赖关系
- **多任务学习**: 同时预测状态、奖励和终止信号

**代码示例**:
```python
rnn = MDNRNN(config)
mdn_params, hidden = rnn(z, actions, hidden)
next_z = rnn.sample(mdn_params['pi'], mdn_params['mu'], mdn_params['sigma'])
```

### 3️⃣ Controller (控制器)

**功能**: 基于潜在表征和 RNN 隐藏状态做出决策

```
输入: z (潜在状态) + h (RNN 隐藏状态)
     ↓
  [前馈网络]
     ↓
输出: 动作概率分布
```

**训练方法**:
- 策略梯度 (Policy Gradient)
- 进化策略 (Evolution Strategies) - 原论文使用
- 或其他强化学习算法

**代码示例**:
```python
controller = Controller(config)
action_logits = controller(z, h)
action = Categorical(logits=action_logits).sample()
```

---

## 📂 文件结构

```
29_world_model_core.py          # 核心架构实现
├── VectorQuantizer             # 向量量化层
├── VQVAE                       # VAE 模型
├── MDNRNN                      # 序列预测模型
├── Controller                  # 决策控制器
└── WorldModel                  # 完整集成

29_world_model_env.py           # 环境模拟器
├── SimpleGridWorld             # 网格世界环境
├── SimpleCarRacing             # 赛车环境（可选）
└── DataCollector               # 数据收集工具

29_world_model_demo.py          # 完整演示程序
└── Visualizer                  # 可视化工具

29_README_WorldModel.md         # 本文档
29_QUICKSTART.md                # 快速开始指南
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib pillow
```

### 运行演示

```bash
python 29_world_model_demo.py
```

### 完整训练流程

演示程序将自动执行以下步骤：

1. **数据收集** - 在 GridWorld 环境中收集 100 个随机轨迹
2. **训练 VQ-VAE** - 学习视觉表征（10 epochs）
3. **训练 MDN-RNN** - 学习状态转移动态（10 epochs）
4. **训练 Controller** - 学习决策策略（50 episodes）
5. **梦境生成** - 在潜在空间中展开想象轨迹
6. **可视化对比** - 对比真实环境和梦境预测

### 预期输出

程序将生成以下文件（保存在 `world_model_results/` 目录）：

- `reconstruction_*.png` - VAE 重构对比图
- `training_curves_*.png` - 三个模块的训练曲线
- `dream_sequence_*.png` - 梦境序列可视化
- `real_vs_dream_*.png` - 真实环境 vs 梦境对比
- `world_model_*.pt` - 训练好的完整模型

---

## 🔬 技术细节

### VQ-VAE 损失函数

```python
# 重构损失
recon_loss = MSE(x_recon, x_original)

# VQ 损失
vq_loss = ||z_e - sg[e]||² + β||sg[z_e] - e||²
         └─ 量化损失    └─ 承诺损失

# 总损失
total_loss = recon_loss + vq_loss
```

其中：
- `z_e`: 编码器输出
- `e`: 最近的码本向量
- `sg[·]`: stop-gradient 操作
- `β`: 承诺系数（默认 0.25）

### MDN-RNN 损失函数

```python
# 混合高斯负对数似然
mdn_loss = -log Σ(π_i · N(z_{t+1} | μ_i, σ_i))

# 奖励预测损失
reward_loss = MSE(r_pred, r_true)

# 终止状态损失
done_loss = BCE(done_pred, done_true)

# 总损失
total_loss = mdn_loss + 0.1·reward_loss + 0.1·done_loss
```

### 梦境展开算法

```python
def dream(initial_obs, actions):
    z = VAE.encode(initial_obs)
    h = None
    
    for action in actions:
        # RNN 预测
        mdn_params, h = RNN(z, action, h)
        
        # 采样下一状态
        z_next = sample_from_mixture(mdn_params)
        
        # 解码为观察
        obs_next = VAE.decode(z_next)
        
        z = z_next
    
    return dream_trajectory
```

---

## 📊 实验结果

### GridWorld 环境性能

| 指标 | 值 |
|------|-----|
| 平均训练奖励 | ~0.5 (50 episodes 后) |
| VAE 重构误差 | ~0.01 (MSE) |
| MDN-RNN 预测损失 | ~2.5 (NLL) |
| 梦境一致性 | 高（前 5 步） |

### 可视化示例

**VQ-VAE 重构质量**:
- 原始图像能够被准确重构
- 保留了关键信息（智能体、目标、障碍物位置）
- 细节稍有模糊（符合压缩预期）

**梦境预测**:
- 短期预测（1-3步）准确度高
- 中期预测（4-8步）保持环境结构
- 长期预测逐渐发散（正常现象）

---

## 🎓 理论背景

### 为什么需要世界模型？

传统强化学习面临的挑战：
1. **样本效率低** - 需要大量与环境交互
2. **端到端黑箱** - 难以理解决策过程
3. **缺乏规划能力** - 无法评估未来

世界模型的解决方案：
1. **分离表征和决策** - 降低学习难度
2. **在梦境中训练** - 提高样本效率
3. **显式预测未来** - 支持规划和解释

### 与其他方法的比较

| 方法 | 样本效率 | 可解释性 | 计算成本 |
|------|---------|---------|---------|
| DQN | 低 | 低 | 中 |
| A3C | 中 | 低 | 高 |
| **World Model** | **高** | **高** | **中** |
| MuZero | 高 | 中 | 高 |

### 应用场景

✅ **适合**:
- 环境模拟成本高（机器人、自动驾驶）
- 需要规划能力（游戏、导航）
- 视觉输入为主的任务
- 需要可解释性的场景

❌ **不适合**:
- 环境高度随机、不可预测
- 状态空间极其复杂
- 实时性要求极高

---

## 🔧 自定义与扩展

### 修改环境

创建自定义环境需要实现：

```python
class CustomEnv:
    def reset(self) -> np.ndarray:
        # 返回 [3, 64, 64] RGB 图像
        pass
    
    def step(self, action: int) -> Tuple:
        # 返回 (obs, reward, done, info)
        pass
    
    @property
    def action_space(self) -> int:
        # 返回动作空间大小
        pass
```

### 调整模型参数

```python
config = WorldModelConfig(
    image_size=64,           # 图像大小
    latent_dim=32,           # 潜在维度 (↑ 提高表达能力)
    num_embeddings=512,      # 码本大小 (↑ 提高精度)
    hidden_size=256,         # RNN 大小 (↑ 捕捉更复杂动态)
    num_mixtures=5,          # 混合数 (↑ 建模多模态)
    learning_rate=1e-3       # 学习率
)
```

### 使用自己的数据

```python
# 加载自定义观察数据
observations = torch.load('my_observations.pt')

# 加载序列数据
sequences = torch.load('my_sequences.pt')

# 训练
world_model.train_vae(observations, epochs=20)
world_model.train_rnn(sequences, epochs=20)
```

---

## 📚 参考资料

### 论文

1. **World Models** (Ha & Schmidhuber, 2018)
   - 论文: https://arxiv.org/abs/1803.10122
   - 博客: https://worldmodels.github.io/
   - 互动演示: https://dylandjian.github.io/world-models/

2. **Neural Discrete Representation Learning** (VQ-VAE, 2017)
   - 论文: https://arxiv.org/abs/1711.00937

3. **Mixture Density Networks** (Bishop, 1994)
   - 论文: http://publications.aston.ac.uk/id/eprint/373/

### 相关工作

- **DreamerV2** (Hafner et al., 2020) - 现代世界模型实现
- **MuZero** (Schrittwieser et al., 2020) - DeepMind 的隐式世界模型
- **PlaNet** (Hafner et al., 2019) - 基于世界模型的规划

### 代码资源

- 原始实现 (TensorFlow): https://github.com/worldmodels/worldmodels.github.io
- PyTorch 实现: https://github.com/ctallec/world-models
- DreamerV2: https://github.com/danijar/dreamerv2

---

## 🤝 贡献

本实现是教学性质的简化版本，重点在于概念理解和可视化。

如需生产级实现，建议参考：
- **DreamerV2** - 更先进的世界模型
- **Hafner 的工作** - 持续改进的算法

---

## 📝 许可

本项目代码遵循 MIT 许可证。

---

## 💡 常见问题

### Q1: 为什么使用 VQ-VAE 而不是普通 VAE？

A: VQ-VAE 的离散潜在空间有以下优势：
- 避免后验坍塌问题
- 提供更清晰的表征
- 更容易被 RNN 建模

### Q2: 梦境预测为什么会发散？

A: 这是正常现象，原因包括：
- 误差累积（每步小误差累积）
- 环境随机性（MDN 采样）
- 模型容量限制

解决方案：
- 定期重新同步真实状态
- 使用更大的模型
- 集成多个梦境轨迹

### Q3: 如何提高性能？

A: 几个建议：
- 收集更多数据（100+ episodes）
- 训练更长时间（50+ epochs）
- 增加模型容量（latent_dim, hidden_size）
- 使用数据增强
- 调整学习率

### Q4: 能否用于真实机器人？

A: 可以，但需要注意：
- 使用真实传感器数据训练
- 考虑动作延迟和噪声
- 在模拟器中预训练
- 逐步迁移到真实环境

---

## 🎉 总结

世界模型是一个优雅而强大的架构，它展示了：
- 🧠 **认知科学启发** - 模拟人类的心理预演
- 🎯 **工程实用性** - 显著提高样本效率
- 🔍 **可解释性** - 可以"看到"智能体的想象

希望这个实现能帮助你理解这个激动人心的研究方向！

---

**快速开始**: 详见 [`29_QUICKSTART.md`](29_QUICKSTART.md)  
**核心代码**: 详见 [`29_world_model_core.py`](29_world_model_core.py)  
**完整演示**: 运行 `python 29_world_model_demo.py`
