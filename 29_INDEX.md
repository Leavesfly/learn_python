# 世界模型 (World Model) - 项目索引

## 📌 快速导航

### 🚀 快速开始
- **新手入门**: [`29_QUICKSTART.md`](29_QUICKSTART.md) - 5分钟快速上手
- **运行演示**: `python 29_world_model_demo.py`
- **查看示例**: 在 `world_model_results/` 目录查看生成的可视化图像

### 📚 文档资源
- **主文档**: [`29_README_WorldModel.md`](29_README_WorldModel.md) - 完整的架构说明和理论背景
- **快速指南**: [`29_QUICKSTART.md`](29_QUICKSTART.md) - 快速上手和参数调优
- **项目总结**: [`29_PROJECT_SUMMARY.md`](29_PROJECT_SUMMARY.md) - 技术架构和实验结果

### 💻 代码文件
- **核心架构**: [`29_world_model_core.py`](29_world_model_core.py) - VQ-VAE, MDN-RNN, Controller
- **环境模拟**: [`29_world_model_env.py`](29_world_model_env.py) - GridWorld, DataCollector
- **完整演示**: [`29_world_model_demo.py`](29_world_model_demo.py) - 端到端工作流程
- **结构测试**: [`29_test_structure.py`](29_test_structure.py) - 代码结构验证

---

## 📖 学习路径

### 初学者路径

1. **理解概念** (15分钟)
   - 阅读 [`29_README_WorldModel.md`](29_README_WorldModel.md) 的"简介"和"核心理念"部分
   - 了解世界模型与传统RL的区别

2. **运行演示** (10分钟)
   ```bash
   # 安装依赖
   pip install torch numpy matplotlib pillow
   
   # 运行演示
   python 29_world_model_demo.py
   ```

3. **查看结果** (5分钟)
   - 观察训练曲线
   - 对比真实环境和梦境预测
   - 理解各个模块的作用

4. **阅读代码** (30分钟)
   - 从 [`29_world_model_core.py`](29_world_model_core.py) 开始
   - 理解 VQ-VAE 的实现
   - 学习 MDN-RNN 的预测机制

### 进阶路径

1. **深入理论** (1小时)
   - 阅读完整的 [`29_README_WorldModel.md`](29_README_WorldModel.md)
   - 理解 VQ 损失函数的推导
   - 学习混合密度网络的原理

2. **参数调优** (1小时)
   - 参考 [`29_QUICKSTART.md`](29_QUICKSTART.md) 的"参数调优指南"
   - 尝试不同的配置
   - 观察对性能的影响

3. **自定义扩展** (2小时)
   - 修改 GridWorld 环境
   - 实现自己的控制器训练算法
   - 添加新的可视化功能

4. **阅读论文** (2小时)
   - World Models: https://arxiv.org/abs/1803.10122
   - VQ-VAE: https://arxiv.org/abs/1711.00937
   - 互动演示: https://worldmodels.github.io/

---

## 🏗️ 项目结构

```
世界模型项目
│
├── 核心代码 (1,413 行)
│   ├── 29_world_model_core.py      # 核心架构 (634行)
│   │   ├── VectorQuantizer         # 向量量化
│   │   ├── VQVAE                   # 表征学习
│   │   ├── MDNRNN                  # 序列预测
│   │   ├── Controller              # 决策控制
│   │   └── WorldModel              # 完整集成
│   │
│   ├── 29_world_model_env.py       # 环境模拟 (366行)
│   │   ├── SimpleGridWorld         # 网格世界
│   │   ├── SimpleCarRacing         # 赛车环境
│   │   └── DataCollector           # 数据收集
│   │
│   └── 29_world_model_demo.py      # 演示程序 (413行)
│       └── Visualizer              # 可视化工具
│
├── 完整文档 (1,478 行)
│   ├── 29_README_WorldModel.md     # 主文档 (450行)
│   ├── 29_QUICKSTART.md            # 快速指南 (473行)
│   ├── 29_PROJECT_SUMMARY.md       # 项目总结 (555行)
│   └── 29_INDEX.md                 # 本文件
│
└── 测试工具
    └── 29_test_structure.py        # 结构测试
```

---

## 🎯 核心组件速查

### VQ-VAE (表征学习)

**文件位置**: [`29_world_model_core.py`](29_world_model_core.py#L48)

**功能**: 将 64×64 RGB 图像压缩到 8×8×32 的潜在空间

**关键方法**:
```python
vae = VQVAE(config)
z_q, vq_loss = vae.encode(observations)      # 编码
x_recon = vae.decode(z_q)                     # 解码
z_flat = vae.get_latent(observations)         # 获取扁平化表征
```

**损失函数**:
- 重构损失: MSE(x_recon, x_original)
- VQ损失: ||z - e||² + β||z - e||²

---

### MDN-RNN (序列预测)

**文件位置**: [`29_world_model_core.py`](29_world_model_core.py#L162)

**功能**: 预测下一状态的概率分布

**关键方法**:
```python
rnn = MDNRNN(config)
mdn_params, hidden = rnn(z, actions, hidden)  # 预测
z_next = rnn.sample(mdn_params['pi'],         # 采样
                    mdn_params['mu'],
                    mdn_params['sigma'])
```

**输出**:
- π (pi): 混合权重
- μ (mu): 均值向量
- σ (sigma): 标准差向量
- reward: 预测奖励
- done: 预测终止

---

### Controller (决策控制)

**文件位置**: [`29_world_model_core.py`](29_world_model_core.py#L344)

**功能**: 基于潜在状态做出决策

**关键方法**:
```python
controller = Controller(config)
action_logits = controller(z, h)              # 获取动作logits
action = controller.get_action(z, h,          # 采样动作
                               deterministic=False)
```

**训练方法**: 策略梯度 (Policy Gradient)

---

### WorldModel (完整集成)

**文件位置**: [`29_world_model_core.py`](29_world_model_core.py#L399)

**功能**: 集成三大组件的完整模型

**关键方法**:
```python
world_model = WorldModel(config)

# 训练流程
world_model.train_vae(observations, epochs=10)
world_model.train_rnn(sequences, epochs=10)
world_model.train_controller(env, episodes=50)

# 梦境生成
dream_data = world_model.dream(initial_obs, actions)

# 模型保存/加载
world_model.save("model.pt")
world_model.load("model.pt")
```

---

## 🔧 配置参数速查

### 基础配置
```python
from world_model_core_29 import WorldModelConfig

config = WorldModelConfig(
    # VQ-VAE 配置
    image_size=64,              # 图像大小
    latent_dim=32,              # 潜在维度
    num_embeddings=512,         # 码本大小
    commitment_cost=0.25,       # 承诺损失系数
    
    # MDN-RNN 配置
    hidden_size=256,            # RNN隐藏层大小
    num_mixtures=5,             # 混合高斯数量
    sequence_length=32,         # 序列长度
    
    # Controller 配置
    action_dim=4,               # 动作空间维度
    controller_hidden=128,      # 控制器隐藏层
    
    # 训练配置
    learning_rate=1e-3,         # 学习率
    batch_size=32,              # 批次大小
    device="cpu"                # 设备 (cpu/cuda)
)
```

### 性能调优建议

| 场景 | latent_dim | hidden_size | num_embeddings | 训练时间 |
|------|-----------|-------------|----------------|---------|
| 快速实验 | 16 | 128 | 256 | ~5分钟 |
| 标准配置 | 32 | 256 | 512 | ~10分钟 |
| 高性能 | 64 | 512 | 1024 | ~30分钟 |

---

## 📊 可视化输出

运行演示后，在 `world_model_results/` 目录生成：

### 1. 训练曲线 (`training_curves_*.png`)
- VQ-VAE 训练损失
- MDN-RNN 训练损失
- Controller 训练奖励

### 2. 重构对比 (`reconstruction_*.png`)
- 原始观察 vs VAE重构
- 验证表征学习质量

### 3. 梦境序列 (`dream_sequence_*.png`)
- 世界模型想象的未来
- 每步的预测奖励

### 4. 真实vs梦境 (`real_vs_dream_*.png`)
- 真实环境轨迹
- 梦境预测轨迹
- 对比预测准确度

---

## 🐛 常见问题

### Q: 运行提示 "ModuleNotFoundError: No module named 'torch'"

**A**: 需要安装依赖
```bash
pip install torch numpy matplotlib pillow
```

### Q: 训练损失不下降怎么办？

**A**: 尝试：
1. 降低学习率: `config.learning_rate = 1e-4`
2. 收集更多数据: `num_episodes=200`
3. 训练更长时间: `epochs=20`

### Q: 梦境预测不准确？

**A**: 这是正常现象，可以：
1. 增加模型容量: `hidden_size=512`
2. 使用更大的码本: `num_embeddings=1024`
3. 只信任短期预测（前5步）

### Q: 如何自定义环境？

**A**: 参考 [`29_world_model_env.py`](29_world_model_env.py) 中的 `SimpleGridWorld`，实现：
- `reset()`: 返回初始观察
- `step(action)`: 执行动作，返回 (obs, reward, done, info)
- `action_space`: 动作空间大小

---

## 📚 参考资料

### 论文
- [World Models (2018)](https://arxiv.org/abs/1803.10122)
- [VQ-VAE (2017)](https://arxiv.org/abs/1711.00937)
- [Mixture Density Networks (1994)](http://publications.aston.ac.uk/id/eprint/373/)

### 互动资源
- [World Models 博客](https://worldmodels.github.io/)
- [互动演示](https://dylandjian.github.io/world-models/)

### 代码仓库
- [原始实现 (TensorFlow)](https://github.com/worldmodels/worldmodels.github.io)
- [PyTorch 实现](https://github.com/ctallec/world-models)
- [DreamerV2](https://github.com/danijar/dreamerv2)

---

## 🎓 与课程其他模块的关系

```
学习路径:
├── [1_*.py] Python 基础
├── [6_pytorch_*.py] PyTorch 入门
├── [12_rl_*.py] 强化学习 (DQN)
├── [15-18_*.py] 智能体系统
├── [27_embodied_*.py] 具身智能
└── [29_*.py] 世界模型 ← 当前位置
```

**前置知识**:
- Python 基础语法
- PyTorch 张量操作
- 强化学习基本概念
- 神经网络训练流程

**后续扩展**:
- Dreamer 算法
- MuZero (隐式世界模型)
- 模型预测控制 (MPC)

---

## ✅ 检查清单

开始使用前，确认：

- [ ] Python 3.8+ 已安装
- [ ] 依赖库已安装 (torch, numpy, matplotlib, pillow)
- [ ] 阅读了快速开始指南
- [ ] 理解了三大核心组件的作用
- [ ] 运行了结构测试: `python 29_test_structure.py`

准备就绪后：

- [ ] 运行演示: `python 29_world_model_demo.py`
- [ ] 查看生成的可视化结果
- [ ] 阅读完整文档: `29_README_WorldModel.md`
- [ ] 尝试调整参数
- [ ] 实现自定义环境

---

## 📞 获取帮助

1. 先查看 [`29_QUICKSTART.md`](29_QUICKSTART.md) 的"常见问题排查"
2. 阅读 [`29_README_WorldModel.md`](29_README_WorldModel.md) 的"常见问题"
3. 检查 [`29_PROJECT_SUMMARY.md`](29_PROJECT_SUMMARY.md) 的技术细节

---

## 🎉 开始学习

**推荐第一步**: 运行演示程序
```bash
python 29_world_model_demo.py
```

然后查看生成的可视化结果，理解世界模型的工作原理！

---

**更新日期**: 2025-10-21  
**项目版本**: v1.0  
**维护者**: AI Learning Project
