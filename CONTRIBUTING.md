# 贡献指南 🤝

感谢你对本项目的关注！我们欢迎所有形式的贡献。

## 📝 贡献方式

你可以通过以下方式为项目做出贡献：

1. **报告Bug** - 发现问题时提交Issue
2. **提出建议** - 分享你的想法和改进建议
3. **修复Bug** - 提交修复问题的Pull Request
4. **添加功能** - 实现新功能或改进现有功能
5. **完善文档** - 改进文档、添加示例、修正错误
6. **分享经验** - 编写教程、制作视频、撰写博客

## 🔧 开发流程

### 1. Fork项目

点击GitHub页面右上角的"Fork"按钮，将项目复制到你的账号下。

### 2. 克隆到本地

```bash
git clone https://github.com/你的用户名/learn_python.git
cd learn_python
```

### 3. 创建分支

为你的改动创建一个新分支：

```bash
git checkout -b feature/你的功能名称
# 或
git checkout -b fix/bug描述
```

### 4. 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"
```

### 5. 进行修改

- 遵循项目的代码风格
- 添加必要的测试
- 更新相关文档

### 6. 运行测试

```bash
# 格式化代码
make format
# 或
black *.py

# 运行测试
make test
# 或
pytest

# 代码检查
make lint
# 或
flake8 *.py
```

### 7. 提交更改

```bash
git add .
git commit -m "简洁明了的提交信息"
```

#### 提交信息规范

使用清晰的提交信息，格式如下：

```
类型: 简短描述（不超过50字符）

详细描述（如果需要）
- 要点1
- 要点2

相关Issue: #123
```

类型包括：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat: 添加新的强化学习算法PPO

- 实现PPO算法核心逻辑
- 添加训练示例
- 更新README文档

相关Issue: #42
```

### 8. 推送到GitHub

```bash
git push origin feature/你的功能名称
```

### 9. 创建Pull Request

1. 访问你的GitHub仓库
2. 点击"New Pull Request"按钮
3. 选择你的分支
4. 填写PR描述，说明改动内容
5. 提交PR等待审核

## 📐 代码规范

### Python代码风格

项目遵循 PEP 8 风格指南，并使用以下工具：

- **Black**: 代码格式化（行长度100）
- **Flake8**: 代码检查
- **MyPy**: 类型检查（可选）

### 代码质量要求

1. **清晰的命名** - 使用有意义的变量和函数名
2. **适当的注释** - 解释复杂逻辑，但避免过度注释
3. **文档字符串** - 为类和函数添加docstring
4. **类型提示** - 尽可能使用类型注解
5. **单一职责** - 每个函数只做一件事

### 示例代码风格

```python
from typing import List, Optional

def calculate_reward(
    state: List[float], 
    action: int, 
    next_state: List[float]
) -> float:
    """
    计算强化学习中的奖励值
    
    Args:
        state: 当前状态向量
        action: 执行的动作
        next_state: 下一个状态向量
        
    Returns:
        float: 计算得到的奖励值
        
    Example:
        >>> state = [0.0, 1.0]
        >>> action = 1
        >>> next_state = [0.5, 1.0]
        >>> reward = calculate_reward(state, action, next_state)
    """
    # 实现逻辑
    reward = sum(next_state) - sum(state)
    return reward
```

## 📚 文档规范

### README文档

- 使用清晰的标题层次
- 提供代码示例
- 包含使用说明
- 标注重要提示

### 代码注释

```python
# 好的注释：解释为什么这样做
# 使用epsilon-greedy策略来平衡探索和利用
if random.random() < epsilon:
    action = random.choice(actions)

# 不好的注释：重复代码
# 如果随机数小于epsilon
if random.random() < epsilon:
    action = random.choice(actions)
```

## 🧪 测试规范

### 编写测试

为新功能添加测试：

```python
def test_q_learning_update():
    """测试Q-Learning更新逻辑"""
    agent = QLearningAgent()
    initial_q = agent.q_table.copy()
    
    # 执行更新
    agent.update(state=0, action=1, reward=1.0, next_state=1)
    
    # 验证Q值已更新
    assert agent.q_table != initial_q
```

### 测试覆盖率

- 目标：代码覆盖率 > 80%
- 重点测试核心算法和关键逻辑
- 边界条件和异常情况

## 📋 Pull Request清单

提交PR前，请确认：

- [ ] 代码遵循项目风格规范
- [ ] 通过所有测试 (`pytest`)
- [ ] 代码格式化完成 (`black`)
- [ ] 通过代码检查 (`flake8`)
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息清晰明确
- [ ] PR描述详细完整

## 🎯 PR审核标准

PR会根据以下标准进行审核：

1. **功能完整性** - 实现了预期功能
2. **代码质量** - 符合规范，易于维护
3. **测试覆盖** - 有足够的测试保证质量
4. **文档完善** - 更新了相关文档
5. **兼容性** - 不破坏现有功能

## 💬 交流讨论

- **提问**: 在Issue中提出问题
- **讨论**: 使用GitHub Discussions
- **反馈**: 在PR中进行代码审查

## 🏆 贡献者

感谢所有为项目做出贡献的开发者！

你的贡献将被记录在项目的贡献者列表中。

## 📄 许可证

通过提交PR，你同意将你的贡献在MIT许可证下发布。

## 🙏 致谢

再次感谢你的贡献！每一个改进都让这个项目变得更好。

---

**Happy Contributing! 期待你的贡献！** 🎉
