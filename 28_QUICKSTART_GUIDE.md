# VLA系统 - 5分钟快速上手指南

## 🎯 什么是VLA系统？

**VLA (Vision-Language-Action)** 是一个端到端的多模态智能系统，能够：
- 👁️ **看懂场景** - 识别物体、理解空间关系
- 💬 **听懂指令** - 理解自然语言命令
- 🤖 **执行动作** - 规划并执行机器人动作

## ⚡ 30秒快速开始

```bash
# 1. 进入项目目录
cd /path/to/learn_python-1

# 2. 运行演示（无需任何依赖！）
python3 28_vla_quickstart.py

# 3. 选择 "1" 查看基础演示
```

## 📝 支持的指令

### 拾取指令
```
"pick the red cube"       # 拾取红色方块
"grab the blue sphere"    # 抓取蓝色球体
"grasp the green cylinder"# 抓取绿色圆柱
```

### 放置指令
```
"place the object"        # 放置物体
"put it down"            # 放下物体
```

### 移动指令
```
"move the cube to left"   # 移动方块到左边
```

## 🎮 交互模式

```bash
python3 28_vla_quickstart.py
# 选择 "3" 进入交互模式

# 然后可以输入:
🎤 请输入指令: pick the red cube
🎤 请输入指令: place the object
🎤 请输入指令: scene          # 查看当前场景
🎤 请输入指令: reset          # 重置场景
🎤 请输入指令: stats          # 查看统计
🎤 请输入指令: quit           # 退出
```

## 💻 代码示例

### 示例1: 最简单的使用

```python
from 28_vla_quickstart import SimpleVLASystem, SimpleEnvironment

# 创建系统和环境
vla = SimpleVLASystem()
env = SimpleEnvironment()
env.reset()

# 获取场景
scene = env.get_scene()

# 处理指令
actions = vla.process_instruction(scene, "pick the red cube")

# 执行动作
vla.execute_actions(actions)
```

### 示例2: 处理多个指令

```python
instructions = [
    "pick the red cube",
    "place the object",
    "pick the blue sphere"
]

for instruction in instructions:
    scene = env.get_scene()
    actions = vla.process_instruction(scene, instruction)
    result = vla.execute_actions(actions, verbose=True)
    print(f"完成！执行了 {len(actions)} 个动作")
```

### 示例3: 查看系统统计

```python
# 执行一些指令后...
stats = vla.stats
print(f"总指令数: {stats['total_instructions']}")
print(f"成功执行: {stats['successful_executions']}")
print(f"总动作数: {len(vla.action_history)}")
```

## 🏗️ 系统工作流程

```
1. 输入
   └─> 视觉场景 + 语言指令

2. 理解
   ├─> 视觉编码: 识别物体、位置、颜色
   └─> 语言编码: 提取意图、目标物体

3. 融合
   └─> 多模态融合: 结合视觉和语言信息

4. 规划
   └─> 动作解码: 生成动作序列

5. 执行
   └─> 机器人执行: MOVE → GRASP → MOVE → RELEASE
```

## 📊 输出示例

```
📷 当前场景:
==================================================
场景 #1
==================================================
机器人位置: (0.0, 0.0, 20.0)

物体列表 (共 5 个):
  obj_0: red cube
    位置: (14.3, -14.2, 12.8)
  obj_1: green cylinder
    位置: (-27.7, 21.8, 6.9)
  obj_2: yellow cube
    位置: (-20.3, -20.6, 8.8)
  obj_3: blue sphere
    位置: (-28.5, -18.1, 5.4)
  obj_4: yellow cylinder
    位置: (-4.2, -26.9, 12.4)
==================================================

🎤 请输入指令: pick the red cube

🤖 执行 4 个动作:
  1. move_to -> (14.3, -14.2, 22.8) (1.0s)
  2. move_to -> (14.3, -14.2, 12.8) (0.5s)
  3. grasp (0.3s)
  4. move_to -> (14.3, -14.2, 22.8) (0.5s)

✅ 完成
```

## 🎨 可用的物体类型

- **形状**: `cube` (方块), `sphere` (球体), `cylinder` (圆柱)
- **颜色**: `red` (红), `green` (绿), `blue` (蓝), `yellow` (黄)

## 🔍 常见问题

### Q1: 如何查看当前场景？
```python
print(env.visualize())
# 或在交互模式输入: scene
```

### Q2: 如何重置场景？
```python
env.reset()
# 或在交互模式输入: reset
```

### Q3: 支持哪些指令格式？
```python
# 所有这些都可以:
"pick the red cube"      ✅
"pick red cube"          ✅
"pick cube"              ✅
"pick the red"           ✅
"grab the blue sphere"   ✅
"place object"           ✅
```

### Q4: 找不到指定物体怎么办？
系统会自动选择最匹配的物体，如果没有完全匹配，会选择：
1. 颜色匹配的物体
2. 形状匹配的物体
3. 第一个可用物体

## 🚀 进阶使用

### 自定义场景

```python
from 28_vla_quickstart import *

env = SimpleEnvironment()
env.objects = [
    VisualObject("obj1", ObjectType.CUBE, (10, 0, 5), "red", 5.0),
    VisualObject("obj2", ObjectType.SPHERE, (-10, 0, 5), "blue", 4.0),
]

print(env.visualize())
```

### 记录和分析动作历史

```python
# 执行一些指令后
for i, action in enumerate(vla.action_history, 1):
    print(f"{i}. {action.action_type.value}")
    if action.target_position:
        print(f"   位置: {action.target_position}")
```

## 📱 命令行快捷方式

```bash
# 基础演示
python3 28_run_demo.py

# 完整演示 (按顺序运行所有演示)
echo "4" | python3 28_vla_quickstart.py

# 交互模式
echo "3" | python3 28_vla_quickstart.py
```

## 🎓 学习路径

1. **第1步**: 运行基础演示 (28_run_demo.py)
2. **第2步**: 尝试交互模式 (选择3)
3. **第3步**: 阅读代码示例
4. **第4步**: 修改场景和指令
5. **第5步**: 查看完整文档 (28_README_VLA_System.md)

## 💡 提示和技巧

1. **多试几种指令**: 系统支持多种表达方式
2. **观察动作序列**: 理解系统如何规划动作
3. **实验不同场景**: reset后会生成随机物体
4. **查看统计信息**: stats命令查看性能指标

## 🔗 相关文档

- 📘 [系统说明](28_README_VLA_System.md)
- 📗 [项目总结](28_PROJECT_SUMMARY_VLA.md)
- 📙 [架构设计](28_VLA_ARCHITECTURE.md)
- 📕 [完整索引](28_INDEX.md)

## 🎉 开始探索！

现在你已经准备好使用VLA系统了！

```bash
python3 28_vla_quickstart.py
```

**祝你玩得开心！** 🚀

---

**快速指南版本**: 1.0  
**最后更新**: 2025-10-17
