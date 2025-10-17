"""
VLA系统演示 - Vision-Language-Action System Demo

展示VLA系统的完整功能：
1. 场景感知与物体识别
2. 自然语言指令理解
3. 动作序列生成与执行
4. 多任务处理能力
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import time
import importlib
vla_core_module = importlib.import_module('28_vla_core')
from importlib import import_module

# 导入所需类
exec('from ' + '28_vla_core' + ' import *')


def demo_basic_vla():
    """基础VLA功能演示"""
    print("\n" + "=" * 60)
    print("🤖 VLA系统基础功能演示")
    print("=" * 60)
    
    # 初始化系统
    vla = VLASystem()
    env = RobotEnvironment()
    
    # 重置环境
    env.reset()
    
    # 显示场景
    print("\n📷 当前场景:")
    print(env.visualize())
    
    # 测试指令列表
    instructions = [
        "pick the red cube",
        "pick the blue sphere",
        "place the object",
    ]
    
    for i, instruction in enumerate(instructions, 1):
        print(f"\n{'─' * 60}")
        print(f"📝 指令 {i}: {instruction}")
        print(f"{'─' * 60}")
        
        # 获取当前场景
        scene = env.get_current_scene()
        
        # 处理指令
        print("⚙️  处理中...")
        actions = vla.process_instruction(scene, instruction)
        
        print(f"✅ 生成 {len(actions)} 个动作:")
        
        # 执行动作
        result = vla.execute_actions(actions, verbose=True)
        
        print(f"\n📊 执行结果:")
        print(f"  - 成功: {result['success']}")
        print(f"  - 总耗时: {result['total_duration']:.2f}s")
        
        time.sleep(0.5)
    
    # 显示系统指标
    print(f"\n{'=' * 60}")
    print("📈 系统性能指标:")
    metrics = vla.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    print("=" * 60)


def demo_multi_step_task():
    """多步骤任务演示"""
    print("\n" + "=" * 60)
    print("🔄 多步骤任务演示")
    print("=" * 60)
    
    vla = VLASystem()
    env = RobotEnvironment()
    env.reset()
    
    print("\n📷 初始场景:")
    print(env.visualize())
    
    # 复杂任务序列
    task_sequence = [
        ("拾取红色物体", "pick the red cube"),
        ("放置到右侧", "place the object"),
        ("拾取蓝色物体", "pick the blue sphere"),
        ("放置到左侧", "place the object"),
    ]
    
    print(f"\n🎯 任务序列 (共{len(task_sequence)}步):")
    for i, (desc, _) in enumerate(task_sequence, 1):
        print(f"  {i}. {desc}")
    
    total_actions = 0
    start_time = time.time()
    
    for step_num, (description, instruction) in enumerate(task_sequence, 1):
        print(f"\n{'━' * 60}")
        print(f"步骤 {step_num}/{len(task_sequence)}: {description}")
        print(f"{'━' * 60}")
        
        scene = env.get_current_scene()
        actions = vla.process_instruction(scene, instruction)
        
        print(f"生成动作: {len(actions)} 个")
        result = vla.execute_actions(actions, verbose=False)
        
        total_actions += len(actions)
        print(f"✓ 完成 (耗时: {result['total_duration']:.2f}s)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("🏆 任务完成总结:")
    print(f"  - 总步骤数: {len(task_sequence)}")
    print(f"  - 总动作数: {total_actions}")
    print(f"  - 总耗时: {total_time:.2f}s")
    print(f"  - 平均每步耗时: {total_time/len(task_sequence):.2f}s")
    print("=" * 60)


def demo_scene_understanding():
    """场景理解能力演示"""
    print("\n" + "=" * 60)
    print("👁️  场景理解能力演示")
    print("=" * 60)
    
    vla = VLASystem()
    env = RobotEnvironment()
    
    # 创建特定场景
    env.reset()
    env.objects = [
        VisualObject("obj1", ObjectType.CUBE, (10, 0, 5), "red", 5.0),
        VisualObject("obj2", ObjectType.SPHERE, (-10, 0, 5), "blue", 4.0),
        VisualObject("obj3", ObjectType.CYLINDER, (0, 15, 5), "green", 6.0),
    ]
    
    print("\n📷 测试场景:")
    print(env.visualize())
    
    scene = env.get_current_scene()
    
    print("\n🧠 视觉编码测试:")
    vision_features = vla.vision_encoder.encode_scene(scene)
    print(f"  - 特征维度: {len(vision_features)}")
    print(f"  - 特征范围: [{vision_features.min():.3f}, {vision_features.max():.3f}]")
    print(f"  - 特征均值: {vision_features.mean():.3f}")
    
    print("\n💬 语言理解测试:")
    test_instructions = [
        "pick the red cube",
        "grasp the blue sphere",
        "move the green cylinder",
    ]
    
    for instruction in test_instructions:
        lang_features, info = vla.language_encoder.encode_instruction(instruction)
        print(f"\n  指令: '{instruction}'")
        print(f"    - 意图: {info['intent']}")
        print(f"    - 目标: {info['target']}")
        print(f"    - 特征维度: {len(lang_features)}")
    
    print("\n🔗 多模态融合测试:")
    lang_features, _ = vla.language_encoder.encode_instruction("pick the red cube")
    fused_features = vla.fusion_module.fuse(vision_features, lang_features)
    print(f"  - 融合特征维度: {len(fused_features)}")
    print(f"  - 融合特征范围: [{fused_features.min():.3f}, {fused_features.max():.3f}]")
    print("=" * 60)


def demo_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("⚡ 性能基准测试")
    print("=" * 60)
    
    vla = VLASystem()
    env = RobotEnvironment()
    
    num_trials = 20
    instructions = [
        "pick the red cube",
        "pick the blue sphere",
        "pick the green cylinder",
        "place the object",
    ]
    
    print(f"\n🔬 测试配置:")
    print(f"  - 测试次数: {num_trials}")
    print(f"  - 指令种类: {len(instructions)}")
    
    processing_times = []
    action_counts = []
    
    print(f"\n🏃 执行测试...")
    for i in range(num_trials):
        env.reset()
        scene = env.get_current_scene()
        instruction = random.choice(instructions)
        
        start = time.time()
        actions = vla.process_instruction(scene, instruction)
        processing_time = time.time() - start
        
        processing_times.append(processing_time)
        action_counts.append(len(actions))
        
        if (i + 1) % 5 == 0:
            print(f"  完成: {i+1}/{num_trials}")
    
    # 统计分析
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    avg_actions = sum(action_counts) / len(action_counts)
    
    print(f"\n📊 性能统计:")
    print(f"  处理时间:")
    print(f"    - 平均: {avg_time*1000:.2f}ms")
    print(f"    - 最大: {max_time*1000:.2f}ms")
    print(f"    - 最小: {min_time*1000:.2f}ms")
    print(f"  动作生成:")
    print(f"    - 平均动作数: {avg_actions:.1f}")
    print(f"    - 最多: {max(action_counts)}")
    print(f"    - 最少: {min(action_counts)}")
    
    # 系统指标
    metrics = vla.get_metrics()
    print(f"\n  系统指标:")
    print(f"    - 总指令数: {metrics['total_instructions']}")
    print(f"    - 成功执行: {metrics['successful_executions']}")
    print(f"    - 成功率: {metrics['successful_executions']/metrics['total_instructions']*100:.1f}%")
    
    print("=" * 60)


def demo_interactive_mode():
    """交互模式演示"""
    print("\n" + "=" * 60)
    print("🎮 VLA交互模式")
    print("=" * 60)
    
    vla = VLASystem()
    env = RobotEnvironment()
    env.reset()
    
    print("\n欢迎使用VLA交互系统!")
    print("可用指令示例:")
    print("  - pick the red cube")
    print("  - pick the blue sphere")
    print("  - place the object")
    print("  - 输入 'scene' 查看当前场景")
    print("  - 输入 'reset' 重置场景")
    print("  - 输入 'metrics' 查看性能指标")
    print("  - 输入 'quit' 退出")
    
    while True:
        print("\n" + "─" * 60)
        user_input = input("🎤 请输入指令: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 再见!")
            break
        elif user_input.lower() == 'scene':
            print(env.visualize())
        elif user_input.lower() == 'reset':
            env.reset()
            print("✅ 场景已重置")
            print(env.visualize())
        elif user_input.lower() == 'metrics':
            print("📈 性能指标:")
            for k, v in vla.get_metrics().items():
                print(f"  {k}: {v}")
        elif user_input:
            scene = env.get_current_scene()
            print("⚙️  处理中...")
            actions = vla.process_instruction(scene, user_input)
            print(f"✅ 生成 {len(actions)} 个动作:")
            result = vla.execute_actions(actions, verbose=True)
            print(f"✓ 执行完成 (耗时: {result['total_duration']:.2f}s)")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "🌟" * 30)
    print("VLA系统完整演示")
    print("🌟" * 30)
    
    demos = [
        ("基础功能", demo_basic_vla),
        ("多步骤任务", demo_multi_step_task),
        ("场景理解", demo_scene_understanding),
        ("性能基准", demo_performance_benchmark),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'▶' * 30}")
        print(f"演示 {i}/{len(demos)}: {name}")
        print(f"{'▶' * 30}")
        demo_func()
        
        if i < len(demos):
            input("\n按Enter继续下一个演示...")
    
    print("\n" + "🌟" * 30)
    print("所有演示完成!")
    print("🌟" * 30)
    
    # 询问是否进入交互模式
    choice = input("\n是否进入交互模式? (y/n): ").strip().lower()
    if choice == 'y':
        demo_interactive_mode()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "basic":
            demo_basic_vla()
        elif mode == "multistep":
            demo_multi_step_task()
        elif mode == "scene":
            demo_scene_understanding()
        elif mode == "benchmark":
            demo_performance_benchmark()
        elif mode == "interactive":
            demo_interactive_mode()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: basic, multistep, scene, benchmark, interactive")
    else:
        run_all_demos()
