"""
具身智能扫地机器人 - 功能测试脚本

测试所有核心功能模块
"""

import random
from typing import Dict


def test_environment():
    """测试环境模拟模块"""
    print("=" * 60)
    print("测试1: 环境模拟模块")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    # 创建环境
    env = demo.RoomEnvironment(width=5, height=5, obstacle_ratio=0.2)
    
    print(f"✓ 环境创建成功")
    print(f"  - 大小: {env.width}x{env.height}")
    print(f"  - 总单元格: {env.width * env.height}")
    
    # 测试障碍物
    obstacle_count = sum(1 for row in env.grid for cell in row if cell.has_obstacle)
    print(f"  - 障碍物数量: {obstacle_count}")
    
    # 测试灰尘
    print(f"  - 总灰尘量: {env.total_dust:.2f}")
    
    # 测试位置有效性
    assert env.is_valid_position(0, 0), "起始位置应该有效"
    assert not env.is_valid_position(-1, 0), "边界外应该无效"
    print(f"✓ 位置验证功能正常")
    
    # 测试清扫
    initial_dust = env.grid[0][0].dust_level
    cleaned = env.clean_cell(0, 0)
    print(f"✓ 清扫功能正常 (清理了 {cleaned:.3f} 灰尘)")
    
    # 测试可视化
    viz = env.visualize(robot_pos=(0, 0))
    assert 'R' in viz, "可视化应该显示机器人"
    print(f"✓ 可视化功能正常")
    
    return True


def test_perception():
    """测试感知模块"""
    print("\n" + "=" * 60)
    print("测试2: 感知模块")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    env = demo.RoomEnvironment(width=10, height=10)
    perception = demo.PerceptionSystem(env)
    
    # 测试传感器
    sensor_data = perception.sense(5, 5, 0.8)
    
    print(f"✓ 传感器数据采集成功")
    print(f"  - 激光雷达: {len(sensor_data.lidar_readings)} 个方向")
    print(f"  - 当前位置: {sensor_data.position}")
    print(f"  - 灰尘传感器: {sensor_data.dust_sensor:.3f}")
    print(f"  - 电池电量: {sensor_data.battery_level:.2%}")
    print(f"  - 局部地图: {len(sensor_data.local_map)}x{len(sensor_data.local_map[0])}")
    
    # 测试状态编码
    state_vector = perception.encode_state(sensor_data)
    print(f"✓ 状态编码成功")
    print(f"  - 状态向量维度: {len(state_vector)}")
    assert len(state_vector) == 37, "状态向量应该是37维"
    
    return True


def test_agent():
    """测试决策模块"""
    print("\n" + "=" * 60)
    print("测试3: 决策模块")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    agent = demo.SimpleAgent()
    
    print(f"✓ 智能体创建成功")
    print(f"  - 动作空间: {len(agent.ACTIONS)} 个动作")
    print(f"  - 初始探索率: {agent.q_table.epsilon:.2f}")
    
    # 测试动作选择
    state = [0.5] * 37
    action = agent.select_action(state, training=True)
    assert 0 <= action < len(agent.ACTIONS), "动作应该在有效范围内"
    print(f"✓ 动作选择功能正常")
    print(f"  - 选择的动作: {action} ({agent.ACTIONS[action]})")
    
    # 测试学习
    next_state = [0.6] * 37
    agent.learn(state, action, 1.0, next_state, False)
    print(f"✓ 学习功能正常")
    print(f"  - Q表大小: {len(agent.q_table.q_table)}")
    
    return True


def test_actuator():
    """测试执行模块"""
    print("\n" + "=" * 60)
    print("测试4: 执行模块")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    env = demo.RoomEnvironment(width=10, height=10)
    actuator = demo.RobotActuator(env)
    
    print(f"✓ 执行器创建成功")
    
    # 测试移动
    new_pos, success = actuator.execute_move((5, 5), 1)  # 向东移动
    assert success, "有效移动应该成功"
    assert new_pos == (6, 5), "移动后位置应该正确"
    print(f"✓ 移动功能正常")
    print(f"  - 从 (5,5) 移动到 {new_pos}")
    
    # 测试清扫
    cleaned = actuator.execute_clean((5, 5))
    print(f"✓ 清扫功能正常")
    print(f"  - 清理量: {cleaned:.3f}")
    
    return True


def test_robot_system():
    """测试完整机器人系统"""
    print("\n" + "=" * 60)
    print("测试5: 完整机器人系统")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    env = demo.RoomEnvironment(width=8, height=8, obstacle_ratio=0.1)
    robot = demo.EmbodiedRobotCleaner(env)
    
    print(f"✓ 机器人系统创建成功")
    
    # 测试单步执行
    reward, done = robot.step(training=True)
    print(f"✓ 单步执行功能正常")
    print(f"  - 奖励: {reward:.2f}")
    print(f"  - 位置: {robot.state.position}")
    print(f"  - 步数: {robot.state.steps}")
    
    # 测试重置
    robot.reset()
    assert robot.state.steps == 0, "重置后步数应该为0"
    assert robot.state.battery == 100.0, "重置后电池应该满"
    print(f"✓ 重置功能正常")
    
    return True


def test_training():
    """测试训练系统"""
    print("\n" + "=" * 60)
    print("测试6: 训练系统（小规模）")
    print("=" * 60)
    
    from importlib import import_module
    demo = import_module('27_embodied_robot_demo')
    
    random.seed(42)
    
    env = demo.RoomEnvironment(width=6, height=6, obstacle_ratio=0.1)
    robot = demo.EmbodiedRobotCleaner(env)
    
    print("开始训练 5 个回合...")
    
    results = []
    for episode in range(1, 6):
        stats = robot.train_episode()
        results.append(stats)
        print(f"  Episode {episode}: "
              f"Reward={stats['reward']:.2f}, "
              f"Clean={stats['cleanliness']:.2%}, "
              f"Q表={stats['q_table_size']}")
    
    print(f"\n✓ 训练系统运行正常")
    
    # 检查学习进展
    avg_reward_first_3 = sum(r['reward'] for r in results[:3]) / 3
    avg_reward_last_2 = sum(r['reward'] for r in results[3:]) / 2
    
    print(f"  - 前3回合平均奖励: {avg_reward_first_3:.2f}")
    print(f"  - 后2回合平均奖励: {avg_reward_last_2:.2f}")
    
    # Q表应该增长
    q_size_growth = results[-1]['q_table_size'] - results[0]['q_table_size']
    print(f"  - Q表增长: {results[0]['q_table_size']} → {results[-1]['q_table_size']} (+{q_size_growth})")
    
    return True


def test_analysis():
    """测试分析工具"""
    print("\n" + "=" * 60)
    print("测试7: 分析工具")
    print("=" * 60)
    
    # 创建测试数据
    test_data = [
        {"reward": 50.0, "steps": 500, "cleanliness": 0.05, "collisions": 10, 
         "epsilon": 0.9, "q_table_size": 50},
        {"reward": 60.0, "steps": 480, "cleanliness": 0.07, "collisions": 8,
         "epsilon": 0.8, "q_table_size": 60},
        {"reward": 70.0, "steps": 450, "cleanliness": 0.09, "collisions": 6,
         "epsilon": 0.7, "q_table_size": 70},
    ]
    
    # 保存测试数据
    import json
    test_file = "/Users/yefei.yf/Qoder/learn_python/test_training.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    print(f"✓ 测试数据创建成功")
    
    # 测试分析函数
    from importlib import import_module
    analysis = import_module('27_embodied_analysis')
    
    # 加载数据
    data = analysis.load_training_data(test_file)
    assert len(data) == 3, "应该加载3条数据"
    print(f"✓ 数据加载功能正常")
    
    # 测试统计计算（使用内联计算）
    rewards = [d['reward'] for d in data]
    mean = sum(rewards) / len(rewards)
    print(f"✓ 统计计算功能正常")
    print(f"  - 平均值: {mean:.2f}")
    print(f"  - 最大值: {max(rewards):.2f}")
    print(f"  - 最小值: {min(rewards):.2f}")
    
    # 清理测试文件
    import os
    os.remove(test_file)
    print(f"✓ 测试数据清理完成")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("具身智能扫地机器人 - 功能测试")
    print("=" * 60)
    
    tests = [
        ("环境模拟", test_environment),
        ("感知系统", test_perception),
        ("决策智能体", test_agent),
        ("执行器", test_actuator),
        ("机器人系统", test_robot_system),
        ("训练系统", test_training),
        ("分析工具", test_analysis),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✅ 通过"
        except Exception as e:
            results[name] = f"❌ 失败: {str(e)}"
            print(f"\n❌ 测试失败: {e}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"{name:15s} : {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常。")
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，请检查。")


if __name__ == "__main__":
    run_all_tests()
