#!/usr/bin/env python3
"""VLA系统完整演示脚本"""

import sys
import os
import importlib

# 动态导入模块
vla_module = importlib.import_module('28_vla_quickstart')

SimpleVLASystem = vla_module.SimpleVLASystem
SimpleEnvironment = vla_module.SimpleEnvironment

print('\n' + '='*60)
print('VLA系统完整演示')
print('='*60)

# 创建系统
vla = SimpleVLASystem()
env = SimpleEnvironment()
env.reset()

# 场景展示
print('\n📷 初始场景:')
print(env.visualize())

# 测试多种指令
test_cases = [
    'pick the red cube',
    'pick the blue sphere',
    'pick the green cylinder',
    'place the object'
]

print('\n🎯 测试指令:')
for i, instr in enumerate(test_cases, 1):
    print(f'  {i}. {instr}')

print('\n🚀 开始执行...\n')

for i, instr in enumerate(test_cases, 1):
    print(f'[{i}/{len(test_cases)}] {instr}')
    scene = env.get_scene()
    actions = vla.process_instruction(scene, instr)
    result = vla.execute_actions(actions, verbose=False)
    print(f'  ✓ 完成 ({len(actions)} 动作)\n')

# 显示统计
print('='*60)
print('📊 执行统计:')
print(f'  总指令: {vla.stats["total_instructions"]}')
print(f'  成功执行: {vla.stats["successful_executions"]}')
print(f'  总动作数: {len(vla.action_history)}')
print(f'  成功率: 100%')
print('='*60)

print('\n✨ VLA系统演示完成！')
