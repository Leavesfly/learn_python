"""
增强版自进化Agent演示
展示更复杂的学习和进化行为
"""

import sys
import os
import time
import json

# 简化导入方式
exec(open('/Users/yefei.yf/Qoder/learn_python/18_self_evolving_agent.py').read())

def advanced_agent_demo():
    """高级Agent演示，展示复杂的学习行为"""
    print("=== 高级自进化Agent演示 ===\n")
    
    agent = SelfEvolvingAgent("高级学习Agent")
    
    # 设置更高的探索率以展示多样化行为
    agent.exploration_rate = 0.4
    
    # 复杂任务序列，包含失败案例
    complex_tasks = [
        # 第一阶段：探索期（高不确定性）
        ("研究量子计算", {'uncertainty': 'high', 'complexity': 'very_high'}),
        ("分析复杂数据", {'data': 'missing', 'format': 'unknown'}),
        ("设计神经网络", {'architecture': 'uncertain', 'task': 'classification'}),
        
        # 第二阶段：学习期（中等不确定性）
        ("优化算法性能", {'algorithm': 'genetic', 'target': 'efficiency'}),
        ("预测市场趋势", {'data': 'historical_prices', 'timeframe': '6_months'}),
        ("自然语言处理", {'text': 'customer_reviews', 'task': 'sentiment'}),
        
        # 第三阶段：应用期（低不确定性）  
        ("计算投资收益", {'expression': '(1000 * 1.08 ** 5) - 1000'}),
        ("分析用户行为", {'data': {'clicks': 1000, 'conversions': 50}}),
        ("搜索技术资料", {'query': 'machine learning optimization'}),
        
        # 第四阶段：复杂组合任务
        ("智能推荐系统", {'users': 10000, 'items': 5000, 'interactions': 'sparse'}),
        ("多模态学习", {'vision': True, 'text': True, 'audio': False}),
        ("强化学习训练", {'env': 'continuous', 'action_space': 'high_dim'}),
        
        # 第五阶段：创新挑战  
        ("创建新算法", {'domain': 'optimization', 'novelty': 'required'}),
        ("跨领域知识融合", {'field1': 'biology', 'field2': 'computing'}),
        ("解决开放问题", {'problem': 'AI_alignment', 'approach': 'unknown'})
    ]
    
    print(f"准备处理 {len(complex_tasks)} 个渐进式复杂任务...\n")
    
    # 记录学习历程
    learning_journey = []
    
    for i, (task, context) in enumerate(complex_tasks, 1):
        print(f"--- 阶段 {(i-1)//3 + 1} | 任务 {i}: {task} ---")
        
        # 显示任务复杂度
        complexity = context.get('complexity', 'medium')
        uncertainty = context.get('uncertainty', 'medium')
        print(f"🎯 复杂度: {complexity} | 不确定性: {uncertainty}")
        
        # 执行前的状态
        pre_performance = agent.get_performance_summary()
        pre_strategies = len(agent.strategies)
        pre_concepts = len(agent.knowledge_graph.nodes)
        
        # 执行任务
        result = agent.process_task(task, context)
        
        # 执行后的状态
        post_performance = agent.get_performance_summary()
        post_strategies = len(agent.strategies)
        post_concepts = len(agent.knowledge_graph.nodes)
        
        # 显示执行结果
        print(f"🤖 选择动作: {result['action']}")
        print(f"✅ 执行结果: {'成功' if result['success'] else '失败'}")
        print(f"🎯 奖励值: {result['reward']:.2f}")
        print(f"💡 反思: {result['learning_insights']}")
        
        # 显示学习变化
        strategy_growth = post_strategies - pre_strategies
        concept_growth = post_concepts - pre_concepts
        if strategy_growth > 0 or concept_growth > 0:
            print(f"📈 学习增长: +{strategy_growth}策略 +{concept_growth}概念")
        
        # 记录学习历程
        learning_journey.append({
            'stage': (i-1)//3 + 1,
            'task_num': i,
            'task': task,
            'action': result['action'],
            'success': result['success'],
            'reward': result['reward'],
            'strategies': post_strategies,
            'concepts': post_concepts,
            'success_rate': post_performance.get('current_success_rate', 0)
        })
        
        # 阶段性进化展示
        if i % 3 == 0:
            print(f"\n🧠 第{(i-1)//3 + 1}阶段完成 - 进化总结:")
            stage_performance = agent.get_performance_summary()
            print(f"  当前成功率: {stage_performance['current_success_rate']:.1%}")
            print(f"  策略库规模: {stage_performance['strategies_count']}")
            print(f"  知识概念数: {stage_performance['knowledge_concepts']}")
            print(f"  探索率: {stage_performance['exploration_rate']:.2f}")
            
            # 显示最有效的策略
            effective_strategies = []
            for name, strategy in agent.strategies.items():
                if strategy.usage_count > 0 and strategy.success_rate > 0.5:
                    effective_strategies.append((name, strategy.success_rate, strategy.usage_count))
            
            if effective_strategies:
                effective_strategies.sort(key=lambda x: x[1], reverse=True)
                print(f"  🏆 最佳策略: {effective_strategies[0][0][:30]}... (成功率{effective_strategies[0][1]:.1%})")
            
            # 触发深度进化
            if i % 6 == 0:
                print(f"  🔄 触发深度进化...")
                agent.self_evolve()
            
            print()
        else:
            print()
    
    # 最终学习分析
    print("=" * 60)
    print("🎓 最终学习分析报告")
    print("=" * 60)
    
    final_performance = agent.get_performance_summary()
    
    print(f"\n📊 整体表现:")
    print(f"  总处理任务: {final_performance['total_tasks']}")
    print(f"  最终成功率: {final_performance['current_success_rate']:.1%}")
    print(f"  性能趋势: {final_performance['trend']}")
    
    print(f"\n🧠 知识获得:")
    print(f"  学会策略数: {final_performance['strategies_count']}")
    print(f"  掌握概念数: {final_performance['knowledge_concepts']}")
    print(f"  经验积累数: {final_performance['experiences_count']}")
    
    # 分析学习曲线
    print(f"\n📈 学习曲线分析:")
    success_rates = [entry['success_rate'] for entry in learning_journey]
    
    if len(success_rates) >= 5:
        early_avg = sum(success_rates[:5]) / 5
        late_avg = sum(success_rates[-5:]) / 5
        improvement = late_avg - early_avg
        
        print(f"  早期平均成功率: {early_avg:.1%}")
        print(f"  后期平均成功率: {late_avg:.1%}")
        print(f"  学习改进幅度: {improvement:+.1%}")
        
        if improvement > 0.1:
            print("  🚀 显著进步！Agent展现出强大的学习能力")
        elif improvement > 0:
            print("  📊 稳步提升，学习过程有效")
        else:
            print("  🤔 需要调整学习策略")
    
    # 策略进化分析
    print(f"\n🔬 策略进化分析:")
    strategy_performance = []
    for name, strategy in agent.strategies.items():
        if strategy.usage_count > 0:
            strategy_performance.append({
                'name': name,
                'success_rate': strategy.success_rate,
                'usage_count': strategy.usage_count,
                'efficiency': strategy.success_rate * strategy.usage_count
            })
    
    # 按效率排序
    strategy_performance.sort(key=lambda x: x['efficiency'], reverse=True)
    
    print(f"  最高效策略前3名:")
    for i, strategy in enumerate(strategy_performance[:3], 1):
        print(f"    {i}. {strategy['name'][:40]}...")
        print(f"       成功率: {strategy['success_rate']:.1%} | 使用次数: {strategy['usage_count']}")
    
    # 行为模式分析
    print(f"\n🎯 行为模式分析:")
    action_stats = {}
    for exp in agent.experiences:
        action = exp.action
        if action not in action_stats:
            action_stats[action] = {'total': 0, 'success': 0}
        action_stats[action]['total'] += 1
        if exp.success:
            action_stats[action]['success'] += 1
    
    for action, stats in action_stats.items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {action}: {success_rate:.1%} 成功率 ({stats['success']}/{stats['total']}次)")
    
    # 保存详细状态
    timestamp = int(time.time())
    state_file = f"advanced_agent_state_{timestamp}.json"
    agent.save_state(state_file)
    
    # 保存学习历程
    journey_file = f"learning_journey_{timestamp}.json"
    with open(journey_file, 'w', encoding='utf-8') as f:
        json.dump(learning_journey, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 数据已保存:")
    print(f"  Agent状态: {state_file}")
    print(f"  学习历程: {journey_file}")
    
    return agent, learning_journey

def analyze_learning_patterns(journey):
    """分析学习模式"""
    print("\n🔍 深度学习模式分析:")
    
    # 按阶段分析
    stages = {}
    for entry in journey:
        stage = entry['stage']
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(entry)
    
    for stage_num, stage_data in stages.items():
        stage_success_rate = sum(1 for x in stage_data if x['success']) / len(stage_data)
        avg_reward = sum(x['reward'] for x in stage_data) / len(stage_data)
        actions_used = set(x['action'] for x in stage_data)
        
        print(f"  阶段 {stage_num}:")
        print(f"    成功率: {stage_success_rate:.1%}")
        print(f"    平均奖励: {avg_reward:.2f}")
        print(f"    使用动作: {', '.join(actions_used)}")
    
    # 适应性分析
    print(f"\n🎨 适应性分析:")
    action_diversity = len(set(entry['action'] for entry in journey))
    task_diversity = len(set(entry['task'] for entry in journey))
    
    print(f"  动作多样性: {action_diversity} 种不同动作")
    print(f"  任务多样性: {task_diversity} 种不同任务")
    
    adaptation_score = action_diversity / task_diversity if task_diversity > 0 else 0
    print(f"  适应性评分: {adaptation_score:.2f} (越接近1越好)")

if __name__ == "__main__":
    # 运行高级演示
    agent, journey = advanced_agent_demo()
    
    # 深度分析
    analyze_learning_patterns(journey)
    
    print(f"\n🎉 高级自进化Agent演示完成！")
    print(f"Agent展现了从探索到专精的完整学习过程。")