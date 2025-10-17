"""
简化版自进化Agent演示
直接在文件中引用主要功能来避免导入问题
"""

import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class SimpleExperience:
    """简化的经验记录"""
    task: str
    action: str
    success: bool
    reward: float
    timestamp: float

class SimpleEvolutionAgent:
    """简化版自进化Agent"""
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
        self.experiences = []
        self.action_success_rates = {}  # 动作成功率
        self.task_action_preferences = {}  # 任务-动作偏好
        self.total_tasks = 0
        self.successful_tasks = 0
        self.exploration_rate = 0.3
        
        # 可用工具
        self.tools = ['search', 'calculate', 'analyze', 'plan']
        
    def choose_action(self, task: str) -> str:
        """选择动作"""
        # 探索vs利用
        if random.random() < self.exploration_rate:
            # 探索：随机选择
            return random.choice(self.tools)
        else:
            # 利用：基于历史经验选择
            if task in self.task_action_preferences:
                # 选择该任务类型最成功的动作
                best_action = max(self.task_action_preferences[task].items(), 
                                key=lambda x: x[1])
                return best_action[0]
            else:
                # 选择全局最成功的动作
                if self.action_success_rates:
                    best_action = max(self.action_success_rates.items(),
                                    key=lambda x: x[1])
                    return best_action[0]
                else:
                    return random.choice(self.tools)
    
    def execute_action(self, action: str, task: str) -> tuple:
        """执行动作并返回成功与否"""
        # 模拟执行结果，不同动作对不同任务的成功率不同
        success_probabilities = {
            'search': {
                '搜索': 0.9, '查询': 0.8, '研究': 0.7, '分析': 0.4
            },
            'calculate': {
                '计算': 0.9, '数学': 0.8, '统计': 0.7, '优化': 0.6
            },
            'analyze': {
                '分析': 0.9, '研究': 0.8, '评估': 0.7, '预测': 0.6
            },
            'plan': {
                '规划': 0.9, '设计': 0.8, '制定': 0.7, '创建': 0.6
            }
        }
        
        # 基于任务关键词确定基础成功率
        base_prob = 0.5
        for keyword, prob in success_probabilities.get(action, {}).items():
            if keyword in task:
                base_prob = prob
                break
        
        # 添加一些随机性
        actual_prob = base_prob + random.uniform(-0.2, 0.2)
        actual_prob = max(0.1, min(0.95, actual_prob))  # 限制在合理范围
        
        success = random.random() < actual_prob
        reward = 1.0 if success else -0.5
        
        return success, reward
    
    def learn_from_experience(self, experience: SimpleExperience):
        """从经验中学习"""
        self.experiences.append(experience)
        
        # 更新动作成功率
        action = experience.action
        if action not in self.action_success_rates:
            self.action_success_rates[action] = 0.5
        
        # 使用指数移动平均更新成功率
        alpha = 0.1
        if experience.success:
            self.action_success_rates[action] = (1 - alpha) * self.action_success_rates[action] + alpha * 1.0
        else:
            self.action_success_rates[action] = (1 - alpha) * self.action_success_rates[action] + alpha * 0.0
        
        # 更新任务-动作偏好
        task_type = experience.task.split('：')[0] if '：' in experience.task else experience.task[:4]
        if task_type not in self.task_action_preferences:
            self.task_action_preferences[task_type] = {}
        
        if action not in self.task_action_preferences[task_type]:
            self.task_action_preferences[task_type][action] = 0.5
            
        if experience.success:
            self.task_action_preferences[task_type][action] = \
                (1 - alpha) * self.task_action_preferences[task_type][action] + alpha * 1.0
        else:
            self.task_action_preferences[task_type][action] = \
                (1 - alpha) * self.task_action_preferences[task_type][action] + alpha * 0.0
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """处理任务"""
        # 选择动作
        action = self.choose_action(task)
        
        # 执行动作
        success, reward = self.execute_action(action, task)
        
        # 创建经验
        experience = SimpleExperience(
            task=task,
            action=action,
            success=success,
            reward=reward,
            timestamp=time.time()
        )
        
        # 学习
        self.learn_from_experience(experience)
        
        # 更新统计
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
            
        return {
            'task': task,
            'action': action,
            'success': success,
            'reward': reward,
            'success_rate': self.successful_tasks / self.total_tasks
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'success_rate': self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0,
            'exploration_rate': self.exploration_rate,
            'learned_preferences': self.task_action_preferences,
            'action_success_rates': self.action_success_rates
        }
    
    def evolve(self):
        """自我进化"""
        # 根据近期表现调整探索率
        if len(self.experiences) >= 10:
            recent_success = sum(1 for exp in self.experiences[-10:] if exp.success) / 10
            if recent_success > 0.8:
                self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
            elif recent_success < 0.5:
                self.exploration_rate = min(0.5, self.exploration_rate + 0.05)

def comprehensive_demo():
    """综合演示"""
    print("=== 简化版自进化Agent综合演示 ===\n")
    
    agent = SimpleEvolutionAgent("学习助手")
    
    # 测试任务序列 - 从简单到复杂
    test_tasks = [
        # 第一阶段：基础任务
        "搜索：Python基础教程",
        "计算：投资收益率",
        "分析：用户行为数据", 
        "规划：学习路线图",
        
        # 第二阶段：中等难度
        "搜索：机器学习算法",
        "计算：模型准确率",
        "分析：数据分布特征",
        "规划：项目开发计划",
        
        # 第三阶段：复杂任务
        "研究：深度学习前沿",
        "优化：神经网络架构",
        "评估：系统性能指标",
        "设计：智能推荐系统",
        
        # 第四阶段：创新挑战
        "创建：新型算法框架",
        "预测：技术发展趋势",
        "制定：AI伦理准则",
        "构建：多模态学习系统"
    ]
    
    print(f"准备执行 {len(test_tasks)} 个渐进式任务\n")
    
    results = []
    
    # 执行任务并观察学习过程
    for i, task in enumerate(test_tasks, 1):
        print(f"--- 任务 {i}: {task} ---")
        
        result = agent.process_task(task)
        results.append(result)
        
        print(f"🎯 选择动作: {result['action']}")
        print(f"✅ 执行结果: {'成功' if result['success'] else '失败'}")
        print(f"📊 当前成功率: {result['success_rate']:.1%}")
        
        # 每4个任务显示一次学习状态
        if i % 4 == 0:
            print(f"\n--- 第{i//4}阶段学习总结 ---")
            performance = agent.get_performance_summary()
            
            print(f"🎯 阶段成功率: {performance['success_rate']:.1%}")
            print(f"🔍 当前探索率: {performance['exploration_rate']:.2f}")
            
            print("🧠 学到的动作偏好:")
            for task_type, actions in performance['learned_preferences'].items():
                best_action = max(actions.items(), key=lambda x: x[1])
                print(f"  {task_type}: 最佳动作 {best_action[0]} (成功率 {best_action[1]:.1%})")
            
            print("🛠️ 工具使用效果:")
            for action, success_rate in performance['action_success_rates'].items():
                print(f"  {action}: {success_rate:.1%}")
            
            # 触发进化
            agent.evolve()
            print(f"🔄 探索率调整为: {agent.exploration_rate:.2f}")
            print()
        else:
            print()
    
    # 最终分析
    print("=" * 60)
    print("🎓 最终学习成果分析")
    print("=" * 60)
    
    final_performance = agent.get_performance_summary()
    
    print(f"\n📈 整体表现:")
    print(f"  总任务数: {final_performance['total_tasks']}")
    print(f"  成功任务: {final_performance['successful_tasks']}")
    print(f"  最终成功率: {final_performance['success_rate']:.1%}")
    
    # 学习曲线分析
    success_rates = [r['success_rate'] for r in results]
    
    print(f"\n📊 学习曲线分析:")
    print(f"  初期成功率 (前4任务): {success_rates[3]:.1%}")
    print(f"  中期成功率 (第8任务): {success_rates[7]:.1%}")
    print(f"  后期成功率 (第12任务): {success_rates[11]:.1%}")
    print(f"  最终成功率: {success_rates[-1]:.1%}")
    
    improvement = success_rates[-1] - success_rates[3]
    print(f"  总体改进: {improvement:+.1%}")
    
    if improvement > 0.1:
        print("  🚀 显著改进！Agent学习效果明显")
    elif improvement > 0:
        print("  📈 稳步提升，学习方向正确")
    else:
        print("  🤔 学习效果有限，需要调优")
    
    # 专业化分析
    print(f"\n🎯 任务专业化分析:")
    task_types = set()
    for task in test_tasks:
        task_type = task.split('：')[0] if '：' in task else task[:4]
        task_types.add(task_type)
    
    for task_type in task_types:
        if task_type in final_performance['learned_preferences']:
            preferences = final_performance['learned_preferences'][task_type]
            best_action = max(preferences.items(), key=lambda x: x[1])
            print(f"  {task_type}任务: 专精 {best_action[0]} (熟练度 {best_action[1]:.1%})")
    
    # 保存学习结果
    timestamp = int(time.time())
    with open(f"learning_results_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump({
            'agent_name': agent.name,
            'final_performance': final_performance,
            'task_results': results,
            'learning_timeline': [
                {
                    'task_num': i+1,
                    'task': test_tasks[i],
                    'action': results[i]['action'],
                    'success': results[i]['success'],
                    'cumulative_success_rate': results[i]['success_rate']
                }
                for i in range(len(test_tasks))
            ]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 学习结果已保存到: learning_results_{timestamp}.json")
    
    return agent, results

if __name__ == "__main__":
    agent, results = comprehensive_demo()
    
    print(f"\n🎉 自进化Agent演示完成！")
    print(f"Agent从随机决策进化为专业化的任务处理系统。")