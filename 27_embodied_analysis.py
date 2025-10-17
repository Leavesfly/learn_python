"""
具身智能扫地机器人 - 训练结果分析工具

分析训练数据，展示学习曲线和性能指标
"""

import json
import math
from typing import List, Dict


def load_training_data(filepath: str) -> List[Dict]:
    """加载训练数据"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"错误：JSON格式错误")
        return []


def calculate_moving_average(values: List[float], window: int = 10) -> List[float]:
    """计算移动平均"""
    if len(values) < window:
        return values
    
    moving_avg = []
    for i in range(len(values) - window + 1):
        avg = sum(values[i:i+window]) / window
        moving_avg.append(avg)
    
    return moving_avg


def print_ascii_chart(values: List[float], title: str, width: int = 60, height: int = 15):
    """打印ASCII图表"""
    if not values:
        print("无数据可显示")
        return
    
    print(f"\n{title}")
    print("=" * width)
    
    # 归一化数据到图表高度
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        print("数据无变化")
        return
    
    # 创建图表
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # 绘制数据点
    for i, val in enumerate(values):
        x = int((i / len(values)) * (width - 1))
        normalized = (val - min_val) / (max_val - min_val)
        y = height - 1 - int(normalized * (height - 1))
        
        if 0 <= x < width and 0 <= y < height:
            chart[y][x] = '●'
    
    # 打印图表
    print(f"Max: {max_val:.2f} ┤", end='')
    for row_idx, row in enumerate(chart):
        if row_idx == 0:
            print(''.join(row))
        else:
            print(' ' * 13 + '│' + ''.join(row))
    
    print(f"Min: {min_val:.2f} └" + "─" * width)
    print(' ' * 14 + f"Episode 1 → {len(values)}")


def analyze_learning_progress(history: List[Dict]):
    """分析学习进度"""
    if not history:
        print("没有训练数据")
        return
    
    print("\n" + "=" * 70)
    print("具身智能机器人训练结果分析")
    print("=" * 70)
    
    # 提取数据
    episodes = len(history)
    rewards = [h["reward"] for h in history]
    cleanliness = [h["cleanliness"] for h in history]
    steps = [h["steps"] for h in history]
    collisions = [h["collisions"] for h in history]
    
    # 1. 整体统计
    print(f"\n📊 整体统计 ({episodes}个回合)")
    print("-" * 70)
    
    print(f"\n奖励 (Reward):")
    print(f"  平均值: {sum(rewards)/len(rewards):.2f}")
    print(f"  最大值: {max(rewards):.2f}")
    print(f"  最小值: {min(rewards):.2f}")
    variance = sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)
    print(f"  标准差: {math.sqrt(variance):.2f}")
    
    print(f"\n清洁度 (Cleanliness):")
    print(f"  平均值: {sum(cleanliness)/len(cleanliness):.2%}")
    print(f"  最大值: {max(cleanliness):.2%}")
    print(f"  最小值: {min(cleanliness):.2%}")
    
    print(f"\n步数 (Steps):")
    print(f"  平均值: {sum(steps)/len(steps):.1f}")
    print(f"  最小值: {min(steps)}")
    print(f"  最大值: {max(steps)}")
    
    print(f"\n碰撞次数 (Collisions):")
    print(f"  平均值: {sum(collisions)/len(collisions):.1f}")
    print(f"  最小值: {min(collisions)}")
    print(f"  最大值: {max(collisions)}")
    
    # 2. 学习进步分析
    if episodes >= 20:
        print(f"\n📈 学习进步分析")
        print("-" * 70)
        
        window = min(10, episodes // 5)
        first_window = history[:window]
        last_window = history[-window:]
        
        first_reward = sum(h["reward"] for h in first_window) / window
        last_reward = sum(h["reward"] for h in last_window) / window
        
        first_clean = sum(h["cleanliness"] for h in first_window) / window
        last_clean = sum(h["cleanliness"] for h in last_window) / window
        
        first_steps = sum(h["steps"] for h in first_window) / window
        last_steps = sum(h["steps"] for h in last_window) / window
        
        first_collision = sum(h["collisions"] for h in first_window) / window
        last_collision = sum(h["collisions"] for h in last_window) / window
        
        print(f"\n前{window}回合 vs 后{window}回合:")
        print(f"  奖励:    {first_reward:7.2f} → {last_reward:7.2f}  "
              f"(变化: {last_reward-first_reward:+7.2f}, {(last_reward-first_reward)/abs(first_reward)*100:+.1f}%)")
        print(f"  清洁度:  {first_clean:7.2%} → {last_clean:7.2%}  "
              f"(变化: {last_clean-first_clean:+7.2%})")
        print(f"  步数:    {first_steps:7.1f} → {last_steps:7.1f}  "
              f"(变化: {last_steps-first_steps:+7.1f})")
        print(f"  碰撞:    {first_collision:7.1f} → {last_collision:7.1f}  "
              f"(变化: {last_collision-first_collision:+7.1f})")
    
    # 3. 绘制学习曲线
    print(f"\n📉 学习曲线")
    print("-" * 70)
    
    # 奖励曲线
    if len(rewards) >= 10:
        ma_rewards = calculate_moving_average(rewards, window=10)
        print_ascii_chart(ma_rewards, "奖励移动平均 (窗口=10)", width=60, height=12)
    else:
        print_ascii_chart(rewards, "奖励", width=60, height=12)
    
    # 清洁度曲线
    if len(cleanliness) >= 10:
        ma_clean = calculate_moving_average(cleanliness, window=10)
        print_ascii_chart(ma_clean, "清洁度移动平均 (窗口=10)", width=60, height=12)
    else:
        print_ascii_chart(cleanliness, "清洁度", width=60, height=12)
    
    # 4. 性能里程碑
    print(f"\n🎯 性能里程碑")
    print("-" * 70)
    
    # 找到清洁度超过特定阈值的首次回合
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    for threshold in thresholds:
        for i, h in enumerate(history):
            if h["cleanliness"] >= threshold:
                print(f"  清洁度达到 {threshold:.0%}: 第 {i+1} 回合")
                break
        else:
            print(f"  清洁度达到 {threshold:.0%}: 未达成")
    
    # 找到最佳回合
    best_clean_idx = cleanliness.index(max(cleanliness))
    best_reward_idx = rewards.index(max(rewards))
    
    print(f"\n  最佳清洁度回合: 第 {best_clean_idx+1} 回合 ({max(cleanliness):.2%})")
    print(f"  最高奖励回合:   第 {best_reward_idx+1} 回合 ({max(rewards):.2f})")
    
    # 5. Q表增长
    if "q_table_size" in history[0]:
        q_sizes = [h["q_table_size"] for h in history]
        print(f"\n🧠 Q表增长")
        print("-" * 70)
        print(f"  初始大小: {q_sizes[0]}")
        print(f"  最终大小: {q_sizes[-1]}")
        print(f"  增长量:   {q_sizes[-1] - q_sizes[0]}")
        
        print_ascii_chart(q_sizes, "Q表大小增长", width=60, height=10)
    
    # 6. 探索率衰减
    if "epsilon" in history[0]:
        epsilons = [h["epsilon"] for h in history]
        print(f"\n🔍 探索率衰减")
        print("-" * 70)
        print(f"  初始探索率: {epsilons[0]:.3f}")
        print(f"  最终探索率: {epsilons[-1]:.3f}")
        
        print_ascii_chart(epsilons, "Epsilon (探索率)", width=60, height=10)


def generate_summary_report(history: List[Dict]) -> str:
    """生成汇总报告"""
    if not history:
        return "无训练数据"
    
    episodes = len(history)
    rewards = [h["reward"] for h in history]
    cleanliness = [h["cleanliness"] for h in history]
    
    report = []
    report.append("\n" + "=" * 70)
    report.append("训练总结报告")
    report.append("=" * 70)
    
    report.append(f"\n总训练回合: {episodes}")
    report.append(f"平均奖励: {sum(rewards)/len(rewards):.2f}")
    report.append(f"平均清洁度: {sum(cleanliness)/len(cleanliness):.2%}")
    report.append(f"最高清洁度: {max(cleanliness):.2%}")
    
    # 评估学习效果
    if episodes >= 20:
        first_10 = cleanliness[:10]
        last_10 = cleanliness[-10:]
        
        first_avg = sum(first_10) / len(first_10)
        last_avg = sum(last_10) / len(last_10)
        improvement = last_avg - first_avg
        
        report.append(f"\n学习效果评估:")
        report.append(f"  前10回合平均清洁度: {first_avg:.2%}")
        report.append(f"  后10回合平均清洁度: {last_avg:.2%}")
        report.append(f"  提升幅度: {improvement:.2%}")
        
        if improvement > 0.2:
            report.append("  ✅ 学习效果: 优秀 (提升>20%)")
        elif improvement > 0.1:
            report.append("  ✅ 学习效果: 良好 (提升>10%)")
        elif improvement > 0.05:
            report.append("  ⚠️  学习效果: 一般 (提升>5%)")
        else:
            report.append("  ❌ 学习效果: 需要改进 (提升<5%)")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    """主函数"""
    # 加载训练数据
    filepath = "/Users/yefei.yf/Qoder/learn_python/embodied_robot_training.json"
    
    print("正在加载训练数据...")
    history = load_training_data(filepath)
    
    if not history:
        print("无法加载训练数据，请先运行训练程序")
        return
    
    print(f"成功加载 {len(history)} 回合的训练数据\n")
    
    # 分析训练结果
    analyze_learning_progress(history)
    
    # 生成汇总报告
    summary = generate_summary_report(history)
    print(summary)
    
    # 保存报告
    report_path = "/Users/yefei.yf/Qoder/learn_python/embodied_robot_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
