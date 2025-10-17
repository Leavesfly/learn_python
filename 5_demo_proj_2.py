import json
import os
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict
import math

class LearningTopic:
    def __init__(self, name, description="", difficulty="medium", estimated_hours=2):
        self.id = int(datetime.now().timestamp() * 1000000)
        self.name = name
        self.description = description
        self.difficulty = difficulty  # easy, medium, hard
        self.estimated_hours = estimated_hours
        self.mastery_level = 0  # 0-100
        self.time_spent = 0  # 分钟
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_studied = None
        self.notes = []
        self.quiz_scores = []

    def add_study_time(self, minutes):
        """添加学习时间"""
        self.time_spent += minutes
        self.last_studied = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def update_mastery(self, level):
        """更新掌握度"""
        self.mastery_level = max(0, min(100, level))

    def add_note(self, note):
        """添加学习笔记"""
        self.notes.append({
            'content': note,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def add_quiz_score(self, score):
        """添加测试分数"""
        self.quiz_scores.append({
            'score': score,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'estimated_hours': self.estimated_hours,
            'mastery_level': self.mastery_level,
            'time_spent': self.time_spent,
            'created_at': self.created_at,
            'last_studied': self.last_studied,
            'notes': self.notes,
            'quiz_scores': self.quiz_scores
        }

    @classmethod
    def from_dict(cls, data):
        topic = cls(data['name'], data['description'], data['difficulty'], data['estimated_hours'])
        topic.id = data['id']
        topic.mastery_level = data['mastery_level']
        topic.time_spent = data['time_spent']
        topic.created_at = data['created_at']
        topic.last_studied = data['last_studied']
        topic.notes = data.get('notes', [])
        topic.quiz_scores = data.get('quiz_scores', [])
        return topic

class StudySession:
    def __init__(self, topic_id, duration_minutes, notes=""):
        self.id = int(datetime.now().timestamp() * 1000000)
        self.topic_id = topic_id
        self.duration_minutes = duration_minutes
        self.notes = notes
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class LearningTracker:
    def __init__(self, data_file="learning_progress.json"):
        self.data_file = data_file
        self.topics = []
        self.study_sessions = []
        self.learning_goals = {}
        self.load_data()

    def load_data(self):
        """加载学习数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.topics = [LearningTopic.from_dict(topic_data) for topic_data in data.get('topics', [])]
                    self.study_sessions = data.get('study_sessions', [])
                    self.learning_goals = data.get('learning_goals', {})
            except (json.JSONDecodeError, KeyError):
                print("数据文件格式错误，将创建新的学习记录")
                self.topics = []
                self.study_sessions = []
                self.learning_goals = {}

    def save_data(self):
        """保存学习数据"""
        data = {
            'topics': [topic.to_dict() for topic in self.topics],
            'study_sessions': self.study_sessions,
            'learning_goals': self.learning_goals
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_topic(self, name, description="", difficulty="medium", estimated_hours=2):
        """添加学习主题"""
        topic = LearningTopic(name, description, difficulty, estimated_hours)
        self.topics.append(topic)
        self.save_data()
        print(f"学习主题 '{name}' 已添加！")
        return topic

    def get_topic_by_id(self, topic_id):
        """根据ID获取主题"""
        for topic in self.topics:
            if topic.id == topic_id:
                return topic
        return None

    def record_study_session(self, topic_id, duration_minutes, notes="", mastery_update=None):
        """记录学习会话"""
        topic = self.get_topic_by_id(topic_id)
        if not topic:
            print("未找到指定的学习主题")
            return

        topic.add_study_time(duration_minutes)
        
        if mastery_update is not None:
            topic.update_mastery(mastery_update)

        session = StudySession(topic_id, duration_minutes, notes)
        self.study_sessions.append(session.__dict__)
        
        self.save_data()
        print(f"已记录 {duration_minutes} 分钟的学习时间")

    def get_learning_statistics(self):
        """获取学习统计信息"""
        total_time = sum(topic.time_spent for topic in self.topics)
        total_topics = len(self.topics)
        completed_topics = len([t for t in self.topics if t.mastery_level >= 80])
        avg_mastery = sum(t.mastery_level for t in self.topics) / total_topics if total_topics > 0 else 0

        # 最近7天的学习时间
        recent_sessions = []
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        for session in self.study_sessions:
            session_time = datetime.strptime(session['timestamp'], "%Y-%m-%d %H:%M:%S")
            if session_time >= seven_days_ago:
                recent_sessions.append(session)

        recent_time = sum(session['duration_minutes'] for session in recent_sessions)

        return {
            'total_time_hours': total_time / 60,
            'total_topics': total_topics,
            'completed_topics': completed_topics,
            'completion_rate': (completed_topics / total_topics * 100) if total_topics > 0 else 0,
            'average_mastery': avg_mastery,
            'recent_week_hours': recent_time / 60,
            'study_streak': self.calculate_study_streak()
        }

    def calculate_study_streak(self):
        """计算连续学习天数"""
        if not self.study_sessions:
            return 0

        # 按日期分组学习会话
        study_dates = set()
        for session in self.study_sessions:
            date = session['timestamp'].split(' ')[0]
            study_dates.add(date)

        # 计算连续天数
        sorted_dates = sorted(study_dates, reverse=True)
        streak = 0
        current_date = datetime.now().date()

        for date_str in sorted_dates:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if date == current_date or date == current_date - timedelta(days=streak):
                streak += 1
                current_date = date
            else:
                break

        return streak

    def generate_progress_report(self):
        """生成学习进度报告"""
        stats = self.get_learning_statistics()
        
        print("\n" + "="*60)
        print("📊 学习进度报告".center(56))
        print("="*60)
        
        print(f"📚 总学习主题: {stats['total_topics']} 个")
        print(f"✅ 已完成主题: {stats['completed_topics']} 个 ({stats['completion_rate']:.1f}%)")
        print(f"⏰ 总学习时间: {stats['total_time_hours']:.1f} 小时")
        print(f"📈 平均掌握度: {stats['average_mastery']:.1f}%")
        print(f"🔥 连续学习: {stats['study_streak']} 天")
        print(f"📅 近一周学习: {stats['recent_week_hours']:.1f} 小时")
        
        # 显示各难度主题分布
        difficulty_stats = defaultdict(int)
        for topic in self.topics:
            difficulty_stats[topic.difficulty] += 1
        
        print(f"\n📋 主题难度分布:")
        for difficulty, count in difficulty_stats.items():
            print(f"  {difficulty}: {count} 个")

        # 显示掌握度分布
        mastery_ranges = {"0-20%": 0, "21-40%": 0, "41-60%": 0, "61-80%": 0, "81-100%": 0}
        for topic in self.topics:
            if topic.mastery_level <= 20:
                mastery_ranges["0-20%"] += 1
            elif topic.mastery_level <= 40:
                mastery_ranges["21-40%"] += 1
            elif topic.mastery_level <= 60:
                mastery_ranges["41-60%"] += 1
            elif topic.mastery_level <= 80:
                mastery_ranges["61-80%"] += 1
            else:
                mastery_ranges["81-100%"] += 1

        print(f"\n🎯 掌握度分布:")
        for range_name, count in mastery_ranges.items():
            print(f"  {range_name}: {count} 个")

    def show_study_recommendations(self):
        """显示学习建议"""
        print("\n💡 个性化学习建议:")
        
        # 找出需要复习的主题
        need_review = [t for t in self.topics if t.mastery_level < 60]
        if need_review:
            print("📖 建议复习的主题:")
            for topic in need_review[:3]:
                print(f"  • {topic.name} (掌握度: {topic.mastery_level}%)")

        # 找出很久没学的主题
        week_ago = datetime.now() - timedelta(days=7)
        need_attention = []
        for topic in self.topics:
            if topic.last_studied:
                last_study = datetime.strptime(topic.last_studied, "%Y-%m-%d %H:%M:%S")
                if last_study < week_ago and topic.mastery_level < 80:
                    need_attention.append(topic)

        if need_attention:
            print("\n⚠️ 很久没学习的主题:")
            for topic in need_attention[:3]:
                days_ago = (datetime.now() - datetime.strptime(topic.last_studied, "%Y-%m-%d %H:%M:%S")).days
                print(f"  • {topic.name} (已经 {days_ago} 天没学)")

        # 学习时间建议
        stats = self.get_learning_statistics()
        if stats['recent_week_hours'] < 5:
            print("\n⏰ 建议增加学习时间，每周至少5小时")
        
        # 连续学习鼓励
        if stats['study_streak'] == 0:
            print("\n🚀 开始你的学习之旅吧！")
        elif stats['study_streak'] < 7:
            print(f"\n🔥 很好！已连续学习{stats['study_streak']}天，继续保持！")
        else:
            print(f"\n🏆 太棒了！已连续学习{stats['study_streak']}天，你是学习达人！")

def create_default_python_topics():
    """创建默认的Python学习主题"""
    topics = [
        ("Python基础语法", "变量、数据类型、运算符", "easy", 3),
        ("控制结构", "if语句、循环、异常处理", "easy", 4),
        ("函数和模块", "函数定义、参数、模块导入", "medium", 5),
        ("数据结构", "列表、字典、集合、元组", "medium", 6),
        ("面向对象编程", "类、继承、多态、封装", "hard", 8),
        ("文件操作", "文件读写、JSON处理", "medium", 3),
        ("正则表达式", "模式匹配、文本处理", "hard", 4),
        ("网络编程", "HTTP请求、API调用", "hard", 6),
        ("数据库操作", "SQLite、MySQL连接", "hard", 5),
        ("Web开发", "Flask/Django框架", "hard", 10)
    ]
    return topics

def main():
    tracker = LearningTracker()
    
    print("🎓 智能学习进度追踪系统")
    print("=" * 50)
    
    while True:
        print("\n主菜单:")
        print("1. 添加学习主题")
        print("2. 记录学习会话")
        print("3. 查看学习主题")
        print("4. 学习进度报告")
        print("5. 学习建议")
        print("6. 管理学习笔记")
        print("7. 初始化Python学习计划")
        print("8. 退出")
        
        choice = input("\n请选择功能 (1-8): ").strip()
        
        if choice == '1':
            name = input("主题名称: ").strip()
            if not name:
                print("主题名称不能为空！")
                continue
            
            description = input("主题描述: ").strip()
            
            print("难度等级: 1-简单 2-中等 3-困难")
            diff_choice = input("选择难度 (1-3): ").strip()
            difficulty_map = {'1': 'easy', '2': 'medium', '3': 'hard'}
            difficulty = difficulty_map.get(diff_choice, 'medium')
            
            try:
                estimated_hours = float(input("预估学习时间(小时): ") or "2")
                tracker.add_topic(name, description, difficulty, estimated_hours)
            except ValueError:
                print("时间格式错误，使用默认值2小时")
                tracker.add_topic(name, description, difficulty, 2)
        
        elif choice == '2':
            if not tracker.topics:
                print("还没有学习主题，请先添加！")
                continue
            
            print("\n选择学习主题:")
            for i, topic in enumerate(tracker.topics, 1):
                mastery_bar = "█" * (topic.mastery_level // 10) + "░" * (10 - topic.mastery_level // 10)
                print(f"{i}. {topic.name} [{mastery_bar}] {topic.mastery_level}%")
            
            try:
                topic_index = int(input("选择主题序号: ")) - 1
                if 0 <= topic_index < len(tracker.topics):
                    selected_topic = tracker.topics[topic_index]
                    
                    duration = int(input("学习时间(分钟): "))
                    notes = input("学习笔记(可选): ").strip()
                    
                    mastery_input = input(f"当前掌握度 ({selected_topic.mastery_level}%), 更新为(0-100, 回车跳过): ").strip()
                    mastery_update = None
                    if mastery_input:
                        mastery_update = int(mastery_input)
                    
                    tracker.record_study_session(selected_topic.id, duration, notes, mastery_update)
                    
                    if notes:
                        selected_topic.add_note(notes)
                        tracker.save_data()
                    
                else:
                    print("无效的主题序号！")
            except ValueError:
                print("请输入有效的数字！")
        
        elif choice == '3':
            if not tracker.topics:
                print("还没有学习主题！")
                continue
            
            print(f"\n📚 学习主题列表 (共{len(tracker.topics)}个):")
            print("-" * 80)
            
            for topic in tracker.topics:
                difficulty_emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}
                emoji = difficulty_emoji.get(topic.difficulty, '⚪')
                
                mastery_bar = "█" * (topic.mastery_level // 10) + "░" * (10 - topic.mastery_level // 10)
                
                print(f"{emoji} {topic.name}")
                print(f"   掌握度: [{mastery_bar}] {topic.mastery_level}%")
                print(f"   学习时间: {topic.time_spent//60}小时{topic.time_spent%60}分钟 / 预估{topic.estimated_hours}小时")
                print(f"   最后学习: {topic.last_studied or '从未学习'}")
                if topic.description:
                    print(f"   描述: {topic.description}")
                print("-" * 80)
        
        elif choice == '4':
            tracker.generate_progress_report()
        
        elif choice == '5':
            tracker.show_study_recommendations()
        
        elif choice == '6':
            if not tracker.topics:
                print("还没有学习主题！")
                continue
            
            print("\n选择查看笔记的主题:")
            for i, topic in enumerate(tracker.topics, 1):
                note_count = len(topic.notes)
                print(f"{i}. {topic.name} ({note_count}条笔记)")
            
            try:
                topic_index = int(input("选择主题序号: ")) - 1
                if 0 <= topic_index < len(tracker.topics):
                    selected_topic = tracker.topics[topic_index]
                    
                    if not selected_topic.notes:
                        print("该主题还没有笔记")
                    else:
                        print(f"\n📝 {selected_topic.name} 的学习笔记:")
                        for i, note in enumerate(selected_topic.notes, 1):
                            print(f"{i}. [{note['timestamp']}]")
                            print(f"   {note['content']}")
                            print()
                else:
                    print("无效的主题序号！")
            except ValueError:
                print("请输入有效的数字！")
        
        elif choice == '7':
            confirm = input("这将创建Python学习计划，是否继续？(y/n): ").strip().lower()
            if confirm == 'y':
                default_topics = create_default_python_topics()
                for name, desc, diff, hours in default_topics:
                    tracker.add_topic(name, desc, diff, hours)
                print("Python学习计划已创建！")
        
        elif choice == '8':
            print("学习愉快，坚持就是胜利！🎓")
            break
        
        else:
            print("无效选择，请重新输入！")

if __name__ == "__main__":
    main()