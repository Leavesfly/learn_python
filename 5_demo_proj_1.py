import json
import os
from datetime import datetime

class Task:
    def __init__(self, title, description="", priority="medium"):
        self.id = int(datetime.now().timestamp() * 1000000)
        self.title = title
        self.description = description
        self.priority = priority  # high, medium, low
        self.completed = False
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.completed_at = None

    def mark_completed(self):
        self.completed = True
        self.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'completed': self.completed,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }

    @classmethod
    def from_dict(cls, data):
        task = cls(data['title'], data['description'], data['priority'])
        task.id = data['id']
        task.completed = data['completed']
        task.created_at = data['created_at']
        task.completed_at = data['completed_at']
        return task

class TaskManager:
    def __init__(self, data_file="tasks.json"):
        self.data_file = data_file
        self.tasks = []
        self.load_tasks()

    def load_tasks(self):
        """从文件加载任务"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                    self.tasks = [Task.from_dict(task_data) for task_data in tasks_data]
            except (json.JSONDecodeError, KeyError):
                print("数据文件格式错误，将创建新的任务列表")
                self.tasks = []

    def save_tasks(self):
        """保存任务到文件"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            tasks_data = [task.to_dict() for task in self.tasks]
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)

    def add_task(self, title, description="", priority="medium"):
        """添加新任务"""
        task = Task(title, description, priority)
        self.tasks.append(task)
        self.save_tasks()
        print(f"任务 '{title}' 已添加成功！")

    def list_tasks(self, show_completed=True):
        """显示任务列表"""
        if not self.tasks:
            print("暂无任务")
            return

        # 按优先级排序
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        sorted_tasks = sorted(self.tasks, key=lambda x: (x.completed, priority_order.get(x.priority, 4)))

        print("\n" + "="*60)
        print("任务列表".center(58))
        print("="*60)

        for i, task in enumerate(sorted_tasks, 1):
            if not show_completed and task.completed:
                continue
            
            status = "✓" if task.completed else "○"
            priority_symbols = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            priority_symbol = priority_symbols.get(task.priority, '⚪')
            
            print(f"{i:2d}. {status} {priority_symbol} {task.title}")
            if task.description:
                print(f"     描述: {task.description}")
            print(f"     创建时间: {task.created_at}")
            if task.completed:
                print(f"     完成时间: {task.completed_at}")
            print("-" * 60)

    def complete_task(self, task_id):
        """标记任务为完成"""
        for task in self.tasks:
            if task.id == task_id:
                if task.completed:
                    print("该任务已经完成了！")
                else:
                    task.mark_completed()
                    self.save_tasks()
                    print(f"任务 '{task.title}' 已标记为完成！")
                return
        print("未找到该任务ID")

    def delete_task(self, task_id):
        """删除任务"""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                deleted_task = self.tasks.pop(i)
                self.save_tasks()
                print(f"任务 '{deleted_task.title}' 已删除！")
                return
        print("未找到该任务ID")

    def get_task_by_index(self, index):
        """通过显示索引获取任务"""
        if 1 <= index <= len(self.tasks):
            # 按优先级排序后获取
            priority_order = {'high': 1, 'medium': 2, 'low': 3}
            sorted_tasks = sorted(self.tasks, key=lambda x: (x.completed, priority_order.get(x.priority, 4)))
            return sorted_tasks[index - 1]
        return None

def main():
    manager = TaskManager()
    
    while True:
        print("\n" + "="*40)
        print("个人任务管理系统".center(36))
        print("="*40)
        print("1. 添加任务")
        print("2. 查看所有任务")
        print("3. 查看未完成任务")
        print("4. 完成任务")
        print("5. 删除任务")
        print("6. 退出")
        print("-"*40)
        
        choice = input("请选择操作 (1-6): ").strip()
        
        if choice == '1':
            title = input("请输入任务标题: ").strip()
            if not title:
                print("任务标题不能为空！")
                continue
                
            description = input("请输入任务描述 (可选): ").strip()
            
            print("请选择优先级:")
            print("1. 高 (high)")
            print("2. 中 (medium)")
            print("3. 低 (low)")
            priority_choice = input("选择优先级 (1-3，默认为2): ").strip()
            
            priority_map = {'1': 'high', '2': 'medium', '3': 'low'}
            priority = priority_map.get(priority_choice, 'medium')
            
            manager.add_task(title, description, priority)
            
        elif choice == '2':
            manager.list_tasks(show_completed=True)
            
        elif choice == '3':
            manager.list_tasks(show_completed=False)
            
        elif choice == '4':
            manager.list_tasks(show_completed=False)
            if not any(not task.completed for task in manager.tasks):
                print("没有未完成的任务！")
                continue
                
            try:
                index = int(input("请输入要完成的任务序号: "))
                task = manager.get_task_by_index(index)
                if task:
                    manager.complete_task(task.id)
                else:
                    print("无效的任务序号！")
            except ValueError:
                print("请输入有效的数字！")
                
        elif choice == '5':
            manager.list_tasks(show_completed=True)
            if not manager.tasks:
                continue
                
            try:
                index = int(input("请输入要删除的任务序号: "))
                task = manager.get_task_by_index(index)
                if task:
                    confirm = input(f"确定要删除任务 '{task.title}' 吗？(y/n): ").strip().lower()
                    if confirm == 'y':
                        manager.delete_task(task.id)
                else:
                    print("无效的任务序号！")
            except ValueError:
                print("请输入有效的数字！")
                
        elif choice == '6':
            print("感谢使用任务管理系统，再见！")
            break
            
        else:
            print("无效选择，请重新输入！")

if __name__ == "__main__":
    main()