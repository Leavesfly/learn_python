"""
VLA系统快速演示 - 纯Python实现（无需numpy）
Vision-Language-Action System Quick Demo

这是一个简化版的VLA系统演示，展示核心功能而不需要额外依赖
"""

import random
import time
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# ========== 数据结构 ==========

class ObjectType(Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"

class ActionType(Enum):
    MOVE_TO = "move_to"
    GRASP = "grasp"
    RELEASE = "release"


@dataclass
class VisualObject:
    object_id: str
    object_type: ObjectType
    position: Tuple[float, float, float]
    color: str
    size: float


@dataclass
class VisualScene:
    scene_id: str
    objects: List[VisualObject]
    robot_position: Tuple[float, float, float]


@dataclass
class RobotAction:
    action_type: ActionType
    target_position: Optional[Tuple[float, float, float]] = None
    target_object_id: Optional[str] = None
    duration: float = 0.5


# ========== 简化的VLA系统 ==========

class SimpleVLASystem:
    """简化的VLA系统 - 纯Python实现"""
    
    def __init__(self):
        self.action_history = []
        self.stats = {
            "total_instructions": 0,
            "successful_executions": 0
        }
    
    def process_instruction(self, scene: VisualScene, instruction: str) -> List[RobotAction]:
        """处理指令并生成动作序列"""
        self.stats["total_instructions"] += 1
        
        # 简单的指令解析
        instruction = instruction.lower()
        
        if "pick" in instruction or "grab" in instruction:
            return self._generate_pick_actions(instruction, scene)
        elif "place" in instruction or "put" in instruction:
            return self._generate_place_actions(scene)
        else:
            return [RobotAction(ActionType.MOVE_TO, (0, 0, 10), duration=1.0)]
    
    def _generate_pick_actions(self, instruction: str, scene: VisualScene) -> List[RobotAction]:
        """生成拾取动作"""
        # 找到目标物体
        target_obj = None
        for obj in scene.objects:
            obj_desc = f"{obj.color} {obj.object_type.value}"
            if obj.color in instruction or obj.object_type.value in instruction:
                target_obj = obj
                break
        
        if not target_obj:
            target_obj = scene.objects[0] if scene.objects else None
        
        if not target_obj:
            return []
        
        above_pos = (target_obj.position[0], target_obj.position[1], 
                    target_obj.position[2] + 10)
        
        return [
            RobotAction(ActionType.MOVE_TO, above_pos, duration=1.0),
            RobotAction(ActionType.MOVE_TO, target_obj.position, duration=0.5),
            RobotAction(ActionType.GRASP, target_object_id=target_obj.object_id, duration=0.3),
            RobotAction(ActionType.MOVE_TO, above_pos, duration=0.5)
        ]
    
    def _generate_place_actions(self, scene: VisualScene) -> List[RobotAction]:
        """生成放置动作"""
        target_pos = (10.0, 10.0, 5.0)
        above_pos = (target_pos[0], target_pos[1], target_pos[2] + 10)
        
        return [
            RobotAction(ActionType.MOVE_TO, above_pos, duration=1.0),
            RobotAction(ActionType.MOVE_TO, target_pos, duration=0.5),
            RobotAction(ActionType.RELEASE, duration=0.3)
        ]
    
    def execute_actions(self, actions: List[RobotAction], verbose: bool = True) -> Dict:
        """执行动作序列"""
        if verbose:
            print(f"\n🤖 执行 {len(actions)} 个动作:")
        
        for i, action in enumerate(actions, 1):
            if verbose:
                print(f"  {i}. {action.action_type.value}", end="")
                if action.target_position:
                    print(f" -> {action.target_position}", end="")
                print(f" ({action.duration}s)")
            
            time.sleep(action.duration * 0.1)
            self.action_history.append(action)
        
        self.stats["successful_executions"] += 1
        return {"success": True, "total_actions": len(actions)}


# ========== 环境模拟 ==========

class SimpleEnvironment:
    """简化的环境模拟器"""
    
    def __init__(self):
        self.objects = []
        self.robot_position = (0.0, 0.0, 20.0)
        self.scene_counter = 0
    
    def reset(self):
        """重置环境"""
        self.objects = []
        self.robot_position = (0.0, 0.0, 20.0)
        
        # 生成随机物体
        colors = ["red", "green", "blue", "yellow"]
        types = [ObjectType.CUBE, ObjectType.SPHERE, ObjectType.CYLINDER]
        
        num_objects = random.randint(3, 5)
        for i in range(num_objects):
            obj = VisualObject(
                object_id=f"obj_{i}",
                object_type=random.choice(types),
                position=(
                    random.uniform(-30, 30),
                    random.uniform(-30, 30),
                    random.uniform(5, 15)
                ),
                color=random.choice(colors),
                size=random.uniform(3, 8)
            )
            self.objects.append(obj)
    
    def get_scene(self) -> VisualScene:
        """获取当前场景"""
        self.scene_counter += 1
        return VisualScene(
            scene_id=f"scene_{self.scene_counter}",
            objects=self.objects.copy(),
            robot_position=self.robot_position
        )
    
    def visualize(self) -> str:
        """可视化场景"""
        lines = ["=" * 50]
        lines.append(f"场景 #{self.scene_counter}")
        lines.append("=" * 50)
        lines.append(f"机器人位置: {self.robot_position}\n")
        lines.append(f"物体列表 (共 {len(self.objects)} 个):")
        
        for obj in self.objects:
            lines.append(f"  {obj.object_id}: {obj.color} {obj.object_type.value}")
            lines.append(f"    位置: ({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f})")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# ========== 演示函数 ==========

def demo_basic():
    """基础功能演示"""
    print("\n" + "🌟" * 25)
    print("VLA系统快速演示 - 基础功能")
    print("🌟" * 25)
    
    vla = SimpleVLASystem()
    env = SimpleEnvironment()
    
    # 重置并显示场景
    env.reset()
    print("\n📷 当前场景:")
    print(env.visualize())
    
    # 测试指令
    instructions = [
        "pick the red cube",
        "pick the blue sphere",
        "place the object"
    ]
    
    for i, instruction in enumerate(instructions, 1):
        print(f"\n{'─' * 50}")
        print(f"指令 {i}: {instruction}")
        print(f"{'─' * 50}")
        
        scene = env.get_scene()
        actions = vla.process_instruction(scene, instruction)
        result = vla.execute_actions(actions, verbose=True)
        
        print(f"✅ 完成 - 执行了 {result['total_actions']} 个动作")
    
    print(f"\n{'=' * 50}")
    print("📊 统计信息:")
    print(f"  总指令数: {vla.stats['total_instructions']}")
    print(f"  成功执行: {vla.stats['successful_executions']}")
    print(f"  总动作数: {len(vla.action_history)}")
    print("=" * 50)


def demo_multi_task():
    """多任务演示"""
    print("\n" + "🌟" * 25)
    print("VLA系统快速演示 - 多任务处理")
    print("🌟" * 25)
    
    vla = SimpleVLASystem()
    env = SimpleEnvironment()
    env.reset()
    
    print("\n📷 场景:")
    print(env.visualize())
    
    tasks = [
        ("拾取红色物体", "pick the red cube"),
        ("放置物体", "place the object"),
        ("拾取蓝色物体", "pick the blue sphere"),
    ]
    
    print(f"\n🎯 任务列表 (共 {len(tasks)} 个):")
    for i, (desc, _) in enumerate(tasks, 1):
        print(f"  {i}. {desc}")
    
    start_time = time.time()
    
    for step, (desc, instruction) in enumerate(tasks, 1):
        print(f"\n{'━' * 50}")
        print(f"任务 {step}/{len(tasks)}: {desc}")
        
        scene = env.get_scene()
        actions = vla.process_instruction(scene, instruction)
        vla.execute_actions(actions, verbose=False)
        
        print(f"✓ 完成 ({len(actions)} 个动作)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 50}")
    print("🏆 任务完成:")
    print(f"  完成任务: {len(tasks)}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均每任务: {total_time/len(tasks):.2f}s")
    print("=" * 50)


def demo_interactive():
    """交互模式"""
    print("\n" + "🌟" * 25)
    print("VLA系统快速演示 - 交互模式")
    print("🌟" * 25)
    
    vla = SimpleVLASystem()
    env = SimpleEnvironment()
    env.reset()
    
    print("\n欢迎使用VLA交互系统!")
    print("\n可用命令:")
    print("  - pick the <color> <shape>  (例如: pick the red cube)")
    print("  - place the object")
    print("  - scene    (显示当前场景)")
    print("  - reset    (重置场景)")
    print("  - stats    (显示统计)")
    print("  - quit     (退出)")
    
    print("\n" + env.visualize())
    
    while True:
        print("\n" + "─" * 50)
        user_input = input("🎤 请输入指令: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("👋 再见!")
            break
        elif user_input.lower() == 'scene':
            print(env.visualize())
        elif user_input.lower() == 'reset':
            env.reset()
            print("✅ 场景已重置")
            print(env.visualize())
        elif user_input.lower() == 'stats':
            print("📊 统计信息:")
            for k, v in vla.stats.items():
                print(f"  {k}: {v}")
            print(f"  总动作数: {len(vla.action_history)}")
        else:
            scene = env.get_scene()
            actions = vla.process_instruction(scene, user_input)
            result = vla.execute_actions(actions, verbose=True)
            print(f"✅ 完成")


def main():
    """主函数"""
    print("\n" + "🤖" * 30)
    print("VLA (Vision-Language-Action) 智能系统")
    print("快速演示版本 - 纯Python实现")
    print("🤖" * 30)
    
    while True:
        print("\n请选择演示模式:")
        print("  1. 基础功能演示")
        print("  2. 多任务处理演示")
        print("  3. 交互模式")
        print("  4. 全部演示")
        print("  0. 退出")
        
        choice = input("\n输入选择 (0-4): ").strip()
        
        if choice == '0':
            print("感谢使用!")
            break
        elif choice == '1':
            demo_basic()
        elif choice == '2':
            demo_multi_task()
        elif choice == '3':
            demo_interactive()
        elif choice == '4':
            demo_basic()
            input("\n按Enter继续...")
            demo_multi_task()
            input("\n按Enter进入交互模式...")
            demo_interactive()
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    random.seed(42)  # 设置随机种子以获得可重复结果
    main()
