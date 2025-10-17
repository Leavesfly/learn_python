"""
VLA系统测试 - 单元测试与集成测试
"""

import unittest
import sys
import importlib

# 动态导入VLA核心模块
vla_core = importlib.import_module('28_vla_core')

VLASystem = vla_core.VLASystem
RobotEnvironment = vla_core.RobotEnvironment
VisionEncoder = vla_core.VisionEncoder
LanguageEncoder = vla_core.LanguageEncoder
MultiModalFusion = vla_core.MultiModalFusion
ActionDecoder = vla_core.ActionDecoder
VisualObject = vla_core.VisualObject
VisualScene = vla_core.VisualScene
ObjectType = vla_core.ObjectType
ActionType = vla_core.ActionType


class TestVisionEncoder(unittest.TestCase):
    """视觉编码器测试"""
    
    def setUp(self):
        self.encoder = VisionEncoder(feature_dim=128)
        self.env = RobotEnvironment()
        self.env.reset()
    
    def test_encode_scene(self):
        """测试场景编码"""
        scene = self.env.get_current_scene()
        features = self.encoder.encode_scene(scene)
        
        self.assertEqual(len(features), 128)
        self.assertTrue(all(-1 <= f <= 2 for f in features))
    
    def test_empty_scene(self):
        """测试空场景"""
        self.env.objects = []
        scene = self.env.get_current_scene()
        features = self.encoder.encode_scene(scene)
        
        self.assertEqual(len(features), 128)


class TestLanguageEncoder(unittest.TestCase):
    """语言编码器测试"""
    
    def setUp(self):
        self.encoder = LanguageEncoder(embedding_dim=64)
    
    def test_encode_pick_instruction(self):
        """测试拾取指令"""
        features, info = self.encoder.encode_instruction("pick the red cube")
        
        self.assertEqual(len(features), 64)
        self.assertEqual(info['intent'], 'pick')
        self.assertIn('red', info['target'])
    
    def test_encode_place_instruction(self):
        """测试放置指令"""
        features, info = self.encoder.encode_instruction("place the object")
        
        self.assertEqual(info['intent'], 'place')
    
    def test_unknown_instruction(self):
        """测试未知指令"""
        features, info = self.encoder.encode_instruction("hello world")
        
        self.assertEqual(info['intent'], 'unknown')


class TestMultiModalFusion(unittest.TestCase):
    """多模态融合测试"""
    
    def setUp(self):
        self.fusion = MultiModalFusion(vision_dim=128, language_dim=64, fusion_dim=128)
    
    def test_fuse_features(self):
        """测试特征融合"""
        vision_features = vla_core.np.random.randn(128)
        language_features = vla_core.np.random.randn(64)
        
        fused = self.fusion.fuse(vision_features, language_features)
        
        self.assertEqual(len(fused), 128)
        norm = vla_core.np.linalg.norm(fused)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestActionDecoder(unittest.TestCase):
    """动作解码器测试"""
    
    def setUp(self):
        self.decoder = ActionDecoder(input_dim=128)
        self.env = RobotEnvironment()
        self.env.reset()
    
    def test_decode_pick_action(self):
        """测试拾取动作解码"""
        features = vla_core.np.random.randn(128)
        instruction_info = {
            'intent': 'pick',
            'target': 'red cube'
        }
        scene = self.env.get_current_scene()
        
        actions = self.decoder.decode(features, instruction_info, scene)
        
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[0].action_type, ActionType.MOVE_TO)
    
    def test_decode_place_action(self):
        """测试放置动作解码"""
        features = vla_core.np.random.randn(128)
        instruction_info = {
            'intent': 'place',
            'target': None
        }
        scene = self.env.get_current_scene()
        
        actions = self.decoder.decode(features, instruction_info, scene)
        
        self.assertGreater(len(actions), 0)


class TestVLASystem(unittest.TestCase):
    """VLA系统集成测试"""
    
    def setUp(self):
        self.vla = VLASystem()
        self.env = RobotEnvironment()
        self.env.reset()
    
    def test_process_instruction(self):
        """测试指令处理"""
        scene = self.env.get_current_scene()
        actions = self.vla.process_instruction(scene, "pick the red cube")
        
        self.assertGreater(len(actions), 0)
        self.assertEqual(self.vla.metrics['total_instructions'], 1)
    
    def test_execute_actions(self):
        """测试动作执行"""
        scene = self.env.get_current_scene()
        actions = self.vla.process_instruction(scene, "pick the red cube")
        
        result = self.vla.execute_actions(actions, verbose=False)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['executed_actions']), len(actions))
    
    def test_multiple_instructions(self):
        """测试多指令处理"""
        scene = self.env.get_current_scene()
        
        instructions = [
            "pick the red cube",
            "place the object",
            "pick the blue sphere"
        ]
        
        for instruction in instructions:
            actions = self.vla.process_instruction(scene, instruction)
            self.assertGreater(len(actions), 0)
        
        self.assertEqual(self.vla.metrics['total_instructions'], 3)
    
    def test_reset(self):
        """测试系统重置"""
        scene = self.env.get_current_scene()
        self.vla.process_instruction(scene, "pick the red cube")
        
        self.vla.reset()
        
        self.assertEqual(self.vla.metrics['total_instructions'], 0)
        self.assertEqual(len(self.vla.action_history), 0)


class TestRobotEnvironment(unittest.TestCase):
    """机器人环境测试"""
    
    def setUp(self):
        self.env = RobotEnvironment()
    
    def test_reset(self):
        """测试环境重置"""
        self.env.reset()
        
        self.assertGreater(len(self.env.objects), 0)
        self.assertEqual(self.env.robot_position, (0.0, 0.0, 20.0))
    
    def test_get_scene(self):
        """测试场景获取"""
        self.env.reset()
        scene = self.env.get_current_scene()
        
        self.assertIsNotNone(scene.scene_id)
        self.assertEqual(len(scene.objects), len(self.env.objects))
    
    def test_visualize(self):
        """测试可视化"""
        self.env.reset()
        viz = self.env.visualize()
        
        self.assertIn("场景可视化", viz)
        self.assertIn("机器人位置", viz)


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("VLA系统单元测试")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestVisionEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestLanguageEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModalFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestActionDecoder))
    suite.addTests(loader.loadTestsFromTestCase(TestVLASystem))
    suite.addTests(loader.loadTestsFromTestCase(TestRobotEnvironment))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
