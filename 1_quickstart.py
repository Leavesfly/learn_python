#!/usr/bin/env python3
"""
快速启动脚本 - Python vs Java 学习系列
自动检查环境并提供学习建议
"""

import sys
import os
from pathlib import Path

# ============================================================================
# 检查Python版本
# ============================================================================

def check_python_version():
    """检查Python版本"""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)
    
    version = sys.version_info
    print(f"\nPython版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3:
        print("❌ 警告: 建议使用Python 3.6+")
        return False
    elif version.minor < 6:
        print("⚠️  建议: 推荐使用Python 3.6+以支持f-string等特性")
        return True
    else:
        print("✅ 版本符合要求")
        return True

# ============================================================================
# 检查文档文件
# ============================================================================

def check_documents():
    """检查所有文档是否存在"""
    print("\n" + "=" * 70)
    print("文档检查")
    print("=" * 70 + "\n")
    
    documents = [
        ("1_diff_java.py", "Python与Java主要区别总览"),
        ("1_basic_syntax_comparison.py", "基础语法深度对比"),
        ("1_type_system_comparison.py", "类型系统深度对比"),
        ("1_oop_comparison.py", "面向对象编程深度对比"),
        ("1_exception_handling.py", "异常处理深度对比"),
        ("1_collections_comparison.py", "集合与数据结构深度对比"),
        ("1_stdlib_comparison.py", "标准库与常用模块对比"),
        ("1_INDEX_LEARNING_GUIDE.py", "学习指南索引"),
        ("1_README_JAVA_TO_PYTHON.md", "README文档"),
    ]
    
    all_exist = True
    for filename, title in documents:
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"✅ {filename:35} ({size:.1f} KB) - {title}")
        else:
            print(f"❌ {filename:35} - 文件不存在")
            all_exist = False
    
    return all_exist

# ============================================================================
# 显示学习路线
# ============================================================================

def show_learning_path():
    """显示推荐的学习路线"""
    print("\n" + "=" * 70)
    print("推荐学习路线")
    print("=" * 70)
    
    path = """
📚 第一阶段：快速入门 (1-2天)
   ├─ 1️⃣  python 1_diff_java.py
   │   → 快速了解Python与Java的主要差异
   │
   ├─ 2️⃣  python 1_basic_syntax_comparison.py
   │   → 掌握基础语法和编程规范
   │
   └─ 💪 练习：改写简单的Java程序为Python

📊 第二阶段：核心掌握 (3-5天)
   ├─ 3️⃣  python 1_type_system_comparison.py
   │   → 理解动态类型系统
   │
   ├─ 4️⃣  python 1_collections_comparison.py
   │   → 熟练使用Python数据结构
   │
   └─ 💪 练习：完成数据处理小项目

🚀 第三阶段：进阶提升 (5-7天)
   ├─ 5️⃣  python 1_oop_comparison.py
   │   → 掌握Python面向对象特性
   │
   ├─ 6️⃣  python 1_exception_handling.py
   │   → 学习异常处理最佳实践
   │
   ├─ 7️⃣  python 1_stdlib_comparison.py
   │   → 熟悉标准库常用模块
   │
   └─ 💪 练习：实现一个完整的应用

🎯 第四阶段：实战应用 (持续学习)
   └─ 选择方向深入：Web/数据分析/自动化/AI
"""
    print(path)

# ============================================================================
# 显示快速命令
# ============================================================================

def show_quick_commands():
    """显示快速命令"""
    print("\n" + "=" * 70)
    print("快速命令")
    print("=" * 70)
    
    commands = """
🔍 查看完整索引:
   python 1_INDEX_LEARNING_GUIDE.py

📖 阅读README:
   cat 1_README_JAVA_TO_PYTHON.md

▶️  运行示例:
   python 1_basic_syntax_comparison.py
   python 1_type_system_comparison.py
   python 1_collections_comparison.py

📝 运行所有示例:
   for f in 1_*_comparison.py; do echo "=== $f ===" && python "$f"; done

🧪 测试代码片段:
   python -i 1_basic_syntax_comparison.py  # 交互模式

💡 获取帮助:
   python -c "help('modules')"  # 查看所有模块
   python -c "import this"      # Python之禅
"""
    print(commands)

# ============================================================================
# 显示学习建议
# ============================================================================

def show_tips():
    """显示学习建议"""
    print("\n" + "=" * 70)
    print("给Java程序员的建议")
    print("=" * 70)
    
    tips = """
✨ 学习心态:
   • 拥抱简洁 - Python强调可读性
   • 相信类型 - 动态类型不是敌人
   • 遵循规范 - PEP 8是你的朋友

🎯 重点关注:
   • ⭐ 缩进规则 (强制性的!)
   • ⭐ 推导式语法 (列表/字典/集合)
   • ⭐ 魔法方法 (__init__, __str__, etc.)
   • ⭐ 上下文管理器 (with语句)
   • ⭐ 生成器和迭代器

⚠️  常见陷阱:
   • 不要混用Tab和空格
   • 注意可变默认参数
   • 理解浅拷贝vs深拷贝
   • 循环中不要修改列表

🔧 推荐工具:
   • IDE: PyCharm, VS Code
   • 格式化: black, autopep8
   • 检查: pylint, flake8
   • 类型检查: mypy
   • 测试: pytest

📚 扩展学习:
   • 官方文档: https://docs.python.org/
   • Real Python: https://realpython.com/
   • PEP 8规范: https://pep8.org/
"""
    print(tips)

# ============================================================================
# 交互式菜单
# ============================================================================

def interactive_menu():
    """交互式菜单"""
    print("\n" + "=" * 70)
    print("交互式菜单")
    print("=" * 70)
    
    while True:
        print("\n请选择:")
        print("  1 - 查看完整学习指南")
        print("  2 - 运行基础语法对比")
        print("  3 - 运行类型系统对比")
        print("  4 - 运行数据结构对比")
        print("  5 - 运行OOP对比")
        print("  6 - 运行异常处理对比")
        print("  7 - 运行标准库对比")
        print("  8 - 显示Python之禅")
        print("  0 - 退出")
        
        choice = input("\n输入选择 (0-8): ").strip()
        
        if choice == '0':
            print("\n祝学习愉快！🐍")
            break
        elif choice == '1':
            os.system("python 1_INDEX_LEARNING_GUIDE.py")
        elif choice == '2':
            os.system("python 1_basic_syntax_comparison.py")
        elif choice == '3':
            os.system("python 1_type_system_comparison.py")
        elif choice == '4':
            os.system("python 1_collections_comparison.py")
        elif choice == '5':
            os.system("python 1_oop_comparison.py")
        elif choice == '6':
            os.system("python 1_exception_handling.py")
        elif choice == '7':
            os.system("python 1_stdlib_comparison.py")
        elif choice == '8':
            import this
        else:
            print("❌ 无效选择，请重试")
        
        input("\n按Enter继续...")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print()
    print("🐍 " * 35)
    print()
    print("     Python vs Java 学习系列 - 快速启动")
    print("     面向Java程序员的Python学习指南")
    print()
    print("🐍 " * 35)
    
    # 检查环境
    if not check_python_version():
        return
    
    # 检查文档
    if not check_documents():
        print("\n❌ 部分文档缺失，请检查!")
        return
    
    # 显示信息
    show_learning_path()
    show_quick_commands()
    show_tips()
    
    # 询问是否进入交互模式
    print("\n" + "=" * 70)
    choice = input("是否进入交互式菜单? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        interactive_menu()
    else:
        print("\n💡 提示: 运行 'python 1_quickstart.py' 可随时启动交互菜单")
        print("\n开始学习吧！建议从 'python 1_INDEX_LEARNING_GUIDE.py' 开始 📚")
        print("\n祝学习愉快！🐍\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出。祝学习愉快！🐍")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
