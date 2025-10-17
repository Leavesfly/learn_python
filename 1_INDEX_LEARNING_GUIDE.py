"""
Python vs Java: 完整学习指南索引
============================
面向Java程序员的Python学习路线图

本系列文档专门为有Java背景的开发者设计，
通过对比的方式帮助快速掌握Python的核心特性。
"""

print("=" * 70)
print("Python vs Java 学习指南 - 文档索引")
print("=" * 70)

# ============================================================================
# 文档列表
# ============================================================================

documents = [
    {
        "file": "1_diff_java.py",
        "title": "Python与Java主要区别总览",
        "topics": [
            "语法差异基础",
            "类型系统概览",
            "面向对象编程初探",
            "内存管理机制",
            "异常处理简介",
            "数据结构对比",
            "性能差异分析",
            "选择建议"
        ],
        "level": "入门",
        "description": "全面概览Python和Java的主要差异，为后续深入学习奠定基础"
    },
    {
        "file": "1_basic_syntax_comparison.py",
        "title": "基础语法深度对比",
        "topics": [
            "代码块与缩进规则",
            "变量声明与赋值",
            "注释与文档字符串",
            "条件语句(if-elif-else)",
            "循环语句(for/while)",
            "函数定义与参数",
            "字符串处理技巧"
        ],
        "level": "基础",
        "description": "详细对比Python和Java的基础语法，掌握Python编程的基本规则"
    },
    {
        "file": "1_type_system_comparison.py",
        "title": "类型系统深度对比",
        "topics": [
            "静态类型 vs 动态类型",
            "鸭子类型理解",
            "类型提示(Type Hints)",
            "内置数据类型",
            "类型转换机制",
            "类型检查(isinstance)",
            "mypy静态检查工具"
        ],
        "level": "基础",
        "description": "深入理解Python的动态类型系统，学会使用类型提示提高代码质量"
    },
    {
        "file": "1_oop_comparison.py",
        "title": "面向对象编程深度对比",
        "topics": [
            "类的定义与构造",
            "访问控制与封装",
            "属性(Properties)",
            "继承与多重继承",
            "抽象类与接口",
            "魔法方法(Dunder Methods)",
            "运算符重载"
        ],
        "level": "进阶",
        "description": "全面掌握Python的面向对象编程，理解与Java OOP的差异"
    },
    {
        "file": "1_exception_handling.py",
        "title": "异常处理深度对比",
        "topics": [
            "try-except-else-finally",
            "异常层次结构",
            "抛出异常(raise)",
            "自定义异常",
            "上下文管理器(with)",
            "异常处理最佳实践",
            "EAFP vs LBYL"
        ],
        "level": "进阶",
        "description": "掌握Python异常处理机制，学会编写健壮的错误处理代码"
    },
    {
        "file": "1_collections_comparison.py",
        "title": "集合与数据结构深度对比",
        "topics": [
            "列表(List) vs ArrayList",
            "元组(Tuple)的独特性",
            "字典(Dict) vs HashMap",
            "集合(Set) vs HashSet",
            "推导式(Comprehensions)",
            "高级集合类型",
            "collections模块"
        ],
        "level": "基础",
        "description": "精通Python内置数据结构，高效处理各类数据操作"
    },
    {
        "file": "1_stdlib_comparison.py",
        "title": "标准库与常用模块对比",
        "topics": [
            "文件I/O操作",
            "路径操作(pathlib)",
            "日期时间处理",
            "JSON处理",
            "正则表达式",
            "数学与随机",
            "系统与环境"
        ],
        "level": "进阶",
        "description": "掌握Python标准库的常用模块，提高开发效率"
    }
]

# ============================================================================
# 打印文档索引
# ============================================================================

def print_index():
    """打印文档索引"""
    for i, doc in enumerate(documents, 1):
        print(f"\n{i}. 【{doc['level']}】{doc['title']}")
        print(f"   文件: {doc['file']}")
        print(f"   说明: {doc['description']}")
        print(f"   主题:")
        for topic in doc['topics']:
            print(f"      • {topic}")

print_index()

# ============================================================================
# 学习路线建议
# ============================================================================

print("\n" + "=" * 70)
print("推荐学习路线")
print("=" * 70)

learning_path = """
第一阶段：快速入门（1-2天）
─────────────────────────
1. 阅读「Python与Java主要区别总览」
   → 建立整体认知，了解两种语言的核心差异

2. 学习「基础语法深度对比」
   → 掌握Python基本语法，能写简单程序
   → 重点：缩进规则、变量、循环、函数

3. 实践练习
   → 将熟悉的Java代码改写为Python
   → 尝试解决简单算法题


第二阶段：核心掌握（3-5天）
─────────────────────────
4. 学习「类型系统深度对比」
   → 理解动态类型的优势和陷阱
   → 学会使用类型提示

5. 学习「集合与数据结构深度对比」
   → 熟练使用list、dict、set、tuple
   → 掌握推导式语法
   → 了解collections模块

6. 实践项目
   → 实现一个简单的数据处理脚本
   → 使用Python的数据结构优势


第三阶段：进阶提升（5-7天）
─────────────────────────
7. 学习「面向对象编程深度对比」
   → 理解Python的OOP特性
   → 掌握魔法方法的使用
   → 学会多重继承

8. 学习「异常处理深度对比」
   → 掌握异常处理最佳实践
   → 学会使用上下文管理器
   → 理解EAFP哲学

9. 学习「标准库与常用模块对比」
   → 熟悉常用标准库
   → 学会查阅文档


第四阶段：实战应用（持续）
─────────────────────────
10. 选择方向深入
    → Web开发: Flask/Django
    → 数据分析: Pandas/NumPy
    → 自动化: Selenium/Requests
    → 机器学习: Scikit-learn/TensorFlow

11. 阅读优秀代码
    → GitHub上的优秀Python项目
    → Python标准库源码

12. 实践项目
    → 完成至少2-3个完整项目
    → 参与开源项目
"""

print(learning_path)

# ============================================================================
# 学习建议
# ============================================================================

print("\n" + "=" * 70)
print("给Java程序员的Python学习建议")
print("=" * 70)

tips = """
心态调整
────────
1. 拥抱简洁
   - Python强调可读性和简洁性
   - 不要试图用Java的方式写Python
   - "There should be one-- and preferably only one --obvious way to do it"

2. 相信鸭子类型
   - 不要过度担心类型安全
   - 编写单元测试来保证正确性
   - 需要时使用类型提示

3. 学习Python风格
   - 阅读PEP 8代码规范
   - 使用pylint/flake8检查代码
   - 学习"Pythonic"的写法


技术要点
────────
1. 重点掌握的差异
   - 缩进敏感（强制性）
   - 动态类型系统
   - 推导式语法
   - 生成器和迭代器
   - 装饰器
   - 上下文管理器

2. 需要转变的思维
   - EAFP vs LBYL
   - 组合优于继承
   - "batteries included"
   - 简单优于复杂

3. 工具链
   - pip: 包管理
   - venv: 虚拟环境
   - pytest: 测试框架
   - black: 代码格式化
   - mypy: 类型检查


避免的陷阱
──────────
1. 常见错误
   - 混用Tab和空格
   - 可变默认参数
   - 循环中修改列表
   - 浅拷贝vs深拷贝

2. 性能陷阱
   - 字符串拼接使用join
   - 生成器代替列表（大数据）
   - 使用内置函数（C实现）

3. 设计陷阱
   - 过度使用类（不是所有东西都要OOP）
   - 忽略Python的函数式特性
   - 不合理的继承层次


资源推荐
────────
1. 官方文档
   - Python官方教程
   - Python标准库文档
   - PEP文档

2. 书籍
   - 《流畅的Python》
   - 《Effective Python》
   - 《Python Cookbook》

3. 在线资源
   - Real Python
   - Python官方文档
   - Stack Overflow

4. 实践平台
   - LeetCode (Python解题)
   - Kaggle (数据分析)
   - GitHub (开源项目)
"""

print(tips)

# ============================================================================
# 快速参考
# ============================================================================

print("\n" + "=" * 70)
print("快速参考对照表")
print("=" * 70)

quick_reference = """
┌──────────────────┬───────────────────────┬───────────────────────┐
│     概念         │        Python         │         Java          │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 代码块           │ 缩进                  │ {}                    │
│ 注释             │ # 和 \"\"\"           │ // 和 /* */          │
│ 变量声明         │ x = 10                │ int x = 10;           │
│ 类型             │ 动态                  │ 静态                  │
│ 字符串           │ 'str' "str" \"\"\"s\"\"\"│ "str"                 │
│ 列表             │ [1, 2, 3]             │ new ArrayList<>()     │
│ 字典             │ {"k": "v"}            │ new HashMap<>()       │
│ 元组             │ (1, 2)                │ 无内置                │
│ 集合             │ {1, 2, 3}             │ new HashSet<>()       │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 函数定义         │ def func():           │ void func()           │
│ 方法定义         │ def method(self):     │ void method()         │
│ 构造函数         │ __init__(self)        │ ClassName()           │
│ 继承             │ class C(Base):        │ class C extends Base  │
│ 接口             │ 非正式(鸭子类型)      │ interface             │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 异常捕获         │ except                │ catch                 │
│ 抛出异常         │ raise                 │ throw                 │
│ 资源管理         │ with                  │ try-with-resources    │
│ 空值             │ None                  │ null                  │
│ 布尔值           │ True/False            │ true/false            │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 导入             │ import module         │ import package.*;     │
│ 包管理           │ pip                   │ Maven/Gradle          │
│ 虚拟环境         │ venv                  │ 无（JVM隔离）         │
│ 打印             │ print()               │ System.out.println()  │
└──────────────────┴───────────────────────┴───────────────────────┘
"""

print(quick_reference)

# ============================================================================
# 结语
# ============================================================================

print("\n" + "=" * 70)
print("结语")
print("=" * 70)

conclusion = """
作为Java程序员学习Python，你已经具备了：
- 编程思维和逻辑能力
- 面向对象的理解
- 算法和数据结构基础
- 软件工程实践经验

这些都是宝贵的资产！

Python学习的关键是：
1. 不要抗拒简洁性 - 享受Python的优雅
2. 不要过度工程化 - Python推崇实用主义
3. 不要忽视文档 - Python有极好的文档文化
4. 多写多练 - 用Python思维写Python代码

记住Python之禅(import this)：
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Readability counts.

祝你Python学习之旅愉快！🐍
"""

print(conclusion)

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("开始学习吧！")
    print("=" * 70)
    print("\n建议按照推荐的学习路线，循序渐进地学习。")
    print("每个文档都可以独立运行，查看实际效果。")
    print("\n运行方式：python <文件名>")
    print("例如：python 1_basic_syntax_comparison.py")
    print("\n祝学习顺利！")

if __name__ == "__main__":
    main()
