"""
Python vs Java: 基础语法对比
============================
面向Java程序员的Python基础语法学习指南

作者: Python学习系列
目标读者: 有Java背景的开发者
学习重点: 掌握Python的语法特点与Java的差异
"""

# ============================================================================
# 1. 代码块与缩进
# ============================================================================

print("=== 1. 代码块与缩进 ===\n")

# Python: 使用缩进表示代码块（强制）
def python_indentation_demo():
    """
    Python关键差异：
    - 使用冒号 : 开始代码块
    - 使用4个空格缩进（PEP 8规范）
    - 不使用大括号
    - 缩进错误会导致语法错误
    """
    x = 10
    if x > 5:
        print("x 大于 5")
        if x > 8:
            print("x 也大于 8")
    else:
        print("x 不大于 5")

"""
Java对比:
public class JavaIndentation {
    public static void javaIndentationDemo() {
        // Java使用大括号 {} 表示代码块
        // 缩进只是为了可读性，不影响代码执行
        int x = 10;
        if (x > 5) {
            System.out.println("x 大于 5");
            if (x > 8) {
                System.out.println("x 也大于 8");
            }
        } else {
            System.out.println("x 不大于 5");
        }
    }
}

学习建议：
1. 配置IDE自动使用4空格缩进
2. 永远不要混用Tab和空格
3. 使用pylint或flake8检查代码规范
"""

# ============================================================================
# 2. 变量声明与赋值
# ============================================================================

print("=== 2. 变量声明与赋值 ===\n")

def python_variables():
    """
    Python变量特点：
    - 无需声明类型（动态类型）
    - 变量名使用snake_case命名
    - 可以同时赋值多个变量
    - 变量可以改变类型
    """
    # 直接赋值，无需声明类型
    user_name = "张三"
    user_age = 25
    is_active = True
    
    print(f"姓名: {user_name}, 年龄: {user_age}, 活跃: {is_active}")
    
    # 多变量同时赋值
    x, y, z = 1, 2, 3
    print(f"x={x}, y={y}, z={z}")
    
    # 变量交换（Python特色）
    a, b = 10, 20
    a, b = b, a  # 交换值，无需临时变量
    print(f"交换后: a={a}, b={b}")
    
    # 变量可以改变类型
    value = 42      # 整数
    value = "文本"  # 字符串
    value = [1, 2]  # 列表
    print(f"最终值: {value}")

"""
Java对比:
public class JavaVariables {
    public static void javaVariables() {
        // Java必须声明类型（静态类型）
        String userName = "张三";  // 驼峰命名
        int userAge = 25;
        boolean isActive = true;
        
        System.out.println("姓名: " + userName + ", 年龄: " + userAge + ", 活跃: " + isActive);
        
        // Java变量类型固定，不能改变
        int value = 42;
        // value = "文本";  // 编译错误！
        
        // 变量交换需要临时变量
        int a = 10, b = 20;
        int temp = a;
        a = b;
        b = temp;
    }
}

关键差异总结：
┌────────────────┬─────────────────┬─────────────────┐
│   特性         │   Python        │   Java          │
├────────────────┼─────────────────┼─────────────────┤
│ 类型声明       │ 不需要          │ 必须声明        │
│ 类型检查       │ 运行时          │ 编译时          │
│ 命名规范       │ snake_case      │ camelCase       │
│ 类型改变       │ 允许            │ 不允许          │
│ 多变量赋值     │ 支持            │ 不直接支持      │
└────────────────┴─────────────────┴─────────────────┘
"""

# ============================================================================
# 3. 注释
# ============================================================================

print("\n=== 3. 注释 ===\n")

# Python单行注释使用 #

"""
Python多行注释使用三引号
可以使用单引号或双引号
这常用于文档字符串（docstring）
"""

def documented_function(param1, param2):
    """
    这是函数的文档字符串（docstring）
    
    Args:
        param1: 第一个参数说明
        param2: 第二个参数说明
    
    Returns:
        返回值说明
    """
    return param1 + param2

# 查看文档字符串
print(documented_function.__doc__)

"""
Java对比:
public class JavaComments {
    // Java单行注释使用 //
    
    /*
     * Java多行注释使用 slash-star
     */
    
    /**
     * JavaDoc文档注释
     * @param param1 第一个参数
     * @param param2 第二个参数
     * @return 返回值说明
     */
    public static int documentedMethod(int param1, int param2) {
        return param1 + param2;
    }
}

学习要点：
- Python的docstring是语言内置特性，可以在运行时访问
- 使用__doc__属性访问文档字符串
- 推荐使用Google风格或NumPy风格的docstring
"""

# ============================================================================
# 4. 条件语句
# ============================================================================

print("\n=== 4. 条件语句 ===\n")

def python_conditionals():
    """
    Python条件语句特点：
    - 使用 elif 代替 else if
    - 条件表达式不需要括号
    - 支持链式比较
    - 有三元运算符
    """
    age = 25
    
    # 基本if-elif-else
    if age < 18:
        print("未成年")
    elif 18 <= age < 60:  # 链式比较（Java不支持）
        print("成年人")
    else:
        print("老年人")
    
    # 三元运算符（条件表达式）
    status = "成年" if age >= 18 else "未成年"
    print(f"状态: {status}")
    
    # Python的真值测试
    empty_list = []
    if empty_list:
        print("列表不为空")
    else:
        print("列表为空")  # 空列表被视为False
    
    # 成员测试
    fruits = ["apple", "banana", "orange"]
    if "apple" in fruits:
        print("找到苹果")
    
    # is vs ==
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    print(f"a == b: {a == b}")  # True (值相等)
    print(f"a is b: {a is b}")  # False (不是同一对象)
    print(f"a is c: {a is c}")  # True (同一对象)

"""
Java对比:
public class JavaConditionals {
    public static void javaConditionals() {
        int age = 25;
        
        // Java使用else if（两个单词）
        if (age < 18) {
            System.out.println("未成年");
        } else if (age >= 18 && age < 60) {  // 需要完整写出条件
            System.out.println("成年人");
        } else {
            System.out.println("老年人");
        }
        
        // Java三元运算符
        String status = (age >= 18) ? "成年" : "未成年";
        
        // Java的真值测试
        List<String> emptyList = new ArrayList<>();
        if (emptyList.isEmpty()) {  // 必须使用isEmpty()方法
            System.out.println("列表为空");
        }
        
        // Java成员测试
        List<String> fruits = Arrays.asList("apple", "banana", "orange");
        if (fruits.contains("apple")) {
            System.out.println("找到苹果");
        }
        
        // == vs equals()
        List<Integer> a = Arrays.asList(1, 2, 3);
        List<Integer> b = Arrays.asList(1, 2, 3);
        List<Integer> c = a;
        System.out.println("a.equals(b): " + a.equals(b));  // true (值相等)
        System.out.println("a == b: " + (a == b));          // false (不是同一对象)
        System.out.println("a == c: " + (a == c));          // true (同一对象)
    }
}

关键差异：
┌────────────────┬─────────────────┬─────────────────┐
│   特性         │   Python        │   Java          │
├────────────────┼─────────────────┼─────────────────┤
│ else if        │ elif            │ else if         │
│ 条件括号       │ 不需要          │ 需要()          │
│ 链式比较       │ 支持            │ 不支持          │
│ 空值测试       │ if not list:    │ list.isEmpty()  │
│ 成员测试       │ in              │ contains()      │
│ 对象比较       │ is/==           │ ==/equals()     │
└────────────────┴─────────────────┴─────────────────┘
"""

# ============================================================================
# 5. 循环语句
# ============================================================================

print("\n=== 5. 循环语句 ===\n")

def python_loops():
    """
    Python循环特点：
    - for循环遍历可迭代对象
    - range()函数生成序列
    - enumerate()同时获取索引和值
    - 支持else子句
    """
    # for循环遍历列表
    fruits = ["apple", "banana", "orange"]
    for fruit in fruits:
        print(f"水果: {fruit}")
    
    # 使用range()
    for i in range(5):  # 0到4
        print(f"数字: {i}")
    
    # enumerate获取索引和值
    for index, fruit in enumerate(fruits):
        print(f"索引 {index}: {fruit}")
    
    # 遍历字典
    person = {"name": "张三", "age": 25, "city": "北京"}
    for key, value in person.items():
        print(f"{key}: {value}")
    
    # while循环
    count = 0
    while count < 3:
        print(f"计数: {count}")
        count += 1
    
    # 循环的else子句（特殊特性）
    for i in range(3):
        print(f"循环 {i}")
    else:
        print("循环正常结束")  # 没有break时执行
    
    # break和continue
    for i in range(10):
        if i == 3:
            continue  # 跳过3
        if i == 7:
            break     # 在7处停止
        print(f"值: {i}")

"""
Java对比:
public class JavaLoops {
    public static void javaLoops() {
        // for-each循环
        String[] fruits = {"apple", "banana", "orange"};
        for (String fruit : fruits) {
            System.out.println("水果: " + fruit);
        }
        
        // 传统for循环
        for (int i = 0; i < 5; i++) {
            System.out.println("数字: " + i);
        }
        
        // 带索引的遍历
        for (int i = 0; i < fruits.length; i++) {
            System.out.println("索引 " + i + ": " + fruits[i]);
        }
        
        // 遍历Map
        Map<String, Object> person = new HashMap<>();
        person.put("name", "张三");
        person.put("age", 25);
        person.put("city", "北京");
        for (Map.Entry<String, Object> entry : person.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // while循环
        int count = 0;
        while (count < 3) {
            System.out.println("计数: " + count);
            count++;
        }
        
        // Java没有循环的else子句
        
        // break和continue
        for (int i = 0; i < 10; i++) {
            if (i == 3) continue;
            if (i == 7) break;
            System.out.println("值: " + i);
        }
    }
}

循环对比总结：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 遍历容器       │ for item in list    │ for(item : list)    │
│ 数字范围       │ for i in range(n)   │ for(int i=0;i<n;i++)│
│ 索引+值        │ enumerate()         │ 手动维护索引        │
│ 字典遍历       │ for k,v in d.items()│ entrySet()          │
│ else子句       │ 支持                │ 不支持              │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 6. 函数定义
# ============================================================================

print("\n=== 6. 函数定义 ===\n")

def python_functions():
    """
    Python函数特点：
    - 使用def关键字
    - 支持默认参数
    - 支持可变参数
    - 支持关键字参数
    - 函数是一等公民
    """
    
    # 基本函数
    def greet(name):
        return f"你好, {name}!"
    
    # 默认参数
    def greet_with_title(name, title="先生"):
        return f"你好, {title}{name}!"
    
    # 可变位置参数 (*args)
    def sum_all(*numbers):
        """接受任意数量的参数"""
        return sum(numbers)
    
    # 可变关键字参数 (**kwargs)
    def print_info(**kwargs):
        """接受任意数量的关键字参数"""
        for key, value in kwargs.items():
            print(f"{key}: {value}")
    
    # 混合参数
    def complex_function(pos1, pos2, *args, keyword1="default", **kwargs):
        """
        参数顺序：
        1. 位置参数
        2. *args (可变位置参数)
        3. 关键字参数
        4. **kwargs (可变关键字参数)
        """
        print(f"位置参数: {pos1}, {pos2}")
        print(f"额外位置参数: {args}")
        print(f"关键字参数: {keyword1}")
        print(f"额外关键字参数: {kwargs}")
    
    # 函数作为参数
    def apply_operation(x, y, operation):
        """函数是一等公民，可以作为参数传递"""
        return operation(x, y)
    
    # Lambda表达式
    add = lambda x, y: x + y
    multiply = lambda x, y: x * y
    
    # 测试函数
    print(greet("张三"))
    print(greet_with_title("张三"))
    print(greet_with_title("李四", "女士"))
    print(f"求和: {sum_all(1, 2, 3, 4, 5)}")
    
    print("\n信息:")
    print_info(name="张三", age=25, city="北京")
    
    print("\n复杂函数:")
    complex_function(1, 2, 3, 4, keyword1="value", extra1="e1", extra2="e2")
    
    print(f"\n应用加法: {apply_operation(10, 5, add)}")
    print(f"应用乘法: {apply_operation(10, 5, multiply)}")

"""
Java对比:
public class JavaFunctions {
    // 基本方法
    public static String greet(String name) {
        return "你好, " + name + "!";
    }
    
    // Java方法重载（而非默认参数）
    public static String greetWithTitle(String name) {
        return greetWithTitle(name, "先生");
    }
    
    public static String greetWithTitle(String name, String title) {
        return "你好, " + title + name + "!";
    }
    
    // Java可变参数（varargs）
    public static int sumAll(int... numbers) {
        int total = 0;
        for (int num : numbers) {
            total += num;
        }
        return total;
    }
    
    // Java没有**kwargs等价物，需要使用Map
    public static void printInfo(Map<String, Object> kwargs) {
        for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
    
    // 函数式接口（Java 8+）
    @FunctionalInterface
    interface Operation {
        int apply(int x, int y);
    }
    
    public static int applyOperation(int x, int y, Operation operation) {
        return operation.apply(x, y);
    }
    
    public static void javaFunctions() {
        System.out.println(greet("张三"));
        System.out.println(greetWithTitle("张三"));
        System.out.println(greetWithTitle("李四", "女士"));
        System.out.println("求和: " + sumAll(1, 2, 3, 4, 5));
        
        // Lambda表达式（Java 8+）
        Operation add = (x, y) -> x + y;
        Operation multiply = (x, y) -> x * y;
        
        System.out.println("应用加法: " + applyOperation(10, 5, add));
        System.out.println("应用乘法: " + applyOperation(10, 5, multiply));
    }
}

函数对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 定义关键字     │ def                 │ 访问修饰符+返回类型 │
│ 默认参数       │ 原生支持            │ 使用方法重载        │
│ 可变参数       │ *args, **kwargs     │ varargs, Map        │
│ 一等公民       │ 是                  │ 需要函数式接口      │
│ Lambda         │ lambda x: x+1       │ x -> x+1            │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 7. 字符串处理
# ============================================================================

print("\n=== 7. 字符串处理 ===\n")

def python_strings():
    """
    Python字符串特点：
    - 不可变
    - 支持多种引号
    - 强大的格式化功能
    - 丰富的字符串方法
    """
    # 字符串定义
    str1 = '单引号'
    str2 = "双引号"
    str3 = """
    三引号多行字符串
    可以跨越多行
    """
    str4 = r"原始字符串\n不转义"  # r前缀
    
    # 字符串格式化
    name = "张三"
    age = 25
    
    # 1. f-string（推荐，Python 3.6+）
    formatted1 = f"我是{name}，今年{age}岁，明年{age+1}岁"
    print(formatted1)
    
    # 2. format方法
    formatted2 = "我是{}，今年{}岁".format(name, age)
    formatted3 = "我是{0}，今年{1}岁，{0}很高兴".format(name, age)
    formatted4 = "我是{name}，今年{age}岁".format(name=name, age=age)
    
    # 3. %格式化（老式，不推荐）
    formatted5 = "我是%s，今年%d岁" % (name, age)
    
    # 字符串操作
    text = "Hello World"
    print(f"大写: {text.upper()}")
    print(f"小写: {text.lower()}")
    print(f"首字母大写: {text.capitalize()}")
    print(f"标题格式: {text.title()}")
    print(f"替换: {text.replace('World', 'Python')}")
    print(f"分割: {text.split()}")
    print(f"去空格: {'  test  '.strip()}")
    print(f"开始于: {text.startswith('Hello')}")
    print(f"结束于: {text.endswith('World')}")
    print(f"查找: {text.find('World')}")
    
    # 字符串拼接
    parts = ["Python", "is", "awesome"]
    joined = " ".join(parts)
    print(f"拼接: {joined}")
    
    # 字符串切片
    text = "0123456789"
    print(f"切片 [2:5]: {text[2:5]}")      # "234"
    print(f"切片 [:5]: {text[:5]}")        # "01234"
    print(f"切片 [5:]: {text[5:]}")        # "56789"
    print(f"切片 [::2]: {text[::2]}")      # "02468"
    print(f"切片 [::-1]: {text[::-1]}")    # "9876543210" (反转)

"""
Java对比:
public class JavaStrings {
    public static void javaStrings() {
        // 字符串定义
        String str1 = "Java只支持双引号";
        String str2 = "多行字符串需要拼接\n" +
                      "使用 + 运算符";
        
        // Java字符串格式化
        String name = "张三";
        int age = 25;
        
        // 1. String.format
        String formatted1 = String.format("我是%s，今年%d岁", name, age);
        
        // 2. 字符串拼接
        String formatted2 = "我是" + name + "，今年" + age + "岁";
        
        // 3. StringBuilder（高效）
        StringBuilder sb = new StringBuilder();
        sb.append("我是").append(name).append("，今年").append(age).append("岁");
        String formatted3 = sb.toString();
        
        
        // Java 15+ Text Blocks (示例：多行文本块)
        
        // 字符串操作
        String text = "Hello World";
        System.out.println("大写: " + text.toUpperCase());
        System.out.println("小写: " + text.toLowerCase());
        System.out.println("替换: " + text.replace("World", "Java"));
        System.out.println("分割: " + Arrays.toString(text.split(" ")));
        System.out.println("去空格: " + "  test  ".trim());
        System.out.println("开始于: " + text.startsWith("Hello"));
        System.out.println("结束于: " + text.endsWith("World"));
        System.out.println("查找: " + text.indexOf("World"));
        
        // 字符串拼接
        List<String> parts = Arrays.asList("Java", "is", "powerful");
        String joined = String.join(" ", parts);
        
        // 字符串截取
        text = "0123456789";
        System.out.println("子串 (2,5): " + text.substring(2, 5));  // "234"
        System.out.println("子串 (0,5): " + text.substring(0, 5));  // "01234"
        System.out.println("子串 (5): " + text.substring(5));       // "56789"
        
        // 反转需要StringBuilder
        String reversed = new StringBuilder(text).reverse().toString();
    }
}

字符串对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 引号类型       │ 单引号/双引号/三引号│ 只支持双引号        │
│ 格式化         │ f-string (f"{var}") │ String.format()     │
│ 多行字符串     │ 三引号              │ Text Blocks(Java15+)│
│ 切片           │ str[start:end]      │ substring()         │
│ 反转           │ str[::-1]           │ StringBuilder.rev() │
│ 拼接性能       │ join()              │ StringBuilder       │
└────────────────┴─────────────────────┴─────────────────────┘
"""

def main():
    """演示所有基础语法差异"""
    print("=" * 70)
    print("Python vs Java: 基础语法对比")
    print("=" * 70)
    
    python_indentation_demo()
    python_variables()
    python_conditionals()
    python_loops()
    python_functions()
    python_strings()
    
    print("\n" + "=" * 70)
    print("学习建议")
    print("=" * 70)
    print("""
    1. 适应缩进：Python强制使用缩进，配置IDE使用4空格
    2. 拥抱动态类型：无需声明类型，但可以使用类型提示
    3. 学习Python风格：阅读PEP 8代码规范
    4. 使用内置函数：len(), range(), enumerate()等
    5. 掌握切片语法：这是Python的强大特性
    6. f-string格式化：Python 3.6+推荐使用
    7. 理解可迭代对象：for循环的核心概念
    8. 利用多重赋值：简化代码，如 a, b = b, a
    
    常见陷阱：
    - 不要混用Tab和空格
    - 注意可变默认参数的坑
    - 理解is和==的区别
    - 字符串是不可变的
    """)

if __name__ == "__main__":
    main()
