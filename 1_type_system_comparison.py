"""
Python vs Java: 类型系统深度对比
============================
面向Java程序员的Python类型系统学习指南

作者: Python学习系列
目标读者: 熟悉Java静态类型的开发者
学习重点: 理解动态类型与静态类型的差异
"""

# ============================================================================
# 1. 类型系统基础对比
# ============================================================================

print("=== 1. 类型系统基础对比 ===\n")

def type_system_basics():
    """
    核心差异：
    - Java: 静态类型 (编译时类型检查)
    - Python: 动态类型 (运行时类型检查)
    """
    
    # Python: 运行时确定类型
    variable = 42
    print(f"变量类型: {type(variable)}")  # <class 'int'>
    
    variable = "现在是字符串"
    print(f"变量类型: {type(variable)}")  # <class 'str'>
    
    variable = [1, 2, 3]
    print(f"变量类型: {type(variable)}")  # <class 'list'>
    
    # 鸭子类型："如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子"
    class Duck:
        def quack(self):
            return "嘎嘎嘎"
    
    class Person:
        def quack(self):
            return "我在模仿鸭子叫"
    
    def make_it_quack(duck_like):
        """只要有quack方法，就能调用"""
        print(duck_like.quack())
    
    make_it_quack(Duck())
    make_it_quack(Person())  # Python不关心类型，只关心行为

"""
Java对比:
public class TypeSystemBasics {
    public static void typeSystemBasics() {
        // Java: 编译时确定类型，不能改变
        int variable = 42;
        // variable = "字符串";  // 编译错误！
        
        // 必须声明类型
        String text = "文本";
        List<Integer> numbers = Arrays.asList(1, 2, 3);
    }
    
    // Java需要接口或继承来实现多态
    interface Quackable {
        String quack();
    }
    
    static class Duck implements Quackable {
        public String quack() {
            return "嘎嘎嘎";
        }
    }
    
    static class Person implements Quackable {
        public String quack() {
            return "我在模仿鸭子叫";
        }
    }
    
    static void makeItQuack(Quackable quackable) {
        System.out.println(quackable.quack());
    }
}

关键差异：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 类型检查       │ 运行时              │ 编译时              │
│ 类型声明       │ 可选                │ 必须                │
│ 类型改变       │ 允许                │ 不允许              │
│ 多态实现       │ 鸭子类型            │ 接口/继承           │
│ 类型安全       │ 运行时错误          │ 编译时错误          │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 2. Python类型提示 (Type Hints)
# ============================================================================

print("\n=== 2. Python类型提示 ===\n")

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

def python_type_hints():
    """
    Python 3.5+ 引入类型提示
    注意：类型提示不强制执行，仅用于静态分析工具
    """
    
    # 基本类型提示
    def greet(name: str, age: int) -> str:
        return f"{name} is {age} years old"
    
    # 集合类型提示
    def process_numbers(numbers: List[int]) -> int:
        """处理整数列表"""
        return sum(numbers)
    
    def get_user_info() -> Dict[str, str]:
        """返回字典"""
        return {"name": "张三", "city": "北京"}
    
    def get_coordinates() -> Tuple[float, float]:
        """返回元组"""
        return (39.9, 116.4)
    
    # Optional类型 (可以是None)
    def find_user(user_id: int) -> Optional[str]:
        """可能返回None"""
        if user_id > 0:
            return "用户名"
        return None
    
    # Union类型 (多种可能的类型)
    def process_id(id_value: Union[int, str]) -> str:
        """可以接受int或str"""
        return str(id_value)
    
    # Any类型 (任意类型)
    def accept_anything(value: Any) -> Any:
        """接受任意类型"""
        return value
    
    # 函数类型
    def apply_function(func: Callable[[int, int], int], x: int, y: int) -> int:
        """接受一个函数作为参数"""
        return func(x, y)
    
    # 测试
    print(greet("张三", 25))
    print(process_numbers([1, 2, 3, 4, 5]))
    print(get_user_info())
    print(get_coordinates())
    print(find_user(1))
    print(find_user(-1))
    print(process_id(123))
    print(process_id("ABC"))
    
    # 类型提示不强制执行！
    # print(greet(123, "年龄"))  # 运行时不会报错，但类型检查工具会警告

"""
Java对比:
import java.util.*;
import java.util.function.BiFunction;

public class JavaTypeSystem {
    // Java类型声明是强制的
    public static String greet(String name, int age) {
        return name + " is " + age + " years old";
    }
    
    // 泛型类型
    public static int processNumbers(List<Integer> numbers) {
        return numbers.stream().mapToInt(Integer::intValue).sum();
    }
    
    public static Map<String, String> getUserInfo() {
        Map<String, String> map = new HashMap<>();
        map.put("name", "张三");
        map.put("city", "北京");
        return map;
    }
    
    // Java没有元组，需要创建类或使用数组
    public static double[] getCoordinates() {
        return new double[]{39.9, 116.4};
    }
    
    // Optional类 (Java 8+)
    public static Optional<String> findUser(int userId) {
        if (userId > 0) {
            return Optional.of("用户名");
        }
        return Optional.empty();
    }
    
    // Java没有Union类型，需要使用继承或重载
    public static String processId(int idValue) {
        return String.valueOf(idValue);
    }
    
    public static String processId(String idValue) {
        return idValue;
    }
    
    // 函数式接口
    public static int applyFunction(BiFunction<Integer, Integer, Integer> func, 
                                   int x, int y) {
        return func.apply(x, y);
    }
    
    public static void main(String[] args) {
        // Java在编译时检查类型
        System.out.println(greet("张三", 25));
        // System.out.println(greet(123, "年龄"));  // 编译错误！
    }
}

类型提示对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 类型标注       │ 可选，仅用于检查    │ 强制，编译时检查    │
│ Optional       │ Optional[T]         │ Optional<T>         │
│ Union类型      │ Union[int, str]     │ 不支持(用重载)      │
│ 泛型           │ List[int]           │ List<Integer>       │
│ 元组           │ Tuple[int, str]     │ 无内置(用类)        │
│ 类型检查工具   │ mypy, pyright       │ javac (编译器)      │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 3. 内置数据类型对比
# ============================================================================

print("\n=== 3. 内置数据类型对比 ===\n")

def python_builtin_types():
    """
    Python丰富的内置类型
    """
    
    # 数值类型
    integer = 42                    # int (整数，无限精度)
    floating = 3.14                 # float (浮点数)
    complex_num = 3 + 4j           # complex (复数)
    boolean = True                  # bool (布尔值)
    
    print(f"整数: {integer}, 类型: {type(integer)}")
    print(f"浮点: {floating}, 类型: {type(floating)}")
    print(f"复数: {complex_num}, 类型: {type(complex_num)}")
    print(f"布尔: {boolean}, 类型: {type(boolean)}")
    
    # Python的int支持任意大小
    big_number = 99999999999999999999999999999999
    print(f"大整数: {big_number}")
    
    # 字符串类型
    text = "Python字符串"           # str (不可变)
    print(f"字符串: {text}, 类型: {type(text)}")
    
    # 字节类型
    byte_data = b"bytes"           # bytes (不可变字节序列)
    byte_array = bytearray(b"mutable")  # bytearray (可变字节序列)
    print(f"字节: {byte_data}, 类型: {type(byte_data)}")
    
    # 序列类型
    my_list = [1, 2, 3, "混合"]     # list (可变列表)
    my_tuple = (1, 2, 3)           # tuple (不可变元组)
    my_range = range(5)            # range (范围对象)
    
    print(f"列表: {my_list}, 类型: {type(my_list)}")
    print(f"元组: {my_tuple}, 类型: {type(my_tuple)}")
    print(f"范围: {list(my_range)}, 类型: {type(my_range)}")
    
    # 映射类型
    my_dict = {"key": "value"}     # dict (字典)
    print(f"字典: {my_dict}, 类型: {type(my_dict)}")
    
    # 集合类型
    my_set = {1, 2, 3}             # set (可变集合)
    my_frozenset = frozenset([1, 2, 3])  # frozenset (不可变集合)
    print(f"集合: {my_set}, 类型: {type(my_set)}")
    
    # None类型
    nothing = None                 # NoneType
    print(f"None: {nothing}, 类型: {type(nothing)}")

"""
Java对比:
import java.util.*;
import java.math.BigInteger;

public class JavaBuiltinTypes {
    public static void javaBuiltinTypes() {
        // 数值类型（基本类型）
        byte b = 127;                    // 8位整数
        short s = 32767;                 // 16位整数
        int i = 2147483647;              // 32位整数
        long l = 9223372036854775807L;   // 64位整数
        float f = 3.14f;                 // 32位浮点
        double d = 3.14159;              // 64位浮点
        boolean bool = true;             // 布尔值
        char c = 'A';                    // 字符
        
        // Java整数有大小限制，大整数需要BigInteger
        BigInteger bigNum = new BigInteger("99999999999999999999999999999999");
        
        // 包装类（对象类型）
        Integer integer = 42;
        Double floating = 3.14;
        Boolean boolObj = true;
        
        // 字符串（对象类型）
        String text = "Java字符串";  // 不可变
        
        // 字节数组
        byte[] byteData = "bytes".getBytes();
        
        // 集合类型（需要泛型）
        List<Object> list = Arrays.asList(1, 2, 3, "混合");
        Map<String, String> map = new HashMap<>();
        Set<Integer> set = new HashSet<>(Arrays.asList(1, 2, 3));
        
        // Java没有元组，没有range
        // Java没有frozenset等价物
        
        // null
        String nothing = null;
    }
}

数据类型对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   类型         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 整数           │ int (无限精度)      │ int/long/BigInteger │
│ 浮点数         │ float               │ float/double        │
│ 复数           │ complex             │ 不支持              │
│ 字符串         │ str                 │ String              │
│ 列表           │ list                │ ArrayList           │
│ 元组           │ tuple               │ 无内置(用数组/类)   │
│ 字典           │ dict                │ HashMap             │
│ 集合           │ set                 │ HashSet             │
│ 空值           │ None                │ null                │
│ 布尔值         │ True/False          │ true/false          │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 4. 类型转换
# ============================================================================

print("\n=== 4. 类型转换 ===\n")

def python_type_conversion():
    """
    Python类型转换
    """
    
    # 隐式转换（很少）
    result = 3 + 4.5  # int自动转为float
    print(f"3 + 4.5 = {result}, 类型: {type(result)}")
    
    # 显式转换（推荐）
    int_val = int("42")           # 字符串转整数
    float_val = float("3.14")     # 字符串转浮点
    str_val = str(42)             # 整数转字符串
    bool_val = bool(1)            # 整数转布尔
    list_val = list("abc")        # 字符串转列表
    tuple_val = tuple([1, 2, 3])  # 列表转元组
    set_val = set([1, 2, 2, 3])   # 列表转集合(去重)
    
    print(f"int('42') = {int_val}")
    print(f"float('3.14') = {float_val}")
    print(f"str(42) = {str_val}")
    print(f"bool(1) = {bool_val}")
    print(f"list('abc') = {list_val}")
    print(f"tuple([1,2,3]) = {tuple_val}")
    print(f"set([1,2,2,3]) = {set_val}")
    
    # 真值测试
    print("\n真值测试:")
    print(f"bool(0) = {bool(0)}")           # False
    print(f"bool('') = {bool('')}")         # False
    print(f"bool([]) = {bool([])}")         # False
    print(f"bool({{}}) = {bool({})}")       # False
    print(f"bool(None) = {bool(None)}")     # False
    print(f"bool(42) = {bool(42)}")         # True
    print(f"bool('text') = {bool('text')}")  # True
    
    # 异常处理
    try:
        invalid = int("not a number")
    except ValueError as e:
        print(f"\n转换错误: {e}")

"""
Java对比:
public class JavaTypeConversion {
    public static void javaTypeConversion() {
        // 隐式转换（自动提升）
        int i = 3;
        double d = i;  // int自动转为double
        double result = 3 + 4.5;  // int自动转为double
        
        // 显式转换（强制转换）
        double d2 = 3.14;
        int i2 = (int) d2;  // 强制转换，会丢失精度
        
        // 包装类转换
        int intVal = Integer.parseInt("42");
        double doubleVal = Double.parseDouble("3.14");
        String strVal = String.valueOf(42);
        boolean boolVal = Boolean.parseBoolean("true");
        
        // 集合转换
        String[] array = {"a", "b", "c"};
        List<String> list = Arrays.asList(array);
        Set<String> set = new HashSet<>(list);
        
        // 异常处理
        try {
            int invalid = Integer.parseInt("not a number");
        } catch (NumberFormatException e) {
            System.out.println("转换错误: " + e.getMessage());
        }
    }
}

类型转换对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   转换         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 字符串→整数    │ int("42")           │ Integer.parseInt()  │
│ 整数→字符串    │ str(42)             │ String.valueOf()    │
│ 强制转换       │ int(3.14)           │ (int) 3.14          │
│ 真值测试       │ bool(value)         │ 不适用              │
│ 自动转换       │ 很少                │ 数值类型自动提升    │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 5. 类型检查与isinstance
# ============================================================================

print("\n=== 5. 类型检查 ===\n")

def python_type_checking():
    """
    Python运行时类型检查
    """
    
    # type()获取类型
    value = 42
    print(f"type(42) = {type(value)}")
    print(f"type(value) == int: {type(value) == int}")
    
    # isinstance()检查类型（推荐）
    print(f"\nisinstance检查:")
    print(f"isinstance(42, int): {isinstance(42, int)}")
    print(f"isinstance(42, (int, float)): {isinstance(42, (int, float))}")
    print(f"isinstance('text', str): {isinstance('text', str)}")
    print(f"isinstance([1,2], list): {isinstance([1, 2], list)}")
    
    # 检查继承关系
    class Animal:
        pass
    
    class Dog(Animal):
        pass
    
    dog = Dog()
    print(f"\nisinstance(dog, Dog): {isinstance(dog, Dog)}")
    print(f"isinstance(dog, Animal): {isinstance(dog, Animal)}")  # True
    print(f"type(dog) == Dog: {type(dog) == Dog}")
    print(f"type(dog) == Animal: {type(dog) == Animal}")  # False
    
    # hasattr检查属性
    class MyClass:
        def __init__(self):
            self.value = 42
        
        def method(self):
            pass
    
    obj = MyClass()
    print(f"\nhasattr检查:")
    print(f"hasattr(obj, 'value'): {hasattr(obj, 'value')}")
    print(f"hasattr(obj, 'method'): {hasattr(obj, 'method')}")
    print(f"hasattr(obj, 'missing'): {hasattr(obj, 'missing')}")
    
    # callable检查是否可调用
    print(f"\ncallable检查:")
    print(f"callable(print): {callable(print)}")
    print(f"callable(42): {callable(42)}")
    print(f"callable(lambda x: x): {callable(lambda x: x)}")

"""
Java对比:
public class JavaTypeChecking {
    public static void javaTypeChecking() {
        // instanceof检查类型
        Object value = 42;
        System.out.println("value instanceof Integer: " + (value instanceof Integer));
        System.out.println("value instanceof Number: " + (value instanceof Number));
        
        // getClass()获取类型
        System.out.println("value.getClass(): " + value.getClass());
        System.out.println("value.getClass() == Integer.class: " + 
                          (value.getClass() == Integer.class));
        
        // 检查继承关系
        class Animal {}
        class Dog extends Animal {}
        
        Dog dog = new Dog();
        System.out.println("dog instanceof Dog: " + (dog instanceof Dog));
        System.out.println("dog instanceof Animal: " + (dog instanceof Animal));
        
        // 反射检查属性和方法
        try {
            Class<?> clazz = dog.getClass();
            System.out.println("有方法: " + (clazz.getMethods().length > 0));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

类型检查对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 获取类型       │ type(obj)           │ obj.getClass()      │
│ 类型判断       │ isinstance(obj, T)  │ obj instanceof T    │
│ 严格相等       │ type(obj) == T      │ obj.getClass() == T │
│ 属性检查       │ hasattr(obj, attr)  │ 反射API             │
│ 可调用检查     │ callable(obj)       │ 反射API             │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 6. 使用mypy进行静态类型检查
# ============================================================================

print("\n=== 6. 静态类型检查工具 ===\n")

def mypy_example():
    """
    虽然Python是动态类型，但可以使用mypy进行静态检查
    """
    
    # 良好的类型提示示例
    def calculate_total(prices: List[float], tax_rate: float = 0.1) -> float:
        """计算总价（含税）"""
        subtotal = sum(prices)
        return subtotal * (1 + tax_rate)
    
    # 使用示例
    result = calculate_total([10.0, 20.0, 30.0])
    print(f"总价: {result}")
    
    # mypy会检查这些错误（但Python运行时不会）
    # wrong = calculate_total(["10", "20"])  # mypy错误：类型不匹配
    # wrong = calculate_total([10, 20], "0.1")  # mypy错误：参数类型错误

"""
使用mypy:
$ pip install mypy
$ mypy your_file.py

mypy配置文件 (mypy.ini):
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

学习建议：
1. 大型项目中使用类型提示
2. 配置mypy或pyright进行类型检查
3. 在IDE中启用类型检查
4. 逐步为现有代码添加类型提示
"""

# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示类型系统差异"""
    print("=" * 70)
    print("Python vs Java: 类型系统深度对比")
    print("=" * 70)
    
    type_system_basics()
    python_type_hints()
    python_builtin_types()
    python_type_conversion()
    python_type_checking()
    mypy_example()
    
    print("\n" + "=" * 70)
    print("学习建议")
    print("=" * 70)
    print("""
    对于Java程序员学习Python类型系统：
    
    1. 拥抱动态类型
       - 无需声明类型，提高开发速度
       - 但需要更多的运行时测试
    
    2. 理解鸭子类型
       - 关注对象的行为，而非类型
       - "如果它看起来像鸭子，那就把它当鸭子"
    
    3. 使用类型提示
       - 大型项目中添加类型提示
       - 使用mypy或pyright进行静态检查
       - 类型提示不影响运行时性能
    
    4. 类型检查最佳实践
       - 使用isinstance()而非type()
       - 检查行为而非类型(hasattr)
       - 遵循EAFP原则(Easier to Ask Forgiveness than Permission)
    
    5. 常见陷阱
       - 类型提示不强制执行
       - 注意可变默认参数
       - 整数除法 / 返回float，使用 // 整除
       - True/False是int的子类
    
    6. 工具推荐
       - mypy: 静态类型检查
       - pyright: Microsoft的类型检查器
       - pylint: 代码质量检查
       - black: 代码格式化
    """)

if __name__ == "__main__":
    main()
