"""
Python与Java的主要区别对比
============================

本文件详细对比了Python和Java两种编程语言的主要差异
包括语法、面向对象、类型系统、性能等方面的区别
"""

# ============================================================================
# 1. 语法差异对比
# ============================================================================

print("=== 1. 语法差异 ===")

# Python: 简洁的语法，使用缩进表示代码块
def python_example():
    """Python函数示例 - 使用缩进和冒号"""
    numbers = [1, 2, 3, 4, 5]
    for num in numbers:
        if num % 2 == 0:
            print(f"偶数: {num}")
        else:
            print(f"奇数: {num}")

# Java对应代码（注释形式展示）:
"""
// Java需要显式的大括号和分号
public class JavaExample {
    public static void javaExample() {
        int[] numbers = {1, 2, 3, 4, 5};
        for (int num : numbers) {
            if (num % 2 == 0) {
                System.out.println("偶数: " + num);
            } else {
                System.out.println("奇数: " + num);
            }
        }
    }
}
"""

# ============================================================================
# 2. 类型系统差异
# ============================================================================

print("\n=== 2. 类型系统差异 ===")

# Python: 动态类型
def python_dynamic_typing():
    """Python动态类型示例"""
    variable = 42          # 整数
    print(f"整数: {variable}, 类型: {type(variable)}")
    
    variable = "Hello"     # 字符串
    print(f"字符串: {variable}, 类型: {type(variable)}")
    
    variable = [1, 2, 3]   # 列表
    print(f"列表: {variable}, 类型: {type(variable)}")

# Python类型提示（可选）
def python_type_hints(name: str, age: int) -> str:
    """Python类型提示示例（运行时不强制）"""
    return f"{name} is {age} years old"

# Java对应代码（注释形式）:
"""
// Java: 静态类型，编译时检查
public class JavaStaticTyping {
    public static void javaStaticTyping() {
        int variable = 42;              // 必须声明类型
        String message = "Hello";       // 类型固定
        List<Integer> numbers = Arrays.asList(1, 2, 3);
    }
    
    public static String javaMethod(String name, int age) {
        return name + " is " + age + " years old";
    }
}
"""

# ============================================================================
# 3. 面向对象编程差异
# ============================================================================

print("\n=== 3. 面向对象编程差异 ===")

# Python: 简单的类定义
class PythonClass:
    """Python类示例"""
    
    # 类变量
    class_variable = "我是类变量"
    
    def __init__(self, name, age):
        """构造函数"""
        self.name = name        # 公有属性
        self._protected = age   # 受保护属性（约定）
        self.__private = "私有" # 私有属性（名称改写）
    
    def public_method(self):
        """公有方法"""
        return f"Hello, {self.name}"
    
    def _protected_method(self):
        """受保护方法（约定）"""
        return self._protected
    
    def __private_method(self):
        """私有方法"""
        return self.__private
    
    @staticmethod
    def static_method():
        """静态方法"""
        return "这是静态方法"
    
    @classmethod
    def class_method(cls):
        """类方法"""
        return f"类方法访问: {cls.class_variable}"

# Python继承
class PythonChild(PythonClass):
    """Python继承示例"""
    
    def __init__(self, name, age, grade):
        super().__init__(name, age)  # 调用父类构造函数
        self.grade = grade
    
    def public_method(self):
        """方法重写"""
        parent_result = super().public_method()
        return f"{parent_result}, 年级: {self.grade}"

# Java对应代码（注释形式）:
"""
// Java: 更严格的访问控制
public class JavaClass {
    // 类变量
    public static String classVariable = "我是类变量";
    
    // 实例变量
    public String name;           // 公有
    protected int age;            // 受保护
    private String privateField;  // 私有
    
    // 构造函数
    public JavaClass(String name, int age) {
        this.name = name;
        this.age = age;
        this.privateField = "私有";
    }
    
    // 公有方法
    public String publicMethod() {
        return "Hello, " + this.name;
    }
    
    // 受保护方法
    protected int protectedMethod() {
        return this.age;
    }
    
    // 私有方法
    private String privateMethod() {
        return this.privateField;
    }
    
    // 静态方法
    public static String staticMethod() {
        return "这是静态方法";
    }
}

// Java继承
public class JavaChild extends JavaClass {
    private String grade;
    
    public JavaChild(String name, int age, String grade) {
        super(name, age);  // 调用父类构造函数
        this.grade = grade;
    }
    
    @Override  // 方法重写注解
    public String publicMethod() {
        String parentResult = super.publicMethod();
        return parentResult + ", 年级: " + this.grade;
    }
}
"""

# ============================================================================
# 4. 内存管理差异
# ============================================================================

print("\n=== 4. 内存管理差异 ===")

def python_memory_management():
    """Python内存管理示例"""
    # Python: 自动垃圾回收
    large_list = [i for i in range(1000000)]
    print(f"创建了包含 {len(large_list)} 个元素的列表")
    
    # Python会自动回收内存，无需手动管理
    del large_list  # 可选的显式删除
    print("列表已删除，内存将被自动回收")

# Java对应概念（注释形式）:
"""
// Java: 自动垃圾回收 + JVM管理
public class JavaMemoryManagement {
    public static void javaMemoryManagement() {
        // Java也有自动垃圾回收
        List<Integer> largeList = new ArrayList<>();
        for (int i = 0; i < 1000000; i++) {
            largeList.add(i);
        }
        System.out.println("创建了包含 " + largeList.size() + " 个元素的列表");
        
        // Java也会自动回收内存
        largeList = null;  // 解除引用
        System.gc();       // 建议进行垃圾回收（不保证立即执行）
    }
}
"""

# ============================================================================
# 5. 异常处理差异
# ============================================================================

print("\n=== 5. 异常处理差异 ===")

def python_exception_handling():
    """Python异常处理示例"""
    try:
        result = 10 / 0  # 会引发ZeroDivisionError
    except ZeroDivisionError as e:
        print(f"捕获到除零错误: {e}")
    except Exception as e:
        print(f"捕获到其他异常: {e}")
    else:
        print("没有异常发生时执行")
    finally:
        print("无论是否有异常都会执行")

# 自定义异常
class CustomPythonException(Exception):
    """自定义Python异常"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Java对应代码（注释形式）:
"""
// Java: 检查异常和非检查异常
public class JavaExceptionHandling {
    public static void javaExceptionHandling() {
        try {
            int result = 10 / 0;  // 会引发ArithmeticException
        } catch (ArithmeticException e) {
            System.out.println("捕获到算术异常: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("捕获到其他异常: " + e.getMessage());
        } finally {
            System.out.println("无论是否有异常都会执行");
        }
    }
    
    // 自定义异常
    public static class CustomJavaException extends Exception {
        public CustomJavaException(String message) {
            super(message);
        }
    }
    
    // 检查异常必须声明或处理
    public static void methodWithCheckedException() throws CustomJavaException {
        throw new CustomJavaException("这是一个检查异常");
    }
}
"""

# ============================================================================
# 6. 数据结构和集合差异
# ============================================================================

print("\n=== 6. 数据结构和集合差异 ===")

def python_data_structures():
    """Python数据结构示例"""
    # 列表 (类似Java的ArrayList)
    python_list = [1, 2, 3, "混合类型", True]
    print(f"Python列表: {python_list}")
    
    # 元组 (不可变)
    python_tuple = (1, 2, 3)
    print(f"Python元组: {python_tuple}")
    
    # 字典 (类似Java的HashMap)
    python_dict = {"name": "张三", "age": 25, "city": "北京"}
    print(f"Python字典: {python_dict}")
    
    # 集合 (类似Java的HashSet)
    python_set = {1, 2, 3, 3, 4}  # 自动去重
    print(f"Python集合: {python_set}")
    
    # 列表推导式
    squares = [x**2 for x in range(5)]
    print(f"列表推导式: {squares}")

# Java对应代码（注释形式）:
"""
// Java: 泛型集合，类型安全
import java.util.*;

public class JavaDataStructures {
    public static void javaDataStructures() {
        // ArrayList (动态数组)
        List<Object> javaList = new ArrayList<>();
        javaList.add(1);
        javaList.add("字符串");
        javaList.add(true);
        System.out.println("Java列表: " + javaList);
        
        // 不可变数组
        int[] javaArray = {1, 2, 3};
        System.out.println("Java数组: " + Arrays.toString(javaArray));
        
        // HashMap
        Map<String, Object> javaMap = new HashMap<>();
        javaMap.put("name", "张三");
        javaMap.put("age", 25);
        javaMap.put("city", "北京");
        System.out.println("Java映射: " + javaMap);
        
        // HashSet
        Set<Integer> javaSet = new HashSet<>();
        javaSet.add(1);
        javaSet.add(2);
        javaSet.add(3);
        javaSet.add(3);  // 重复元素会被忽略
        System.out.println("Java集合: " + javaSet);
        
        // Stream API (Java 8+)
        List<Integer> squares = Arrays.asList(0, 1, 2, 3, 4)
            .stream()
            .map(x -> x * x)
            .collect(Collectors.toList());
        System.out.println("Stream处理: " + squares);
    }
}
"""

# ============================================================================
# 7. 性能和执行差异
# ============================================================================

print("\n=== 7. 性能和执行差异 ===")

import time

def python_performance_demo():
    """Python性能示例"""
    start_time = time.time()
    
    # Python: 解释执行，相对较慢
    total = 0
    for i in range(1000000):
        total += i
    
    end_time = time.time()
    print(f"Python计算耗时: {end_time - start_time:.4f} 秒")
    print(f"结果: {total}")

# Java对应性能（注释形式）:
"""
// Java: 编译为字节码，JVM优化，通常更快
public class JavaPerformance {
    public static void javaPerformanceDemo() {
        long startTime = System.currentTimeMillis();
        
        long total = 0;
        for (int i = 0; i < 1000000; i++) {
            total += i;
        }
        
        long endTime = System.currentTimeMillis();
        System.out.println("Java计算耗时: " + (endTime - startTime) + " 毫秒");
        System.out.println("结果: " + total);
    }
}
"""

# ============================================================================
# 8. 主要差异总结
# ============================================================================

def print_summary():
    """打印Python与Java的主要差异总结"""
    print("\n" + "="*60)
    print("Python与Java主要差异总结")
    print("="*60)
    
    differences = [
        ("语法风格", "Python: 简洁、优雅、缩进敏感", "Java: 详细、严格、大括号结构"),
        ("类型系统", "Python: 动态类型、鸭子类型", "Java: 静态类型、编译时检查"),
        ("编译执行", "Python: 解释执行", "Java: 编译为字节码，JVM执行"),
        ("性能", "Python: 相对较慢，但开发效率高", "Java: 性能较好，优化程度高"),
        ("内存管理", "Python: 引用计数+垃圾回收", "Java: 自动垃圾回收"),
        ("平台依赖", "Python: 跨平台，需要Python解释器", "Java: 一次编写，到处运行（JVM）"),
        ("学习曲线", "Python: 容易学习，语法简单", "Java: 学习曲线较陡，概念较多"),
        ("应用领域", "Python: 数据科学、AI、Web、脚本", "Java: 企业应用、Web后端、Android"),
        ("社区生态", "Python: 丰富的科学计算库", "Java: 成熟的企业级框架"),
        ("开发效率", "Python: 快速原型开发", "Java: 大型项目维护性好")
    ]
    
    for aspect, python_feature, java_feature in differences:
        print(f"\n{aspect}:")
        print(f"  • {python_feature}")
        print(f"  • {java_feature}")

# ============================================================================
# 9. 选择建议
# ============================================================================

def print_choice_recommendations():
    """打印选择建议"""
    print("\n" + "="*60)
    print("选择建议")
    print("="*60)
    
    print("\n选择Python的场景:")
    print("• 数据分析和科学计算")
    print("• 机器学习和人工智能")
    print("• 快速原型开发")
    print("• 自动化脚本和工具")
    print("• Web开发（Django、Flask）")
    print("• 初学者学习编程")
    
    print("\n选择Java的场景:")
    print("• 大型企业级应用")
    print("• Android移动应用开发")
    print("• 高性能Web后端服务")
    print("• 分布式系统开发")
    print("• 银行和金融系统")
    print("• 需要严格类型检查的项目")

# ============================================================================
# 主函数 - 演示所有差异
# ============================================================================

def main():
    """主函数 - 演示Python与Java的差异"""
    print("Python与Java编程语言对比演示")
    print("="*60)
    
    # 执行各个演示函数
    python_example()
    python_dynamic_typing()
    
    # 创建对象演示
    print("\n=== 面向对象演示 ===")
    obj = PythonClass("小明", 20)
    print(obj.public_method())
    print(PythonClass.static_method())
    print(PythonClass.class_method())
    
    child_obj = PythonChild("小红", 18, "高三")
    print(child_obj.public_method())
    
    # 其他演示
    python_memory_management()
    python_exception_handling()
    python_data_structures()
    python_performance_demo()
    
    # 打印总结
    print_summary()
    print_choice_recommendations()

if __name__ == "__main__":
    main()