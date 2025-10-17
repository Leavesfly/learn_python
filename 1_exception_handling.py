"""
Python vs Java: 异常处理深度对比
============================
面向Java程序员的Python异常处理学习指南

作者: Python学习系列
目标读者: 熟悉Java异常处理的开发者
学习重点: 掌握Python异常处理机制
"""

# ============================================================================
# 1. 基本异常处理
# ============================================================================

print("=== 1. 基本异常处理 ===\n")

def basic_exception_handling():
    """
    Python异常处理基础
    - try-except: 捕获异常
    - else: 无异常时执行
    - finally: 总是执行
    """
    
    # 基本try-except
    try:
        result = 10 / 2
        print(f"结果: {result}")
    except ZeroDivisionError as e:
        print(f"捕获除零错误: {e}")
    
    # 捕获多个异常
    try:
        value = int("not a number")
    except (ValueError, TypeError) as e:
        print(f"捕获值错误或类型错误: {e}")
    
    # try-except-else-finally
    try:
        file_content = "模拟文件内容"
        print(f"读取文件: {file_content}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    else:
        print("文件读取成功，无异常")  # 没有异常时执行
    finally:
        print("清理资源（总是执行）")  # 无论是否有异常都执行

"""
Java对比:
public class BasicExceptionHandling {
    public static void basicExceptionHandling() {
        // 基本try-catch
        try {
            int result = 10 / 2;
            System.out.println("结果: " + result);
        } catch (ArithmeticException e) {
            System.out.println("捕获算术异常: " + e.getMessage());
        }
        
        // 捕获多个异常（Java 7+）
        try {
            int value = Integer.parseInt("not a number");
        } catch (NumberFormatException | NullPointerException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        // try-catch-finally（Java没有else子句）
        try {
            String fileContent = "模拟文件内容";
            System.out.println("读取文件: " + fileContent);
        } catch (Exception e) {
            System.out.println("捕获异常: " + e.getMessage());
        } finally {
            System.out.println("清理资源（总是执行）");
        }
    }
}

关键差异：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 捕获异常       │ except              │ catch               │
│ else子句       │ 支持                │ 不支持              │
│ finally子句    │ 支持                │ 支持                │
│ 多异常         │ except (E1, E2)     │ catch (E1 | E2)     │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 2. 异常层次结构
# ============================================================================

print("\n=== 2. 异常层次结构 ===\n")

def exception_hierarchy():
    """
    Python异常层次：
    - BaseException (所有异常的基类)
      - Exception (常规异常的基类)
        - ValueError, TypeError, etc.
      - SystemExit (系统退出)
      - KeyboardInterrupt (用户中断)
    """
    
    # 常见异常类型
    exceptions_demo = [
        ("ValueError", lambda: int("abc")),
        ("KeyError", lambda: {}["missing_key"]),
        ("IndexError", lambda: [1, 2, 3][10]),
        ("ZeroDivisionError", lambda: 10 / 0),
    ]
    
    for name, func in exceptions_demo:
        try:
            func()
        except Exception as e:
            print(f"{name}: {type(e).__name__} - {e}")
    
    # 额外示例（注释形式）
    # TypeError: "string" + 123
    # AttributeError: "string".missing_method()

"""
Java对比:
public class ExceptionHierarchy {
    /*
    Java异常层次：
    - Throwable
      - Error (系统错误，不应捕获)
      - Exception
        - RuntimeException (非检查异常)
          - NullPointerException
          - IndexOutOfBoundsException
          - etc.
        - IOException (检查异常)
        - SQLException (检查异常)
        - etc.
    */
    
    public static void exceptionHierarchy() {
        // RuntimeException示例
        try {
            String str = null;
            str.length();  // NullPointerException
        } catch (NullPointerException e) {
            System.out.println("空指针异常");
        }
        
        try {
            int[] arr = {1, 2, 3};
            int val = arr[10];  // ArrayIndexOutOfBoundsException
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("数组越界异常");
        }
    }
}

异常类型对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   Python异常   │   含义              │   Java等价          │
├────────────────┼─────────────────────┼─────────────────────┤
│ ValueError     │ 值错误              │ IllegalArgumentEx   │
│ TypeError      │ 类型错误            │ ClassCastException  │
│ KeyError       │ 键不存在            │ (Map无此异常)       │
│ IndexError     │ 索引越界            │ IndexOutOfBounds    │
│ AttributeError │ 属性不存在          │ NoSuchFieldEx       │
│ ZeroDivisionErr│ 除零错误            │ ArithmeticException │
│ FileNotFoundErr│ 文件未找到          │ FileNotFoundEx      │
│ IOError        │ IO错误              │ IOException         │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 3. 抛出异常
# ============================================================================

print("\n=== 3. 抛出异常 ===\n")

def raise_exceptions():
    """
    Python抛出异常：
    - raise关键字
    - 不区分检查异常和非检查异常
    """
    
    def validate_age(age):
        """验证年龄"""
        if age < 0:
            raise ValueError("年龄不能为负数")
        if age > 150:
            raise ValueError("年龄不合理")
        return age
    
    # 测试抛出异常
    try:
        validate_age(-5)
    except ValueError as e:
        print(f"年龄验证失败: {e}")
    
    # 重新抛出异常
    def process_data(data):
        try:
            result = int(data)
        except ValueError:
            print("转换失败，重新抛出异常")
            raise  # 重新抛出当前异常
    
    try:
        process_data("invalid")
    except ValueError as e:
        print(f"捕获重新抛出的异常: {e}")
    
    # 抛出新异常（异常链）
    def outer_function():
        try:
            inner_function()
        except ValueError as e:
            raise RuntimeError("外层处理失败") from e  # 异常链
    
    def inner_function():
        raise ValueError("内层错误")
    
    try:
        outer_function()
    except RuntimeError as e:
        print(f"异常链: {e}")
        print(f"原始异常: {e.__cause__}")

"""
Java对比:
public class RaiseExceptions {
    // 抛出异常
    public static int validateAge(int age) throws IllegalArgumentException {
        if (age < 0) {
            throw new IllegalArgumentException("年龄不能为负数");
        }
        if (age > 150) {
            throw new IllegalArgumentException("年龄不合理");
        }
        return age;
    }
    
    // 重新抛出异常
    public static void processData(String data) throws NumberFormatException {
        try {
            int result = Integer.parseInt(data);
        } catch (NumberFormatException e) {
            System.out.println("转换失败，重新抛出异常");
            throw e;  // 重新抛出
        }
    }
    
    // 异常链
    public static void outerFunction() throws RuntimeException {
        try {
            innerFunction();
        } catch (IllegalArgumentException e) {
            throw new RuntimeException("外层处理失败", e);  // 异常链
        }
    }
    
    public static void innerFunction() throws IllegalArgumentException {
        throw new IllegalArgumentException("内层错误");
    }
}

抛出异常对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 抛出异常       │ raise Exception     │ throw new Exception │
│ 重新抛出       │ raise               │ throw e             │
│ 异常链         │ raise E from e      │ new E(msg, e)       │
│ 声明异常       │ 不需要              │ throws声明          │
│ 检查异常       │ 无区分              │ 必须处理或声明      │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 4. 自定义异常
# ============================================================================

print("\n=== 4. 自定义异常 ===\n")

# 简单自定义异常
class ValidationError(Exception):
    """数据验证异常"""
    pass

# 带属性的自定义异常
class BusinessError(Exception):
    """业务异常"""
    
    def __init__(self, message, code=None):
        super().__init__(message)
        self.message = message
        self.code = code
    
    def __str__(self):
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

# 异常层次
class DatabaseError(Exception):
    """数据库异常基类"""
    pass

class ConnectionError(DatabaseError):
    """连接错误"""
    pass

class QueryError(DatabaseError):
    """查询错误"""
    pass

def test_custom_exceptions():
    """测试自定义异常"""
    
    # 使用简单自定义异常
    try:
        raise ValidationError("用户名格式不正确")
    except ValidationError as e:
        print(f"验证错误: {e}")
    
    # 使用带属性的异常
    try:
        raise BusinessError("库存不足", code="E1001")
    except BusinessError as e:
        print(f"业务错误: {e}")
        print(f"错误代码: {e.code}")
    
    # 捕获异常层次
    try:
        raise QueryError("SQL语法错误")
    except DatabaseError as e:  # 可以捕获父类异常
        print(f"数据库错误: {e}")

test_custom_exceptions()

"""
Java对比:
// 简单自定义异常
public class ValidationException extends Exception {
    public ValidationException(String message) {
        super(message);
    }
}

// 带属性的自定义异常
public class BusinessException extends Exception {
    private String code;
    
    public BusinessException(String message, String code) {
        super(message);
        this.code = code;
    }
    
    public String getCode() {
        return code;
    }
    
    @Override
    public String toString() {
        if (code != null) {
            return "[" + code + "] " + getMessage();
        }
        return getMessage();
    }
}

// 异常层次
public class DatabaseException extends Exception {
    public DatabaseException(String message) {
        super(message);
    }
}

public class ConnectionException extends DatabaseException {
    public ConnectionException(String message) {
        super(message);
    }
}

自定义异常对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 继承基类       │ Exception           │ Exception           │
│ 构造函数       │ __init__            │ Constructor         │
│ 字符串表示     │ __str__             │ toString()          │
│ 自定义属性     │ self.attr           │ private field       │
│ 检查异常       │ 无区分              │ extends Exception   │
│ 非检查异常     │ 无区分              │ extends RuntimeEx   │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 5. 上下文管理器与资源管理
# ============================================================================

print("\n=== 5. 上下文管理器 ===\n")

def context_managers():
    """
    Python的with语句（上下文管理器）
    自动管理资源的获取和释放
    """
    
    # 文件操作（自动关闭文件）
    # with open("file.txt", "w") as f:
    #     f.write("Hello, World!")
    # # 文件自动关闭
    
    print("with语句示例（文件会自动关闭）")
    
    # 自定义上下文管理器（类方式）
    class ManagedResource:
        """自定义资源管理器"""
        
        def __init__(self, name):
            self.name = name
        
        def __enter__(self):
            """进入上下文时调用"""
            print(f"获取资源: {self.name}")
            return self  # 返回给as后的变量
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """退出上下文时调用"""
            print(f"释放资源: {self.name}")
            if exc_type:
                print(f"处理异常: {exc_type.__name__}: {exc_val}")
            return False  # False表示不抑制异常
    
    # 使用自定义上下文管理器
    with ManagedResource("数据库连接") as resource:
        print(f"使用资源: {resource.name}")
    
    # 即使发生异常也会清理资源
    try:
        with ManagedResource("文件句柄") as resource:
            print(f"使用资源: {resource.name}")
            raise ValueError("模拟异常")
    except ValueError:
        print("异常已被捕获")
    
    # contextlib.contextmanager装饰器
    from contextlib import contextmanager
    
    @contextmanager
    def managed_resource(name):
        """使用生成器创建上下文管理器"""
        print(f"获取资源: {name}")
        try:
            yield name  # yield前是__enter__，后是__exit__
        finally:
            print(f"释放资源: {name}")
    
    with managed_resource("网络连接") as resource:
        print(f"使用资源: {resource}")

context_managers()

"""
Java对比:
import java.io.*;

public class ResourceManagement {
    // Java 7+ try-with-resources
    public static void tryWithResources() {
        // 自动关闭资源
        try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
            String line = reader.readLine();
            System.out.println(line);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // reader自动关闭
    }
    
    // 自定义可关闭资源
    static class ManagedResource implements AutoCloseable {
        private String name;
        
        public ManagedResource(String name) {
            this.name = name;
            System.out.println("获取资源: " + name);
        }
        
        public void use() {
            System.out.println("使用资源: " + name);
        }
        
        @Override
        public void close() {
            System.out.println("释放资源: " + name);
        }
    }
    
    public static void useCustomResource() {
        try (ManagedResource resource = new ManagedResource("数据库连接")) {
            resource.use();
        }  // 自动调用close()
    }
}

资源管理对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 资源管理       │ with语句            │ try-with-resources  │
│ 进入方法       │ __enter__           │ (构造函数)          │
│ 退出方法       │ __exit__            │ close()             │
│ 实现接口       │ 无需                │ AutoCloseable       │
│ 异常处理       │ __exit__参数        │ try-catch           │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 6. 异常处理最佳实践
# ============================================================================

print("\n=== 6. 异常处理最佳实践 ===\n")

def best_practices():
    """
    Python异常处理最佳实践
    """
    
    # 1. 具体异常优于通用异常
    print("1. 捕获具体异常:")
    try:
        value = int("abc")
    except ValueError as e:  # 好：捕获具体异常
        print(f"  值错误: {e}")
    # except Exception as e:  # 差：太宽泛
    
    # 2. EAFP vs LBYL
    print("\n2. EAFP (Easier to Ask Forgiveness than Permission):")
    data = {"name": "张三"}
    
    # EAFP风格（Python推荐）
    try:
        age = data["age"]
    except KeyError:
        age = 0
    print(f"  年龄: {age}")
    
    # LBYL风格（Look Before You Leap）
    if "city" in data:
        city = data["city"]
    else:
        city = "未知"
    print(f"  城市: {city}")
    
    # 3. 异常不应用于正常流程控制
    print("\n3. 不要用异常控制流程:")
    # 差的做法
    # for i in range(1000):
    #     try:
    #         if i == 100:
    #             raise StopIteration
    #     except StopIteration:
    #         break
    
    # 好的做法
    for i in range(100):
        if i == 100:
            break
    
    # 4. 使用finally清理资源（或使用with）
    print("\n4. 资源清理:")
    resource = None
    try:
        resource = "模拟资源"
        print(f"  使用资源: {resource}")
    finally:
        if resource:
            print(f"  清理资源: {resource}")
    
    # 5. 提供有用的错误信息
    print("\n5. 有用的错误信息:")
    def divide(a, b):
        if b == 0:
            raise ValueError(f"除数不能为零: {a} / {b}")
        return a / b
    
    try:
        divide(10, 0)
    except ValueError as e:
        print(f"  {e}")
    
    # 6. 日志记录异常
    print("\n6. 日志记录:")
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logging.exception("  除零错误发生")  # 自动记录堆栈

best_practices()

"""
最佳实践总结：

1. 捕获具体异常
   - 避免使用except Exception
   - 只捕获你能处理的异常

2. 使用EAFP风格
   - Python推荐"先做再说"
   - 而不是"三思而后行"

3. 不滥用异常
   - 异常用于异常情况
   - 不用于正常流程控制

4. 资源管理
   - 优先使用with语句
   - 或者使用finally清理

5. 提供清晰的错误信息
   - 包含足够的上下文
   - 便于调试

6. 日志记录
   - 使用logging模块
   - logging.exception()记录堆栈

7. 不要忽略异常
   - 避免空的except块
   - 至少记录日志

8. 异常链
   - 使用raise...from保留原始异常
   - 提供完整的错误上下文
"""

# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示异常处理差异"""
    print("=" * 70)
    print("Python vs Java: 异常处理深度对比")
    print("=" * 70)
    
    basic_exception_handling()
    exception_hierarchy()
    raise_exceptions()
    
    print("\n" + "=" * 70)
    print("学习要点")
    print("=" * 70)
    print("""
    对于Java程序员学习Python异常处理：
    
    1. 没有检查异常
       - Python不区分检查异常和非检查异常
       - 不需要throws声明
       - 更灵活但需要更多文档说明
    
    2. else子句
       - try-except-else-finally
       - else在无异常时执行
       - Java没有这个特性
    
    3. EAFP vs LBYL
       - Python推荐EAFP（先做再说）
       - Java倾向LBYL（先检查再做）
       - 这是哲学差异
    
    4. with语句
       - 自动资源管理
       - 类似Java的try-with-resources
       - 实现__enter__和__exit__
    
    5. 异常链
       - raise...from保留原始异常
       - 类似Java的构造函数传入cause
    
    6. 异常是对象
       - 可以添加自定义属性
       - 可以重写__str__方法
    
    常见陷阱：
    - 不要使用空的except块
    - 不要捕获BaseException
    - 不要用异常控制流程
    - 记得清理资源
    """)

if __name__ == "__main__":
    main()
