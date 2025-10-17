"""
Python vs Java: 面向对象编程深度对比
============================
面向Java程序员的Python OOP学习指南

作者: Python学习系列
目标读者: 熟悉Java OOP的开发者
学习重点: 掌握Python面向对象的独特特性
"""

# ============================================================================
# 1. 类的定义与构造
# ============================================================================

print("=== 1. 类的定义与构造 ===\n")

class PythonPerson:
    """
    Python类定义特点：
    - 使用class关键字
    - __init__方法是构造函数
    - self是实例的引用（类似Java的this）
    - 没有严格的访问控制修饰符
    """
    
    # 类变量（所有实例共享）
    species = "Homo sapiens"
    count = 0
    
    def __init__(self, name, age):
        """
        构造函数
        self: 实例引用（必须显式声明）
        """
        # 实例变量
        self.name = name
        self.age = age
        PythonPerson.count += 1
    
    def introduce(self):
        """实例方法"""
        return f"我是{self.name}，{self.age}岁"
    
    @classmethod
    def get_count(cls):
        """类方法 - 访问类变量"""
        return f"共创建了{cls.count}个Person对象"
    
    @staticmethod
    def is_adult(age):
        """静态方法 - 不访问实例或类"""
        return age >= 18
    
    def __str__(self):
        """字符串表示（类似Java的toString）"""
        return f"Person(name={self.name}, age={self.age})"
    
    def __repr__(self):
        """开发者友好的字符串表示"""
        return f"PythonPerson('{self.name}', {self.age})"

# 测试
person1 = PythonPerson("张三", 25)
person2 = PythonPerson("李四", 30)

print(person1.introduce())
print(PythonPerson.get_count())
print(f"是否成年: {PythonPerson.is_adult(25)}")
print(f"str: {str(person1)}")
print(f"repr: {repr(person1)}")

"""
Java对比:
public class JavaPerson {
    // 类变量（静态变量）
    public static String species = "Homo sapiens";
    private static int count = 0;
    
    // 实例变量
    private String name;
    private int age;
    
    // 构造函数
    public JavaPerson(String name, int age) {
        this.name = name;  // this是隐式的
        this.age = age;
        JavaPerson.count++;
    }
    
    // 实例方法
    public String introduce() {
        return "我是" + this.name + "，" + this.age + "岁";
    }
    
    // 静态方法
    public static String getCount() {
        return "共创建了" + count + "个Person对象";
    }
    
    public static boolean isAdult(int age) {
        return age >= 18;
    }
    
    // toString方法
    @Override
    public String toString() {
        return "Person(name=" + name + ", age=" + age + ")";
    }
}

关键差异：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 构造函数       │ __init__(self)      │ ClassName()         │
│ 实例引用       │ self (显式)         │ this (隐式)         │
│ 类方法         │ @classmethod        │ static方法          │
│ 静态方法       │ @staticmethod       │ static方法          │
│ 字符串表示     │ __str__, __repr__   │ toString()          │
│ 访问控制       │ 约定（_前缀）       │ public/private/etc  │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 2. 访问控制与封装
# ============================================================================

print("\n=== 2. 访问控制与封装 ===\n")

class PythonEncapsulation:
    """
    Python的访问控制基于命名约定：
    - 无前缀: 公有
    - 单下划线_: 受保护（约定，不强制）
    - 双下划线__: 私有（名称改写）
    """
    
    def __init__(self):
        self.public_attr = "公有属性"
        self._protected_attr = "受保护属性"
        self.__private_attr = "私有属性"
    
    def public_method(self):
        """公有方法"""
        return "这是公有方法"
    
    def _protected_method(self):
        """受保护方法（约定）"""
        return "这是受保护方法"
    
    def __private_method(self):
        """私有方法（名称改写）"""
        return "这是私有方法"
    
    def access_private(self):
        """在类内部可以访问私有成员"""
        return self.__private_method()

# 测试访问控制
obj = PythonEncapsulation()

# 公有成员可以直接访问
print(f"公有属性: {obj.public_attr}")
print(f"公有方法: {obj.public_method()}")

# 受保护成员（约定不要访问，但技术上可以）
print(f"受保护属性: {obj._protected_attr}")
print(f"受保护方法: {obj._protected_method()}")

# 私有成员（名称改写，不能直接访问）
# print(obj.__private_attr)  # AttributeError
# 但可以通过名称改写访问（不推荐）
# print(f"私有属性（通过名称改写）: {obj._PythonEncapsulation__private_attr}")

# 通过公有方法访问私有成员
print(f"访问私有方法: {obj.access_private()}")

"""
Java对比:
public class JavaEncapsulation {
    // 严格的访问控制修饰符
    public String publicAttr = "公有属性";      // 任何地方可访问
    protected String protectedAttr = "受保护";   // 子类和同包可访问
    String defaultAttr = "包私有";               // 同包可访问
    private String privateAttr = "私有属性";     // 仅类内部可访问
    
    public void publicMethod() {
        System.out.println("公有方法");
    }
    
    protected void protectedMethod() {
        System.out.println("受保护方法");
    }
    
    void defaultMethod() {
        System.out.println("包私有方法");
    }
    
    private void privateMethod() {
        System.out.println("私有方法");
    }
    
    public void accessPrivate() {
        // 在类内部可以访问私有成员
        privateMethod();
    }
}

// 测试访问控制
JavaEncapsulation obj = new JavaEncapsulation();
System.out.println(obj.publicAttr);    // OK
System.out.println(obj.protectedAttr); // OK（同包或子类）
// System.out.println(obj.privateAttr); // 编译错误！

访问控制对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   级别         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 公有           │ name                │ public              │
│ 受保护         │ _name (约定)        │ protected           │
│ 包私有         │ 无                  │ (默认)              │
│ 私有           │ __name (名称改写)   │ private             │
│ 强制性         │ 弱（约定为主）      │ 强（编译时检查）    │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 3. 属性（Properties）
# ============================================================================

print("\n=== 3. 属性（Properties）===\n")

class PythonProperties:
    """
    Python的@property装饰器
    提供类似Java的getter/setter，但语法更简洁
    """
    
    def __init__(self, temperature):
        self._temperature = temperature  # 使用下划线前缀
    
    @property
    def temperature(self):
        """获取温度（getter）"""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        """设置温度（setter）"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._temperature = value
    
    @temperature.deleter
    def temperature(self):
        """删除温度（deleter）"""
        print("删除温度属性")
        del self._temperature
    
    # 只读属性（只有getter）
    @property
    def fahrenheit(self):
        """摄氏度转华氏度"""
        return self._temperature * 9/5 + 32

# 测试属性
obj = PythonProperties(25)
print(f"温度: {obj.temperature}°C")  # 调用getter
print(f"华氏度: {obj.fahrenheit}°F")  # 只读属性

obj.temperature = 30  # 调用setter
print(f"新温度: {obj.temperature}°C")

try:
    obj.temperature = -300  # 触发验证
except ValueError as e:
    print(f"错误: {e}")

"""
Java对比:
public class JavaProperties {
    private double temperature;
    
    // 构造函数
    public JavaProperties(double temperature) {
        this.temperature = temperature;
    }
    
    // Getter方法
    public double getTemperature() {
        return this.temperature;
    }
    
    // Setter方法
    public void setTemperature(double value) {
        if (value < -273.15) {
            throw new IllegalArgumentException("温度不能低于绝对零度");
        }
        this.temperature = value;
    }
    
    // 只读属性（只有getter）
    public double getFahrenheit() {
        return this.temperature * 9/5 + 32;
    }
}

// 使用
JavaProperties obj = new JavaProperties(25);
System.out.println("温度: " + obj.getTemperature() + "°C");
System.out.println("华氏度: " + obj.getFahrenheit() + "°F");

obj.setTemperature(30);  // 必须调用setter方法

属性对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ Getter         │ @property           │ getXxx()方法        │
│ Setter         │ @xxx.setter         │ setXxx()方法        │
│ 访问语法       │ obj.attr            │ obj.getAttr()       │
│ 设置语法       │ obj.attr = value    │ obj.setAttr(value)  │
│ 只读属性       │ 只定义getter        │ 只定义getter        │
│ 语法简洁性     │ 非常简洁            │ 较冗长              │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 4. 继承
# ============================================================================

print("\n=== 4. 继承 ===\n")

class Animal:
    """基类"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self) -> str:
        """基类方法"""
        return "某种声音"
    
    def info(self):
        return f"{self.name} is an animal"

class Dog(Animal):
    """单继承"""
    
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造函数
        self.breed = breed
    
    def speak(self):
        """方法重写"""
        return "汪汪汪"
    
    def info(self):
        """调用父类方法"""
        parent_info = super().info()
        return f"{parent_info}, breed: {self.breed}"

# 多重继承
class Swimmer:
    def swim(self):
        return "游泳中"

class Flyer:
    def fly(self):
        return "飞行中"

class Duck(Animal, Swimmer, Flyer):
    """多重继承（Python支持，Java不支持）"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def speak(self):
        return "嘎嘎嘎"

# 测试继承
dog = Dog("旺财", "金毛")
print(dog.speak())
print(dog.info())

duck = Duck("唐老鸭")
print(duck.speak())
print(duck.swim())
print(duck.fly())

# MRO（方法解析顺序）
print(f"\nDuck的MRO: {[cls.__name__ for cls in Duck.__mro__]}")

"""
Java对比:
// 基类
public class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public String speak() {
        return "某种声音";
    }
    
    public String info() {
        return this.name + " is an animal";
    }
}

// 单继承
public class Dog extends Animal {
    private String breed;
    
    public Dog(String name, String breed) {
        super(name);  // 调用父类构造函数
        this.breed = breed;
    }
    
    @Override
    public String speak() {
        return "汪汪汪";
    }
    
    @Override
    public String info() {
        String parentInfo = super.info();
        return parentInfo + ", breed: " + this.breed;
    }
}

// Java不支持多重继承，使用接口
interface Swimmer {
    String swim();
}

interface Flyer {
    String fly();
}

public class Duck extends Animal implements Swimmer, Flyer {
    public Duck(String name) {
        super(name);
    }
    
    @Override
    public String speak() {
        return "嘎嘎嘎";
    }
    
    @Override
    public String swim() {
        return "游泳中";
    }
    
    @Override
    public String fly() {
        return "飞行中";
    }
}

继承对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 单继承         │ class Dog(Animal)   │ extends Animal      │
│ 多重继承       │ 支持                │ 不支持              │
│ 接口           │ 非正式接口(鸭子类型)│ interface关键字     │
│ 调用父类       │ super()             │ super               │
│ MRO            │ C3线性化算法        │ 单继承链            │
│ 抽象类         │ ABC模块             │ abstract关键字      │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 5. 抽象类与接口
# ============================================================================

print("\n=== 5. 抽象类与接口 ===\n")

from abc import ABC, abstractmethod

class AbstractShape(ABC):
    """
    抽象基类（使用ABC模块）
    """
    
    def __init__(self, color):
        self.color = color
    
    @abstractmethod
    def area(self) -> float:
        """抽象方法 - 子类必须实现"""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """抽象方法"""
        pass
    
    def describe(self):
        """具体方法 - 子类可以继承"""
        return f"这是一个{self.color}的形状"

class Circle(AbstractShape):
    """实现抽象类"""
    
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius
    
    def area(self):
        """实现抽象方法"""
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        """实现抽象方法"""
        return 2 * 3.14159 * self.radius

# 测试抽象类
# shape = AbstractShape("红色")  # TypeError: 不能实例化抽象类
circle = Circle("红色", 5)
print(f"面积: {circle.area()}")
print(f"周长: {circle.perimeter()}")
print(circle.describe())

"""
Java对比:
// 抽象类
public abstract class AbstractShape {
    protected String color;
    
    public AbstractShape(String color) {
        this.color = color;
    }
    
    // 抽象方法
    public abstract double area();
    public abstract double perimeter();
    
    // 具体方法
    public String describe() {
        return "这是一个" + this.color + "的形状";
    }
}

// 实现抽象类
public class Circle extends AbstractShape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
    
    @Override
    public double perimeter() {
        return 2 * Math.PI * radius;
    }
}

// 接口
public interface Drawable {
    void draw();
    
    // Java 8+ 可以有默认实现
    default void display() {
        System.out.println("显示图形");
    }
}

抽象类对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 抽象类         │ ABC基类             │ abstract class      │
│ 抽象方法       │ @abstractmethod     │ abstract method     │
│ 接口           │ 非正式(鸭子类型)    │ interface           │
│ 多接口         │ 多重继承            │ implements多个      │
│ 实例化         │ TypeError           │ 编译错误            │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 6. 魔法方法（Magic Methods / Dunder Methods）
# ============================================================================

print("\n=== 6. 魔法方法 ===\n")

class Vector:
    """
    Python魔法方法示例
    魔法方法以双下划线开头和结尾
    """
    
    def __init__(self, x, y):
        """构造函数"""
        self.x = x
        self.y = y
    
    def __str__(self):
        """str()函数调用"""
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        """repr()函数调用"""
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        """+ 运算符重载"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """- 运算符重载"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """* 运算符重载"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        """== 运算符重载"""
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        """len()函数调用"""
        return 2
    
    def __getitem__(self, index):
        """索引访问 v[0]"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("索引超出范围")
    
    def __call__(self):
        """使对象可调用"""
        return f"向量的模: {(self.x**2 + self.y**2)**0.5}"

# 测试魔法方法
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(f"v1: {v1}")
print(f"v1 + v2: {v1 + v2}")
print(f"v1 - v2: {v1 - v2}")
print(f"v1 * 3: {v1 * 3}")
print(f"v1 == v2: {v1 == v2}")
print(f"len(v1): {len(v1)}")
print(f"v1[0]: {v1[0]}, v1[1]: {v1[1]}")
print(f"v1(): {v1()}")

"""
Java对比:
public class Vector {
    private double x, y;
    
    public Vector(double x, double y) {
        this.x = x;
        this.y = y;
    }
    
    @Override
    public String toString() {
        return "Vector(" + x + ", " + y + ")";
    }
    
    // Java不支持运算符重载，需要定义方法
    public Vector add(Vector other) {
        return new Vector(this.x + other.x, this.y + other.y);
    }
    
    public Vector subtract(Vector other) {
        return new Vector(this.x - other.x, this.y - other.y);
    }
    
    public Vector multiply(double scalar) {
        return new Vector(this.x * scalar, this.y * scalar);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Vector)) return false;
        Vector other = (Vector) obj;
        return this.x == other.x && this.y == other.y;
    }
    
    // Java没有__len__, __getitem__, __call__等价物
}

// 使用
Vector v1 = new Vector(1, 2);
Vector v2 = new Vector(3, 4);
Vector v3 = v1.add(v2);  // 必须调用方法，不能用+

常用魔法方法：
┌────────────────┬─────────────────────┬─────────────────────┐
│   Python       │   用途              │   Java等价          │
├────────────────┼─────────────────────┼─────────────────────┤
│ __init__       │ 构造函数            │ ClassName()         │
│ __str__        │ 字符串表示          │ toString()          │
│ __repr__       │ 开发者字符串        │ toString()          │
│ __add__        │ + 运算符            │ 不支持              │
│ __eq__         │ == 运算符           │ equals()            │
│ __len__        │ len()函数           │ 不支持              │
│ __getitem__    │ 索引访问            │ 不支持              │
│ __call__       │ 对象可调用          │ 不支持              │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示面向对象编程差异"""
    print("=" * 70)
    print("Python vs Java: 面向对象编程深度对比")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("学习建议")
    print("=" * 70)
    print("""
    对于Java程序员学习Python OOP：
    
    1. 理解self
       - self必须显式声明（不像Java的this）
       - 方法的第一个参数总是self
       - 命名可以是其他名称，但约定使用self
    
    2. 访问控制约定
       - Python没有严格的访问控制
       - 使用_前缀表示"请不要访问"
       - 使用__前缀触发名称改写
       - 相信开发者的自觉性
    
    3. 使用@property
       - 比Java的getter/setter更简洁
       - 保持属性访问的语法
       - 可以添加验证逻辑
    
    4. 多重继承
       - Python支持多重继承
       - 理解MRO（方法解析顺序）
       - 使用super()正确调用父类方法
    
    5. 掌握魔法方法
       - 运算符重载：__add__, __sub__等
       - 容器协议：__len__, __getitem__等
       - 上下文管理：__enter__, __exit__
       - 这是Python OOP的强大特性
    
    6. 抽象类和接口
       - 使用ABC模块定义抽象类
       - Python没有正式的接口概念
       - 鸭子类型替代接口
    
    7. 组合优于继承
       - Python和Java都推荐这个原则
       - 使用组合构建灵活的对象
    
    常见陷阱：
    - 忘记在方法中写self参数
    - 类变量vs实例变量的混淆
    - 可变默认参数的陷阱
    - 多重继承的复杂性
    """)

if __name__ == "__main__":
    main()
