# 设计模式示例 - 常见的设计模式实现

# 1. 单例模式 (Singleton Pattern)
class Singleton:
    """单例模式 - 确保一个类只有一个实例"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.value = 0
            self._initialized = True
    
    def increment(self):
        self.value += 1
        return self.value


# 2. 工厂模式 (Factory Pattern)
from abc import ABC, abstractmethod

class Animal(ABC):
    """抽象动物类"""
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "汪汪!"

class Cat(Animal):
    def speak(self):
        return "喵喵!"

class Bird(Animal):
    def speak(self):
        return "叽叽喳喳!"

class AnimalFactory:
    """动物工厂"""
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == 'dog':
            return Dog()
        elif animal_type.lower() == 'cat':
            return Cat()
        elif animal_type.lower() == 'bird':
            return Bird()
        else:
            raise ValueError(f"不支持的动物类型: {animal_type}")


# 3. 观察者模式 (Observer Pattern)
class Subject:
    """被观察者"""
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        """添加观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self):
        """通知所有观察者"""
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        """设置状态并通知观察者"""
        self._state = state
        self.notify()

class Observer:
    """观察者"""
    def __init__(self, name):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} 收到状态更新: {state}")


# 4. 装饰器模式 (Decorator Pattern)
class Coffee:
    """基础咖啡类"""
    def cost(self):
        return 10
    
    def description(self):
        return "基础咖啡"

class CoffeeDecorator:
    """咖啡装饰器基类"""
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()

class MilkDecorator(CoffeeDecorator):
    """牛奶装饰器"""
    def cost(self):
        return self._coffee.cost() + 3
    
    def description(self):
        return self._coffee.description() + " + 牛奶"

class SugarDecorator(CoffeeDecorator):
    """糖装饰器"""
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + " + 糖"


# 5. 策略模式 (Strategy Pattern)
class PaymentStrategy(ABC):
    """支付策略抽象类"""
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    """信用卡支付"""
    def __init__(self, card_number):
        self.card_number = card_number
    
    def pay(self, amount):
        return f"使用信用卡 {self.card_number} 支付 {amount} 元"

class AlipayPayment(PaymentStrategy):
    """支付宝支付"""
    def __init__(self, account):
        self.account = account
    
    def pay(self, amount):
        return f"使用支付宝账户 {self.account} 支付 {amount} 元"

class WechatPayment(PaymentStrategy):
    """微信支付"""
    def __init__(self, phone):
        self.phone = phone
    
    def pay(self, amount):
        return f"使用微信 {self.phone} 支付 {amount} 元"

class PaymentContext:
    """支付上下文"""
    def __init__(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: PaymentStrategy):
        self._strategy = strategy
    
    def execute_payment(self, amount):
        return self._strategy.pay(amount)


# 6. 建造者模式 (Builder Pattern)
class Computer:
    """电脑类"""
    def __init__(self):
        self.cpu = None
        self.memory = None
        self.storage = None
        self.gpu = None
    
    def __str__(self):
        return f"电脑配置 - CPU: {self.cpu}, 内存: {self.memory}, 存储: {self.storage}, 显卡: {self.gpu}"

class ComputerBuilder:
    """电脑建造者"""
    def __init__(self):
        self.computer = Computer()
    
    def set_cpu(self, cpu):
        self.computer.cpu = cpu
        return self
    
    def set_memory(self, memory):
        self.computer.memory = memory
        return self
    
    def set_storage(self, storage):
        self.computer.storage = storage
        return self
    
    def set_gpu(self, gpu):
        self.computer.gpu = gpu
        return self
    
    def build(self):
        return self.computer


# 演示代码
def demonstrate_patterns():
    print("=== 设计模式演示 ===\n")
    
    # 1. 单例模式演示
    print("1. 单例模式:")
    s1 = Singleton()
    s2 = Singleton()
    print(f"s1 和 s2 是同一个实例: {s1 is s2}")
    print(f"s1.increment(): {s1.increment()}")
    print(f"s2.value: {s2.value}")
    print()
    
    # 2. 工厂模式演示
    print("2. 工厂模式:")
    dog = AnimalFactory.create_animal('dog')
    cat = AnimalFactory.create_animal('cat')
    print(f"狗: {dog.speak()}")
    print(f"猫: {cat.speak()}")
    print()
    
    # 3. 观察者模式演示
    print("3. 观察者模式:")
    subject = Subject()
    observer1 = Observer("观察者1")
    observer2 = Observer("观察者2")
    
    subject.attach(observer1)
    subject.attach(observer2)
    subject.set_state("新状态A")
    print()
    
    # 4. 装饰器模式演示
    print("4. 装饰器模式:")
    coffee = Coffee()
    coffee_with_milk = MilkDecorator(coffee)
    coffee_with_milk_and_sugar = SugarDecorator(coffee_with_milk)
    
    print(f"{coffee_with_milk_and_sugar.description()}: {coffee_with_milk_and_sugar.cost()} 元")
    print()
    
    # 5. 策略模式演示
    print("5. 策略模式:")
    context = PaymentContext(CreditCardPayment("1234-5678-9012-3456"))
    print(context.execute_payment(100))
    
    context.set_strategy(AlipayPayment("user@example.com"))
    print(context.execute_payment(200))
    print()
    
    # 6. 建造者模式演示
    print("6. 建造者模式:")
    computer = (ComputerBuilder()
                .set_cpu("Intel i7")
                .set_memory("16GB")
                .set_storage("512GB SSD")
                .set_gpu("RTX 3080")
                .build())
    print(computer)


if __name__ == "__main__":
    demonstrate_patterns()