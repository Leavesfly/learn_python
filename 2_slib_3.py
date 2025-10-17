# 数据结构和算法相关
import collections
import heapq
import itertools
import functools
import re
import random
import math
import statistics

def collections_demo():
    """collections模块示例"""
    print("=== collections 数据结构 ===")
    
    # Counter - 计数器
    text = "hello world python programming"
    counter = collections.Counter(text.replace(' ', ''))
    print(f"字符计数: {counter.most_common(5)}")
    
    # defaultdict - 默认字典
    dd = collections.defaultdict(list)
    words = ["apple", "banana", "cherry", "avocado", "blueberry"]
    for word in words:
        dd[word[0]].append(word)
    print(f"按首字母分组: {dict(dd)}")
    
    # deque - 双端队列
    dq = collections.deque([1, 2, 3])
    dq.appendleft(0)
    dq.append(4)
    print(f"双端队列: {list(dq)}")
    
    # namedtuple - 命名元组
    Person = collections.namedtuple('Person', ['name', 'age', 'city'])
    person = Person('张三', 25, '北京')
    print(f"命名元组: {person}, 姓名: {person.name}")

def heapq_demo():
    """堆队列示例"""
    print("\n=== heapq 堆队列 ===")
    
    # 创建堆
    numbers = [5, 2, 8, 1, 9, 3]
    heapq.heapify(numbers)
    print(f"堆化后: {numbers}")
    
    # 获取最小值
    smallest = heapq.heappop(numbers)
    print(f"最小值: {smallest}, 剩余: {numbers}")
    
    # 添加元素
    heapq.heappush(numbers, 0)
    print(f"添加0后: {numbers}")
    
    # 获取最大的3个元素
    data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    largest = heapq.nlargest(3, data)
    smallest = heapq.nsmallest(3, data)
    print(f"最大的3个: {largest}, 最小的3个: {smallest}")

def itertools_demo():
    """itertools迭代工具示例"""
    print("\n=== itertools 迭代工具 ===")
    
    # 无限迭代器
    counter = itertools.count(1, 2)  # 从1开始，步长为2
    print(f"前5个奇数: {list(itertools.islice(counter, 5))}")
    
    # 排列组合
    items = ['A', 'B', 'C']
    permutations = list(itertools.permutations(items, 2))
    combinations = list(itertools.combinations(items, 2))
    print(f"排列: {permutations}")
    print(f"组合: {combinations}")
    
    # 分组
    data = [1, 1, 2, 2, 2, 3, 3]
    groups = [(k, list(g)) for k, g in itertools.groupby(data)]
    print(f"分组: {groups}")
    
    # 链式连接
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    chained = list(itertools.chain(list1, list2))
    print(f"链式连接: {chained}")

def functools_demo():
    """functools函数工具示例"""
    print("\n=== functools 函数工具 ===")
    
    # reduce
    numbers = [1, 2, 3, 4, 5]
    sum_result = functools.reduce(lambda x, y: x + y, numbers)
    print(f"累加结果: {sum_result}")
    
    # partial - 偏函数
    def multiply(x, y, z):
        return x * y * z
    
    double = functools.partial(multiply, 2)
    result = double(3, 4)  # 相当于 multiply(2, 3, 4)
    print(f"偏函数结果: {result}")
    
    # lru_cache - 缓存装饰器
    @functools.lru_cache(maxsize=None)
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    print(f"斐波那契数列第10项: {fibonacci(10)}")
    print(f"缓存信息: {fibonacci.cache_info()}")

def regex_demo():
    """正则表达式示例"""
    print("\n=== re 正则表达式 ===")
    
    text = "联系方式：手机 138-1234-5678，邮箱 zhangsan@example.com，电话 010-12345678"
    
    # 查找手机号
    phone_pattern = r'1[3-9]\d-\d{4}-\d{4}'
    phones = re.findall(phone_pattern, text)
    print(f"手机号: {phones}")
    
    # 查找邮箱
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    print(f"邮箱: {emails}")
    
    # 替换
    masked_text = re.sub(r'1[3-9]\d-\d{4}-\d{4}', '***-****-****', text)
    print(f"手机号脱敏: {masked_text}")

def random_math_stats_demo():
    """随机数、数学和统计示例"""
    print("\n=== random, math, statistics ===")
    
    # random - 随机数
    print(f"随机整数: {random.randint(1, 100)}")
    print(f"随机浮点数: {random.random():.3f}")
    
    choices = ['苹果', '香蕉', '橙子', '葡萄']
    print(f"随机选择: {random.choice(choices)}")
    
    sample_data = random.sample(range(1, 101), 5)
    print(f"随机采样: {sample_data}")
    
    # math - 数学函数
    print(f"平方根: {math.sqrt(16)}")
    print(f"对数: {math.log10(100)}")
    print(f"三角函数: sin(π/2) = {math.sin(math.pi/2):.3f}")
    
    # statistics - 统计函数
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"平均值: {statistics.mean(data)}")
    print(f"中位数: {statistics.median(data)}")
    print(f"标准差: {statistics.stdev(data):.3f}")

if __name__ == "__main__":
    collections_demo()
    heapq_demo()
    itertools_demo()
    functools_demo()
    regex_demo()
    random_math_stats_demo()