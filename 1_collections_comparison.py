"""
Python vs Java: 集合与数据结构深度对比
============================
面向Java程序员的Python集合学习指南

作者: Python学习系列
目标读者: 熟悉Java集合框架的开发者
学习重点: 掌握Python内置数据结构及其操作
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter, deque, OrderedDict, namedtuple

# ============================================================================
# 1. 列表 (List) vs ArrayList
# ============================================================================

print("=== 1. 列表 (List) ===\n")

def python_lists():
    """
    Python列表特点：
    - 动态大小
    - 可以包含不同类型元素
    - 支持丰富的操作
    """
    
    # 创建列表
    numbers = [1, 2, 3, 4, 5]
    mixed = [1, "text", 3.14, True]  # 可以混合类型
    empty = []
    
    # 列表操作
    print("基本操作:")
    numbers.append(6)  # 添加元素
    print(f"  添加后: {numbers}")
    
    numbers.insert(0, 0)  # 在指定位置插入
    print(f"  插入后: {numbers}")
    
    numbers.remove(3)  # 删除指定值
    print(f"  删除3后: {numbers}")
    
    popped = numbers.pop()  # 删除并返回最后一个元素
    print(f"  弹出: {popped}, 剩余: {numbers}")
    
    # 列表访问
    print("\n访问操作:")
    print(f"  第一个: {numbers[0]}")
    print(f"  最后一个: {numbers[-1]}")
    print(f"  切片[1:3]: {numbers[1:3]}")
    print(f"  反转: {numbers[::-1]}")
    
    # 列表方法
    print("\n列表方法:")
    numbers_copy = [5, 2, 8, 1, 9]
    numbers_copy.sort()  # 原地排序
    print(f"  排序: {numbers_copy}")
    
    numbers_copy.reverse()  # 原地反转
    print(f"  反转: {numbers_copy}")
    
    print(f"  计数1: {[1, 2, 1, 3, 1].count(1)}")
    print(f"  索引: {[1, 2, 3].index(2)}")
    
    # 列表推导式（强大特性）
    print("\n列表推导式:")
    squares = [x**2 for x in range(5)]
    print(f"  平方: {squares}")
    
    evens = [x for x in range(10) if x % 2 == 0]
    print(f"  偶数: {evens}")
    
    matrix = [[i*j for j in range(3)] for i in range(3)]
    print(f"  矩阵: {matrix}")

python_lists()

"""
Java对比:
import java.util.*;

public class JavaLists {
    public static void javaLists() {
        // 创建列表
        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        List<Object> mixed = new ArrayList<>();  // 可以用Object但不推荐
        List<Integer> empty = new ArrayList<>();
        
        // 列表操作
        numbers.add(6);  // 添加元素
        numbers.add(0, 0);  // 在指定位置插入
        numbers.remove(Integer.valueOf(3));  // 删除指定值
        Integer popped = numbers.remove(numbers.size() - 1);  // 删除最后一个
        
        // 访问操作
        int first = numbers.get(0);
        int last = numbers.get(numbers.size() - 1);
        List<Integer> subList = numbers.subList(1, 3);  // 不包括3
        
        // 列表方法
        List<Integer> numbersCopy = new ArrayList<>(Arrays.asList(5, 2, 8, 1, 9));
        Collections.sort(numbersCopy);  // 排序
        Collections.reverse(numbersCopy);  // 反转
        
        int count = Collections.frequency(Arrays.asList(1, 2, 1, 3, 1), 1);
        int index = Arrays.asList(1, 2, 3).indexOf(2);
        
        // Stream API（Java 8+，类似列表推导）
        List<Integer> squares = IntStream.range(0, 5)
            .map(x -> x * x)
            .boxed()
            .collect(Collectors.toList());
        
        List<Integer> evens = IntStream.range(0, 10)
            .filter(x -> x % 2 == 0)
            .boxed()
            .collect(Collectors.toList());
    }
}

列表对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 创建           │ [1, 2, 3]           │ new ArrayList<>()   │
│ 添加           │ list.append(x)      │ list.add(x)         │
│ 插入           │ list.insert(i, x)   │ list.add(i, x)      │
│ 删除           │ list.remove(x)      │ list.remove(x)      │
│ 访问           │ list[i]             │ list.get(i)         │
│ 切片           │ list[1:3]           │ list.subList(1,3)   │
│ 排序           │ list.sort()         │ Collections.sort()  │
│ 推导式         │ [x for x in list]   │ Stream API          │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 2. 元组 (Tuple) vs 不可变
# ============================================================================

print("\n=== 2. 元组 (Tuple) ===\n")

def python_tuples():
    """
    Python元组特点：
    - 不可变序列
    - 可以作为字典的键
    - 常用于函数返回多个值
    """
    
    # 创建元组
    coordinates = (10, 20)
    single = (1,)  # 单元素元组需要逗号
    empty = ()
    
    # 元组解包
    x, y = coordinates
    print(f"坐标解包: x={x}, y={y}")
    
    # 多值返回
    def get_user_info():
        return "张三", 25, "北京"  # 自动打包为元组
    
    name, age, city = get_user_info()  # 解包
    print(f"用户信息: {name}, {age}, {city}")
    
    # 命名元组
    from collections import namedtuple
    
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(10, 20)
    print(f"命名元组: {p}, x={p.x}, y={p.y}")
    
    # 元组作为字典键
    location_data = {
        (0, 0): "原点",
        (1, 0): "东",
        (0, 1): "北",
    }
    print(f"位置: {location_data[(0, 0)]}")

python_tuples()

"""
Java对比:
public class JavaTuples {
    // Java没有内置元组，需要自定义类
    static class Pair<T, U> {
        public final T first;
        public final U second;
        
        public Pair(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }
    
    static class Triple<T, U, V> {
        public final T first;
        public final U second;
        public final V third;
        
        public Triple(T first, U second, V third) {
            this.first = first;
            this.second = second;
            this.third = third;
        }
    }
    
    // 多值返回
    public static Triple<String, Integer, String> getUserInfo() {
        return new Triple<>("张三", 25, "北京");
    }
    
    public static void javaTuples() {
        Pair<Integer, Integer> coordinates = new Pair<>(10, 20);
        int x = coordinates.first;
        int y = coordinates.second;
        
        Triple<String, Integer, String> user = getUserInfo();
        String name = user.first;
        int age = user.second;
        String city = user.third;
        
        // 使用不可变对象作为Map键
        Map<Pair<Integer, Integer>, String> locationData = new HashMap<>();
        locationData.put(new Pair<>(0, 0), "原点");
    }
}

元组对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   特性         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 内置支持       │ 是                  │ 否（需自定义类）    │
│ 创建           │ (1, 2)              │ new Pair<>(1, 2)    │
│ 访问           │ tuple[0]            │ pair.first          │
│ 解包           │ a, b = tuple        │ 不支持              │
│ 命名元组       │ namedtuple          │ 自定义类            │
│ 不可变性       │ 是                  │ 使用final字段       │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 3. 字典 (Dict) vs HashMap
# ============================================================================

print("\n=== 3. 字典 (Dict) ===\n")

def python_dicts():
    """
    Python字典特点：
    - 键值对存储
    - 键必须是不可变类型
    - Python 3.7+保持插入顺序
    """
    
    # 创建字典
    user = {"name": "张三", "age": 25, "city": "北京"}
    empty = {}
    
    # 字典操作
    print("基本操作:")
    print(f"  获取name: {user['name']}")
    print(f"  get方法: {user.get('email', '未设置')}")  # 带默认值
    
    user["email"] = "test@example.com"  # 添加/修改
    print(f"  添加email: {user}")
    
    del user["city"]  # 删除
    print(f"  删除city: {user}")
    
    # 字典方法
    print("\n字典方法:")
    print(f"  keys: {list(user.keys())}")
    print(f"  values: {list(user.values())}")
    print(f"  items: {list(user.items())}")
    
    # 遍历字典
    print("\n遍历:")
    for key, value in user.items():
        print(f"  {key}: {value}")
    
    # 字典推导式
    print("\n字典推导式:")
    squares = {x: x**2 for x in range(5)}
    print(f"  平方字典: {squares}")
    
    # defaultdict
    from collections import defaultdict
    
    word_count = defaultdict(int)  # 默认值为0
    for word in ["apple", "banana", "apple"]:
        word_count[word] += 1
    print(f"\n  defaultdict: {dict(word_count)}")
    
    # Counter
    from collections import Counter
    
    counter = Counter(["apple", "banana", "apple", "cherry", "banana", "apple"])
    print(f"  Counter: {counter}")
    print(f"  most_common: {counter.most_common(2)}")

python_dicts()

"""
Java对比:
import java.util.*;

public class JavaMaps {
    public static void javaMaps() {
        // 创建Map
        Map<String, Object> user = new HashMap<>();
        user.put("name", "张三");
        user.put("age", 25);
        user.put("city", "北京");
        
        // Map操作
        String name = (String) user.get("name");
        String email = (String) user.getOrDefault("email", "未设置");
        
        user.put("email", "test@example.com");  // 添加/修改
        user.remove("city");  // 删除
        
        // Map方法
        Set<String> keys = user.keySet();
        Collection<Object> values = user.values();
        Set<Map.Entry<String, Object>> entries = user.entrySet();
        
        // 遍历
        for (Map.Entry<String, Object> entry : user.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // Stream创建Map（Java 8+）
        Map<Integer, Integer> squares = IntStream.range(0, 5)
            .boxed()
            .collect(Collectors.toMap(x -> x, x -> x * x));
        
        // LinkedHashMap（保持插入顺序）
        Map<String, Integer> ordered = new LinkedHashMap<>();
        
        // TreeMap（排序）
        Map<String, Integer> sorted = new TreeMap<>();
    }
}

字典对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 创建           │ {"k": "v"}          │ new HashMap<>()     │
│ 访问           │ dict[key]           │ map.get(key)        │
│ 默认值         │ dict.get(k, default)│ getOrDefault()      │
│ 添加           │ dict[key] = value   │ map.put(key, value) │
│ 删除           │ del dict[key]       │ map.remove(key)     │
│ 遍历           │ for k,v in items()  │ for Entry           │
│ 顺序           │ 保持插入顺序(3.7+)  │ LinkedHashMap       │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 4. 集合 (Set) vs HashSet
# ============================================================================

print("\n=== 4. 集合 (Set) ===\n")

def python_sets():
    """
    Python集合特点：
    - 无序、唯一元素
    - 支持数学集合操作
    """
    
    # 创建集合
    numbers = {1, 2, 3, 4, 5}
    duplicates = {1, 2, 2, 3, 3, 3}  # 自动去重
    print(f"去重: {duplicates}")
    
    empty = set()  # 注意：{}是空字典，不是空集合
    
    # 集合操作
    print("\n基本操作:")
    numbers.add(6)
    print(f"  添加: {numbers}")
    
    numbers.remove(3)  # 不存在会报错
    print(f"  删除: {numbers}")
    
    numbers.discard(10)  # 不存在不报错
    print(f"  discard: {numbers}")
    
    # 集合运算
    print("\n集合运算:")
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    
    print(f"  并集: {a | b}")
    print(f"  交集: {a & b}")
    print(f"  差集: {a - b}")
    print(f"  对称差: {a ^ b}")
    
    print(f"  子集: {set([1, 2]) <= a}")
    print(f"  超集: {a >= set([1, 2])}")
    
    # 集合推导式
    print("\n集合推导式:")
    squares = {x**2 for x in range(10)}
    print(f"  平方集合: {squares}")
    
    # frozenset（不可变集合）
    frozen = frozenset([1, 2, 3])
    # frozen.add(4)  # AttributeError: 不可变
    print(f"  frozenset: {frozen}")

python_sets()

"""
Java对比:
import java.util.*;

public class JavaSets {
    public static void javaSets() {
        // 创建Set
        Set<Integer> numbers = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
        Set<Integer> duplicates = new HashSet<>(Arrays.asList(1, 2, 2, 3, 3, 3));
        Set<Integer> empty = new HashSet<>();
        
        // Set操作
        numbers.add(6);
        numbers.remove(3);
        // Java的remove不存在元素时返回false，不报错
        
        // 集合运算（需要手动实现）
        Set<Integer> a = new HashSet<>(Arrays.asList(1, 2, 3, 4));
        Set<Integer> b = new HashSet<>(Arrays.asList(3, 4, 5, 6));
        
        // 并集
        Set<Integer> union = new HashSet<>(a);
        union.addAll(b);
        
        // 交集
        Set<Integer> intersection = new HashSet<>(a);
        intersection.retainAll(b);
        
        // 差集
        Set<Integer> difference = new HashSet<>(a);
        difference.removeAll(b);
        
        // 子集检查
        boolean isSubset = a.containsAll(new HashSet<>(Arrays.asList(1, 2)));
        
        // Stream创建Set
        Set<Integer> squares = IntStream.range(0, 10)
            .map(x -> x * x)
            .boxed()
            .collect(Collectors.toSet());
        
        // Collections.unmodifiableSet（不可变Set）
        Set<Integer> frozen = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(1, 2, 3)));
    }
}

集合对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 创建           │ {1, 2, 3}           │ new HashSet<>()     │
│ 添加           │ set.add(x)          │ set.add(x)          │
│ 删除           │ set.remove(x)       │ set.remove(x)       │
│ 并集           │ a | b               │ addAll()            │
│ 交集           │ a & b               │ retainAll()         │
│ 差集           │ a - b               │ removeAll()         │
│ 不可变集合     │ frozenset           │ unmodifiableSet()   │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 5. 高级集合类型
# ============================================================================

print("\n=== 5. 高级集合类型 ===\n")

def advanced_collections():
    """Python的高级集合类型"""
    
    # deque（双端队列）
    from collections import deque
    
    dq = deque([1, 2, 3])
    dq.append(4)       # 右端添加
    dq.appendleft(0)   # 左端添加
    print(f"deque: {dq}")
    
    dq.pop()           # 右端弹出
    dq.popleft()       # 左端弹出
    print(f"弹出后: {dq}")
    
    # OrderedDict（有序字典，Python 3.7+普通dict已有序）
    from collections import OrderedDict
    
    ordered = OrderedDict()
    ordered['first'] = 1
    ordered['second'] = 2
    ordered['third'] = 3
    print(f"\nOrderedDict: {ordered}")
    
    # ChainMap（链式字典）
    from collections import ChainMap
    
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'b': 3, 'c': 4}
    chain = ChainMap(dict1, dict2)
    print(f"\nChainMap: {chain}")
    print(f"  查找'b': {chain['b']}")  # 返回第一个找到的
    
    # heapq（堆）
    import heapq
    
    numbers = [5, 2, 8, 1, 9]
    heapq.heapify(numbers)  # 原地转换为最小堆
    print(f"\n堆: {numbers}")
    print(f"  最小值: {heapq.heappop(numbers)}")
    
    heapq.heappush(numbers, 3)
    print(f"  添加3后: {numbers}")

advanced_collections()

"""
Java对比:
import java.util.*;

public class JavaAdvancedCollections {
    public static void advancedCollections() {
        // Deque（双端队列）
        Deque<Integer> deque = new ArrayDeque<>();
        deque.addLast(1);
        deque.addLast(2);
        deque.addFirst(0);
        deque.removeLast();
        deque.removeFirst();
        
        // LinkedHashMap（有序Map）
        Map<String, Integer> ordered = new LinkedHashMap<>();
        ordered.put("first", 1);
        ordered.put("second", 2);
        
        // PriorityQueue（优先队列/堆）
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        heap.offer(5);
        heap.offer(2);
        heap.offer(8);
        int min = heap.poll();  // 获取最小值
    }
}
"""

# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示集合与数据结构差异"""
    print("=" * 70)
    print("Python vs Java: 集合与数据结构深度对比")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("学习要点")
    print("=" * 70)
    print("""
    对于Java程序员学习Python集合：
    
    1. 列表 vs ArrayList
       - Python列表更简洁：[1, 2, 3]
       - 支持负索引：list[-1]
       - 强大的切片：list[1:3]
       - 列表推导式：[x for x in range(10)]
    
    2. 元组的独特性
       - Java没有内置元组
       - 用于多值返回和解包
       - 不可变，可作为字典键
       - namedtuple提供命名访问
    
    3. 字典 vs HashMap
       - Python字典语法更简洁
       - Python 3.7+保持插入顺序
       - 字典推导式很强大
       - defaultdict和Counter很实用
    
    4. 集合运算
       - Python支持数学集合运算符
       - |, &, -, ^ 更直观
       - Java需要方法调用
    
    5. 推导式
       - 列表推导：[x for x in iterable]
       - 字典推导：{k: v for k, v in items}
       - 集合推导：{x for x in iterable}
       - 比Java Stream更简洁
    
    6. 内置vs导入
       - list, dict, set, tuple是内置的
       - defaultdict, Counter等需要导入
       - collections模块很重要
    
    常见陷阱：
    - {}是空字典，不是空集合（用set()）
    - 列表是可变的，字典键必须不可变
    - 字典.keys()返回视图，不是列表
    - 切片不包括结束索引
    """)

if __name__ == "__main__":
    main()

