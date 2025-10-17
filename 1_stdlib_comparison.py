"""
Python vs Java: 标准库与常用模块对比
============================
面向Java程序员的Python标准库学习指南

作者: Python学习系列
目标读者: 熟悉Java标准库的开发者
学习重点: 掌握Python常用标准库模块
"""

import os
import sys
import datetime
import json
import re
import math
import random
from pathlib import Path

# ============================================================================
# 1. 文件I/O操作
# ============================================================================

print("=== 1. 文件I/O操作 ===\n")

def file_io_operations():
    """
    Python文件操作
    """
    
    # 写文件（自动关闭）
    print("文件写入:")
    with open("demo.txt", "w", encoding="utf-8") as f:
        f.write("第一行\n")
        f.write("第二行\n")
        f.writelines(["第三行\n", "第四行\n"])
    print("  文件已写入")
    
    # 读文件
    print("\n文件读取:")
    with open("demo.txt", "r", encoding="utf-8") as f:
        content = f.read()  # 读取全部
        print(f"  全部内容:\n{content}")
    
    # 按行读取
    with open("demo.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"  行列表: {lines}")
    
    # 迭代读取（内存高效）
    with open("demo.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            print(f"  行{i}: {line.strip()}")
    
    # 追加模式
    with open("demo.txt", "a", encoding="utf-8") as f:
        f.write("追加的行\n")
    
    # 二进制模式
    with open("demo.bin", "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    
    # 清理
    os.remove("demo.txt")
    os.remove("demo.bin")

file_io_operations()

"""
Java对比:
import java.io.*;
import java.nio.file.*;

public class JavaFileIO {
    public static void fileIOOperations() throws IOException {
        // 写文件（Java 7+ try-with-resources）
        try (BufferedWriter writer = new BufferedWriter(
                new FileWriter("demo.txt", StandardCharsets.UTF_8))) {
            writer.write("第一行\n");
            writer.write("第二行\n");
        }
        
        // 读文件
        try (BufferedReader reader = new BufferedReader(
                new FileReader("demo.txt", StandardCharsets.UTF_8))) {
            String content = reader.lines()
                .collect(Collectors.joining("\n"));
        }
        
        // 按行读取
        List<String> lines = Files.readAllLines(Paths.get("demo.txt"));
        
        // 迭代读取
        try (BufferedReader reader = new BufferedReader(new FileReader("demo.txt"))) {
            String line;
            int i = 1;
            while ((line = reader.readLine()) != null) {
                System.out.println("行" + i + ": " + line);
                i++;
            }
        }
        
        // 追加模式
        try (BufferedWriter writer = new BufferedWriter(
                new FileWriter("demo.txt", true))) {
            writer.write("追加的行\n");
        }
        
        // 删除文件
        Files.delete(Paths.get("demo.txt"));
    }
}

文件I/O对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 打开文件       │ open(file, mode)    │ new FileWriter()    │
│ 自动关闭       │ with语句            │ try-with-resources  │
│ 读取全部       │ f.read()            │ Files.readString()  │
│ 按行读取       │ f.readlines()       │ Files.readAllLines()│
│ 迭代读取       │ for line in f       │ while循环           │
│ 写入           │ f.write()           │ writer.write()      │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 2. 路径操作
# ============================================================================

print("\n=== 2. 路径操作 ===\n")

def path_operations():
    """
    Python路径操作（推荐使用pathlib）
    """
    
    # pathlib（现代方式）
    print("pathlib方式:")
    path = Path("folder/subfolder/file.txt")
    print(f"  路径: {path}")
    print(f"  父目录: {path.parent}")
    print(f"  文件名: {path.name}")
    print(f"  扩展名: {path.suffix}")
    print(f"  无扩展名: {path.stem}")
    
    # 路径拼接
    base = Path("folder")
    full_path = base / "subfolder" / "file.txt"
    print(f"  拼接路径: {full_path}")
    
    # 创建目录
    test_dir = Path("test_folder")
    test_dir.mkdir(exist_ok=True)
    print(f"  创建目录: {test_dir}")
    
    # 创建文件
    test_file = test_dir / "test.txt"
    test_file.write_text("测试内容", encoding="utf-8")
    print(f"  创建文件: {test_file}")
    
    # 检查存在
    print(f"  文件存在: {test_file.exists()}")
    print(f"  是文件: {test_file.is_file()}")
    print(f"  是目录: {test_dir.is_dir()}")
    
    # 列出目录
    print(f"  目录内容: {list(test_dir.iterdir())}")
    
    # 通配符
    # print(f"  *.txt文件: {list(test_dir.glob('*.txt'))}")
    
    # os.path方式（传统）
    print("\nos.path方式:")
    import os.path
    
    filepath = "folder/file.txt"
    print(f"  目录: {os.path.dirname(filepath)}")
    print(f"  文件名: {os.path.basename(filepath)}")
    print(f"  拼接: {os.path.join('folder', 'file.txt')}")
    print(f"  绝对路径: {os.path.abspath('.')}")
    
    # 清理
    test_file.unlink()
    test_dir.rmdir()

path_operations()

"""
Java对比:
import java.nio.file.*;

public class JavaPathOperations {
    public static void pathOperations() throws IOException {
        // Java NIO Path
        Path path = Paths.get("folder/subfolder/file.txt");
        System.out.println("路径: " + path);
        System.out.println("父目录: " + path.getParent());
        System.out.println("文件名: " + path.getFileName());
        
        // 路径拼接
        Path base = Paths.get("folder");
        Path fullPath = base.resolve("subfolder").resolve("file.txt");
        
        // 创建目录
        Path testDir = Paths.get("test_folder");
        Files.createDirectories(testDir);
        
        // 创建文件
        Path testFile = testDir.resolve("test.txt");
        Files.writeString(testFile, "测试内容");
        
        // 检查存在
        boolean exists = Files.exists(testFile);
        boolean isFile = Files.isRegularFile(testFile);
        boolean isDir = Files.isDirectory(testDir);
        
        // 列出目录
        Files.list(testDir).forEach(System.out::println);
        
        // 删除
        Files.delete(testFile);
        Files.delete(testDir);
    }
}

路径操作对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 路径对象       │ Path("path")        │ Paths.get("path")   │
│ 路径拼接       │ path / "file"       │ path.resolve("file")│
│ 创建目录       │ path.mkdir()        │ Files.createDir()   │
│ 写文件         │ path.write_text()   │ Files.writeString() │
│ 读文件         │ path.read_text()    │ Files.readString()  │
│ 检查存在       │ path.exists()       │ Files.exists()      │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 3. 日期时间操作
# ============================================================================

print("\n=== 3. 日期时间操作 ===\n")

def datetime_operations():
    """
    Python日期时间操作
    """
    
    from datetime import datetime, date, time, timedelta
    
    # 当前时间
    now = datetime.now()
    print(f"当前时间: {now}")
    print(f"当前日期: {date.today()}")
    
    # 创建特定日期
    birthday = datetime(1990, 5, 15, 10, 30, 0)
    print(f"生日: {birthday}")
    
    # 格式化
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"格式化: {formatted}")
    
    # 解析
    parsed = datetime.strptime("2024-01-15", "%Y-%m-%d")
    print(f"解析: {parsed}")
    
    # 时间运算
    tomorrow = now + timedelta(days=1)
    print(f"明天: {tomorrow}")
    
    week_ago = now - timedelta(weeks=1)
    print(f"一周前: {week_ago}")
    
    # 时间差
    diff = now - birthday
    print(f"年龄(天): {diff.days}")
    
    # 时间戳
    timestamp = now.timestamp()
    print(f"时间戳: {timestamp}")
    
    from_timestamp = datetime.fromtimestamp(timestamp)
    print(f"从时间戳: {from_timestamp}")

datetime_operations()

"""
Java对比:
import java.time.*;
import java.time.format.DateTimeFormatter;

public class JavaDateTimeOperations {
    public static void dateTimeOperations() {
        // 当前时间
        LocalDateTime now = LocalDateTime.now();
        LocalDate today = LocalDate.now();
        
        // 创建特定日期
        LocalDateTime birthday = LocalDateTime.of(1990, 5, 15, 10, 30, 0);
        
        // 格式化
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        String formatted = now.format(formatter);
        
        // 解析
        LocalDate parsed = LocalDate.parse("2024-01-15");
        
        // 时间运算
        LocalDateTime tomorrow = now.plusDays(1);
        LocalDateTime weekAgo = now.minusWeeks(1);
        
        // 时间差
        Duration diff = Duration.between(birthday, now);
        long days = diff.toDays();
        
        // 时间戳
        Instant instant = now.toInstant(ZoneOffset.UTC);
        long timestamp = instant.getEpochSecond();
        LocalDateTime fromTimestamp = LocalDateTime.ofInstant(
            Instant.ofEpochSecond(timestamp), ZoneId.systemDefault());
    }
}

日期时间对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 当前时间       │ datetime.now()      │ LocalDateTime.now() │
│ 创建           │ datetime(y,m,d)     │ LocalDateTime.of()  │
│ 格式化         │ strftime()          │ format()            │
│ 解析           │ strptime()          │ parse()             │
│ 时间运算       │ + timedelta()       │ plus/minus方法      │
│ 时间差         │ datetime1 - dt2     │ Duration.between()  │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 4. JSON处理
# ============================================================================

print("\n=== 4. JSON处理 ===\n")

def json_operations():
    """
    Python JSON处理
    """
    
    import json
    
    # Python对象转JSON
    data = {
        "name": "张三",
        "age": 25,
        "skills": ["Python", "Java", "Go"],
        "active": True,
        "score": 95.5
    }
    
    # 转为JSON字符串
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    print("JSON字符串:")
    print(json_str)
    
    # 解析JSON字符串
    parsed = json.loads(json_str)
    print(f"\n解析结果: {parsed}")
    print(f"姓名: {parsed['name']}")
    
    # 写入文件
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\nJSON已写入文件")
    
    # 从文件读取
    with open("data.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"从文件加载: {loaded}")
    
    # 清理
    os.remove("data.json")

json_operations()

"""
Java对比:
import com.google.gson.Gson;  // 或使用Jackson
import com.google.gson.GsonBuilder;

public class JavaJSONOperations {
    static class Person {
        String name;
        int age;
        List<String> skills;
        boolean active;
        double score;
    }
    
    public static void jsonOperations() {
        Gson gson = new GsonBuilder()
            .setPrettyPrinting()
            .create();
        
        // Java对象转JSON
        Person person = new Person();
        person.name = "张三";
        person.age = 25;
        person.skills = Arrays.asList("Python", "Java", "Go");
        person.active = true;
        person.score = 95.5;
        
        String jsonStr = gson.toJson(person);
        System.out.println(jsonStr);
        
        // 解析JSON
        Person parsed = gson.fromJson(jsonStr, Person.class);
        
        // 写入文件
        try (FileWriter writer = new FileWriter("data.json")) {
            gson.toJson(person, writer);
        }
        
        // 从文件读取
        try (FileReader reader = new FileReader("data.json")) {
            Person loaded = gson.fromJson(reader, Person.class);
        }
    }
}

JSON对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 库             │ json (内置)         │ Gson/Jackson        │
│ 序列化         │ json.dumps()        │ gson.toJson()       │
│ 反序列化       │ json.loads()        │ gson.fromJson()     │
│ 文件写入       │ json.dump(obj, f)   │ gson.toJson(obj, w) │
│ 文件读取       │ json.load(f)        │ gson.fromJson(r, c) │
│ 类型           │ 字典/列表           │ POJO类              │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 5. 正则表达式
# ============================================================================

print("\n=== 5. 正则表达式 ===\n")

def regex_operations():
    """
    Python正则表达式
    """
    
    import re
    
    text = "联系电话: 138-1234-5678, Email: test@example.com"
    
    # 搜索
    phone_match = re.search(r'(\d{3})-(\d{4})-(\d{4})', text)
    if phone_match:
        print(f"找到电话: {phone_match.group()}")
        print(f"区号: {phone_match.group(1)}")
    
    # 查找所有
    numbers = re.findall(r'\d+', text)
    print(f"所有数字: {numbers}")
    
    # 替换
    masked = re.sub(r'\d{4}', '****', text)
    print(f"掩码: {masked}")
    
    # 分割
    parts = re.split(r'[,:]', text)
    print(f"分割: {parts}")
    
    # 编译正则（性能优化）
    pattern = re.compile(r'\w+@\w+\.\w+')
    email_match = pattern.search(text)
    if email_match:
        print(f"邮箱: {email_match.group()}")
    
    # 匹配验证
    is_valid_email = bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', 'test@example.com'))
    print(f"邮箱验证: {is_valid_email}")

regex_operations()

"""
Java对比:
import java.util.regex.*;

public class JavaRegexOperations {
    public static void regexOperations() {
        String text = "联系电话: 138-1234-5678, Email: test@example.com";
        
        // 搜索
        Pattern phonePattern = Pattern.compile("(\\d{3})-(\\d{4})-(\\d{4})");
        Matcher phoneMatcher = phonePattern.matcher(text);
        if (phoneMatcher.find()) {
            System.out.println("找到电话: " + phoneMatcher.group());
            System.out.println("区号: " + phoneMatcher.group(1));
        }
        
        // 查找所有
        Pattern numberPattern = Pattern.compile("\\d+");
        Matcher numberMatcher = numberPattern.matcher(text);
        while (numberMatcher.find()) {
            System.out.println(numberMatcher.group());
        }
        
        // 替换
        String masked = text.replaceAll("\\d{4}", "****");
        
        // 分割
        String[] parts = text.split("[,:]");
        
        // 匹配验证
        boolean isValidEmail = "test@example.com".matches("^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$");
    }
}

正则表达式对比：
┌────────────────┬─────────────────────┬─────────────────────┐
│   操作         │   Python            │   Java              │
├────────────────┼─────────────────────┼─────────────────────┤
│ 搜索           │ re.search()         │ matcher.find()      │
│ 匹配           │ re.match()          │ matcher.matches()   │
│ 查找所有       │ re.findall()        │ 循环find()          │
│ 替换           │ re.sub()            │ replaceAll()        │
│ 分割           │ re.split()          │ split()             │
│ 编译           │ re.compile()        │ Pattern.compile()   │
└────────────────┴─────────────────────┴─────────────────────┘
"""

# ============================================================================
# 6. 数学与随机
# ============================================================================

print("\n=== 6. 数学与随机 ===\n")

def math_and_random():
    """
    Python数学和随机操作
    """
    
    import math
    import random
    
    # 数学函数
    print("数学函数:")
    print(f"  π = {math.pi}")
    print(f"  e = {math.e}")
    print(f"  sqrt(16) = {math.sqrt(16)}")
    print(f"  pow(2, 3) = {math.pow(2, 3)}")
    print(f"  ceil(3.2) = {math.ceil(3.2)}")
    print(f"  floor(3.8) = {math.floor(3.8)}")
    print(f"  sin(π/2) = {math.sin(math.pi/2)}")
    
    # 随机数
    print("\n随机数:")
    print(f"  random(): {random.random()}")  # 0-1之间
    print(f"  randint(1, 10): {random.randint(1, 10)}")  # 1-10之间整数
    print(f"  uniform(1.0, 10.0): {random.uniform(1.0, 10.0)}")  # 浮点数
    
    # 随机选择
    items = ['a', 'b', 'c', 'd', 'e']
    print(f"  choice: {random.choice(items)}")
    print(f"  sample(3): {random.sample(items, 3)}")
    
    # 打乱
    shuffled = items.copy()
    random.shuffle(shuffled)
    print(f"  shuffle: {shuffled}")

math_and_random()

"""
Java对比:
import java.util.*;

public class JavaMathAndRandom {
    public static void mathAndRandom() {
        // 数学函数
        System.out.println("π = " + Math.PI);
        System.out.println("e = " + Math.E);
        System.out.println("sqrt(16) = " + Math.sqrt(16));
        System.out.println("pow(2, 3) = " + Math.pow(2, 3));
        System.out.println("ceil(3.2) = " + Math.ceil(3.2));
        System.out.println("floor(3.8) = " + Math.floor(3.8));
        System.out.println("sin(π/2) = " + Math.sin(Math.PI/2));
        
        // 随机数
        Random random = new Random();
        System.out.println("nextDouble(): " + random.nextDouble());
        System.out.println("nextInt(10): " + random.nextInt(10));
        
        // 随机选择（需要自己实现）
        List<String> items = Arrays.asList("a", "b", "c", "d", "e");
        String choice = items.get(random.nextInt(items.size()));
        
        // 打乱
        List<String> shuffled = new ArrayList<>(items);
        Collections.shuffle(shuffled);
    }
}
"""

# ============================================================================
# 7. 系统与环境
# ============================================================================

print("\n=== 7. 系统与环境 ===\n")

def system_operations():
    """
    Python系统操作
    """
    
    import sys
    import os
    import platform
    
    # 系统信息
    print("系统信息:")
    print(f"  平台: {platform.system()}")
    print(f"  版本: {platform.version()}")
    print(f"  架构: {platform.machine()}")
    print(f"  Python版本: {sys.version}")
    
    # 环境变量
    print("\n环境变量:")
    print(f"  PATH: {os.environ.get('PATH', '未设置')[:50]}...")
    
    # 设置环境变量
    os.environ['MY_VAR'] = 'my_value'
    print(f"  MY_VAR: {os.environ['MY_VAR']}")
    
    # 命令行参数
    print(f"\n命令行参数: {sys.argv}")
    
    # 当前工作目录
    print(f"当前目录: {os.getcwd()}")
    
    # 执行系统命令
    # result = os.system('echo "Hello"')
    print("可以使用subprocess模块执行命令")

system_operations()

"""
Java对比:
public class JavaSystemOperations {
    public static void systemOperations() {
        // 系统信息
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("版本: " + System.getProperty("os.version"));
        System.out.println("架构: " + System.getProperty("os.arch"));
        System.out.println("Java版本: " + System.getProperty("java.version"));
        
        // 环境变量
        String path = System.getenv("PATH");
        
        // 系统属性
        System.setProperty("my.property", "value");
        String value = System.getProperty("my.property");
        
        // 命令行参数（通过main方法）
        // public static void main(String[] args)
        
        // 当前目录
        String currentDir = System.getProperty("user.dir");
    }
}
"""

# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示标准库差异"""
    print("=" * 70)
    print("Python vs Java: 标准库与常用模块对比")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("学习要点")
    print("=" * 70)
    print("""
    对于Java程序员学习Python标准库：
    
    1. 文件I/O
       - with语句自动管理资源
       - pathlib提供面向对象的路径操作
       - 比Java的File类更简洁
    
    2. 日期时间
       - datetime模块类似Java 8的java.time
       - timedelta用于时间运算
       - strftime/strptime格式化和解析
    
    3. JSON
       - json是内置模块
       - 直接转换Python字典
       - 不需要定义POJO类
    
    4. 正则表达式
       - re模块功能强大
       - 语法与Java基本相同
       - findall更方便
    
    5. 常用模块
       - os: 操作系统接口
       - sys: 系统参数和函数
       - pathlib: 现代路径操作
       - collections: 高级数据结构
       - itertools: 迭代器工具
       - functools: 函数工具
    
    6. 包管理
       - pip安装第三方包
       - requirements.txt管理依赖
       - 虚拟环境venv隔离项目
    
    推荐学习的模块：
    - requests: HTTP请求（类似Java的HttpClient）
    - pandas: 数据分析（Java无直接等价）
    - numpy: 数值计算
    - flask/django: Web框架
    - pytest: 测试框架
    """)

if __name__ == "__main__":
    main()
