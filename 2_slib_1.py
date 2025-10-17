# 文件操作和路径处理
import os
import pathlib
from datetime import datetime, timedelta
import json
import csv

def file_operations_demo():
    """文件操作和路径处理示例"""
    print("=== 文件操作和路径处理 ===")
    
    # os 模块 - 操作系统接口
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 列出当前目录文件
    files = os.listdir('.')
    print(f"当前目录文件: {files}")
    
    # pathlib 模块 - 面向对象的路径操作
    path = pathlib.Path('.')
    python_files = list(path.glob('*.py'))
    print(f"Python文件: {python_files}")
    
    # 创建示例文件
    sample_file = pathlib.Path('sample.txt')
    sample_file.write_text('这是一个示例文件\n包含多行内容')
    print(f"创建文件: {sample_file.absolute()}")
    
    # 读取文件
    content = sample_file.read_text()
    print(f"文件内容: {content}")

def datetime_demo():
    """日期时间处理示例"""
    print("\n=== 日期时间处理 ===")
    
    # 获取当前时间
    now: datetime = datetime.now()
    print(f"当前时间: {now}")
    
    # 格式化时间
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"格式化时间: {formatted_time}")
    
    # 时间计算
    tomorrow = now + timedelta(days=1)
    print(f"明天: {tomorrow.strftime('%Y-%m-%d')}")
    
    # 一周前
    week_ago = now - timedelta(weeks=1)
    print(f"一周前: {week_ago.strftime('%Y-%m-%d')}")

def json_demo():
    """JSON处理示例"""
    print("\n=== JSON处理 ===")
    
    # Python对象转JSON
    data = {
        "name": "张三",
        "age": 30,
        "city": "北京",
        "hobbies": ["读书", "游泳", "编程"]
    }
    
    json_string = json.dumps(data, ensure_ascii=False, indent=2)
    print(f"JSON字符串:\n{json_string}")
    
    # JSON转Python对象
    parsed_data = json.loads(json_string)
    print(f"解析后的数据: {parsed_data}")
    
    # 写入JSON文件
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("数据已写入 data.json 文件")

def csv_demo():
    """CSV处理示例"""
    print("\n=== CSV处理 ===")
    
    # 写入CSV文件
    csv_data = [
        ['姓名', '年龄', '城市'],
        ['张三', 25, '北京'],
        ['李四', 30, '上海'],
        ['王五', 28, '广州']
    ]
    
    with open('people.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print("CSV数据已写入 people.csv 文件")
    
    # 读取CSV文件
    with open('people.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        print("CSV文件内容:")
        for row in reader:
            print(row)

if __name__ == "__main__":
    file_operations_demo()
    datetime_demo()
    json_demo()
    csv_demo()