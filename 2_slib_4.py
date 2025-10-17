# 系统和进程相关
import sys
import subprocess
import threading
import multiprocessing
import queue
import logging
import argparse
import configparser
import pickle

def system_info_demo():
    """系统信息示例"""
    print("=== 系统信息 ===")
    
    print(f"Python版本: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"命令行参数: {sys.argv}")
    print(f"模块搜索路径: {sys.path[:3]}...")  # 只显示前3个

def subprocess_demo():
    """子进程示例"""
    print("\n=== 子进程处理 ===")
    
    try:
        # 执行系统命令
        result = subprocess.run(['echo', '你好，世界！'], 
                              capture_output=True, text=True)
        print(f"命令输出: {result.stdout.strip()}")
        
        # 获取当前日期
        result = subprocess.run(['date'], 
                              capture_output=True, text=True)
        print(f"当前日期: {result.stdout.strip()}")
        
    except Exception as e:
        print(f"执行命令失败: {e}")

def threading_demo():
    """多线程示例"""
    print("\n=== 多线程处理 ===")
    
    def worker(name, delay):
        for i in range(3):
            print(f"线程 {name}: 工作 {i+1}")
            threading.Event().wait(delay)
        print(f"线程 {name}: 完成")
    
    # 创建线程
    threads = []
    for i in range(2):
        t = threading.Thread(target=worker, args=(f"Worker-{i+1}", 0.5))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("所有线程完成")

def queue_demo():
    """队列示例"""
    print("\n=== 队列处理 ===")
    
    # 创建队列
    q = queue.Queue()
    
    # 生产者
    def producer():
        for i in range(5):
            item = f"item-{i+1}"
            q.put(item)
            print(f"生产: {item}")
            threading.Event().wait(0.1)
    
    # 消费者
    def consumer():
        while True:
            try:
                item = q.get(timeout=1)
                print(f"消费: {item}")
                q.task_done()
            except queue.Empty:
                break
    
    # 启动生产者和消费者
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    
    producer_thread.start()
    consumer_thread.start()
    
    producer_thread.join()
    consumer_thread.join()

def logging_demo():
    """日志记录示例"""
    print("\n=== 日志记录 ===")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('MyApp')
    
    logger.debug("这是调试信息")
    logger.info("这是信息日志")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    print("日志已记录到 app.log 文件")

def argparse_demo():
    """命令行参数解析示例"""
    print("\n=== 命令行参数解析 ===")
    
    # 模拟命令行参数
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['script.py', '--name', '张三', '--age', '25', '--verbose']
    
    parser = argparse.ArgumentParser(description='示例程序')
    parser.add_argument('--name', type=str, default='匿名', help='姓名')
    parser.add_argument('--age', type=int, default=0, help='年龄')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    print(f"解析的参数: 姓名={args.name}, 年龄={args.age}, 详细模式={args.verbose}")
    
    # 恢复原始命令行参数
    sys.argv = original_argv

def config_demo():
    """配置文件示例"""
    print("\n=== 配置文件处理 ===")
    
    # 创建配置文件
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'debug': 'False',
        'log_level': 'INFO'
    }
    config['database'] = {
        'host': 'localhost',
        'port': '3306',
        'user': 'admin'
    }
    config['api'] = {
        'url': 'https://api.example.com',
        'timeout': '30'
    }
    
    # 写入配置文件
    with open('config.ini', 'w', encoding='utf-8') as f:
        config.write(f)
    
    # 读取配置文件
    config_reader = configparser.ConfigParser()
    config_reader.read('config.ini', encoding='utf-8')
    
    print(f"数据库配置: {dict(config_reader['database'])}")
    print(f"API配置: {dict(config_reader['api'])}")

def pickle_demo():
    """序列化示例"""
    print("\n=== 对象序列化 ===")
    
    # 要序列化的数据
    data = {
        'name': '张三',
        'scores': [85, 92, 78, 96],
        'info': {'age': 25, 'city': '北京'}
    }
    
    # 序列化到文件
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    # 从文件反序列化
    with open('data.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"原始数据: {data}")
    print(f"加载数据: {loaded_data}")
    print(f"数据相等: {data == loaded_data}")

if __name__ == "__main__":
    system_info_demo()
    subprocess_demo()
    threading_demo()
    queue_demo()
    logging_demo()
    argparse_demo()
    config_demo()
    pickle_demo()