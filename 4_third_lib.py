# Python常用第三方库学习指南
"""
这是一个Python第三方库学习指南，涵盖了最常用和实用的第三方库
每个库都包含安装方法、基本用法和实际示例

注意：运行前需要先安装相应的库：
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn flask django fastapi pytest
"""

# ==================== 网络请求库 ====================

def requests_demo():
    """Requests - HTTP库"""
    print("=== Requests HTTP库 ===")
    print("安装: pip install requests")
    
    try:
        import requests
        
        # GET请求
        response = requests.get('https://httpbin.org/json')
        if response.status_code == 200:
            data = response.json()
            print(f"GET请求成功: {list(data.keys())}")
        
        # POST请求
        payload = {'name': '张三', 'age': 25}
        response = requests.post('https://httpbin.org/post', json=payload)
        print(f"POST请求状态码: {response.status_code}")
        
        # 带参数的请求
        params = {'q': 'python', 'page': 1}
        response = requests.get('https://httpbin.org/get', params=params)
        print(f"带参数请求URL: {response.url}")
        
    except ImportError:
        print("请先安装: pip install requests")
    except Exception as e:
        print(f"网络请求失败: {e}")

# ==================== 网页解析库 ====================

def beautifulsoup_demo():
    """Beautiful Soup - HTML/XML解析"""
    print("\n=== Beautiful Soup HTML解析 ===")
    print("安装: pip install beautifulsoup4")
    
    try:
        from bs4 import BeautifulSoup
        
        # 示例HTML
        html_content = """
        <html>
            <head><title>示例页面</title></head>
            <body>
                <div class="content">
                    <h1 id="title">欢迎来到Python世界</h1>
                    <p class="intro">这是一个学习示例</p>
                    <ul class="list">
                        <li>Python基础</li>
                        <li>Web开发</li>
                        <li>数据分析</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找元素
        title = soup.find('title').text
        print(f"页面标题: {title}")
        
        h1_tag = soup.find('h1', {'id': 'title'})
        print(f"H1内容: {h1_tag.text}")
        
        # 查找所有列表项
        li_tags = soup.find_all('li')
        print(f"列表项: {[li.text for li in li_tags]}")
        
        # CSS选择器
        content_div = soup.select_one('.content')
        print(f"Content div包含 {len(content_div.find_all())} 个子元素")
        
    except ImportError:
        print("请先安装: pip install beautifulsoup4")

# ==================== 数据处理库 ====================

def pandas_demo():
    """Pandas - 数据分析"""
    print("\n=== Pandas 数据分析 ===")
    print("安装: pip install pandas")
    
    try:
        import pandas as pd
        
        # 创建数据
        data = {
            '姓名': ['张三', '李四', '王五', '赵六'],
            '年龄': [25, 30, 35, 28],
            '城市': ['北京', '上海', '广州', '深圳'],
            '薪资': [8000, 12000, 15000, 10000]
        }
        
        df = pd.DataFrame(data)
        print("数据框:")
        print(df)
        
        # 基本统计
        print(f"\n年龄统计:")
        print(df['年龄'].describe())
        
        # 数据筛选
        high_salary = df[df['薪资'] > 10000]
        print(f"\n高薪人员:")
        print(high_salary[['姓名', '薪资']])
        
        # 分组统计
        city_avg_salary = df.groupby('城市')['薪资'].mean()
        print(f"\n各城市平均薪资:")
        print(city_avg_salary)
        
        # 保存到CSV
        df.to_csv('员工数据.csv', index=False, encoding='utf-8-sig')
        print("\n数据已保存到 员工数据.csv")
        
    except ImportError:
        print("请先安装: pip install pandas")

def numpy_demo():
    """NumPy - 科学计算"""
    print("\n=== NumPy 科学计算 ===")
    print("安装: pip install numpy")
    
    try:
        import numpy as np
        
        # 创建数组
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([[1, 2], [3, 4], [5, 6]])
        
        print(f"一维数组: {arr1}")
        print(f"二维数组:\n{arr2}")
        print(f"数组形状: {arr2.shape}")
        
        # 数组运算
        print(f"数组平方: {arr1 ** 2}")
        print(f"数组求和: {np.sum(arr1)}")
        print(f"数组平均值: {np.mean(arr1)}")
        
        # 创建特殊数组
        zeros = np.zeros((2, 3))
        ones = np.ones((2, 3))
        random_arr = np.random.rand(3, 3)
        
        print(f"零数组:\n{zeros}")
        print(f"随机数组:\n{random_arr}")
        
        # 数组索引和切片
        print(f"第一行: {arr2[0]}")
        print(f"第一列: {arr2[:, 0]}")
        
    except ImportError:
        print("请先安装: pip install numpy")

# ==================== 数据可视化库 ====================

def matplotlib_demo():
    """Matplotlib - 基础绘图"""
    print("\n=== Matplotlib 基础绘图 ===")
    print("安装: pip install matplotlib")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建数据
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 折线图
        ax1.plot(x, y1, label='sin(x)', color='blue')
        ax1.plot(x, y2, label='cos(x)', color='red')
        ax1.set_title('三角函数图')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True)
        
        # 柱状图
        categories = ['A', 'B', 'C', 'D']
        values = [23, 45, 56, 78]
        ax2.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
        ax2.set_title('柱状图示例')
        ax2.set_ylabel('数值')
        
        plt.tight_layout()
        plt.savefig('matplotlib_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("图表已保存为 matplotlib_demo.png")
        
    except ImportError:
        print("请先安装: pip install matplotlib")

def seaborn_demo():
    """Seaborn - 统计绘图"""
    print("\n=== Seaborn 统计绘图 ===")
    print("安装: pip install seaborn")
    
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        
        # 创建示例数据
        np.random.seed(42)
        data = pd.DataFrame({
            '身高': np.random.normal(170, 10, 100),
            '体重': np.random.normal(65, 15, 100),
            '性别': np.random.choice(['男', '女'], 100),
            '年龄': np.random.randint(20, 60, 100)
        })
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 散点图
        sns.scatterplot(data=data, x='身高', y='体重', hue='性别', ax=ax1)
        ax1.set_title('身高体重关系图')
        
        # 箱线图
        sns.boxplot(data=data, x='性别', y='身高', ax=ax2)
        ax2.set_title('性别身高分布')
        
        # 直方图
        sns.histplot(data=data, x='年龄', bins=20, ax=ax3)
        ax3.set_title('年龄分布')
        
        # 热力图
        correlation = data[['身高', '体重', '年龄']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('相关性热力图')
        
        plt.tight_layout()
        plt.savefig('seaborn_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("统计图表已保存为 seaborn_demo.png")
        
    except ImportError:
        print("请先安装: pip install seaborn pandas matplotlib numpy")

# ==================== Web开发框架 ====================

def flask_demo():
    """Flask - 轻量级Web框架"""
    print("\n=== Flask 轻量级Web框架 ===")
    print("安装: pip install flask")
    
    try:
        from flask import Flask, jsonify, request
        
        app = Flask(__name__)
        
        # 示例数据
        users = [
            {'id': 1, 'name': '张三', 'email': 'zhangsan@example.com'},
            {'id': 2, 'name': '李四', 'email': 'lisi@example.com'}
        ]
        
        @app.route('/')
        def home():
            return '<h1>欢迎来到Flask应用！</h1><p>访问 /api/users 查看用户列表</p>'
        
        @app.route('/api/users', methods=['GET'])
        def get_users():
            return jsonify(users)
        
        @app.route('/api/users', methods=['POST'])
        def create_user():
            data = request.get_json()
            new_user = {
                'id': len(users) + 1,
                'name': data.get('name'),
                'email': data.get('email')
            }
            users.append(new_user)
            return jsonify(new_user), 201
        
        print("Flask应用示例代码已准备就绪")
        print("运行方式: python -c \"from this_file import flask_demo; app.run(debug=True)\"")
        print("然后访问 http://localhost:5000")
        
        return app
        
    except ImportError:
        print("请先安装: pip install flask")

def fastapi_demo():
    """FastAPI - 现代Web框架"""
    print("\n=== FastAPI 现代Web框架 ===")
    print("安装: pip install fastapi uvicorn")
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import List
        
        app = FastAPI(title="FastAPI示例", description="一个简单的API示例")
        
        # 数据模型
        class User(BaseModel):
            id: int
            name: str
            email: str
            age: int = None
        
        # 示例数据
        users_db = [
            User(id=1, name="张三", email="zhangsan@example.com", age=25),
            User(id=2, name="李四", email="lisi@example.com", age=30)
        ]
        
        @app.get("/")
        def read_root():
            return {"message": "欢迎来到FastAPI！", "docs": "/docs"}
        
        @app.get("/users", response_model=List[User])
        def get_users():
            return users_db
        
        @app.post("/users", response_model=User)
        def create_user(user: User):
            users_db.append(user)
            return user
        
        @app.get("/users/{user_id}", response_model=User)
        def get_user(user_id: int):
            for user in users_db:
                if user.id == user_id:
                    return user
            return {"error": "用户未找到"}
        
        print("FastAPI应用示例代码已准备就绪")
        print("运行方式: uvicorn filename:app --reload")
        print("然后访问 http://localhost:8000/docs 查看API文档")
        
        return app
        
    except ImportError:
        print("请先安装: pip install fastapi uvicorn")

# ==================== 测试库 ====================

def pytest_demo():
    """Pytest - 测试框架"""
    print("\n=== Pytest 测试框架 ===")
    print("安装: pip install pytest")
    
    try:
        import pytest
        
        # 被测试的函数
        def add(a, b):
            return a + b
        
        def divide(a, b):
            if b == 0:
                raise ValueError("除数不能为零")
            return a / b
        
        class Calculator:
            def multiply(self, a, b):
                return a * b
        
        # 测试函数（通常放在单独的test_文件中）
        def test_add():
            assert add(2, 3) == 5
            assert add(-1, 1) == 0
            assert add(0, 0) == 0
        
        def test_divide():
            assert divide(10, 2) == 5
            assert divide(9, 3) == 3
        
        def test_divide_by_zero():
            with pytest.raises(ValueError):
                divide(10, 0)
        
        def test_calculator():
            calc = Calculator()
            assert calc.multiply(3, 4) == 12
        
        # Fixture示例
        @pytest.fixture
        def sample_data():
            return [1, 2, 3, 4, 5]
        
        def test_with_fixture(sample_data):
            assert len(sample_data) == 5
            assert sum(sample_data) == 15
        
        print("测试示例已准备就绪")
        print("创建test_文件并运行: pytest test_filename.py")
        print("或运行所有测试: pytest")
        
        # 手动运行一些测试作为演示
        print("\n运行示例测试:")
        test_add()
        test_divide()
        test_calculator()
        print("所有基础测试通过!")
        
    except ImportError:
        print("请先安装: pip install pytest")

# ==================== 实用工具库 ====================

def other_useful_libraries():
    """其他实用库介绍"""
    print("\n=== 其他常用第三方库 ===")
    
    libraries = [
        {
            'name': 'Pillow (PIL)',
            'install': 'pip install Pillow',
            'description': '图像处理库，用于打开、处理、保存图片',
            'usage': 'from PIL import Image; img = Image.open("photo.jpg")'
        },
        {
            'name': 'SQLAlchemy',
            'install': 'pip install sqlalchemy',
            'description': 'Python SQL工具包和对象关系映射(ORM)库',
            'usage': 'from sqlalchemy import create_engine'
        },
        {
            'name': 'Celery',
            'install': 'pip install celery',
            'description': '分布式任务队列，用于异步任务处理',
            'usage': 'from celery import Celery'
        },
        {
            'name': 'Scrapy',
            'install': 'pip install scrapy',
            'description': '专业的网页爬虫框架',
            'usage': 'scrapy startproject myproject'
        },
        {
            'name': 'Click',
            'install': 'pip install click',
            'description': '创建命令行界面的库',
            'usage': 'import click; @click.command()'
        },
        {
            'name': 'Jinja2',
            'install': 'pip install jinja2',
            'description': '模板引擎，用于生成动态HTML',
            'usage': 'from jinja2 import Template'
        },
        {
            'name': 'PyYAML',
            'install': 'pip install pyyaml',
            'description': 'YAML格式文件处理',
            'usage': 'import yaml; yaml.load(file)'
        },
        {
            'name': 'python-dotenv',
            'install': 'pip install python-dotenv',
            'description': '从.env文件加载环境变量',
            'usage': 'from dotenv import load_dotenv'
        }
    ]
    
    for lib in libraries:
        print(f"\n📚 {lib['name']}")
        print(f"   安装: {lib['install']}")
        print(f"   描述: {lib['description']}")
        print(f"   用法: {lib['usage']}")

def learning_resources():
    """学习资源推荐"""
    print("\n=== 学习资源推荐 ===")
    
    resources = [
        "📖 官方文档 - 每个库的官方文档是最权威的学习资料",
        "🌐 PyPI (pypi.org) - Python包索引，查找和了解库的信息",
        "📺 YouTube - 搜索库名+tutorial获取视频教程",
        "📱 GitHub - 查看库的源代码和示例项目",
        "📚 Real Python - 高质量的Python教程网站",
        "🎓 Coursera/edX - 在线课程平台的Python课程",
        "📖 《Python Tricks》- Brett Slatkin著，进阶技巧",
        "📝 Stack Overflow - 遇到问题时的首选求助平台"
    ]
    
    for resource in resources:
        print(f"  {resource}")

if __name__ == "__main__":
    print("🐍 Python常用第三方库学习指南")
    print("=" * 50)
    
    # 网络和数据获取
    requests_demo()
    beautifulsoup_demo()
    
    # 数据处理和分析
    pandas_demo()
    numpy_demo()
    
    # 数据可视化
    matplotlib_demo()
    seaborn_demo()
    
    # Web开发
    flask_demo()
    fastapi_demo()
    
    # 测试
    pytest_demo()
    
    # 其他库介绍
    other_useful_libraries()
    
    # 学习资源
    learning_resources()
    
    print("\n" + "=" * 50)
    print("🎯 建议学习顺序:")
    print("1. requests (网络请求)")
    print("2. pandas + numpy (数据处理)")
    print("3. matplotlib (可视化)")
    print("4. flask/fastapi (Web开发)")
    print("5. pytest (测试)")
    print("6. 根据需要学习其他专业库")