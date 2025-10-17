#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目环境初始化脚本
自动检查和设置Python开发环境
"""

import sys
import subprocess
import platform
import os
from pathlib import Path


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    """打印成功信息"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """打印错误信息"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """打印信息"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")


def check_python_version():
    """检查Python版本"""
    print_info("检查Python版本...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print_error("需要Python 3.8或更高版本")
        return False


def check_pip():
    """检查pip是否可用"""
    print_info("检查pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print_success(f"pip已安装: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print_error("pip未安装或不可用")
        return False


def check_venv():
    """检查是否在虚拟环境中"""
    print_info("检查虚拟环境...")
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print_success("当前在虚拟环境中")
    else:
        print_warning("当前不在虚拟环境中")
        print_warning("建议创建虚拟环境: python -m venv venv")
    
    return in_venv


def install_dependencies():
    """安装项目依赖"""
    print_info("安装项目依赖...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("未找到requirements.txt文件")
        return False
    
    try:
        print("正在安装依赖包，请稍候...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        print_success("依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print_error("依赖安装失败")
        print_error(e.stderr)
        return False


def install_dev_dependencies():
    """安装开发依赖"""
    print_info("安装开发依赖...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
            capture_output=True,
            text=True,
            check=True
        )
        print_success("开发依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print_warning("开发依赖安装失败（可选）")
        return False


def verify_installation():
    """验证关键包是否安装成功"""
    print_info("验证核心依赖...")
    
    packages = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'matplotlib': 'Matplotlib'
    }
    
    all_success = True
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} 安装成功")
        except ImportError:
            print_error(f"{name} 安装失败")
            all_success = False
    
    return all_success


def run_example():
    """运行示例程序"""
    print_info("运行示例程序...")
    
    example_file = Path("1_quickstart.py")
    if not example_file.exists():
        print_warning("示例文件不存在，跳过")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, str(example_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        print_success("示例程序运行成功")
        return True
    except subprocess.TimeoutExpired:
        print_success("示例程序正在运行（已超时但正常）")
        return True
    except subprocess.CalledProcessError as e:
        print_warning("示例程序运行出错（可能正常）")
        return True


def print_next_steps():
    """打印后续步骤"""
    print_header("🎉 环境设置完成！")
    
    print(f"{Colors.OKGREEN}{Colors.BOLD}后续步骤:{Colors.ENDC}\n")
    
    steps = [
        ("1️⃣ ", "激活虚拟环境（如果还未激活）", "source venv/bin/activate"),
        ("2️⃣ ", "查看学习指南", "python 1_INDEX_LEARNING_GUIDE.py"),
        ("3️⃣ ", "运行PyTorch教程", "python 6_pytorch_1_basics.py"),
        ("4️⃣ ", "探索强化学习", "python 12_rl_0.py"),
        ("5️⃣ ", "阅读项目文档", "查看 README.md"),
    ]
    
    for emoji, desc, cmd in steps:
        print(f"{Colors.OKCYAN}{emoji}{desc}{Colors.ENDC}")
        if cmd:
            print(f"   {Colors.OKBLUE}$ {cmd}{Colors.ENDC}\n")
    
    print(f"\n{Colors.BOLD}项目结构:{Colors.ENDC}")
    print(f"  • 1_*.py          - Python基础教程")
    print(f"  • 6_*.py          - PyTorch教程")
    print(f"  • 12_rl_*.py      - 强化学习系列")
    print(f"  • 15_multi_*.py   - 多智能体系统")
    print(f"  • 19_rag_*.py     - RAG系统")
    print(f"  • 26_mcp_*.py     - MCP架构")
    
    print(f"\n{Colors.BOLD}有用的命令:{Colors.ENDC}")
    if platform.system() != "Windows":
        print(f"  • make help       - 查看所有Make命令")
        print(f"  • make test       - 运行测试")
        print(f"  • make format     - 格式化代码")
    
    print(f"\n{Colors.OKGREEN}祝学习愉快！Happy Learning! 🚀{Colors.ENDC}\n")


def main():
    """主函数"""
    print_header("AI智能体技术学习项目 - 环境初始化")
    
    # 显示系统信息
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"Python路径: {sys.executable}")
    
    # 检查环境
    checks = [
        ("Python版本", check_python_version),
        ("pip工具", check_pip),
    ]
    
    print_header("环境检查")
    for name, check_func in checks:
        if not check_func():
            print_error(f"{name}检查失败，无法继续")
            sys.exit(1)
    
    # 检查虚拟环境（警告但不强制）
    check_venv()
    
    # 询问是否安装依赖
    print_header("依赖安装")
    response = input(f"\n{Colors.BOLD}是否安装项目依赖？(y/n): {Colors.ENDC}").lower()
    
    if response in ['y', 'yes', '']:
        if not install_dependencies():
            print_error("依赖安装失败")
            sys.exit(1)
        
        # 询问是否安装开发依赖
        response = input(f"\n{Colors.BOLD}是否安装开发工具（pytest, black等）？(y/n): {Colors.ENDC}").lower()
        if response in ['y', 'yes']:
            install_dev_dependencies()
        
        # 验证安装
        print_header("验证安装")
        if not verify_installation():
            print_warning("部分依赖未成功安装，但可以继续")
        
        # 运行示例
        response = input(f"\n{Colors.BOLD}是否运行示例程序？(y/n): {Colors.ENDC}").lower()
        if response in ['y', 'yes']:
            run_example()
    
    # 显示后续步骤
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}操作已取消{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
