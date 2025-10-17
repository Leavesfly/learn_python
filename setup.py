#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI智能体技术学习项目 - 安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
readme_file = Path(__file__).parent / "1_PROJECT_SUMMARY.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# 读取requirements文件
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="learn-python-ai-agents",
    version="1.0.0",
    author="山泽",
    author_email="your.email@example.com",
    description="AI智能体技术教学与实践项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/learn_python",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # 可以添加命令行工具
            # "ai-agent=module:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
