# Makefile for AI Agent Learning Project
# =======================================

.PHONY: help install install-dev clean test format lint type-check run-examples all

# 默认目标
help:
	@echo "AI智能体技术学习项目 - 可用命令:"
	@echo ""
	@echo "  make install          - 安装项目依赖"
	@echo "  make install-dev      - 安装项目依赖（包含开发工具）"
	@echo "  make clean            - 清理临时文件和缓存"
	@echo "  make test             - 运行测试"
	@echo "  make format           - 格式化代码（使用black）"
	@echo "  make lint             - 代码风格检查（使用flake8）"
	@echo "  make type-check       - 类型检查（使用mypy）"
	@echo "  make run-examples     - 运行示例程序"
	@echo "  make all              - 执行完整的检查流程"
	@echo ""

# 安装基础依赖
install:
	pip install -r requirements.txt

# 安装开发依赖
install-dev:
	pip install -e ".[dev]"
	pip install -e ".[jupyter]"

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.py,cover" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	@echo "清理完成！"

# 运行测试
test:
	@echo "运行测试..."
	pytest -v --cov=. --cov-report=term-missing

# 代码格式化
format:
	@echo "格式化代码..."
	black *.py --line-length 100

# 代码风格检查
lint:
	@echo "检查代码风格..."
	flake8 *.py --max-line-length=100 --extend-ignore=E203,W503

# 类型检查
type-check:
	@echo "类型检查..."
	mypy *.py --ignore-missing-imports

# 运行示例程序
run-examples:
	@echo "运行快速开始示例..."
	python 1_quickstart.py
	@echo ""
	@echo "运行强化学习示例..."
	python 12_rl_0.py

# 完整检查流程
all: format lint type-check test
	@echo ""
	@echo "所有检查完成！✅"

# 初始化虚拟环境
venv:
	python -m venv venv
	@echo "虚拟环境已创建！"
	@echo "激活命令:"
	@echo "  macOS/Linux: source venv/bin/activate"
	@echo "  Windows: venv\\Scripts\\activate"

# 导出依赖
freeze:
	pip freeze > requirements-freeze.txt
	@echo "依赖已导出到 requirements-freeze.txt"

# 检查过时的包
outdated:
	pip list --outdated
