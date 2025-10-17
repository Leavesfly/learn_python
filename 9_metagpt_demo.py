# MetaGPT使用演示
# filename: metagpt_demo.py

"""
MetaGPT 使用演示
这个脚本展示了 MetaGPT 的基本使用方法，包括安装、配置和运行示例
"""

import asyncio
import os
from metagpt.software_company import SoftwareCompany
from metagpt.roles import ProductManager, Architect, ProjectManager, Engineer
from metagpt.team import Team
from metagpt.actions import WriteCode
from metagpt.schema import Message


def setup_environment():
    """
    设置 MetaGPT 环境
    需要配置 API Key 和其他必要设置
    """
    # 设置 OpenAI API Key（必需）
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    
    # 可选：设置其他 LLM 配置
    # os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"
    # os.environ["AZURE_OPENAI_API_KEY"] = "your_azure_api_key"
    
    print("✅ 环境配置完成")


async def demo_1_simple_software_company():
    """
    演示1: 简单的软件公司运行
    这是最基本的使用方式，类似于运行 startup.py
    """
    print("\n=== 演示1: 简单软件公司 ===")
    
    # 创建软件公司实例
    company = SoftwareCompany()
    
    # 定义项目需求
    idea = "开发一个简单的待办事项管理应用"
    
    print(f"项目需求: {idea}")
    print("正在生成项目方案...")
    
    # 运行软件公司流程
    result = await company.run(idea)
    
    print("✅ 项目方案生成完成")
    print(f"生成的文档数量: {len(result)}")
    
    return result


async def demo_2_custom_team():
    """
    演示2: 自定义团队配置
    展示如何创建自定义的智能体团队
    """
    print("\n=== 演示2: 自定义团队 ===")
    
    # 创建自定义团队
    team = Team()
    
    # 添加不同角色的智能体
    team.hire([
        ProductManager(),  # 产品经理
        Architect(),      # 架构师
        ProjectManager(), # 项目经理
        Engineer(),       # 工程师
    ])
    
    # 定义需求
    requirement = "设计一个在线聊天系统"
    
    print(f"团队需求: {requirement}")
    print("团队成员协作中...")
    
    # 团队协作执行任务
    result = await team.run(requirement)
    
    print("✅ 团队协作完成")
    return result


async def demo_3_single_agent():
    """
    演示3: 单个智能体使用
    展示如何使用单个智能体完成特定任务
    """
    print("\n=== 演示3: 单个智能体 ===")
    
    # 创建工程师智能体
    engineer = Engineer()
    
    # 创建编程任务消息
    message = Message(
        content="请编写一个 Python 函数来计算斐波那契数列",
        role="user"
    )
    
    print("工程师智能体正在编写代码...")
    
    # 执行编程任务
    result = await engineer.run(message)
    
    print("✅ 代码编写完成")
    print("生成的代码:")
    print(result)
    
    return result


async def demo_4_step_by_step():
    """
    演示4: 分步骤执行
    展示 MetaGPT 的详细执行过程
    """
    print("\n=== 演示4: 分步骤执行 ===")
    
    # 1. 产品需求分析
    pm = ProductManager()
    requirement = "开发一个简单的计算器应用"
    
    print("1. 产品经理分析需求...")
    prd = await pm.run(requirement)
    
    # 2. 系统架构设计
    architect = Architect()
    print("2. 架构师设计系统架构...")
    architecture = await architect.run(prd)
    
    # 3. 项目管理
    project_manager = ProjectManager()
    print("3. 项目经理制定开发计划...")
    plan = await project_manager.run(architecture)
    
    # 4. 代码实现
    engineer = Engineer()
    print("4. 工程师实现代码...")
    code = await engineer.run(plan)
    
    print("✅ 完整开发流程执行完成")
    
    return {
        'prd': prd,
        'architecture': architecture,
        'plan': plan,
        'code': code
    }


def demo_5_command_line_usage():
    """
    演示5: 命令行使用方法
    展示如何通过命令行运行 MetaGPT
    """
    print("\n=== 演示5: 命令行使用 ===")
    print("以下是常用的命令行使用方法:")
    
    commands = [
        "# 基本使用",
        "python startup.py '开发一个井字棋游戏'",
        "",
        "# 指定输出目录",
        "python startup.py '设计一个博客系统' --project-path ./my_project",
        "",
        "# 使用不同的 LLM",
        "python startup.py '创建一个聊天机器人' --llm-api azure",
        "",
        "# 查看帮助",
        "python startup.py --help",
        "",
        "# 使用新版本语法 (v0.5+)",
        "metagpt '开发一个命令行黑杰克游戏'",
    ]
    
    for cmd in commands:
        print(cmd)


async def main():
    """
    主函数 - 运行所有演示
    """
    print("🚀 MetaGPT 使用演示开始")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    try:
        # 运行演示1: 简单软件公司
        await demo_1_simple_software_company()
        
        # 运行演示2: 自定义团队
        await demo_2_custom_team()
        
        # 运行演示3: 单个智能体
        await demo_3_single_agent()
        
        # 运行演示4: 分步骤执行
        await demo_4_step_by_step()
        
        # 演示5: 命令行使用
        demo_5_command_line_usage()
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查 API Key 配置和网络连接")
    
    print("\n🎉 所有演示完成!")


# 安装指南
def installation_guide():
    """
    MetaGPT 安装指南
    """
    print("📦 MetaGPT 安装指南")
    print("=" * 30)
    
    install_commands = [
        "# 方法1: 稳定版本安装",
        "pip install metagpt",
        "",
        "# 方法2: 开发版本安装",
        "git clone https://github.com/geekan/MetaGPT.git",
        "cd MetaGPT",
        "pip install -e .",
        "",
        "# 安装额外依赖",
        "pip install metagpt[extra]",
        "",
        "# 验证安装",
        "python -c 'import metagpt; print(metagpt.__version__)'",
    ]
    
    for cmd in install_commands:
        print(cmd)


# 配置指南
def configuration_guide():
    """
    MetaGPT 配置指南
    """
    print("\n⚙️  MetaGPT 配置指南")
    print("=" * 30)
    
    print("""
1. 设置 API Key:
   export OPENAI_API_KEY="your_openai_api_key"
   
2. 创建配置文件 ~/.metagpt/config.yaml:
   llm:
     api_type: "openai"
     model: "gpt-4"
     api_key: "your_api_key"
   
3. 高级配置:
   - 支持多种 LLM: OpenAI, Azure, Anthropic
   - 可配置输出目录和日志级别
   - 支持代理设置和超时配置
   
4. 检查配置:
   python -c "from metagpt.config import config; print(config)"
    """)


if __name__ == "__main__":
    print("MetaGPT 完整使用演示")
    print("🔧 首先查看安装和配置指南...")
    
    # 显示安装指南
    installation_guide()
    
    # 显示配置指南
    configuration_guide()
    
    print("\n" + "="*50)
    print("⚠️  注意: 运行演示前请确保:")
    print("1. 已安装 MetaGPT")
    print("2. 已配置 OpenAI API Key")
    print("3. 网络连接正常")
    print("="*50)
    
    # 询问是否运行演示
    run_demo = input("\n是否运行演示代码? (y/n): ").lower().strip()
    
    if run_demo == 'y':
        # 运行异步主函数
        asyncio.run(main())
    else:
        print("演示代码已准备就绪，您可以根据需要单独运行各个部分。")