import subprocess
import sys
import os

# ==========================================
# 演示专用配置
# 注意：演示过程中请勿打开此文件，以免泄露 Key
# ==========================================
API_KEY = "sk-271209a24e4f4d8b845632bab5663488"
PROVIDER = "qwen"

def run_demo():
    # 1. 获取指令
    query = ""
    if len(sys.argv) > 1:
        # 如果命令行带了参数，直接使用
        query = " ".join(sys.argv[1:])
    else:
        # 否则交互式输入
        print("\n=== Qwen Agent 演示启动器 ===")
        print("请输入演示指令 (直接回车使用默认测试用例):")
        try:
            user_input = input("> ").strip()
            if user_input:
                query = user_input
            else:
                # 默认测试用例
                query = "打开百度 ，等待页面加载，然后在搜索框（屏幕中央偏上）输入'长城汽车'并回车"
        except KeyboardInterrupt:
            print("\n取消演示")
            return

    # 2. 构造命令
    # 确保使用虚拟环境的 python
    python_exe = ".venv/bin/python"
    if not os.path.exists(python_exe):
        # 如果找不到虚拟环境，尝试使用系统 python
        python_exe = sys.executable

    cmd = [
        python_exe,
        "main.py",
        "--provider", PROVIDER,
        "--api_key", API_KEY,
        "--query", query
    ]

    # 3. 执行命令
    print(f"\n🚀 正在启动演示...")
    print(f"📋 执行任务: {query}")
    print("-" * 50)
    
    try:
        # 使用 subprocess.run 执行，这样 Key 不会显示在终端历史记录中
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 演示已手动停止")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    run_demo()
