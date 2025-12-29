# Computer Use (Qwen-VL 适配版)

本项目完全开源免费，旨在帮助国内开发者更便捷地体验和研究基于视觉大模型的浏览器自动化技术，尤其致力于**浏览器自动化** **辅助web自动化测试**，探索 AI 驱动的新一代测试方案。

## ✨ 主要特性

- **国内模型适配**：深度适配阿里云 Qwen-VL-Max 模型，解决国内访问 Gemini 困难的问题。
- **中文环境优化**：
    - 默认搜索引擎调整为百度。
    - 优化了 System Prompt，使其更懂中文指令。
    - 修复了浏览器默认页面的加载问题。
- **智能体增强**：
    - 增加了“任务完成”检测机制，自动判断任务结束并移交人类接管。
    - 实现了对纯文本回复的 Fallback 解析，提升模型指令执行的稳定性。
- **演示友好**：以我个人为例，我利用 `juyufeng.py` 脚本，支持隐藏 API Key 进行安全演示。


## 🚀 快速开始

### 1. 安装

**克隆仓库**

```bash
git clone https://github.com/your-username/computer-use-qwen.git
cd computer-use-qwen
```

**配置 Python 环境并安装依赖**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**安装 Playwright 浏览器**

```bash
# 安装 Playwright 及其依赖
playwright install
```

### 2. 配置

你需要一个阿里云百炼的 API Key (DashScope)。

**设置环境变量 (推荐)**

```bash
export DASHSCOPE_API_KEY="你的_DASHSCOPE_API_KEY"
```

或者在运行命令时通过 `--api_key` 参数传入。

### 3. 运行

**交互式演示 (推荐)**

我们提供了一个方便的演示脚本，支持隐藏 Key 并在终端交互式输入指令：

```bash
python juyufeng.py "打开B站搜索黑神话悟空"
```

**命令行运行**

你也可以使用原始的 `main.py` 进行更细致的控制：

```bash
# 使用 Qwen 模型
python main.py --provider qwen --query "打开百度搜索xxx"

# 如果没有设置环境变量，可以手动传入 Key
python main.py --provider qwen --api_key "sk-..." --query "打开百度搜索xxx"
```

## 🛠️ 二次开发指南

本项目核心逻辑位于 `agent_qwen.py` (Qwen 智能体实现) 和 `main.py` (入口与参数处理)。

如果你想适配其他 OpenAI 兼容接口的模型，可以参考 `agent_qwen.py` 中的 `QwenAgent` 类实现。

## 🫡 致敬与鸣谢

本项目基于 Google 的 [Computer Use Preview](https://github.com/google-gemini/computer-use-preview) 项目构建。感谢 Google 团队开源了如此优秀的浏览器自动化框架，为社区提供了宝贵的探索基础。

Original Project Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0.

## 📄 许可证

本项目遵循 [Apache License 2.0](LICENSE) 协议。
你可以免费使用、修改和分发本项目，但请保留原始版权声明和协议文件。

## 💬 交流与讨论

![一起讨论吧](一起讨论吧.jpg)
