# 落球法液体黏度自动测量系统

> 基于机器视觉自动追踪与误差修正的落球法液体黏度测量实验改进

## 📁 仓库结构

本仓库是项目的**外层包装仓库**，核心代码位于子模块中：

- **[`falling-ball-viscometer-core`](https://github.com/huangdouding/falling-ball-viscometer-core)** — 核心代码库（检测、追踪、黏度计算、GUI）
  - 子模块路径：`B/`
  - 包含完整的 20+ 条提交历史和全部源代码

## 🚀 快速开始

```bash
# 克隆并初始化子模块
git clone --recursive https://github.com/huangdouding/falling-ball-viscometer-.git
cd falling-ball-viscometer-
git submodule update --init --recursive

# 进入核心代码目录
cd B

# 安装依赖
pip install -r requirements.txt

# 运行 GUI
python gui_app.py
```

## 📖 详细文档

项目完整文档（算法说明、配置参数、使用指南）位于核心仓库的 [README.md](https://github.com/huangdouding/falling-ball-viscometer-core)。

## 🏷️ 版本

最新版本：**[v1.1.0 - 视差修正版](https://github.com/huangdouding/falling-ball-viscometer-/releases/tag/v1.1.0)**
