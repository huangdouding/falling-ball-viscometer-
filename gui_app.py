#!/usr/bin/env python3
"""
落球法液体黏度自动分析系统 — GUI 入口

用法：
    python gui_app.py

或直接双击运行。

基于机器视觉自动追踪与误差修正的落球法液体黏度测量实验改进。
"""

import sys
import os

# 确保 src 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 提前固定 matplotlib 后端，避免 Windows 下导入时卡死
os.environ.setdefault("MPLBACKEND", "QtAgg")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.gui.main_window import MainWindow


def main():
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("落球法液体黏度自动分析系统")
    app.setOrganizationName("PhysicsLab")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
