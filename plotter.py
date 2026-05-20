"""
plotter.py — 可视化模块

功能:
  - 球心轨迹图 (叠加在视频帧上)
  - y-t 位移曲线
  - v-t 速度曲线
  - 终端速度区标注
  - 三方法对比柱状图
"""

import os

import matplotlib
matplotlib.use("Agg")          # 无头渲染，不弹出窗口
import matplotlib.pyplot as plt
import numpy as np

# 中文字体修复 — 直接注册 Windows 字体
import matplotlib.font_manager as fm
for _fp in [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]:
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass
for _f in ["Microsoft YaHei", "SimHei"]:
    if any(_f in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

from config import OUTPUT_DIR

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})


def set_output_dir(path: str = None):
    """设置输出目录"""
    d = path or OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    return d


def save(name: str, fig=None):
    """保存图片到输出目录"""
    d = set_output_dir()
    path = os.path.join(d, name)
    (fig or plt).savefig(path, bbox_inches="tight")
    print(f"[图] 已保存 → {path}")
    return path


def plot_yt(t: np.ndarray, y_mm: np.ndarray, label: str = "实验数据",
            title: str = "小球 y-t 位移曲线", filename: str = "yt_curve.png"):
    """
    绘制 y-t 位移曲线
    参数:
      t: 时间 (s)
      y_mm: 位移 (mm)，y 正方向向下
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y_mm, "b.", markersize=2, label=label)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("位移 (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(filename, fig)


def plot_vt(t: np.ndarray, v_mm_s: np.ndarray, label: str = "速度",
            title: str = "小球 v-t 速度曲线", filename: str = "vt_curve.png"):
    """绘制 v-t 速度曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, v_mm_s, "r-", linewidth=1, label=label)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("速度 (mm/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(filename, fig)


def plot_terminal_zone(t: np.ndarray, y_mm: np.ndarray,
                       t_zone: tuple, v_t: float = None,
                       filename: str = "terminal_zone.png"):
    """
    绘制 y-t 并标注终端速度区
    参数:
      t, y_mm: 完整轨迹
      t_zone: (t_start, t_end) 终端区起止时间
      v_t: 终端速度 (mm/s)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y_mm, "b.", markersize=2, label="全部数据")

    # 高亮终端区
    mask = (t >= t_zone[0]) & (t <= t_zone[1])
    ax.plot(t[mask], y_mm[mask], "r.", markersize=3, label="终端速度区")

    # 线性拟合线
    if v_t is not None:
        t_fit = np.linspace(t_zone[0], t_zone[1], 100)
        y_fit = y_mm[mask][0] + v_t * (t_fit - t[mask][0])  # 近似
        ax.plot(t_fit, y_fit, "r--", linewidth=2,
                label=f"v_t = {v_t:.2f} mm/s")

    ax.axvline(t_zone[0], color="gray", linestyle=":", alpha=0.6)
    ax.axvline(t_zone[1], color="gray", linestyle=":", alpha=0.6)

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("位移 (mm)")
    ax.set_title("终端速度区自动判定结果")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save(filename, fig)


def plot_comparison(methods: list, means: list, stds: list,
                    filename: str = "comparison.png"):
    """
    三方法对比柱状图
    参数:
      methods: 方法名列表 ["秒表法", "Tracker 法", "OpenCV 法"]
      means:   黏度均值列表
      stds:    标准差列表
    """
    colors = ["#D85D5D", "#F6B73C", "#0395BC"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("黏度 η (Pa·s)")
    ax.set_title("三方法测量结果对比")
    ax.grid(axis="y", alpha=0.3)
    return save(filename, fig)


def plot_correction(labels: list, values: list,
                    filename: str = "correction_comparison.png"):
    """修正前后黏度对比图"""
    colors = ["#607D8B", "#0395BC"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.4,
                  edgecolor="white", alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.6,
                f"{val:.4f}", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")

    ax.set_ylabel("黏度 η (Pa·s)")
    ax.set_title("修正前后黏度对比")
    ax.grid(axis="y", alpha=0.3)
    return save(filename, fig)


def plot_trajectory_on_frame(frame, xs, ys, filename="trajectory_overlay.png"):
    """将轨迹叠加在视频帧上"""
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(frame)
    ax.plot(xs, ys, "r-", linewidth=1, alpha=0.7, label="球心轨迹")
    ax.scatter(xs[::5], ys[::5], s=10, c="r", alpha=0.4)
    ax.set_title("球心轨迹叠加图")
    ax.legend()
    return save(filename, fig)
