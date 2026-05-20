"""
图表输出模块（v6 分段绘制 + 点类型颜色区分）

提供功能：
  1. trajectory_plot.png     — 时间-位移轨迹，分段绘制，不连跨越 NaN 的线
  2. velocity_plot.png       — 时间-速度曲线，分段绘制
  3. fit_plot.png            — 终端区 y-t 线性拟合图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
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

_ZH_FONT = None
for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
    if any(_f in f.name for f in fm.fontManager.ttflist):
        _ZH_FONT = _f
        break
if _ZH_FONT:
    plt.rcParams["font.sans-serif"] = [_ZH_FONT]
plt.rcParams["axes.unicode_minus"] = False

import logging
logger = logging.getLogger(__name__)

# ============================================================
#  点类型配色
# ============================================================
_POINT_STYLES = {
    "raw":          {"color": "#2196F3", "marker": "o", "label": "raw detection",  "ms": 4, "z": 3},
    "predicted":    {"color": "#FF9800", "marker": "^", "label": "prediction",     "ms": 5, "z": 2},
    "interpolated": {"color": "#4CAF50", "marker": "s", "label": "interpolation",  "ms": 5, "z": 2},
}
_OUTLIER_STYLE = {"color": "#F44336", "marker": "x", "label": "outlier (rejected)", "ms": 6, "z": 5}
_LINE_STYLE = {"color": "#90CAF9", "linewidth": 0.8, "alpha": 0.5}


def _split_segments(t: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> list:
    """将连续有效点分成段，遇到无效/NaN 即断开。

    Returns:
        [(t_seg, y_seg), ...]  每段为有效数据子数组
    """
    segments = []
    in_seg = False
    start = 0
    for i in range(len(t)):
        v = valid_mask[i] if i < len(valid_mask) else False
        if v and not np.isnan(y[i]) and not np.isnan(t[i]):
            if not in_seg:
                start = i
                in_seg = True
        else:
            if in_seg:
                seg = (t[start:i], y[start:i])
                if len(seg[0]) >= 2:
                    segments.append(seg)
                in_seg = False
    if in_seg:
        seg = (t[start:], y[start:])
        if len(seg[0]) >= 2:
            segments.append(seg)
    return segments


def plot_trajectory(traj_df: pd.DataFrame, terminal_region: dict, output_dir: str):
    """绘制 y-t 轨迹图，按连续有效段分段绘制。

    点类型颜色：
      - 蓝色 ● raw detection
      - 橙色 ▲ prediction
      - 绿色 ■ interpolation
      - 红色 ✗ outlier (rejected)
    """
    t = traj_df["time_s"].values
    y = traj_df["y_m"].values
    pt = traj_df["point_type"].values if "point_type" in traj_df.columns else None
    valid = traj_df["valid"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- 连续段浅色连线（仅 raw 有效段） ----
    raw_valid = (pt == "raw") & valid & np.isfinite(t) & np.isfinite(y) \
        if pt is not None else valid
    segments = _split_segments(t, y, raw_valid)
    for t_seg, y_seg in segments:
        ax.plot(t_seg, y_seg, "-", color=_LINE_STYLE["color"],
                linewidth=_LINE_STYLE["linewidth"], alpha=_LINE_STYLE["alpha"], zorder=1)

    # ---- 散点：各点类型 ----
    for ptype, sty in _POINT_STYLES.items():
        if pt is not None:
            mask = (pt == ptype) & valid & np.isfinite(t) & np.isfinite(y)
        else:
            mask = valid if ptype == "raw" else np.zeros(len(t), dtype=bool)
        if mask.sum() > 0:
            ax.plot(t[mask], y[mask], linestyle="None",
                    marker=sty["marker"], color=sty["color"], markersize=sty["ms"],
                    label=sty["label"], zorder=sty["z"])

    # ---- 离群点（valid=False 但 type=raw 且坐标有限） ----
    if pt is not None:
        outlier_mask = (pt == "raw") & ~valid & np.isfinite(t) & np.isfinite(y)
        if outlier_mask.sum() > 0:
            ax.plot(t[outlier_mask], y[outlier_mask], linestyle="None",
                    marker=_OUTLIER_STYLE["marker"], color=_OUTLIER_STYLE["color"],
                    markersize=_OUTLIER_STYLE["ms"], label=_OUTLIER_STYLE["label"],
                    zorder=_OUTLIER_STYLE["z"])

    # ---- 终端速度区高亮 ----
    if terminal_region.get("found", False):
        t_start = terminal_region["start_time_s"]
        t_end = terminal_region["end_time_s"]
        mask = (t >= t_start) & (t <= t_end) & raw_valid
        if mask.sum() > 0:
            ax.plot(t[mask], y[mask], "r.-", markersize=5,
                    label=f"terminal ({t_start:.2f}s - {t_end:.2f}s)", zorder=3)
            ax.axvspan(t_start, t_end, alpha=0.12, color="red")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y displacement (m)")
    ax.set_title("Ball Fall: y-t Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    out_path = os.path.join(output_dir, "trajectory_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("轨迹图已保存: %s", out_path)


def plot_velocity(velocity_df: pd.DataFrame, terminal_region: dict, output_dir: str):
    """绘制 v-t 速度曲线图。

    显示三种速度（互不干扰）：
      1. 浅灰细线·散点: 逐帧差分速度 v_frame_diff（仅作 debug 参考，不可靠）
      2. 橙色点/线:     局部滑动窗口 y-t 拟合速度 v_local_window（趋势参考）
      3. 绿色虚线:       终端区间 y-t 线性拟合斜率 v_terminal_fit（★ 黏度计算依据）

    红色阴影：自动判定的终端速度区间。
    """
    t = velocity_df["time_s"].values
    v_frame = velocity_df["v_m_s"].values           # 逐帧差分（debug）
    v_smooth = velocity_df["smooth_v_m_s"].values   # 滑动窗口拟合（趋势参考）
    valid = velocity_df["valid"].values
    pt = velocity_df.get("point_type", None)

    if valid.sum() < 2:
        logger.warning("有效速度点不足 2 个，跳过速度图。")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- (1) 逐帧差分速度 v_frame_diff（浅灰细线，debug 参考） ----
    vel_valid = valid & np.isfinite(t) & np.isfinite(v_frame)
    segments = _split_segments(t, v_frame, vel_valid)
    for t_seg, v_seg in segments:
        ax.plot(t_seg, v_seg, "-", color="#D0D0D0", linewidth=0.5, alpha=0.4, zorder=1)
    ax.plot(t[vel_valid], v_frame[vel_valid], linestyle="None",
            marker=".", color="#A0A0A0", markersize=2, alpha=0.3,
            zorder=1, label="v_frame_diff (debug)")

    # ---- (1b) 平滑速度散点（v_smooth 仍按点类型着色，显示数据来源） ----
    for ptype, sty in _POINT_STYLES.items():
        mask = (pt == ptype) if pt is not None else np.zeros(len(t), dtype=bool)
        if mask.sum() > 0:
            ax.plot(t[mask], v_smooth[mask], linestyle="None",
                    marker=sty["marker"], color=sty["color"], markersize=sty["ms"],
                    alpha=0.5, label=f"{sty['label']} (win-fit)", zorder=sty["z"])

    # ---- (2) 局部滑动窗口 y-t 拟合速度（橙色点/线，trend reference） ----
    wv = terminal_region.get("window_velocities", [])
    if wv:
        times = np.array([w["time"] for w in wv])
        vals = np.array([w["v"] for w in wv])
        n_show = min(len(times), 150)
        if n_show < len(times):
            idx = np.linspace(0, len(times) - 1, n_show, dtype=int)
            times = times[idx]
            vals = vals[idx]
        ax.plot(times, vals, "-", color="#FF9800", linewidth=0.8, alpha=0.4,
                zorder=2, label="_window_line")
        ax.plot(times, vals, linestyle="None",
                marker=".", color="#FF9800", markersize=4, alpha=0.7,
                zorder=3, label="v_local_window (ref)")

    # ---- (3) 终端速度 v_terminal_fit（绿色虚线，★ 黏度计算依据） ----
    if terminal_region.get("found", False):
        t_start = terminal_region["start_time_s"]
        t_end = terminal_region["end_time_s"]
        v_t = terminal_region["terminal_velocity_m_s"]
        r2 = terminal_region.get("r2", 0)
        cv = terminal_region.get("cv", 0)
        ax.axhline(y=v_t, color="green", linestyle="--", linewidth=2.0,
                   label=f"v_terminal_fit = {v_t:.4f} m/s  (R²={r2:.4f}, Cv={cv:.4f})",
                   zorder=5)
        ax.axvspan(t_start, t_end, alpha=0.12, color="red", label=f"terminal interval")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("Velocity v-t: three velocity types (terminal fit = viscosity basis)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "velocity_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("速度图已保存: %s", out_path)

    out_path = os.path.join(output_dir, "velocity_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("速度图已保存: %s", out_path)


def plot_terminal_fit(traj_df: pd.DataFrame, terminal_region: dict, output_dir: str):
    """绘制终端速度区 y-t 线性拟合图。

    只使用 raw 点，标注 v_t, R², Cv。
    """
    if not terminal_region.get("found", False):
        logger.warning("未找到终端速度区，跳过拟合图。")
        return

    t_start = terminal_region["start_time_s"]
    t_end = terminal_region["end_time_s"]
    v_t = terminal_region["terminal_velocity_m_s"]
    r2 = terminal_region["r2"]
    cv = terminal_region["cv"]

    t = traj_df["time_s"].values
    y = traj_df["y_m"].values
    pt = traj_df["point_type"].values if "point_type" in traj_df.columns else None
    valid = traj_df["valid"].values

    # 只使用 raw 有效点
    raw_mask = (pt == "raw") & valid if pt is not None else valid
    mask = (t >= t_start) & (t <= t_end) & raw_mask & np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        logger.warning("终端区 raw 有效点不足 3 个，跳过拟合图。")
        return

    t_fit = t[mask]
    y_fit = y[mask]
    y_pred = v_t * t_fit + terminal_region["fit_intercept"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_fit, y_fit, "bo", markersize=4, label="数据点 (raw)", zorder=2)
    ax.plot(t_fit, y_pred, "r-", linewidth=2, label="线性拟合", zorder=3)

    info_text = (
        f"v_t = {v_t:.6f} m/s\n"
        f"R²  = {r2:.6f}\n"
        f"Cv  = {cv:.4f}\n"
        f"区间: {t_start:.3f}s - {t_end:.3f}s"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y (m)")
    ax.set_title("终端速度区 y-t 线性拟合")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "fit_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("拟合图已保存: %s", out_path)


def plot_all(traj_df: pd.DataFrame, velocity_df: pd.DataFrame,
             terminal_region: dict, output_dir: str):
    """生成所有图表。"""
    plot_trajectory(traj_df, terminal_region, output_dir)
    plot_velocity(velocity_df, terminal_region, output_dir)
    plot_terminal_fit(traj_df, terminal_region, output_dir)
