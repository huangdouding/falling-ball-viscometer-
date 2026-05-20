"""
轨迹后处理

将 trajectory post-processing 从 tracking.py 中提取为独立模块。

功能：
  1. 短间隙线性插值（≤3 帧）
  2. 异常跳变剔除（dx/dt/回跳检查）
  3. y 归一化（以首帧为原点）
  4. 轨迹段质量评估（R², Cv）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def interpolate_short_gaps(
    df: pd.DataFrame,
    max_gap: int = 3,
    inplace: bool = False,
) -> pd.DataFrame:
    """对 ≤max_gap 帧的短间隙做线性插值。

    只插值有效检测点之间的小间隙，保证轨迹连续性。
    被插值的点标记 point_type = "interpolated"。
    """
    result = df if inplace else df.copy()
    valid = result["valid"].values
    x = result["x_px"].values
    y = result["y_px"].values
    point_type = result["point_type"].values

    n = len(result)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        # 找下一个有效点
        j = i + 1
        while j < n and not valid[j]:
            j += 1
        if j >= n or not valid[j]:
            break
        gap = j - i - 1
        if 1 <= gap <= max_gap:
            x1, y1 = x[i], y[i]
            x2, y2 = x[j], y[j]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                result.at[i + k, "x_px"] = x1 + t * (x2 - x1)
                result.at[i + k, "y_px"] = y1 + t * (y2 - y1)
                result.at[i + k, "valid"] = True
                result.at[i + k, "point_type"] = "interpolated"
        i = j

    return result


def remove_outliers(
    df: pd.DataFrame,
    dx_max: float = 20.0,
    dy_back_tol: float = 5.0,
    dist_max: float = 80.0,
    inplace: bool = False,
) -> pd.DataFrame:
    """剔除异常跳变点。

    Args:
        dx_max: 水平方向最大变化 (px)
        dy_back_tol: 允许向上回跳的最大值 (px)
        dist_max: 帧间最大位移 (px)
    """
    result = df if inplace else df.copy()
    valid = result["valid"].values
    x = result["x_px"].values.astype(float)
    y = result["y_px"].values.astype(float)

    n = len(result)
    removed = 0
    for i in range(1, n):
        if not valid[i] or not valid[i - 1]:
            continue
        dx = abs(x[i] - x[i - 1])
        dy = y[i] - y[i - 1]   # 向下为正
        dist = np.hypot(dx, dy)

        reason = None
        if dx > dx_max:
            reason = f"dx跳变({dx:.1f}>{dx_max})"
        elif dy < -dy_back_tol:
            reason = f"y回跳({dy:.1f}<-{-dy_back_tol})"
        elif dist > dist_max:
            reason = f"总位移过大({dist:.1f}>{dist_max})"

        if reason:
            result.at[i, "valid"] = False
            result.at[i, "point_type"] = "outlier"
            if result.at[i, "y_m"] is not None:
                result.at[i, "y_m"] = np.nan
            removed += 1
            logger.debug("帧 %d: 异常点剔除 (%s)", i, reason)

    if removed > 0:
        logger.info("轨迹后处理: 已剔除 %d 个异常点", removed)
    return result


def normalize_y(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """以首帧 y 为原点归一化。

    下落分析需要 y 坐标从 0 开始递增（向下为正）。
    对同时有 y_px 和 y_m 列的 DataFrame 生效。
    """
    result = df if inplace else df.copy()

    for col in ["y_px", "y_m"]:
        if col not in result.columns:
            continue
        vals = result[col].values.astype(float)
        valid_mask = result["valid"].values & np.isfinite(vals)
        if valid_mask.sum() == 0:
            continue
        first_valid = np.where(valid_mask)[0][0]
        origin = vals[first_valid]
        vals[valid_mask] = vals[valid_mask] - origin
        vals[~valid_mask] = np.nan
        result[col] = vals

    return result


def assess_segment_quality(
    t: np.ndarray,
    y: np.ndarray,
    point_types: np.ndarray | None = None,
    min_raw_points: int = 10,
) -> tuple[float | None, float | None]:
    """评估 y-t 段的线性拟合质量。

    只使用 point_type == "raw" 的点进行评估。

    Returns:
        (r2, cv): 拟合优度和速度变异系数
    """
    if point_types is not None:
        raw_mask = (point_types == "raw")
    else:
        raw_mask = np.ones(len(t), dtype=bool)

    valid = raw_mask & np.isfinite(t) & np.isfinite(y)
    if valid.sum() < min_raw_points:
        return None, None

    t_v = t[valid]
    y_v = y[valid]

    # 线性拟合 y = slope * t + intercept
    A = np.vstack([t_v, np.ones_like(t_v)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y_v, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None

    # R²
    y_pred = slope * t_v + intercept
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Cv: 差分速度的变异系数
    if len(t_v) >= 3:
        v = np.diff(y_v) / np.diff(t_v)
        v_mean = np.mean(v)
        if v_mean > 0:
            v_std = np.std(v, ddof=1)
            cv = v_std / v_mean
        else:
            cv = 1.0
    else:
        cv = 1.0

    return r2, cv


def compute_displacement(df: pd.DataFrame) -> dict:
    """计算轨迹的基本运动学统计。

    Returns:
        dict with: total_frames, valid_count, raw_count, pred_count, interp_count,
                   y_displacement_px, y_displacement_m, x_std_px, duration_s
    """
    valid_mask = df["valid"].values
    raw_mask = (df["point_type"] == "raw").values
    pred_mask = (df["point_type"] == "predicted").values
    interp_mask = (df["point_type"] == "interpolated").values

    total = len(df)
    valid_count = int(valid_mask.sum())
    raw_count = int(raw_mask.sum())
    pred_count = int(pred_mask.sum())
    interp_count = int(interp_mask.sum())

    result = {
        "total_frames": total,
        "valid_count": valid_count,
        "raw_count": raw_count,
        "pred_count": pred_count,
        "interp_count": interp_count,
        "valid_ratio": valid_count / total if total > 0 else 0.0,
    }

    if "y_px" in df.columns and valid_count >= 2:
        y_vals = df["y_px"].values[valid_mask]
        result["y_displacement_px"] = float(abs(y_vals[-1] - y_vals[0]))
        result["y_min_px"] = float(np.min(y_vals))
        result["y_max_px"] = float(np.max(y_vals))

    if "y_m" in df.columns and valid_count >= 2:
        y_vals = df["y_m"].values[valid_mask]
        result["y_displacement_m"] = float(abs(y_vals[-1] - y_vals[0]))

    if "x_px" in df.columns and raw_count >= 3:
        x_raw = df["x_px"].values[raw_mask]
        result["x_std_px"] = float(np.std(x_raw))

    if "time_s" in df.columns:
        times = df["time_s"].values
        result["duration_s"] = float(times[-1] - times[0])

    return result
