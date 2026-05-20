"""
终端速度拟合

从 terminal_region.py 提取并简化。

核心原则：
  1. 终端速度 = y-t 线性拟合斜率（纯位置拟合，不涉及帧差速度）
  2. 滑动窗口仅用于终端区判定（哪段 y-t 最接近直线）
  3. 只使用 "raw" 类型点拟合
  4. 无合格段时返回 found=False（不清真回退）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def find_terminal_velocity(
    traj_df: pd.DataFrame,
    velocity_df: pd.DataFrame | None = None,
    config: dict | None = None,
    *,
    debug_callback: Callable | None = None,
) -> dict:
    """自动判定终端速度区并拟合终端速度。

    Args:
        traj_df: 轨迹 DataFrame（含 time_s, y_m, point_type 列）
        velocity_df: 速度 DataFrame（可选，仅用于 debug 显示）
        config: 配置参数

    Returns:
        dict with:
          found: bool
          terminal_velocity_m_s: float | None
          start_time_s, end_time_s: 终端区起止时间
          r2: 拟合优度
          cv: 速度变异系数
          message: 状态信息
    """
    cfg = {
        "terminal_window_sec": 0.8,
        "terminal_ignore_start_sec": 0.3,
        "terminal_ignore_end_sec": 0.5,
        "r2_threshold": 0.999,
        "cv_threshold": 0.03,
    }
    if config:
        cfg.update({k: config[k] for k in cfg if k in config})

    window_sec = float(cfg["terminal_window_sec"])
    ignore_start = float(cfg["terminal_ignore_start_sec"])
    ignore_end = float(cfg["terminal_ignore_end_sec"])
    r2_threshold = float(cfg["r2_threshold"])
    cv_threshold = float(cfg["cv_threshold"])

    # 只使用 raw 点
    raw_mask = (traj_df["point_type"] == "raw").values
    t = traj_df["time_s"].values
    y = traj_df["y_m"].values

    raw_idx = np.where(raw_mask)[0]
    if len(raw_idx) < 10:
        return _not_found(f"raw 点不足 ({len(raw_idx)} < 10)")

    t_raw = t[raw_idx]
    y_raw = y[raw_idx]

    # 排除首尾段
    t_min, t_max = t_raw[0], t_raw[-1]
    span = t_max - t_min
    t_start = t_min + ignore_start
    t_end = t_max - ignore_end
    if t_end - t_start < window_sec:
        return _not_found(
            f"排除首尾后可用时长不足 ({t_end - t_start:.3f}s < {window_sec}s)"
        )

    # ---- 滑动窗口 y-t 线性拟合 ----
    dt_frame = np.median(np.diff(t_raw)) if len(t_raw) > 1 else 1/240
    window_frames = max(3, int(window_sec / dt_frame))

    candidates = []
    for i in range(len(t_raw) - window_frames + 1):
        t_win = t_raw[i:i + window_frames]
        y_win = y_raw[i:i + window_frames]

        vt, intercept, r2 = _fit_y_t(t_win, y_win)
        if vt is None:
            continue

        # Cv: 窗口内速度变异系数
        cv = _compute_cv(t_win, y_win)

        candidates.append({
            "start_idx": raw_idx[i],
            "end_idx": raw_idx[i + window_frames - 1],
            "start_time": t_win[0],
            "end_time": t_win[-1],
            "v_t": vt,
            "r2": r2,
            "cv": cv,
            "intercept": intercept,
            "n_points": window_frames,
        })

    if not candidates:
        return _not_found("滑动窗口拟合全部失败")

    if debug_callback:
        debug_callback(f"[DEBUG] 滑动窗口: {len(candidates)} 个候选, "
                       f"窗口 {window_frames} 帧 ({window_sec}s)")

    # ---- 筛选：连续 R² 合格窗口 → 候选群组 ----
    groups = _find_groups(candidates, r2_threshold)
    if not groups:
        if debug_callback:
            r2_max = max(c["r2"] for c in candidates)
            debug_callback(f"[DEBUG] 无群组通过 R²>{r2_threshold} (最大 R²={r2_max:.6f})")
        # 回退：放宽 R² 要求
        groups = _find_groups(candidates, max(0.95, r2_threshold * 0.98))
        if not groups:
            return _not_found("所有候选 R² 均过低")

    # ---- 从群组中选最优 ----
    best = _select_best_group(groups, candidates, cv_threshold)

    if debug_callback:
        debug_callback(
            f"[INFO] 终端速度: v_t={best['v_t']:.6f} m/s, "
            f"R²={best['r2']:.6f}, Cv={best['cv']:.4f}, "
            f"时段 {best['start_time']:.3f}s-{best['end_time']:.3f}s"
        )

    return {
        "found": True,
        "terminal_velocity_m_s": best["v_t"],
        "start_time_s": best["start_time"],
        "end_time_s": best["end_time"],
        "start_frame": int(best["start_idx"]),
        "end_frame": int(best["end_idx"]),
        "r2": best["r2"],
        "cv": best["cv"],
        "fit_intercept": best["intercept"],
        "method": "y_t_sliding_window",
        "window_velocities": [c["v_t"] for c in candidates],
        "candidates_table": candidates,
        "message": f"v_t={best['v_t']:.6f} m/s (R²={best['r2']:.4f})",
    }


def _fit_y_t(t: np.ndarray, y: np.ndarray) -> tuple:
    """y-t 线性拟合。

    y = v_t * t + intercept

    Returns:
        (slope, intercept, r2) 或 (None, None, None)
    """
    if len(t) < 3:
        return None, None, None
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None, None

    y_pred = slope * t + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(slope), float(intercept), float(r2)


def _compute_cv(t: np.ndarray, y: np.ndarray) -> float:
    """计算 y-t 窗口内速度变异系数 Cv。"""
    if len(t) < 3:
        return 1.0
    v = np.diff(y) / np.diff(t)
    v_mean = np.mean(v)
    if v_mean > 0:
        return float(np.std(v, ddof=1) / v_mean)
    return 1.0


def _find_groups(candidates: list, r2_threshold: float) -> list[dict]:
    """找出连续 R² 合格窗口组成的群组。"""
    groups = []
    current = []
    for c in candidates:
        if c["r2"] >= r2_threshold:
            current.append(c)
        else:
            if len(current) >= 2:
                groups.append({
                    "candidates": current,
                    "start_time": current[0]["start_time"],
                    "end_time": current[-1]["end_time"],
                    "count": len(current),
                })
            current = []
    if len(current) >= 2:
        groups.append({
            "candidates": current,
            "start_time": current[0]["start_time"],
            "end_time": current[-1]["end_time"],
            "count": len(current),
        })
    return groups


def _select_best_group(
    groups: list[dict],
    all_candidates: list[dict],
    cv_threshold: float,
) -> dict:
    """从群组中选最佳终端速度。

    评分：点数(40) + Cv 合格(30) + 时长(20) + R²(10)
    """
    scored = []
    for g in groups:
        group_cands = g["candidates"]
        duration = g["end_time"] - g["start_time"]
        count = g["count"]
        avg_r2 = np.mean([c["r2"] for c in group_cands])
        avg_cv = np.mean([c["cv"] for c in group_cands])
        cv_ok = avg_cv <= cv_threshold

        # 评分
        score = 0
        score += min(40, count)                     # 窗口数
        score += 30 if cv_ok else 0                 # Cv 合格
        score += min(20, duration * 10)             # 时长
        score += min(10, avg_r2 * 10)               # R²

        # 取群组中间窗口的参数
        mid = group_cands[len(group_cands) // 2]

        scored.append({
            "score": score,
            "v_t": mid["v_t"],
            "r2": avg_r2,
            "cv": avg_cv,
            "start_time": g["start_time"],
            "end_time": g["end_time"],
            "start_idx": group_cands[0]["start_idx"],
            "end_idx": group_cands[-1]["end_idx"],
            "intercept": mid["intercept"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0]


def _not_found(message: str) -> dict:
    return {
        "found": False,
        "terminal_velocity_m_s": None,
        "start_time_s": None,
        "end_time_s": None,
        "start_frame": None,
        "end_frame": None,
        "r2": None,
        "cv": None,
        "method": "none",
        "window_velocities": [],
        "candidates_table": [],
        "message": message,
    }
