"""
速度计算模块（v3 — 滑动窗口 y-t 线性拟合，仅用于速度图显示）

重要：本模块的输出 velocity_df 仅供 GUI 速度图和 debug 使用。
★ 最终终端速度 v_terminal_fit 来自 terminal_region.py 的终端区间 y-t 整体线性拟合。
★ 黏度计算始终使用 terminal_region.y-t 拟合斜率，不依赖本模块。

方向约定：
  - 图像 y 轴向下为正
  - 小球下落时 y_m 增大 → dy/dt 为正
  - 因此 v = dy/dt 即正值为下落速度

三种速度类型（互不干扰）：
  1. v_frame_diff (v_m_s):          逐帧中心差分，1px抖动→~0.028 m/s误差，仅 debug
  2. v_local_window (smooth_v_m_s):  滑动窗口 y-t 拟合，用于速度图趋势参考
  3. v_terminal_fit:                 终端区间 y-t 整体线性拟合斜率，★ 黏度计算依据
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_velocity(traj_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """计算速度（v3 滑动窗口 y-t 线性拟合，仅用于显示和 debug）。

    ★ 最终黏度使用 terminal_region 的终端区间 y-t 整体拟合斜率，
      不依赖本模块的输出值。

    参数：
        traj_df: 轨迹 DataFrame（需含 time_s, y_m, valid, point_type）
        config: 配置字典
            - velocity_window_sec: 速度拟合窗口时长（秒），默认 0.50

    返回：
        velocity_df: 包含以下列的 DataFrame
            time_s: float              — 时间
            v_m_s: float               — 逐帧中心差分速度（仅 debug 参考，不可靠）
            smooth_v_m_s: float        — 滑动窗口 y-t 拟合速度（速度图趋势参考）
            v_fit_r2: float            — 窗口拟合 R²（质量控制）
            valid: bool
            point_type: str
    """
    use_for_velocity = traj_df["point_type"].isin(["raw", "interpolated"]).values \
        if "point_type" in traj_df.columns else traj_df["valid"].values
    valid_mask = use_for_velocity & traj_df["valid"].values

    time = traj_df["time_s"].values
    y = traj_df["y_m"].values

    n = len(traj_df)
    v_raw = np.full(n, np.nan)
    v_fit = np.full(n, np.nan)
    v_fit_r2 = np.full(n, np.nan)
    valid_out = np.full(n, False, dtype=bool)

    if valid_mask.sum() < 3:
        logger.warning("有效数据点少于 3 个，无法计算速度。")
        return pd.DataFrame({
            "time_s": time,
            "v_m_s": v_raw,
            "smooth_v_m_s": v_fit,
            "v_fit_r2": v_fit_r2,
            "valid": valid_out,
            "point_type": traj_df.get("point_type", None),
        })

    # ---- 估计 fps ----
    dt = np.diff(time[valid_mask])
    dt_valid = dt[dt > 0]
    if len(dt_valid) > 0:
        fps = 1.0 / np.median(dt_valid)
    else:
        fps = config.get("manual_fps", 240.0)

    # ---- 窗口大小 ----
    window_sec = config.get("velocity_window_sec", 0.50)
    half_window = max(2, int(round(window_sec * fps / 2.0)))
    logger.info("速度计算: 滑动窗口 %d 帧 (%.3f s @ %.1f fps)",
                half_window * 2 + 1, window_sec, fps)

    # ---- 只在有效点的连续段上计算速度 ----
    segments = _find_valid_segments(valid_mask)

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < 3:
            continue

        idx = np.arange(seg_start, seg_end)
        t_seg = time[idx]
        y_seg = y[idx]

        # A. 逐帧中心差分（保留作 debug 参考）
        v_seg_raw = np.full(len(idx), np.nan)
        for i in range(len(idx)):
            if i == 0:
                dt0 = t_seg[i + 1] - t_seg[i]
                if dt0 > 0:
                    v_seg_raw[i] = (y_seg[i + 1] - y_seg[i]) / dt0
            elif i == len(idx) - 1:
                dt0 = t_seg[i] - t_seg[i - 1]
                if dt0 > 0:
                    v_seg_raw[i] = (y_seg[i] - y_seg[i - 1]) / dt0
            else:
                dt0 = t_seg[i + 1] - t_seg[i - 1]
                if dt0 > 0:
                    v_seg_raw[i] = (y_seg[i + 1] - y_seg[i - 1]) / dt0

        v_raw[idx] = v_seg_raw

        # B. 滑动窗口 y-t 线性拟合速度（主要结果）
        v_seg_fit = np.full(len(idx), np.nan)
        r2_seg = np.full(len(idx), np.nan)
        for i in range(len(idx)):
            win_start = max(0, i - half_window)
            win_end = min(len(idx), i + half_window + 1)
            win_n = win_end - win_start
            if win_n < 3:
                continue

            t_win = t_seg[win_start:win_end]
            y_win = y_seg[win_start:win_end]
            finite = np.isfinite(t_win) & np.isfinite(y_win)
            if finite.sum() < 3:
                continue

            slope, r2 = _linear_fit_slope_r2(t_win[finite], y_win[finite])
            if slope is not None:
                v_seg_fit[i] = slope
                r2_seg[i] = r2

        v_fit[idx] = v_seg_fit
        v_fit_r2[idx] = r2_seg
        valid_out[idx] = True

    pt_col = traj_df["point_type"].values if "point_type" in traj_df.columns else None
    velocity_df = pd.DataFrame({
        "time_s": time,
        "v_m_s": v_raw,
        "smooth_v_m_s": v_fit,
        "v_fit_r2": v_fit_r2,
        "valid": valid_out,
        "point_type": pt_col,
    })

    n_valid = valid_out.sum()
    if n_valid > 0:
        v_mean = np.nanmean(np.abs(v_fit[valid_out]))
        logger.info("速度计算完成: %d 个有效速度点, 平均窗口拟合速度 %.4f m/s", n_valid, v_mean)

    return velocity_df


def _linear_fit_slope_r2(t: np.ndarray, y: np.ndarray) -> tuple:
    """对 (t, y) 做最小二乘线性拟合，返回 (slope, r2) 或 (None, None)。"""
    if len(t) < 3:
        return None, None
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None

    y_pred = slope * t + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def _find_valid_segments(valid_mask: np.ndarray) -> list:
    """找到连续有效段的 (start, end) 列表（左闭右开）。"""
    segments = []
    i = 0
    n = len(valid_mask)
    while i < n:
        if valid_mask[i]:
            j = i
            while j < n and valid_mask[j]:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1
    return segments
