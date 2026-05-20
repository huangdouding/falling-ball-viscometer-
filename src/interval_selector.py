"""
轨迹区间选择器

从 tracking.py 的 detect_track_interval 提取并简化。

核心原则：
  1. 手动区间优先（manual_start_frame / manual_end_frame）
  2. 自动检测失败时 NEVER 回退到 0-0
  3. 失败时返回明确的失败信息，让上层决定如何处理
  4. 所有候选段输出详细质量指标供调试

【区间来源】
  - manual_override: 用户手动指定
  - auto_track: 自动检测成功
  - (不再有 0-0 静默回退)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Callable

from src.trajectory_filter import assess_segment_quality

logger = logging.getLogger(__name__)


@dataclass
class TrackInterval:
    """轨迹区间选择结果。"""
    success: bool = False
    interval_source: str = "auto_track"

    # 区间边界
    start_frame: int = 0
    end_frame: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    total_frames: int = 0
    duration_s: float = 0.0

    # 统计
    valid_frames_in_track: int = 0
    termination_reason: str = ""

    # 候选段详情
    candidates_summary: str = ""
    candidates_table: list = field(default_factory=list)
    candidates_table_all: list = field(default_factory=list)


def select_interval(
    traj_df: pd.DataFrame,
    config: dict,
    roi: list[float] | None,
    *,
    debug_callback: Callable | None = None,
) -> TrackInterval:
    """选择分析区间。

    优先使用 manual_start_frame / manual_end_frame（用户手动指定）。
    自动检测失败时返回 success=False，由上层决定后续处理。
    """
    n = len(traj_df)
    if n == 0:
        return TrackInterval(
            success=False,
            termination_reason="空轨迹数据",
        )

    # ── 手动区间 ──
    manual_start = int(config.get("manual_start_frame", 0))
    manual_end = int(config.get("manual_end_frame", 0))

    if manual_start > 0 or manual_end > 0:
        return _select_manual(traj_df, manual_start, manual_end, config, debug_callback)

    # ── 自动区间 ──
    return _select_auto(traj_df, config, roi, debug_callback)


def _select_manual(
    traj_df: pd.DataFrame,
    manual_start: int,
    manual_end: int,
    config: dict,
    debug_callback: Callable | None,
) -> TrackInterval:
    """手动区间选择。"""
    n = len(traj_df)

    if manual_start <= 0:
        manual_start = 0
    if manual_end <= 0:
        manual_end = n - 1
    manual_start = max(0, min(manual_start, n - 1))
    manual_end = max(manual_start, min(manual_end, n - 1))

    seg_df = traj_df.iloc[manual_start:manual_end + 1]
    raw_in = int((seg_df["point_type"] == "raw").values.sum())
    pred_in = int((seg_df["point_type"] == "predicted").values.sum())
    interp_in = int((seg_df["point_type"] == "interpolated").values.sum())

    t_start = float(traj_df.iloc[manual_start]["time_s"])
    t_end = float(traj_df.iloc[manual_end]["time_s"])
    duration = t_end - t_start

    if debug_callback:
        debug_callback(
            f"[INFO] 手动分析区间: 帧 {manual_start}~{manual_end}, "
            f"时长 {duration:.3f}s, "
            f"raw={raw_in}, pred={pred_in}, interp={interp_in}"
        )

    return TrackInterval(
        success=True,
        interval_source="manual_override",
        start_frame=manual_start,
        end_frame=manual_end,
        start_time_s=t_start,
        end_time_s=t_end,
        total_frames=manual_end - manual_start + 1,
        duration_s=duration,
        valid_frames_in_track=raw_in,
        termination_reason="手动指定区间",
        candidates_summary=f"手动区间: 帧 {manual_start}~{manual_end}, 时长 {duration:.3f}s",
    )


def _select_auto(
    traj_df: pd.DataFrame,
    config: dict,
    roi: list[float] | None,
    debug_callback: Callable | None,
) -> TrackInterval:
    """自动区间选择。

    在所有候选轨迹段中，按质量评分选择最优段。
    无合格段时返回 success=False（不再回退到 0-0）。
    """
    # 参数
    lost_threshold = int(config.get("lost_threshold", 15))
    outside_threshold = int(config.get("outside_threshold", 10))
    stop_y_ratio = float(config.get("stop_y_ratio", 0.95))
    bottom_margin = float(config.get("bottom_margin_ratio", 0.03))
    ignore_start_s = float(config.get("terminal_ignore_start_sec", 0.3))
    ignore_end_s = float(config.get("terminal_ignore_end_sec", 0.5))
    terminal_window_sec = float(config.get("terminal_window_sec", 0.8))
    min_total_track_duration = ignore_start_s + ignore_end_s + terminal_window_sec

    # 质量门槛
    min_raw_count = int(config.get("min_track_valid_points", 30))
    min_raw_rate = float(config.get("min_track_raw_rate", 0.30))
    min_displacement_px = float(config.get("min_track_displacement_px", 30.0))
    min_r2_raw = float(config.get("quality_r2_threshold", 0.990))

    # ROI
    _roi = roi if roi is not None else config.get("roi")
    if _roi is None or len(_roi) != 4:
        return _fallback_full_video(traj_df)

    rx, ry, rw, rh = float(_roi[0]), float(_roi[1]), float(_roi[2]), float(_roi[3])
    use_roi_for_interval = bool(config.get("use_roi_for_interval", False))
    stop_y = ry + rh * stop_y_ratio if use_roi_for_interval else np.inf
    bottom_y = ry + rh * (1.0 - bottom_margin) if use_roi_for_interval else np.inf

    # 数据准备
    x_arr = traj_df["x_px"].values
    y_arr = traj_df["y_px"].values
    raw_mask = (traj_df["point_type"] == "raw").values
    pred_mask = (traj_df["point_type"] == "predicted").values
    interp_mask = (traj_df["point_type"] == "interpolated").values
    has_coord = np.isfinite(x_arr) & np.isfinite(y_arr)
    if use_roi_for_interval:
        in_roi = _compute_in_roi_mask(x_arr, y_arr, has_coord, rx, ry, rw, rh)
    else:
        in_roi = has_coord.copy()
    valid_in_roi = has_coord & in_roi

    # 找候选段
    candidates = _find_segments(
        raw_mask & in_roi, raw_mask, pred_mask, interp_mask,
        has_coord, in_roi, valid_in_roi,
        x_arr, y_arr, len(traj_df),
        lost_threshold, outside_threshold, stop_y, bottom_y,
    )

    if debug_callback:
        debug_callback(
            f"[DEBUG] 全视频扫描: 共发现 {len(candidates)} 个候选轨迹段"
        )

    # 评估每个候选段
    for c in candidates:
        _evaluate_candidate(c, traj_df, raw_mask, pred_mask, interp_mask,
                           x_arr, y_arr, in_roi,
                           ignore_start_s, ignore_end_s, terminal_window_sec,
                           min_total_track_duration, min_raw_count, min_raw_rate,
                           min_displacement_px, min_r2_raw,
                           debug_callback)

    valid_candidates = [c for c in candidates if not c.get("rejected", False)]

    # ── 无合格段：明确失败 ──
    if not valid_candidates:
        reason = _build_failure_reason(candidates, min_total_track_duration)
        if debug_callback:
            debug_callback(reason)

        return TrackInterval(
            success=False,
            interval_source="auto_track",
            termination_reason=reason,
            candidates_summary=_build_summary(candidates, valid_candidates),
            candidates_table=[],
            candidates_table_all=[dict(c) for c in candidates],
        )

    # ── 选最佳段 ──
    best = _select_best(valid_candidates)

    if debug_callback:
        debug_callback(
            f"[INFO] 选中最佳段: 帧 {best['start_frame']}~{best['end_frame']}, "
            f"时长 {best['duration_s']:.3f}s, raw={best['raw_count']}, "
            f"R²={best['r2']:.4f}, 得分 {best.get('score', 0):.1f}"
        )

    return TrackInterval(
        success=True,
        interval_source="auto_track",
        start_frame=best["start_frame"],
        end_frame=best["end_frame"],
        start_time_s=float(traj_df.iloc[best["start_frame"]]["time_s"]),
        end_time_s=float(traj_df.iloc[best["end_frame"]]["time_s"]),
        total_frames=best["end_frame"] - best["start_frame"] + 1,
        duration_s=best["duration_s"],
        valid_frames_in_track=best["raw_count"],
        termination_reason=best.get("termination_reason", "自动检测"),
        candidates_summary=_build_summary(candidates, valid_candidates),
        candidates_table=[dict(c) for c in valid_candidates],
        candidates_table_all=[dict(c) for c in candidates],
    )


# ── 内部函数 ──

def _compute_in_roi_mask(x_arr, y_arr, has_coord, rx, ry, rw, rh):
    """计算在 ROI 内的掩码。"""
    in_roi = np.zeros(len(x_arr), dtype=bool)
    if has_coord.any():
        in_roi[has_coord] = (
            (x_arr[has_coord] >= rx) & (x_arr[has_coord] <= rx + rw) &
            (y_arr[has_coord] >= ry) & (y_arr[has_coord] <= ry + rh)
        )
    return in_roi


def _find_segments(
    raw_in_roi, raw_mask, pred_mask, interp_mask,
    has_coord, in_roi, valid_in_roi,
    x_arr, y_arr, n,
    lost_threshold, outside_threshold, stop_y, bottom_y,
) -> list:
    """扫描全视频找所有候选轨迹段。"""
    candidates = []
    seg_start = seg_end = None
    seg_raw_count = seg_display_count = 0
    gap_count = outside_count = 0

    for i in range(n):
        if valid_in_roi[i]:
            if seg_start is None:
                seg_start = i
                seg_raw_count = seg_display_count = 0
            seg_end = i
            seg_display_count += 1
            if raw_in_roi[i]:
                seg_raw_count += 1
            gap_count = 0
            outside_count = 0
        else:
            if seg_start is None:
                continue
            gap_count += 1
            if has_coord[i] and not in_roi[i]:
                outside_count += 1

            # 终止条件
            term_reason = None
            if outside_count >= outside_threshold:
                term_reason = f"离开ROI(连续{outside_count}帧)"
            elif has_coord[i] and not np.isnan(y_arr[i]) and y_arr[i] > stop_y:
                term_reason = f"超过终止线(y={y_arr[i]:.0f})"
            elif has_coord[i] and not np.isnan(y_arr[i]) and y_arr[i] > bottom_y:
                term_reason = f"接近底部(y={y_arr[i]:.0f})"
            elif gap_count > lost_threshold:
                term_reason = f"连续丢失超过{lost_threshold}帧"

            if term_reason:
                candidates.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "termination_reason": term_reason,
                    "raw_count": seg_raw_count,
                    "display_count": seg_display_count,
                })
                seg_start = seg_end = None
                seg_raw_count = seg_display_count = 0
                gap_count = outside_count = 0

    if seg_start is not None and seg_end is not None:
        candidates.append({
            "start_frame": seg_start, "end_frame": seg_end,
            "termination_reason": "到达视频末帧",
            "raw_count": seg_raw_count,
            "display_count": seg_display_count,
        })

    return candidates


def _evaluate_candidate(c, traj_df, raw_mask, pred_mask, interp_mask,
                        x_arr, y_arr, in_roi,
                        ignore_start_s, ignore_end_s, terminal_window_sec,
                        min_total_track_duration, min_raw_count, min_raw_rate,
                        min_displacement_px, min_r2_raw,
                        debug_callback):
    """评估单个候选段的质量。"""
    seg_start, seg_end = c["start_frame"], c["end_frame"]
    seg_len = seg_end - seg_start + 1
    seg_ts = float(traj_df.iloc[seg_start]["time_s"])
    seg_te = float(traj_df.iloc[seg_end]["time_s"])
    seg_dur = seg_te - seg_ts

    # 统计
    seg_raw = int(raw_mask[seg_start:seg_end + 1].sum())
    seg_pred = int((pred_mask & in_roi)[seg_start:seg_end + 1].sum())
    seg_interp = int((interp_mask & in_roi)[seg_start:seg_end + 1].sum())
    seg_display = seg_raw + seg_pred + seg_interp
    raw_rate = seg_raw / seg_len if seg_len > 0 else 0.0

    # y 位移（仅 raw）
    ys = y_arr[seg_start:seg_end + 1]
    raw_seg = raw_mask[seg_start:seg_end + 1]
    raw_ys = ys[raw_seg]
    y_disp = abs(float(raw_ys[-1] - raw_ys[0])) if len(raw_ys) >= 2 else 0.0

    # x 标准差
    xs = x_arr[seg_start:seg_end + 1]
    raw_xs = xs[raw_seg]
    x_std = float(np.std(raw_xs)) if len(raw_xs) >= 3 else 0.0

    # y-t 拟合
    seg_df = traj_df.iloc[seg_start:seg_end + 1]
    r2, _cv = assess_segment_quality(
        seg_df["time_s"].values,
        seg_df["y_m"].values,
        seg_df["point_type"].values if "point_type" in seg_df.columns else None,
    )
    r2 = r2 if r2 is not None else 0.0

    usable = seg_dur - ignore_start_s - ignore_end_s

    # 拒绝判定
    reject_reasons = []
    if seg_dur < min_total_track_duration:
        reject_reasons.append(
            f"总轨迹时长不足({seg_dur:.3f}s < {min_total_track_duration:.1f}s"
            f" = ignore_start+ignore_end+window)"
        )
    elif usable < terminal_window_sec:
        reject_reasons.append(
            f"排除起止段后可用时长不足({usable:.3f}s < {terminal_window_sec}s)")
    if seg_raw < min_raw_count:
        reject_reasons.append(f"raw点数不足({seg_raw} < {min_raw_count})")
    if raw_rate < min_raw_rate:
        reject_reasons.append(f"raw率过低({raw_rate:.1%} < {min_raw_rate:.0%})")
    if y_disp < min_displacement_px:
        reject_reasons.append(f"位移过小({y_disp:.0f}px < {min_displacement_px:.0f}px)")
    if r2 < min_r2_raw and seg_raw >= 10:
        reject_reasons.append(f"R²过低({r2:.4f} < {min_r2_raw})")

    c.update({
        "duration_s": seg_dur,
        "display_count": seg_display,
        "raw_count": seg_raw,
        "predicted_count": seg_pred,
        "interpolated_count": seg_interp,
        "raw_rate": raw_rate,
        "y_displacement_px": y_disp,
        "x_std_px": x_std,
        "r2": r2,
        "usable_after_ignore_s": usable,
        "rejected": len(reject_reasons) > 0,
        "reject_reasons": reject_reasons,
    })

    if debug_callback:
        status = "[拒绝]" if c["rejected"] else "[通过]"
        debug_callback(
            f"  段 帧{seg_start}~{seg_end}: "
            f"时长={seg_dur:.3f}s, raw={seg_raw}, "
            f"raw_rate={raw_rate:.1%}, 位移={y_disp:.0f}px, "
            f"R²={r2:.4f}, 可用={usable:.3f}s {status}"
        )
        for rr in reject_reasons:
            debug_callback(f"    -> 拒绝: {rr}")


def _select_best(valid_candidates: list) -> dict:
    """从多个合格段中选择最优。"""
    if len(valid_candidates) == 1:
        best = valid_candidates[0]
        best["score"] = 100.0
        return best

    max_dur = max(c["duration_s"] for c in valid_candidates) or 0.001
    max_raw = max(c["raw_count"] for c in valid_candidates) or 1
    max_rate = max(c["raw_rate"] for c in valid_candidates) or 0.001
    max_r2 = max(c["r2"] for c in valid_candidates) or 0.001
    max_disp = max(c["y_displacement_px"] for c in valid_candidates) or 1.0
    max_xstd = max(c["x_std_px"] for c in valid_candidates) or 1.0
    min_xstd = min(c["x_std_px"] for c in valid_candidates) or 0.0

    for c in valid_candidates:
        dur_score = c["duration_s"] / max_dur * 35
        raw_score = c["raw_count"] / max_raw * 30
        rate_score = c["raw_rate"] / max_rate * 15
        r2_score = c["r2"] / max_r2 * 10
        disp_score = c["y_displacement_px"] / max_disp * 5
        x_range = max_xstd - min_xstd if max_xstd > min_xstd else 1.0
        x_score = (1.0 - (c["x_std_px"] - min_xstd) / x_range) * 5
        c["score"] = dur_score + raw_score + rate_score + r2_score + disp_score + x_score

    return max(valid_candidates, key=lambda c: c["score"])


def _fallback_full_video(traj_df: pd.DataFrame) -> TrackInterval:
    """无 ROI 时使用全视频首末 raw 帧。"""
    raw_mask = (traj_df["point_type"] == "raw").values
    raw_frames = np.where(raw_mask)[0]
    if len(raw_frames) == 0:
        return TrackInterval(success=False, termination_reason="全视频无 raw 帧")

    return TrackInterval(
        success=True,
        interval_source="auto_track",
        start_frame=int(raw_frames[0]),
        end_frame=int(raw_frames[-1]),
        start_time_s=float(traj_df.iloc[raw_frames[0]]["time_s"]),
        end_time_s=float(traj_df.iloc[raw_frames[-1]]["time_s"]),
        total_frames=int(raw_frames[-1] - raw_frames[0] + 1),
        duration_s=float(traj_df.iloc[raw_frames[-1]]["time_s"] - traj_df.iloc[raw_frames[0]]["time_s"]),
        valid_frames_in_track=len(raw_frames),
        termination_reason="无ROI（全视频首末raw帧）",
    )


def _build_failure_reason(candidates: list, min_total_dur: float) -> str:
    """构造失败原因文本，包括各段的具体筛选拒绝原因。"""
    if not candidates:
        return "未发现任何候选轨迹段，请检查 ROI 设置和检测参数"

    longest = max(candidates, key=lambda c: c.get("duration_s", 0))
    rej = longest.get("reject_reasons", [])
    rej_str = ""
    if rej:
        rej_str = "; " + "; ".join(rej if isinstance(rej, list) else [rej])

    return (
        f"自动段均不满足质量要求: "
        f"最长段={longest['duration_s']:.3f}s"
        f"(需≥{min_total_dur:.1f}s 总轨迹时长)"
        f"{rej_str}"
    )


def _build_summary(all_candidates: list, valid_candidates: list) -> str:
    """构造候选段摘要文本。"""
    valid_keys = {(c["start_frame"], c["end_frame"]) for c in valid_candidates}
    lines = [
        f"全视频共发现 {len(all_candidates)} 个候选轨迹段, "
        f"其中 {len(valid_candidates)} 个通过质量筛选:"
    ]
    for i, c in enumerate(all_candidates):
        is_valid = (c["start_frame"], c["end_frame"]) in valid_keys
        tag = "[通过]" if is_valid else "[拒绝]"
        lines.append(
            f"  段{i+1} {tag}: 帧{c['start_frame']}~{c['end_frame']}, "
            f"时长={c.get('duration_s', 0):.3f}s, "
            f"raw={c.get('raw_count', 0)}, "
            f"R²={c.get('r2', 0):.4f}, "
            f"位移={c.get('y_displacement_px', 0):.0f}px, "
            f"终止: {c.get('termination_reason', '')}"
        )
        for rr in c.get("reject_reasons", []):
            lines.append(f"        {rr}")
    return "\n".join(lines)
