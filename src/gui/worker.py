"""
后台工作线程

在 QThread 中运行完整的视频分析管线，避免 GUI 卡死。
"""

import os
import sys
import traceback
from PySide6.QtCore import QThread, Signal

# 确保能导入 src 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.tracking import process_video
from src.velocity import compute_velocity
from src.terminal_region import find_terminal_region
from src.viscosity import compute_viscosity
from src.report import generate_summary
from src.utils import load_config, ensure_output_dir, normalize_config_keys

# 新版管线
from src.pipeline import run_pipeline as run_new_pipeline


class AnalysisWorker(QThread):
    """后台分析线程。"""

    progress = Signal(str)          # 状态消息
    frame_processed = Signal(int, int)  # (current, total)
    finished = Signal(dict)         # 分析结果
    error = Signal(str)             # 错误消息

    def __init__(self, config: dict, video_path: str, output_dir: str, parent=None):
        super().__init__(parent)
        self._config = config
        self.final_config = None  # 在 run() 中设置归一化后的 config
        self._video_path = video_path
        self._output_dir = output_dir
        self._is_cancelled = False

    def run(self):
        try:
            ensure_output_dir(self._output_dir)

            # ★ 归一化配置并存为最终配置
            config = normalize_config_keys(self._config)
            self.final_config = config
            self.progress.emit(
                "[CONFIG] parameter sources: default=config.yaml, "
                "saved_config=config/settings.json, manual_override=UI, "
                "video_metadata=fps/resolution only"
            )
            self.progress.emit(
                "[INFO] 主算法入口: stable tracking.py + BallDetector; "
                "pipeline.py is experimental fallback when explicitly enabled"
            )

            # 保存配置副本
            import yaml
            config_copy_path = os.path.join(self._output_dir, "config_used.yaml")
            with open(config_copy_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            self.progress.emit("[INFO] 开始读取视频...")

            # ★ 新版管线（use_new_pipeline=True 时使用）
            if config.get("use_new_pipeline", False):
                self.progress.emit("[INFO] 使用新版管线 (pipeline.py)...")
                pipeline_result = run_new_pipeline(
                    self._video_path, config, self._output_dir,
                    debug_callback=lambda msg: self.progress.emit(msg),
                    progress_callback=lambda c, t: self.frame_processed.emit(c, t),
                )
                if self._is_cancelled:
                    return

                if not pipeline_result["success"]:
                    tr = pipeline_result.get("terminal_region", {})
                    self.error.emit(
                        f"新版管线分析失败。\n\n"
                        f"原因: {tr.get('message', '未知')}\n\n"
                        f"建议:\n"
                        f"  - 检查 ROI 是否正确框选小球下落通道\n"
                        f"  - 调整检测参数\n"
                        f"  - 设置 manual_start_frame/manual_end_frame 手动指定区间\n"
                        f"  - 或切换回旧版管线"
                    )
                    return

                traj_df = pipeline_result["traj_df"]
                velocity_df = pipeline_result["velocity_df"]
                terminal_region = pipeline_result["terminal_region"]
                viscosity_result = pipeline_result["viscosity_result"]
                track_interval = pipeline_result["track_interval"]

                if config.get("save_plots", True):
                    self.progress.emit("[INFO] 生成图表...")
                    from src.plotting import plot_all
                    plot_all(traj_df, velocity_df, terminal_region, self._output_dir)

                self.progress.emit("[INFO] 生成结果摘要...")
                try:
                    from src.report import generate_summary
                    generate_summary(
                        self._video_path, config, traj_df, velocity_df,
                        terminal_region, viscosity_result, self._output_dir,
                    )
                except Exception as e:
                    self.progress.emit(f"[WARN] 报告输出遇到可恢复错误: {e}")

                result = {
                    "traj_df": traj_df,
                    "traj_df_full": pipeline_result.get("traj_df_full", traj_df),
                    "velocity_df": velocity_df,
                    "terminal_region": terminal_region,
                    "viscosity_result": viscosity_result,
                    "output_dir": self._output_dir,
                    "analysis_start_frame": track_interval.start_frame,
                    "analysis_end_frame": track_interval.end_frame,
                    "analysis_total_frames": track_interval.total_frames,
                    "analysis_duration_s": track_interval.duration_s,
                    "interval_source": track_interval.interval_source,
                    "termination_reason": track_interval.termination_reason,
                    "candidates_summary": track_interval.candidates_summary,
                    "candidates_table": track_interval.candidates_table,
                    "video_total_frames": len(traj_df) if traj_df is not None else 0,
                    "valid_frames_in_analysis": track_interval.valid_frames_in_track,
                    "valid_rate_in_analysis": (track_interval.valid_frames_in_track /
                                                max(track_interval.total_frames, 1) * 100),
                    "valid_frames_all_video": pipeline_result.get("stats", {}).get("valid_raw_all", 0),
                    "valid_rate_all_video": pipeline_result.get("stats", {}).get("valid_raw_ratio", 0) * 100,
                }
                self.progress.emit(f"[INFO] 分析完成，结果已保存至 {self._output_dir}")
                self.finished.emit(result)
                return

            # ---- 旧版管线 (默认) ----
            proc_result = process_video(self._video_path, config)
            traj_df = proc_result["traj_df"]
            if self._is_cancelled:
                return

            csv_path = os.path.join(self._output_dir, "trajectory.csv")
            traj_df.to_csv(csv_path, index=False, float_format="%.8f", encoding="utf-8")
            self.progress.emit(f"[INFO] 轨迹已保存: {csv_path}")

            valid_count = proc_result["valid_frames"]
            total_count = proc_result["total_frames"]
            valid_ratio = proc_result["valid_ratio"] * 100

            # 显示各点类型数量
            if traj_df is not None and "point_type" in traj_df.columns:
                raw_n = int((traj_df["point_type"] == "raw").sum())
                pred_n = int((traj_df["point_type"] == "predicted").sum())
                interp_n = int((traj_df["point_type"] == "interpolated").sum())
                self.progress.emit(
                    f"[INFO] 点类型统计: raw={raw_n}  predicted={pred_n}  interpolated={interp_n}"
                )

            self.progress.emit(
                f"[INFO] 有效识别 {valid_count}/{total_count} 帧 "
                f"({valid_ratio:.1f}%)"
            )
            self.progress.emit(
                f"[INFO] 最长连续有效段: {proc_result['longest_valid_segment_frames']} 帧 "
                f"({proc_result['longest_valid_segment_sec']:.3f}s)"
            )
            if traj_df is not None and "point_type" in traj_df.columns:
                raw_n = int((traj_df["point_type"] == "raw").sum())
                pred_n = int((traj_df["point_type"] == "predicted").sum())
                interp_n = int((traj_df["point_type"] == "interpolated").sum())
                raw_ratio = raw_n / max(total_count, 1) * 100.0
                pred_ratio = pred_n / max(total_count, 1) * 100.0
                interp_ratio = interp_n / max(total_count, 1) * 100.0
                self.progress.emit(
                    f"[INFO] 识别构成: raw={raw_n} ({raw_ratio:.1f}%), "
                    f"predicted={pred_n} ({pred_ratio:.1f}%), "
                    f"interpolated={interp_n} ({interp_ratio:.1f}%)"
                )

            # 显示四层统计口径和拒绝原因
            rc_rate = proc_result.get("raw_candidate_rate", -1)
            sd_rate = proc_result.get("selected_detection_rate", -1)
            vr_rate = proc_result.get("valid_real_point_rate", -1)
            rr_counts = proc_result.get("reject_reason_counts", {})
            if rc_rate >= 0:
                self.progress.emit(
                    f"[INFO] 四层检测率:\n"
                    f"  raw_candidate_rate（BallDetector找到候选）= {rc_rate:.1f}%\n"
                    f"  selected_detection_rate（raw+predicted）= {sd_rate:.1f}%\n"
                    f"  valid_real_point_rate（仅raw点）= {vr_rate:.1f}%\n"
                    f"  类型: raw={raw_n}, predicted={pred_n}, "
                    f"interpolated={interp_n}"
                )
                if rr_counts:
                    top5 = sorted(rr_counts.items(), key=lambda x: -x[1])[:5]
                    rr_str = " | ".join(f"{k}x{v}" for k, v in top5)
                    self.progress.emit(
                        f"[INFO] 拒绝原因 Top5: {rr_str}"
                    )

            # 新版判据：不只看全视频识别率，而是看是否存在高质量的物理轨迹段
            has_quality = proc_result.get("has_quality_segment", False)
            seg_len = proc_result.get("longest_valid_segment_frames", 0)
            seg_r2 = proc_result.get("longest_segment_r2", 0.0)
            seg_cv = proc_result.get("longest_segment_cv", 1.0)

            if valid_count < 3:
                self.error.emit(
                    "有效识别帧数不足（< 3 帧）。\n\n"
                    "建议：\n"
                    "  - 调整 ROI 确保框住小球下落通道\n"
                    "  - 切换颜色模式（深色小球/浅色小球）\n"
                    "  - 调整面积范围（min_area_px / max_area_px）\n"
                    "  - 在首帧手动点击小球中心指定初始位置\n"
                    "  - 使用「测试当前帧识别」配合调试视图检查参数"
                )
                return

            if not has_quality and total_count > 20:
                self.progress.emit(
                    f"[WARN] 未找到满足物理质量的轨迹段\n"
                    f"  最长连续段: {seg_len} 帧, R²={seg_r2:.6f}, Cv={seg_cv:.4f}\n"
                    f"  期望: ≥{config.get('quality_min_frames', 80)} 帧, "
                    f"R²≥{config.get('quality_r2_threshold', 0.995)}, "
                    f"Cv<{config.get('quality_cv_threshold', 0.05)}\n"
                    f"  有效识别: {valid_count}/{total_count} ({valid_ratio:.1f}%)\n"
                    f"  但即使低识别率，仍将继续后续分析。"
                )
                if traj_df is not None and "rejection" in traj_df.columns:
                    rejected = traj_df.loc[
                        traj_df["rejection"].fillna("").astype(str) != "",
                        "rejection"
                    ].astype(str)
                    if len(rejected) > 0:
                        top_reasons = rejected.value_counts().head(5)
                        summary = " | ".join(
                            f"{reason} x{count}" for reason, count in top_reasons.items()
                        )
                        self.progress.emit(
                            f"[WARN] 高频拒绝原因: {summary}"
                        )
                # 不阻断，继续后续分析

            # ---- 检测有效轨迹区间（自动 + 手动优先级） ----
            from src.tracking import detect_track_interval

            video_total_frames = len(traj_df)
            raw_mask_full = (traj_df["point_type"] == "raw").values
            valid_raw_all = int(raw_mask_full.sum())

            roi = config.get("roi")
            track_info = detect_track_interval(
                traj_df, config, roi,
                debug_callback=lambda msg: self.progress.emit(msg),
            )

            interval_source = track_info.get("interval_source", "auto_track")
            analysis_start = track_info["track_start_frame"]
            analysis_end = track_info["track_end_frame"]

            # 日志：区间检测结果
            candidates_summary = track_info.get("candidates_summary", "")
            self.progress.emit(
                f"[INFO] 轨迹区间检测结果:\n"
                f"  interval_source = {interval_source}\n"
                f"  final_analysis_start_frame = {analysis_start}\n"
                f"  final_analysis_end_frame = {analysis_end}\n"
                f"  终止原因: {track_info['termination_reason']}\n"
                f"  区间时长: {track_info['track_duration_s']:.3f} s\n"
                f"{candidates_summary}"
            )
            all_segments = track_info.get("candidates_table_all", [])
            if all_segments:
                longest = max(all_segments, key=lambda c: c.get("duration_s", 0.0))
                reject_text = longest.get("reject_reasons", [])
                if isinstance(reject_text, list):
                    reject_text = " | ".join(reject_text)
                self.progress.emit(
                    f"[INFO] 最长候选段详情: "
                    f"frames={longest.get('start_frame')}~{longest.get('end_frame')}, "
                    f"duration={longest.get('duration_s', 0.0):.3f}s, "
                    f"raw={longest.get('raw_count', 0)}, "
                    f"predicted={longest.get('predicted_count', 0)}, "
                    f"interpolated={longest.get('interpolated_count', 0)}, "
                    f"raw_rate={longest.get('raw_rate', 0.0):.1%}, "
                    f"usable_after_ignore={longest.get('usable_after_ignore_s', 0.0):.3f}s, "
                    f"termination={longest.get('termination_reason', '')}"
                )
                if reject_text:
                    self.progress.emit(
                        f"[INFO] 最长候选段被拒原因: {reject_text}"
                    )

            # ★ 检查区间检测是否成功
            try:
                all_candidates = track_info.get("candidates_table_all", [])
                if all_candidates:
                    _fields = [
                        "start_frame", "end_frame", "duration_s",
                        "raw_count", "display_count", "predicted_count", "interpolated_count",
                        "raw_rate", "y_displacement_px", "x_std_px", "r2",
                        "usable_after_ignore_s", "termination_reason",
                        "rejected", "reject_reasons",
                    ]
                    seg_rows = []
                    for c in all_candidates:
                        row = {f: c.get(f, "") for f in _fields}
                        if isinstance(row["reject_reasons"], list):
                            row["reject_reasons"] = " | ".join(row["reject_reasons"])
                        seg_rows.append(row)
                    seg_df = __import__("pandas", fromlist=["DataFrame"]).DataFrame(seg_rows)
                    seg_csv = os.path.join(self._output_dir, "segment_debug.csv")
                    seg_df.to_csv(seg_csv, index=False, float_format="%.6f", encoding="utf-8")
                    self.progress.emit(f"[INFO] 候选段数据已保存: {seg_csv}")
            except Exception as e:
                self.progress.emit(f"[WARN] 导出 segment_debug.csv 失败: {e}")

            if not track_info.get("success", True):
                # 构造详细失败信息
                term_reason = track_info.get('termination_reason', '未知')
                extra_info = ""
                if "总轨迹时长不足" in term_reason or "自动段均不满足" in term_reason:
                    all_segs = track_info.get("candidates_table_all", [])
                    if all_segs:
                        # 找出各候选段的拒绝原因统计
                        reason_counter = {}
                        for s in all_segs:
                            rrs = s.get("reject_reasons", [])
                            if isinstance(rrs, list):
                                for rr in rrs:
                                    key = rr.split("(")[0].strip().split("=")[0].strip()
                                    if key:
                                        reason_counter[key] = reason_counter.get(key, 0) + 1
                        if reason_counter:
                            top_reasons = sorted(reason_counter.items(), key=lambda x: -x[1])
                            extra_info = "\n各段拒绝原因统计:\n"
                            for r, c in top_reasons[:5]:
                                extra_info += f"  - {r}: {c}次\n"
                    # 添加检测率信息
                    rc_rate = proc_result.get("raw_candidate_rate", -1)
                    sd_rate = proc_result.get("selected_detection_rate", -1)
                    vr_rate = proc_result.get("valid_real_point_rate", -1)
                    if rc_rate >= 0:
                        extra_info += (
                            f"\n检测率信息:\n"
                            f"  raw_candidate_rate (BallDetector找到候选): {rc_rate:.1f}%\n"
                            f"  selected_detection_rate (raw+predicted): {sd_rate:.1f}%\n"
                            f"  valid_real_point_rate (仅raw点): {vr_rate:.1f}%\n"
                        )

                self.error.emit(
                    f"轨迹区间检测失败，无法继续分析。\n\n"
                    f"原因:\n{term_reason}{extra_info}\n\n"
                    f"建议:\n"
                    f"  1. 检查 frame_debug_detailed.csv 查看逐帧拒绝原因\n"
                    f"  2. 检查 ROI 是否正确框选小球下落通道\n"
                    f"  3. 调整检测参数（面积范围、颜色模式等）\n"
                    f"  4. 在「识别参数」面板中设置 manual_start_frame / manual_end_frame\n"
                    f"     手动指定分析区间\n"
                    f"  5. 使用「测试当前帧识别」配合调试视图检查检测效果"
                )
                return

            # ---- 导出 segment_debug.csv（所有候选段详细指标） ----
            try:
                all_candidates = track_info.get("candidates_table_all", [])
                if all_candidates:
                    _fields = [
                        "start_frame", "end_frame", "duration_s",
                        "raw_count", "display_count", "predicted_count", "interpolated_count",
                        "raw_rate", "y_displacement_px", "x_std_px", "r2",
                        "usable_after_ignore_s", "termination_reason",
                        "rejected", "reject_reasons",
                    ]
                    seg_rows = []
                    for c in all_candidates:
                        row = {f: c.get(f, "") for f in _fields}
                        if isinstance(row["reject_reasons"], list):
                            row["reject_reasons"] = " | ".join(row["reject_reasons"])
                        seg_rows.append(row)
                    seg_df = __import__("pandas", fromlist=["DataFrame"]).DataFrame(seg_rows)
                    seg_csv = os.path.join(self._output_dir, "segment_debug.csv")
                    seg_df.to_csv(seg_csv, index=False, float_format="%.6f", encoding="utf-8")
                    self.progress.emit(f"[INFO] 候选段数据已保存: {seg_csv}")
            except Exception as e:
                self.progress.emit(f"[WARN] 导出 segment_debug.csv 失败: {e}")

            # 边界保护
            analysis_start = max(0, min(analysis_start, video_total_frames - 1))
            analysis_end = max(analysis_start, min(analysis_end, video_total_frames - 1))
            analysis_total_frames = analysis_end - analysis_start + 1

            # 切分分析区间数据
            df_analysis = traj_df.iloc[analysis_start:analysis_end + 1].copy()
            raw_mask_a = (df_analysis["point_type"] == "raw").values
            valid_raw_analysis = int(raw_mask_a.sum())

            # 日志输出完整统计
            self.progress.emit(
                f"[INFO] 统计口径:\n"
                f"  video_total_frames = {video_total_frames}\n"
                f"  interval_source = {interval_source}\n"
                f"  final_analysis_start_frame = {analysis_start}\n"
                f"  final_analysis_end_frame = {analysis_end}\n"
                f"  analysis_total_frames = {analysis_total_frames}\n"
                f"  区间内raw识别 = {valid_raw_analysis} / {analysis_total_frames}"
                f" ({valid_raw_analysis / analysis_total_frames * 100:.1f}%)\n"
                f"  全视频raw识别 = {valid_raw_all} / {video_total_frames}"
                f" ({valid_raw_all / video_total_frames * 100:.1f}%)\n"
                f"  终止原因: {track_info['termination_reason']}"
            )

            # ---- 2. 速度（使用分析区间数据） ----
            self.progress.emit("[INFO] 计算速度...")
            velocity_df = compute_velocity(df_analysis, config)
            if self._is_cancelled:
                return

            csv_v_path = os.path.join(self._output_dir, "velocity.csv")
            velocity_df.to_csv(csv_v_path, index=False, float_format="%.8f", encoding="utf-8")

            # ---- 3. 终端速度区（使用分析区间数据） ----
            self.progress.emit("[INFO] 自动判定终端速度区...")

            # 预检：分析区间时长是否足够
            analysis_duration = df_analysis["time_s"].values[-1] - df_analysis["time_s"].values[0]
            ignore_start = float(config.get("terminal_ignore_start_sec", 0.3))
            ignore_end = float(config.get("terminal_ignore_end_sec", 0.5))
            min_window = float(config.get("terminal_window_sec", 0.8))
            usable_duration = analysis_duration - ignore_start - ignore_end

            if usable_duration < min_window:
                self.progress.emit(
                    f"[WARN] 分析区间时长不足，跳过终端速度判定:\n"
                    f"  分析时长 = {analysis_duration:.3f}s\n"
                    f"  忽略首尾 = {ignore_start}s + {ignore_end}s\n"
                    f"  可用时长 = {usable_duration:.3f}s < 滑动窗口 = {min_window}s\n"
                    f"  → 请扩大分析区间（手动设置 manual_start_frame/manual_end_frame）"
                )
                terminal_region = {
                    "success": False, "found": False,
                    "v_t": None, "terminal_velocity_m_s": None,
                    "start_time_s": None, "end_time_s": None,
                    "start_frame": None, "end_frame": None,
                    "r2": None, "cv": None,
                    "method": "skipped",
                    "window_velocities": [],
                    "candidates_table": [],
                    "message": f"分析区间可用时长不足 ({usable_duration:.3f}s < {min_window}s)",
                }
            else:
                terminal_region = find_terminal_region(df_analysis, velocity_df, config)

            if self._is_cancelled:
                return

            if not terminal_region["found"]:
                self.progress.emit(f"[WARN] {terminal_region['message']}")
            else:
                self.progress.emit(
                    f"[INFO] 终端速度区: {terminal_region['start_time_s']:.3f}s"
                    f" — {terminal_region['end_time_s']:.3f}s, "
                    f"v_t = {terminal_region['terminal_velocity_m_s']:.6f} m/s"
                )

            # ---- 4. 黏度 ----
            viscosity_result = None
            if terminal_region["found"]:
                self.progress.emit("[INFO] 计算黏度...")
                try:
                    # ★ 从标准字段（mm）转换为 SI 单位传给黏度函数
                    viscosity_result = compute_viscosity(
                        terminal_velocity_m_s=terminal_region["terminal_velocity_m_s"],
                        ball_radius_m=config["ball_radius_mm"] / 1000.0,
                        ball_density_kg_m3=config["ball_density_kg_m3"],
                        liquid_density_kg_m3=config["liquid_density_kg_m3"],
                        cylinder_radius_m=config["cylinder_radius_mm"] / 1000.0,
                        liquid_height_m=config["liquid_height_mm"] / 1000.0,
                        g_m_s2=config.get("gravity_m_s2", 9.8),
                        temperature_c=config.get("temperature_c"),
                        enable_wall_correction=config.get("enable_wall_correction", True),
                    )
                    # 追加参考黏度到结果字典
                    ref_visc = config.get("reference_viscosity_pa_s")
                    viscosity_result["reference_viscosity_pa_s"] = ref_visc
                    viscosity_result["temperature_c"] = config.get("temperature_c", "?")

                    eta_final = viscosity_result["eta_final_pa_s"]
                    wall_tag = "壁面修正后" if config.get("enable_wall_correction", True) else "基础"
                    self.progress.emit(
                        f"[INFO] 黏度结果: η_basic={viscosity_result['eta_basic_pa_s']:.6f} Pa·s, "
                        f"η_final({wall_tag})={eta_final:.6f} Pa·s, "
                        f"Re = {viscosity_result['reynolds_number']:.4f}"
                    )
                except (ValueError, ZeroDivisionError) as e:
                    self.progress.emit(f"[ERROR] 黏度计算失败: {e}")

            # ---- 5. 图表（使用分析区间数据） ----
            if config.get("save_plots", True):
                self.progress.emit("[INFO] 生成图表...")
                from src.plotting import plot_all
                plot_all(df_analysis, velocity_df, terminal_region, self._output_dir)

            # ---- 6. 报告 ----
            self.progress.emit("[INFO] 生成结果摘要...")
            try:
                generate_summary(
                    self._video_path, config, df_analysis, velocity_df,
                    terminal_region, viscosity_result, self._output_dir,
                )
            except Exception as e:
                self.progress.emit(f"[WARN] 报告输出过程中遇到可恢复错误: {e}")

            result = {
                "traj_df": df_analysis,  # 仅分析区间内的轨迹
                "traj_df_full": traj_df,  # 全视频轨迹（用于全视频统计叠加显示）
                "velocity_df": velocity_df,
                "terminal_region": terminal_region,
                "viscosity_result": viscosity_result,
                "output_dir": self._output_dir,
                # 最终分析区间
                "analysis_start_frame": analysis_start,
                "analysis_end_frame": analysis_end,
                "analysis_total_frames": analysis_total_frames,
                "analysis_duration_s": track_info.get("track_duration_s", 0),
                "interval_source": interval_source,
                "termination_reason": track_info.get("termination_reason", ""),
                "candidates_summary": track_info.get("candidates_summary", ""),
                "candidates_table": track_info.get("candidates_table", []),
                # 统计
                "video_total_frames": video_total_frames,
                "valid_frames_in_analysis": valid_raw_analysis,
                "valid_rate_in_analysis": valid_raw_analysis / analysis_total_frames * 100 if analysis_total_frames > 0 else 0.0,
                "valid_frames_all_video": valid_raw_all,
                "valid_rate_all_video": valid_raw_all / video_total_frames * 100 if video_total_frames > 0 else 0.0,
                # 四层检测率
                "raw_candidate_rate": proc_result.get("raw_candidate_rate", -1),
                "selected_detection_rate": proc_result.get("selected_detection_rate", -1),
                "valid_real_point_rate": proc_result.get("valid_real_point_rate", -1),
                "reject_reason_counts": proc_result.get("reject_reason_counts", {}),
            }

            self.progress.emit(f"[INFO] 分析完成，结果已保存至 {self._output_dir}")
            self.finished.emit(result)

        except Exception as e:
            self.progress.emit(f"[ERROR] {str(e)}")
            self.error.emit(f"分析过程中发生错误:\n{str(e)}\n{traceback.format_exc()}")

    def cancel(self):
        """请求取消分析。"""
        self._is_cancelled = True
        self.progress.emit("[INFO] 用户请求停止分析...")
