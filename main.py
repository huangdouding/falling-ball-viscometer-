#!/usr/bin/env python3
"""
落球法液体黏度自动测量系统 — 主入口

基于机器视觉自动追踪与误差修正的落球法液体黏度测量实验改进。

用法：
    python main.py --config config.yaml

工作流程：
  视频输入 → 逐帧读取 → OpenCV 识别球心 → 坐标换算
  → y-t 轨迹 → v-t 速度 → 自动判稳 → 终端速度 v_t
  → 黏度计算（理想+修正） → 输出 CSV/图表/报告
"""

import os
import sys
import argparse
import logging

import pandas as pd

# 确保 src 模块可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    setup_logging,
    load_config,
    check_required_params,
    ensure_output_dir,
    normalize_config_keys,
)
from src.tracking import process_video
from src.velocity import compute_velocity
from src.terminal_region import find_terminal_region
from src.viscosity import compute_viscosity
from src.plotting import plot_all
from src.report import generate_summary

logger = logging.getLogger(__name__)


def _safe_print(text: str):
    """安全打印到控制台，GBK 环境下替换无法编码的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _choose_video_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""

    root = tk.Tk()
    root.withdraw()
    root.update()
    video_path = filedialog.askopenfilename(
        title="选择实验视频",
        filetypes=[
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
            ("所有文件", "*.*"),
        ],
    )
    root.destroy()
    return video_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="落球法液体黏度自动测量系统 — 基于 OpenCV 自动追踪"
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(script_dir, "config.yaml"),
        help="配置文件路径 (默认: main.py 同目录下的 config.yaml)"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="视频文件路径；可把视频拖到脚本/快捷方式上自动传入"
    )
    parser.add_argument(
        "--video", "-v",
        dest="video_option",
        help="视频文件路径，优先级高于 config.yaml"
    )
    parser.add_argument(
        "--new-pipeline", "-n",
        action="store_true",
        help="use new pipeline (pipeline.py)"
    )
    args = parser.parse_args()

    # 1. 初始化
    setup_logging()
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)

    _safe_print("=" * 60)
    _safe_print("  落球法液体黏度自动测量系统")
    _safe_print("  基于机器视觉自动追踪与误差修正")
    _safe_print("=" * 60)

    # 2. 读取配置
    if not os.path.exists(config_path):
        _safe_print(f"\n[错误] 配置文件不存在: {config_path}")
        _safe_print("请确保 config.yaml 位于当前目录。")
        sys.exit(1)

    config = load_config(config_path)
    config = normalize_config_keys(config)
    logger.info(
        "parameter sources: default/config=%s, video_metadata=fps/resolution only, "
        "manual_override=CLI arguments; physical parameters are not overwritten by video import",
        config_path,
    )
    logger.info(
        "main algorithm entry: stable tracking.py + BallDetector; "
        "pipeline.py is experimental fallback via --new-pipeline"
    )
    video_path = args.video_option or args.video_path or config.get("video_path")
    if not video_path:
        _safe_print("\n请选择要处理的视频文件...")
        video_path = _choose_video_file()
    if isinstance(video_path, str) and video_path:
        config["video_path"] = _resolve_path(video_path, config_dir)
    output_dir = _resolve_path(config.get("output_dir", "data/results"), config_dir)
    config["output_dir"] = output_dir

    # 3. 检查必填参数（含视频路径是否存在）
    if not check_required_params(config):
        _safe_print("\n上述问题解决后可重新运行: python main.py --config config.yaml")
        sys.exit(1)

    # 4. 准备输出目录
    ensure_output_dir(output_dir)

    # 复制配置文件到输出目录（供追溯）
    import shutil
    shutil.copy2(config_path, os.path.join(output_dir, "config_copy.yaml"))

    video_path = config["video_path"]
    _safe_print(f"\n[1/7] 读取视频: {video_path}")
    logger.info("开始处理视频: %s", video_path)

    # -- new-pipeline --
    if args.new_pipeline:
        _safe_print("  [use new pipeline (pipeline.py)]")
        try:
            from src.pipeline import run_pipeline
            result = run_pipeline(
                video_path, config, output_dir,
                debug_callback=lambda msg: _safe_print("  " + str(msg)),
            )
        except Exception as e:
            logger.error("new pipeline failed: %s", str(e))
            _safe_print("\n[ERROR] new pipeline failed: " + str(e))
            sys.exit(1)
        if result["success"]:
            _safe_print("\n[DONE] analysis ok, results saved to " + output_dir)
        else:
            tr_msg = result.get("terminal_region", {}).get("message", "unknown")
            _safe_print("\n[WARN] pipeline finished but: " + tr_msg)
        return

    # -- old pipeline (default) --
    try:
        proc_result = process_video(video_path, config)
    except Exception as e:
        logger.error("视频处理失败: %s", str(e))
        _safe_print(f"\n[错误] 视频处理失败: {e}")
        sys.exit(1)

    traj_df = proc_result["traj_df"]
    valid_count = proc_result["valid_frames"]
    total_frames = proc_result["total_frames"]
    valid_ratio_val = proc_result["valid_ratio"]

    # 保存轨迹 CSV
    csv_path = os.path.join(output_dir, "trajectory.csv")
    traj_df.to_csv(csv_path, index=False, float_format="%.8f")
    logger.info("轨迹数据已保存: %s", csv_path)

    # 6. 检查是否识别到有效点
    if valid_count < 2:
        _safe_print(f"\n[2/7] 追踪结果: 有效识别点仅 {valid_count} 个")
        _safe_print(
            "\n[警告] 未检测到足够的小球轨迹点。\n"
            "  可能原因:\n"
            "    - 阈值参数不合适 (threshold_value / threshold_mode)\n"
            "    - ROI 区域不正确\n"
            "    - 面积范围 (min_area_px / max_area_px) 不匹配\n"
            "    - 小球与背景对比度不足\n"
            "    - 光照条件需要调整\n"
            "  建议: 调整 config.yaml 中的检测参数后重试。"
        )
        if config.get("save_plots", True):
            from src.plotting import plot_trajectory
            plot_trajectory(traj_df, {"found": False}, output_dir)
        generate_summary(
            video_path, config, traj_df,
            None, {"found": False, "message": "未检测到小球。"},
            None, output_dir
        )
        sys.exit(0)

    _safe_print(f"\n[2/7] 追踪完成: 有效识别 {valid_count}/{total_frames} 帧 "
          f"({100.0*valid_ratio_val:.1f}%)")
    _safe_print(f"  最长连续有效段: {proc_result['longest_valid_segment_frames']} 帧"
          f" ({proc_result['longest_valid_segment_sec']:.3f}s)")

    # 7. 计算速度
    _safe_print("\n[3/7] 计算速度...")
    velocity_df = compute_velocity(traj_df, config)

    csv_v_path = os.path.join(output_dir, "velocity.csv")
    velocity_df.to_csv(csv_v_path, index=False, float_format="%.8f")
    logger.info("速度数据已保存: %s", csv_v_path)

    # 8. 寻找终端速度区
    _safe_print("\n[4/7] 自动判定终端速度区...")
    terminal_region = find_terminal_region(traj_df, velocity_df, config)

    if not terminal_region["found"]:
        _safe_print(f"\n  {terminal_region['message']}")
    else:
        _safe_print(f"\n  终端速度区: {terminal_region['start_time_s']:.3f}s"
              f" — {terminal_region['end_time_s']:.3f}s")
        _safe_print(f"  终端速度 v_t (y-t 线性拟合斜率) = {terminal_region['terminal_velocity_m_s']:.6f} m/s")
        _safe_print(f"  拟合 R² = {terminal_region['r2']:.6f}")
        _safe_print(f"  速度变异系数 Cv = {terminal_region['cv']:.4f}")

    # 保存终端速度区候选窗口表
    candidates_table = terminal_region.get("candidates_table", [])
    if candidates_table:
        cand_path = os.path.join(output_dir, "terminal_candidates.csv")
        pd.DataFrame(candidates_table).to_csv(cand_path, index=False)
        logger.info("终端速度候选窗口表已保存: %s", cand_path)
        _safe_print(f"  候选窗口: {len(candidates_table)} 个 (已保存至 terminal_candidates.csv)")

    # 9. 计算黏度
    _safe_print("\n[5/7] 计算黏度...")
    viscosity_result = None
    if terminal_region["found"]:
        try:
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
                enable_reynolds_correction=config.get("enable_reynolds_correction", True),
            )
            _safe_print(f"  理想黏度 η_basic: {viscosity_result['eta_basic_pa_s']:.6f} Pa·s")
            wf = viscosity_result.get('wall_correction_factor', 1.0)
            if wf > 1.0:
                _safe_print(f"  壁面修正因子:     {wf:.4f}")
            rf = viscosity_result['reynolds_factor']
            if rf > 1.0:
                _safe_print(f"  雷诺数修正因子:   {rf:.4f}")
            _safe_print(f"  最终黏度 η_final: {viscosity_result['eta_final_pa_s']:.6f} Pa·s")
            _safe_print(f"  雷诺数 Re:        {viscosity_result['reynolds_number']:.4f}")

            if viscosity_result.get("warnings"):
                for w in viscosity_result["warnings"]:
                    _safe_print(f"  [注意] {w.split(chr(10))[0]}")

        except (ValueError, ZeroDivisionError) as e:
            _safe_print(f"  [错误] 黏度计算失败: {e}")
            viscosity_result = None
    else:
        _safe_print("  未找到终端速度区，跳过黏度计算。")

    # 10. 生成图表
    _safe_print("\n[6/7] 生成图表...")
    if config.get("save_plots", True):
        plot_all(traj_df, velocity_df, terminal_region, output_dir)

    # 11. 生成汇总报告
    _safe_print("\n[7/7] 生成结果摘要...")
    generate_summary(
        video_path, config, traj_df, velocity_df,
        terminal_region, viscosity_result, output_dir,
    )

    # 12. 完成
    _safe_print(f"\n所有输出已保存至: {os.path.abspath(output_dir)}/")
    _safe_print("完成。")


if __name__ == "__main__":
    main()
