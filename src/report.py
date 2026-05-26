"""
报告输出模块

生成 result_summary.txt，汇总视频信息、检测结果、黏度计算结果。
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_print_report(text: str):
    """安全打印到控制台，GBK 环境下替换无法编码的字符为 '?'。"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 替换当前控制台编码无法处理的字符
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)
        logger.warning("控制台输出包含当前编码(%s)不支持的字符，已替换。", enc)


def generate_summary(
    video_path: str,
    config: dict,
    traj_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    terminal_region: dict,
    viscosity_result: Optional[dict],
    output_dir: str,
):
    """生成结果摘要文本文件。"""
    lines = []
    sep = "=" * 65

    lines.append(sep)
    lines.append("  落球法液体黏度自动测量系统 — 结果摘要")
    lines.append(sep)
    lines.append("")

    # ---- 视频信息 ----
    lines.append("【视频信息】")
    lines.append(f"  视频路径:            {video_path}")
    lines.append(f"  总帧数:              {len(traj_df)}")
    lines.append(f"  有效识别帧数:        {int(traj_df['valid'].sum())}")
    valid_ratio = 100.0 * traj_df["valid"].sum() / len(traj_df) if len(traj_df) > 0 else 0.0
    lines.append(f"  有效识别比例:        {valid_ratio:.1f}%")

    fps = config.get("manual_fps")
    if fps is None:
        dt = traj_df["time_s"].diff().median()
        fps = 1.0 / dt if dt and dt > 0 else None
    lines.append(f"  帧率:                {fps:.2f} fps" if fps else "  帧率:                未知")

    scale = config.get("scale_mm_per_px")
    if scale:
        lines.append(f"  比例尺:              {scale:.6f} mm/px")
    lines.append("")

    # ---- 追踪统计 ----
    lines.append("【追踪统计】")
    if traj_df["valid"].sum() > 0:
        valid_traj = traj_df[traj_df["valid"]]
        y_min = valid_traj["y_m"].min()
        y_max = valid_traj["y_m"].max()
        lines.append(f"  下落总位移:          {abs(y_max - y_min):.4f} m")
        lines.append(f"  y 范围:              {y_min:.4f} ~ {y_max:.4f} m")
        lines.append(f"  平均圆度:            {valid_traj['circularity'].mean():.4f}")
    lines.append("")

    # ---- 终端速度区 ----
    lines.append("【终端速度区】")
    if terminal_region.get("found", False):
        lines.append(f"  区间:                {terminal_region['start_time_s']:.4f}s"
                     f"  — {terminal_region['end_time_s']:.4f}s")
        lines.append(f"  起始帧:              {terminal_region['start_frame']}")
        lines.append(f"  结束帧:              {terminal_region['end_frame']}")
        lines.append(f"  终端速度 v_t:        {terminal_region['terminal_velocity_m_s']:.6f} m/s")
        lines.append(f"    (来自 y-t 线性拟合斜率)")
        lines.append(f"  拟合 R²:             {terminal_region['r2']:.6f}")
        lines.append(f"  速度变异系数 Cv:     {terminal_region['cv']:.4f}")
        n_candidates = len(terminal_region.get("candidates_table", []))
        lines.append(f"  候选窗口数:          {n_candidates}")
    else:
        lines.append(f"  {terminal_region.get('message', '未找到终端速度区。')}")
    lines.append("")

    # ---- 实验参数（全部使用标准字段 ball_radius_mm 等，转换为 SI 显示） ----
    lines.append("【实验参数】")
    br_mm = config.get('ball_radius_mm', None)
    lines.append(f"  小球半径 r:          {br_mm / 1000.0:.6f} m" if br_mm else "  小球半径 r:          ? m")
    lines.append(f"  小球密度 ρ_s:        {config.get('ball_density_kg_m3', '?')} kg/m³")
    lines.append(f"  液体密度 ρ_l:        {config.get('liquid_density_kg_m3', '?')} kg/m³")
    cr_mm = config.get('cylinder_radius_mm', None)
    lines.append(f"  量筒内半径 R:        {cr_mm / 1000.0:.6f} m" if cr_mm else "  量筒内半径 R:        ? m")
    lh_mm = config.get('liquid_height_mm', None)
    lines.append(f"  液柱高度 h:          {lh_mm / 1000.0:.6f} m" if lh_mm else "  液柱高度 h:          ? m")
    lines.append(f"  温度:                {config.get('temperature_c', '?')} °C")
    lines.append(f"  重力加速度 g:        {config.get('gravity_m_s2', '?')} m/s²")
    lines.append("")

    # ---- 黏度结果 ----
    lines.append("【黏度计算结果】")
    if viscosity_result is not None:
        lines.append(f"  理想黏度 η_basic:    {viscosity_result['eta_basic_pa_s']:.6f} Pa·s")
        wf = viscosity_result.get('wall_correction_factor', 1.0)
        if wf > 1.0:
            lines.append(f"  壁面修正因子:        {wf:.4f}")
            lines.append(f"    (1 + 2.4·r/R)(1 + 3.3·r/h)")
        rf = viscosity_result.get('reynolds_factor', 1.0)
        if rf > 1.0:
            lines.append(f"  雷诺数修正因子:        {rf:.4f}")
            lines.append(f"    (因子 = 1 + 3/16·Re)")
        lines.append(f"  最终黏度 η_final:   {viscosity_result['eta_final_pa_s']:.6f} Pa·s")
        lines.append(f"  雷诺数 Re:           {viscosity_result['reynolds_number']:.4f}")

        if viscosity_result.get("warnings"):
            lines.append("")
            lines.append("  [注意事项]")
            for w in viscosity_result["warnings"]:
                for line in w.split("\n"):
                    lines.append(f"    {line}")

        # 结论
        lines.append("")
        lines.append("【结论】")
        if terminal_region.get("found", False):
            lines.append(
                "  该视频中自动识别到稳定终端速度区，线性拟合优度满足设定阈值，\n"
                "  已计算得到液体黏度。结果仍需结合比例尺标定、温度记录和重复实验\n"
                "  进行不确定度分析。"
            )
        else:
            lines.append(
                "  已计算黏度值，但终端速度区判定可能存在问题，\n"
                "  请检查判稳参数或视频质量。"
            )
    else:
        lines.append("  未能得到可靠黏度结果。")
        if not terminal_region.get("found", False):
            lines.append(f"  原因: {terminal_region.get('message', '未找到终端速度区。')}")
    lines.append("")

    # 如果因为缺少视频或参数而失败
    lines.append(sep)
    lines.append("  程序: falling_ball_viscosity")
    lines.append("  项目: 基于机器视觉自动追踪与误差修正的落球法液体黏度测量实验改进")
    lines.append(sep)

    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "result_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("结果摘要已保存: %s", out_path)

    # 安全打印到控制台（兼容 Windows GBK）
    _safe_print_report("\n" + "\n".join(lines))
