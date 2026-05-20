"""
黏度计算模块

基于斯托克斯公式计算液体黏度，包含壁面修正和雷诺数验证。

【符号约定】
  公式中出现的 r、R、h 均为半径/高度值（非直径）。

  基础公式（斯托克斯定律）：
      η_basic = 2 * r² * g * (ρ_s - ρ_l) / (9 * v_t)

  壁面与液柱高度修正（使用半径 r、R，非直径 d、D）：
      η_wall = η_basic / correction
      correction = (1 + 2.4 * r / R) * (1 + 3.3 * r / h)

    其中：
      r — 小球半径 (m)
      R — 量筒内半径 (m)  ← 注意是半径，不是直径
      h — 液柱高度 (m)

  雷诺数验证：
      Re = 2 * r * v_t * ρ_l / η_wall
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_viscosity(
    terminal_velocity_m_s: float,
    ball_radius_m: float,
    ball_density_kg_m3: float,
    liquid_density_kg_m3: float,
    cylinder_radius_m: float,
    liquid_height_m: float,
    g_m_s2: float = 9.8,
    temperature_c: Optional[float] = None,
    enable_wall_correction: bool = True,
) -> dict:
    """计算液体黏度。

    参数：
        terminal_velocity_m_s: 终端速度 (m/s)，正值表示下落
        ball_radius_m: 小球半径 (m)
        ball_density_kg_m3: 小球密度 (kg/m³)
        liquid_density_kg_m3: 液体密度 (kg/m³)
        cylinder_radius_m: 量筒内半径 (m)
        liquid_height_m: 液柱高度 (m)
        g_m_s2: 重力加速度 (m/s²)
        temperature_c: 温度 (°C)，可选
        enable_wall_correction: 是否启用壁面修正，默认 True

    返回 dict：
        eta_basic_pa_s: float        — 理想 Stokes 黏度 (Pa·s)
        correction_factor: float     — 壁面修正因子
        eta_wall_pa_s: float         — 壁面修正后黏度 (Pa·s)
        eta_final_pa_s: float        — 最终输出黏度 (依据 enable_wall_correction)
        enable_wall_correction: bool — 是否启用壁面修正
        r_over_R: float              — 球径/筒径比
        r_over_h: float              — 球径/液高比
        r_m: float                   — 小球半径 (m)
        R_m: float                   — 量筒内半径 (m)
        h_m: float                   — 液柱高度 (m)
        reynolds_number: float       — 雷诺数
        warnings: list[str]          — 警告信息
    """
    warnings = []

    # ---- 输入验证 ----
    if terminal_velocity_m_s <= 0:
        raise ValueError(
            f"终端速度必须为正数 (当前值: {terminal_velocity_m_s} m/s)。\n"
            f"请检查速度计算或方向约定是否正确。"
        )

    if ball_radius_m <= 0:
        raise ValueError(f"小球半径必须为正数 (当前值: {ball_radius_m} m)")

    if ball_density_kg_m3 <= liquid_density_kg_m3:
        warnings.append(
            f"小球密度 ({ball_density_kg_m3} kg/m³) 不大于液体密度 "
            f"({liquid_density_kg_m3} kg/m³)。\n"
            f"  小球可能无法下沉，请检查参数。"
        )

    if cylinder_radius_m <= 0:
        raise ValueError(f"量筒半径必须为正数 (当前值: {cylinder_radius_m} m)")

    if liquid_height_m <= 0:
        raise ValueError(f"液柱高度必须为正数 (当前值: {liquid_height_m} m)")

    # ---- 基础黏度 ----
    r = ball_radius_m
    rho_s = ball_density_kg_m3
    rho_l = liquid_density_kg_m3
    g = g_m_s2
    v_t = terminal_velocity_m_s

    eta_basic = (2.0 * r**2 * g * (rho_s - rho_l)) / (9.0 * v_t)

    # ---- 壁面与液柱高度修正 ----
    # correction = (1 + 2.4 * r / R) * (1 + 3.3 * r / h)
    #   r — 小球半径 (m), R — 量筒内半径 (m), h — 液柱高度 (m)
    R = cylinder_radius_m
    h = liquid_height_m
    r_over_R = r / R
    r_over_h = r / h
    wall_factor_radial = 1.0 + 2.4 * r / R
    wall_factor_height = 1.0 + 3.3 * r / h
    correction = wall_factor_radial * wall_factor_height

    logger.info(
        "Wall correction details (Ladenburg-Faxen, divide correction):\n"
        "  r (ball radius) = %.6f m\n"
        "  R (cylinder radius) = %.6f m\n"
        "  h (liquid height) = %.6f m\n"
        "  r/R = %.6f\n"
        "  r/h = %.6f\n"
        "  wall_factor_radial = 1 + 2.4*(r/R) = %.6f\n"
        "  wall_factor_height = 1 + 3.3*(r/h) = %.6f\n"
        "  correction_factor = wall_factor_radial * wall_factor_height = %.6f\n"
        "  eta_wall = eta_basic / correction_factor",
        r, R, h, r_over_R, r_over_h,
        wall_factor_radial, wall_factor_height, correction,
    )

    eta_wall = eta_basic / correction

    if correction > 1.10:
        logger.warning(
            "Wall correction factor is large (correction=%.4f > 1.10). "
            "Check cylinder/ball dimensions or eccentric fall.",
            correction,
        )

    # ---- 雷诺数 ----
    Re = 2.0 * r * v_t * rho_l / eta_wall

    if Re > 1.0:
        warnings.append(
            f"雷诺数 Re = {Re:.2f} > 1，当前可能不满足低雷诺数条件。\n"
            f"  斯托克斯公式要求 Re << 1。\n"
            f"  建议: 使用更小直径的小球或更高黏度的液体。\n"
            f"  当前结果需要谨慎解释，应考虑雷诺数修正。"
        )
    elif Re > 0.1:
        warnings.append(
            f"雷诺数 Re = {Re:.3f}，略高于完全层流条件。\n"
            f"  建议考虑雷诺数修正项。"
        )

    # ---- 温度信息 ----
    if temperature_c is not None:
        logger.info("Liquid temperature: %.1f C", temperature_c)

    logger.info(
        "Viscosity results:\n"
        "  eta_basic = %.6f Pa*s  (Stokes)\n"
        "  correction_factor = %.4f\n"
        "  eta_wall  = %.6f Pa*s  (wall-corrected)\n"
        "  Re      = %.4f",
        eta_basic, correction, eta_wall, Re
    )

    # Decide final viscosity output
    eta_final = eta_wall if enable_wall_correction else eta_basic
    logger.info(
        "  eta_final = %.6f Pa*s  (enable_wall_correction=%s)",
        eta_final, enable_wall_correction,
    )

    return {
        "eta_basic_pa_s": float(eta_basic),
        "correction_factor": float(correction),
        "eta_wall_pa_s": float(eta_wall),
        "eta_final_pa_s": float(eta_final),
        "enable_wall_correction": enable_wall_correction,
        "r_over_R": float(r_over_R),
        "r_over_h": float(r_over_h),
        "r_m": float(r),
        "R_m": float(R),
        "h_m": float(h),
        "reynolds_number": float(Re),
        "warnings": warnings,
    }
