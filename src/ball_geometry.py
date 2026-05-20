"""
小球几何参数计算

统一管理直径、半径、比例尺换算，避免 radius/diameter 混淆。

【用法】
    geo = compute_ball_geometry(ball_diameter_mm=1.5, scale_mm_per_px=0.2869)
    # geo.ball_radius_mm == 0.75
    # geo.expected_diameter_px == 5.23
    # geo.size_quality == "acceptable"
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# —— 尺寸品质分级（基于像素直径） ——
# good:       ≥ 12 px  → 任何检测方法都稳定
# acceptable: ≥ 10 px  → 多数方法可工作
# poor:       ≥  8 px  → blob 可能不稳定，需 bg_sub
# bad:        <  5 px  → 无法可靠检测

_QUALITY_GRADES = [
    ("good", 12.0, "任何检测方法都稳定"),
    ("acceptable", 10.0, "多数方法可工作"),
    ("poor", 5.0, "仅背景差分可用"),
    ("bad", 0.0, "无法可靠检测"),
]


@dataclass
class BallGeometry:
    """小球几何参数（全部根据直径 + 比例尺自动计算）。"""
    # —— 输入 ——
    ball_diameter_mm: float          # 小球直径 (mm)
    scale_mm_per_px: float           # 比例尺 (mm/px)

    # —— 计算值 ——
    ball_radius_mm: float            # 小球半径 (mm)

    expected_diameter_px: float      # 预期像素直径
    expected_radius_px: float        # 预期像素半径
    expected_area_px: float          # 预期像素面积

    size_quality: str                # good / acceptable / poor / bad
    quality_description: str         # 中文说明

    # —— 警告 ——
    size_warning: str                # 空字符串 = 无警告

    @property
    def is_reliable(self) -> bool:
        """是否可可靠检测（至少 poor 以上）。"""
        return self.size_quality not in ("bad",)

    @property
    def quality_score(self) -> float:
        """数值化品质评分（0~100）。"""
        grades = {g[0]: i for i, g in enumerate(_QUALITY_GRADES)}
        max_g = len(_QUALITY_GRADES) - 1
        return (max_g - grades.get(self.size_quality, max_g)) / max_g * 100


def compute_ball_geometry(
    ball_diameter_mm: float,
    scale_mm_per_px: float,
    *,
    ball_radius_mm: float | None = None,
) -> BallGeometry:
    """计算小球几何参数。

    参数：
        ball_diameter_mm: 小球直径 (mm)。如果只知半径，传半径*2。
        scale_mm_per_px: 比例尺 (mm/px)。
        ball_radius_mm: 可选，如果传了会用此值校验直径一致性。

    返回：
        BallGeometry dataclass
    """
    # —— 参数校验 ——
    if ball_diameter_mm <= 0:
        raise ValueError(f"小球直径必须为正数 (当前: {ball_diameter_mm} mm)")
    if scale_mm_per_px <= 0:
        raise ValueError(f"比例尺必须为正数 (当前: {scale_mm_per_px} mm/px)")

    # —— 一致性校验 ——
    if ball_radius_mm is not None and ball_radius_mm > 0:
        implied_diameter = ball_radius_mm * 2
        if abs(implied_diameter - ball_diameter_mm) > 1e-6:
            raise ValueError(
                f"ball_diameter_mm ({ball_diameter_mm}) 与 ball_radius_mm ({ball_radius_mm}) "
                f"不一致: 直径≠半径×2 ({implied_diameter})"
            )

    # —— 计算 ——
    ball_radius_mm = ball_diameter_mm / 2.0
    expected_diameter_px = ball_diameter_mm / scale_mm_per_px
    expected_radius_px = expected_diameter_px / 2.0
    expected_area_px = np.pi * expected_radius_px ** 2

    # —— 分级 ——
    size_quality = "bad"
    quality_description = _QUALITY_GRADES[-1][2]
    for name, threshold, desc in _QUALITY_GRADES:
        if expected_diameter_px >= threshold:
            size_quality = name
            quality_description = desc
            break

    # —— 警告 ——
    size_warning = ""
    if size_quality == "bad":
        size_warning = (
            f"像素直径仅 {expected_diameter_px:.1f} px (< 5 px)，无法可靠检测。"
        )
    elif size_quality == "poor":
        size_warning = (
            f"像素直径仅 {expected_diameter_px:.1f} px (< 10 px)，"
            f"仅背景差分法可能有效，建议换用更大小球或提高分辨率。"
        )

    return BallGeometry(
        ball_diameter_mm=ball_diameter_mm,
        scale_mm_per_px=scale_mm_per_px,
        ball_radius_mm=ball_radius_mm,
        expected_diameter_px=float(expected_diameter_px),
        expected_radius_px=float(expected_radius_px),
        expected_area_px=float(expected_area_px),
        size_quality=size_quality,
        quality_description=quality_description,
        size_warning=size_warning,
    )
