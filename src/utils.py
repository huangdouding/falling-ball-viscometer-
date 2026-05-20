"""
工具函数模块

提供：
- 检查必填参数
- 读取配置文件
- 单位换算
- 日志打印
"""

import os
import sys
import yaml
import logging


def setup_logging(output_dir: str = None):
    """配置日志输出到控制台，兼容 GBK 终端。"""
    # 让 stdout 能安全处理 Unicode 字符（Windows GBK 终端）
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 避免重复添加 handler
    if not root.handlers:
        root.addHandler(handler)


def load_config(config_path: str) -> dict:
    """读取 YAML 配置文件。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ---- 标准字段映射 ----

STANDARD_PHYSICS_KEYS = {
    "scale_mm_per_px",
    "ball_diameter_mm",
    "ball_radius_mm",
    "ball_density_kg_m3",
    "liquid_density_kg_m3",
    "cylinder_radius_mm",
    "liquid_height_mm",
    "temperature_c",
    "reference_viscosity_pa_s",
    "enable_wall_correction",
    "gravity_m_s2",
    "manual_fps",
}

_OLD_TO_STANDARD = {
    "scale": "scale_mm_per_px",
    "mm_per_px": "scale_mm_per_px",
    "diameter": "ball_diameter_mm",
    "ball_diameter": "ball_diameter_mm",
    "ball_diameter_m": "ball_diameter_mm",
    "diameter_m": "ball_diameter_mm",
    "radius": "ball_radius_mm",
    "ball_radius": "ball_radius_mm",
    "r_mm": "ball_radius_mm",
    "rho_s": "ball_density_kg_m3",
    "ball_density": "ball_density_kg_m3",
    "rho_l": "liquid_density_kg_m3",
    "liquid_density": "liquid_density_kg_m3",
    "cylinder_radius": "cylinder_radius_mm",
    "R_mm": "cylinder_radius_mm",
    "liquid_height": "liquid_height_mm",
    "h_mm": "liquid_height_mm",
    "fps": "manual_fps",
    "video_fps": "manual_fps",
    "g_m_s2": "gravity_m_s2",
    "reference_viscosity": "reference_viscosity_pa_s",
}

_SI_TO_MM = {
    "ball_diameter_m": "ball_diameter_mm",
    "ball_radius_m": "ball_radius_mm",
    "cylinder_radius_m": "cylinder_radius_mm",
    "liquid_height_m": "liquid_height_mm",
}


# ---- 检测器配置文件 ----
# 每个 profile 定义一组预设参数覆盖，用户通过 config["detector_profile"] 选择
DETECTOR_PROFILES = {
    "stable_demo": {
        "_profile_description": "稳定演示模式：放宽质量门槛，优先输出结果",
        "r2_threshold": 0.95,
        "quality_r2_threshold": 0.95,
        "cv_threshold": 0.12,
        "terminal_min_real_detection_rate": 0.60,
        "min_track_raw_rate": 0.20,
        "quality_cv_threshold": 0.10,
    },
    "strict_physics": {
        "_profile_description": "严格物理模式：高标准质量要求，用于正式分析",
        "r2_threshold": 0.995,
        "quality_r2_threshold": 0.995,
        "cv_threshold": 0.05,
        "terminal_min_real_detection_rate": 0.85,
        "min_track_raw_rate": 0.40,
        "quality_cv_threshold": 0.03,
    },
}


def _apply_detector_profile(config: dict) -> dict:
    """根据 config["detector_profile"] 应用预设参数覆盖。

    仅覆盖用户在 config 中未显式设置的参数（profile 作为默认值）。
    如果用户同时设置了 detector_profile 和具体参数，用户参数优先。
    """
    profile_name = config.get("detector_profile")
    if not profile_name or profile_name not in DETECTOR_PROFILES:
        return config

    profile = DETECTOR_PROFILES[profile_name]
    for key, value in profile.items():
        if key.startswith("_"):
            continue
        if key not in config or config[key] is None:
            config[key] = value

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "已应用检测器配置文件: %s (%s)",
        profile_name,
        profile.get("_profile_description", ""),
    )
    return config


def normalize_config_keys(config: dict) -> dict:
    """标准化配置字典中的字段名。

    1. 将旧字段名映射为标准字段名
    2. 将 SI 单位值（m）转换为 mm 值
    3. 移除重复/遗留的旧字段名
    4. 保留所有其他字段不变
    """
    result = dict(config)

    # 1. 映射旧字段名 → 标准字段名
    for old_key, standard_key in _OLD_TO_STANDARD.items():
        if old_key in result and old_key != standard_key:
            if standard_key not in result or result[standard_key] is None:
                result[standard_key] = result[old_key]
            del result[old_key]

    # 2. 直径→半径转换：ball_diameter_mm 自动生成 ball_radius_mm
    #    如果同时有直径和半径且不一致，以半径为准并记录
    if "ball_diameter_mm" in result:
        diam = result["ball_diameter_mm"]
        if "ball_radius_mm" not in result or result["ball_radius_mm"] is None:
            result["ball_radius_mm"] = diam / 2.0
        else:
            # 已有半径 → 校验一致性
            implied_diam = result["ball_radius_mm"] * 2.0
            if abs(implied_diam - diam) > 1e-6:
                import warnings as _w
                _w.warn(
                    f"ball_diameter_mm ({diam}) 与 ball_radius_mm ({result['ball_radius_mm']}) "
                    f"不一致，使用 ball_radius_mm × 2 = {implied_diam} 作为直径"
                )
        del result["ball_diameter_mm"]

    # 3. SI→mm 转换：当只有 m 键且没有 mm 键时，生成 mm 键
    for si_key, mm_key in _SI_TO_MM.items():
        if si_key in result:
            if mm_key not in result or result[mm_key] is None:
                result[mm_key] = result[si_key] * 1000.0
            del result[si_key]

    # 4. 清理重复键（只处理物理参数，不碰检测参数如 threshold_mode / color_mode）
    for kept, removed in [
        ("terminal_ignore_start_sec", "ignore_start_sec"),
        ("terminal_ignore_end_sec", "ignore_end_sec"),
    ]:
        if kept in result and removed in result:
            del result[removed]

    # 5. 移除旧的 g_m_s2（已映射到 gravity_m_s2 后保留 gravity_m_s2）
    if "g_m_s2" in result and "gravity_m_s2" in result:
        del result["g_m_s2"]

    # 6. 应用检测器配置文件（stable_demo / strict_physics）
    result = _apply_detector_profile(result)

    return result


def check_required_params(config: dict):
    """检查必填的实验参数是否已填写。"""
    video_path = config.get("video_path", "")
    if not video_path or "replace_with_real_video" in video_path:
        print(
            "[警告] 视频文件路径未设置或仍为占位路径。\n"
            f"   当前值: {video_path}\n"
            "   请在 config.yaml 中将 video_path 改为真实视频路径。"
        )
        return False

    if not os.path.exists(video_path):
        print(
            f"[警告] 视频文件不存在: {video_path}\n"
            "   请在 config.yaml 中设置正确的 video_path。"
        )
        return False

    scale = config.get("scale_mm_per_px")
    if scale is None:
        print(
            "[警告] scale_mm_per_px 未设置。\n"
            "   请在 config.yaml 中填写比例尺（mm/px）。"
        )
        return False

    required_physics = [
        "ball_radius_mm", "ball_density_kg_m3",
        "liquid_density_kg_m3", "cylinder_radius_mm",
        "liquid_height_mm",
    ]
    missing = [p for p in required_physics if config.get(p) is None]
    if missing:
        print(
            f"[警告] 以下物理参数未设置: {', '.join(missing)}\n"
            f"   请在 config.yaml 中填写。"
        )
        return False

    return True


def ensure_output_dir(output_dir: str):
    """确保输出目录存在。"""
    os.makedirs(output_dir, exist_ok=True)


def mm_per_px_to_m_per_px(scale_mm_per_px: float) -> float:
    """将 mm/px 转换为 m/px。"""
    return scale_mm_per_px / 1000.0
