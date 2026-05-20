"""
背景模型构建与维护

从 tracking.py 提取，提供更灵活的背景建模策略。

支持：
  - 中值合成（默认）
  - 平均值合成
  - 高斯混合模型（MOG2）
  - 滑动窗口更新
"""

from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_background_median(
    frames: list[np.ndarray],
) -> np.ndarray:
    """中值法合成背景。

    对 >= 20 帧效果较好，能有效滤除运动目标。
    帧数不足时回退到单帧。
    """
    if not frames:
        raise ValueError("无帧数据，无法构建背景")
    if len(frames) == 1:
        return frames[0]
    bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    return bg


def build_background_mean(frames: list[np.ndarray]) -> np.ndarray:
    """平均法合成背景。"""
    if not frames:
        raise ValueError("无帧数据，无法构建背景")
    bg = np.mean(np.array(frames), axis=0).astype(np.uint8)
    return bg


def compute_diff_map(
    frame_gray: np.ndarray,
    background_gray: np.ndarray,
) -> np.ndarray:
    """计算帧与背景的差分图。

    返回 0~255 uint8 的差分图，值越大表示与背景差异越大。
    """
    return cv2.absdiff(frame_gray, background_gray)


def threshold_motion(
    diff_map: np.ndarray,
    threshold: int = 30,
    morph_ksize: int = 3,
) -> np.ndarray:
    """将差分图二值化，获取运动区域掩码。

    Args:
        diff_map: 差分图 (uint8)
        threshold: 二值化阈值 (默认 30)
        morph_ksize: 形态学核大小，0=不处理

    Returns:
        运动区域掩码 (uint8, 0/255)
    """
    _, mask = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
    if morph_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_foreground_contours(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 5000,
) -> list[tuple[int, int, int, int]]:
    """从运动掩码中提取前景轮廓的外接矩形。

    Returns:
        [(x, y, w, h), ...] 按面积降序排列
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            result.append((x, y, w, h))
    result.sort(key=lambda r: r[2] * r[3], reverse=True)
    return result
