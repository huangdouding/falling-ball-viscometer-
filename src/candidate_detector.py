"""
简化候选检测器（3 方法融合 + 5 维评分）

替代原 ball_detector.py 的 5 方法 + 10 维评分系统。

设计原则：
  1. 只保留 3 种经实际验证有效的方法
  2. 5 维评分，权重清晰，满分 100
  3. 每帧仅在局部搜索窗口内检测
  4. HoughCircles 默认禁用

【检测方法】
  1. background_subtraction — 背景差分，运动目标主方法
  2. blob (SimpleBlobDetector) — 暗 blob 检测，无运动时后备
  3. threshold_contour (Otsu+轮廓) — 通用后备

【评分维度（满分 100）】
  - pred_score (30): 与预测位置的距离
  - motion_score (25): 背景差分强度
  - size_score (20): 半径匹配度
  - contrast_score (15): 小球 vs 局部背景对比度
  - axis_score (10): 下落中心线对齐度
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Callable

# —— 评分权重 ——
W_PRED = 30       # 预测位置接近度
W_MOTION = 25     # 运动/差分强度
W_SIZE = 20       # 尺寸匹配度
W_CONTRAST = 15   # 对比度
W_AXIS = 10       # 下落中心线对齐

TOTAL_WEIGHT = W_PRED + W_MOTION + W_SIZE + W_CONTRAST + W_AXIS  # = 100


class CandidateDetector:
    """简化小球候选检测器。"""

    def __init__(self, config: dict, background: np.ndarray | None = None):
        # —— 预期尺寸 ——
        self.expected_radius_px_min = float(config.get("expected_radius_px_min", 1.0))
        self.expected_radius_px_max = float(config.get("expected_radius_px_max", 6.0))
        self.expected_radius_mid = (self.expected_radius_px_min + self.expected_radius_px_max) / 2.0

        # —— 面积范围 ——
        self.min_area_px = float(config.get("min_area_px", 10))
        self.max_area_px = float(config.get("max_area_px", 500))

        # —— 圆度 ——
        self.min_circularity = float(config.get("min_circularity", 0.15))

        # —— 背景 ——
        self.background = background

        # —— 颜色模式 ——
        self.color_mode = config.get("color_mode", "dark_ball_on_bright_bg")

        # —— 长线排除 ——
        self.enable_long_line = config.get("enable_long_line_rejection", True)

        # —— 中心线 ——
        self.fall_axis_x: float | None = config.get("fall_axis_x")
        self.allowed_axis_deviation = float(config.get("allowed_axis_deviation_px", 50))

        # —— Blob 检测器 ——
        self._blob_detector = self._make_blob_detector()

        # —— 高斯模糊核 ——
        self._gaussian_ksize = int(config.get("gaussian_blur_ksize", 5))
        if self._gaussian_ksize % 2 == 0:
            self._gaussian_ksize += 1

    # ── Blob Detector ──

    def _make_blob_detector(self):
        p = cv2.SimpleBlobDetector_Params()
        p.filterByArea = True
        p.minArea = self.min_area_px
        p.maxArea = self.max_area_px
        p.filterByCircularity = True
        p.minCircularity = max(0.01, self.min_circularity)
        p.filterByColor = True
        p.blobColor = 0                     # dark blob
        p.filterByConvexity = True
        p.minConvexity = 0.7
        p.filterByInertia = False
        return cv2.SimpleBlobDetector_create(p)

    # ── 主检测入口 ──

    def detect(
        self,
        gray: np.ndarray,
        search_rect: tuple[int, int, int, int] | None = None,
        *,
        predict_pos: tuple[float, float] | None = None,
        frame_idx: int = 0,
    ) -> list[dict]:
        """在搜索窗口内检测小球候选。

        Args:
            gray: 灰度图 (ROI 裁剪后)
            search_rect: 搜索矩形 (x, y, w, h) 相对 gray 坐标
            predict_pos: 预测位置 (px, py)，用于 pred_score
            frame_idx: 帧号（仅用于 debug）

        Returns:
            按评分降序排列的候选列表，每个候选是 dict:
              cx, cy, radius, area, circularity, contrast,
              score, pred_score, motion_score, size_score,
              contrast_score, axis_score, source
        """
        # 搜索窗口裁剪
        if search_rect is not None:
            sx, sy, sw, sh = search_rect
            sx = max(0, sx); sy = max(0, sy)
            sw = min(sw, gray.shape[1] - sx)
            sh = min(sh, gray.shape[0] - sy)
            if sw <= 0 or sh <= 0:
                return []
            sub_gray = gray[sy:sy+sh, sx:sx+sw].copy()
            offset_x, offset_y = sx, sy
        else:
            sub_gray = gray
            offset_x, offset_y = 0, 0

        # 收集所有候选
        candidates: list[dict] = []

        # 方法1: 背景差分（有背景时）
        if self.background is not None:
            bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY) \
                if len(self.background.shape) == 3 else self.background
            if bg_gray.shape == gray.shape:
                diff_map = cv2.absdiff(gray, bg_gray)
            else:
                diff_map = None
        else:
            diff_map = None

        if diff_map is not None:
            motion_cands = self._detect_bg_sub(sub_gray, diff_map, offset_x, offset_y)
            for c in motion_cands:
                c["_motion"] = True
            candidates.extend(motion_cands)

        # 方法2: Blob
        blob_cands = self._detect_blob(sub_gray, offset_x, offset_y)
        for c in blob_cands:
            if diff_map is not None:
                # 计算 blob 位置的差分值
                cx_int, cy_int = int(c["cx"]), int(c["cy"])
                if 0 <= cy_int < diff_map.shape[0] and 0 <= cx_int < diff_map.shape[1]:
                    r = max(1, int(c["radius"]))
                    y0 = max(0, cy_int - r); y1 = min(diff_map.shape[0], cy_int + r)
                    x0 = max(0, cx_int - r); x1 = min(diff_map.shape[1], cx_int + r)
                    c["_mean_diff"] = float(np.mean(diff_map[y0:y1, x0:x1]))
                else:
                    c["_mean_diff"] = 0.0
        candidates.extend(blob_cands)

        # 方法3: 阈值+轮廓（后备）
        if not candidates:
            contour_cands = self._detect_threshold_contour(sub_gray, offset_x, offset_y)
            candidates.extend(contour_cands)

        # 去重
        if len(candidates) > 1:
            candidates = self._deduplicate(candidates)

        # 长线排除
        if self.enable_long_line and candidates:
            candidates = [c for c in candidates if not self._is_long_line(
                sub_gray, c, offset_x, offset_y)]

        if not candidates:
            return []

        # 评分
        for c in candidates:
            c["score"] = self._score(c, gray, predict_pos)

        # 降序排列
        candidates.sort(key=lambda c: c["score"], reverse=True)

        return candidates

    # ── 各检测方法 ──

    def _detect_bg_sub(
        self, sub_gray: np.ndarray, diff_map: np.ndarray,
        ox: int, oy: int,
    ) -> list[dict]:
        """背景差分法：运动区域 → 轮廓 → 候选。"""
        _, motion_mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # 仅搜索子图区域的差分
        sub_diff = diff_map[oy:oy+sub_gray.shape[0], ox:ox+sub_gray.shape[1]] \
            if (oy + sub_gray.shape[0] <= diff_map.shape[0] and
                ox + sub_gray.shape[1] <= diff_map.shape[1]) else diff_map

        sub_mask = motion_mask[oy:oy+sub_gray.shape[0], ox:ox+sub_gray.shape[1]] \
            if (oy + sub_gray.shape[0] <= motion_mask.shape[0] and
                ox + sub_gray.shape[1] <= motion_mask.shape[1]) else motion_mask

        contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px or area > self.max_area_px:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.expected_radius_px_min * 0.5 or radius > self.expected_radius_px_max * 2.0:
                continue
            # 圆度
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            # 局部对比度
            contrast = self._local_contrast(sub_gray, cx, cy, radius)
            # 平均差分强度
            mean_diff = self._mean_in_circle(sub_diff, cx, cy, radius)

            candidates.append({
                "cx": cx + ox, "cy": cy + oy,
                "radius": radius, "area": area,
                "circularity": circularity,
                "contrast": contrast,
                "_mean_diff": mean_diff,
                "source": "bg_sub",
            })
        return candidates

    def _detect_blob(self, sub_gray: np.ndarray, ox: int, oy: int) -> list[dict]:
        """SimpleBlobDetector 检测。"""
        keypoints = self._blob_detector.detect(sub_gray)
        candidates = []
        for kp in keypoints:
            cx, cy = kp.pt
            r = kp.size / 2.0
            area = np.pi * r * r
            contrast = self._local_contrast(sub_gray, cx, cy, r)
            candidates.append({
                "cx": cx + ox, "cy": cy + oy,
                "radius": r, "area": area,
                "circularity": kp.response / 100.0 if kp.response else 0.5,
                "contrast": contrast,
                "_mean_diff": 0.0,
                "source": "blob",
            })
        return candidates

    def _detect_threshold_contour(
        self, sub_gray: np.ndarray, ox: int, oy: int
    ) -> list[dict]:
        """Otsu 阈值 + 轮廓检测。"""
        blurred = cv2.GaussianBlur(sub_gray, (self._gaussian_ksize, self._gaussian_ksize), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px or area > self.max_area_px:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.expected_radius_px_min * 0.3:
                continue
            if radius > self.expected_radius_px_max * 3.0:
                continue
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
            if circularity < self.min_circularity * 0.5:
                continue
            contrast = self._local_contrast(sub_gray, cx, cy, radius)
            candidates.append({
                "cx": cx + ox, "cy": cy + oy,
                "radius": radius, "area": area,
                "circularity": circularity,
                "contrast": contrast,
                "_mean_diff": 0.0,
                "source": "threshold_contour",
            })
        return candidates

    # ── 长线排除 ──

    def _is_long_line(
        self, sub_gray: np.ndarray, c: dict,
        ox: int, oy: int,
    ) -> bool:
        """判断候选是否为长线结构（刻度线/管壁边缘）。"""
        cx, cy = c["cx"], c["cy"]
        r = max(3, int(c["radius"] * 1.5))

        # 裁剪候选周围区域
        ix = int(cx) - ox; iy = int(cy) - oy
        x0 = max(0, ix - r); y0 = max(0, iy - r)
        x1 = min(sub_gray.shape[1], ix + r)
        y1 = min(sub_gray.shape[0], iy + r)
        patch = sub_gray[y0:y1, x0:x1]
        if patch.size < 9:
            return False

        # Sobel 边缘响应
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)

        # 长线特征：边缘集中在单一方向
        angle = np.arctan2(sobely, sobelx)
        directionality = np.std(angle) if np.std(angle) > 0 else 0
        # 方向性太集中 → 可能是直线
        if directionality < 0.5 and np.mean(mag) > 30:
            return True
        return False

    # ── 评分 ──

    def _score(
        self,
        c: dict,
        gray: np.ndarray,
        predict_pos: tuple[float, float] | None,
    ) -> float:
        """5 维综合评分，满分 100。"""
        scores = {}

        # 1. pred_score (30) — 与预测位置的距离
        if predict_pos:
            dx = abs(c["cx"] - predict_pos[0])
            dy = abs(c["cy"] - predict_pos[1])
            dist = np.hypot(dx, dy)
            # 距离 0 → 30 分，距离 > 100px → 0 分
            scores["pred_score"] = max(0, W_PRED * (1.0 - dist / 100.0))
        else:
            scores["pred_score"] = W_PRED * 0.5  # 无预测时不扣满

        # 2. motion_score (25) — 运动强度
        mean_diff = c.get("_mean_diff", 0)
        # diff 0-255, 阈值 15-80 之间线性评分
        md = min(mean_diff / 40.0, 1.0) if mean_diff > 15 else max(0, mean_diff / 15.0 * 0.3)
        scores["motion_score"] = W_MOTION * md

        # 3. size_score (20) — 半径匹配度
        r = c["radius"]
        r_err = abs(r - self.expected_radius_mid) / max(self.expected_radius_mid, 0.1)
        size_ratio = min(r / self.expected_radius_mid, self.expected_radius_mid / max(r, 0.1))
        scores["size_score"] = W_SIZE * min(1.0, size_ratio)

        # 4. contrast_score (15) — 对比度
        contrast = c.get("contrast", 0)
        # 对比度 0-50+，10 以上满分
        scores["contrast_score"] = W_CONTRAST * min(1.0, contrast / 15.0)

        # 5. axis_score (10) — 下落中心线对齐
        if self.fall_axis_x is not None:
            dist_axis = abs(c["cx"] - self.fall_axis_x)
            scores["axis_score"] = max(0, W_AXIS * (1.0 - dist_axis / self.allowed_axis_deviation))
        else:
            scores["axis_score"] = W_AXIS * 0.8

        c.update(scores)
        return sum(scores.values())

    # ── 辅助 ──

    def _local_contrast(self, gray: np.ndarray, cx: float, cy: float, r: float) -> float:
        """计算小球区域的局部对比度（中心 vs 外环）。"""
        h, w = gray.shape
        cx_i, cy_i = int(cx), int(cy)
        r_i = max(1, int(r))

        # 中心区域
        inner_r = max(1, int(r_i * 0.6))
        y0 = max(0, cy_i - inner_r); y1 = min(h, cy_i + inner_r)
        x0 = max(0, cx_i - inner_r); x1 = min(w, cx_i + inner_r)
        inner = gray[y0:y1, x0:x1]
        if inner.size == 0:
            return 0.0
        inner_mean = float(np.mean(inner))

        # 外环区域
        outer_r = r_i * 2
        y0 = max(0, cy_i - outer_r); y1 = min(h, cy_i + outer_r)
        x0 = max(0, cx_i - outer_r); x1 = min(w, cx_i + outer_r)
        outer = gray[y0:y1, x0:x1].copy()
        # 挖掉中心
        mask = np.ones_like(outer, dtype=np.uint8) * 255
        cv2.circle(mask, (outer_r, outer_r), inner_r, 0, -1)
        outer_values = outer[mask == 255]
        if len(outer_values) == 0:
            return 0.0
        outer_mean = float(np.mean(outer_values))

        diff = abs(outer_mean - inner_mean)
        return diff

    def _mean_in_circle(self, img: np.ndarray, cx: float, cy: float, r: float) -> float:
        """计算圆形区域的均值。"""
        h, w = img.shape[:2]
        cx_i, cy_i = int(cx), int(cy)
        r_i = max(1, int(r))
        y0 = max(0, cy_i - r_i); y1 = min(h, cy_i + r_i)
        x0 = max(0, cx_i - r_i); x1 = min(w, cx_i + r_i)
        patch = img[y0:y1, x0:x1]
        return float(np.mean(patch)) if patch.size > 0 else 0.0

    def _deduplicate(self, candidates: list[dict], dist_thresh: float = 5.0) -> list[dict]:
        """合并重叠候选。"""
        deduped = []
        for c in candidates:
            cx, cy = c["cx"], c["cy"]
            dup = False
            for k in deduped:
                if np.hypot(cx - k["cx"], cy - k["cy"]) < dist_thresh:
                    dup = True
                    break
            if not dup:
                deduped.append(c)
        return deduped
