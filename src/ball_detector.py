"""
小球识别模块 — 多方法检测 + 综合评分（v3 运动优先版）

核心策略：
  1. 完整视频分析：bg_sub 产生运动候选 → blob 仅做后备（无运动=低分）
  2. 单帧测试：blob → threshold_contour（无背景时）
  3. 下落中心线约束：距离 fall_axis_x 太远硬排除
  4. 长线结构排除：刻度线/弧线候选被严重扣分
  5. 评分权重：motion(25) > axis(20) > continuity(15) > size(15) > isolation(10) > center_band(10) > contrast(5)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class BallDetector:
    """单帧小球检测器（运动优先 + 多方法 + 综合评分）。"""

    def __init__(self, config: dict, background: np.ndarray = None):
        # ---------- 媒体类型 ----------
        self._image_mode = config.get("image_mode", False)  # 图片模式禁止 bg_sub

        # ---------- 检测方法 ----------
        self.detect_method = config.get("detect_method", "auto")

        # ---------- 颜色模式 ----------
        self.threshold_mode = config.get("threshold_mode", "dark_ball_on_bright_bg")

        # ---------- Blob 检测参数 ----------
        self.min_area_px = config.get("min_area_px", 2)
        self.max_area_px = config.get("max_area_px", 80)
        self.min_circularity = config.get("min_circularity", 0.05)

        # ---------- 预期半径范围 ----------
        self.expected_radius_px_min = config.get("expected_radius_px_min", 1.0)
        self.expected_radius_px_max = config.get("expected_radius_px_max", 4.0)

        # ---------- 对比度参数 ----------
        self.contrast_threshold = config.get("contrast_threshold", 5.0)
        self.inner_ratio = 0.65
        self.outer_ring_min = 3
        self.outer_ring_max = 18

        # ---------- 局部亮度 ----------
        self.local_window = 37
        self.local_bright_threshold = 200
        self.local_bright_ratio = 0.50

        # ---------- 中心运动带约束 ----------
        self.center_band_enabled = config.get("center_band_enabled", True)
        self.center_band_left_ratio = config.get("center_band_left_ratio", 0.30)
        self.center_band_right_ratio = config.get("center_band_right_ratio", 0.70)

        # ---------- 下落中心线约束 ----------
        self.fall_axis_x = config.get("fall_axis_x", None)  # 全局坐标
        self.allowed_axis_deviation_px = config.get("allowed_axis_deviation_px", 50)

        # ---------- 长线排除 ----------
        self.enable_long_line_rejection = config.get("enable_long_line_rejection", True)

        # ---------- 追踪 — 核心参数 ----------
        self.max_jump_px = config.get("max_jump_px", 80)
        self._prev_cx: Optional[float] = None
        self._prev_cy: Optional[float] = None
        self._search_center: Optional[Tuple[float, float]] = None
        # 非对称搜索窗口（追踪模式下使用）
        self.search_win_w = config.get("search_win_w", 60)       # x: ±60 px
        self.search_win_y_up = config.get("search_win_y_up", 20)  # y: 向上 20 px
        self.search_win_y_down = config.get("search_win_y_down", 120)  # y: 向下 120 px

        # 预测追踪
        self._predicted_cx: Optional[float] = None
        self._predicted_cy: Optional[float] = None
        self._velocity_x: float = 0.0  # px/frame (ROI 坐标)
        self._velocity_y: float = 0.0
        self._consecutive_lost: int = 0

        # ---------- ROI（使用 detect_roi 做物理裁剪，回退到 roi） ----------
        self.roi = config.get("roi")
        self.detect_roi = config.get("detect_roi")  # 窄检测带（frame 坐标）

        # ---------- 预处理参数 ----------
        self.gaussian_blur_ksize = config.get("gaussian_blur_ksize", 5)
        self.morph_kernel_size = config.get("morph_kernel_size", 3)

        # ---------- 大球模式 ----------
        self.large_ball_mode = config.get("large_ball_mode", False)
        if self.large_ball_mode:
            # Large balls need softer shape checks, but the radius gate should
            # still come from the selected physical ball radius.
            self.min_area_px = max(1, config.get("min_area_px", 2))
            # max_area_px 已经由 _compute_dynamic_detection_params 计算
            self.min_circularity = 0.05  # 圆度不作为硬门槛
            # 大球模式对比度阈值降低
            self.contrast_threshold = max(2.0, config.get("contrast_threshold", 3.0))
            # 长线拒绝敏感度降低（大球反光容易误判为长线）
            self.enable_long_line_rejection = False
            self.expected_radius_px_min = config.get("expected_radius_px_min", 1.0)
            self.expected_radius_px_max = config.get("expected_radius_px_max", 20.0)
        else:
            self.min_circularity = max(0.05, config.get("min_circularity", 0.05))
            self.contrast_threshold = config.get("contrast_threshold", 5.0)
            self.enable_long_line_rejection = config.get("enable_long_line_rejection", True)
            if self.min_circularity > 0.20:
                self.min_circularity = 0.15  # 统一降低圆度硬门槛

        # ---------- 背景模型 ----------
        self.background = background

        # ---------- BlobDetector ----------
        # Blob detector 参数根据模式调整
        self._blob_params = self._make_blob_params()
        self._blob_detector = cv2.SimpleBlobDetector_create(self._blob_params)

    # ================================================================
    #  BlobDetector 参数
    # ================================================================
    def _make_blob_params(self):
        p = cv2.SimpleBlobDetector_Params()
        p.filterByArea = True
        p.minArea = float(self.min_area_px)
        p.maxArea = float(self.max_area_px)
        # OpenCV 要求 0 < minCircularity <= maxCircularity 即使 filterByCircularity=False
        if self.large_ball_mode:
            p.filterByCircularity = False
            p.minCircularity = 0.01
        else:
            p.filterByCircularity = True
            p.minCircularity = float(max(0.01, self.min_circularity))
        p.filterByColor = True
        p.blobColor = 0
        # 大球模式：降低凸性要求
        if self.large_ball_mode:
            p.filterByConvexity = False
            p.minConvexity = 0.5
        else:
            p.filterByConvexity = True
            p.minConvexity = 0.7
        p.filterByInertia = False
        return p

    # ================================================================
    #  主检测入口
    # ================================================================
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> dict:
        """单帧检测。"""
        roi_offset = (0, 0)
        debug = {}

        # ---- ROI 裁剪（优先 detect_roi，再 fallback 到 roi） ----
        crop_roi = self.detect_roi if self.detect_roi is not None else self.roi
        if crop_roi is not None:
            x, y, w, h = [int(v) for v in crop_roi]
            x = max(0, x); y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            if w > 0 and h > 0:
                img = frame[y:y + h, x:x + w].copy()
                roi_offset = (x, y)
            else:
                img = frame.copy()
        else:
            img = frame.copy()

        debug["roi_frame"] = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        debug["gray"] = gray
        roi_h, roi_w = gray.shape

        # ---- 背景差分图（用于运动 vs 静态判别） ----
        diff_map = None
        if self.background is not None:
            if len(self.background.shape) == 3:
                bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
            else:
                bg_gray = self.background
            if bg_gray.shape == gray.shape and gray.dtype == bg_gray.dtype:
                diff_map = cv2.absdiff(gray, bg_gray)
                debug["diff_map"] = diff_map
            else:
                debug["diff_map"] = None

        # ---- 执行检测 ----
        method = self.detect_method
        candidates = []
        method_used = ""
        method_history = []

        if method == "auto":
            candidates, method_used, method_history = self._detect_auto(gray, debug)
        elif method == "blob":
            candidates, method_history = self._detect_blob(gray, debug)
            method_used = "blob" if candidates else ""
        elif method == "threshold_contour":
            candidates, method_history = self._detect_threshold_contour(gray, debug)
            method_used = "threshold_contour" if candidates else ""
        elif method == "background_subtraction":
            candidates, method_history = self._detect_background_subtraction(gray, debug)
            method_used = "background_subtraction" if candidates else ""
        else:
            candidates, method_used, method_history = self._detect_auto(gray, debug)

        debug["method_history"] = method_history
        debug["method_used"] = method_used

        # 下落中心线（用于评分和搜索窗口参考）
        axis_x_roi = self._get_fall_axis_x_roi(roi_offset[0], roi_w)

        # ---- 1. 搜索窗口过滤（追踪模式 — 非对称窗口） ----
        all_pre_search = list(candidates)  # 保留原始候选用于扩展搜索
        search_expanded = False
        debug["search_window_filtered"] = 0
        debug["pre_search_candidates"] = len(candidates)
        debug["search_window_active"] = self._search_center is not None
        if self._search_center is not None:
            scx, scy = self._search_center
            before_sw = len(candidates)

            # 非对称窗口：x ± search_win_w, y 向下 search_win_y_down, 向上 search_win_y_up
            wx = self.search_win_w
            wy_up = self.search_win_y_up
            wy_down = self.search_win_y_down
            candidates = [c for c in candidates
                          if abs(c["cx"] - scx) <= wx
                          and -wy_up <= (c["cy"] - scy) <= wy_down]

            # 无候选时逐步扩大窗口
            if not candidates:
                for expand in [1.5, 2.0, 3.0]:
                    wx_e = wx * expand
                    wy_up_e = wy_up * expand
                    wy_down_e = wy_down * expand
                    candidates = [c for c in all_pre_search
                                  if abs(c["cx"] - scx) <= wx_e
                                  and -wy_up_e <= (c["cy"] - scy) <= wy_down_e]
                    if candidates:
                        search_expanded = True
                        debug["search_window_expanded"] = f"{expand}x"
                        break
            debug["search_window_filtered"] = before_sw - len(candidates)

        # ---- 1b. 预测回退：搜索窗口无候选时，用速度预测结果 ----
        used_fallback = False
        if not candidates and self._prev_cx is not None and self._consecutive_lost < 10:
            pred_x, pred_y = self.predict_next_position()
            if pred_x is not None:
                used_fallback = True
                # predict_next_position 返回 ROI 坐标（_prev_cx 已使用 ROI 坐标）
                pred_x_roi = pred_x
                pred_y_roi = pred_y
                # 更新内部状态保证追踪连续性（使用 ROI 坐标）
                self._prev_cx = pred_x_roi
                self._prev_cy = pred_y_roi
                self._consecutive_lost += 1
                # 用预测位置构造一个"软"候选，评分较低但能让追踪继续
                fallback_radius = (self.expected_radius_px_min + self.expected_radius_px_max) / 2.0
                fallback_area = np.pi * fallback_radius ** 2
                fallback_candidate = {
                    "cx": pred_x_roi, "cy": pred_y_roi,
                    "radius": fallback_radius, "area": fallback_area,
                    "circularity": 0.5, "contrast": 0.0,
                    "inner_mean": 0.0, "outer_mean": 0.0,
                    "source": "fallback_prediction",
                    "fallback": True,
                }
                candidates = [fallback_candidate]
                debug["fallback_prediction"] = (pred_x_roi, pred_y_roi)
                debug["search_window_expanded"] = False

        debug["search_window_expanded"] = search_expanded

        # ---- 补齐缺失的调试图（GUI 需要完整字段） ----
        if "binary_mask" not in debug:
            debug["binary_mask"] = np.zeros_like(gray)
        if np.count_nonzero(debug.get("binary_mask", np.zeros(1))) == 0:
            debug["binary_mask_empty"] = True
        if "contour_preview" not in debug:
            debug["contour_preview"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if "candidate_preview" not in debug:
            debug["candidate_preview"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug["original"] = img.copy()

        # ---- 4. 无候选 ----
        if not candidates:
            fail_reason = self._diagnose_failure(debug, gray)
            # 失败时的 final_preview：标注失败原因
            fail_final = img.copy()
            _overlay = fail_final.copy()
            cv2.rectangle(_overlay, (2, 2), (280, 80), (0, 0, 0), -1)
            fail_final = cv2.addWeighted(fail_final, 0.6, _overlay, 0.4, 0)
            cv2.putText(fail_final, f"Frame: {frame_idx}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(fail_final, "Detected: NO", (6, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(fail_final, f"Method: {method_used or 'none'}", (6, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(fail_final, f"Reason: {fail_reason[:60]}", (6, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)
            debug["final_preview"] = fail_final
            return self._make_result(
                found=False, fail_reason=fail_reason,
                roi_offset=roi_offset, debug=debug,
                method_info=method_used or "none",
            )

        # ---- 5. 综合评分 + 选最优 ----
        candidates = self._score_candidates(candidates, roi_w, gray, axis_x_roi,
                                            diff_map=diff_map)
        best = candidates[0]

        # 判断是否为预测回退
        is_fallback = best.get("fallback", False)

        x_global = best["cx"] + roi_offset[0]
        y_global = best["cy"] + roi_offset[1]

        # ★ 不再在 detect() 内部自动更新追踪状态。
        # 追踪状态仅由 tracking.py 确认点有效后通过 set_prev() 更新，
        # 避免被 tracking.py 后续拒绝的点污染搜索中心和速度估计。
        # 预测回退的临时状态保留在 candidates 的 fallback 标记中。

        # ---- 6. 构建候选排序图 candidate_preview ----
        h_img, w_img = img.shape[:2]
        _CANVAS_W = w_img + 260  # 右侧信息栏
        _CANVAS_H = max(h_img, 320)
        cand_preview = np.full((_CANVAS_H, _CANVAS_W, 3), 30, dtype=np.uint8)
        cand_preview[0:h_img, 0:w_img] = img.copy()
        # 右侧信息栏背景
        cv2.rectangle(cand_preview, (w_img, 0), (_CANVAS_W, _CANVAS_H), (40, 40, 40), -1)

        y_text = 16
        for i, c in enumerate(candidates[:10]):
            rank = i + 1
            # 颜色规则
            rj = c.get("reject_reason", "")
            if i == 0 and not rj:
                col = (0, 255, 0)       # 最终选中：绿色
            elif rj:
                col = (0, 0, 255)       # 被排除：红色
            else:
                col = (0, 255, 255)     # 通过但未选中：黄色

            # 左侧画圈
            cv2.circle(cand_preview, (int(c["cx"]), int(c["cy"])),
                       int(c["radius"]) + 2, col, 1)
            if i == 0 and not rj:
                cv2.circle(cand_preview, (int(c["cx"]), int(c["cy"])), 3, (255, 0, 0), -1)

            # 右侧文字
            dist_axis = c.get("dist_to_axis", abs(c["cx"] - axis_x_roi)) if axis_x_roi else 0
            text_x = w_img + 6
            cv2.putText(cand_preview, f"#{rank} s={c['score']:.0f}",
                        (text_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            y_text += 14
            cv2.putText(cand_preview,
                        f"  ({c['cx']:.0f},{c['cy']:.0f}) r={c['radius']:.1f}",
                        (text_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)
            y_text += 13
            cv2.putText(cand_preview,
                        f"  diff={c.get('diff_score',0):.0f}(md={c.get('mean_diff',0):.1f}) src={c.get('source','?')}",
                        (text_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1)
            y_text += 13
            if rj:
                cv2.putText(cand_preview, f"  REJ: {rj[:40]}",
                            (text_x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 80, 255), 1)
                y_text += 13
            y_text += 3  # 间距
        debug["candidate_preview"] = cand_preview

        # ---- 7. 构建最终标注图 final_preview（简化） ----
        final = img.copy()
        # 最终选中小球：绿色加粗圆圈 + 蓝色中心点
        cv2.circle(final, (int(best["cx"]), int(best["cy"])),
                   int(best["radius"]), (0, 255, 0), 2)
        cv2.circle(final, (int(best["cx"]), int(best["cy"])), 3, (255, 0, 0), -1)
        # 小球旁边小标签
        cv2.putText(final, f"({x_global:.0f},{y_global:.0f})",
                    (int(best["cx"]) + int(best["radius"]) + 5, int(best["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # 左上角信息框
        _overlay = final.copy()
        cv2.rectangle(_overlay, (2, 2), (240, 110), (0, 0, 0), -1)
        final = cv2.addWeighted(final, 0.6, _overlay, 0.4, 0)
        best_diff = best.get("diff_score", 0)
        best_md = best.get("mean_diff", 0)
        info_lines = [
            f"Frame: {frame_idx}",
            f"Detected: YES",
            f"Center: ({x_global:.0f}, {y_global:.0f})",
            f"Radius: {best['radius']:.1f} px  Score: {best['score']:.0f}",
            f"Diff: {best_diff:.0f} (mean={best_md:.1f})  src={method_used}",
        ]
        for j, line in enumerate(info_lines):
            cv2.putText(final, line, (6, 16 + j * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        debug["final_preview"] = final

        # ---- 7. 返回（候选坐标统一转为全局坐标） ----
        return self._make_result(
            found=True,
            fallback=is_fallback,
            x_px=x_global, y_px=y_global,
            radius_px=best["radius"],
            area_px=best["area"],
            circularity=best.get("circularity", 0),
            confidence=best["score"],
            candidates=[{
                "cx": c["cx"] + roi_offset[0],
                "cy": c["cy"] + roi_offset[1],
                "radius": c["radius"], "area": c["area"],
                "contrast": c.get("contrast", 0),
                "score": c["score"],
                "diff_score": c.get("diff_score", 0),
                "mean_diff": c.get("mean_diff", 0),
                "isolation": c.get("isolation_score", 0),
                "center_band": c.get("center_band_score", 0),
                "size_score": c.get("size_score", 0),
                "continuity": c.get("motion_continuity", 0),
                "axis_score": c.get("axis_score", 0),
                "motion_bonus": c.get("motion_bonus", 0),
                "long_line_penalty": c.get("long_line_penalty", 0),
                "source_method": c.get("source", "?"),
                "dist_to_axis": abs(c["cx"] + roi_offset[0] - (axis_x_roi + roi_offset[0])),
                "reject_reason": c.get("reject_reason", ""),
                "rank": i + 1,
            } for i, c in enumerate(candidates)],
            roi_offset=roi_offset, debug=debug,
            method_info=f"{'pred' if is_fallback else method_used}(s={best['score']:.0f})",
        )

    # ================================================================
    #  自动检测策略
    # ================================================================
    def _detect_auto(self, gray: np.ndarray, debug: dict) -> Tuple[list, str, list]:
        """自动检测：运行所有适用方法，合并候选统一评分。

        图片模式: blob + threshold_contour
        视频模式: bg_sub + blob + threshold_contour
        """
        method_history = []

        if self._image_mode:
            methods = [
                ("blob", self._detect_blob),
                ("threshold_contour", self._detect_threshold_contour),
            ]
            if self.large_ball_mode:
                # 大球模式在图片模式下加入自适应阈值
                methods.insert(1, ("adaptive_threshold", self._detect_adaptive_threshold))
        else:
            methods = [
                ("background_subtraction", self._detect_background_subtraction),
                ("blob", self._detect_blob),
                ("threshold_contour", self._detect_threshold_contour),
            ]
            if self.large_ball_mode:
                # 大球模式：自适应阈值优先于普通阈值
                methods.insert(2, ("adaptive_threshold", self._detect_adaptive_threshold))

        all_candidates = []
        for method_name, method_fn in methods:
            cands, hist = method_fn(gray, debug)
            method_history.extend(hist)
            all_candidates.extend(cands)

        # 去重
        if all_candidates:
            all_candidates = self._deduplicate_candidates(all_candidates)

        # ★ 大球模式无候选时尝试 HoughCircles 后备
        if not all_candidates and self.large_ball_mode:
            hough_cands, hough_hist = self._detect_hough_circles(gray, debug)
            method_history.extend(hough_hist)
            all_candidates.extend(hough_cands)

        return all_candidates, "merged", method_history

    def _deduplicate_candidates(self, candidates: list,
                                 distance_threshold: float = 5.0) -> list:
        """合并重叠候选（中心距离 < distance_threshold 者保留较优者）。"""
        if len(candidates) <= 1:
            return candidates

        # 按面积接近预期排序作为初步优先级
        exp_r_mid = (self.expected_radius_px_min + self.expected_radius_px_max) / 2.0
        candidates.sort(key=lambda c: abs(c["radius"] - exp_r_mid))

        kept = []
        for c in candidates:
            cx, cy = c["cx"], c["cy"]
            is_dup = False
            for k in kept:
                if np.hypot(cx - k["cx"], cy - k["cy"]) < distance_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)

        return kept

    # ================================================================
    #  方法 1: SimpleBlobDetector + 对比度验证
    # ================================================================
    def _detect_blob(self, gray: np.ndarray, debug: dict) -> Tuple[list, list]:
        """SimpleBlobDetector 找候选 + 对比度验证。"""
        keypoints = self._blob_detector.detect(gray)
        debug["raw_blob_count"] = len(keypoints)
        debug["total_contours"] = len(keypoints)

        h, w = gray.shape
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        candidates = []
        for kp in keypoints:
            cx, cy = kp.pt
            r = kp.size / 2.0
            area = np.pi * r * r

            # 计算对比度（供评分使用，不作硬门槛）
            contrast, inner_mean, outer_mean = self._compute_contrast(gray, cx, cy, r)

            candidates.append({
                "cx": cx, "cy": cy, "radius": r, "area": area,
                "circularity": 0.0,
                "contrast": contrast, "inner_mean": inner_mean, "outer_mean": outer_mean,
                "source": "blob",
            })
            cv2.circle(binary_mask, (int(cx), int(cy)), max(2, int(r)), 255, -1)

        debug["binary_mask"] = binary_mask

        # 轮廓预览 — 保留各方法自己的原始输出
        contour_canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for kp in keypoints:
            cv2.drawKeypoints(contour_canvas, [kp], np.array([]),
                              (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for c in candidates:
            cv2.circle(contour_canvas, (int(c["cx"]), int(c["cy"])),
                       int(c["radius"]), (0, 0, 255), 2)
        debug["contour_preview"] = contour_canvas

        history_entry = {"method": "blob", "candidates": len(candidates), "raw": len(keypoints)}
        debug.setdefault("method_history", []).append(history_entry)
        return candidates, [history_entry]

    # ================================================================
    #  方法 2: Otsu 阈值 + 轮廓检测
    # ================================================================
    def _detect_threshold_contour(self, gray: np.ndarray, debug: dict) -> Tuple[list, list]:
        """Otsu 二值化 + 形态学 + 轮廓筛选。

        对于大球模式，使用更大的形态学闭运算 kernel 合并破碎轮廓。
        circularity 不作为硬拒绝条件，而是在评分中处理。
        """
        k = self.gaussian_blur_ksize
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学操作：闭运算合并破碎轮廓
        ks = self.morph_kernel_size
        if self.large_ball_mode:
            # 大球模式用更大的 kernel
            ks = max(ks, 5)
            if ks % 2 == 0:
                ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if self.large_ball_mode:
            # 再额外做一次闭运算合并更大范围的破碎轮廓
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks + 2, ks + 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)

        debug["binary_mask"] = binary

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug["total_contours"] = len(contours)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px or area > self.max_area_px:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 1:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0

            # ★ 关键改动：大球模式下 circularity 不硬拒绝，留到评分环节处理
            if not self.large_ball_mode and circularity < self.min_circularity:
                continue

            # 计算对比度
            contrast, inner_mean, outer_mean = self._compute_contrast(gray, cx, cy, radius)

            candidates.append({
                "cx": cx, "cy": cy, "radius": radius, "area": area,
                "circularity": circularity,
                "contrast": contrast, "inner_mean": inner_mean, "outer_mean": outer_mean,
                "source": "threshold_contour",
            })

        contour_canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            cv2.drawContours(contour_canvas, [cnt], -1, (0, 255, 0), 1)
        for c in candidates:
            cv2.circle(contour_canvas, (int(c["cx"]), int(c["cy"])),
                       int(c["radius"]), (0, 0, 255), 2)
        debug["contour_preview"] = contour_canvas

        history_entry = {"method": "threshold_contour", "candidates": len(candidates),
                         "raw": len(contours)}
        debug.setdefault("method_history", []).append(history_entry)
        return candidates, [history_entry]

    # ================================================================
    #  方法 3: 背景差分法
    # ================================================================
    def _detect_background_subtraction(self, gray: np.ndarray, debug: dict) -> Tuple[list, list]:
        """背景差分 + 自适应阈值 + 轮廓筛选。"""
        if self.background is None:
            history_entry = {"method": "background_subtraction", "candidates": 0,
                             "error": "no background model"}
            debug.setdefault("method_history", []).append(history_entry)
            return [], [history_entry]

        # 提取灰度背景
        if len(self.background.shape) == 3:
            bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        else:
            bg_gray = self.background

        # 统一 dtype
        if gray.dtype != bg_gray.dtype:
            bg_gray = bg_gray.astype(gray.dtype)

        # 通道数检查
        if bg_gray.ndim != 2 or gray.ndim != 2:
            history_entry = {
                "method": "background_subtraction", "candidates": 0,
                "error": f"channel mismatch bg={bg_gray.ndim}d gray={gray.ndim}d; both must be single-channel",
            }
            debug.setdefault("method_history", []).append(history_entry)
            debug["bg_sub_skipped"] = "channel mismatch; both must be single-channel"
            return [], [history_entry]

        # shape 保护
        if bg_gray.shape != gray.shape:
            history_entry = {
                "method": "background_subtraction", "candidates": 0,
                "error": f"shape mismatch bg={bg_gray.shape} gray={gray.shape}; rebuild background required",
            }
            debug.setdefault("method_history", []).append(history_entry)
            debug["bg_sub_skipped"] = "shape mismatch; rebuild background required"
            return [], [history_entry]

        diff = cv2.absdiff(gray, bg_gray)

        # 自适应阈值：基于差分图的均值+标准差
        diff_mean = diff.mean()
        diff_std = diff.std()
        thresh_val = max(15, int(diff_mean + diff_std * 1.5))
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        debug["binary_mask"] = binary
        debug["bg_sub_threshold"] = thresh_val

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug["total_contours"] = len(contours)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px or area > self.max_area_px:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 1:
                continue

            # 计算对比度（供评分使用，不作硬门槛）
            contrast, inner_mean, outer_mean = self._compute_contrast(gray, cx, cy, radius)

            candidates.append({
                "cx": cx, "cy": cy, "radius": radius, "area": area,
                "circularity": 0.0,
                "contrast": contrast, "inner_mean": inner_mean, "outer_mean": outer_mean,
                "source": "bg_sub",
            })

        contour_canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            cv2.drawContours(contour_canvas, [cnt], -1, (0, 255, 0), 1)
        for c in candidates:
            cv2.circle(contour_canvas, (int(c["cx"]), int(c["cy"])),
                       int(c["radius"]), (0, 0, 255), 2)
        debug["contour_preview"] = contour_canvas

        history_entry = {"method": "background_subtraction", "candidates": len(candidates),
                         "raw": len(contours), "threshold": thresh_val}
        debug.setdefault("method_history", []).append(history_entry)
        return candidates, [history_entry]

    # ================================================================
    #  方法 4: 局部自适应阈值 + 闭运算
    # ================================================================
    def _detect_adaptive_threshold(self, gray: np.ndarray, debug: dict) -> Tuple[list, list]:
        """局部自适应阈值 + 形态学闭运算 + 轮廓筛选。

        对大球模式下轮廓破碎/反光的情况更鲁棒。
        仅在 large_ball_mode 或检测区域较亮时使用。
        """
        h, w = gray.shape
        candidates = []

        # 自适应阈值参数
        block_size = 21  # 必须为奇数
        c_value = -5     # 使阈值更严格（暗区域更易被选中）

        # 高斯模糊预处理
        k = self.gaussian_blur_ksize
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 两种阈值方式
        binary_sets = []

        # 方式1: 均值自适应阈值
        th1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
        binary_sets.append(("adaptive_mean", th1))

        # 方式2: 高斯自适应阈值
        th2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value
        )
        binary_sets.append(("adaptive_gaussian", th2))

        # 方式3: 局部对比度增强后 Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        _, th3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_sets.append(("clahe_otsu", th3))

        # 形态学闭运算合并破碎轮廓
        ks = self.morph_kernel_size
        if ks % 2 == 0:
            ks += 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, ks - 2), max(3, ks - 2)))

        for method_name, binary in binary_sets:
            # 闭运算 + 开运算
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            debug[f"adaptive_contours_{method_name}"] = len(contours)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area_px or area > self.max_area_px:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if radius < 1:
                    continue

                # 不按圆度硬过滤
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0

                contrast, inner_mean, outer_mean = self._compute_contrast(gray, cx, cy, radius)

                candidates.append({
                    "cx": cx, "cy": cy, "radius": radius, "area": area,
                    "circularity": circularity,
                    "contrast": contrast, "inner_mean": inner_mean, "outer_mean": outer_mean,
                    "source": f"adaptive_{method_name}",
                })

        # 去重
        if candidates:
            candidates = self._deduplicate_candidates(candidates, distance_threshold=4.0)

        hist_entry = {"method": "adaptive_threshold", "candidates": len(candidates),
                      "raw": len(candidates)}
        debug.setdefault("method_history", []).append(hist_entry)
        return candidates, [hist_entry]

    # ================================================================
    #  方法 5: HoughCircles 后备
    # ================================================================
    def _detect_hough_circles(self, gray: np.ndarray, debug: dict) -> Tuple[list, list]:
        """HoughCircles 圆形检测作为后备。

        仅在大球模式下启用，且只在预测位置附近局部搜索以避免误检。
        """
        candidates = []

        if not self.large_ball_mode:
            debug["hough_skipped"] = "not_large_ball_mode"
            hist_entry = {"method": "hough_circles", "candidates": 0, "error": "only in large_ball_mode"}
            debug.setdefault("method_history", []).append(hist_entry)
            return [], [hist_entry]

        h, w = gray.shape
        search_center_x = w // 2
        search_center_y = h // 2
        search_radius = min(w, h) // 3

        # 如果有预测位置，在预测位置附近搜索
        if self._search_center is not None:
            scx, scy = self._search_center
            # 确保搜索中心在图像范围内
            if 0 <= scx < w and 0 <= scy < h:
                search_center_x = int(scx)
                search_center_y = int(scy)
                search_radius = self.search_win_y_down

        # 局部 ROI 裁剪
        x0 = max(0, search_center_x - search_radius)
        x1 = min(w, search_center_x + search_radius)
        y0 = max(0, search_center_y - search_radius)
        y1 = min(h, search_center_y + search_radius)

        if x1 - x0 < 20 or y1 - y0 < 20:
            hist_entry = {"method": "hough_circles", "candidates": 0, "error": "search region too small"}
            debug.setdefault("method_history", []).append(hist_entry)
            return [], [hist_entry]

        roi_gray = gray[y0:y1, x0:x1]

        # 高斯模糊降低噪点
        k = self.gaussian_blur_ksize
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(roi_gray, (k, k), 0)

        # 动态 Hough 参数
        min_radius = max(3, int(self.expected_radius_px_min * 0.6))
        max_radius = max(min_radius + 2, int(self.expected_radius_px_max * 1.5))
        dp = 1.2  # 累加器分辨率
        min_dist = max(10, int(self.expected_radius_px_min * 3))
        param1 = 50  # Canny 高阈值
        param2 = 15  # 累加器阈值（越低越敏感）

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )

        if circles is not None and len(circles.shape) >= 2:
            circles = np.round(circles[0, :]).astype("int")
            for (cx_roi, cy_roi, r) in circles:
                cx_global = cx_roi + x0
                cy_global = cy_roi + y0
                area = np.pi * r * r

                if area < self.min_area_px or area > self.max_area_px:
                    continue

                contrast, inner_mean, outer_mean = self._compute_contrast(gray, cx_global, cy_global, r)

                candidates.append({
                    "cx": float(cx_global), "cy": float(cy_global),
                    "radius": float(r), "area": area,
                    "circularity": 0.85,
                    "contrast": contrast, "inner_mean": inner_mean, "outer_mean": outer_mean,
                    "source": "hough",
                })

        debug["hough_circles_count"] = len(candidates)
        hist_entry = {"method": "hough_circles", "candidates": len(candidates)}
        debug.setdefault("method_history", []).append(hist_entry)
        return candidates, [hist_entry]

    # ================================================================
    #  下落中心线
    # ================================================================
    def _get_fall_axis_x_roi(self, roi_offset_x: int, roi_w: int) -> float:
        """获取 ROI 坐标系中的下落中心线 x 坐标。"""
        if self.fall_axis_x is not None:
            return self.fall_axis_x - roi_offset_x
        # 默认 ROI 中心
        return roi_w / 2.0

    def _compute_axis_score_soft(self, cx: float, axis_x: float) -> float:
        """计算中心线接近度评分 (0~15) — 软评分，不硬排除。

        距中心线 < deviation: 15→5 线性衰减
        距中心线 deviation~2x: 5→0 线性衰减
        距中心线 > 2x deviation: 0（通过低总分自然排除）
        """
        dist = abs(cx - axis_x)
        d_max = self.allowed_axis_deviation_px
        if dist <= d_max:
            return 15.0 - (dist / d_max) * 10.0
        elif dist <= 2.0 * d_max:
            return 5.0 * (1.0 - (dist - d_max) / d_max)
        else:
            return 0.0

    def _compute_predicted_position_score(self, cx: float, cy: float) -> float:
        """计算预测位置评分 (0~25)。

        有预测位置：距预测位置越近越高
        无预测（首帧）：中性 12.5
        """
        if self._search_center is None:
            return 12.5
        scx, scy = self._search_center
        dist = np.hypot(cx - scx, cy - scy)
        max_dist = self.max_jump_px
        if dist < max_dist:
            return max(0.0, 25.0 * (1.0 - dist / max_dist))
        else:
            return max(0.0, 5.0 - (dist - max_dist) / max_dist * 5.0)

    def _compute_size_score(self, r: float) -> float:
        """计算尺寸评分 (0~25)，仅用于综合评分排序，不做硬过滤。

        使用高斯型衰减，sigma 取 expected_radius 的 0.6 倍以容忍运动模糊/折射。
        大球因折射/反光可能半径偏大，不设硬截止。
        """
        r_min = self.expected_radius_px_min
        r_max = self.expected_radius_px_max
        r_expected = (r_min + r_max) / 2.0
        sigma = max(1.0, 0.60 * r_expected)
        # 高斯型评分（宽松，远离预期尺寸逐渐衰减，无硬截止）
        score = 25.0 * np.exp(-0.5 * ((r - r_expected) / sigma) ** 2)
        return max(0.0, score)

    # ================================================================
    #  运动连续性评分（新增）
    # ================================================================
    def _compute_motion_continuity(self, cx: float, cy: float) -> float:
        """评估候选点的运动连续性 (0~10)。

        惩罚：
          - x 方向横向跳动
          - y 方向明显上跳（违反下落单调性）
          - 速度大幅偏离预期
        """
        score = 5.0  # 中性值

        if self._prev_cx is None or self._prev_cy is None:
            return score  # 首帧无历史，给中性分

        # 1. x 方向稳定性 (0~2.5)
        dx = abs(cx - self._prev_cx)
        if dx < 10:
            score += 2.5
        elif dx > 50:
            score -= 2.5

        # 2. y 方向单调性 (-3~+1.5): 小球应下落 (y 增加)
        dy = cy - self._prev_cy
        if dy < -20:
            score -= 3.0   # 明显上跳 → 重扣
        elif dy < -5:
            score -= 1.0   # 轻微上跳 → 轻扣
        elif dy > 0:
            score += 1.5   # 正常下落 → 奖励

        # 3. 速度一致性 (0~2)
        if abs(self._velocity_y) > 0.5:
            expected_dy = self._velocity_y * 1.0  # dt = 1 帧
            dy_deviation = abs(dy - expected_dy)
            if dy_deviation < expected_dy * 0.3:
                score += 2.0    # 接近预期速度 → 奖励
            elif dy_deviation > expected_dy * 2.0:
                score -= 2.0    # 速度异常 → 扣分

        return max(0.0, min(10.0, score))

    # ================================================================
    #  长线结构排除（新增）
    # ================================================================
    def _reject_long_line_v2(self, gray: np.ndarray,
                             cx: float, cy: float,
                             radius: float) -> Tuple[float, str]:
        """检查候选点是否在长线结构上（v4: 仅扣分，不硬排除）。

        核心改进：识别候选中心像素所在的连通域，
        只有该连通域自身是长线才重扣，旁边有刻度线仅轻扣。

        Returns:
            (penalty, reason)  — penalty 上限 20，不再返回 hard-reject bool
        """
        if not self.enable_long_line_rejection:
            return 0.0, ""

        h, w = gray.shape
        patch_size = 41
        half = patch_size // 2
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + half + 1)
        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + half + 1)

        patch = gray[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        if ph < 10 or pw < 10:
            return 0.0, ""

        # 二值化提取暗结构
        _, binary = cv2.threshold(patch, 100, 255, cv2.THRESH_BINARY_INV)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 找到候选中心像素所在的连通域标签
        center_px, center_py = half, half
        if 0 <= center_py < labels.shape[0] and 0 <= center_px < labels.shape[1]:
            own_label = labels[center_py, center_px]
        else:
            own_label = 0  # 背景

        own_penalty = 0.0
        own_reason = ""
        nearby_penalty = 0.0
        nearby_reason = ""

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1:
                continue
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            cx_i, cy_i = centroids[i]

            # 只分析离候选中心足够近的连通域
            dist_to_center = np.hypot(cx_i - center_px, cy_i - center_py)
            if dist_to_center > max(radius * 3, 10):
                continue

            is_own = (i == own_label)

            # ---- 长宽比检查 ----
            if min(width, height) > 0:
                aspect = max(width, height) / min(width, height)
                if aspect > 3.0:
                    if is_own:
                        # 候选自身是长线 → 重扣
                        own_penalty = max(own_penalty, 20.0)
                        own_reason = f"own_is_line(aspect={aspect:.1f})"
                    else:
                        # 旁边有长线 → 轻扣
                        nearby_penalty = max(nearby_penalty, 5.0)
                        nearby_reason = f"nearby_line(aspect={aspect:.1f})"
                elif aspect > 2.0:
                    pen = (aspect - 2.0) * 10.0
                    if is_own and pen > own_penalty:
                        own_penalty = pen
                        own_reason = f"own_aspect={aspect:.1f}"
                    elif not is_own and pen > nearby_penalty:
                        nearby_penalty = pen
                        nearby_reason = f"nearby_aspect={aspect:.1f}"

            # ---- 接触边界 ----
            touches = (left <= 1 or top <= 1 or
                       left + width >= pw - 2 or
                       top + height >= ph - 2)
            if touches and area > 15 and is_own:
                pen = min(8.0, area * 0.2)
                if pen > own_penalty:
                    own_penalty = pen
                    own_reason = f"touches_boundary(area={area})"

            # ---- 水平延展（仅对自身） ----
            if is_own and height > 0 and width > height * 3 and width > 10:
                own_penalty = max(own_penalty, 18.0)
                own_reason = f"horizontal_streak(w={width},h={height})"

        # ---- Sobel 边缘检查 ----
        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        center_region = grad[max(0, half-5):min(ph, half+6),
                             max(0, half-5):min(pw, half+6)]
        if center_region.size > 0 and center_region.max() > 80:
            h_edges = np.abs(sobel_y[half-4:half+5, half-4:half+5]).mean()
            v_edges = np.abs(sobel_x[half-4:half+5, half-4:half+5]).mean()
            edge_pen = min(5.0, max(h_edges, v_edges) * 0.1)
            if edge_pen > own_penalty and edge_pen > nearby_penalty:
                nearby_penalty = edge_pen
                nearby_reason = f"strong_edge(h={h_edges:.0f},v={v_edges:.0f})"

        # ---- 形态学直线检测 ----
        h_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        v_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        h_lines = cv2.morphologyEx(patch, cv2.MORPH_OPEN, h_line_kernel)
        v_lines = cv2.morphologyEx(patch, cv2.MORPH_OPEN, v_line_kernel)

        near_h = np.count_nonzero(h_lines[half-4:half+5, half-4:half+5]) > 8
        near_v = np.count_nonzero(v_lines[half-4:half+5, half-4:half+5]) > 8
        morph_pen = 3.0 if (near_h or near_v) else 0.0

        # 汇总：自身问题权重高，旁边问题权重低
        total_penalty = own_penalty + nearby_penalty * 0.5 + morph_pen
        total_penalty = min(20.0, total_penalty)

        if own_reason:
            reason = own_reason
        elif nearby_reason:
            reason = nearby_reason
        elif morph_pen > 0:
            reason = f"on_line(h={int(near_h)},v={int(near_v)})"
        else:
            reason = ""

        return total_penalty, reason

    # ================================================================
    #  综合评分（v5 运动优先 + 背景差分）
    # ================================================================
    def _score_candidates(self, candidates: list, roi_w: int,
                          gray: np.ndarray = None,
                          axis_x: float = None,
                          diff_map: np.ndarray = None) -> list:
        """评分选最优（v5 背景差分主导）。

        评分策略:
          - diff_score (0~25): 背景差分值 → 运动的候选高分，静止的(刻度线/管壁)低分
          - size_score (0~25): 尺寸匹配度（高斯型衰减，远离预期尺寸严重扣分）
          - pred_score (0~20): 预测位置接近度
          - continuity (0~15): 运动连续性(方向/速度)
          - iso_score (0~15): 孤立小黑点评分
          - ax_score (0~10): 中心线接近度
          - cb_score (0~10): 中心运动带
          - motion_bonus (0~5): bg_sub 来源奖励
          - contrast_score (0~5): 对比度
          - ll_penalty (-20~0): 长线结构扣分

        总分上限约 130，真正的小球通常 ≥ 75 分。
        """
        for c in candidates:
            cx = float(c["cx"])
            cy = float(c["cy"])
            r = float(c["radius"])
            is_motion = c.get("source") == "bg_sub"

            # 0. 背景差分评分 (0~25) — 运动的才是真正的小球
            diff_score = 0.0
            if diff_map is not None:
                # 以候选中心周围取小窗计算平均 diff
                hs = max(3, int(r * 1.5))
                x0 = max(0, int(cx) - hs)
                x1 = min(diff_map.shape[1], int(cx) + hs + 1)
                y0 = max(0, int(cy) - hs)
                y1 = min(diff_map.shape[0], int(cy) + hs + 1)
                patch = diff_map[y0:y1, x0:x1]
                if patch.size > 0:
                    mean_diff = float(patch.mean())
                    # 经验阈值: diff > 15 有明显运动; < 5 基本静止
                    if mean_diff > 15:
                        diff_score = min(25.0, 15.0 + mean_diff * 0.4)
                    elif mean_diff > 5:
                        diff_score = mean_diff * 1.0
                    else:
                        diff_score = max(0.0, mean_diff * 0.5)
            c["diff_score"] = diff_score
            c["mean_diff"] = float(patch.mean()) if diff_map is not None and patch.size > 0 else 0.0

            # 1. 预测位置评分 (0~20)
            pred_score = self._compute_predicted_position_score(cx, cy)
            c["prediction_score"] = pred_score

            # 2. 运动连续性评分 (0~15): 方向/速度/稳定性
            continuity = self._compute_motion_continuity(cx, cy)
            c["motion_continuity"] = continuity

            # 3. 孤立小黑点评分 (0~15)
            iso_score = 7.5
            if gray is not None:
                iso_score = min(15.0, self._compute_isolation_score(gray, cx, cy) * 1.0)
            c["isolation_score"] = iso_score

            # 4. 尺寸分 (0~25) — 高斯型衰减，远离预期尺寸严重扣分
            size_score = self._compute_size_score(r)
            c["size_score"] = size_score

            # 5. 中心线接近度 (0~10) — 软评分
            ax_score = 5.0
            if axis_x is not None:
                ax_score = self._compute_axis_score_soft(cx, axis_x) * 0.67
            c["axis_score"] = ax_score

            # 6. 中心运动带 (0~10)
            cb_score = self._compute_center_band_score(cx, roi_w) / 3.0
            if not is_motion:
                cb_score *= 0.5
            c["center_band_score"] = cb_score

            # 7. 运动来源奖励 (0~5) — bg_sub 方法额外奖励
            motion_bonus = 5.0 if is_motion else 0.0
            c["motion_bonus"] = motion_bonus

            # 8. 对比度 (0~10) — 大球模式提高对比度权重
            contrast = c.get("contrast", 0)
            contrast_pass = contrast >= self.contrast_threshold
            if self.large_ball_mode:
                # 大球模式：对比度放宽 + 权重翻倍
                contrast_pass = contrast >= (self.contrast_threshold * 0.7)
                contrast_score = min(contrast * 0.15, 10.0) if contrast_pass else max(0, contrast * 0.03)
            else:
                contrast_score = min(contrast * 0.08, 5.0) if contrast_pass else 0.0
            c["contrast_score"] = contrast_score
            c["contrast_pass"] = contrast_pass

            # 9. 长线扣分 (-20~0)
            ll_penalty = 0.0
            ll_reason = ""
            if gray is not None and self.enable_long_line_rejection:
                ll_penalty, ll_reason = self._reject_long_line_v2(gray, cx, cy, r)
            c["long_line_penalty"] = ll_penalty

            # ---- 被静态/结构剔除（低 diff + 长线 + 低 continuity） ----
            reject_reasons = []
            if diff_map is not None:
                if diff_score < 2.0 and not self.large_ball_mode:
                    reject_reasons.append("static_background")
                elif diff_score < 1.0 and self.large_ball_mode:
                    reject_reasons.append("static_background")
            if ll_penalty > 15:
                reject_reasons.append("on_line_structure")
            if self._prev_cx is not None:
                if continuity < 1.0:
                    reject_reasons.append("bad_motion")
                elif continuity < 2.0 and not self.large_ball_mode:
                    reject_reasons.append("bad_motion")
            if contrast_pass is False and diff_score < 5.0 and not self.large_ball_mode:
                reject_reasons.append("low_contrast+static")
            c["reject_reason"] = "; ".join(reject_reasons) if reject_reasons else ""

            # ---- 总分（大球模式适当调低总分要求） ----
            raw = (diff_score + pred_score + continuity + iso_score
                   + size_score + ax_score + cb_score
                   + motion_bonus + contrast_score - ll_penalty)
            # 大球模式给额外补正（因为很多子项天然偏低）
            if diff_map is not None and diff_score < 2.0:
                raw -= 6.0
            if self.large_ball_mode:
                raw += 10.0
            c["score"] = max(0.0, raw)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # ================================================================
    #  局部亮度判断
    # ================================================================
    def _check_local_brightness(self, gray: np.ndarray,
                                cx: float, cy: float) -> bool:
        """检查候选点周围局部区域是否为亮背景。"""
        half = self.local_window // 2
        x0 = max(0, int(cx) - half)
        x1 = min(gray.shape[1], int(cx) + half + 1)
        y0 = max(0, int(cy) - half)
        y1 = min(gray.shape[0], int(cy) + half + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        bright_ratio = np.count_nonzero(patch > self.local_bright_threshold) / patch.size
        return bright_ratio > self.local_bright_ratio

    # ================================================================
    #  对比度计算
    # ================================================================
    def _compute_contrast(self, gray: np.ndarray,
                          cx: float, cy: float,
                          radius: float) -> Tuple[float, float, float]:
        """计算内圈/外圈对比度。"""
        h, w = gray.shape
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        r_inner = max(1, int(radius * self.inner_ratio))
        cv2.circle(mask_inner, (int(cx), int(cy)), r_inner, 255, -1)
        r_outer_min = int(radius) + self.outer_ring_min
        r_outer_max = int(radius) + self.outer_ring_max
        cv2.circle(mask_outer, (int(cx), int(cy)), r_outer_max, 255, -1)
        cv2.circle(mask_outer, (int(cx), int(cy)), r_outer_min, 0, -1)
        inner_pixels = gray[mask_inner > 0]
        outer_pixels = gray[mask_outer > 0]
        inner_mean = float(inner_pixels.mean()) if len(inner_pixels) > 0 else 0.0
        outer_mean = float(outer_pixels.mean()) if len(outer_pixels) > 0 else 0.0
        return outer_mean - inner_mean, inner_mean, outer_mean

    # ================================================================
    #  中心运动带评分
    # ================================================================
    def _compute_center_band_score(self, cx: float, roi_w: int) -> float:
        """计算中心运动带评分 (0~30)。"""
        if not self.center_band_enabled or roi_w <= 0:
            return 15.0

        left = roi_w * self.center_band_left_ratio
        right = roi_w * self.center_band_right_ratio

        if left <= cx <= right:
            center = (left + right) / 2.0
            max_dist = (right - left) / 2.0
            dist = abs(cx - center)
            ratio = dist / max_dist if max_dist > 0 else 0
            return max(10.0, 30.0 - ratio * 20.0)
        else:
            dist_to_band = min(abs(cx - left), abs(cx - right))
            return max(0.0, 10.0 - dist_to_band / (roi_w * 0.1) * 10.0)

    # ================================================================
    #  孤立小黑点评分
    # ================================================================
    def _compute_isolation_score(self, gray: np.ndarray,
                                 cx: float, cy: float) -> float:
        """判断候选点是否为孤立小黑点（非刻度线/管壁边缘）(0~15)。"""
        patch_size = 21
        half = patch_size // 2
        x0 = max(0, int(cx) - half)
        x1 = min(gray.shape[1], int(cx) + half + 1)
        y0 = max(0, int(cy) - half)
        y1 = min(gray.shape[0], int(cy) + half + 1)

        patch = gray[y0:y1, x0:x1]
        if patch.size < patch_size * patch_size * 0.5:
            return 7.5

        _, binary = cv2.threshold(patch, 100, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            return 0.0

        center = patch_size // 2
        best_comp = -1
        best_dist = float('inf')
        best_area = 0
        for i in range(1, num_labels):
            cx_i, cy_i = centroids[i]
            dist = np.hypot(cx_i - center, cy_i - center)
            a = stats[i, cv2.CC_STAT_AREA]
            if dist < best_dist:
                best_dist = dist
                best_comp = i
                best_area = a

        if best_comp < 0:
            return 0.0

        left = stats[best_comp, cv2.CC_STAT_LEFT]
        top = stats[best_comp, cv2.CC_STAT_TOP]
        width = stats[best_comp, cv2.CC_STAT_WIDTH]
        height = stats[best_comp, cv2.CC_STAT_HEIGHT]
        ph, pw = patch.shape[:2]

        score = 15.0

        touches = (left <= 1 or top <= 1 or
                   left + width >= pw - 2 or
                   top + height >= ph - 2)
        if touches:
            score -= 5.0

        if min(width, height) > 0:
            aspect = max(width, height) / min(width, height)
            if aspect > 1.5:
                score -= min(6.0, (aspect - 1.5) * 3.0)

        center_ratio = best_dist / (patch_size / 2)
        score -= center_ratio * 4.0

        sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        center_region = grad[half - 3:half + 4, half - 3:half + 4]
        if center_region.max() > 60:
            score -= 3.0

        return max(0.0, score)

    # ================================================================
    #  失败诊断
    # ================================================================
    def _diagnose_failure(self, debug: dict, gray: np.ndarray) -> str:
        """诊断识别失败原因。"""
        method_history = debug.get("method_history", [])
        if not method_history:
            return "检测方法未返回任何候选点。"

        lines = []
        for entry in method_history:
            m = entry["method"]
            n_cand = entry.get("candidates", 0)
            n_raw = entry.get("raw", 0)
            err = entry.get("error", "")
            if err:
                lines.append(f"  {m}: {err}")
            else:
                lines.append(f"  {m}: {n_raw} 原始候选 → {n_cand} 通过")

        sw_filtered = debug.get("search_window_filtered", 0)
        if sw_filtered:
            lines.append(f"  搜索窗口过滤: {sw_filtered} 个")
        axis_rejected = debug.get("axis_rejected", 0)
        if axis_rejected:
            lines.append(f"  中心线过滤: {axis_rejected} 个")

        lines.append("")
        lines.append("  建议:")
        if gray.size > 0:
            mean_gray = gray.mean()
            if mean_gray < 30:
                lines.append("    - ROI 区域过暗 (均值 {:.0f})".format(mean_gray))
            elif mean_gray > 230:
                lines.append("    - ROI 区域过亮 (均值 {:.0f})".format(mean_gray))
            else:
                lines.append("    - 检查「面积范围」(min={}, max={})".format(
                    self.min_area_px, self.max_area_px))
                lines.append("    - 降低「最小圆度」({})".format(self.min_circularity))
                lines.append("    - 降低「对比度阈值」({})".format(self.contrast_threshold))
                lines.append("    - 确认 ROI 框住了管子亮区域")

        return "\n".join(lines)

    # ================================================================
    #  构造返回 dict
    # ================================================================
    def _make_result(self, found: bool = False,
                     fallback: bool = False,
                     x_px: float = None, y_px: float = None,
                     radius_px: float = None, area_px: float = None,
                     circularity: float = None, confidence: float = None,
                     candidates: list = None,
                     fail_reason: str = "",
                     roi_offset: Tuple[int, int] = (0, 0),
                     debug: dict = None,
                     thresh_val: float = None,
                     method_info: str = "") -> dict:
        return {
            "found": found,
            "fallback": fallback,
            "x_px": x_px,
            "y_px": y_px,
            "radius_px": radius_px,
            "area_px": area_px,
            "circularity": circularity,
            "confidence": confidence,
            "method_used": method_info,
            "threshold_value": thresh_val,
            "roi": self.roi,
            "roi_offset": roi_offset,
            "candidates": candidates or [],
            "fail_reason": fail_reason,
            "debug_images": debug or {},
            "pre_search_candidates": (debug or {}).get("pre_search_candidates", 0),
            "search_window_active": (debug or {}).get("search_window_active", False),
        }

    # ================================================================
    #  reset / set_prev / search window
    # ================================================================
    def reset(self):
        """重置所有追踪状态。"""
        self._prev_cx = None
        self._prev_cy = None
        self._search_center = None
        self._predicted_cx = None
        self._predicted_cy = None
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._consecutive_lost = 0

    def set_prev(self, cx: float, cy: float):
        """设置上一帧的检测结果，更新速度估计并预测下一帧搜索中心。"""
        if self._prev_cx is not None and self._prev_cy is not None:
            self._velocity_x = cx - self._prev_cx
            self._velocity_y = cy - self._prev_cy

        self._prev_cx = cx
        self._prev_cy = cy

        # 预测下一帧位置作为搜索窗口中心
        pred_x, pred_y = self.predict_next_position()
        if pred_x is not None:
            self._predicted_cx = pred_x
            self._predicted_cy = pred_y
            self._search_center = (pred_x, pred_y)
        else:
            self._search_center = (cx, cy)

        self._consecutive_lost = 0

    def get_prev(self) -> Tuple[Optional[float], Optional[float]]:
        return self._prev_cx, self._prev_cy

    def set_search_center(self, cx: float, cy: float):
        """设置搜索窗口中心（用于追踪模式）。"""
        self._search_center = (cx, cy)

    def clear_search_center(self):
        """清除搜索窗口，回到全 ROI 搜索。"""
        self._search_center = None

    def lost_frame(self):
        """通知检测器上一帧追踪丢失（用于 tracking.py 异常拒绝同步）。

        递增连续丢失计数，不更新 prev 位置，让检测器知道需要扩大搜索。
        """
        self._consecutive_lost += 1

    def set_search_window(self, w: int, y_up: int, y_down: int):
        """动态调整搜索窗口大小（用于丢帧后扩大搜索）。"""
        self.search_win_w = w
        self.search_win_y_up = y_up
        self.search_win_y_down = y_down

    def predict_next_position(self) -> Tuple[Optional[float], Optional[float]]:
        """基于当前位置和速度预测下一帧位置。"""
        if self._prev_cx is None:
            return None, None
        dt = 1.0  # 假设 1 帧间隔
        pred_x = self._prev_cx + self._velocity_x * dt
        pred_y = self._prev_cy + self._velocity_y * dt
        return pred_x, pred_y

    def get_fall_axis_x(self) -> Optional[float]:
        """获取全局坐标中的下落中心线。"""
        return self.fall_axis_x

    def set_fall_axis_x(self, x: float):
        """设置下落中心线（全局坐标）。"""
        self.fall_axis_x = x
