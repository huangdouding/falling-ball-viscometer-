"""
ball_tracker.py — OpenCV 球心识别核心模块

功能管线:
  读取视频 → 标定 → 预处理 → 球心检测 → 输出坐标 CSV

用法:
  from ball_tracker import BallTracker
  tracker = BallTracker("videos/sample.mp4")
  tracker.run()
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np

import config


class BallTracker:
    """落球视频球心追踪器"""

    def __init__(self, video_path: str, cfg=None):
        self.video_path = Path(video_path)
        self.cfg = cfg or config

        # 视频属性（在 open_video 中填充）
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.frame_w = None
        self.frame_h = None

        # 标定
        self.scale = None               # mm/pixel

        # 背景模型
        self.background = None

        # 结果
        self.results = []               # [(frame_id, x, y, t, radius)]
        self.timestamps = []            # t 数组 (s)
        self.y_positions = []           # y 数组 (pixel)

    # ------------------------------------------------------------------
    # 视频 I/O
    # ------------------------------------------------------------------

    def open_video(self):
        """打开视频并读取基本属性"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[视频信息] {self.video_path.name}  "
              f"{self.frame_w}x{self.frame_h}  "
              f"{self.fps:.2f} fps  "
              f"{self.total_frames} 帧")

    def read_frame(self, frame_id: int) -> np.ndarray:
        """跳转到指定帧并读取"""
        if self.cap is None:
            raise RuntimeError("视频未打开，请先调用 open_video()")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration(f"无法读取第 {frame_id} 帧")
        return frame

    def close(self):
        if self.cap:
            self.cap.release()

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------

    def calibrate(self, known_mm: float = None, pixel_length: float = None):
        """
        标定：计算 mm/pixel 比例
        如果 pixel_length 未指定，在第一个视频帧上交互式选取标尺区域
        可后续替换为自动标定
        """
        known_mm = known_mm or self.cfg.CALIB_OBJECT_MM
        if pixel_length is not None:
            self.scale = known_mm / pixel_length
        else:
            # 默认值，待视频到位后标定
            self.scale = known_mm / 500.0
        print(f"[标定] {self.scale:.4f} mm/pixel")
        return self.scale

    # ------------------------------------------------------------------
    # 图像预处理
    # ------------------------------------------------------------------

    def build_background(self, n_frames: int = 30):
        """取前 n_frames 中值作为背景"""
        frames = []
        for i in range(min(n_frames, self.total_frames)):
            try:
                frames.append(self.read_frame(i))
            except StopIteration:
                break
        if frames:
            self.background = np.median(np.array(frames), axis=0).astype(np.uint8)
            print(f"[背景] 使用前 {len(frames)} 帧建立背景模型")
        # 回到开头
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理管线:
          灰度 → 高斯模糊 → 背景扣除(可选) → 阈值(二值化)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        k = self.cfg.GAUSSIAN_BLUR_KSIZE
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 背景扣除
        if self.cfg.BG_SUBTRACT and self.background is not None:
            bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
            fg = cv2.absdiff(blurred, bg_gray)
        else:
            fg = blurred

        # 二值化
        method = self.cfg.THRESH_METHOD
        if method == "fixed":
            _, thresh = cv2.threshold(fg, self.cfg.THRESH_FIXED_VALUE, 255, cv2.THRESH_BINARY)
        elif method == "otsu":
            _, thresh = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            thresh = cv2.adaptiveThreshold(
                fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 4
            )
        else:
            raise ValueError(f"未知阈值方法: {method}")

        return thresh

    # ------------------------------------------------------------------
    # 球心检测
    # ------------------------------------------------------------------

    def detect_ball(self, thresh: np.ndarray) -> tuple:
        """
        在二值图中检测小球，返回 (x, y, radius) 或 (None, None, None)
        使用轮廓检测 + 圆形筛选
        """
        method = self.cfg.DETECT_METHOD

        if method == "contour":
            return self._detect_by_contour(thresh)
        elif method == "hough":
            return self._detect_by_hough(thresh)
        else:
            raise ValueError(f"未知检测方法: {method}")

    def _detect_by_contour(self, thresh: np.ndarray) -> tuple:
        """轮廓检测法：findContours → 面积/圆度筛选 → 最小外接圆"""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 忽略太小区域
                continue

            # 最小外接圆
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # 半径筛选
            r_min, r_max = self.cfg.MIN_BALL_RADIUS, self.cfg.MAX_BALL_RADIUS
            if not (r_min < radius < r_max):
                continue

            # 圆度筛选: 4π·面积 / 周长²
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < self.cfg.BALL_CIRCULARITY_MIN:
                    continue

            candidates.append((int(cx), int(cy), int(radius)))

        if not candidates:
            return None, None, None

        # 返回面积最大的候选者（应该就是小球）
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates[0]

    def _detect_by_hough(self, thresh: np.ndarray) -> tuple:
        """霍夫圆变换法（备选）"""
        circles = cv2.HoughCircles(
            thresh, cv2.HOUGH_GRADIENT,
            dp=1, minDist=50,
            param1=50, param2=30,
            minRadius=self.cfg.MIN_BALL_RADIUS,
            maxRadius=self.cfg.MAX_BALL_RADIUS
        )
        if circles is not None:
            cx, cy, r = circles[0][0]
            return int(cx), int(cy), int(r)
        return None, None, None

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        """执行完整追踪管线"""
        self.open_video()

        # 标定
        self.calibrate()

        # 构建背景
        if self.cfg.BG_SUBTRACT:
            self.build_background(n_frames=max(10, self.cfg.BG_FRAME_OFFSET))

        print("[开始] 逐帧识别小球...")
        self.results = []
        max_f = self.cfg.MAX_FRAMES or self.total_frames

        for f_id in range(self.cfg.SKIP_FRAMES, max_f):
            try:
                frame = self.read_frame(f_id)
            except StopIteration:
                break

            t = f_id / self.fps                        # 时间 (s)

            thresh = self.preprocess(frame)
            x, y, r = self.detect_ball(thresh)

            if x is not None:
                self.results.append((f_id, x, y, t, r))
                self.timestamps.append(t)
                self.y_positions.append(y)
            else:
                self.results.append((f_id, -1, -1, t, 0))   # 未检测到

            if (f_id + 1) % 100 == 0:
                print(f"  已处理 {f_id + 1}/{max_f} 帧 ...")

        self.close()
        print(f"[完成] 共处理 {len(self.results)} 帧, "
              f"识别成功率: "
              f"{sum(1 for r in self.results if r[1] >= 0) / len(self.results) * 100:.1f}%")

        self.save_csv()

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def save_csv(self, path: str = None):
        """保存球心坐标到 CSV"""
        if path is None:
            os.makedirs(self.cfg.DATA_DIR, exist_ok=True)
            path = os.path.join(self.cfg.DATA_DIR, self.cfg.CSV_FILENAME)

        header = ["frame", "x_px", "y_px", "t_s", "radius_px"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.results)
        print(f"[CSV] 已保存 → {path}")

    def get_y_mm(self) -> np.ndarray:
        """返回 y 坐标 (mm)，剔除未检测到的帧"""
        if self.scale is None:
            raise RuntimeError("未标定，请先调用 calibrate()")
        y_arr = np.array(self.y_positions)
        t_arr = np.array(self.timestamps)
        # 如果 y 原点在顶部，翻转使 y 向下为正
        y_mm = y_arr * self.scale
        return t_arr, y_mm

    def get_velocity(self, smooth: int = None):
        """中心差分计算速度 (mm/s)"""
        t, y = self.get_y_mm()
        v = np.gradient(y, t)
        # 平滑
        if smooth is None:
            smooth = self.cfg.SMOOTH_WINDOW
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            v = np.convolve(v, kernel, mode="same")
        return t, v


def demo():
    """快速测试：用单帧截图验证检测效果"""
    import sys
    if len(sys.argv) < 2:
        print("用法: python ball_tracker.py <视频路径>")
        return
    tracker = BallTracker(sys.argv[1])
    tracker.open_video()
    tracker.calibrate()

    # 取中间一帧测试
    mid_frame = tracker.total_frames // 2
    frame = tracker.read_frame(mid_frame)
    thresh = tracker.preprocess(frame)
    x, y, r = tracker.detect_ball(thresh)

    if x is not None:
        print(f"检测到球心: ({x}, {y}), 半径: {r} px")
        # 标注
        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

        from matplotlib import pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("原始帧")
        plt.subplot(122)
        plt.imshow(thresh, cmap="gray")
        plt.title("二值化")
        plt.tight_layout()
        plt.savefig("output/demo_detection.png", dpi=150)
        print("检测结果图已保存 → output/demo_detection.png")
    else:
        print("未检测到小球")


if __name__ == "__main__":
    demo()
