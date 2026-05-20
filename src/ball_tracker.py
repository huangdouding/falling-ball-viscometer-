"""
局部轨迹追踪器

将检测从"每帧全局目标检测"改为"基于物理约束的局部轨迹追踪"。

核心设计：
  1. 状态机：tracking → lost → reacquire → stopped
  2. Alpha-beta 滤波器预测下一帧位置
  3. 以预测位置为中心的局部搜索窗口（非对称，下方向更大）
  4. 物理约束：y 单调下降、速度上限、x 通道宽度
  5. 容忍短间隙（≤10 帧预测补位）

【状态机】
  tracking:   正常追踪，在预测位置附近搜索
  lost:       连续未检测到有效候选，用预测位置补位
  reacquire:  在 lost 多帧后扩大搜索窗口尝试重新捕获
  stopped:    超出底部边界或用户终止

【坐标约定】
  - x_px, y_px: 像素坐标 (y 向下为正)
  - velocity 向下为正
  - 所有搜索窗口单位为像素
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# —— 追踪状态 ——

class TrackerState:
    TRACKING = "tracking"
    LOST = "lost"
    REACQUIRE = "reacquire"
    STOPPED = "stopped"


@dataclass
class TrackedPoint:
    """单帧追踪结果。"""
    frame_idx: int
    x_px: float
    y_px: float
    state: str                      # tracking / lost / reacquire
    point_type: str                 # raw / predicted
    confidence: float = 1.0         # 0~1
    score: float = 0.0              # 原始检测评分
    circularity: float = 0.0
    fallback_reason: str = ""       # 补位原因


@dataclass
class TrackerConfig:
    """追踪器配置参数。"""
    # ── 搜索窗口（像素） ──
    search_win_w: int = 60          # 左右搜索半径
    search_win_y_up: int = 20       # 向上搜索（帧间下落不应向上）
    search_win_y_down: int = 120    # 向下搜索

    # ── 运动约束 ──
    max_jump_px: float = 80.0       # 帧间最大位移
    dx_max_px: float = 20.0         # 水平方向最大变化
    dy_back_tol_px: float = 5.0     # 允许向上"回跳"的最大像素

    # ── 丢帧策略 ──
    predict_max_frames: int = 10    # 最多连续预测补位帧数
    lost_threshold: int = 10        # 超过此帧数视为 lost
    reacquire_search_scale: float = 2.0  # lost 后搜索窗口放大倍数

    # ── Alpha-beta 滤波器 ──
    alpha: float = 0.7              # 位置平滑
    beta: float = 0.3               # 速度平滑

    # ── 终止条件 ──
    stop_y_ratio: float = 0.92      # 超出 ROI 底部比率时停止

    @classmethod
    def from_config(cls, config: dict) -> "TrackerConfig":
        return cls(
            search_win_w=config.get("search_win_w", 60),
            search_win_y_up=config.get("search_win_y_up", 20),
            search_win_y_down=config.get("search_win_y_down", 120),
            max_jump_px=float(config.get("max_jump_px", 80)),
            dx_max_px=float(config.get("dx_max", 20)),
            dy_back_tol_px=float(config.get("dy_back_tol", 5)),
            predict_max_frames=int(config.get("predict_max_frames", 10)),
            lost_threshold=int(config.get("lost_threshold", 10)),
            reacquire_search_scale=float(config.get("reacquire_search_scale", 2.0)),
            alpha=float(config.get("tracker_alpha", 0.7)),
            beta=float(config.get("tracker_beta", 0.3)),
            stop_y_ratio=float(config.get("stop_y_ratio", 0.92)),
        )


# —— 追踪器 ——

class BallTracker:
    """基于物理约束的局部轨迹追踪器。"""

    def __init__(self, config: TrackerConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = TrackerConfig.from_config(config)
        elif isinstance(config, TrackerConfig):
            self.cfg = config
        else:
            self.cfg = TrackerConfig()

        self._state = TrackerState.STOPPED

        # 当前位置/速度
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._vx: float = 0.0
        self._vy: float = 0.0          # 帧间速度 (px/frame)

        # 追踪历史
        self._trace: list[TrackedPoint] = []

        # 丢帧计数器
        self._lost_count: int = 0
        self._total_predicted: int = 0

        # 搜索窗口
        self._search_rect: list[int] = [0, 0, 0, 0]  # [x, y, w, h]

    # ── 属性 ──

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state != TrackerState.STOPPED

    @property
    def position(self) -> tuple[float, float]:
        return (self._cx, self._cy)

    @property
    def velocity(self) -> tuple[float, float]:
        """帧间速度 (px/frame)。vx, vy — vy 向下为正。"""
        return (self._vx, self._vy)

    @property
    def lost_count(self) -> int:
        return self._lost_count

    @property
    def trace_count(self) -> int:
        return len(self._trace)

    @property
    def raw_count(self) -> int:
        return sum(1 for p in self._trace if p.point_type == "raw")

    @property
    def search_rect(self) -> list[int]:
        """当前搜索矩形 [x, y, w, h]。"""
        return self._search_rect

    # ── 初始化 ──

    def start(self, cx: float, cy: float, frame_idx: int = 0):
        """以初始位置启动追踪。"""
        self._cx = cx
        self._cy = cy
        self._vx = 0.0
        self._vy = 0.0
        self._state = TrackerState.TRACKING
        self._lost_count = 0
        self._total_predicted = 0
        self._trace = []

        self._trace.append(TrackedPoint(
            frame_idx=frame_idx,
            x_px=cx, y_px=cy,
            state=TrackerState.TRACKING,
            point_type="raw",
            confidence=1.0,
        ))

        self._update_search_window()
        logger.info(
            "追踪器启动: 初始位置 (%.1f, %.1f) 帧 %d",
            cx, cy, frame_idx,
        )

    # ── 核心预测 ──

    def predict_next(self, dt: float = 1.0) -> tuple[float, float]:
        """预测下一帧位置。

        使用 alpha-beta 滤波器：
          pred_cx = cx + vx * dt
          pred_cy = cy + vy * dt
        """
        pred_cx = self._cx + self._vx * dt
        pred_cy = self._cy + self._vy * dt
        return (pred_cx, pred_cy)

    def update(self, detected_cx: float, detected_cy: float,
               frame_idx: int, score: float = 0.0,
               circularity: float = 0.0) -> TrackedPoint:
        """用检测到的位置更新追踪器。

        alpha-beta 平滑：
          cx += alpha * (detected_cx - pred_cx)
          cy += alpha * (detected_cy - pred_cy)
          vx += beta * (detected_cx - pred_cx) / dt
          vy += beta * (detected_cy - pred_cy) / dt
        """
        pred_cx, pred_cy = self.predict_next()

        # alpha-beta 平滑
        self._cx = pred_cx + self.cfg.alpha * (detected_cx - pred_cx)
        self._cy = pred_cy + self.cfg.alpha * (detected_cy - pred_cy)
        self._vx += self.cfg.beta * (detected_cx - pred_cx)
        self._vy += self.cfg.beta * (detected_cy - pred_cy)

        # 速度上限
        self._vx = np.clip(self._vx, -self.cfg.dx_max_px, self.cfg.dx_max_px)
        self._vy = max(0.0, self._vy)  # vy 不应为负（自由落体/tracking模式）

        self._state = TrackerState.TRACKING
        self._lost_count = 0

        pt = TrackedPoint(
            frame_idx=frame_idx,
            x_px=self._cx, y_px=self._cy,
            state=TrackerState.TRACKING,
            point_type="raw",
            confidence=1.0,
            score=score,
            circularity=circularity,
        )
        self._trace.append(pt)
        self._update_search_window()
        return pt

    def predict_fallback(self, frame_idx: int,
                         reason: str = "检测失败") -> TrackedPoint:
        """检测失败时，用预测位置补位。

        如果连续预测帧数超过阈值，状态变为 LOST/REACQUIRE。
        """
        pred_cx, pred_cy = self.predict_next()
        self._cx = pred_cx
        self._cy = pred_cy
        # vy 保持，不衰减（惯性预测）

        self._lost_count += 1
        self._total_predicted += 1

        if self._lost_count > self.cfg.lost_threshold:
            self._state = TrackerState.REACQUIRE
        elif self._lost_count > 3:
            self._state = TrackerState.LOST
        else:
            self._state = TrackerState.TRACKING

        pt = TrackedPoint(
            frame_idx=frame_idx,
            x_px=self._cx, y_px=self._cy,
            state=self._state,
            point_type="predicted",
            confidence=max(0.0, 1.0 - self._lost_count / self.cfg.lost_threshold),
            fallback_reason=reason,
        )
        self._trace.append(pt)

        if self._lost_count == 1:
            logger.info(
                "帧 %d: 首次丢失，用预测值补位 (%.1f, %.1f)",
                frame_idx, self._cx, self._cy,
            )
        elif self._lost_count % 5 == 0:
            logger.info(
                "帧 %d: 已连续丢失 %d 帧",
                frame_idx, self._lost_count,
            )

        self._update_search_window(expanded=self._state != TrackerState.TRACKING)
        return pt

    def stop(self, reason: str = "手动停止"):
        """停止追踪。"""
        self._state = TrackerState.STOPPED
        logger.info("追踪器停止: %s (已追踪 %d 帧)", reason, len(self._trace))

    # ── 搜索窗口 ──

    def _update_search_window(self, expanded: bool = False):
        """更新搜索矩形。

        非对称窗口：向下搜索范围 > 向上搜索范围。
        expanded 用于 reacquire 模式。
        """
        scale = self.cfg.reacquire_search_scale if expanded else 1.0
        w = int(self.cfg.search_win_w * scale)
        y_up = int(self.cfg.search_win_y_up * scale)
        y_down = int(self.cfg.search_win_y_down * scale)

        self._search_rect = [
            int(self._cx) - w,
            int(self._cy) - y_up,
            w * 2,
            y_up + y_down,
        ]

    def get_search_window(self, frame_width: int, frame_height: int,
                          expanded: bool = False) -> tuple[int, int, int, int]:
        """获取裁减到图像边界内的搜索窗口。

        Returns:
            (x, y, w, h) 可用于 frame[y:y+h, x:x+w]
        """
        scale = self.cfg.reacquire_search_scale if expanded else 1.0
        half_w = int(self.cfg.search_win_w * scale)
        y_up = int(self.cfg.search_win_y_up * scale)
        y_down = int(self.cfg.search_win_y_down * scale)

        x = max(0, int(self._cx) - half_w)
        y = max(0, int(self._cy) - y_up)
        x2 = min(frame_width, int(self._cx) + half_w)
        y2 = min(frame_height, int(self._cy) + y_down)
        return (x, y, x2 - x, y2 - y)

    # ── 物理约束检查 ──

    def check_physical(self, cx: float, cy: float) -> tuple[bool, str]:
        """检查检测候选是否满足物理约束。

        Returns:
            (is_ok: bool, reason: str)
        """
        dx = abs(cx - self._cx)
        dy = cy - self._cy  # 向下为正

        if dx > self.cfg.dx_max_px:
            return False, f"水平跳变过大 dx={dx:.1f} > {self.cfg.dx_max_px}"

        if dx > self.cfg.max_jump_px:
            return False, f"总跳变过大 dist={np.sqrt(dx**2 + dy**2):.1f} > {self.cfg.max_jump_px}"

        if dy < -self.cfg.dy_back_tol_px:
            return False, f"y 回跳过大 dy={dy:.1f} < -{self.cfg.dy_back_tol_px}"

        return True, ""

    def is_beyond_stop_y(self, cy_px: float, roi_rect: list[int] | None) -> bool:
        """检查 y 坐标是否超出底部终止线。"""
        if roi_rect is None:
            return False
        stop_y = roi_rect[1] + roi_rect[3] * self.cfg.stop_y_ratio
        return cy_px > stop_y

    # ── 结果导出 ──

    def get_trace_df(self) -> "pd.DataFrame":
        """导出追踪记录为 DataFrame。"""
        import pandas as pd
        records = []
        for p in self._trace:
            records.append({
                "frame": p.frame_idx,
                "x_px": p.x_px,
                "y_px": p.y_px,
                "state": p.state,
                "point_type": p.point_type,
                "confidence": p.confidence,
                "score": p.score,
                "circularity": p.circularity,
            })
        return pd.DataFrame(records)

    def reset(self):
        """重置追踪器到初始状态。"""
        self._state = TrackerState.STOPPED
        self._cx = self._cy = 0.0
        self._vx = self._vy = 0.0
        self._lost_count = 0
        self._total_predicted = 0
        self._trace = []
        self._search_rect = [0, 0, 0, 0]
