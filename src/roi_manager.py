"""
ROI 管理器

管理显示 ROI 和检测 ROI 的分离：
  - display_roi: 用户在视频上框选的区域（像素坐标 [x, y, w, h]）
  - detection_roi: 实际用于检测的区域（自动收窄/扩展）

支持：
  - 手动初始点设置
  - ROI 边界保护
  - 多 ROI 模式（全视频 / 固定 ROI / 追踪跟随）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ROIState:
    """ROI 状态快照。"""
    display_roi: list[int] | None = None      # [x, y, w, h] 用户框选
    detection_roi: list[int] | None = None     # [x, y, w, h] 实际检测
    init_point: tuple[int, int] | None = None  # (x, y) 用户指定的初始点
    roi_type: str = "fixed"                    # "fixed" | "full_frame" | "tracking"
    auto_shrink: bool = False                  # 是否自动收窄 ROI


class ROIManager:
    """ROI 管理器，统一管理 ROI 状态转换。"""

    def __init__(self, frame_width: int = 0, frame_height: int = 0):
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._state = ROIState()
        self._fall_axis_x: float | None = None  # 下落中心线 X 坐标

    # ── 属性 ──

    @property
    def state(self) -> ROIState:
        return self._state

    @property
    def has_roi(self) -> bool:
        return self._state.display_roi is not None

    @property
    def has_init_point(self) -> bool:
        return self._state.init_point is not None

    # ── 设置 ──

    def set_frame_size(self, width: int, height: int):
        self._frame_width = width
        self._frame_height = height

    def set_display_roi(self, x: int, y: int, w: int, h: int):
        """设置用户框选的显示 ROI。"""
        x, y, w, h = self._clamp_roi(x, y, w, h)
        self._state.display_roi = [x, y, w, h]
        self._sync_detection_roi()

    def set_init_point(self, x: int, y: int):
        """设置手动指定的初始点（首帧小球中心）。"""
        x = max(0, min(x, self._frame_width - 1))
        y = max(0, min(y, self._frame_height - 1))
        self._state.init_point = (x, y)

    def clear_init_point(self):
        self._state.init_point = None

    def set_auto_shrink(self, enabled: bool):
        self._state.auto_shrink = enabled
        self._sync_detection_roi()

    def clear_roi(self):
        """清除 ROI，回退到全视频检测。"""
        self._state.display_roi = None
        self._state.detection_roi = None
        self._state.roi_type = "full_frame"

    # ── 检测 ROI 同步 ──

    def _sync_detection_roi(self):
        """根据 display_roi 和 auto_shrink 生成 detection_roi。"""
        if self._state.display_roi is None:
            self._state.detection_roi = None
            return

        x, y, w, h = self._state.display_roi

        if self._state.auto_shrink and self._fall_axis_x is not None:
            # 收窄：以下落中心线为中心，左右各 w/4
            half_w = max(20, w // 4)
            new_x = max(0, int(self._fall_axis_x) - half_w)
            new_w = min(half_w * 2, self._frame_width - new_x)
            self._state.detection_roi = [new_x, y, new_w, h]
        else:
            self._state.detection_roi = [x, y, w, h]

    def update_fall_axis(self, fall_axis_x: float):
        """更新下落中心线估计值，触发 detection_roi 同步。"""
        self._fall_axis_x = fall_axis_x
        if self._state.auto_shrink:
            self._sync_detection_roi()

    # ── 辅助 ──

    def _clamp_roi(self, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        x = max(0, min(x, self._frame_width - 1))
        y = max(0, min(y, self._frame_height - 1))
        w = max(10, min(w, self._frame_width - x))
        h = max(10, min(h, self._frame_height - y))
        return x, y, w, h

    def roi_to_dict(self) -> dict:
        """导出到 config dict，供后端分析管线使用。"""
        d = {}
        if self._state.display_roi is not None:
            d["roi"] = self._state.display_roi
        if self._state.init_point is not None:
            d["init_x_px"] = self._state.init_point[0]
            d["init_y_px"] = self._state.init_point[1]
        if self._state.auto_shrink:
            d["auto_shrink_roi"] = True
        return d

    @classmethod
    def from_config(cls, config: dict, frame_width: int = 0, frame_height: int = 0) -> "ROIManager":
        """从 config dict 恢复 ROI 状态。"""
        mgr = cls(frame_width, frame_height)
        roi = config.get("roi")
        if roi is not None and len(roi) == 4:
            mgr.set_display_roi(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        ix = config.get("init_x_px")
        iy = config.get("init_y_px")
        if ix is not None and iy is not None:
            mgr.set_init_point(int(ix), int(iy))
        if config.get("auto_shrink_roi"):
            mgr.set_auto_shrink(True)
        fa = config.get("fall_axis_x")
        if fa is not None:
            mgr.update_fall_axis(float(fa))
        return mgr

    def get_init_search_rect(self, radius_px: float = 30) -> list[int]:
        """获取以初始点为中心的搜索矩形 [x, y, w, h]。

        用于首帧在初始点附近搜索小球，而非全图搜索。
        """
        if self._state.init_point is None:
            return [0, 0, self._frame_width, self._frame_height]
        ix, iy = self._state.init_point
        r = max(10, int(radius_px))
        return [
            max(0, ix - r),
            max(0, iy - r),
            min(self._frame_width, ix + r) - max(0, ix - r),
            min(self._frame_height, iy + r) - max(0, iy - r),
        ]
