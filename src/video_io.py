"""
视频输入/输出工具函数

负责：
- 打开视频文件并读取属性（fps、总帧数、分辨率）
- 帧率获取与时间戳计算
"""

import cv2
import numpy as np


class VideoReader:
    """封装 OpenCV 视频读取，统一帧率与时间戳处理。"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.width = None
        self.height = None

    def open(self):
        """打开视频并读取基本属性。"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"无法打开视频文件: {self.video_path}\n"
                f"请检查路径是否正确，文件是否损坏。"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.fps <= 0:
            raise ValueError(
                f"无法读取视频帧率 (fps={self.fps})，请检查视频文件。"
            )

    def read_frame(self):
        """读取下一帧，返回 (ret, frame)。"""
        if self.cap is None:
            raise RuntimeError("请先调用 open() 打开视频。")
        return self.cap.read()

    def get_frame_time(self, frame_idx: int) -> float:
        """根据帧索引计算时间 (s)。"""
        if self.fps is None or self.fps <= 0:
            return 0.0
        return frame_idx / self.fps

    def reset(self):
        """回到视频开头。"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriterWrapper:
    """封装标注视频写入。"""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {output_path}")
        self.fps = fps

    def write_frame(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
