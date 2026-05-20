"""
对话框模块
"""

import yaml
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFormLayout, QDialogButtonBox, QFileDialog,
    QMessageBox, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt


class ScaleCalibrationDialog(QDialog):
    """比例尺标定对话框，显示详细标定信息。"""

    def __init__(self, p1: tuple, p2: tuple, pixel_distance: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定比例尺")
        self.setModal(True)
        self.resize(360, 280)

        self._result_scale = 0.0

        layout = QVBoxLayout(self)

        # 标定点信息
        info_group = QGroupBox("标定信息")
        grid = QGridLayout(info_group)
        grid.addWidget(QLabel("第一点坐标:"), 0, 0)
        grid.addWidget(QLabel(f"({p1[0]}, {p1[1]})"), 0, 1)
        grid.addWidget(QLabel("第二点坐标:"), 1, 0)
        grid.addWidget(QLabel(f"({p2[0]}, {p2[1]})"), 1, 1)
        grid.addWidget(QLabel("像素距离:"), 2, 0)
        self._px_label = QLabel(f"{pixel_distance:.1f} px")
        grid.addWidget(self._px_label, 2, 1)
        layout.addWidget(info_group)

        # 输入真实距离
        input_group = QGroupBox("输入真实距离")
        input_form = QFormLayout(input_group)

        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.001, 10000.0)
        self._dist_spin.setDecimals(2)
        self._dist_spin.setValue(50.0)
        self._dist_spin.setSuffix(" mm")
        self._dist_spin.valueChanged.connect(self._update)
        input_form.addRow("真实距离:", self._dist_spin)

        self._scale_display = QLabel("= 0.000000 mm/px")
        self._scale_display.setStyleSheet("font-weight: bold; font-size: 13px;")
        input_form.addRow("比例尺:", self._scale_display)
        layout.addWidget(input_group)

        self._px_dist = pixel_distance
        self._update()

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update(self):
        if self._px_dist <= 0:
            return
        real_mm = self._dist_spin.value()
        scale = real_mm / self._px_dist
        self._result_scale = scale
        self._scale_display.setText(f"= {scale:.6f} mm/px")

    def get_scale(self) -> float:
        return self._result_scale


class LoadConfigDialog(QDialog):
    """加载配置确认对话框。"""

    def __init__(self, config_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("加载配置")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"将从以下文件加载配置:"))
        layout.addWidget(QLabel(f"  {config_path}"))

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            preview = QLabel(content[:600])
            preview.setWordWrap(True)
            preview.setStyleSheet(
                "background: #f5f5f5; padding: 8px; font-family: Consolas; font-size: 9px;"
            )
            layout.addWidget(preview)
        except Exception:
            pass

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
