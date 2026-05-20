"""
结果展示区（重构版）

Tab 1: 轨迹图 y-t
Tab 2: 速度图 v-t
Tab 3: 终端速度拟合图（仅终端区数据）
Tab 4: 黏度结果表
Tab 5: 日志
"""

import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QTableWidget,
    QTableWidgetItem, QTextEdit, QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")

# ---- 中文字体修复 ----
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_font_paths = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
for _fp in _font_paths:
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"]
except Exception:
    pass

plt.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib 画布嵌入 Qt。"""

    def __init__(self, parent=None, width=7, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class ResultTabs(QTabWidget):
    """结果展示 Tab 页。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)

        self._traj_canvas = MplCanvas(self)
        self.addTab(self._traj_canvas, "轨迹图 y-t")

        self._vel_canvas = MplCanvas(self)
        self.addTab(self._vel_canvas, "速度图 v-t")

        self._fit_canvas = MplCanvas(self)
        self.addTab(self._fit_canvas, "拟合图")

        self._result_table = QTableWidget()
        self.addTab(self._result_table, "黏度结果")

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        self.addTab(self._log_view, "日志")

        self.clear_all()

    # ---- 日志 ----
    def append_log(self, msg: str):
        self._log_view.append(msg)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self._log_view.clear()

    # ---- 轨迹图 ----
    def update_trajectory_plot(self, traj_df, terminal_region: dict):
        ax = self._traj_canvas.axes
        ax.clear()
        valid = traj_df["valid"].values
        t = traj_df["time_s"].values
        y = traj_df["y_m"].values
        if valid.sum() < 2:
            ax.text(0.5, 0.5, "有效数据点不足", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            self._traj_canvas.draw()
            return
        ax.plot(t[valid], y[valid], "b.-", markersize=3, label="轨迹")
        if terminal_region.get("found", False):
            ts = terminal_region["start_time_s"]
            te = terminal_region["end_time_s"]
            mask = (t >= ts) & (t <= te) & valid
            if mask.sum():
                ax.plot(t[mask], y[mask], "r.-", markersize=5,
                        label=f"终端区 ({ts:.2f}s-{te:.2f}s)")
                ax.axvspan(ts, te, alpha=0.12, color="red")
        ax.set_xlabel("Time t / s")
        ax.set_ylabel("Position y / m")
        ax.set_title("Trajectory y-t")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        self._traj_canvas.draw()

    # ---- 速度图 ----
    def update_velocity_plot(self, velocity_df, terminal_region: dict):
        ax = self._vel_canvas.axes
        ax.clear()
        valid = velocity_df["valid"].values
        t = velocity_df["time_s"].values
        v_frame = velocity_df["v_m_s"].values          # 逐帧差分（debug）
        v_win = velocity_df["smooth_v_m_s"].values     # 滑动窗口拟合（趋势参考）

        if valid.sum() < 2:
            ax.text(0.5, 0.5, "有效数据点不足", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            self._vel_canvas.draw()
            return

        # 逐帧差分速度（浅灰细线，仅 debug）
        ax.plot(t[valid], v_frame[valid], "-", color="#D0D0D0", linewidth=0.5,
                alpha=0.4, zorder=1, label="v_frame_diff (debug)")
        # 滑动窗口拟合速度（蓝色）
        ax.plot(t[valid], v_win[valid], "b.-", markersize=3, zorder=2,
                label="v_local_window (参考)")

        if terminal_region.get("found", False):
            ts = terminal_region["start_time_s"]
            te = terminal_region["end_time_s"]
            vt = terminal_region["terminal_velocity_m_s"]
            r2 = terminal_region.get("r2", 0)
            cv = terminal_region.get("cv", 0)
            mask = (t >= ts) & (t <= te) & valid
            if mask.sum():
                ax.plot(t[mask], v_win[mask], "r.-", markersize=5, label="终端区", zorder=3)
            ax.axhline(y=vt, color="green", linestyle="--", linewidth=2,
                       label=f"v_terminal_fit = {vt:.4f} m/s  (R²={r2:.4f})", zorder=4)
            ax.axvspan(ts, te, alpha=0.12, color="red")
        ax.set_xlabel("Time t / s")
        ax.set_ylabel("Velocity v / (m/s)")
        ax.set_title("Velocity v-t (terminal fit = viscosity basis)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._vel_canvas.draw()

    # ---- 拟合图（只拟合终端速度区间） ----
    def update_fit_plot(self, traj_df, terminal_region: dict):
        ax = self._fit_canvas.axes
        ax.clear()
        if not terminal_region.get("found", False):
            ax.text(0.5, 0.5, "未找到终端速度区", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            self._fit_canvas.draw()
            return
        ts = terminal_region["start_time_s"]
        te = terminal_region["end_time_s"]
        vt = terminal_region["terminal_velocity_m_s"]
        r2 = terminal_region["r2"]
        cv = terminal_region["cv"]
        intercept = terminal_region.get("fit_intercept", 0)

        valid = traj_df["valid"].values
        t = traj_df["time_s"].values
        y = traj_df["y_m"].values
        mask = (t >= ts) & (t <= te) & valid
        if mask.sum() < 3:
            ax.text(0.5, 0.5, "终端区数据点不足", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            self._fit_canvas.draw()
            return

        t_fit = t[mask]
        y_fit = y[mask]
        y_pred = vt * t_fit + intercept

        ax.plot(t_fit, y_fit, "bo", markersize=4, label="Data (terminal region)")
        ax.plot(t_fit, y_pred, "r-", linewidth=2, label="Linear fit")

        info = (f"v_t = {vt:.6f} m/s\n"
                f"R²  = {r2:.6f}\n"
                f"Cv  = {cv:.4f}\n"
                f"Interval: {ts:.3f}s - {te:.3f}s")
        ax.text(0.05, 0.95, info, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("Time t / s")
        ax.set_ylabel("Position y / m")
        ax.set_title("Terminal Region Linear Fit")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._fit_canvas.draw()

    # ---- 结果表格 ----
    def update_result_table(self, terminal_region: dict, viscosity_result: dict = None,
                            target_summary: dict = None):
        rows = []
        tr = terminal_region or {}
        ts = target_summary or {}

        # ---- 统计口径行 ----
        video_total = ts.get("video_total_frames", 0)
        interval_src = ts.get("interval_source", "auto_track")
        a_start = ts.get("analysis_start_frame", 0)
        a_end = ts.get("analysis_end_frame", 0)
        a_total = ts.get("analysis_total_frames", 0)
        a_dur = ts.get("analysis_duration_s", 0)
        term_reason = ts.get("termination_reason", "")
        valid_in = ts.get("valid_frames_in_analysis", 0)
        valid_rate_in = ts.get("valid_rate_in_analysis", 0)
        valid_all = ts.get("valid_frames_all_video", 0)
        valid_rate_all = ts.get("valid_rate_all_video", 0)

        src_label = "手动指定" if interval_src == "manual_override" else "自动检测"
        rows.append(("视频总帧数", f"{video_total}"))
        rows.append(("—" * 20, f"──── 分析区间 ({src_label}) ────"))
        rows.append(("区间来源", f"{interval_src}"))
        rows.append(("分析区间",
                     f"帧 {a_start} ~ 帧 {a_end}"))
        rows.append(("分析区间时长", f"{a_dur:.3f} s"))
        rows.append(("终止原因", term_reason))

        if tr.get("found", False):
            t_start = tr.get("start_frame", "?")
            t_end = tr.get("end_frame", "?")
            t_dur = tr.get("end_time_s", 0) - tr.get("start_time_s", 0)
            rows.append(("—" * 20, "──── 终端速度区间 ────"))
            rows.append(("终端速度区间",
                         f"帧 {t_start} ~ 帧 {t_end}"))
            if isinstance(t_dur, (int, float)):
                rows.append(("终端速度区间时长", f"{t_dur:.3f} s"))
            if a_total > 0:
                rows.append(("分析区间内有效识别(raw)",
                             f"{valid_in} / {a_total}  ({valid_rate_in:.1f}%)"))

        else:
            if a_total > 0:
                rows.append(("分析区间内有效识别(raw)",
                             f"{valid_in} / {a_total}"))
            rows.append(("状态", tr.get("message", "未找到终端速度区")))
            rows.append(("程序结论", "未能得到可靠黏度结果。"))

        # 全视频统计
        if video_total > 0:
            rows.append(("—" * 20, "──── 全视频统计 ────"))
            rows.append(("全视频有效识别",
                         f"{valid_all} / {video_total}  ({valid_rate_all:.1f}%)"))

        if tr.get("found", False):
            # 继续输出已有的终端速度精确定量信息
            rows.append(("终端速度区间时间",
                         f"{tr['start_time_s']:.4f}s — {tr['end_time_s']:.4f}s"))
            rows.append(("终端速度 v_t", f"{tr['terminal_velocity_m_s']:.6f} m/s"))
            rows.append(("拟合 R²", f"{tr['r2']:.6f}"))
            rows.append(("速度变异系数 Cv", f"{tr['cv']:.4f}"))

            if viscosity_result:
                vb = viscosity_result['eta_basic_pa_s']
                cf = viscosity_result['correction_factor']
                vw = viscosity_result['eta_wall_pa_s']
                re = viscosity_result['reynolds_number']
                wall_on = viscosity_result.get('enable_wall_correction', True)

                rows.append(("─" * 20, "──── 黏度结果 ────"))
                rows.append(("基础黏度 η_basic", f"{vb:.6f} Pa·s"))
                rows.append(("壁面修正因子", f"{cf:.4f}"))
                rows.append(("修正后黏度 η_wall", f"{vw:.6f} Pa·s"))
                rows.append(("最终黏度 η_final",
                             f"{viscosity_result['eta_final_pa_s']:.6f} Pa·s"
                             f"  ({'η_wall = η_basic / correction' if wall_on else 'η_basic 无修正'})"))
                rows.append(("雷诺数 Re", f"{re:.4f}"))

                # 参考黏度对比（如果设置了）
                ref_visc = viscosity_result.get("reference_viscosity_pa_s")
                if ref_visc is not None and ref_visc > 0:
                    basic_err = abs(vb - ref_visc) / ref_visc * 100
                    wall_err = abs(vw - ref_visc) / ref_visc * 100
                    closer = "基础黏度 eta_basic" if basic_err <= wall_err else "壁面修正 eta_wall"
                    rows.append(("─" * 20, "──── 参考值对比 ────"))
                    rows.append(("参考黏度 η_ref", f"{ref_visc:.6f} Pa·s"))
                    rows.append(("基础黏度相对误差",
                                 f"{basic_err:.2f}%  (η_basic vs η_ref)"))
                    rows.append(("壁面修正后相对误差",
                                 f"{wall_err:.2f}%  (η_wall vs η_ref)"))
                    rows.append(("参考值下更接近", closer))
                else:
                    rows.append(("─" * 20, "──── 参考值对比 ────"))
                    rows.append(("参考黏度 η_ref", "未设置参考值"))
                    rows.append(("基础黏度相对误差", "未设置参考值"))
                    rows.append(("壁面修正后相对误差", "未设置参考值"))

                # 提示信息
                hints = self._build_result_hints(viscosity_result)
                if hints:
                    rows.append(("─" * 20, "──── 提示 ────"))
                    for h in hints:
                        rows.append(("", h))
        else:
            rows.append(("─" * 20, "──── 终端速度 ────"))
            rows.append(("状态", tr.get("message", "未找到终端速度区")))
            rows.append(("程序结论", "未能得到可靠黏度结果。"))

        self._result_table.setRowCount(len(rows))
        self._result_table.setColumnCount(2)
        self._result_table.setHorizontalHeaderLabels(["参数", "数值"])
        self._result_table.horizontalHeader().setStretchLastSection(True)

        # 设置第一列宽度
        self._result_table.setColumnWidth(0, 180)

        for i, (key, val) in enumerate(rows):
            item_key = QTableWidgetItem(key)
            item_val = QTableWidgetItem(val)
            if key.startswith("─"):
                # 分隔行用灰色
                item_key.setForeground(Qt.gray)
                item_val.setForeground(Qt.gray)
            elif key.startswith("提示") or (key == "" and val):
                item_val.setForeground(Qt.darkYellow)
            self._result_table.setItem(i, 0, item_key)
            self._result_table.setItem(i, 1, item_val)

        self._result_table.resizeColumnsToContents()
        self._result_table.setColumnWidth(0, max(self._result_table.columnWidth(0), 180))
        self.setCurrentIndex(3)

    @staticmethod
    def _build_result_hints(viscosity_result: dict) -> list:
        """根据黏度结果构造提示信息。"""
        hints = []
        vb = viscosity_result.get('eta_basic_pa_s', 0)
        vw = viscosity_result.get('eta_wall_pa_s', 0)
        cf = viscosity_result.get('correction_factor', 1.0)
        ref_visc = viscosity_result.get("reference_viscosity_pa_s")

        # 提示 1: 修正因子过大
        if cf > 1.10:
            hints.append(
                "壁面修正因子较大 (>{:.2f})，建议核对量筒内径、小球半径和是否偏心下落。".format(1.10)
            )

        # 提示 2: 有参考值时，检查基础值接近但修正后偏离
        if ref_visc is not None and ref_visc > 0 and cf > 1.05:
            basic_err = abs(vb - ref_visc) / ref_visc * 100
            wall_err = abs(vw - ref_visc) / ref_visc * 100
            if wall_err > basic_err * 1.5 and basic_err < 10:
                hints.append(
                    "当前基础黏度与参考值较接近 ({:.2f}%)，但壁面修正因子较大 ({:.4f})，"
                    "修正后结果 ({:.2f}%) 偏低。结果对量筒内半径 R 极其敏感，"
                    "请核准 R 是否为量筒内半径，并确认小球是否沿中心轴线下落。"
                    .format(basic_err, cf, wall_err)
                )

        return hints

    # ---- 清空 ----
    def clear_all(self):
        for canvas in [self._traj_canvas, self._vel_canvas, self._fit_canvas]:
            canvas.axes.clear()
            canvas.axes.text(0.5, 0.5, "等待分析结果...",
                             ha="center", va="center",
                             transform=canvas.axes.transAxes, fontsize=13)
            canvas.draw()
        self._result_table.setRowCount(0)
        self._result_table.setColumnCount(0)
        self.clear_log()
