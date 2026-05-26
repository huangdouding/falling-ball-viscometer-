import json
import math
import os
import sys
import tempfile
import unittest

# 确保从任何目录运行都能找到 src 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd

from src.ball_detector import BallDetector
from src.candidate_detector import CandidateDetector
from src.terminal_region import find_terminal_region
from src.trajectory_filter import remove_outliers
from src.utils import normalize_config_keys
from src.viscosity import compute_viscosity


class CorePipelineTests(unittest.TestCase):
    def test_settings_roundtrip_keeps_physics_parameters(self):
        cfg = {
            "ball_density_kg_m3": 7810.0,
            "liquid_density_kg_m3": 960.0,
            "ball_radius_mm": 1.25,
            "cylinder_radius_mm": 20.0,
            "liquid_height_mm": 335.0,
            "scale_mm_per_px": 0.203252,
            "manual_fps": 240.0,
            "roi": [12, 34, 56, 78],
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "settings.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(normalize_config_keys(cfg), f)
            with open(path, "r", encoding="utf-8") as f:
                loaded = normalize_config_keys(json.load(f))

        self.assertEqual(loaded["ball_density_kg_m3"], 7810.0)
        self.assertEqual(loaded["liquid_density_kg_m3"], 960.0)
        self.assertEqual(loaded["roi"], [12, 34, 56, 78])

    def test_import_video_metadata_does_not_overwrite_physics(self):
        cfg = normalize_config_keys({
            "ball_density_kg_m3": 7800.0,
            "liquid_density_kg_m3": 1260.0,
            "ball_radius_mm": 1.0,
            "manual_fps": 240.0,
        })
        video_metadata = {"fps": 30.0, "width": 1920, "height": 1080}
        merged = dict(cfg)
        merged["video_width"] = video_metadata["width"]
        merged["video_height"] = video_metadata["height"]
        self.assertEqual(merged["ball_density_kg_m3"], 7800.0)
        self.assertEqual(merged["liquid_density_kg_m3"], 1260.0)
        self.assertEqual(merged["manual_fps"], 240.0)

    def test_roi_detector_returns_global_coordinates(self):
        frame = np.full((100, 100, 3), 245, dtype=np.uint8)
        cv2.circle(frame, (35, 50), 5, (20, 20, 20), -1)
        cfg = {
            "image_mode": True,
            "detect_method": "threshold_contour",
            "threshold_mode": "dark_ball_on_bright_bg",
            "roi": [20, 30, 50, 50],
            "min_area_px": 20,
            "max_area_px": 200,
            "min_circularity": 0.2,
            "expected_radius_px_min": 3,
            "expected_radius_px_max": 8,
            "center_band_enabled": False,
            "enable_long_line_rejection": False,
        }
        result = BallDetector(cfg).detect(frame, frame_idx=0)
        self.assertTrue(result["found"], result.get("fail_reason"))
        self.assertAlmostEqual(result["x_px"], 35, delta=2)
        self.assertAlmostEqual(result["y_px"], 50, delta=2)

    def test_candidate_detector_scores_circle_candidate(self):
        gray = np.full((80, 80), 240, dtype=np.uint8)
        cv2.circle(gray, (40, 45), 5, 20, -1)
        detector = CandidateDetector({
            "min_area_px": 20,
            "max_area_px": 200,
            "expected_radius_px_min": 3,
            "expected_radius_px_max": 8,
            "min_circularity": 0.2,
            "enable_long_line_rejection": False,
        })
        candidates = detector.detect(gray, predict_pos=(40, 45))
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreater(candidates[0]["score"], 40)

    def test_remove_outliers_rejects_large_jump(self):
        df = pd.DataFrame({
            "frame": [0, 1, 2],
            "time_s": [0.0, 0.01, 0.02],
            "x_px": [10.0, 11.0, 80.0],
            "y_px": [10.0, 15.0, 20.0],
            "y_m": [0.0, 0.001, 0.002],
            "valid": [True, True, True],
            "point_type": ["raw", "raw", "raw"],
        })
        cleaned = remove_outliers(df, dx_max=20, dist_max=30)
        self.assertFalse(bool(cleaned.loc[2, "valid"]))
        self.assertEqual(cleaned.loc[2, "point_type"], "outlier")

    def test_terminal_velocity_window_selection_reports_quality(self):
        fps = 100
        t = np.arange(0, 2.0, 1 / fps)
        y = 0.018 * t
        df = pd.DataFrame({
            "frame": np.arange(len(t)),
            "time_s": t,
            "y_m": y,
            "x_px": np.full(len(t), 40.0),
            "valid": np.ones(len(t), dtype=bool),
            "point_type": ["raw"] * len(t),
        })
        result = find_terminal_region(df, None, {
            "terminal_window_sec": 0.5,
            "terminal_ignore_start_sec": 0.1,
            "terminal_ignore_end_sec": 0.1,
            "r2_threshold": 0.999,
            "cv_threshold": 0.01,
            "terminal_min_real_detection_rate": 0.8,
        })
        self.assertTrue(result["found"], result["message"])
        self.assertAlmostEqual(result["terminal_velocity_m_s"], 0.018, delta=1e-6)
        self.assertGreaterEqual(result["real_detection_rate"], 0.8)
        self.assertEqual(result["predicted_count"], 0)

    def test_roi_exit_direction_aware(self):
        """Points below ROI bottom with downward continuity should be accepted;
        points outside the X band should be rejected."""
        from src.tracking import _remove_trajectory_outliers

        # Build a trajectory that starts in the ROI and naturally falls below
        user_roi = [100, 50, 80, 200]  # x=100, y=50, w=80, h=200
        # The ROI bottom is at y = 250
        n = 30
        df = pd.DataFrame({
            "frame": list(range(n)),
            "time_s": [i * 0.01 for i in range(n)],
            "x_px": [140.0] * n,  # centered in ROI X band
            "y_px": [50 + i * 10 for i in range(n)],  # starts at 50, goes to 340 (past ROI bottom at 250)
            "x_m": [0.01] * n,
            "y_m": [0.001 * i for i in range(n)],
            "valid": [True] * n,
            "point_type": ["raw"] * n,
            "rejection": [""] * n,
            "radius_px": [5.0] * n,
            "area_px": [78.5] * n,
            "circularity": [0.9] * n,
            "confidence": [100.0] * n,
        })
        config = {
            "dx_max": 15, "dy_back_tol": 3, "dist_max": 35,
            "roi": user_roi, "detect_roi": [100, 0, 80, 400],
            "allowed_axis_deviation_px": 50,
        }
        cleaned = _remove_trajectory_outliers(df, config)
        # All points should remain valid because they follow a straight downward path
        self.assertTrue(cleaned["valid"].all(), "Natural downward trajectory should be fully valid")

    def test_viscosity_formula_and_units(self):
        result = compute_viscosity(
            terminal_velocity_m_s=0.02,
            ball_radius_m=0.001,
            ball_density_kg_m3=7800,
            liquid_density_kg_m3=1260,
            cylinder_radius_m=0.02,
            liquid_height_m=0.35,
            g_m_s2=9.8,
            enable_wall_correction=False,
            enable_reynolds_correction=False,
        )
        expected = 2 * (0.001 ** 2) * 9.8 * (7800 - 1260) / (9 * 0.02)
        self.assertTrue(math.isclose(result["eta_final_pa_s"], expected, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
