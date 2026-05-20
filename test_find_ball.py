"""Final verification: tracking + interval + terminal quality."""
import sys, os
sys.path.insert(0, 'src')
import yaml, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(message)s')
with open('config.yaml') as f:
    config = yaml.safe_load(f)

from src.tracking import process_video, detect_track_interval
video_path = 'videos/' + os.listdir('videos')[0]
result = process_video(video_path, config)
traj_df = result['traj_df']

valid = traj_df[traj_df['valid'] == True]
raw = valid[valid['point_type'] == 'raw']

print("=" * 60)
print("TRACKING RESULTS")
print("=" * 60)
print(f"Total frames: {result['total_frames']} ({result['total_frames']/240:.3f}s)")
print(f"Valid points: {len(valid)} ({len(valid)/240:.3f}s)")
print(f"  raw: {len(raw)}, interpolated: {len(valid)-len(raw)}")
print(f"Valid ratio: {result['valid_ratio']*100:.1f}%")
print(f"Longest segment: {result['longest_valid_segment_frames']} fr = {result['longest_valid_segment_sec']:.3f}s")
print(f"Segment R^2: {result['longest_segment_r2']:.4f}")
print(f"Segment Cv: {result['longest_segment_cv']:.4f}")
print(f"Has quality segment: {result['has_quality_segment']}")

# Ball displacement
y_min, y_max = valid['y_px'].min(), valid['y_px'].max()
x_min, x_max = valid['x_px'].min(), valid['x_px'].max()
print(f"\nPosition range: x=[{x_min:.1f}, {x_max:.1f}] Δx={x_max-x_min:.1f}px")
print(f"                y=[{y_min:.1f}, {y_max:.1f}] Δy={y_max-y_min:.1f}px")

# Interval detection
print("\n" + "=" * 60)
print("INTERVAL DETECTION")
print("=" * 60)
try:
    interval = detect_track_interval(traj_df, config)
    if interval:
        start_fr, end_fr = interval
        start_t = start_fr / 240
        end_t = end_fr / 240
        seg = traj_df.iloc[start_fr:end_fr+1]
        seg_valid = seg[seg['valid'] == True]
        dx = seg_valid['y_px'].max() - seg_valid['y_px'].min()
        print(f"Best interval: frame {start_fr}-{end_fr} ({start_t:.3f}s-{end_t:.3f}s)")
        print(f"  duration: {(end_fr-start_fr+1)/240:.3f}s")
        print(f"  valid points: {len(seg_valid)}")
        print(f"  y displacement: {dx:.1f}px")
        r2 = result['longest_segment_r2']
        print(f"  R^2: {r2:.4f}")
        # Quality requirements
        min_dur = 0.3 + 0.5 + 0.8  # ignore_start + ignore_end + terminal_window
        print(f"\nQuality requirements:")
        print(f"  min duration {min_dur:.1f}s: {'PASS' if (end_fr-start_fr+1)/240 >= min_dur else 'FAIL'}")
        print(f"  min R^2 0.990: {'PASS' if r2 >= 0.99 else 'FAIL'}")
        print(f"  min 30px displacement: {'PASS' if dx >= 30 else 'FAIL'}")
    else:
        print("No valid interval found")
except Exception as e:
    print(f"Interval detection error: {e}")

print("\n" + "=" * 60)
print("CONFIG CHANGES MADE")
print("=" * 60)
changes = [
    ("auto_size_params: false", "Keeps user's manual detection params, prevents code override"),
    ("expected_radius_px_max: 8.0", "Matches visual ball radius (6-7px) for better size scoring"),
    ("min_area_px: 40", "Filters out small artifacts (<40px^2), keeps ball (~60px^2)"),
    ("max_area_px: 2000", "Accommodates larger ball appearance from blooming/blur"),
    ("fall_axis_x: 347.0", "Guides detector to ball's column, penalizes off-axis artifacts"),
    ("roi: [280, 0, 140, 1280]", "Excludes left/right artifacts, focuses on ball region"),
    ("enable_prediction_fill: true", "Fallback prediction when ball briefly undetected"),
    ("dy_back_tol: 8", "Relaxed vertical tolerance for motion continuity"),
    ("dist_max: 45, dx_max: 20", "Relaxed distance tolerance"),
    ("bg_sub threshold 15→8", "Lowered diff threshold to detect slow-moving ball at 240fps"),
    ("tracking params respect manual config", "No autmatic override of user-set params"),
]
for c, why in changes:
    print(f"  • {c}")
    print(f"    → {why}")
