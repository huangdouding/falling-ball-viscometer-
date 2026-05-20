"""程序全局配置参数"""

# ---------- 视频参数 ----------
VIDEO_PATH = "videos/sample.mp4"      # 默认输入视频路径
SKIP_FRAMES = 0                        # 跳过头帧（释放抖动）
MAX_FRAMES = None                      # 最多处理帧数（None=全部）

# ---------- 图像预处理 ----------
GAUSSIAN_BLUR_KSIZE = 5                # 高斯模糊核大小
THRESH_METHOD = "adaptive"             # 阈值方法: "fixed" / "adaptive" / "otsu"
THRESH_FIXED_VALUE = 80                # fixed 模式下的固定阈值
BG_SUBTRACT = True                     # 是否启用背景扣除
BG_FRAME_OFFSET = 10                   # 取前 N 帧平均作为背景

# ---------- 球心检测 ----------
DETECT_METHOD = "contour"              # "contour" / "hough" / "moment"
MIN_BALL_RADIUS = 5                    # 最小球半径(像素)
MAX_BALL_RADIUS = 80                   # 最大球半径(像素)
BALL_CIRCULARITY_MIN = 0.7             # 最小圆度 (4π·面积/周长²)

# ---------- 标定 ----------
CALIB_OBJECT_MM = 50.0                 # 标尺实际长度 (mm)
CALIB_OBJECT_PIXELS = None             # 标尺在图像中的像素长度 (手动指定或自动)

# ---------- 输出 ----------
OUTPUT_DIR = "output"
DATA_DIR = "data"
CSV_FILENAME = "ball_track.csv"

# ---------- 终端速度判据 (Phase 3) ----------
VELOCITY_WINDOW = 30                   # 滑动窗口大小 (帧数)
MAX_CV = 0.05                          # 速度变异系数阈值 Cv = σ/v̄
MIN_R2 = 0.995                         # 线性拟合 R² 阈值
SMOOTH_WINDOW = 5                      # 速度平滑窗口

# ---------- 黏度参数 (Phase 5) ----------
BALL_DIAMETER_MM = 2.0                 # 小球直径 (mm)
BALL_DENSITY = 7.86                    # 小球密度 (g/cm³)
LIQUID_DENSITY = 0.96                  # 液体密度 (g/cm³)
GRAVITY = 9.794                        # 本地重力加速度 (m/s²)
CONTAINER_RADIUS_MM = 25.0             # 量筒内径半径 (mm)
LIQUID_HEIGHT_MM = 300.0               # 液柱高度 (mm)
