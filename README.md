# 落球法液体黏度自动测量系统

基于 **机器视觉自动追踪与误差修正** 的落球法液体黏度测量实验改进。

## 项目用途

本程序通过 OpenCV 自动识别透明液体中下落小球的球心位置，逐帧追踪小球轨迹，
自动判断终端速度区并计算终端速度，进而得到液体黏度（含壁面修正和雷诺数验证）。

## 项目结构

```
falling_ball_viscosity/
│
├── main.py                 # 主入口
├── config.yaml             # 配置文件（视频路径、物理参数、检测阈值等）
├── requirements.txt        # Python 依赖
├── README.md
│
├── src/
│   ├── video_io.py         # 视频读写工具
│   ├── ball_detector.py    # 单帧小球检测
│   ├── tracking.py         # 逐帧追踪 → 轨迹 DataFrame
│   ├── velocity.py         # 速度计算与平滑
│   ├── terminal_region.py  # 终端速度区自动判定
│   ├── viscosity.py        # 黏度计算（理想 + 壁面修正）
│   ├── plotting.py         # 图表输出（y-t、v-t、拟合图）
│   ├── report.py           # 结果摘要报告
│   └── utils.py            # 配置读取、参数检查等工具
│
├── data/
│   ├── input_videos/       # 放入真实视频
│   └── results/            # 输出：CSV、图表、标注视频、报告
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 准备真实视频

将拍摄的小球下落视频放入 `data/input_videos/` 目录。

### 2. 修改配置文件

编辑 `config.yaml`，修改以下**必填项**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `video_path` | 视频文件路径 | `data/input_videos/experiment_01.mp4` |
| `scale_mm_per_px` | 比例尺 (mm/px)，通过标尺照片计算 | `0.12` |
| `ball_radius_m` | 小球半径 (m) | `0.001` |
| `ball_density_kg_m3` | 小球密度 (kg/m³) | `7800` |
| `liquid_density_kg_m3` | 液体密度 (kg/m³) | `1260` |
| `cylinder_radius_m` | 量筒内半径 (m) | `0.025` |
| `liquid_height_m` | 液柱高度 (m) | `0.40` |

可选调整的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `roi` | 感兴趣区域 [x, y, w, h]，缩小搜索范围提高稳定性 | `null` (全帧) |
| `threshold_value` | 二值化阈值 (0-255)，`null` 则 Otsu 自动 | `null` |
| `min_area_px` / `max_area_px` | 小球在画面中的像素面积范围 | 20 ~ 2000 |
| `max_jump_px` | 帧间最大跳变像素，超过则标记为异常 | 80 |
| `cv_threshold` | 速度变异系数阈值 | 0.05 |
| `r2_threshold` | 线性拟合 R² 阈值 | 0.995 |

### 3. 运行程序

```bash
python main.py --config config.yaml
```

或使用默认路径：

```bash
python main.py
```

## 输出说明

程序运行后，所有输出保存在 `data/results/` 目录：

| 文件 | 说明 |
|------|------|
| `trajectory.csv` | 轨迹数据：帧号、时间、球心坐标 (px/m)、半径、面积、圆度、有效性标记 |
| `velocity.csv` | 速度数据：时间、原始速度、平滑后速度、有效性标记 |
| `marked_video.mp4` | 标注视频：显示每帧检测到的球心位置和坐标 |
| `trajectory_plot.png` | y-t 轨迹图，高亮终端速度区 |
| `velocity_plot.png` | v-t 速度曲线图，标注终端速度和区间 |
| `fit_plot.png` | 终端速度区 y-t 线性拟合图，标注 v_t、R²、Cv |
| `result_summary.txt` | 结果摘要：视频信息、追踪统计、黏度结果 |
| `config_copy.yaml` | 本次运行使用的配置副本 |

## 拍摄视频建议

为保证程序识别效果，拍摄时请注意：

1. **背景干净** — 白色哑光背景板，与深色小球形成明显对比
2. **光线均匀** — 避免强反光或局部过曝
3. **相机固定** — 使用三脚架，避免画面抖动
4. **标尺同平面** — 标尺与小球下落路径尽量在同一平面，减少透视误差
5. **帧率足够** — 建议 60 fps 以上，推荐 120 fps 慢动作模式
6. **小球不碰壁** — 尽量从量筒中心轴线释放，避免碰壁和明显偏心
7. **完整过程** — 视频包含小球从释放到落到底部的完整过程
8. **竖直下落** — 保证量筒竖直，减少倾斜带来的额外误差

## 参数调整建议

如果首次运行识别效果不佳：

- **检测不到小球**：降低 `threshold_value`，或改用 `adaptive_threshold: true`；
  检查 `min_area_px` / `max_area_px` 是否覆盖小球实际大小；
  设置 `roi` 缩小搜索区域
- **识别不稳定/跳变**：减小 `max_jump_px`；增大 `gaussian_blur_ksize`；
  检查光照是否均匀
- **找不到终端区**：适当放宽 `cv_threshold` 或 `r2_threshold`；
  检查视频中小球是否已进入匀速段

## 依赖

- Python >= 3.8
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0（可选，用于 Savitzky-Golay 平滑）
- pyyaml >= 6.0
