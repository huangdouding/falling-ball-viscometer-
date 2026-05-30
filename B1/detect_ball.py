"""
落球法小球检测脚本（单张截图版）
用法：python detect_ball.py --image <图片路径>
"""

import cv2
import numpy as np
import argparse
import os


def preprocess(img, blur_ksize=5, method="adaptive"):
    """灰度→高斯模糊→二值化"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
    elif method == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # 形态学开运算去除噪点
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, blurred, binary


def detect_by_contour(binary, original, min_area=30, max_area=5000, circularity_min=0.5):
    """轮廓检测法：面积+圆度筛选"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # 圆度 = 4π * area / perimeter²，正圆为 1
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        candidates.append({
            "cx": int(x),
            "cy": int(y),
            "r": int(radius),
            "area": area,
            "circularity": circularity,
        })

    # 按圆度排序，取最圆的
    candidates.sort(key=lambda p: p["circularity"], reverse=True)
    return candidates


def get_search_roi(gray):
    h, w = gray.shape
    if h >= 1000 and h / w > 1.7:
        return int(w * 0.42), int(h * 0.16), int(w * 0.80), int(h * 0.58)
    if h < 900:
        return 0, 0, w, h
    return int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)


def detect_dark_spots(gray, original):
    x0, y0, x1, y1 = get_search_roi(gray)
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 20
    params.maxThreshold = 190
    params.thresholdStep = 5
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.08
    params.filterByConvexity = True
    params.minConvexity = 0.15
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(crop)
    candidates = []

    for kp in keypoints:
        x, y = kp.pt
        r = kp.size / 2
        if x < 12 or x > w - 12 or y < 20 or y > h - 20:
            continue
        if not (2 <= r <= 16):
            continue

        xi, yi = int(round(x)), int(round(y))
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - xi) ** 2 + (yy - yi) ** 2
        inner = dist2 <= max(2, r * 0.65) ** 2
        ring = (dist2 >= (r + 3) ** 2) & (dist2 <= (r + 18) ** 2)
        if not inner.any() or not ring.any():
            continue

        inner_mean = float(crop[inner].mean())
        ring_mean = float(crop[ring].mean())
        contrast = ring_mean - inner_mean
        if contrast < 10:
            continue

        py0 = max(0, yi - 18)
        py1 = min(h, yi + 19)
        px0 = max(0, xi - 18)
        px1 = min(w, xi + 19)
        patch = crop[py0:py1, px0:px1]
        bright_ratio = float((patch > 170).mean())
        very_bright_ratio = float((patch > 200).mean())

        # 局部亮度判断：patch 中 >200 的占比高则为亮背景
        local_bright = very_bright_ratio > 0.50

        if local_bright:
            if r < 2 or r > 12:
                continue
        else:
            if very_bright_ratio > 0.02 or bright_ratio > 0.18:
                continue
            if r < 4 and bright_ratio > 0.08:
                continue

        if gray.shape[0] < 900:
            gx = xi + x0
            gy = yi + y0
            if not (0.40 * gray.shape[1] <= gx <= 0.70 * gray.shape[1] and 0.25 * gray.shape[0] <= gy <= 0.55 * gray.shape[0]):
                continue

        small_penalty = 18 if r < 4 else 0
        size_score = 14 - abs(r - 7) * 1.4
        darkness_score = max(0, 150 - inner_mean) * 0.25
        score = contrast * 1.5 + darkness_score + size_score - bright_ratio * 60 - very_bright_ratio * 140 - small_penalty
        if local_bright:
            score = contrast * 1.8 + darkness_score + size_score - abs(yi - h * 0.65) * 0.03
        elif gray.shape[0] >= 1000:
            preferred_x = x0 + (x1 - x0) * 0.88
            score += max(0, 45 - abs((xi + x0) - preferred_x)) * 1.2

        candidates.append({
            "cx": xi + x0,
            "cy": yi + y0,
            "r": int(round(r)),
            "area": np.pi * r * r,
            "circularity": 1.0,
            "contrast": contrast,
            "inner_mean": inner_mean,
            "score": score,
            "bright_ratio": bright_ratio,
            "roi": (x0, y0, x1, y1),
        })

    candidates.sort(key=lambda p: p["score"], reverse=True)
    return candidates


def detect_by_hough(gray, original, min_radius=5, max_radius=100):
    """霍夫圆检测法（备用）"""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=30,
        minRadius=min_radius, maxRadius=max_radius
    )
    candidates = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for (x, y, r) in circles:
            candidates.append({"cx": x, "cy": y, "r": r, "area": np.pi * r * r, "circularity": 1.0})
    return candidates


def run(image_path):
    if not os.path.exists(image_path):
        print(f"[错误] 图片不存在: {image_path}")
        return

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"[信息] 图片尺寸: {w}x{h}")

    # 预处理
    gray, blurred, binary = preprocess(img)

    # 当前截图可视化很差：优先找浅背景上的小暗点，并压低数字/刻度干扰
    candidates = detect_dark_spots(gray, img)

    # 如果小暗点检测没找到，再退回轮廓检测和霍夫圆
    if not candidates:
        candidates = detect_by_contour(binary, img)
    if not candidates:
        print("[信息] 轮廓检测未找到候选，尝试霍夫圆检测...")
        candidates = detect_by_hough(gray, img)

    # 可视化结果
    result = img.copy()

    if candidates:
        best = candidates[0]
        print(f"[成功] 检测到小球！")
        print(f"       圆心: ({best['cx']}, {best['cy']})")
        print(f"       半径: {best['r']} px")
        print(f"       面积: {best['area']:.0f} px^2")
        print(f"       圆度: {best['circularity']:.3f}")
        if "contrast" in best:
            print(f"       局部对比度: {best['contrast']:.1f}")
            print(f"       评分: {best['score']:.1f}")

        # 画圆和圆心
        cv2.circle(result, (best['cx'], best['cy']), best['r'], (0, 255, 0), 2)
        cv2.circle(result, (best['cx'], best['cy']), 3, (0, 0, 255), -1)
        cv2.putText(result, f"({best['cx']}, {best['cy']})",
                    (best['cx'] - 60, best['cy'] - best['r'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("[失败] 未检测到小球，请调整阈值参数或检查图片")

    # 拼接预览图（全部转3通道）
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # 统一高度
    target_h = 400
    def resize_to_h(img, h):
        scale = h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        return cv2.resize(img, (new_w, h))

    result   = resize_to_h(result, target_h)
    binary_bgr = resize_to_h(binary_bgr, target_h)
    gray_bgr  = resize_to_h(gray_bgr, target_h)
    blurred_bgr = resize_to_h(blurred_bgr, target_h)

    top_row = np.hstack((result, binary_bgr))
    bottom_row = np.hstack((gray_bgr, blurred_bgr))

    # 缩放以适应屏幕
    max_w = 1600
    if top_row.shape[1] > max_w:
        scale = max_w / top_row.shape[1]
        new_h = int(top_row.shape[0] * scale)
        top_row = cv2.resize(top_row, (max_w, new_h))
        bottom_row = cv2.resize(bottom_row, (max_w, new_h))

    preview = np.vstack((top_row, bottom_row))

    # 保存结果（支持中文路径）
    out_dir = os.path.dirname(image_path) or "."
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}_result.png")
    cv2.imencode('.png', preview)[1].tofile(out_path)
    print(f"[保存] 结果图: {out_path}")

    print(f"[完成] 结果图已保存至: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="落球法小球检测（单张截图）")
    parser.add_argument("--image", "-i", required=True, help="图片路径")
    parser.add_argument("--method", choices=["adaptive", "otsu", "binary"], default="adaptive",
                        help="二值化方法 (default: adaptive)")
    args = parser.parse_args()
    run(args.image)
