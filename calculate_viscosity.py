#!/usr/bin/env python3
"""
落球法液体黏度手动计算工具

交互式输入实验数据，计算 Stokes 基础黏度、壁面修正 (Ladenburg-Faxen)、
雷诺数修正 (Oseen) 及综合修正后的黏度与相对误差。
支持 1~3 组数据对比。
"""

import math
import sys

# 设置 stdout 编码为 UTF-8，确保中文在 Windows 终端正常显示
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 标准黏度参考值 (Pa.s)
# ============================================================
STANDARD_VISCOSITY = {
    30: 0.451,
    40: 0.231,
}


# ============================================================
# 输入辅助
# ============================================================

def input_float(prompt, default=None):
    """输入浮点数，default=None 时必须输入有效值"""
    while True:
        s = input(prompt).strip()
        if not s:
            if default is not None:
                return default
            print("  此项不能为空，请输入数字。")
            continue
        try:
            return float(s)
        except ValueError:
            print("  输入无效，请输入数字。")


def input_int(prompt, default=None):
    """输入整数，default=None 时必须输入"""
    while True:
        s = input(prompt).strip()
        if not s:
            if default is not None:
                return default
            print("  此项不能为空，请输入整数。")
            continue
        try:
            return int(s)
        except ValueError:
            print("  输入无效，请输入整数。")


def input_times(n=6):
    """输入 n 个下落时间"""
    times = []
    for i in range(n):
        t = input_float(f"    第 {i + 1} 个时间 (s): ")
        times.append(t)
    return times


# ============================================================
# 黏度计算
# ============================================================

def compute_all(r, t, L, R, h, rho_s, rho_l, g=9.8):
    """
    计算各阶段黏度 (SI 单位: m, s, kg/m3, Pa.s)。

    返回:
        (eta_basic, eta_wall, eta_re, eta_combined)  单位: Pa.s
    """
    v = L / t

    # ---- Stokes 基础黏度 ----
    # eta = 2 * r^2 * g * (rho_s - rho_l) / (9 * v)
    eta_basic = (2.0 * r ** 2 * g * (rho_s - rho_l)) / (9.0 * v)

    # ---- 壁面修正 (Ladenburg-Faxen, 仅横向) ----
    # k_wall = 1 + 2.4 * r/R
    k_wall = 1.0 + 2.4 * r / R
    eta_wall = eta_basic / k_wall

    # ---- 雷诺数修正 (Oseen, 单独修正, 无壁面) ----
    # 隐式方程: eta = eta_basic / (1 + 3/16 * Re)
    # Re = 2 * r * v * rho_l / eta
    eta_re = eta_basic
    for _ in range(20):
        Re = 2.0 * r * v * rho_l / eta_re
        k_re = 1.0 + 3.0 / 16.0 * Re
        eta_re_new = eta_basic / k_re
        if abs(eta_re_new - eta_re) < 1e-12:
            break
        eta_re = eta_re_new

    # ---- 综合修正 (壁面 + 雷诺数) ----
    eta_combined = eta_wall  # 初始猜测（已含壁面修正）
    for _ in range(20):
        Re = 2.0 * r * v * rho_l / eta_combined
        k_re = 1.0 + 3.0 / 16.0 * Re
        eta_new = eta_basic / (k_wall * k_re)
        if abs(eta_new - eta_combined) < 1e-12:
            break
        eta_combined = eta_new

    return eta_basic, eta_wall, eta_re, eta_combined


def relative_error(calculated, standard):
    """计算相对误差 (%)"""
    if standard == 0:
        return float('inf')
    return abs(calculated - standard) / standard * 100


# ============================================================
# 输出格式化
# ============================================================

W = 118  # 表格宽度


def separator(ch):
    """生成表格横线"""
    return ch + "-" * (W - 2) + ch


def print_group_table(results, times, label, r_mm, temp, standard,
                      L_mm, D_mm, h_mm):
    """打印单组详细结果表格"""
    avg_t = sum(times) / len(times)
    n = len(times)

    # 标题行
    print()
    print(separator("+"))
    title = (f"  {label}  |  r={r_mm:.2f}mm  |  T={temp}C  |  "
             f"L={L_mm:.0f}mm  |  D={D_mm:.0f}mm  |  h={h_mm:.0f}mm  |  "
             f"eta0={standard:.4f}Pa.s")
    print(f"| {title:<{W - 3}}|")
    print(separator("|"))

    # 列标题
    col = (f"| {'序号':>3} | {'t(s)':>8} | "
           f"{'eta_Stokes':>10} | {'误差%':>7} | "
           f"{'eta_壁面':>10} | {'误差%':>7} | "
           f"{'eta_Re':>10} | {'误差%':>7} | "
           f"{'eta_综合':>10} | {'误差%':>7} |")
    print(col)
    print(separator("|"))

    # 每行数据
    eta_avgs = [0.0] * 4
    for i in range(n):
        t = times[i]
        eta_b, eta_w, eta_re, eta_c = results[i]
        e_b = relative_error(eta_b, standard)
        e_w = relative_error(eta_w, standard)
        e_re = relative_error(eta_re, standard)
        e_c = relative_error(eta_c, standard)

        print(f"| {i + 1:>3} | {t:>8.4f} | "
              f"{eta_b:>10.4f} | {e_b:>6.2f}% | "
              f"{eta_w:>10.4f} | {e_w:>6.2f}% | "
              f"{eta_re:>10.4f} | {e_re:>6.2f}% | "
              f"{eta_c:>10.4f} | {e_c:>6.2f}% |")

        for j in range(4):
            eta_avgs[j] += results[i][j]

    # 平均值行
    for j in range(4):
        eta_avgs[j] /= n

    e_b_avg = relative_error(eta_avgs[0], standard)
    e_w_avg = relative_error(eta_avgs[1], standard)
    e_re_avg = relative_error(eta_avgs[2], standard)
    e_c_avg = relative_error(eta_avgs[3], standard)

    print(separator("|"))
    print(f"| {'平均':>3} | {avg_t:>8.4f} | "
          f"{eta_avgs[0]:>10.4f} | {e_b_avg:>6.2f}% | "
          f"{eta_avgs[1]:>10.4f} | {e_w_avg:>6.2f}% | "
          f"{eta_avgs[2]:>10.4f} | {e_re_avg:>6.2f}% | "
          f"{eta_avgs[3]:>10.4f} | {e_c_avg:>6.2f}% |")
    print(separator("+"))

    return eta_avgs


def print_summary(all_avg_results, n_groups):
    """打印组间对比汇总"""
    if n_groups < 2:
        return

    W2 = 100
    print()
    line = "+" + "-" * (W2 - 2) + "+"
    print(line)
    print(f"|{'  组间对比汇总（平均值）':>{W2 - 2}}|")
    print("|" + "-" * (W2 - 2) + "|")

    col = (f"| {'组别':<16} | "
           f"{'eta_Stokes':>10} | {'误差%':>7} | "
           f"{'eta_壁面':>10} | {'误差%':>7} | "
           f"{'eta_综合':>10} | {'误差%':>7} |")
    print(col)
    print("|" + "-" * (W2 - 2) + "|")

    for label, standard, avg in all_avg_results:
        eta_b, eta_w, eta_re, eta_c = avg
        e_b = relative_error(eta_b, standard)
        e_w = relative_error(eta_w, standard)
        e_c = relative_error(eta_c, standard)

        print(f"| {label:<16} | "
              f"{eta_b:>10.4f} | {e_b:>6.2f}% | "
              f"{eta_w:>10.4f} | {e_w:>6.2f}% | "
              f"{eta_c:>10.4f} | {e_c:>6.2f}% |")

    print(line)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("   落球法液体黏度手动计算工具")
    print("=" * 60)

    # ---- 公共参数 ----
    print("\n--- 公共参数 ---")
    rho_s = input_float("小球密度 (kg/m^3) [默认 7850]: ", default=7850.0)
    rho_l = input_float("液体密度 (kg/m^3) [默认 960]: ", default=960.0)
    g = input_float("重力加速度 (m/s^2) [默认 9.8]: ", default=9.8)

    D_mm = input_float("量筒内径 (mm): ")
    R = D_mm / 2.0 / 1000.0                      # 半径 -> m

    h_mm = input_float("液面高度 / 液柱高度 (mm): ")  # 壁面修正 r/h 用
    h = h_mm / 1000.0

    L_mm = input_float("下落距离 (mm) [时间对应的下落距离]: ")  # v = L/t
    L = L_mm / 1000.0

    # ---- 标准黏度提示 ----
    print("\n标准黏度参考值:")
    for t, v in STANDARD_VISCOSITY.items():
        print(f"  {t}C  ->  {v} Pa.s")
    print("  (其他温度需手动输入标准黏度)")

    # ---- 组数 ----
    n_groups = int(input_int("\n要计算的组数 (1~6): "))
    n_groups = max(1, min(6, n_groups))

    all_avg_results = []

    for g_idx in range(n_groups):
        print(f"\n{'-' * 50}")
        print(f"  第 {g_idx + 1} 组数据输入")
        print(f"{'-' * 50}")

        label = input("  组别标签 (如 30C-3mm): ").strip()
        if not label:
            label = f"组{g_idx + 1}"

        temp = input_float("  温度 (C): ")
        if temp in STANDARD_VISCOSITY:
            standard = STANDARD_VISCOSITY[temp]
            print(f"    -> 自动选用标准黏度: {standard} Pa.s")
        else:
            standard = input_float(f"  输入 {temp:.0f}C 下的标准黏度 (Pa.s): ")

        r_mm = input_float("  小球半径 (mm): ")
        r = r_mm / 1000.0

        print("  请输入 6 个下落时间 (s):")
        times = input_times(6)

        # 计算
        results = [compute_all(r, t, L, R, h, rho_s, rho_l, g)
                   for t in times]

        # 打印表格
        avg_res = print_group_table(
            results, times, label, r_mm, temp, standard,
            L_mm, D_mm, h_mm,
        )
        all_avg_results.append((label, standard, avg_res))

    # ---- 汇总 ----
    print_summary(all_avg_results, n_groups)

    print("\n计算完毕！")


if __name__ == "__main__":
    main()
