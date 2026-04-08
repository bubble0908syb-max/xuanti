import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# ================= 配置参数 =================
fs = 20000  # 采样率 20kHz
# 请确保下面两个路径是你本地真实存在的路径
original_file_path = r'./data/0.4mm leak/0.4mm-1-1-4.xlsx'
denoised_file_path = r'./denoised_data/0.4mm leak/0.4mm-1-1-4_denoised.csv'

# 中文显示字体设置 (防止画图时中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def get_spectrum(signal, fs):
    """
    计算单边频谱
    """
    N = len(signal)
    yf = fft(signal)
    xf = np.linspace(0.0, fs / 2.0, N // 2)
    # 归一化双边频谱并取一半
    y_mag = 2.0 / N * np.abs(yf[0:N // 2])
    return xf, y_mag


def calculate_metrics(raw, denoised):
    """
    计算评价指标
    """
    noise = raw - denoised

    # 1. 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean(noise ** 2))

    # 2. 原始信号的标准差 vs 去噪后信号的标准差
    std_raw = np.std(raw)
    std_denoised = np.std(denoised)

    # 3. 互相关系数 (Correlation)
    corr = np.corrcoef(raw, denoised)[0, 1]

    return rmse, std_raw, std_denoised, corr


def main():
    # 1. 读取数据
    print(f"正在加载数据...\n原文件: {original_file_path}\n去噪文件: {denoised_file_path}")
    try:
        # 💡 核心修复：自动判断原始文件是 xlsx 还是 csv
        if original_file_path.endswith('.xlsx') or original_file_path.endswith('.xls'):
            raw_data = pd.read_excel(original_file_path, header=None, engine='openpyxl').iloc[:, -1].values.flatten()
        else:
            raw_data = pd.read_csv(original_file_path, header=None).iloc[:, -1].values.flatten()

        # 去噪后的数据我们之前统一保存为了 CSV 格式
        denoised_data = pd.read_csv(denoised_file_path, header=None).iloc[:, -1].values.flatten()

    except FileNotFoundError:
        print("❌ 找不到文件，请检查文件路径和文件名是否填写正确！")
        return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return

    # 为了画图清晰，截取前 2000 个点 (0.1秒) 展示
    display_len = min(20000, len(raw_data))
    t = np.arange(display_len) / fs

    raw_disp = raw_data[:display_len]
    denoised_disp = denoised_data[:display_len]

    # 2. 计算定量指标
    rmse, std_raw, std_denoised, corr = calculate_metrics(raw_data, denoised_data)
    print("\n================ 去噪效果定量评估 ================")
    print(f"互相关系数 (Correlation): {corr:.4f} (推荐范围: 0.85~0.99，说明有效保留了原始特征)")
    print(f"均方根误差 (RMSE): {rmse:.4f} (表示滤除了多少幅值的噪声)")
    print(f"信号标准差: 原始 {std_raw:.4f} -> 去噪后 {std_denoised:.4f} (波动减小)")
    print("==================================================\n")

    # 3. 计算频域
    xf, raw_fft = get_spectrum(raw_data, fs)
    _, denoised_fft = get_spectrum(denoised_data, fs)

    # 4. 可视化绘图
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('VMD 去噪效果对比分析', fontsize=16)

    # ==== 图 1：时域波形对比 ====
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, raw_disp, color='lightgray', label='原始含噪信号', alpha=0.8)
    ax1.plot(t, denoised_disp, color='red', label='VMD 去噪后信号', linewidth=1.5)
    ax1.set_title('局部时域波形对比 (前0.1秒)')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ==== 图 2：滤除的噪声波形 ====
    ax2 = plt.subplot(2, 2, 2)
    noise_disp = raw_disp - denoised_disp
    ax2.plot(t, noise_disp, color='blue', alpha=0.7)
    ax2.set_title('被滤除的噪声信号')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('幅值')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # ==== 图 3：原始信号频谱 ====
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(xf, raw_fft, color='gray')
    ax3.set_title('原始信号频谱 (全频段)')
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('幅值')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # ==== 图 4：去噪后信号频谱 ====
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(xf, denoised_fft, color='red')
    ax4.set_title('VMD去噪后信号频谱 (高频杂波减少)')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('幅值')
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()