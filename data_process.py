import os
import numpy as np
import pandas as pd
from vmdpy import VMD
from scipy.stats import pearsonr

# ================= 数据参数 =================
fs = 20000  # 采样率 20kHz
categories = ['0.4mm leak', '2mm leak', '4mm leak', 'no Leak']
input_base_dir = './data'  # 替换为你的原始数据根目录
output_base_dir = './denoised_data'  # 去噪后保存的根目录

# ================= VMD 参数 =================
alpha = 2000  # 数据保真度容忍度 (惩罚因子)，通常取 1000-2000
tau = 0.  # 噪声容忍度
K = 5  # 分解的模态数 (需要根据你的实际信号调整，通常 4-8 较好)
DC = 0  # 是否包含直流分量
init = 1  # 均匀初始化 omega
tol = 1e-7  # 收敛容忍度


def vmd_denoise(signal):
    """
    对单条信号进行 VMD 去噪
    """
    # 1. 运行 VMD 分解
    # u: 分解得到的 IMF 分量
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    # 2. 基于相关系数的模态筛选与重构
    reconstructed_signal = np.zeros_like(signal)
    correlation_threshold = 0.1  # 相关系数阈值，可根据实际去噪效果微调

    for i in range(K):
        imf = u[i, :]
        # 计算当前 IMF 与原始信号的相关系数
        corr, _ = pearsonr(signal, imf)

        # 如果相关系数大于阈值，认为是有效成分，参与重构
        if abs(corr) > correlation_threshold:
            reconstructed_signal += imf

    return reconstructed_signal


def main():
    # 创建输出根目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 遍历 4 个分类文件夹
    for category in categories:
        input_dir = os.path.join(input_base_dir, category)
        output_dir = os.path.join(output_base_dir, category)

        # 确保输出子文件夹存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"正在处理类别: {category} ...")

        # 假设每个文件夹下有 50 个文件，遍历处理
        if not os.path.exists(input_dir):
            print(f"警告：找不到文件夹 {input_dir}")
            continue

        for filename in os.listdir(input_dir):
            # 假设文件是 csv 格式，如果是一维数组格式（如 npy）请更改加载方式
            if filename.endswith('.csv'):
                file_path = os.path.join(input_dir, filename)

                # 读取数据 (这里假设 CSV 只有一列数据，没有表头)
                # 如果是 .npy 文件，使用: data = np.load(file_path)
                data = pd.read_csv(file_path, header=None).values.flatten()

                # 执行 VMD 去噪
                denoised_data = vmd_denoise(data)

                # 保存去噪后的数据
                save_path = os.path.join(output_dir, filename)
                pd.DataFrame(denoised_data).to_csv(save_path, index=False, header=False)

        print(f"类别 {category} 处理并保存完毕！")


if __name__ == '__main__':
    main()