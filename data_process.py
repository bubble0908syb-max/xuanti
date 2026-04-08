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
alpha = 2000
tau = 0.
K = 5
DC = 0
init = 1
tol = 1e-7


def vmd_denoise(signal):
    """
    对单条信号进行 VMD 去噪
    """
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    reconstructed_signal = np.zeros_like(signal)

    # 💡 提高阈值：0.2 或 0.3 可以滤除更多低相关性的噪声模态
    correlation_threshold = 0.2

    selected_imfs = []  # 记录被选中的模态

    for i in range(K):
        imf = u[i, :]
        corr, _ = pearsonr(signal, imf)

        if abs(corr) > correlation_threshold:
            reconstructed_signal += imf
            selected_imfs.append(i + 1)

    return reconstructed_signal, selected_imfs


def main():
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for category in categories:
        input_dir = os.path.join(input_base_dir, category)
        output_dir = os.path.join(output_base_dir, category)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n================ 正在处理类别: {category} ================")

        if not os.path.exists(input_dir):
            print(f"警告：找不到文件夹 {input_dir}")
            continue

        file_list = os.listdir(input_dir)
        processed_count = 0

        for filename in file_list:
            # 💡 修复：同时支持 xlsx, xls, csv
            if filename.endswith(('.csv', '.xlsx', '.xls')):
                file_path = os.path.join(input_dir, filename)
                processed_count += 1

                try:
                    # 💡 修复读取逻辑：区分 csv 和 xlsx
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path, header=None)
                    else:
                        df = pd.read_excel(file_path, header=None, engine='openpyxl')

                    # 确保只取最后一列作为信号（防止把时间列也读进去）
                    # 如果只有1列，就是取第1列；如果有两列（比如time, value），取value列
                    data = df.iloc[:, -1].values.flatten()

                    # 执行 VMD 去噪
                    denoised_data, selected = vmd_denoise(data)

                    # 打印每个文件的去噪情况，方便你调试
                    print(f"[{processed_count}] 处理 {filename} -> 保留了 IMF 分量: {selected}")

                    # 💡 统一保存为 CSV，速度更快
                    new_filename = filename.rsplit('.', 1)[0] + '_denoised.csv'
                    save_path = os.path.join(output_dir, new_filename)
                    pd.DataFrame(denoised_data).to_csv(save_path, index=False, header=False)

                except Exception as e:
                    print(f"处理 {filename} 时出错: {e}")

        if processed_count == 0:
            print(f"⚠️ 警告：在 {category} 文件夹下没有找到有效的 csv 或 xlsx 文件！")
        else:
            print(f"✅ 类别 {category} 共处理了 {processed_count} 个文件！")


if __name__ == '__main__':
    main()