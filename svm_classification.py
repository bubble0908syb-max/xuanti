import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= 1. 配置参数 =================
# 确保这里的类别名称与你文件夹名称完全一致！
categories = ['no Leak', '0.4mm leak', '2mm leak', '4mm leak']
data_dir = './denoised_data'  # 使用去噪后的数据
results_dir = './results'  # 💡 新增：保存结果的文件夹

# 创建结果保存目录
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 中文显示字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 特征提取函数 =================
def extract_time_features(signal):
    """
    提取单条信号的 10 个时域特征
    """
    mean = np.mean(signal)
    std = np.std(signal, ddof=1)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    skew = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)

    mean_abs = np.mean(np.abs(signal))
    sqr_amp = (np.mean(np.sqrt(np.abs(signal)))) ** 2

    shape_factor = rms / mean_abs if mean_abs != 0 else 0
    crest_factor = peak / rms if rms != 0 else 0
    impulse_factor = peak / mean_abs if mean_abs != 0 else 0
    margin_factor = peak / sqr_amp if sqr_amp != 0 else 0

    return [mean, std, rms, peak, skew, kurtosis, shape_factor, crest_factor, impulse_factor, margin_factor]


# ================= 3. 主程序 =================
def main():
    X = []
    y = []

    print("开始提取时域特征...")
    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        if not os.path.exists(folder_path):
            print(f"⚠️ 找不到文件夹: {folder_path}")
            continue

        file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for filename in file_list:
            file_path = os.path.join(folder_path, filename)
            try:
                signal = pd.read_csv(file_path, header=None).values.flatten()
                features = extract_time_features(signal)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("❌ 没有提取到任何特征！")
        return

    print(f"\n特征提取完成！数据集总大小: {X.shape[0]} 个样本。")

    # 💡 新增：1. 保存特征到 CSV，方便以后直接加载进行其他算法(如RF, KNN)的对比
    feature_names = ['Mean', 'Std', 'RMS', 'Peak', 'Skewness', 'Kurtosis',
                     'Shape_Factor', 'Crest_Factor', 'Impulse_Factor', 'Margin_Factor']
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features['Label_Num'] = y
    df_features['Label_Name'] = [categories[i] for i in y]
    features_save_path = os.path.join(results_dir, 'extracted_features.csv')
    df_features.to_csv(features_save_path, index=False)
    print(f"✅ 特征数据已保存至: {features_save_path}")

    # ================= 4. 数据划分与标准化 =================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================= 5. SVM 模型训练 =================
    print("\n正在训练 SVM 分类器 (RBF 核)...")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # 💡 新增：2. 保存训练好的 SVM 模型和标准化器 (Scaler)
    model_path = os.path.join(results_dir, 'svm_model.pkl')
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    joblib.dump(svm_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ 模型与标准化器已保存至: {results_dir} 文件夹")

    # ================= 6. 模型评估 =================
    y_pred = svm_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试集分类准确率 (Accuracy): {acc * 100:.2f}%")

    report = classification_report(y_test, y_pred, target_names=categories)
    print("\n================ 分类报告 ================")
    print(report)

    # 💡 新增：3. 保存分类报告到 txt 文本
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"模型准确率 (Accuracy): {acc * 100:.2f}%\n\n")
        f.write("详细分类报告:\n")
        f.write(report)
    print(f"✅ 分类报告已保存至: {report_path}")

    # ================= 7. 绘制并保存混淆矩阵 =================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title(f'SVM 管道泄漏分类混淆矩阵\n准确率: {acc * 100:.2f}%', fontsize=14)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()

    # 💡 新增：4. 保存混淆矩阵图片 (一定要在 plt.show() 之前保存)
    img_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(img_path, dpi=300)  # dpi=300 保证图片是论文级别的高清图
    print(f"✅ 混淆矩阵图片已保存至: {img_path}")

    plt.show()


if __name__ == '__main__':
    main()