import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 纯手写数据预处理工具 =================

def manual_train_test_split(X, y, test_size=0.3, random_state=42):
    """纯手写：划分训练集和测试集"""
    np.random.seed(random_state)
    # 打乱索引
    indices = np.random.permutation(X.shape[0])
    split_idx = int(X.shape[0] * (1 - test_size))

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class ManualStandardScaler:
    """纯手写：Z-Score 数据标准化"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 防止除以0
        self.std_[self.std_ == 0] = 1e-8
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_


# ================= 2. 纯手写底层 SVM (二分类) =================

class LinearSVM_Binary:
    """
    基于次梯度下降法 (Sub-gradient Descent) 的线性 SVM (二分类)
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  # 学习率
        self.lambda_param = lambda_param  # 正则化惩罚参数 (类似 sklearn 里的 1/C)
        self.n_iters = n_iters  # 迭代次数
        self.w = None  # 权重向量
        self.b = None  # 偏置
        self.loss_history = []  # 记录损失下降过程

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降迭代
        for _ in range(self.n_iters):
            loss = 0
            for idx, x_i in enumerate(X):
                # 核心逻辑：判断是否满足 margin >= 1
                # y_i * (w·x_i - b) >= 1
                decision = y[idx] * (np.dot(x_i, self.w) - self.b)

                if decision >= 1:
                    # 分类正确且在间隔边界外：只惩罚权重大小 (L2 正则化)
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # 分类错误或在间隔边界内：Hinge Loss 产生次梯度
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = y[idx]
                    loss += 1 - decision

                # 更新参数
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # 记录平均 Loss，用于画图
            self.loss_history.append((self.lambda_param * np.dot(self.w, self.w) + loss / n_samples))

    def decision_function(self, X):
        """计算样本到超平面的距离 (用于多分类打分)"""
        return np.dot(X, self.w) - self.b


# ================= 3. 纯手写 OvR 多分类 SVM 封装 =================

class MultiClassSVM_OvR:
    """
    基于 One-vs-Rest (一对多) 策略的多分类 SVM
    """

    def __init__(self, learning_rate=0.005, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = []  # 存放多个二分类器
        self.classes = []  # 记录类别标签

    def fit(self, X, y):
        self.classes = np.unique(y)

        # 针对每一个类别，训练一个"是该类 vs 不是该类"的二分类器
        for c in self.classes:
            print(f"正在训练二分类器: [类别 {c}] vs [其他所有类别] ...")
            # 构建二分类标签：当前类别为 1，其他为 -1
            y_binary = np.where(y == c, 1, -1)

            model = LinearSVM_Binary(learning_rate=self.lr,
                                     lambda_param=self.lambda_param,
                                     n_iters=self.n_iters)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        # 让所有二分类器给样本打分
        # shape: (n_classes, n_samples)
        scores = np.array([model.decision_function(X) for model in self.models])

        # 沿着列(针对每个样本)，找出得分最高的模型索引
        best_class_indices = np.argmax(scores, axis=0)
        return self.classes[best_class_indices]


# ================= 4. 主程序：加载数据并训练 =================

def main():
    print("1. 正在加载特征数据...")
    try:
        df = pd.read_csv('./results/extracted_features.csv')
    except Exception as e:
        print("❌ 读取文件失败，请确保 extracted_features.csv 存在！")
        return

    # 前10列是特征，倒数第二列是数字标签 (0, 1, 2, 3)
    X = df.iloc[:, :-2].values
    y = df['Label_Num'].values
    target_names = ['no Leak', '0.4mm leak', '2mm leak', '4mm leak']

    print("2. 划分数据集 (70% 训练, 30% 测试)...")
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.3, random_state=42)

    print("3. 手写特征标准化...")
    scaler = ManualStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n4. 开始训练手写 SVM (梯度下降法)...")
    # 你可以调整学习率 lr 和迭代次数 n_iters 观察效果
    my_svm = MultiClassSVM_OvR(learning_rate=0.005, lambda_param=0.01, n_iters=1500)
    my_svm.fit(X_train_scaled, y_train)

    print("\n5. 模型评估...")
    y_pred = my_svm.predict(X_test_scaled)

    # 纯手写计算准确率
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"✅ 自研 SVM 测试集准确率: {accuracy * 100:.2f}%\n")

    # ================= 画出其中一个模型的 Loss 曲线 =================
    plt.figure(figsize=(8, 5))
    # 我们画第一个模型(no Leak vs Rest)的 Loss 下降曲线
    plt.plot(my_svm.models[0].loss_history, color='b', linewidth=2)
    plt.title('底层 SVM (No Leak vs Rest) 梯度下降 Loss 曲线', fontsize=14)
    plt.xlabel('迭代次数 (Iterations)', fontsize=12)
    plt.ylabel('Hinge Loss + L2 正则化惩罚', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('./results/manual_svm_loss.png', dpi=300)
    print("✅ 损失下降曲线图已保存至: ./results/manual_svm_loss.png")
    plt.show()

    # (可选) 绘制混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'自研手写 SVM 混淆矩阵\n准确率: {accuracy * 100:.2f}%')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('./results/manual_svm_cm.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()