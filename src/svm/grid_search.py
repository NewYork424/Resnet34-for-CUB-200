import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os, sys
import time
from dataset import DatasetSVM

# 数据集根目录（可从命令行参数传入），需包含 train/ 与 val/ 子目录
# 用法示例（PowerShell/命令提示符）:
# python d:\ProgramingFiles\ML\project\src\SVM\svm.py d:\path\to\CUB200
dataset_root = "./data"
mode = 1 # 1 for attributes, 2 for raw images

# 构建数据集（来自 src/dataset.py）
train_data_path = os.path.join(dataset_root, 'train')
test_data_path = os.path.join(dataset_root, 'val')
train_data = DatasetSVM(train_data_path, img_only=(mode==2), attr_only=(mode==1))
test_data = DatasetSVM(test_data_path, img_only=(mode==2), attr_only=(mode==1))
X_imgs_train, X_attrs_train, Y_train = train_data.get_data()
X_imgs_test, X_attrs_test, Y_test = test_data.get_data()

print(f"Training samples: {len(Y_train)}, Testing samples: {len(Y_test)}")

# 选择属性特征作为分类依据（默认）
X_train = X_attrs_train
X_test = X_attrs_test

# 可选：使用原始图片像素作为特征
# 单张图片形状 [H, W, 3]，需展开为一维向量
if mode == 2:
    print("Using raw image pixels as features.")
    n_train_samples = X_imgs_train.shape[0]
    n_test_samples = X_imgs_test.shape[0]
    X_train = X_imgs_train.reshape(n_train_samples, -1).astype(np.float32)
    X_test = X_imgs_test.reshape(n_test_samples, -1).astype(np.float32)
else:
    print("Using attribute vectors as features.")

# 标准化训练集和测试集
print("Standardizing features...")
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# --- 训练并评估多种SVM参数组合 ---

# 定义你想要测试的超参数组
# 这是一个更全面的测试列表，用于系统地比较不同参数的影响
# C: 正则化参数。通常在 [0.1, 1, 10, 100] 中选择。
# gamma: RBF核的系数。通常在 [0.001, 0.01, 0.1, 1] 中选择。
param_sets = [
    # 1. 线性核 (Linear Kernel): 主要观察C值的影响
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'linear', 'C': 10.0},
    {'kernel': 'linear', 'C': 100.0},

    # 2. RBF核 (高斯核): 观察C和gamma的组合影响
    # 固定 gamma，改变 C
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 1.0},
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 10.0},
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 100.0},
    # 固定 C，改变 gamma
    {'kernel': 'rbf', 'gamma': 0.001, 'C': 10.0},
    # {'kernel': 'rbf', 'gamma': 0.01, 'C': 10.0}, # 已在上面测试
    {'kernel': 'rbf', 'gamma': 0.1, 'C': 10.0},

    # 3. 多项式核 (Polynomial Kernel): 观察不同阶数(degree)的影响
    {'kernel': 'poly', 'degree': 2, 'C': 1.0},
    {'kernel': 'poly', 'degree': 3, 'C': 1.0},

    # 4. Sigmoid核 (Sigmoid Kernel): 新增
    {'kernel': 'sigmoid', 'C': 1.0}
]

results = []

for params in param_sets:
    print("-" * 30)
    print(f"Training with parameters: {params}")
    
    # 定义支持向量机
    svm = SVC(**params, random_state=1) # 添加 random_state 保证结果可复现
    
    # 训练模型并计时
    start_time = time.time()
    svm.fit(X_train_std, Y_train)
    training_time = time.time() - start_time
    
    # 使用测试集进行数据预测
    accuracy = svm.score(X_test_std, Y_test)
    Y_pred = svm.predict(X_test_std)    # 用训练好的分类器svm预测数据X_test_std的标签
    # 记录结果
    result_entry = {
        'params': params,
        'accuracy': accuracy,
        'misclassified_samples': (Y_test != Y_pred).sum(),
        'training_time': training_time
    }
    results.append(result_entry)

# --- 打印最终对比结果 ---
print("\n" + "="*50)
print("Final Comparison of SVM Parameters")
print("="*50)
# 按准确率降序排序，方便查看最佳参数
for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    print(f"Params: {str(res['params']):<45} \
    | Accuracy: {res['accuracy']:.4f} \
    | Misclassified: {res['misclassified_samples']}")