import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import time
from dataset import DatasetSVM


dataset_root = "./data"
mode = 1  # 1: 属性特征，2: 原始图片像素

train_data_path = os.path.join(dataset_root, 'train')
test_data_path = os.path.join(dataset_root, 'val')
train_data = DatasetSVM(train_data_path, img_only=(mode == 2), attr_only=(mode == 1))
test_data = DatasetSVM(test_data_path, img_only=(mode == 2), attr_only=(mode == 1))
X_imgs_train, X_attrs_train, Y_train = train_data.get_data()
X_imgs_test, X_attrs_test, Y_test = test_data.get_data()

print(f"Training samples: {len(Y_train)}, Testing samples: {len(Y_test)}")

# 默认用属性特征
X_train = X_attrs_train
X_test = X_attrs_test

if mode == 2:
    print("Using raw image pixels as features.")
    n_train_samples = X_imgs_train.shape[0]
    n_test_samples = X_imgs_test.shape[0]
    X_train = X_imgs_train.reshape(n_train_samples, -1).astype(np.float32)
    X_test = X_imgs_test.reshape(n_test_samples, -1).astype(np.float32)
else:
    print("Using attribute vectors as features.")

print("Standardizing features...")
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

param_sets = [
    {'kernel': 'linear', 'C': 0.1},
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'linear', 'C': 10.0},
    {'kernel': 'linear', 'C': 100.0},
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 1.0},
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 10.0},
    {'kernel': 'rbf', 'gamma': 0.01, 'C': 100.0},
    {'kernel': 'rbf', 'gamma': 0.001, 'C': 10.0},
    {'kernel': 'rbf', 'gamma': 0.1, 'C': 10.0},
    {'kernel': 'poly', 'degree': 2, 'C': 1.0},
    {'kernel': 'poly', 'degree': 3, 'C': 1.0},
    {'kernel': 'sigmoid', 'C': 1.0}
]

results = []

for params in param_sets:
    print("-" * 30)
    print(f"Training with parameters: {params}")
    svm = SVC(**params, random_state=1)
    start_time = time.time()
    svm.fit(X_train_std, Y_train)
    training_time = time.time() - start_time
    accuracy = svm.score(X_test_std, Y_test)
    Y_pred = svm.predict(X_test_std)
    result_entry = {
        'params': params,
        'accuracy': accuracy,
        'misclassified_samples': (Y_test != Y_pred).sum(),
        'training_time': training_time
    }
    results.append(result_entry)

print("\n" + "=" * 50)
print("Final Comparison of SVM Parameters")
print("=" * 50)
for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    print(f"Params: {str(res['params']):<45} "
          f"| Accuracy: {res['accuracy']:.4f} "
          f"| Misclassified: {res['misclassified_samples']}")