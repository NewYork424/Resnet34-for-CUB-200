import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import joblib  # [新增] 用于保存 sklearn 模型
from utils.dataset import UnifiedBirdDataset

def run_svm_task(cfg, logger):
    svm_cfg = cfg['svm']
    data_cfg = cfg['data']
    output_dir = cfg['output_dir'] # [新增] 获取输出目录
    
    feature_type = svm_cfg.get('feature_type', 'attr')
    logger.info(f"Running SVM with feature_type={feature_type}")

    # 加载数据
    train_ds = UnifiedBirdDataset(data_cfg['root'], 'train', img_size=data_cfg['img_size'], class_num = svm_cfg['class_num'])
    val_ds = UnifiedBirdDataset(data_cfg['root'], 'val', img_size=data_cfg['img_size'], class_num = svm_cfg['class_num'])
    
    X_train, y_train = train_ds.get_sklearn_data(feature_type)
    X_test, y_test = val_ds.get_sklearn_data(feature_type)
    
    logger.info(f"Data loaded. Train shape: {X_train.shape}")

    # 标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 训练
    logger.info("Training SVM...")
    clf = SVC(kernel=svm_cfg['kernel'], C=svm_cfg['C'], gamma=svm_cfg['gamma'], degree=svm_cfg['degree'])
    clf.fit(X_train, y_train)
    
    # 评估
    acc = clf.score(X_test, y_test)
    logger.info(f"SVM Accuracy: {acc:.4f}")

    # [新增] 保存结果
    # 1. 保存模型
    model_path = os.path.join(output_dir, "svm_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(sc, scaler_path)
    logger.info(f"Model saved to {model_path}")

    # 2. 保存结果文本
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Model: SVM\n")
        f.write(f"Kernel: {svm_cfg['kernel']}\n")
        f.write(f"Accuracy: {acc:.4f}\n")