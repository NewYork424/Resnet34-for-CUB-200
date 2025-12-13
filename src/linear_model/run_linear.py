import argparse
import os
import sys
import numpy as np
import pickle # [新增]

# 使用绝对导入
from linear_model.linear_model import LinearClassifier, IdentityFeatures, Polynomial2Features, RBFRandomFourierFeatures
from utils.dataset import UnifiedBirdDataset

def load_attrs_split(root: str, class_num: int = 10):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_ds = UnifiedBirdDataset(train_dir, img_only=False, attr_only=True, class_num=class_num)
    val_ds = UnifiedBirdDataset(val_dir, img_only=False, attr_only=True, class_num=class_num)
    tr_imgs, tr_attrs, tr_labels = train_ds.get_data()
    va_imgs, va_attrs, va_labels = val_ds.get_data()

    X_tr = np.asarray(tr_attrs, dtype=np.float32)
    X_va = np.asarray(va_attrs, dtype=np.float32)
    y_tr = np.asarray(tr_labels, dtype=np.int64)
    y_va = np.asarray(va_labels, dtype=np.int64)
    return X_tr, y_tr, X_va, y_va

def build_feature_map(name: str, rff_dim: int, rff_gamma: float, seed: int):
    name = name.lower()
    if name == "identity":
        return IdentityFeatures()
    elif name == "poly2":
        return Polynomial2Features()
    elif name == "rbf":
        return RBFRandomFourierFeatures(n_components=rff_dim, gamma=rff_gamma, random_state=seed)
    else:
        raise ValueError(f"Unknown feature_map: {name}")

def run_linear_task(cfg, logger):
    """
    统一接口调用的线性模型任务函数
    """
    lin_cfg = cfg['linear']
    data_cfg = cfg['data']
    output_dir = cfg['output_dir'] # [新增]
    
    feature_type = lin_cfg.get('feature_type', 'attr')
    logger.info(f"Running Linear Model with feature_type={feature_type}")

    # 加载数据
    train_ds = UnifiedBirdDataset(data_cfg['root'], 'train', img_size=data_cfg['img_size'], class_num = lin_cfg['class_num'])
    val_ds = UnifiedBirdDataset(data_cfg['root'], 'val', img_size=data_cfg['img_size'], class_num = lin_cfg['class_num'])
    
    X_train, y_train = train_ds.get_sklearn_data(feature_type)
    X_test, y_test = val_ds.get_sklearn_data(feature_type)
    
    logger.info(f"Data loaded. Train shape: {X_train.shape}")

    # 构建特征映射
    seed = cfg.get('seed', 42)
    fmap = build_feature_map(lin_cfg['feature_map'], lin_cfg['rff_dim'], lin_cfg['rff_gamma'], seed)
    
    # 初始化分类器
    clf = LinearClassifier(
        lr=lin_cfg['lr'],
        epochs=lin_cfg['epochs'],
        batch_size=lin_cfg.get('batch_size', 256),
        reg=lin_cfg['reg'],
        standardize=lin_cfg.get('standardize', True),
        feature_map=fmap,
        random_state=seed,
        verbose=False
    )
    
    # 训练
    logger.info("Training Linear Classifier...")
    clf.fit(X_train, y_train)
    
    # 评估
    acc = clf.score(X_test, y_test)
    logger.info(f"Linear Model Accuracy: {acc:.4f}")

    # [新增] 保存结果
    # 1. 保存模型对象 (Pickle)
    model_path = os.path.join(output_dir, "linear_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {model_path}")

    # 2. 保存权重 (可选，方便单独读取)
    weights_path = os.path.join(output_dir, "weights.npz")
    np.savez(weights_path, W=clf.W_, b=clf.b_)

    # 3. 保存结果文本
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Model: Linear Classifier\n")
        f.write(f"Feature Map: {lin_cfg['feature_map']}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

# 保留 main 用于单独测试 (可选)
def main():
    parser = argparse.ArgumentParser(description="Linear classification on attribute features")
    parser.add_argument("--data-root", type=str, default="./data/", help="数据集根目录，包含 train/ 与 val/ 子目录")
    parser.add_argument("--feature-map", type=str, default="identity", choices=["identity", "poly2", "rbf"], help="特征映射类型")
    parser.add_argument("--rff-dim", type=int, default=512, help="RBF 随机傅里叶特征维度")
    parser.add_argument("--rff-gamma", type=float, default=0.5, help="RBF gamma (1/(2*sigma^2))")
    parser.add_argument("--lr", type=float, default=5e-2, help="学习率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--reg", type=float, default=1e-4, help="L2 正则系数")
    parser.add_argument("--no-standardize", action="store_true", help="不进行标准化")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    X_tr, y_tr, X_va, y_va = load_attrs_split(args.data_root, 10)

    fmap = build_feature_map(args.feature_map, args.rff_dim, args.rff_gamma, args.seed)
    clf = LinearClassifier(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        reg=args.reg,
        standardize=not args.no_standardize,
        feature_map=fmap,
        random_state=args.seed,
        verbose=True
    )

    clf.fit(X_tr, y_tr)
    tr_acc = clf.score(X_tr, y_tr)
    va_acc = clf.score(X_va, y_va)
    print(f"Train acc: {tr_acc:.4f}")
    print(f"Val   acc: {va_acc:.4f}")

if __name__ == "__main__":
    main()