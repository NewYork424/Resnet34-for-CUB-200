import argparse
import itertools
import json
import os
import sys
import time
import numpy as np

from dataset import DatasetLinear
from linear_model import (
    LinearClassifier,
    IdentityFeatures,
    Polynomial2Features,
    RBFRandomFourierFeatures,
)

def load_attrs_split(root: str):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_ds = DatasetLinear(train_dir, img_only=False, attr_only=True)
    val_ds = DatasetLinear(val_dir, img_only=False, attr_only=True)

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

def parameter_grid(grid: dict):
    keys = list(grid.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in (grid[k] for k in keys)]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def main():
    parser = argparse.ArgumentParser(description="Grid Search for LinearClassifier on attribute features")
    parser.add_argument("--data-root", type=str, default="./data", help="数据集根目录，包含 train/ 与 val/ 子目录")
    parser.add_argument("--grid", type=str, default="", help="JSON 字符串或 JSON 文件路径，覆盖默认搜索空间")
    parser.add_argument("--save-csv", type=str, default="", help="保存搜索结果到 CSV")
    args = parser.parse_args()

    # 默认搜索空间
    default_grid = {
        "feature_map": ["poly2"], # "identity", 
        "lr": [5e-2, 1e-1],
        "epochs": [200],
        "batch_size": [256],
        "reg": [1e-4, 1e-2],
        "standardize": [True],
        "seed": [42],

        # 仅在 feature_map == "rbf" 时有效
        "rff_dim": [256, 512],
        "rff_gamma": [0.25, 0.5, 1.0],
    }

    # 支持通过 JSON 字符串或文件覆盖网格
    if args.grid:
        if os.path.isfile(args.grid):
            with open(args.grid, "r", encoding="utf-8") as f:
                user_grid = json.load(f)
        else:
            user_grid = json.loads(args.grid)
        default_grid.update(user_grid)

    print(f"[INFO] Loading dataset from: {args.data_root}")
    X_tr, y_tr, X_va, y_va = load_attrs_split(args.data_root)
    print(f"[INFO] Train: X={X_tr.shape}, Val: X={X_va.shape}, n_classes={int(np.max(y_tr))+1}")

    # 统计组合数（考虑 feature_map 依赖关系）
    all_params = list(parameter_grid(default_grid))
    # 实际遍历时会跳过不适用的 rbf 参数组合
    total = len(all_params)

    best = {
        "val_acc": -1.0,
        "params": None,
        "train_acc": None,
    }
    results = []

    t_all = time.time()
    run_idx = 0
    for params in all_params:
        fmap_name = params["feature_map"]
        # 对非 RBF 映射，忽略 rff_* 参数
        rff_dim = params.get("rff_dim", 256)
        rff_gamma = params.get("rff_gamma", 0.5)

        # 构建特征映射
        fmap = build_feature_map(fmap_name, rff_dim, rff_gamma, params["seed"])

        # 构建模型
        clf = LinearClassifier(
            lr=float(params["lr"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            reg=float(params["reg"]),
            standardize=bool(params["standardize"]),
            feature_map=fmap,
            random_state=int(params["seed"]),
            verbose=False,  # 网格搜索关闭详细日志
        )

        t0 = time.time()
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        val_acc = clf.score(X_va, y_va)
        elapsed = time.time() - t0

        run_idx += 1
        summary = {
            "feature_map": fmap_name,
            "lr": params["lr"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "reg": params["reg"],
            "standardize": params["standardize"],
            "seed": params["seed"],
            "rff_dim": rff_dim if fmap_name == "rbf" else None,
            "rff_gamma": rff_gamma if fmap_name == "rbf" else None,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "time_sec": round(elapsed, 3),
        }
        results.append(summary)

        if val_acc > best["val_acc"]:
            best["val_acc"] = float(val_acc)
            best["train_acc"] = float(train_acc)
            best["params"] = summary

        print(f"[{run_idx}/{total}] fmap={fmap_name:7s}, lr={params['lr']:.3g}, ep={params['epochs']}, "
              f"bs={params['batch_size']}, reg={params['reg']:.1e}, "
              f"{('rff_dim='+str(rff_dim)+', rff_gamma='+str(rff_gamma)) if fmap_name=='rbf' else ' '}"
              f" | train={train_acc:.4f}, val={val_acc:.4f}, time={elapsed:.2f}s")

    print("\n[RESULT] Best val_acc: %.4f (train_acc=%.4f)" % (best["val_acc"], best["train_acc"]))
    print("[RESULT] Best params:")
    print(json.dumps(best["params"], indent=2, ensure_ascii=False))

    # 保存 CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["feature_map", "lr", "epochs", "batch_size", "reg", "standardize", "seed", "rff_dim", "rff_gamma", "train_acc", "val_acc", "time_sec"]
            writer.writerow(header)
            for r in results:
                writer.writerow([r[h] for h in header])
        print(f"[INFO] Saved results to: {args.save_csv}")

    print("[INFO] Done. Elapsed: %.2fs" % (time.time() - t_all))

if __name__ == "__main__":
    main()