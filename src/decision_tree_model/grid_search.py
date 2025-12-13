import argparse
import itertools
import json
import os
import sys
import time
import numpy as np

from decision_tree import C45DecisionTreeClassifier, CARTDecisionTreeClassifier  # noqa: E402
# 注意：这里假设 DatasetTree 存在，或者你可以改为使用 UnifiedBirdDataset
from dataset import DatasetTree  # noqa: E402


def parameter_grid(grid: dict):
    """
    简易版 ParameterGrid 生成器
    """
    keys = list(grid.keys())
    values = [v if isinstance(v, (list, tuple)) else [v] for v in (grid[k] for k in keys)]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_model(algo: str, params: dict):
    """
    根据 algo 构造模型实例
    """
    kwargs = dict(params)
    kwargs.pop('algo', None)
    if algo.lower() == 'c45':
        return C45DecisionTreeClassifier(**kwargs)
    elif algo.lower() == 'cart':
        return CARTDecisionTreeClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def load_dataset(data_root: str):
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')
    train_ds = DatasetTree(train_path, img_only=False, attr_only=True)
    val_ds = DatasetTree(val_path, img_only=False, attr_only=True)

    tr_imgs, tr_attrs, tr_labels = train_ds.get_data()
    va_imgs, va_attrs, va_labels = val_ds.get_data()

    X_tr = np.asarray(tr_attrs, dtype=object)
    y_tr = np.asarray(tr_labels)
    X_va = np.asarray(va_attrs, dtype=object)
    y_va = np.asarray(va_labels)
    return X_tr, y_tr, X_va, y_va


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Decision Tree (CART/C4.5)")
    parser.add_argument("--data-root", type=str, default="./data", help="数据集根目录，包含 train/ 与 val/")
    parser.add_argument("--grid", type=str, default="", help="JSON 字符串或 JSON 文件路径，覆盖默认搜索空间")
    parser.add_argument("--save-csv", type=str, default="", help="保存搜索结果到 CSV")
    parser.add_argument("--plot-best", action="store_true", help="对最优模型进行可视化绘图")
    args = parser.parse_args()

    # [修改] 默认搜索空间，加入剪枝参数
    default_grid = {
        "algo": ["c45", "cart"],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 5],
        "min_impurity_decrease": [0.0, 1e-3],
        "pre_pruning": [True, False],  # [新增]
        "post_pruning": [True, False], # [新增]
        "random_state": [42],
    }

    if args.grid:
        if os.path.isfile(args.grid):
            with open(args.grid, "r", encoding="utf-8") as f:
                user_grid = json.load(f)
        else:
            user_grid = json.loads(args.grid)
        default_grid.update(user_grid)

    print(f"[INFO] Loading dataset from: {args.data_root}")
    X_tr, y_tr, X_va, y_va = load_dataset(args.data_root)
    print(f"[INFO] Train: {X_tr.shape}, Val: {X_va.shape}")

    best_score = -1.0
    best_params = None
    best_model = None
    results = []

    total = 1
    for v in default_grid.values():
        total *= (len(v) if isinstance(v, (list, tuple)) else 1)
    print(f"[INFO] Total combinations: {total}")

    start_all = time.time()
    for i, params in enumerate(parameter_grid(default_grid), start=1):
        algo = params.get("algo", "c45")
        model = build_model(algo, params)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_acc = model.score(X_tr, y_tr)
        val_acc = model.score(X_va, y_va)
        elapsed = time.time() - t0

        results.append({
            "algo": algo,
            "params": {k: v for k, v in params.items() if k != "algo"},
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "time_sec": round(elapsed, 3),
        })

        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            best_model = model

        print(f"[{i}/{total}] algo={algo}, pre={params.get('pre_pruning')}, post={params.get('post_pruning')}, "
              f"val_acc={val_acc:.4f}, time={elapsed:.2f}s")

    print("\n[RESULT] Best val_acc: %.4f" % best_score)
    print("[RESULT] Best params:", best_params)

    # 保存 CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["algo", "params", "train_acc", "val_acc", "time_sec"])
            for r in results:
                writer.writerow([r["algo"], json.dumps(r["params"], ensure_ascii=False), r["train_acc"], r["val_acc"], r["time_sec"]])
        print(f"[INFO] Saved results to: {args.save_csv}")

    # 可视化最优模型
    if args.plot_best and best_model is not None:
        try:
            best_model.plot()
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

    print("[INFO] Done. Elapsed: %.2fs" % (time.time() - start_all))


if __name__ == "__main__":
    main()