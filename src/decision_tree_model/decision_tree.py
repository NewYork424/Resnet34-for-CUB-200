from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import math
import copy

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class Node:
    # 叶子
    is_leaf: bool
    prediction: Optional[int] = None  # 叶子预测的类别
    class_counts: Optional[np.ndarray] = None  # 节点样本的类别计数

    # 内部节点（连续）
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    # 内部节点（离散）
    children: Dict[Any, "Node"] = field(default_factory=dict)

    # 元信息
    depth: int = 0
    n_samples: int = 0
    impurity: float = 0.0

    def is_continuous_split(self) -> bool:
        return self.threshold is not None

    def majority_class(self) -> Optional[int]:
        if self.class_counts is None or len(self.class_counts) == 0:
            return self.prediction
        return int(np.argmax(self.class_counts))


class DecisionTreeBase:
    """
    通用决策树基类，支持连续与离散特征。
    新增支持预剪枝(pre_pruning)和后剪枝(post_pruning)。
    """
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        pre_pruning: bool = True,   # [新增] 是否开启预剪枝
        post_pruning: bool = False, # [新增] 是否开启后剪枝
        val_ratio: float = 0.2,     # [新增] 后剪枝用于验证的数据比例
    ):
        assert criterion in ("gini", "entropy", "gain_ratio")
        self.criterion = criterion
        
        # 原始参数保存
        self._original_max_depth = max_depth
        self._original_min_samples_split = max(2, int(min_samples_split))
        self._original_min_samples_leaf = max(1, int(min_samples_leaf))
        self._original_min_impurity_decrease = float(min_impurity_decrease)
        
        self.random_state = random_state
        self.pre_pruning = pre_pruning
        self.post_pruning = post_pruning
        self.val_ratio = val_ratio

        # 实际使用的参数（在 fit 时确定）
        self.max_depth = max_depth
        self.min_samples_split = self._original_min_samples_split
        self.min_samples_leaf = self._original_min_samples_leaf
        self.min_impurity_decrease = self._original_min_impurity_decrease

        self.n_classes_: int = 0
        self.feature_types_: List[str] = []
        self.root_: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    # -------------------------- 公共接口 --------------------------
    def fit(self, X: Union[np.ndarray, List[List[Any]]], y: Union[np.ndarray, List[Any]], feature_types: Optional[List[str]] = None):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], "X, y 形状不匹配"
        
        # [预剪枝逻辑]
        if not self.pre_pruning:
            # 如果关闭预剪枝，强制放宽限制，让树完全生长
            self.max_depth = float('inf')
            self.min_samples_split = 2
            self.min_samples_leaf = 1
            self.min_impurity_decrease = 0.0
        else:
            # 恢复原始限制
            self.max_depth = self._original_max_depth
            self.min_samples_split = self._original_min_samples_split
            self.min_samples_leaf = self._original_min_samples_leaf
            self.min_impurity_decrease = self._original_min_impurity_decrease

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # 推断特征类型
        if feature_types is None:
            self.feature_types_ = [self._infer_feature_type(X[:, j]) for j in range(X.shape[1])]
        else:
            assert len(feature_types) == X.shape[1], "feature_types 长度需等于特征维数"
            self.feature_types_ = feature_types

        # [后剪枝逻辑] 数据划分
        if self.post_pruning:
            # 划分训练集和验证集 (Reduced Error Pruning)
            n_total = len(y_encoded)
            n_val = int(n_total * self.val_ratio)
            indices = np.random.permutation(n_total)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            
            X_train, y_train = X[train_idx], y_encoded[train_idx]
            X_val, y_val = X[val_idx], y_encoded[val_idx]
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None

        # 构建树
        self.root_ = self._build_tree(X_train, y_train, depth=0)

        # 执行后剪枝
        if self.post_pruning and self.root_ is not None and X_val is not None:
            self._prune(self.root_, X_val, y_val)

        return self

    def predict(self, X: Union[np.ndarray, List[List[Any]]]) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        preds = [self._predict_one(self.root_, row) for row in X]
        return self.classes_[preds]

    def predict_proba(self, X: Union[np.ndarray, List[List[Any]]]) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        out = []
        for row in X:
            node = self._traverse(self.root_, row)
            counts = node.class_counts if node and node.class_counts is not None else np.zeros(self.n_classes_)
            proba = counts / max(1, counts.sum())
            out.append(proba)
        return np.vstack(out)

    def score(self, X, y) -> float:
        y = np.asarray(y)
        pred = self.predict(X)
        return float((pred == y).mean())

    def export_dict(self) -> Dict:
        assert self.root_ is not None
        return self._node_to_dict(self.root_)

    def plot(self, figsize: Tuple[int, int] = (12, 8), font_size: int = 10):
        if plt is None:
            raise RuntimeError("matplotlib 未安装")
        assert self.root_ is not None
        leafs = self._leaf_count(self.root_)
        depth = self._tree_depth(self.root_)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.axis("off")
        x_scale = 1.0 / (leafs + 1e-9)
        y_scale = 1.0 / (depth + 1)
        self._plot_node(ax, self.root_, 0.5, 1.0, x_scale, y_scale, font_size, [0.0], leafs)
        plt.tight_layout()
        plt.show()

    # -------------------------- 剪枝实现 --------------------------
    
    def _prune(self, node: Node, X_val: np.ndarray, y_val: np.ndarray):
        """
        后剪枝核心逻辑 (Reduced Error Pruning)
        自底向上：先递归处理子节点，再判断当前节点是否需要剪枝。
        """
        if node.is_leaf:
            return

        # 1. 将验证数据分发到子节点
        feat = node.feature_index
        ftype = self.feature_types_[feat]
        
        if ftype == "continuous":
            thr = node.threshold
            # 注意：这里需要转 float 比较
            try:
                col_val = X_val[:, feat].astype(float)
                left_mask = col_val <= thr
                right_mask = ~left_mask
            except:
                # 如果转换失败，无法判断，停止该分支剪枝
                return 

            if node.left:
                self._prune(node.left, X_val[left_mask], y_val[left_mask])
            if node.right:
                self._prune(node.right, X_val[right_mask], y_val[right_mask])
                
        else:
            # 离散特征
            col_val = X_val[:, feat]
            for val, child in node.children.items():
                mask = (col_val == val)
                self._prune(child, X_val[mask], y_val[mask])

        # 2. 尝试剪枝：比较 "保持子树" 与 "剪成叶子" 在验证集上的误差
        
        # 计算当前子树在验证集上的正确数
        # 注意：此时子节点可能已经被剪枝成叶子了
        acc_tree = self._evaluate_subtree(node, X_val, y_val)
        
        # 计算如果当前节点变成叶子，在验证集上的正确数
        # 变成叶子后的预测值是 majority_class
        majority = node.majority_class()
        if majority is None:
            return
            
        acc_leaf = np.sum(y_val == majority)

        # 如果变成叶子后，准确率没有下降（>=），则剪枝
        # (REP 通常要求严格提升，或者 >= 都可以，这里用 >= 偏向简化模型)
        if acc_leaf >= acc_tree:
            node.is_leaf = True
            node.left = None
            node.right = None
            node.children = {}
            node.prediction = majority
            # print(f"Pruned node at depth {node.depth}, samples {node.n_samples}")

    def _evaluate_subtree(self, node: Node, X: np.ndarray, y: np.ndarray) -> int:
        """计算子树在给定数据上的正确预测数量"""
        if len(y) == 0:
            return 0
        # 使用当前（可能部分剪枝的）子树进行预测
        # 为了效率，这里不调用全量 predict，而是局部递归
        correct = 0
        for i in range(len(y)):
            pred = self._predict_one(node, X[i])
            if pred == y[i]:
                correct += 1
        return correct

    # -------------------------- 内部构建逻辑 (保持不变) --------------------------
    def _infer_feature_type(self, col: np.ndarray) -> str:
        is_number = np.all([isinstance(v, (int, float, np.integer, np.floating)) for v in col])
        return "continuous" if is_number else "discrete"

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
        impurity_parent = self._impurity(y)

        node = Node(
            is_leaf=False,
            prediction=int(np.argmax(counts)),
            class_counts=counts,
            depth=depth,
            n_samples=n_samples,
            impurity=float(impurity_parent),
        )

        # 停止条件 (受 pre_pruning 控制的参数影响)
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           impurity_parent <= 1e-12 or \
           self.n_classes_ == 1:
            node.is_leaf = True
            return node

        best = self._best_split(X, y)
        if best is None:
            node.is_leaf = True
            return node

        feat, info = best
        ftype = self.feature_types_[feat]

        if ftype == "continuous":
            thr = info["threshold"]
            left_mask = X[:, feat].astype(float) <= thr
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                node.is_leaf = True
                return node

            impurity_left = self._impurity(y[left_mask])
            impurity_right = self._impurity(y[right_mask])
            weighted = (left_mask.sum() * impurity_left + right_mask.sum() * impurity_right) / n_samples
            gain = impurity_parent - weighted
            if gain < self.min_impurity_decrease:
                node.is_leaf = True
                return node

            node.feature_index = feat
            node.threshold = float(thr)
            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
            return node
        else:
            values = self._unique_values(X[:, feat])
            split_masks = [(val, X[:, feat] == val) for val in values]
            if any(mask.sum() < self.min_samples_leaf for _, mask in split_masks):
                node.is_leaf = True
                return node

            weighted_imp = 0.0
            for _, mask in split_masks:
                weighted_imp += (mask.sum() * self._impurity(y[mask]))
            weighted_imp /= max(1, n_samples)
            gain = impurity_parent - weighted_imp
            if gain < self.min_impurity_decrease:
                node.is_leaf = True
                return node

            node.feature_index = feat
            node.children = {}
            for val, mask in split_masks:
                node.children[val] = self._build_tree(X[mask], y[mask], depth + 1)
            return node

    def _unique_values(self, col: np.ndarray) -> List[Any]:
        seen = {}
        for v in col:
            seen.setdefault(v, True)
        return list(seen.keys())

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion in ("entropy", "gain_ratio"):
            return self._entropy(y)
        else:
            raise ValueError("未知 criterion")

    def _gini(self, y: np.ndarray) -> float:
        m = len(y)
        if m == 0: return 0.0
        p = np.bincount(y, minlength=self.n_classes_) / m
        return 1.0 - np.sum(p * p)

    def _entropy(self, y: np.ndarray) -> float:
        m = len(y)
        if m == 0: return 0.0
        p = np.bincount(y, minlength=self.n_classes_) / m
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, Dict[str, Any]]]:
        n_samples, n_features = X.shape
        base_imp = self._impurity(y)
        best_feat = None
        best_info = None
        best_score = -np.inf

        for feat in range(n_features):
            ftype = self.feature_types_[feat]
            col = X[:, feat]
            if ftype == "continuous":
                try:
                    vals = col.astype(float)
                except Exception:
                    ftype = "discrete"
            
            if ftype == "continuous":
                res = self._best_split_continuous(vals, y, base_imp)
                if res is None: continue
                thr, score = res
                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_info = {"type": "continuous", "threshold": float(thr)}
            else:
                res = self._best_split_discrete(col, y, base_imp)
                if res is None: continue
                score = res
                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_info = {"type": "discrete"}
        if best_feat is None: return None
        return best_feat, best_info

    def _best_split_continuous(self, vals: np.ndarray, y: np.ndarray, base_imp: float) -> Optional[Tuple[float, float]]:
        order = np.argsort(vals, kind="mergesort")
        vals_sorted = vals[order]
        y_sorted = y[order]
        m = len(y_sorted)
        diffs = vals_sorted[1:] != vals_sorted[:-1]
        cand_idx = np.where(diffs)[0]
        if cand_idx.size == 0: return None

        K = self.n_classes_
        left_counts = np.zeros((cand_idx.size, K), dtype=np.int64)
        right_counts = np.zeros((cand_idx.size, K), dtype=np.int64)
        cumsum = np.zeros((m, K), dtype=np.int64)
        for i in range(m):
            cumsum[i, y_sorted[i]] += 1
            if i > 0: cumsum[i] += cumsum[i - 1]
        for j, i in enumerate(cand_idx):
            left_counts[j] = cumsum[i]
            right_counts[j] = cumsum[-1] - cumsum[i]

        left_sizes = left_counts.sum(axis=1)
        right_sizes = right_counts.sum(axis=1)

        if self.criterion == "gini":
            left_gini = 1.0 - np.sum((left_counts / np.clip(left_sizes[:, None], 1, None))**2, axis=1)
            right_gini = 1.0 - np.sum((right_counts / np.clip(right_sizes[:, None], 1, None))**2, axis=1)
            weighted = (left_sizes * left_gini + right_sizes * right_gini) / (left_sizes + right_sizes)
            gain = base_imp - weighted
            best_i = int(np.argmax(gain))
            best_gain = float(gain[best_i])
        else:
            def entropy_from_counts(counts):
                s = counts.sum(axis=1, keepdims=True)
                p = counts / np.clip(s, 1, None)
                p = np.where(p > 0, p, 1.0)
                return -np.sum(p * np.log2(p), axis=1)
            left_ent = entropy_from_counts(left_counts)
            right_ent = entropy_from_counts(right_counts)
            weighted = (left_sizes * left_ent + right_sizes * right_ent) / (left_sizes + right_sizes)
            info_gain = base_imp - weighted
            if self.criterion == "entropy":
                best_i = int(np.argmax(info_gain))
                best_gain = float(info_gain[best_i])
            else:
                p_left = left_sizes / (left_sizes + right_sizes)
                p_right = 1.0 - p_left
                split_info = -(p_left * np.log2(np.clip(p_left, 1e-12, 1.0)) + p_right * np.log2(np.clip(p_right, 1e-12, 1.0)))
                gain_ratio = np.where(split_info > 0, info_gain / split_info, 0.0)
                best_i = int(np.argmax(gain_ratio))
                best_gain = float(gain_ratio[best_i])

        if not np.isfinite(best_gain): return None
        i = cand_idx[best_i]
        thr = (vals_sorted[i] + vals_sorted[i + 1]) / 2.0
        return float(thr), float(best_gain)

    def _best_split_discrete(self, col: np.ndarray, y: np.ndarray, base_imp: float) -> Optional[float]:
        values = self._unique_values(col)
        if len(values) <= 1: return None
        masks = [(col == v) for v in values]
        sizes = np.array([m.sum() for m in masks], dtype=float)
        if np.any(sizes < self.min_samples_leaf): return None

        if self.criterion == "gini":
            child_imp = 0.0
            for m in masks: child_imp += m.sum() * self._gini(y[m])
            child_imp /= len(y)
            return base_imp - child_imp
        else:
            child_imp = 0.0
            split_info = 0.0
            n = len(y)
            for m in masks:
                w = m.sum()
                if w == 0: continue
                child_imp += w * self._entropy(y[m])
                p = w / n
                split_info += -p * math.log2(p)
            child_imp /= n
            info_gain = base_imp - child_imp
            if self.criterion == "entropy": return info_gain
            else:
                if split_info <= 1e-12: return 0.0
                return info_gain / split_info

    def _predict_one(self, node: Node, row: np.ndarray) -> int:
        node = self._traverse(node, row)
        return node.majority_class()

    def _traverse(self, node: Node, row: np.ndarray) -> Node:
        while node and not node.is_leaf:
            feat = node.feature_index
            if node.is_continuous_split():
                val = float(row[feat])
                node = node.left if val <= node.threshold else node.right
            else:
                val = row[feat]
                node = node.children.get(val, None)
                if node is None: break
        return node if node else Node(is_leaf=True, prediction=None, class_counts=None)

    def _node_to_dict(self, node: Node) -> Union[Dict, int]:
        if node.is_leaf:
            if node.prediction is None: return -1
            return int(self.classes_[node.prediction])
        feat = node.feature_index
        if node.is_continuous_split():
            thr = _fmt_thr(node.threshold)
            return {int(feat): {f"<= {thr}": self._node_to_dict(node.left), f"> {thr}": self._node_to_dict(node.right)}}
        else:
            branch = {}
            for val, child in node.children.items(): branch[val] = self._node_to_dict(child)
            return {int(feat): branch}

    def _leaf_count(self, node: Node) -> int:
        if node is None: return 0
        if node.is_leaf: return 1
        if node.is_continuous_split(): return self._leaf_count(node.left) + self._leaf_count(node.right)
        return sum(self._leaf_count(c) for c in node.children.values())

    def _tree_depth(self, node: Node) -> int:
        if node is None or node.is_leaf: return 1
        if node.is_continuous_split(): return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))
        return 1 + (max([self._tree_depth(c) for c in node.children.values()]) if node.children else 0)

    def _plot_node(self, ax, node: Node, x: float, y: float, x_scale: float, y_scale: float, font_size: int, x_offset: List[float], total_leafs: int):
        if node.is_leaf:
            ax.text(x, y, f"Leaf\nclass={int(self.classes_[node.majority_class()])}\nN={node.n_samples}", ha="center", va="center", bbox=dict(boxstyle="round", fc="#dff0d8", ec="#333"), fontsize=font_size)
            return
        if node.is_continuous_split():
            title = f"[{node.feature_index}] <= { _fmt_thr(node.threshold) }\nN={node.n_samples}"
            ax.text(x, y, title, ha="center", va="center", bbox=dict(boxstyle="round", fc="#d9edf7", ec="#333"), fontsize=font_size)
            left_leafs = self._leaf_count(node.left)
            right_leafs = self._leaf_count(node.right)
            lx = x - (left_leafs / total_leafs) * 0.5
            ly = y - y_scale
            self._plot_node(ax, node.left, lx, ly, x_scale, y_scale, font_size, x_offset, total_leafs)
            ax.annotate("", xy=(lx, ly + 0.03), xytext=(x, y - 0.03), arrowprops=dict(arrowstyle="-"))
            rx = x + (right_leafs / total_leafs) * 0.5
            ry = y - y_scale
            self._plot_node(ax, node.right, rx, ry, x_scale, y_scale, font_size, x_offset, total_leafs)
            ax.annotate("", xy=(rx, ry + 0.03), xytext=(x, y - 0.03), arrowprops=dict(arrowstyle="-"))
        else:
            title = f"[{node.feature_index}] (discrete)\nN={node.n_samples}"
            ax.text(x, y, title, ha="center", va="center", bbox=dict(boxstyle="round", fc="#d9edf7", ec="#333"), fontsize=font_size)
            child_leafs = [self._leaf_count(c) for c in node.children.values()]
            total = sum(child_leafs)
            if total == 0: total = 1
            cum = 0.0
            for (val, child), nlf in zip(node.children.items(), child_leafs):
                cx = x - 0.5 + (cum + nlf / 2) / total
                cy = y - y_scale
                self._plot_node(ax, child, cx, cy, x_scale, y_scale, font_size, x_offset, total_leafs)
                ax.annotate(str(val), xy=(cx, cy + 0.03), xytext=(x, y - 0.03), textcoords="data", ha="center", fontsize=font_size-1, arrowprops=dict(arrowstyle="-"))
                cum += nlf


class CARTDecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, **kwargs):
        super().__init__(criterion="gini", **kwargs)


class C45DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, **kwargs):
        super().__init__(criterion="gain_ratio", **kwargs)


def _fmt_thr(v: float) -> str:
    if v is None: return "None"
    s = f"{float(v):.6f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s


# -------------------------- 使用示例（注释） --------------------------
# X = np.array([
#     [5.1, 3.5, "A"],
#     [4.9, 3.0, "B"],
#     [6.2, 3.4, "A"],
#     [5.9, 3.0, "C"],
# ], dtype=object)
# y = np.array([0, 0, 1, 1])
#
# # 自动推断：数值列视为连续，字符串列视为离散
# clf = C45DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_impurity_decrease=0.0)
# clf.fit(X, y)
# print("acc:", clf.score(X, y))
# print("tree:", clf.export_dict())
# clf.plot()