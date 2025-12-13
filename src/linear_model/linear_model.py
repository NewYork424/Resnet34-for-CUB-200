import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------- 特征标准化 ----------
@dataclass
class Standardizer:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

# ---------- 特征映射 ----------
class IdentityFeatures:
    def fit(self, X: np.ndarray):
        self.in_dim_ = X.shape[1]
        self.out_dim_ = self.in_dim_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

class Polynomial2Features:
    """
    二次多项式扩展：phi(x) = [x, x_i*x_j (i<=j)]
    输出维度 = D + D*(D+1)/2
    """
    def fit(self, X: np.ndarray):
        D = X.shape[1]
        self.in_dim_ = D
        self.out_dim_ = D + (D*(D+1))//2
        # 预先缓存上三角索引
        self.triu_idx_ = np.triu_indices(D)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        quad = (X[:, :, None] * X[:, None, :])[:, self.triu_idx_[0], self.triu_idx_[1]]  # [N, D*(D+1)/2]
        return np.concatenate([X, quad], axis=1)

class RBFRandomFourierFeatures:
    """
    随机傅里叶特征近似 RBF 核：
      phi(x) = sqrt(2/M) * cos(xW + b),  W~N(0, 2*gamma*I), b~U(0, 2pi)
    参数:
      n_components: 输出维度 M
      gamma: RBF 参数 (1 / (2*sigma^2))
      random_state: 随机种子
    """
    def __init__(self, n_components: int = 256, gamma: float = 0.5, random_state: Optional[int] = 42):
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        D = X.shape[1]
        self.W_ = rng.normal(loc=0.0, scale=np.sqrt(2*self.gamma), size=(D, self.n_components)).astype(np.float32)
        self.b_ = rng.uniform(low=0.0, high=2*np.pi, size=(self.n_components,)).astype(np.float32)
        self.in_dim_ = D
        self.out_dim_ = self.n_components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W_ + self.b_
        return np.sqrt(2.0 / self.n_components) * np.cos(Z)

# ---------- 线性分类器（Softmax 回归） ----------
class LinearClassifier:
    """
    多类线性分类器（Softmax 回归）
    - 优化：小批量 SGD
    - 正则：L2
    """
    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 50,
        batch_size: int = 128,
        reg: float = 1e-4,
        standardize: bool = True,
        feature_map: Optional[object] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True
    ):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.reg = float(reg)
        self.standardize = bool(standardize)
        self.feature_map = feature_map if feature_map is not None else IdentityFeatures()
        self.random_state = random_state
        self.verbose = verbose

        self.std_ = Standardizer() if self.standardize else None
        self.W_: Optional[np.ndarray] = None  # [D, K]
        self.b_: Optional[np.ndarray] = None  # [K]
        self.n_classes_: int = 0

    # ---------- 公共接口 ----------
    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        N, D = X.shape
        self.n_classes_ = int(np.max(y)) + 1

        # 标准化
        if self.std_ is not None:
            X = self.std_.fit_transform(X)

        # 特征映射
        self.feature_map.fit(X)
        X = self.feature_map.transform(X)
        N, Dm = X.shape

        # 参数初始化
        self.W_ = rng.normal(scale=0.01, size=(Dm, self.n_classes_)).astype(np.float32)
        self.b_ = np.zeros((self.n_classes_,), dtype=np.float32)

        # 训练
        for ep in range(self.epochs):
            idx = rng.permutation(N)
            X_shuf, y_shuf = X[idx], y[idx]
            for start in range(0, N, self.batch_size):
                end = min(N, start + self.batch_size)
                xb = X_shuf[start:end]   # [B, Dm]
                yb = y_shuf[start:end]   # [B]

                # 前向
                logits = xb @ self.W_ + self.b_  # [B, K]
                logits -= logits.max(axis=1, keepdims=True)  # 稳定
                exp = np.exp(logits, dtype=np.float32)
                probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)  # [B, K]

                # 损失
                loss = -np.log(np.clip(probs[np.arange(len(yb)), yb], 1e-12, None)).mean()
                loss += 0.5 * self.reg * float((self.W_ * self.W_).sum())

                # 反向
                grad = probs
                grad[np.arange(len(yb)), yb] -= 1.0
                grad /= len(yb)
                gW = xb.T @ grad + self.reg * self.W_
                gb = grad.sum(axis=0)

                # 更新
                self.W_ -= self.lr * gW
                self.b_ -= self.lr * gb

            if self.verbose and ((ep + 1) % max(1, self.epochs // 10) == 0 or ep == self.epochs - 1):
                acc = self.score(X, y, already_mapped=True)
                print(f"[Epoch {ep+1:3d}] loss={loss:.4f} acc={acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray, already_mapped: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not already_mapped:
            if self.std_ is not None:
                X = self.std_.transform(X)
            X = self.feature_map.transform(X)
        logits = X @ self.W_ + self.b_
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
        return probs

    def predict(self, X: np.ndarray, already_mapped: bool = False) -> np.ndarray:
        probs = self.predict_proba(X, already_mapped=already_mapped)
        return probs.argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray, already_mapped: bool = False) -> float:
        y = np.asarray(y).reshape(-1)
        pred = self.predict(X, already_mapped=already_mapped)
        return float((pred == y).mean())