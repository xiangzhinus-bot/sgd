# network.py
import numpy as np
import math
from typing import Tuple

class NeuralNetwork:
    """
    单隐藏层回归 MLP（优化）
    - float32 计算；通信用 float64
    - tanh 复用激活值；ReLU/LeakyReLU 原地掩码
    - 与 sgd.py 接口：compute_loss_and_gradient / predict / get/set_parameters
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        dtype=np.float32,
        init_method: str | None = None,
        leaky_slope: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.activation_name = activation
        self.leaky_slope = np.array(leaky_slope, dtype=self.dtype)

        if init_method is None:
            init_method = "he" if activation in ("relu", "leaky_relu") else "xavier"
        self._initialize_parameters(init_method)

        self.X = None
        self.z1 = None
        self.h1 = None
        self.yhat = None

    def _initialize_parameters(self, method: str):
        m, H = self.input_dim, self.hidden_dim
        if method == "xavier":
            std_hidden = math.sqrt(2.0 / (m + H))
            std_out = math.sqrt(2.0 / (H + 1))
        elif method == "he":
            std_hidden = math.sqrt(2.0 / m)
            std_out = math.sqrt(2.0 / H)
        else:
            raise ValueError(f"Unknown init method: {method}")

        self.W1 = np.random.normal(0.0, std_hidden, (H, m)).astype(self.dtype)
        self.b1 = np.zeros(H, dtype=self.dtype)
        self.W2 = np.random.normal(0.0, std_out, H).astype(self.dtype)
        self.b2 = np.zeros(1, dtype=self.dtype)

    def _act_forward(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            h = z.copy()
            np.maximum(h, 0, out=h)
            return h
        elif self.activation_name == "leaky_relu":
            h = z.copy()
            neg = h < 0
            h[neg] *= self.leaky_slope
            return h
        elif self.activation_name == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != self.dtype:
            X = X.astype(self.dtype, copy=False)
        self.X = X
        self.z1 = X @ self.W1.T + self.b1
        self.h1 = self._act_forward(self.z1)
        self.yhat = self.h1 @ self.W2 + self.b2
        return self.yhat.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X.astype(self.dtype, copy=False) @ self.W1.T + self.b1
        h = self._act_forward(z)
        y = h @ self.W2 + self.b2
        return y.ravel()

    def _act_backward(self, dh: np.ndarray, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            dz = dh.copy()
            dz[z <= 0] = 0
            return dz
        elif self.activation_name == "leaky_relu":
            dz = dh.copy()
            neg = (z < 0)
            dz[neg] *= self.leaky_slope
            return dz
        elif self.activation_name == "tanh":
            return dh * (1.0 - h * h)
        else:
            raise ValueError

    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> Tuple[float, np.ndarray]:
        B = X.shape[0]
        yhat = self.forward(X)
        y = y.astype(self.dtype, copy=False)

        err = yhat - y
        mse = 0.5 * np.mean(err * err)  # 注意：这是 0.5*MSE（不含 sqrt）
        l2 = 0.5 * weight_decay * ((self.W1 * self.W1).sum() + (self.W2 * self.W2).sum())
        loss = float(mse + l2)

        dL_dy = err.astype(self.dtype, copy=False) / np.float32(B)

        gW2 = dL_dy @ self.h1
        gb2 = dL_dy.sum()

        dL_dh1 = np.outer(dL_dy, self.W2)
        dL_dz1 = self._act_backward(dL_dh1, self.z1, self.h1)

        gW1 = dL_dz1.T @ self.X
        gb1 = dL_dz1.sum(axis=0)

        if weight_decay > 0:
            gW2 += weight_decay * self.W2
            gW1 += weight_decay * self.W1

        grad = np.concatenate([
            self.W2.ravel(), np.array([gb2], dtype=self.dtype),
            self.W1.ravel(), gb1.ravel()
        ]).astype(np.float64, copy=False)
        return loss, grad

    def get_parameters(self) -> np.ndarray:
        theta = np.concatenate([
            self.W2.ravel(), self.b2.ravel(),
            self.W1.ravel(), self.b1.ravel()
        ]).astype(np.float64, copy=False)
        return theta

    def set_parameters(self, theta: np.ndarray):
        theta = theta.astype(self.dtype, copy=False)
        H, m = self.hidden_dim, self.input_dim
        idx = 0
        self.W2 = theta[idx: idx + H]; idx += H
        self.b2 = theta[idx: idx + 1]; idx += 1
        self.W1 = theta[idx: idx + H * m].reshape(H, m); idx += H * m
        self.b1 = theta[idx: idx + H]
