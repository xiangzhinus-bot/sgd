import numpy as np
import math
from typing import Tuple

# -------- Activation functions --------
class ActivationFunctions:
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
        return np.where(x >= 0, x, negative_slope * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
        out = np.ones_like(x)
        out[x < 0] = negative_slope
        return out

# -------- Simple 1-hidden-layer regressor --------
class NeuralNetwork:
    """
    Input m -> Hidden H (activation) -> Output 1 (linear)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        rng: np.random.Generator | None = None,
        init_method: str = "xavier",
        leaky_slope: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rng = rng if rng is not None else np.random.default_rng()
        self.leaky_slope = leaky_slope

        if activation == "tanh":
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        elif activation == "relu":
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == "leaky_relu":
            # bind slope via small wrappers
            self.activation = lambda x: ActivationFunctions.leaky_relu(x, self.leaky_slope)
            self.activation_derivative = lambda x: ActivationFunctions.leaky_relu_derivative(x, self.leaky_slope)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._initialize_parameters(init_method)

    def _initialize_parameters(self, method: str):
        m, H = self.input_dim, self.hidden_dim
        if method == "xavier":
            std_hidden = math.sqrt(2.0 / (m + H))
            std_output = math.sqrt(2.0 / (H + 1))
        elif method == "he":
            std_hidden = math.sqrt(2.0 / m)
            std_output = math.sqrt(2.0 / H)
        else:
            raise ValueError(f"Unknown init method: {method}")

        self.W_hidden = self.rng.normal(0.0, std_hidden, size=(H, m))
        self.b_hidden = np.zeros(H)
        self.W_output = self.rng.normal(0.0, std_output, size=(H,))
        self.b_output = 0.0

    # ---- parameter pack/unpack (for MPI sync) ----
    def get_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.W_output.ravel(),
            np.array([self.b_output]),
            self.W_hidden.ravel(),
            self.b_hidden.ravel()
        ])

    def set_parameters(self, theta: np.ndarray):
        H, m = self.hidden_dim, self.input_dim
        idx = 0
        self.W_output = theta[idx: idx + H]; idx += H
        self.b_output = float(theta[idx]); idx += 1
        self.W_hidden = theta[idx: idx + H * m].reshape(H, m); idx += H * m
        self.b_hidden = theta[idx: idx + H]

    # ---- forward / predict ----
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = X @ self.W_hidden.T + self.b_hidden
        h = self.activation(z)
        yhat = h @ self.W_output + self.b_output
        return yhat, h, z

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat, _, _ = self.forward(X)
        return yhat

    # ---- loss + grad (MSE + L2) ----
    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> Tuple[float, np.ndarray]:
        B = X.shape[0]
        yhat, h, z = self.forward(X)
        err = yhat - y
        mse = 0.5 * np.mean(err ** 2)
        l2 = 0.5 * weight_decay * (np.sum(self.W_hidden ** 2) + np.sum(self.W_output ** 2))
        loss = mse + l2

        dL_dy = err / B
        grad_W_out = dL_dy @ h
        grad_b_out = np.sum(dL_dy)

        dh = np.outer(dL_dy, self.W_output)
        dz = dh * self.activation_derivative(z)
        grad_W_hid = dz.T @ X
        grad_b_hid = np.sum(dz, axis=0)

        grad_W_out += weight_decay * self.W_output
        grad_W_hid += weight_decay * self.W_hidden

        grad = np.concatenate([
            grad_W_out.ravel(),
            np.array([grad_b_out]),
            grad_W_hid.ravel(),
            grad_b_hid.ravel()
        ])
        return float(loss), grad
