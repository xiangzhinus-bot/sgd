# network.py
import numpy as np
from typing import Tuple
import math

class ActivationFunctions:
    @staticmethod
    def sigmoid(x): x = np.clip(x, -500, 500); return 1.0 / (1.0 + np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x): s = ActivationFunctions.sigmoid(x); return s * (1 - s)
    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def tanh_derivative(x): return 1 - np.tanh(x) ** 2
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_derivative(x): return (x > 0).astype(float)
    @staticmethod
    def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01): return np.where(x > 0, 1.0, alpha)

class NeuralNetwork:
    """
    Single-hidden-layer regression network.
    Param layout in flat vector:
      [W_out (H), b_out (1), W_hid (H*m), b_hid (H)]
    """
    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh", rng: np.random.Generator | None = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.rng = rng if rng is not None else np.random.default_rng()

        if activation == "sigmoid":
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == "tanh":
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        elif activation == "relu":
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == "leaky_relu":
            self.activation = ActivationFunctions.leaky_relu
            self.activation_derivative = ActivationFunctions.leaky_relu_derivative
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._initialize_parameters()

    def _initialize_parameters(self):
        m, H = self.input_dim, self.hidden_dim
        std_hidden = math.sqrt(2.0 / (m + H))
        std_output = math.sqrt(2.0 / (H + 1))
        self.W_hidden = self.rng.normal(0.0, std_hidden, size=(H, m)).astype(np.float64)
        self.b_hidden = self.rng.normal(0.0, std_hidden, size=(H,)).astype(np.float64)
        self.W_output = self.rng.normal(0.0, std_output, size=(H,)).astype(np.float64)
        self.b_output = float(self.rng.normal(0.0, std_output))

    # ---- parameter vector helpers ----
    def get_parameters(self) -> np.ndarray:
        return np.concatenate([self.W_output.ravel(), np.array([self.b_output]), self.W_hidden.ravel(), self.b_hidden.ravel()]).astype(np.float64)

    def set_parameters(self, theta: np.ndarray):
        H, m = self.hidden_dim, self.input_dim
        idx = 0
        self.W_output = theta[idx:idx + H].astype(np.float64); idx += H
        self.b_output = float(theta[idx]); idx += 1
        self.W_hidden = theta[idx:idx + H * m].reshape(H, m).astype(np.float64); idx += H * m
        self.b_hidden = theta[idx:idx + H].astype(np.float64)

    # ---- forward / loss / grad ----
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = X @ self.W_hidden.T + self.b_hidden  # (B, H)
        h = self.activation(z)                   # (B, H)
        yhat = h @ self.W_output + self.b_output # (B,)
        return yhat, h, z

    def predict(self, X: np.ndarray) -> np.ndarray:
        yhat, _, _ = self.forward(X); return yhat

    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        MSE + L2 (weight_decay * 0.5 * ||theta||^2) on weights (not biases).
        """
        B = X.shape[0]
        yhat, h, z = self.forward(X)
        diff = (yhat - y)                                # (B,)
        mse = 0.5 * np.mean(diff ** 2)

        # L2 regularization on weights only
        l2 = 0.5 * weight_decay * (np.sum(self.W_hidden ** 2) + np.sum(self.W_output ** 2))
        loss = mse + l2

        # Backprop
        dL_dyhat = diff / B                              # (B,)
        grad_W_out = dL_dyhat @ h                        # (H,)
        grad_b_out = np.sum(dL_dyhat)

        dh = np.outer(dL_dyhat, self.W_output)           # (B, H)
        dz = dh * self.activation_derivative(z)          # (B, H)

        grad_W_hid = dz.T @ X                            # (H, m)
        grad_b_hid = np.sum(dz, axis=0)                  # (H,)

        # Add weight decay gradients
        grad_W_out += weight_decay * self.W_output
        grad_W_hid += weight_decay * self.W_hidden

        grad = np.concatenate([grad_W_out.ravel(), np.array([grad_b_out]), grad_W_hid.ravel(), grad_b_hid.ravel()])
        return float(loss), grad

    def compute_rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X).astype(np.float64)
        return float(np.sqrt(np.mean((pred - y) ** 2)))
