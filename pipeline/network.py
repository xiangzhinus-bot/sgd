import numpy as np
from typing import Callable, Tuple, List
import math

class ActivationFunctions:
    """激活函数及其导数"""
    
    @staticmethod
    def sigmoid(x):
        # 防止溢出
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

class NeuralNetwork:
    """单隐层神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'sigmoid'):
        """
        初始化神经网络
        
        Args:
            input_dim: 输入维度 (m)
            hidden_dim: 隐藏层神经元数量 (n)
            activation: 激活函数类型
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        
        # 设置激活函数
        if activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        elif activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == 'leaky_relu':
            self.activation = ActivationFunctions.leaky_relu
            self.activation_derivative = ActivationFunctions.leaky_relu_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # 初始化参数
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化网络参数"""
        # Xavier/Glorot 初始化
        std_hidden = math.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        std_output = math.sqrt(2.0 / (self.hidden_dim + 1))
        
        # 隐藏层权重 w_jk (hidden_dim x input_dim)
        self.W_hidden = np.random.normal(0, std_hidden, (self.hidden_dim, self.input_dim))
        # 隐藏层偏置 w_j,m+1 (hidden_dim,)
        self.b_hidden = np.random.normal(0, std_hidden, (self.hidden_dim,))
        
        # 输出层权重 w_j (hidden_dim,)
        self.W_output = np.random.normal(0, std_output, (self.hidden_dim,))
        # 输出层偏置 w_n+1
        self.b_output = np.random.normal(0, std_output)
    
    def get_parameters(self) -> np.ndarray:
        """获取所有参数的扁平化向量"""
        return np.concatenate([
            self.W_output.flatten(),
            [self.b_output],
            self.W_hidden.flatten(),
            self.b_hidden.flatten()
        ])
    
    def set_parameters(self, theta: np.ndarray):
        """从扁平化向量设置参数"""
        idx = 0
        
        # 输出层权重
        self.W_output = theta[idx:idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        
        # 输出层偏置
        self.b_output = theta[idx]
        idx += 1
        
        # 隐藏层权重
        hidden_weights_size = self.hidden_dim * self.input_dim
        self.W_hidden = theta[idx:idx + hidden_weights_size].reshape(self.hidden_dim, self.input_dim)
        idx += hidden_weights_size
        
        # 隐藏层偏置
        self.b_hidden = theta[idx:idx + self.hidden_dim].copy()
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_dim)
            
        Returns:
            output: 网络输出 (batch_size,)
            hidden_output: 隐藏层输出 (batch_size, hidden_dim)
            hidden_input: 隐藏层输入 (batch_size, hidden_dim)
        """
        # 隐藏层计算
        hidden_input = np.dot(X, self.W_hidden.T) + self.b_hidden  # (batch_size, hidden_dim)
        hidden_output = self.activation(hidden_input)  # (batch_size, hidden_dim)
        
        # 输出层计算
        output = np.dot(hidden_output, self.W_output) + self.b_output  # (batch_size,)
        
        return output, hidden_output, hidden_input
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        output, _, _ = self.forward(X)
        return output
    
    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算损失和梯度
        
        Args:
            X: 输入数据 (batch_size, input_dim)
            y: 目标值 (batch_size,)
            
        Returns:
            loss: 损失值
            gradient: 参数梯度的扁平化向量
        """
        batch_size = X.shape[0]
        
        # 前向传播
        output, hidden_output, hidden_input = self.forward(X)
        
        # 计算损失 (MSE)
        loss = 0.5 * np.mean((output - y) ** 2)
        
        # 反向传播
        # 输出层梯度
        output_error = (output - y) / batch_size  # (batch_size,)
        
        grad_W_output = np.dot(output_error, hidden_output)  # (hidden_dim,)
        grad_b_output = np.sum(output_error)
        
        # 隐藏层梯度
        hidden_error = np.outer(output_error, self.W_output)  # (batch_size, hidden_dim)
        hidden_error *= self.activation_derivative(hidden_input)  # (batch_size, hidden_dim)
        
        grad_W_hidden = np.dot(hidden_error.T, X)  # (hidden_dim, input_dim)
        grad_b_hidden = np.sum(hidden_error, axis=0)  # (hidden_dim,)
        
        # 组装梯度向量
        gradient = np.concatenate([
            grad_W_output.flatten(),
            [grad_b_output],
            grad_W_hidden.flatten(),
            grad_b_hidden.flatten()
        ])
        
        return loss, gradient
    
    def compute_rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算RMSE"""
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return np.sqrt(mse)