import numpy as np
from mpi4py import MPI
from typing import List, Tuple, Optional, Dict, Any
import time
import random
from network import NeuralNetwork

class MPISGDTrainer:
    """基于MPI的随机梯度下降训练器"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 activation: str = 'sigmoid',
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 max_epochs: int = 100,
                 tolerance: float = 1e-6,
                 patience: int = 10):
        """
        初始化训练器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层神经元数量
            activation: 激活函数
            learning_rate: 学习率
            batch_size: 批次大小
            max_epochs: 最大迭代次数
            tolerance: 损失变化容忍度
            patience: 早停耐心值
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.patience = patience
        
        # 初始化神经网络
        self.network = NeuralNetwork(input_dim, hidden_dim, activation)
        
        # 训练历史
        self.history = {
            'loss': [],
            'epoch_times': []
        }
        
        # 收敛标志
        self.converged = False
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _generate_minibatch_indices(self, n_samples: int, epoch: int) -> List[int]:
        """生成小批次索引"""
        # 使用epoch作为随机种子确保每次迭代都不同
        random.seed(42 + epoch + self.rank)
        
        if self.batch_size >= n_samples:
            return list(range(n_samples))
        
        # 随机采样不重复的索引
        indices = random.sample(range(n_samples), min(self.batch_size, n_samples))
        return indices
    
    def _compute_local_gradient(self, X: np.ndarray, y: np.ndarray, epoch: int) -> Tuple[float, np.ndarray]:
        """计算本地梯度"""
        n_samples = X.shape[0]
        
        if n_samples == 0:
            # 如果没有数据，返回零梯度
            param_size = len(self.network.get_parameters())
            return 0.0, np.zeros(param_size)
        
        # 生成小批次索引
        batch_indices = self._generate_minibatch_indices(n_samples, epoch)
        
        # 获取小批次数据
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # 计算损失和梯度
        loss, gradient = self.network.compute_loss_and_gradient(X_batch, y_batch)
        
        return loss, gradient
    
    def _all_reduce_gradient(self, local_loss: float, local_gradient: np.ndarray, local_samples: int) -> Tuple[float, np.ndarray]:
        """使用MPI All-Reduce计算全局梯度"""
        # 收集所有进程的样本数
        all_samples = self.comm.allgather(local_samples)
        total_samples = sum(all_samples)
        
        if total_samples == 0:
            return 0.0, local_gradient
        
        # 加权平均损失和梯度
        weighted_loss = local_loss * local_samples
        weighted_gradient = local_gradient * local_samples
        
        # All-reduce求和
        global_weighted_loss = self.comm.allreduce(weighted_loss, op=MPI.SUM)
        global_weighted_gradient = self.comm.allreduce(weighted_gradient, op=MPI.SUM)
        
        # 计算全局平均值
        global_loss = global_weighted_loss / total_samples
        global_gradient = global_weighted_gradient / total_samples
        
        return global_loss, global_gradient
    
    def _update_parameters(self, gradient: np.ndarray):
        """更新网络参数"""
        current_params = self.network.get_parameters()
        new_params = current_params - self.learning_rate * gradient
        self.network.set_parameters(new_params)
    
    def _check_convergence(self, current_loss: float) -> bool:
        """检查是否收敛"""
        if len(self.history['loss']) == 0:
            self.best_loss = current_loss
            return False
        
        # 检查损失是否改善
        if current_loss < self.best_loss - self.tolerance:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 早停检查
        if self.patience_counter >= self.patience:
            if self.rank == 0:
                print(f"早停: 损失在 {self.patience} 个epoch内没有显著改善")
            return True
        
        return False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        训练神经网络
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            
        Returns:
            训练结果字典
        """
        if self.rank == 0:
            print(f"开始训练: {self.size} 个进程, 激活函数={self.activation}, "
                  f"批次大小={self.batch_size}, 隐藏层={self.hidden_dim}")
            print(f"本地训练样本数: {X_train.shape[0] if len(X_train.shape) > 0 else 0}")
        
        start_time = time.time()
        
        # 广播初始参数确保所有进程一致
        if self.rank == 0:
            initial_params = self.network.get_parameters()
        else:
            initial_params = None
        
        initial_params = self.comm.bcast(initial_params, root=0)
        self.network.set_parameters(initial_params)
        
        # 训练循环
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # 计算本地梯度
            local_loss, local_gradient = self._compute_local_gradient(X_train, y_train, epoch)
            local_samples = X_train.shape[0] if len(X_train.shape) > 0 else 0
            
            # 全局梯度聚合
            global_loss, global_gradient = self._all_reduce_gradient(
                local_loss, local_gradient, local_samples
            )
            
            # 更新参数
            self._update_parameters(global_gradient)
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history['loss'].append(global_loss)
            self.history['epoch_times'].append(epoch_time)
            
            # 打印进度
            if self.rank == 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, "
                      f"Loss: {global_loss:.6f}, "
                      f"Time: {epoch_time:.3f}s")
            
            # 检查收敛
            if self._check_convergence(global_loss):
                self.converged = True
                break
        
        total_time = time.time() - start_time
        
        if self.rank == 0:
            print(f"训练完成! 总时间: {total_time:.2f}s, "
                  f"最终损失: {self.history['loss'][-1]:.6f}")
            if self.converged:
                print(f"在第 {len(self.history['loss'])} 个epoch收敛")
        
        return {
            'converged': self.converged,
            'final_loss': self.history['loss'][-1],
            'total_time': total_time,
            'epochs': len(self.history['loss']),
            'history': self.history.copy()
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估结果字典
        """
        local_samples = X_test.shape[0] if len(X_test.shape) > 0 else 0
        
        if local_samples > 0:
            # 计算本地RMSE和损失
            predictions = self.network.predict(X_test)
            local_mse = np.mean((predictions - y_test) ** 2)
            local_rmse = np.sqrt(local_mse)
            local_loss = 0.5 * local_mse
        else:
            local_mse = 0.0
            local_rmse = 0.0
            local_loss = 0.0
        
        # 收集所有进程的结果
        all_samples = self.comm.allgather(local_samples)
        total_samples = sum(all_samples)
        
        if total_samples == 0:
            return {'rmse': 0.0, 'loss': 0.0, 'samples': 0}
        
        # 加权平均
        weighted_mse = local_mse * local_samples
        weighted_loss = local_loss * local_samples
        
        global_weighted_mse = self.comm.allreduce(weighted_mse, op=MPI.SUM)
        global_weighted_loss = self.comm.allreduce(weighted_loss, op=MPI.SUM)
        
        global_mse = global_weighted_mse / total_samples
        global_rmse = np.sqrt(global_mse)
        global_loss = global_weighted_loss / total_samples
        
        return {
            'rmse': global_rmse,
            'loss': global_loss,
            'samples': total_samples
        }
    
    def get_parameters(self) -> np.ndarray:
        """获取当前参数"""
        return self.network.get_parameters()
    
    def set_parameters(self, params: np.ndarray):
        """设置参数"""
        self.network.set_parameters(params)
    
    def get_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return self.history.copy()