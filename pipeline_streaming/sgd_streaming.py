# sgd_streaming.py (Corrected)

import numpy as np
from mpi4py import MPI
from typing import Tuple, Dict, List
import time
from network import NeuralNetwork

class MPISGDTrainer:
    # ... (init and other methods are the same) ...
    def __init__(self, input_dim: int, hidden_dim: int, activation: str, learning_rate: float, max_epochs: int, tolerance: float, patience: int):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.input_dim, self.hidden_dim, self.activation, self.learning_rate = input_dim, hidden_dim, activation, learning_rate
        self.max_epochs, self.tolerance, self.patience = max_epochs, tolerance, patience
        self.network = NeuralNetwork(input_dim, hidden_dim, activation)
        self.history = {'loss': [], 'epoch_times': []}
        self.converged, self.best_loss, self.patience_counter = False, float('inf'), 0

    def sync_parameters(self):
        params = self.network.get_parameters() if self.rank == 0 else None
        self.network.set_parameters(self.comm.bcast(params, root=0))

    def run_epoch(self, line_iterator, process_and_split_func, preprocessor, header) -> Tuple[np.ndarray, float, int]:
        param_size = len(self.network.get_parameters())
        local_gradient_sum = np.zeros(param_size, dtype=np.float64)
        local_loss_sum, local_samples_count = 0.0, 0
        rows_buf, BATCH_SIZE_LINES = [], 50000

        for i, line in enumerate(line_iterator, 1):
            rows_buf.append(line)
            if i % BATCH_SIZE_LINES == 0:
                X_train_b, y_train_b, _, _ = process_and_split_func(rows_buf, preprocessor, header) # Pass header
                if X_train_b.shape[0] > 0:
                    loss, grad = self.network.compute_loss_and_gradient(X_train_b, y_train_b)
                    local_gradient_sum += grad * X_train_b.shape[0]
                    local_loss_sum += loss * X_train_b.shape[0]
                    local_samples_count += X_train_b.shape[0]
                rows_buf.clear()

        if rows_buf:
            X_train_b, y_train_b, _, _ = process_and_split_func(rows_buf, preprocessor, header) # Pass header
            if X_train_b.shape[0] > 0:
                loss, grad = self.network.compute_loss_and_gradient(X_train_b, y_train_b)
                local_gradient_sum += grad * X_train_b.shape[0]
                local_loss_sum += loss * X_train_b.shape[0]
                local_samples_count += X_train_b.shape[0]
        
        return local_gradient_sum, local_loss_sum, local_samples_count

    def update_parameters(self, global_gradient: np.ndarray):
        current_params = self.network.get_parameters()
        self.network.set_parameters(current_params - self.learning_rate * global_gradient)

    def check_convergence(self, current_loss: float) -> bool:
        if self.rank != 0: return False
        self.history['loss'].append(current_loss)
        if current_loss < self.best_loss - self.tolerance:
            self.best_loss, self.patience_counter = current_loss, 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience:
            print(f"Early stopping after {self.patience} epochs with no improvement.")
            self.converged = True
            return True
        return False

    def evaluate(self, line_iterator, process_and_split_func, preprocessor, header) -> Dict[str, float]:
        local_sse_scaled, local_sse_unscaled, local_samples = 0.0, 0.0, 0
        rows_buf, BATCH_SIZE_LINES = [], 50000

        for i, line in enumerate(line_iterator, 1):
            rows_buf.append(line)
            if i % BATCH_SIZE_LINES == 0:
                _, _, X_test_b, y_test_b = process_and_split_func(rows_buf, preprocessor, header) # Pass header
                if X_test_b.shape[0] > 0:
                    pred_s = self.network.predict(X_test_b)
                    local_sse_scaled += np.sum((pred_s - y_test_b)**2)
                    pred_u = preprocessor.inverse_transform_target(pred_s)
                    y_u = preprocessor.inverse_transform_target(y_test_b)
                    local_sse_unscaled += np.sum((pred_u - y_u)**2)
                    local_samples += X_test_b.shape[0]
                rows_buf.clear()
        
        if rows_buf:
            _, _, X_test_b, y_test_b = process_and_split_func(rows_buf, preprocessor, header) # Pass header
            if X_test_b.shape[0] > 0:
                pred_s = self.network.predict(X_test_b)
                local_sse_scaled += np.sum((pred_s - y_test_b)**2)
                pred_u = preprocessor.inverse_transform_target(pred_s)
                y_u = preprocessor.inverse_transform_target(y_test_b)
                local_sse_unscaled += np.sum((pred_u - y_u)**2)
                local_samples += X_test_b.shape[0]

        send_buf = np.array([local_sse_scaled, local_sse_unscaled, local_samples], dtype=np.float64)
        recv_buf = np.empty_like(send_buf)
        self.comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
        global_sse_s, global_sse_u, total_samples = recv_buf
        
        rmse_s = np.sqrt(global_sse_s / total_samples) if total_samples > 0 else 0.0
        rmse_u = np.sqrt(global_sse_u / total_samples) if total_samples > 0 else 0.0
        
        return {'rmse_scaled': rmse_s, 'rmse_unscaled': rmse_u}