import numpy as np
from mpi4py import MPI
from typing import Tuple, Optional
from network import NeuralNetwork

# -------- MPI: synchronize model params --------
def sync_parameters(net: NeuralNetwork, comm: MPI.Comm):
    theta = net.get_parameters() if comm.rank == 0 else None
    theta = comm.bcast(theta, root=0)
    net.set_parameters(theta)

# -------- shard helpers --------
def _split_shard(total: int, rank: int, size: int) -> Tuple[int, int]:
    per = total // size
    rem = total % size
    lo = rank * per + min(rank, rem)
    hi = lo + per + (1 if rank < rem else 0)
    return lo, hi

def _block_order(local_lo: int, local_hi: int, batch_size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    blocks = []
    for s in range(local_lo, local_hi, batch_size):
        e = min(s + batch_size, local_hi)
        blocks.append((s, e))
    rng.shuffle(blocks)
    return blocks

# -------- optimizer step (clip + momentum) --------
def update_with_clipping_momentum(
    net: NeuralNetwork,
    grad: np.ndarray,
    lr: float,
    clip_threshold: float,
    momentum: float = 0.0,
    momentum_buffer: Optional[np.ndarray] = None
) -> np.ndarray:
    gnorm = np.linalg.norm(grad)
    if gnorm > clip_threshold and gnorm > 0:
        grad = grad * (clip_threshold / gnorm)

    update = grad
    if momentum > 0:
        if momentum_buffer is None:
            momentum_buffer = np.zeros_like(grad)
        momentum_buffer = momentum * momentum_buffer + grad
        update = momentum_buffer

    theta = net.get_parameters()
    net.set_parameters(theta - lr * update)
    return momentum_buffer

# -------- 1 epoch train (mmap) --------
def train_one_epoch_mmap(
    net: NeuralNetwork, X_mm: np.memmap, y_mm: np.memmap,
    local_lo: int, local_hi: int, batch_size: int, lr: float,
    weight_decay: float, clip_threshold: float, momentum: float,
    momentum_buffer: Optional[np.ndarray], comm: MPI.Comm,
    epoch: int, base_seed: int
) -> Tuple[float, Optional[np.ndarray]]:
    rank = comm.rank
    rng = np.random.default_rng(seed=base_seed + epoch * 100 + rank)
    param_dim = len(net.get_parameters())

    global_loss_sum_total = 0.0
    global_count_total = 0

    for (s, e) in _block_order(local_lo, local_hi, batch_size, rng):
        Xb = np.asarray(X_mm[s:e])
        yb = np.asarray(y_mm[s:e])

        loss, grad = net.compute_loss_and_gradient(Xb, yb, weight_decay=weight_decay)

        B = e - s
        send_buffer = np.empty(param_dim + 2, dtype=np.float64)
        send_buffer[:param_dim] = grad * B     # scale by batch for correct averaging
        send_buffer[-2] = loss * B
        send_buffer[-1] = float(B)

        recv_buffer = np.empty_like(send_buffer)
        comm.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        global_grad_sum = recv_buffer[:param_dim]
        global_loss_sum = recv_buffer[-2]
        global_count = int(recv_buffer[-1])

        global_grad_avg = global_grad_sum / global_count

        momentum_buffer = update_with_clipping_momentum(
            net, global_grad_avg, lr, clip_threshold, momentum, momentum_buffer
        )

        global_loss_sum_total += global_loss_sum
        global_count_total += global_count

    avg_epoch_loss = global_loss_sum_total / global_count_total
    return float(avg_epoch_loss), momentum_buffer

# -------- evaluation (RMSE) --------
def evaluate_rmse_mmap(
    net: NeuralNetwork, X_mm: np.memmap, y_mm: np.memmap,
    local_lo: int, local_hi: int, eval_batch: int, comm: MPI.Comm
) -> float:
    sse_local = 0.0
    n_local = 0

    for start in range(local_lo, local_hi, eval_batch):
        end = min(start + eval_batch, local_hi)
        Xb = np.asarray(X_mm[start:end])
        yb = np.asarray(y_mm[start:end])

        pred = net.predict(Xb)
        sse_local += np.sum((pred - yb) ** 2)
        n_local += (end - start)

    send_buffer = np.array([sse_local, n_local], dtype=np.float64)
    recv_buffer = np.empty_like(send_buffer)
    comm.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
    sse_global, n_global = recv_buffer[0], int(recv_buffer[1])
    return float(np.sqrt(sse_global / n_global) if n_global > 0 else np.nan)
