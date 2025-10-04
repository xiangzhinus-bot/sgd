# sgd.py
import time, math
import numpy as np
from mpi4py import MPI
from typing import Tuple, Optional, Iterable
from threading import Thread
from queue import Queue
from network import NeuralNetwork

# ---------- data shard ----------
def _split_shard(total: int, rank: int, size: int) -> Tuple[int, int]:
    per = total // size
    rem = total % size
    lo = rank * per + min(rank, rem)
    hi = lo + per + (1 if rank < rem else 0)
    return lo, hi

# ---------- affine perm ----------
def _permute_affine(base_idx: np.ndarray, N: int, a: int, b: int) -> np.ndarray:
    return ((a * base_idx.astype(np.int64)) + b) % np.int64(N)

# ---------- superbatch iterator ----------
def _superbatch_batches_via_prp(
    X_mm: np.memmap, y_mm: np.memmap,
    N: int, a: int, b: int,
    logical_lo: int, logical_hi: int,
    batch_size: int, q_super: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    Q = q_super * batch_size
    d = X_mm.shape[1]

    for p in range(logical_lo, logical_hi, Q):
        L = min(Q, logical_hi - p)
        if L <= 0:
            break

        logical = np.arange(p, p + L, dtype=np.int64)
        ids = _permute_affine(logical, N, a, b)

        sids = np.sort(ids)
        cut = np.where(np.diff(sids) != 1)[0] + 1
        runs = np.split(sids, cut)

        buf_X = np.empty((L, d), dtype=X_mm.dtype)
        buf_y = np.empty((L,), dtype=y_mm.dtype)
        pos = 0
        for r in runs:
            lo = int(r[0]); hi = int(r[-1]) + 1
            blkX = np.asarray(X_mm[lo:hi])
            blky = np.asarray(y_mm[lo:hi])
            n = hi - lo
            buf_X[pos:pos+n] = blkX
            buf_y[pos:pos+n] = blky
            pos += n

        order = np.argsort(ids)
        inv = np.empty_like(order)
        inv[order] = np.arange(L)
        Xrand = buf_X[inv]
        yrand = buf_y[inv]

        for i in range(0, L, batch_size):
            yield Xrand[i:i+batch_size], yrand[i:i+batch_size]

# ---------- async prefetch ----------
def _prefetch_iter(batches_iter: Iterable[Tuple[np.ndarray, np.ndarray]], prefetch: int):
    if prefetch <= 0:
        for item in batches_iter:
            yield item
        return
    q: Queue = Queue(maxsize=prefetch)
    STOP = object()
    def worker():
        try:
            for item in batches_iter:
                q.put(item)
        finally:
            q.put(STOP)
    Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is STOP:
            break
        yield item

# ---------- sync ----------
def sync_parameters(net: NeuralNetwork, comm: MPI.Comm):
    theta = net.get_parameters() if comm.rank == 0 else None
    theta = comm.bcast(theta, root=0)
    net.set_parameters(theta)

# ---------- update ----------
def update_with_clipping_momentum(
    net: NeuralNetwork, grad: np.ndarray, lr: float,
    clip_threshold: float, momentum: float = 0.0,
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

# ---------- one epoch (timed + step logs) ----------
def train_one_epoch_mmap(
    net: NeuralNetwork, X_mm: np.memmap, y_mm: np.memmap,
    N: int, a: int, b: int,
    logical_lo: int, logical_hi: int,
    batch_size: int, lr: float, weight_decay: float, clip_threshold: float,
    momentum: float, momentum_buffer: Optional[np.ndarray], comm: MPI.Comm,
    prefetch: int = 2, q_super: int = 8,
    debug: bool = False, debug_first_batches: int = 1,
    history_granularity: str = "epoch", log_every_steps: int = 1, start_step: int = 0
):
    """
    返回:
      avg_epoch_loss, momentum_buffer, stats(dict), end_global_step, step_logs(list)
    step_logs（仅 rank0/step 粒度）： [{'k': global_step, 'train_rmse': ..., 'lr': lr}, ...]
    """
    rank = comm.rank
    param_dim = len(net.get_parameters())
    global_loss_sum_total = 0.0
    global_count_total = 0

    io_time = 0.0
    compute_time = 0.0
    comm_time = 0.0
    grad_norm_first = None

    batches = _superbatch_batches_via_prp(
        X_mm, y_mm, N, a, b, logical_lo, logical_hi, batch_size, q_super
    )
    batches = _prefetch_iter(batches, prefetch=prefetch)
    it = iter(batches)

    step_logs = []
    global_step = int(start_step)
    printed = False
    printed_dead = 0

    while True:
        t_io0 = time.time()
        try:
            Xb, yb = next(it)
        except StopIteration:
            break
        io_time += (time.time() - t_io0)

        if debug and printed_dead < debug_first_batches and net.activation_name == "relu":
            dead_frac = float(((Xb @ net.W1.T + net.b1) <= 0).mean())
            if rank == 0:
                print(f"[debug] dead_frac(batch{printed_dead})={dead_frac:.3f}")
            printed_dead += 1

        t_comp0 = time.time()
        loss, grad = net.compute_loss_and_gradient(Xb, yb, weight_decay=weight_decay)
        B = Xb.shape[0]
        compute_time += (time.time() - t_comp0)

        # step 级训练 RMSE（基于刚才的前向）
        sse_local = float(np.sum((net.yhat.ravel() - yb) ** 2))
        cnt_local = float(B)
        sse_buf = np.array([sse_local, cnt_local], dtype=np.float64)
        sse_glob = np.empty_like(sse_buf)
        t_comm0 = time.time()
        comm.Allreduce(sse_buf, sse_glob, op=MPI.SUM)
        comm_time += (time.time() - t_comm0)
        train_rmse_step = math.sqrt(sse_glob[0] / max(1.0, sse_glob[1]))

        # grad/loss allreduce
        send = np.empty(param_dim + 2, dtype=np.float64)
        send[:param_dim] = grad * B
        send[-2] = loss * B
        send[-1] = float(B)

        t_comm1 = time.time()
        recv = np.empty_like(send)
        comm.Allreduce(send, recv, op=MPI.SUM)
        comm_time += (time.time() - t_comm1)

        global_grad_sum = recv[:param_dim]
        global_loss_sum = recv[-2]
        global_count = int(recv[-1])
        global_grad_avg = global_grad_sum / global_count

        if debug and not printed and rank == 0:
            grad_norm_first = float(np.linalg.norm(global_grad_avg))
            print(f"[debug] grad_norm_first_batch={grad_norm_first:.6e}")
            printed = True

        momentum_buffer = update_with_clipping_momentum(
            net, global_grad_avg, lr, clip_threshold, momentum, momentum_buffer
        )

        global_loss_sum_total += global_loss_sum
        global_count_total += global_count

        global_step += 1
        if history_granularity == "step" and (global_step % max(1, log_every_steps) == 0):
            if rank == 0:
                step_logs.append({'k': int(global_step), 'train_rmse': float(train_rmse_step), 'lr': float(lr)})

    avg_epoch_loss = global_loss_sum_total / max(1, global_count_total)
    stats = {
        "io_time": io_time,
        "compute_time": compute_time,
        "comm_time": comm_time,
        "grad_norm_first": float(grad_norm_first) if grad_norm_first is not None else float("nan"),
    }
    return float(avg_epoch_loss), momentum_buffer, stats, global_step, step_logs

# ---------- eval ----------
def evaluate_rmse_mmap(
    net: NeuralNetwork, X_mm: np.memmap, y_mm: np.memmap,
    lo: int, hi: int, eval_batch: int, comm: MPI.Comm
) -> float:
    sse_local = 0.0
    n_local = 0
    for s in range(lo, hi, eval_batch):
        e = min(s + eval_batch, hi)
        Xb = np.asarray(X_mm[s:e])
        yb = np.asarray(y_mm[s:e])
        pred = net.predict(Xb)
        sse_local += np.sum((pred - yb) ** 2)
        n_local += (e - s)
    send = np.array([sse_local, n_local], dtype=np.float64)
    recv = np.empty_like(send)
    comm.Allreduce(send, recv, op=MPI.SUM)
    sse_g, n_g = recv[0], int(recv[1])
    return float(np.sqrt(sse_g / n_g) if n_g > 0 else np.nan)
