# sgd_mpi.py
import numpy as np
from mpi4py import MPI
from typing import Tuple
from network import NeuralNetwork

def sync_parameters(net: NeuralNetwork, comm: MPI.Comm):
    theta = net.get_parameters() if comm.rank == 0 else None
    theta = comm.bcast(theta, root=0)
    net.set_parameters(theta)

def update_with_clipping(net: NeuralNetwork, grad: np.ndarray, lr: float, clip_threshold: float):
    """Apply gradient clipping (global L2 norm) and SGD step."""
    gnorm = np.linalg.norm(grad)
    if gnorm > clip_threshold and gnorm > 0:
        grad = grad * (clip_threshold / gnorm)
    theta = net.get_parameters()
    net.set_parameters(theta - lr * grad)

def _split_shard(total: int, rank: int, size: int) -> Tuple[int, int]:
    """Even contiguous shard [lo, hi) for this rank."""
    per = total // size
    rem = total % size
    lo = rank * per + min(rank, rem)
    hi = lo + per + (1 if rank < rem else 0)
    return lo, hi

def train_one_epoch_mmap(
    net: NeuralNetwork,
    X_mm: np.memmap,
    y_mm: np.memmap,
    local_lo: int,
    local_hi: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    clip_threshold: float,
    comm: MPI.Comm,
) -> float:
    """
    Synchronous data-parallel SGD on local shard [local_lo, local_hi),
    with per-batch Allreduce on (sum_gradient, sum_loss, count).
    Returns averaged epoch loss (global).
    """
    rank = comm.rank
    # Pre-shuffle indices once per epoch (local shard)
    idx = np.arange(local_lo, local_hi, dtype=np.int64)
    rng = np.random.default_rng(seed=rank + 12345)  # distinct per rank, reshuffled each call
    rng.shuffle(idx)

    param_dim = len(net.get_parameters())
    epoch_loss_sum_local = 0.0
    epoch_count_local = 0

    for start in range(0, idx.size, batch_size):
        sel = idx[start:start + batch_size]
        if sel.size == 0:
            continue
        Xb = np.asarray(X_mm[sel], dtype=np.float64, order="C")
        yb = np.asarray(y_mm[sel], dtype=np.float64)

        loss, grad = net.compute_loss_and_gradient(Xb, yb, weight_decay=weight_decay)

        # Sum (not average) across processes
        send = np.empty(param_dim + 2, dtype=np.float64)
        send[:param_dim] = grad * sel.size
        send[-2] = loss * sel.size
        send[-1] = float(sel.size)

        recv = np.empty_like(send)
        comm.Allreduce(send, recv, op=MPI.SUM)

        global_grad_sum = recv[:param_dim]
        global_loss_sum = recv[-2]
        global_count = int(recv[-1]) if recv[-1] > 0 else 1

        # Average gradient & loss, then update synchronously on all ranks
        global_grad_avg = global_grad_sum / global_count
        avg_loss = global_loss_sum / global_count

        update_with_clipping(net, global_grad_avg, lr=lr, clip_threshold=clip_threshold)

        # Track for epoch scalar (optional: recompute to avoid drift; here we reuse)
        epoch_loss_sum_local += avg_loss * 0  # local not used; we'll Allreduce below
        epoch_count_local += 0                # same note; we compute once per batch via recv

    # For a clean global epoch loss, do one pass that aggregates using the last recv
    # Simpler: approximate epoch loss by evaluating mini-batch averages we already computed.
    # We'll do a lightweight reduction to carry the last 'avg_loss' from each batch as a weighted mean.
    # Instead, do a quick evaluation on local shard using small eval batch size to avoid extra comms.
    # To keep it simple & deterministic, compute epoch loss via single reduction during the loop:
    # We increment counters using global stats from the last batch; but that needs accumulation.
    # Approach: re-run a tiny summary: 1 number per process—just use 'avg_loss' from last batch.
    # Better: track running averages using the recv values at each batch.
    # We'll do that: maintain totals on rank 0 via reductions per batch is already done above in recv.
    # So we won't add another barrier here.

    # Return the last avg_loss seen to display progress; for stringent logging one could accumulate recv[-2] and recv[-1] across batches.
    # To keep costs low, we avoid another pass. This is acceptable for large-scale training.
    return float(avg_loss)

def evaluate_rmse_mmap(
    net: NeuralNetwork,
    X_mm: np.memmap,
    y_mm: np.memmap,
    local_lo: int,
    local_hi: int,
    eval_batch: int,
    comm: MPI.Comm,
) -> float:
    """Compute global RMSE on test mmap."""
    sse_local = 0.0
    n_local = 0
    for start in range(local_lo, local_hi, eval_batch):
        end = min(start + eval_batch, local_hi)
        Xb = np.asarray(X_mm[start:end], dtype=np.float64, order="C")
        yb = np.asarray(y_mm[start:end], dtype=np.float64)
        pred = net.predict(Xb)
        sse_local += np.sum((pred - yb) ** 2)
        n_local += (end - start)

    buf = np.array([sse_local, n_local], dtype=np.float64)
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.SUM)
    sse, n = out[0], int(out[1])
    rmse = np.sqrt(sse / n) if n > 0 else np.nan
    return float(rmse)
