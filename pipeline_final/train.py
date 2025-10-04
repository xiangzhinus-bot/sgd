# train.py
import argparse
import os
import time
import json
import math
import numpy as np
from mpi4py import MPI

from network import NeuralNetwork
from sgd import sync_parameters, train_one_epoch_mmap, evaluate_rmse_mmap, _split_shard

def infer_rows(path: str, dtype: np.dtype, n_features: int) -> int:
    itemsize = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(path)
    rows = total_bytes // (itemsize * n_features)
    assert rows * n_features * itemsize == total_bytes, "File size mismatch"
    return rows

def open_memmaps(prefix: str, ext: str, n_features: int, dtype: str):
    dt = np.dtype(dtype)
    xtr = f"{prefix}_X_train.{ext}"
    ytr = f"{prefix}_y_train.{ext}"
    xte = f"{prefix}_X_test.{ext}"
    yte = f"{prefix}_y_test.{ext}"

    n_train = infer_rows(xtr, dt, n_features)
    n_test  = infer_rows(xte, dt, n_features)

    X_train = np.memmap(xtr, dtype=dt, mode="r", shape=(n_train, n_features))
    y_train = np.memmap(ytr, dtype=dt, mode="r", shape=(n_train,))
    X_test  = np.memmap(xte, dtype=dt, mode="r", shape=(n_test, n_features))
    y_test  = np.memmap(yte, dtype=dt, mode="r", shape=(n_test,))
    return X_train, y_train, X_test, y_test

def _rand_affine_keys(N: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    while True:
        a = int(rng.integers(1, N))
        if math.gcd(a, N) == 1:
            break
    b = int(rng.integers(0, N))
    return a, b

def main():
    ap = argparse.ArgumentParser(description="MPI SGD (memmap + PRP superbatch)")
    ap.add_argument("--prefix", type=str, default="processed")
    ap.add_argument("--ext", type=str, default="mmap")
    ap.add_argument("--features", type=int, required=True)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh","relu","leaky_relu"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant","step","cosine"])
    ap.add_argument("--lr_scale", type=str, default="sqrt", choices=["none","linear","sqrt"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr_step_epochs", type=int, default=10)
    ap.add_argument("--lr_gamma", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=1e-6)
    ap.add_argument("--eval_batch", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--q_super", type=int, default=8, help="Q = q_super * batch")
    ap.add_argument("--prefetch", type=int, default=2, help="async prefetch queue size (0=off)")
    # debug & 曲线粒度
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_first_batches", type=int, default=1)
    ap.add_argument("--history_granularity", type=str, default="epoch", choices=["epoch","step"],
                    help="记录训练历史的粒度：epoch 或 step（每次梯度更新）")
    ap.add_argument("--log_every_steps", type=int, default=1, help="step 粒度下，每隔多少更新记录一次")
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    if rank == 0:
        print(f"[MPI {size} ranks] Loading data with prefix '{args.prefix}' (ext .{args.ext}) ...")

    Xtr, ytr, Xte, yte = open_memmaps(args.prefix, args.ext, args.features, args.dtype)
    n_train = Xtr.shape[0]
    n_test  = Xte.shape[0]

    net = NeuralNetwork(input_dim=args.features, hidden_dim=args.hidden, activation=args.activation, dtype=np.float32)
    sync_parameters(net, comm)

    tr_lo, tr_hi = _split_shard(n_train, rank, size)
    te_lo, te_hi = _split_shard(n_test, rank, size)

    lr_scale_factor = {"none": 1.0, "linear": size, "sqrt": np.sqrt(size)}[args.lr_scale]
    lr_max = args.lr * lr_scale_factor

    if rank == 0:
        print(f"Global batch size: {args.batch * size}, Max LR: {lr_max:.3g}")
        print("Starting training...")

    best_rmse, best_epoch, patience_ctr = np.inf, -1, 0
    momentum_buffer = None
    history = []          # epoch 粒度
    step_history = []     # step 粒度
    global_step = 0
    prev_theta = net.get_parameters().copy() if rank == 0 else None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if rank == 0:
            a, b = _rand_affine_keys(n_train, args.seed + epoch)
            if args.debug:
                sample = ((a * np.arange(5, dtype=np.int64) + b) % n_train).tolist()
                print(f"[debug] epoch={epoch} PRP keys: a={a} b={b}, first5={sample}")
        else:
            a = b = 0
        a = comm.bcast(a, root=0)
        b = comm.bcast(b, root=0)

        if epoch <= args.warmup and args.warmup > 0:
            lr_t = args.lr + (lr_max - args.lr) * (epoch / args.warmup)
        else:
            if args.lr_schedule == "cosine":
                progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
                lr_t = args.lr + 0.5 * (lr_max - args.lr) * (1 + np.cos(np.pi * progress))
            elif args.lr_schedule == "step":
                steps = (epoch - 1) // max(1, args.lr_step_epochs)
                lr_t = lr_max * (args.lr_gamma ** steps)
            else:
                lr_t = lr_max

        ep_loss, momentum_buffer, stats, global_step, step_logs = train_one_epoch_mmap(
            net, Xtr, ytr,
            n_train, a, b,
            tr_lo, tr_hi,
            batch_size=args.batch, lr=lr_t,
            weight_decay=args.weight_decay, clip_threshold=args.clip,
            momentum=args.momentum, momentum_buffer=momentum_buffer,
            comm=comm, prefetch=args.prefetch, q_super=args.q_super,
            debug=args.debug, debug_first_batches=args.debug_first_batches,
            history_granularity=args.history_granularity, log_every_steps=args.log_every_steps,
            start_step=global_step
        )
        if rank == 0 and step_logs:
            step_history.extend(step_logs)

        test_rmse = evaluate_rmse_mmap(net, Xte, yte, te_lo, te_hi, args.eval_batch, comm)
        ep_time = time.time() - t0

        io_sum   = comm.reduce(stats["io_time"], op=MPI.SUM, root=0)
        comp_sum = comm.reduce(stats["compute_time"], op=MPI.SUM, root=0)
        com_sum  = comm.reduce(stats["comm_time"], op=MPI.SUM, root=0)
        grad0    = stats["grad_norm_first"]
        grad0    = comm.bcast(grad0, root=0)

        stop = np.array(0, dtype=np.int32)
        if rank == 0:
            theta = net.get_parameters()
            delta = float(np.linalg.norm(theta - prev_theta))
            prev_theta = theta.copy()
            if args.debug:
                print(f"[debug] ||Δθ|| (epoch {epoch}) = {delta:.6e}")

            print(f"Epoch {epoch:03d}: loss={ep_loss:.6f}, test_rmse={test_rmse:.6f}, "
                  f"lr={lr_t:.4g}, time={ep_time:.2f}s | io={io_sum:.2f}s, compute={comp_sum:.2f}s, comm={com_sum:.2f}s, "
                  f"grad0={grad0:.3e}")

            history.append({
                'epoch': epoch, 'loss': ep_loss, 'test_rmse': test_rmse, 'lr': lr_t, 'time': ep_time,
                'io_time_sum': io_sum, 'compute_time_sum': comp_sum, 'comm_time_sum': com_sum,
                'grad_norm_first': grad0, 'q_super': args.q_super, 'prefetch': args.prefetch, 'world_size': size
            })
            if test_rmse + args.min_delta < best_rmse:
                best_rmse, best_epoch, patience_ctr = test_rmse, epoch, 0
            else:
                patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                stop[...] = 1
        comm.Bcast(stop, root=0)
        if stop.item() == 1:
            break

    if rank == 0:
        print(f"\nTraining finished. Best test RMSE: {best_rmse:.6f} at epoch {best_epoch}")
        if args.log_file:
            os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
            with open(args.log_file, 'w') as f:
                json.dump({
                    'config': vars(args),
                    'best_rmse': best_rmse,
                    'best_epoch': best_epoch,
                    'history': history,            # epoch 粒度
                    'step_history': step_history   # step 粒度（R(theta_k) vs k）
                }, f, indent=2)
            print(f"Training history saved to {args.log_file}")

if __name__ == "__main__":
    main()
