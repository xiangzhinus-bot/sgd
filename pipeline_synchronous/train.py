import argparse
import os
import time
import json
import numpy as np
from mpi4py import MPI

from network import NeuralNetwork
from sgd import sync_parameters, train_one_epoch_mmap, evaluate_rmse_mmap, _split_shard

# -------- mmap helpers --------
def infer_rows(path: str, dtype: np.dtype, n_features: int) -> int:
    itemsize = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(path)
    total_items = total_bytes // itemsize
    rows = total_items // n_features
    assert rows * n_features * itemsize == total_bytes, \
        f"File size mismatch for {path}. Check dtype/features."
    return rows

def open_memmaps(prefix: str, n_features: int, dtype: str, ext: str = "mmap"):
    """
    Opens {prefix}_X_train.{ext}, {prefix}_y_train.{ext}, {prefix}_X_test.{ext}, {prefix}_y_test.{ext}
    ext can be 'mmap' or 'mmp' etc. (with or without dot)
    """
    dt = np.dtype(dtype)
    ext = ext.lstrip(".")

    xtr_path = f"{prefix}_X_train.{ext}"
    ytr_path = f"{prefix}_y_train.{ext}"
    xte_path = f"{prefix}_X_test.{ext}"
    yte_path = f"{prefix}_y_test.{ext}"

    n_train = infer_rows(xtr_path, dt, n_features)
    n_test = infer_rows(xte_path, dt, n_features)

    X_train = np.memmap(xtr_path, dtype=dt, mode="r", shape=(n_train, n_features))
    y_train = np.memmap(ytr_path, dtype=dt, mode="r", shape=(n_train,))
    X_test  = np.memmap(xte_path, dtype=dt, mode="r", shape=(n_test, n_features))
    y_test  = np.memmap(yte_path, dtype=dt, mode="r", shape=(n_test,))
    return X_train, y_train, X_test, y_test

def main():
    ap = argparse.ArgumentParser(description="MPI SGD on numpy.memmap files (no standardization).")
    ap.add_argument("--prefix", type=str, default="processed", help="Prefix of mmap files")
    ap.add_argument("--ext", type=str, default="mmap", help="File extension (e.g., mmap, mmp).")
    ap.add_argument("--features", type=int, required=True, help="Number of feature columns")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "leaky_relu"])

    ap.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (per-rank before scaling)")
    ap.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "step", "cosine"])
    ap.add_argument("--lr_scale", type=str, default="sqrt", choices=["none", "linear", "sqrt"])
    ap.add_argument("--warmup", type=int, default=5, help="Warmup epochs")

    ap.add_argument("--lr_step_epochs", type=int, default=10, help="(step schedule) decay every N epochs after warmup")
    ap.add_argument("--lr_gamma", type=float, default=0.5, help="(step schedule) multiply LR by this factor at each step")

    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0, help="Gradient clipping threshold")
    ap.add_argument("--batch", type=int, default=65536, help="Per-rank mini-batch size")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="Early-stop patience (epochs)")
    ap.add_argument("--min_delta", type=float, default=1e-6, help="Min improvement for patience")
    ap.add_argument("--eval_batch", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--momentum", type=float, default=0.9)

    ap.add_argument("--log_file", type=str, default=None, help="If set, save JSON training history here")
    args = ap.parse_args()

    # --- MPI ---
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    if rank == 0:
        print(f"[MPI {size} ranks] Loading data with prefix '{args.prefix}' (ext .{args.ext}) ...")

    Xtr, ytr, Xte, yte = open_memmaps(args.prefix, args.features, args.dtype, args.ext)

    # --- model ---
    net = NeuralNetwork(input_dim=args.features, hidden_dim=args.hidden, activation=args.activation)
    sync_parameters(net, comm)

    # --- shards ---
    tr_lo, tr_hi = _split_shard(Xtr.shape[0], rank, size)
    te_lo, te_hi = _split_shard(Xte.shape[0], rank, size)

    # --- LR scaling ---
    lr_scale_factor = {"none": 1.0, "linear": size, "sqrt": np.sqrt(size)}.get(args.lr_scale, 1.0)
    lr_max = args.lr * lr_scale_factor

    if rank == 0:
        print(f"Global batch size: {args.batch * size}, Max LR: {lr_max:.3g}")
        print("Starting training...")

    best_rmse, best_epoch, patience_ctr = np.inf, -1, 0
    momentum_buffer = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- LR schedule w/ warmup ----
        if args.warmup > 0 and epoch <= args.warmup:
            lr_t = args.lr + (lr_max - args.lr) * (epoch / args.warmup)
        else:
            if args.lr_schedule == "cosine":
                # progress \in [0,1]
                progress = (epoch - max(args.warmup, 1)) / max(1, args.epochs - max(args.warmup, 1))
                lr_t = args.lr + 0.5 * (lr_max - args.lr) * (1 + np.cos(np.pi * progress))
            elif args.lr_schedule == "step":
                steps_since_warm = max(0, epoch - args.warmup)
                decays = steps_since_warm // max(1, args.lr_step_epochs)
                lr_t = lr_max * (args.lr_gamma ** decays)
            else:  # constant
                lr_t = lr_max

        # ---- train one epoch ----
        epoch_loss, momentum_buffer = train_one_epoch_mmap(
            net, Xtr, ytr, tr_lo, tr_hi, batch_size=args.batch, lr=lr_t,
            weight_decay=args.weight_decay, clip_threshold=args.clip,
            momentum=args.momentum, momentum_buffer=momentum_buffer,
            comm=comm, epoch=epoch, base_seed=args.seed
        )

        # ---- evaluate ----
        test_rmse = evaluate_rmse_mmap(net, Xte, yte, te_lo, te_hi, args.eval_batch, comm)

        # ---- early stop (rank 0) ----
        stop_flag = np.array(0, dtype=np.int32)
        if rank == 0:
            dur = time.time() - t0
            print(f"Epoch {epoch:03d}: loss={epoch_loss:.6f}, test_rmse={test_rmse:.6f}, lr={lr_t:.4g}, time={dur:.2f}s")
            history.append({"epoch": epoch, "loss": epoch_loss, "test_rmse": test_rmse, "lr": lr_t, "time": dur})

            if test_rmse + args.min_delta < best_rmse:
                best_rmse, best_epoch, patience_ctr = test_rmse, epoch, 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                stop_flag[...] = 1

        comm.Bcast(stop_flag, root=0)
        if stop_flag.item() == 1:
            break

    if rank == 0:
        print(f"\nTraining finished. Best test RMSE: {best_rmse:.6f} at epoch {best_epoch}")
        if args.log_file:
            with open(args.log_file, "w") as f:
                json.dump({"config": vars(args), "best_rmse": best_rmse, "best_epoch": best_epoch, "history": history}, f, indent=2)
            print(f"Training history saved to {args.log_file}")

if __name__ == "__main__":
    main()
