# train_mmap.py
import argparse, os
import numpy as np
from mpi4py import MPI

from network import NeuralNetwork
from sgd_mpi import sync_parameters, train_one_epoch_mmap, evaluate_rmse_mmap, _split_shard

def infer_rows(path: str, dtype: np.dtype, n_features: int) -> int:
    """Compute #rows from file size (works for contiguous C-order memmap)."""
    itemsize = np.dtype(dtype).itemsize
    total_items = os.path.getsize(path) // itemsize
    rows = total_items // n_features
    assert rows * n_features * itemsize == os.path.getsize(path), "Size mismatch: check dtype/features."
    return rows

def open_memmaps(prefix: str, n_features: int, dtype: str):
    dt = np.dtype(dtype)
    xtr = f"{prefix}_X_train.mmap"
    ytr = f"{prefix}_y_train.mmap"
    xte = f"{prefix}_X_test.mmap"
    yte = f"{prefix}_y_test.mmap"

    n_train = infer_rows(xtr, dt, n_features)
    n_test  = infer_rows(xte, dt, n_features)

    X_train = np.memmap(xtr, dtype=dt, mode="r", shape=(n_train, n_features))
    y_train = np.memmap(ytr, dtype=dt, mode="r", shape=(n_train,))
    X_test  = np.memmap(xte, dtype=dt, mode="r", shape=(n_test,  n_features))
    y_test  = np.memmap(yte, dtype=dt, mode="r", shape=(n_test,))

    return (X_train, y_train, X_test, y_test)

def main():
    parser = argparse.ArgumentParser(description="MPI-SGD on numpy.memmap files")
    parser.add_argument("--prefix", type=str, default="processed", help="Prefix of mmap files")
    parser.add_argument("--features", type=int, required=True, help="Number of feature columns in X")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Dtype used in mmaps")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "sigmoid", "relu", "leaky_relu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping L2 threshold")
    parser.add_argument("--batch", type=int, default=65536, help="Mini-batch size (per rank)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=262144)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    if rank == 0:
        print(f"[MPI {size} ranks] Loading mmaps with prefix='{args.prefix}' ...")

    Xtr, ytr, Xte, yte = open_memmaps(args.prefix, args.features, args.dtype)

    # Build model
    net = NeuralNetwork(input_dim=args.features, hidden_dim=args.hidden, activation=args.activation)

    # Sync initial parameters from rank 0
    sync_parameters(net, comm)

    # Shard ranges
    tr_lo, tr_hi = _split_shard(Xtr.shape[0], rank, size)
    te_lo, te_hi = _split_shard(Xte.shape[0], rank, size)

    best_rmse = np.inf
    best_epoch = -1
    patience_ctr = 0

    if rank == 0:
        print(f"Train rows: {Xtr.shape[0]}, Test rows: {Xte.shape[0]}, Features: {args.features}")
        print(f"Local shard (rank {rank}): train [{tr_lo}, {tr_hi}), test [{te_lo}, {te_hi})")
        print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_loss = train_one_epoch_mmap(
            net, Xtr, ytr, tr_lo, tr_hi, batch_size=args.batch,
            lr=args.lr, weight_decay=args.weight_decay, clip_threshold=args.clip, comm=comm
        )

        # Evaluate (scaled domain)
        rmse = evaluate_rmse_mmap(net, Xte, yte, te_lo, te_hi, eval_batch=args.eval_batch, comm=comm)

        # Rank 0 prints & handles early stopping
        stop_flag = np.array(0, dtype=np.int32)
        if rank == 0:
            print(f"Epoch {epoch:03d}: loss~{epoch_loss:.6f}, rmse(test, scaled)={rmse:.6f}")
            if rmse + 1e-9 < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} (best rmse={best_rmse:.6f} @ {best_epoch})")
                stop_flag[...] = 1

        # Broadcast stop decision
        comm.Bcast(stop_flag, root=0)
        if stop_flag.item() == 1:
            break

        # Keep parameters synchronized implicitly (they are updated identically on all ranks).
        # If you add rank-specific noise later, add an explicit sync_parameters() here.

    if rank == 0:
        print(f"Training finished. Best test RMSE (scaled): {best_rmse:.6f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
