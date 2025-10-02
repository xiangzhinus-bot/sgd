# train2.py (版本：保存历史数据，不绘图)
import argparse
import os
import time
import numpy as np
from mpi4py import MPI
import csv # 1. 引入csv库用于保存文件

# 从您的项目中导入
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
    # ... 其他所有参数 ...
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "sigmoid", "relu", "leaky_relu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=262144)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    if rank == 0:
        print("=================================================")
        print(f"Starting MPI Training with {size} processes.")
        print("Chosen Parameters:")
        for arg, value in vars(args).items():
            print(f"  --{arg}: {value}")
        print("-------------------------------------------------")

    Xtr, ytr, Xte, yte = open_memmaps(args.prefix, args.features, args.dtype)

    net = NeuralNetwork(input_dim=args.features, hidden_dim=args.hidden, activation=args.activation)
    sync_parameters(net, comm)

    tr_lo, tr_hi = _split_shard(Xtr.shape[0], rank, size)
    te_lo, te_hi = _split_shard(Xte.shape[0], rank, size)

    best_rmse = np.inf
    best_epoch = -1
    patience_ctr = 0
    
    history = {
        'epoch': [],
        'loss': [],
        'test_rmse': []
    }

    if rank == 0:
        print("Starting training...")
        start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch_mmap(
            net, Xtr, ytr, tr_lo, tr_hi, batch_size=args.batch,
            lr=args.lr, weight_decay=args.weight_decay, clip_threshold=args.clip, comm=comm
        )
        test_rmse = evaluate_rmse_mmap(net, Xte, yte, te_lo, te_hi, eval_batch=args.eval_batch, comm=comm)

        stop_flag = np.array(0, dtype=np.int32)
        if rank == 0:
            print(f"Epoch {epoch:03d}: loss~{epoch_loss:.6f}, rmse(test, scaled)={test_rmse:.6f}")
            history['epoch'].append(epoch)
            history['loss'].append(epoch_loss)
            history['test_rmse'].append(test_rmse)
            if test_rmse + 1e-9 < best_rmse:
                best_rmse = test_rmse
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                stop_flag[...] = 1
        comm.Bcast(stop_flag, root=0)
        if stop_flag.item() == 1:
            break

    final_train_rmse = evaluate_rmse_mmap(net, Xtr, ytr, tr_lo, tr_hi, eval_batch=args.eval_batch, comm=comm)
    final_test_rmse = evaluate_rmse_mmap(net, Xte, yte, te_lo, te_hi, eval_batch=args.eval_batch, comm=comm)

    if rank == 0:
        end_time = time.time()
        training_time = end_time - start_time
        print("-------------------------------------------------")
        print("Training finished.")
        print(f"Total training time with {size} processes: {training_time:.2f} seconds.")
        print(f"Final RMSE on Training Data: {final_train_rmse:.6f}")
        print(f"Final RMSE on Test Data: {final_test_rmse:.6f}")
        print(f"Best test RMSE during training: {best_rmse:.6f} at epoch {best_epoch}")
        
        # --- 主要修改点 ---
        # 2. 移除所有matplotlib绘图代码，替换为保存CSV文件的代码
        if history['epoch']:
            filename = f'training_history_ranks_{size}.csv'
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # 写入表头
                    writer.writerow(['epoch', 'loss', 'test_rmse'])
                    # 写入数据
                    for i in range(len(history['epoch'])):
                        writer.writerow([history['epoch'][i], history['loss'][i], history['test_rmse'][i]])
                print(f"训练历史已保存至 '{filename}'")
            except IOError as e:
                print(f"错误：无法写入文件 {filename}. 原因: {e}")
        # --- 修改结束 ---
        print("=================================================")

if __name__ == "__main__":
    main()
