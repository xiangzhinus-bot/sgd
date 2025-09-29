# main_pipeline.py (With Tuned Hyperparameters for Convergence)

from mpi4py import MPI
import pandas as pd
import os
import io
import gc
import numpy as np

# Import from your custom modules
from preprocessing import DataPreprocessor, train_test_split_local
from sgd import MPISGDTrainer

# --- MPI Initialization ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================
# STEP 1 to 4: DATA LOADING AND PREPROCESSING (Unchanged)
# =============================================================================
file_path = "nytaxi2022.csv"
preprocessor = None

if rank == 0:
    print("--- STEP 1: Rank 0 is fitting the preprocessor on a data sample ---")
    df_sample = pd.read_csv(file_path, nrows=100000, low_memory=False)
    preprocessor = DataPreprocessor()
    preprocessor.fit(df_sample)
    print("\nPreprocessor fitted successfully. Broadcasting to all processes...")
    del df_sample
    gc.collect()
else:
    preprocessor = None

preprocessor = comm.bcast(preprocessor, root=0)
comm.Barrier()
if rank == 0:
    print("\n--- STEP 2: All processes received the fitted preprocessor. ---")

if rank == 0:
    file_size = os.path.getsize(file_path)
    bytes_per_process = file_size // size
    offsets = [0]
    with open(file_path, 'rb') as f:
        for i in range(1, size):
            f.seek(i * bytes_per_process)
            f.readline()
            offsets.append(f.tell())
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
else:
    offsets = None
    header = None
offsets = comm.bcast(offsets, root=0)
header = comm.bcast(header, root=0)

def iter_lines_in_range(path, start, end, encoding="utf-8"):
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            _ = f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line.decode(encoding, errors="ignore")

def process_and_split_batch(rows_buffer, preprocessor_obj):
    if not rows_buffer: return None, None, None, None
    df_batch = pd.read_csv(io.StringIO("".join(rows_buffer)), names=header, header=None, engine="c", low_memory=True)
    X_scaled, y_scaled = preprocessor_obj.transform(df_batch)
    if X_scaled.shape[0] > 0:
        return train_test_split_local(X_scaled, y_scaled, seed=(42 + rank))
    return None, None, None, None

my_start = offsets[rank]
my_end = offsets[rank + 1] if rank < size - 1 else os.path.getsize(file_path)
line_iter = iter_lines_in_range(file_path, my_start, my_end)

batch_size_lines = 100_000
rows_buf = []
X_train_local, y_train_local, X_test_local, y_test_local = [], [], [], []

if rank == 0:
    print("\n--- STEP 3: Starting parallel data loading and transformation. ---\n")

for i, line in enumerate(line_iter, 1):
    rows_buf.append(line)
    if i % batch_size_lines == 0:
        X_train_b, y_train_b, X_test_b, y_test_b = process_and_split_batch(rows_buf, preprocessor)
        if X_train_b is not None and len(X_train_b) > 0:
            X_train_local.append(X_train_b); y_train_local.append(y_train_b)
            X_test_local.append(X_test_b); y_test_local.append(y_test_b)
        rows_buf.clear()
        gc.collect()

if rows_buf:
    X_train_b, y_train_b, X_test_b, y_test_b = process_and_split_batch(rows_buf, preprocessor)
    if X_train_b is not None and len(X_train_b) > 0:
        X_train_local.append(X_train_b); y_train_local.append(y_train_b)
        X_test_local.append(X_test_b); y_test_local.append(y_test_b)
    rows_buf.clear()

if X_train_local:
    X_train_local = np.vstack(X_train_local)
    y_train_local = np.concatenate(y_train_local)
    X_test_local = np.vstack(X_test_local)
    y_test_local = np.concatenate(y_test_local)
else:
    feature_dim = len(preprocessor.get_feature_names())
    X_train_local = np.empty((0, feature_dim)); y_train_local = np.empty(0)
    X_test_local = np.empty((0, feature_dim)); y_test_local = np.empty(0)

comm.Barrier()
if rank == 0:
    print("\n--- STEP 4: Data preprocessing complete. All processes hold their local data. ---")


# =============================================================================
# STEP 5: INITIALIZE AND TRAIN THE NEURAL NETWORK
# =============================================================================
comm.Barrier()
if rank == 0:
    print("\n--- STEP 5: Initializing and training the distributed neural network. ---\n")

# --- Model Hyperparameters ---
# --- FIX: Adjusted for stability and convergence ---
INPUT_DIM = X_train_local.shape[1]
HIDDEN_DIM = 64              # Increased hidden size for more capacity with tanh
ACTIVATION = 'tanh'          # Switched to tanh, which is more stable than relu
LEARNING_RATE = 0.001        # Drastically reduced learning rate to prevent explosion
BATCH_SIZE = 256             # Larger batch size can also help stabilize training
MAX_EPOCHS = 200             # Increased epochs because of smaller learning rate
PATIENCE = 10                # Increased patience for early stopping

# Each process initializes its own trainer
trainer = MPISGDTrainer(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    activation=ACTIVATION,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE
)

# Train the model on the local data chunks. The trainer handles all MPI communication.
train_results = trainer.train(X_train_local, y_train_local)


# =============================================================================
# STEP 6: EVALUATE THE MODEL AND PRINT FINAL RESULTS (Unchanged)
# =============================================================================
comm.Barrier()
if rank == 0:
    print("\n--- STEP 6: Evaluating the final model. ---\n")

eval_results = trainer.evaluate(X_test_local, y_test_local)

# --- Calculate final RMSE in original dollars ---
predictions_scaled = trainer.network.predict(X_test_local)
predictions_unscaled = preprocessor.inverse_transform_target(predictions_scaled)
y_test_unscaled = preprocessor.inverse_transform_target(y_test_local)

local_sse = np.sum((predictions_unscaled - y_test_unscaled)**2)
global_sse = comm.allreduce(local_sse, op=MPI.SUM)
total_test_samples = comm.allreduce(len(X_test_local), op=MPI.SUM)

if rank == 0:
    final_rmse = np.sqrt(global_sse / total_test_samples) if total_test_samples > 0 else 0.0

    print("--- Final Results ---")
    print(f"Training converged: {train_results['converged']} in {train_results['epochs']} epochs.")
    print(f"Total training time: {train_results['total_time']:.2f} seconds.")
    print(f"Final training loss (scaled): {train_results['final_loss']:.6f}")
    print("-" * 25)
    print(f"Test Set RMSE (scaled): {eval_results['rmse']:.6f}")
    print(f"Test Set RMSE (unscaled): ${final_rmse:.2f}")
    print("-" * 25)
    print("\nEnd-to-end pipeline complete.")