# pipeline_streaming.py (Corrected)

from mpi4py import MPI
import pandas as pd
import os
import io
import numpy as np
import time

from preprocessing import DataPreprocessor, train_test_split_local
from sgd_streaming import MPISGDTrainer

# --- MPI Initialization ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================
# STEP 1 & 2: PREPROCESSOR, FILE CHUNKS, AND HEADER
# =============================================================================
file_path = "nytaxi2022.csv"
preprocessor = None
header = None

if rank == 0:
    print("--- STEP 1: Fitting the preprocessor & reading header ---")
    df_sample = pd.read_csv(file_path, nrows=100000, low_memory=False)
    preprocessor = DataPreprocessor()
    preprocessor.fit(df_sample)
    print("Preprocessor fitted successfully. Broadcasting...")
    del df_sample
    
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    
    file_size = os.path.getsize(file_path)
    bytes_per_process = file_size // size
    offsets = [0]
    with open(file_path, 'rb') as f:
        for i in range(1, size):
            f.seek(i * bytes_per_process); f.readline()
            offsets.append(f.tell())
else:
    offsets = None

preprocessor = comm.bcast(preprocessor, root=0)
header = comm.bcast(header, root=0)
offsets = comm.bcast(offsets, root=0)

my_start = offsets[rank]
my_end = offsets[rank + 1] if rank < size - 1 else os.path.getsize(file_path)

# =============================================================================
# STEP 3: INITIALIZE MODEL AND HELPERS
# =============================================================================
INPUT_DIM = len(preprocessor.get_feature_names())
HIDDEN_DIM = 64
ACTIVATION = 'tanh'
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
TOLERANCE = 1e-6 # Define the missing hyperparameter

trainer = MPISGDTrainer(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, activation=ACTIVATION,
                        learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS,
                        tolerance=TOLERANCE, # <-- FIX: Pass the missing argument
                        patience=PATIENCE)
trainer.sync_parameters()

def iter_lines_in_range(path, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0: _ = f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line: break
            yield line.decode("utf-8", errors="ignore")

def process_and_split(rows_buffer, preprocessor_obj, header_list):
    if not rows_buffer:
        return np.array([]).reshape(0, INPUT_DIM), np.array([]), np.array([]).reshape(0, INPUT_DIM), np.array([])
    df = pd.read_csv(io.StringIO("".join(rows_buffer)), names=header_list, header=None, engine="c", low_memory=True)
    X, y = preprocessor_obj.transform(df)
    if X.shape[0] > 0:
        return train_test_split_local(X, y, seed=(42 + rank))
    return np.array([]).reshape(0, INPUT_DIM), np.array([]), np.array([]).reshape(0, INPUT_DIM), np.array([])

# =============================================================================
# STEP 4: PER-EPOCH TRAINING LOOP
# =============================================================================
if rank == 0: print("\n--- STEP 2: Starting stable per-epoch training ---")
total_train_start_time = time.time()

for epoch in range(MAX_EPOCHS):
    epoch_start_time = time.time()
    
    line_iter = iter_lines_in_range(file_path, my_start, my_end)
    local_grad_sum, local_loss_sum, local_samples = trainer.run_epoch(line_iter, process_and_split, preprocessor, header)
    
    send_buf = np.append(local_grad_sum, [local_loss_sum, local_samples])
    recv_buf = np.empty_like(send_buf)
    comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
    
    global_grad_sum, global_loss_sum, global_samples = recv_buf[:-2], recv_buf[-2], recv_buf[-1]

    avg_epoch_loss = global_loss_sum / global_samples if global_samples > 0 else 0.0
    
    if rank == 0 and (epoch + 1) % 5 == 0:
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}, Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
    # All processes must check for convergence to know when to break the loop
    if trainer.check_convergence(avg_epoch_loss):
        # Use a collective operation to ensure all processes break together
        converged_flag = 1
    else:
        converged_flag = 0
    
    global_converged_flag = comm.allreduce(converged_flag, op=MPI.MAX)
    if global_converged_flag == 1:
        break # All processes break the loop

    if global_samples > 0:
        global_avg_gradient = global_grad_sum / global_samples
        trainer.update_parameters(global_avg_gradient)

if rank == 0:
    total_time = time.time() - total_train_start_time
    print(f"\nTraining complete in {total_time:.2f}s")

# =============================================================================
# STEP 5: FINAL EVALUATION
# =============================================================================
if rank == 0: print("\n--- STEP 3: Evaluating the final model ---")

line_iter_eval = iter_lines_in_range(file_path, my_start, my_end)
eval_results = trainer.evaluate(line_iter_eval, process_and_split, preprocessor, header)

if rank == 0:
    final_loss = trainer.history['loss'][-1] if trainer.history['loss'] else float('nan')
    print("\n--- Final Results ---")
    print(f"Training converged: {trainer.converged} in {len(trainer.history['loss'])} epochs.")
    print(f"Final training loss (scaled): {final_loss:.6f}")
    print("-" * 25)
    print(f"Test Set RMSE (scaled): {eval_results['rmse_scaled']:.6f}")
    print(f"Test Set RMSE (unscaled): ${eval_results['rmse_unscaled']:.2f}")
    print("-" * 25)
    print("End-to-end pipeline complete.")