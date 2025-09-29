# parallel_preprocess_test.py (Corrected and Optimized)

from mpi4py import MPI
import pandas as pd
import os
import io
import gc
import numpy as np

# Import from your custom preprocessing module
from preprocessing import DataPreprocessor, train_test_split_local

# --- MPI Initialization ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

file_path = "nytaxi2022.csv"
preprocessor = None

# STEP 1: RANK 0 - FITS THE PREPROCESSOR ON A SAMPLE AND BROADCASTS IT
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
    print("\n--- STEP 2: All processes received the fitted preprocessor. Starting parallel data loading and transformation. ---\n")

# STEP 2: ALL RANKS - DETERMINE FILE CHUNKS
if rank == 0:
    file_size = os.path.getsize(file_path)
    bytes_per_process = file_size // size
    offsets = [0]
    with open(file_path, 'rb') as f:
        for i in range(1, size):
            f.seek(i * bytes_per_process)
            f.readline() # Seek to the next newline
            offsets.append(f.tell())
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
else:
    offsets = None
    header = None

offsets = comm.bcast(offsets, root=0)
header = comm.bcast(header, root=0)

# STEP 3: ALL RANKS - PROCESS THEIR ASSIGNED CHUNK IN BATCHES (FIXED LOGIC)
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

# Helper function to avoid repeating code
def process_and_split_batch(rows_buffer, preprocessor_obj):
    if not rows_buffer:
        return None, None, None, None

    df_batch = pd.read_csv(
        io.StringIO("".join(rows_buffer)),
        names=header,
        header=None,
        engine="c",
        low_memory=True
    )
    
    X_scaled, y_scaled = preprocessor_obj.transform(df_batch)
    
    if X_scaled.shape[0] > 0:
        return train_test_split_local(
            X_scaled, 
            y_scaled, 
            seed=(42 + rank)
        )
    return None, None, None, None

my_start = offsets[rank]
my_end = offsets[rank + 1] if rank < size - 1 else os.path.getsize(file_path)
line_iter = iter_lines_in_range(file_path, my_start, my_end)

batch_size_lines = 100_000
rows_buf = []
total_rows_read = 0

X_train_local, y_train_local = [], []
X_test_local, y_test_local = [], []

print(f"Rank {rank} starting to process chunk from byte {my_start} to {my_end}.")

for i, line in enumerate(line_iter, 1):
    rows_buf.append(line)
    total_rows_read += 1
    # --- EFFICIENT BATCHING ---
    # Only process a batch when it's full.
    if i % batch_size_lines == 0:
        X_train_b, y_train_b, X_test_b, y_test_b = process_and_split_batch(rows_buf, preprocessor)
        if X_train_b is not None:
            X_train_local.append(X_train_b)
            y_train_local.append(y_train_b)
            X_test_local.append(X_test_b)
            y_test_local.append(y_test_b)
        
        rows_buf.clear()
        gc.collect()

# --- EFFICIENT FINAL BATCH ---
# After the loop, process any remaining lines in the buffer.
if rows_buf:
    X_train_b, y_train_b, X_test_b, y_test_b = process_and_split_batch(rows_buf, preprocessor)
    if X_train_b is not None:
        X_train_local.append(X_train_b)
        y_train_local.append(y_train_b)
        X_test_local.append(X_test_b)
        y_test_local.append(y_test_b)
    
    rows_buf.clear()
    gc.collect()

# Final consolidation of local data
if X_train_local:
    X_train_local = np.vstack(X_train_local)
    y_train_local = np.concatenate(y_train_local)
    X_test_local = np.vstack(X_test_local)
    y_test_local = np.concatenate(y_test_local)
else: 
    feature_dim = len(preprocessor.get_feature_names())
    X_train_local = np.empty((0, feature_dim))
    y_train_local = np.empty(0)
    X_test_local = np.empty((0, feature_dim))
    y_test_local = np.empty(0)

print(
    f"Process {rank} finished. "
    f"Total rows read: {total_rows_read}. "
    f"Train samples: {len(X_train_local)}, "
    f"Test samples: {len(X_test_local)}"
)

# STEP 4: GATHER RESULTS (Verification)
comm.Barrier()
local_counts = (len(X_train_local), len(X_test_local))
all_counts = comm.gather(local_counts, root=0)

if rank == 0:
    print("\n--- STEP 4: Final Tally ---")
    total_train = sum(item[0] for item in all_counts)
    total_test = sum(item[1] for item in all_counts)
    total_processed = total_train + total_test
    
    print(f"Total processed train samples across all processes: {total_train}")
    print(f"Total processed test samples across all processes: {total_test}")
    if total_processed > 0:
        print(f"Overall test ratio: {total_test / total_processed:.2%}")
    print("\nEnd-to-end parallel data preprocessing test complete.")