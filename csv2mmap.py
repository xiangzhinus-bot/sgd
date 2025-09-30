import pandas as pd
import numpy as np
import os
import argparse
import time
from mpi4py import MPI
from sklearn.preprocessing import minmax_scale

# --- Configuration ---
# Columns to be used for processing
SELECTED_COLUMNS = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
    'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
    'payment_type', 'extra', 'total_amount'
]
TARGET_COLUMN = 'total_amount'
# Datetime format for parsing
TIME_FORMAT = '%m/%d/%Y %I:%M:%S %p'
# Data type for final mmap files
FINAL_DTYPE = np.float32
# Train/Test split ratio
TEST_RATIO = 0.3
# Seed for reproducible train/test split
RANDOM_SEED = 42
# Number of rows to read from CSV at a time
CHUNK_SIZE = 50000

def preprocess_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning and feature engineering on a DataFrame chunk.

    Args:
        df_chunk: A pandas DataFrame chunk read from the CSV.

    Returns:
        A cleaned DataFrame with engineered features, ready for scaling.
        Returns an empty DataFrame if the chunk is invalid after processing.
    """
    # 1. Select relevant columns
    try:
        df = df_chunk[SELECTED_COLUMNS].copy()
    except KeyError as e:
        print(f"Warning: Missing column {e} in a chunk. Skipping.")
        return pd.DataFrame()

    # 2. Handle datetime features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format=TIME_FORMAT, errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=TIME_FORMAT, errors='coerce')
    
    # Calculate trip duration in minutes
    duration = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df['time_on_taxi'] = duration
    
    # Drop original datetime columns
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    # 3. Drop rows with any missing values after processing
    df.dropna(inplace=True)
    
    # Reorder columns to ensure target is last
    cols = [col for col in df.columns if col != TARGET_COLUMN] + [TARGET_COLUMN]
    
    return df[cols].astype(FINAL_DTYPE)


def process_csv_to_mmap(csv_file_path: str, output_prefix: str = 'processed'):
    """
    Main function to coordinate the parallel processing of a CSV file into
    four memory-mapped files (X_train, X_test, y_train, y_test).
    
    Args:
        csv_file_path: Path to the input CSV file.
        output_prefix: Prefix for the output .mmap files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # --- Rank 0: Initial Setup ---
    if rank == 0:
        print(f"🚀 Starting parallel preprocessing with {size} processes.")
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found at {csv_file_path}")
            comm.Abort() # Terminate all processes
        
        # Read header to determine column count later
        header = pd.read_csv(csv_file_path, nrows=0).columns.tolist()
        
        # Calculate byte chunks for each process to read
        file_size = os.path.getsize(csv_file_path)
        bytes_per_process = file_size // size
        offsets = [0] * (size + 1)
        with open(csv_file_path, 'rb') as f:
            # Skip header line for chunk calculation
            header_offset = len(f.readline())
            offsets[0] = header_offset
            for i in range(1, size):
                f.seek(i * bytes_per_process)
                f.readline() # Align to the start of a new line
                offsets[i] = f.tell()
            offsets[size] = file_size
    else:
        header = None
        offsets = None

    # Broadcast header and offsets to all processes
    header = comm.bcast(header, root=0)
    offsets = comm.bcast(offsets, root=0)

    # Each process determines its chunk of the file
    my_start_offset = offsets[rank]
    my_end_offset = offsets[rank + 1]
    my_chunk_bytes = my_end_offset - my_start_offset

    if my_chunk_bytes <= 0:
        # This process has no work to do
        local_valid_rows = 0
        local_min = np.array([], dtype=FINAL_DTYPE)
        local_max = np.array([], dtype=FINAL_DTYPE)
    else:
        # --- Pass 1: Gather statistics (min, max, row count) ---
        local_valid_rows = 0
        local_min_list, local_max_list = [], []
        
        # Read the assigned byte range using chunking
        with open(csv_file_path, 'r') as f:
            f.seek(my_start_offset)
            reader = pd.read_csv(f, chunksize=CHUNK_SIZE, names=header, engine='c',
                                 nrows=(my_end_offset - my_start_offset) // 100) # Estimate rows

            for chunk in reader:
                if f.tell() > my_end_offset and rank != size-1:
                    # Ensure we don't read past our assigned boundary
                    # (except for the last process)
                    break
                
                processed_df = preprocess_chunk(chunk)
                if not processed_df.empty:
                    local_valid_rows += len(processed_df)
                    local_min_list.append(processed_df.min().values)
                    local_max_list.append(processed_df.max().values)
        
        if local_min_list:
            local_min = np.min(np.array(local_min_list), axis=0)
            local_max = np.max(np.array(local_max_list), axis=0)
        else: # Handle case where a process finds no valid rows
            local_min = np.array([], dtype=FINAL_DTYPE)
            local_max = np.array([], dtype=FINAL_DTYPE)
            local_valid_rows = 0

    # Aggregate statistics using MPI Allreduce
    # Sum of valid rows from all processes
    total_valid_rows = comm.allreduce(local_valid_rows, op=MPI.SUM)
    
    # Need to handle empty chunks before reducing min/max
    has_data = 1 if local_min.size > 0 else 0
    num_features = 0
    if has_data:
        num_features = len(local_min)
    
    # Determine the number of features from any rank that has data
    num_features = comm.allreduce(num_features, op=MPI.MAX)
    
    # Initialize with neutral values for min/max reduction
    if not has_data:
        local_min = np.full(num_features, np.inf, dtype=FINAL_DTYPE)
        local_max = np.full(num_features, -np.inf, dtype=FINAL_DTYPE)

    # Use capital 'A' Allreduce with separate buffers for numpy arrays
    global_min = np.empty_like(local_min)
    global_max = np.empty_like(local_max)
    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)

    if rank == 0:
        if total_valid_rows == 0:
            print("Error: No valid data found after preprocessing.")
            comm.Abort()
        print(f"PASS 1 COMPLETE: Found {total_valid_rows} valid rows.")
        print("Global min/max calculated. Proceeding to Pass 2.")

    # --- Rank 0: Create mmap files and determine split indices ---
    if rank == 0:
        # Calculate shapes for train/test splits
        n_features = num_features - 1 # All columns except the last one (target)
        n_test = int(total_valid_rows * TEST_RATIO)
        n_train = total_valid_rows - n_test
        
        # Define mmap file paths
        mmap_paths = {
            'X_train': f'{output_prefix}_X_train.mmap', 'y_train': f'{output_prefix}_y_train.mmap',
            'X_test': f'{output_prefix}_X_test.mmap', 'y_test': f'{output_prefix}_y_test.mmap'
        }
        
        # Create empty mmap files
        for path in mmap_paths.values():
            if os.path.exists(path):
                os.remove(path)
                
        _ = np.memmap(mmap_paths['X_train'], dtype=FINAL_DTYPE, mode='w+', shape=(n_train, n_features))
        _ = np.memmap(mmap_paths['y_train'], dtype=FINAL_DTYPE, mode='w+', shape=(n_train,))
        _ = np.memmap(mmap_paths['X_test'], dtype=FINAL_DTYPE, mode='w+', shape=(n_test, n_features))
        _ = np.memmap(mmap_paths['y_test'], dtype=FINAL_DTYPE, mode='w+', shape=(n_test,))
        
        # Create shuffled indices and identify which ones are for testing
        indices = np.arange(total_valid_rows)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
        test_indices_set = set(indices[:n_test])
    else:
        mmap_paths = None
        test_indices_set = None

    # Broadcast necessary info for Pass 2
    mmap_paths = comm.bcast(mmap_paths, root=0)
    test_indices_set = comm.bcast(test_indices_set, root=0)
    
    # --- Determine write offsets for each process ---
    # Each process calculates its global starting index for valid rows
    # The scan operation gives a running total across processes
    scan_sum = comm.scan(local_valid_rows, op=MPI.SUM)
    global_start_idx = scan_sum - local_valid_rows
    
    # Wait for all processes to reach this point before writing
    comm.Barrier()
    
    # --- Pass 2: Scale data and write to mmap files ---
    # Open all mmap files in read/write mode
    # Need to broadcast shapes for proper memmap opening
    n_features = num_features - 1
    n_test = comm.bcast(len(test_indices_set) if rank == 0 else None, root=0)
    n_train = comm.bcast(total_valid_rows - n_test if rank == 0 else None, root=0)

    # Open all mmap files in read/write mode WITH PROPER SHAPES
    X_train_mmap = np.memmap(mmap_paths['X_train'], dtype=FINAL_DTYPE, mode='r+', shape=(n_train, n_features))
    y_train_mmap = np.memmap(mmap_paths['y_train'], dtype=FINAL_DTYPE, mode='r+', shape=(n_train,))
    X_test_mmap = np.memmap(mmap_paths['X_test'], dtype=FINAL_DTYPE, mode='r+', shape=(n_test, n_features))
    y_test_mmap = np.memmap(mmap_paths['y_test'], dtype=FINAL_DTYPE, mode='r+', shape=(n_test,))
    
    # Calculate how many train/test items this process is responsible for
    # This is needed to calculate the write start positions
    local_test_count = sum(1 for i in range(global_start_idx, global_start_idx + local_valid_rows) if i in test_indices_set)
    local_train_count = local_valid_rows - local_test_count

    # Scan again to find write start positions for train/test mmap files
    train_write_start = comm.scan(local_train_count, op=MPI.SUM) - local_train_count
    test_write_start = comm.scan(local_test_count, op=MPI.SUM) - local_test_count
    
    train_ptr, test_ptr = 0, 0
    current_global_idx = global_start_idx

    if my_chunk_bytes > 0:
        with open(csv_file_path, 'r') as f:
            f.seek(my_start_offset)
            reader = pd.read_csv(f, chunksize=CHUNK_SIZE, names=header, engine='c',
                                 nrows=(my_end_offset - my_start_offset) // 100) # Estimate

            for chunk in reader:
                if f.tell() > my_end_offset and rank != size - 1:
                    break
                    
                processed_df = preprocess_chunk(chunk)
                if processed_df.empty:
                    continue
                
                # Scale the entire chunk using global min/max
                scaled_data = minmax_scale(processed_df, feature_range=(0, 1), axis=0)
                
                # Separate features (X) and target (y)
                X_scaled = scaled_data[:, :-1]
                y_scaled = scaled_data[:, -1]
                
                # Write each row to the correct mmap file
                for i in range(len(processed_df)):
                    if current_global_idx in test_indices_set:
                        X_test_mmap[test_write_start + test_ptr] = X_scaled[i]
                        y_test_mmap[test_write_start + test_ptr] = y_scaled[i]
                        test_ptr += 1
                    else:
                        X_train_mmap[train_write_start + train_ptr] = X_scaled[i]
                        y_train_mmap[train_write_start + train_ptr] = y_scaled[i]
                        train_ptr += 1
                    current_global_idx += 1
    
    # Ensure all changes are written to disk and wait for all processes
    X_train_mmap.flush()
    y_train_mmap.flush()
    X_test_mmap.flush()
    y_test_mmap.flush()
    
    comm.Barrier()
    
    if rank == 0:
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"✅ SUCCESS: Preprocessing and conversion completed in {total_time:.2f} seconds.")
        print("Output files created:")
        for name, path in mmap_paths.items():
            print(f"  - {mmap_paths['X_train']} (shape: {(n_train, n_features)}, dtype: {FINAL_DTYPE})")
            print(f"  - {mmap_paths['y_train']} (shape: {(n_train,)}, dtype: {FINAL_DTYPE})")
            print(f"  - {mmap_paths['X_test']} (shape: {(n_test, n_features)}, dtype: {FINAL_DTYPE})")
            print(f"  - {mmap_paths['y_test']} (shape: {(n_test,)}, dtype: {FINAL_DTYPE})")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel CSV to Memmap Preprocessing using MPI.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    process_csv_to_mmap(args.csv_file)