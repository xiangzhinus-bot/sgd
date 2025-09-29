# debug_dates.py

import pandas as pd

FILE_PATH = 'nytaxi2022.csv'
# Let's read 1000 rows starting from row 2,000,000 to get a sample
# from a different part of the file.
try:
    df_sample = pd.read_csv(
        FILE_PATH,
        skiprows=range(1, 2000000), # Skip the first 2M rows
        nrows=1000,                  # Read the next 1k rows
        header=0                     # The first row we read is the header
    )
    # If the above fails because the file is smaller, read from the start
except (ValueError, StopIteration):
     df_sample = pd.read_csv(FILE_PATH, nrows=1000)


print("--- Unique values from 'tpep_pickup_datetime' ---")
print(df_sample['tpep_pickup_datetime'].unique()[:10]) # Print first 10 unique values

print("\n--- Unique values from 'tpep_dropoff_datetime' ---")
print(df_sample['tpep_dropoff_datetime'].unique()[:10])

print("\n\n--- How to interpret the format ---")
print("Look at the output above and match the format to the codes below:")
print("  %Y: 4-digit year (e.g., 2022)")
print("  %m: 2-digit month (e.g., 01, 12)")
print("  %d: 2-digit day (e.g., 01, 31)")
print("  %H: 2-digit hour (24-hour clock, e.g., 08, 23)")
print("  %M: 2-digit minute (e.g., 05, 59)")
print("  %S: 2-digit second (e.g., 01, 45)")
print("\nExample: If you see '01/31/2022 14:55:03', the format is '%m/%d/%Y %H:%M:%S'")
print("Example: If you see '2022-01-31 14:55:03', the format is '%Y-%m-%d %H:%M:%S'")