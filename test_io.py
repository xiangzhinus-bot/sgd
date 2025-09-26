# file_path = 'nytaxi2022.csv'

# with open(file_path, 'r', encoding='utf8') as f:
#     row_count = sum(1 for line in f)

# print(f"总行数: {row_count}")

# import pandas as pd

# df_col = pd.read_csv(file_path, usecols=['total_amount']) 
# row_count = df_col.shape[0]
# print(f"（优化 Pandas）数据行数: {row_count}")

from mpi4py import MPI
import pandas as pd
import os
import io

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

file_path = "nytaxi2022.csv"

# 步骤 1: 只有主进程处理文件分块
if rank == 0:
    file_size = os.path.getsize(file_path)
    # 计算每个进程大致需要处理的字节数
    bytes_per_process = file_size // size
    
    # 一个列表，存储每个进程的起始读取位置
    offsets = [0]
    
    with open(file_path, 'rb') as f: # 以二进制模式读取，方便处理字节
        # 确定每个进程的起始偏移量
        for i in range(1, size):
            # 找到一个大致的分割点
            split_point = i * bytes_per_process
            # 移动文件指针到这个位置
            f.seek(split_point)
            
            # 向前扫描，直到找到下一个换行符
            # 这确保了每个块都从一个完整行的开头开始
            while True:
                char = f.read(1)
                if not char or char == b'\n':
                    break
            # 记录下一个进程的起始位置
            offsets.append(f.tell())
            
    # 主进程读取 CSV 的表头
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    
    print(f"主进程已确定每个进程的起始偏移量：{offsets}")
else:
    # 其他进程接收广播
    offsets = None
    header = None

# 步骤 2: 广播偏移量和表头
offsets = comm.bcast(offsets, root=0)
header = comm.bcast(header, root=0)

# 步骤 3: 每个进程独立读取其分配的块 —— 流式 + 分批
import io, gc
from pathlib import Path

def iter_lines_in_range(path, start, end, encoding="utf-8"):
    """按字节范围 [start, end) 逐行产出 UTF-8 文本行（不含跨越 end 的半行）"""
    with open(path, "rb") as f:
        f.seek(start)
        if start != 0:
            _ = f.readline()  # 丢弃半行
        while True:
            pos_before = f.tell()
            if pos_before >= end:
                break
            line = f.readline()
            if not line:
                break
            pos_after = f.tell()
            if pos_after > end:
                break  # 跨 end 的半行丢弃
            yield line.decode(encoding, errors="ignore")

my_start = offsets[rank]
my_end   = offsets[rank + 1] if rank < size - 1 else os.path.getsize(file_path)

line_iter = iter_lines_in_range(file_path, my_start, my_end)

# —— 选项 A：分批解析后“就地训练”（推荐：最省内存）——
# 你可以把这里的 “TODO: your_train_step(df_batch)” 换成你自己的训练逻辑
batch_size_lines = 100_000   # 依据内存调整：50k~300k
rows_buf = []
total_rows = 0
batch_idx = 0

for i, line in enumerate(line_iter, 1):
    rows_buf.append(line)
    if i % batch_size_lines == 0:
        batch_idx += 1
        df_batch = pd.read_csv(
            io.StringIO("".join(rows_buf)),
            names=header,
            header=None,
            engine="c",
            low_memory=True  # 降低内存占用
        )
        total_rows += len(df_batch)

        # TODO: your_train_step(df_batch)  # 在这一批上训练/统计
        # 例如：累计梯度、写本地中间结果等

        # 释放内存
        rows_buf.clear()
        del df_batch
        gc.collect()

# 收尾
if rows_buf:
    batch_idx += 1
    df_batch = pd.read_csv(
        io.StringIO("".join(rows_buf)),
        names=header,
        header=None,
        engine="c",
        low_memory=True
    )
    total_rows += len(df_batch)
    # TODO: your_train_step(df_batch)
    rows_buf.clear()
    del df_batch
    gc.collect()

print(f"进程 {rank} 成功（流式）处理了 {total_rows} 行数据。")

# —— 可选的 选项 B：若必须先落盘，再训练（避免持有整块内存）——
# 如果你想把每个进程的数据写成多个小 Parquet 再训练，把上面 TODO 改为：
#   out_dir = Path(f"_nytaxi_chunks_rank{rank}")
#   out_dir.mkdir(exist_ok=True)
#   df_batch.to_parquet(out_dir / f"part-{batch_idx:05d}.parquet", index=False)
# 最后所有 rank 处理完后，再用 Dask/Polars/Pandas 读取这些 parquet 做训练。
