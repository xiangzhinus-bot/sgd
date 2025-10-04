#!/usr/bin/env python3
import subprocess
import json
import os
from pathlib import Path
import itertools
import time
from copy import deepcopy

# =========================================================
# 固定配置（不使用环境变量）
# =========================================================

# 多机主机名（留空则单机）。示例：["TUFGAMINGA15", "XIAOXINPRO14"]
HOSTS: list[str] = []
# 如果你有 OpenMPI hostfile，在这里填路径；否则设为 None
HOSTFILE: str | None = None
# 额外 mpiexec 参数（可留空）
EXTRA_MPI_ARGS: list[str] = []  # 例如 ["--oversubscribe"] 或 ["-mca","btl","^openib"]

# 结果输出目录
RESULT_DIR = Path("experiment_results")

# 是否自动放大 superbatch（会覆盖网格里的 q_super）
FORCE_AUTO_Q_SUPER: bool = True
# 目标 “superbatch 样本数”：batch * q_super ≈ TARGET_SUPER_SAMPLES
TARGET_SUPER_SAMPLES: int = 1_000_000
# 安全上限，防止内存过大
MAX_Q_SUPER: int = 128

# 训练曲线记录粒度（配合 train.py 的新开关）
HISTORY_GRANULARITY = "step"   # "step" 或 "epoch"
LOG_EVERY_STEPS = 10           # step 粒度下，每隔多少次更新记录一次（太细会很大）

# =========================================================
# 网格与固定参数（需与 train.py 完全一致）
# =========================================================

EXPERIMENT_GRID = {
    "activation": ["tanh", "relu", "leaky_relu"],
    # 覆盖 >=5 个 batch size（按你机器内存适当调整）
    "batch": [8192, 16384, 32768, 65536, 131072],
    "hidden": [64, 128],
    "mpi_processes": [4, 8],
    # I/O 相关（若 FORCE_AUTO_Q_SUPER=True，将被自动覆盖）
    "q_super": [8],
    "prefetch": [2, 3],
}

BASE_CONFIG = {
    "prefix": "processed",
    "ext": "mmap",
    "features": 8,
    "dtype": "float32",
    "lr": 0.001,
    "lr_schedule": "cosine",
    "lr_scale": "sqrt",
    "warmup": 5,
    "lr_step_epochs": 10,
    "lr_gamma": 0.5,
    "weight_decay": 1e-5,
    "clip": 1.0,
    "epochs": 10,
    "patience": 3,
    "min_delta": 1e-6,
    "eval_batch": 262144,
    "seed": 2025,
    "momentum": 0.9,
}

# 子进程环境：限制 BLAS 线程为 1，避免“进程×线程”过订阅
PINNED_ENV = dict(os.environ)
PINNED_ENV["OMP_NUM_THREADS"] = "1"
PINNED_ENV["OPENBLAS_NUM_THREADS"] = "1"
PINNED_ENV["MKL_NUM_THREADS"] = "1"
PINNED_ENV["NUMEXPR_NUM_THREADS"] = "1"


# =========================================================
# 内部函数
# =========================================================

def _effective_config(cfg: dict) -> dict:
    """按需覆盖 q_super：使 batch * q_super ≈ TARGET_SUPER_SAMPLES。"""
    eff = deepcopy(cfg)
    if FORCE_AUTO_Q_SUPER:
        b = int(eff.get("batch", 65536))
        q = max(1, TARGET_SUPER_SAMPLES // b)
        q = min(q, MAX_Q_SUPER)  # 上限保护
        eff["q_super"] = q
    return eff


def _exp_name_from_config(cfg: dict) -> str:
    """稳定可读的实验名（按 key 排序；mpi_processes 放前面）。"""
    items = list(cfg.items())
    items.sort(key=lambda kv: (kv[0] != "mpi_processes", kv[0]))
    return "__".join(f"{k}-{v}" for k, v in items)


def run_experiment(config, base, output_dir: Path):
    eff_cfg = _effective_config(config)
    exp_name = _exp_name_from_config(eff_cfg)

    log_json   = output_dir / f"{exp_name}.json"
    stdout_txt = output_dir / f"{exp_name}.stdout.txt"
    stderr_txt = output_dir / f"{exp_name}.stderr.txt"
    cmd_txt    = output_dir / f"{exp_name}.cmd.txt"

    if log_json.exists():
        print(f"Skipping {exp_name} (already done)")
        return {"name": exp_name, "skipped": True}

    print(f"\n{'='*80}\nRunning: {exp_name}\n{'='*80}")

    # 构造 mpiexec 命令
    cmd = ["mpiexec"]
    if HOSTFILE:
        cmd += ["-hostfile", HOSTFILE]
    elif HOSTS:
        cmd += ["-host", ",".join(HOSTS)]
    cmd += ["-n", str(eff_cfg["mpi_processes"])]
    if EXTRA_MPI_ARGS:
        cmd += EXTRA_MPI_ARGS

    # Python + 脚本
    cmd += ["python", "train.py"]

    # 可变参数（排除 mpi_processes）
    for k, v in eff_cfg.items():
        if k == "mpi_processes":
            continue
        cmd += [f"--{k}", str(v)]

    # 固定参数
    for k, v in base.items():
        cmd += [f"--{k}", str(v)]

    # 日志文件
    cmd += ["--history_granularity", HISTORY_GRANULARITY,
            "--log_every_steps", str(LOG_EVERY_STEPS)]
    cmd += ["--log_file", str(log_json)]

    # 写出命令用于复现
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd_txt.write_text(" ".join(cmd))

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600,
            env=PINNED_ENV,
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        stdout_txt.write_text(result.stdout)
        return {"name": exp_name, "success": True, "time": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"✗ Failed (rc={e.returncode}) after {elapsed:.2f}s")
        print("--- STDERR (tail) ---")
        print(e.stderr[-1000:])
        stderr_txt.write_text(e.stderr)
        return {"name": exp_name, "success": False, "error": e.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        print("✗ Timeout (1h)")
        return {"name": exp_name, "success": False, "error": "TimeoutExpired"}


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 笛卡尔积生成配置
    keys, values = zip(*EXPERIMENT_GRID.items())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    print(f"Starting {len(configs)} experiments...")
    all_results = []
    for cfg in configs:
        res = run_experiment(cfg, BASE_CONFIG, RESULT_DIR)
        if res:
            all_results.append(res)

    summary = {
        "base_config": BASE_CONFIG,
        "force_auto_q_super": FORCE_AUTO_Q_SUPER,
        "target_super_samples": TARGET_SUPER_SAMPLES,
        "max_q_super": MAX_Q_SUPER,
        "history_granularity": HISTORY_GRANULARITY,
        "log_every_steps": LOG_EVERY_STEPS,
        "results": all_results,
    }
    (RESULT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for r in all_results if r.get("success"))
    print(f"\n{'='*80}\nFinished: {ok}/{len(all_results)} successful."
          f"\nSummary saved to {RESULT_DIR / '_summary.json'}\n{'='*80}")


if __name__ == "__main__":
    main()
