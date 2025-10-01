#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path
import itertools
import time

# --- Hyperparameter search grid ---
EXPERIMENT_GRID = {
    "activation": ["tanh", "relu", "leaky_relu"],
    "batch": [16384, 32768],
    "hidden": [64, 128],
    "mpi_processes": [4, 8]
}

# --- Fixed parameters (must match train.py args exactly) ---
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

def run_experiment(config, base, output_dir: Path):
    exp_name = "__".join([f"{k}-{v}" for k, v in config.items()])
    log_file = output_dir / f"{exp_name}.json"

    if log_file.exists():
        print(f"Skipping {exp_name} (already done)")
        return

    print(f"\n{'='*80}\nRunning: {exp_name}\n{'='*80}")

    # ✅ Build command (NO underscore → dash conversion)
    cmd = [
        "mpiexec", "-n", str(config["mpi_processes"]),
        "python", "train.py"
    ]

    # add variable hyperparams
    for key, value in config.items():
        if key != "mpi_processes":
            cmd += [f"--{key}", str(value)]

    # add fixed params
    for key, value in base.items():
        cmd += [f"--{key}", str(value)]

    # add log file
    cmd += ["--log_file", str(log_file)]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        (output_dir / f"{exp_name}.stdout.txt").write_text(result.stdout)
        return {"name": exp_name, "success": True, "time": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"✗ Failed (rc={e.returncode}) after {elapsed:.2f}s")
        print(e.stderr[-1000:])
        (output_dir / f"{exp_name}.stderr.txt").write_text(e.stderr)
        return {"name": exp_name, "success": False, "error": e.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        print("✗ Timeout (1h)")
        return {"name": exp_name, "success": False, "error": "TimeoutExpired"}

def main():
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)

    keys, values = zip(*EXPERIMENT_GRID.items())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    print(f"Starting {len(configs)} experiments...")
    all_results = []
    for cfg in configs:
        res = run_experiment(cfg, BASE_CONFIG, output_dir)
        if res:
            all_results.append(res)

    # Save summary
    summary = {"base_config": BASE_CONFIG, "results": all_results}
    (output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for r in all_results if r["success"])
    print(f"\n{'='*80}\nFinished: {ok}/{len(all_results)} successful.\nSummary saved.\n{'='*80}")

if __name__ == "__main__":
    main()

