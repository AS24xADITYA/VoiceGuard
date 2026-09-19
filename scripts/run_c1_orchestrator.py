"""Orchestrator to launch 4 parallel workers across the 713 row groups of the eval partition."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
python_exe = repo_root / "backend" / "venv" / "Scripts" / "python.exe"

parquet_file = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" / "datasets--Bisher--ASVspoof_2019_LA" / "snapshots" / "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c" / "data" / "test-00000-of-00001.parquet"
protocol_file = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" / "datasets--Nemez1z--asvspoof-2019-la" / "snapshots" / "4ad1b061fa6dc502b30888ee467ed4115fdb44eb" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
checkpoint_file = repo_root / "backend" / "models" / "acoustic.pth"
worker_script = repo_root / "scripts" / "run_c1_worker.py"
aggregator_script = repo_root / "scripts" / "aggregate_c1.py"

scratch_dir = repo_root / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

splits = [
    (0, 0, 178),
    (1, 178, 356),
    (2, 356, 534),
    (3, 534, 713),
]

processes = []
log_files = []
print("Launching 4 parallel evaluation workers with unbuffered file logging...")
t0 = time.time()

for w_id, s_rg, e_rg in splits:
    out_jsonl = scratch_dir / f"c1_part_{w_id}.jsonl"
    log_file_path = scratch_dir / f"worker_{w_id}.log"
    log_fp = open(log_file_path, "a", encoding="utf-8")
    log_files.append(log_fp)

    cmd = [
        str(python_exe),
        "-u",
        str(worker_script),
        "--worker-id", str(w_id),
        "--start-rg", str(s_rg),
        "--end-rg", str(e_rg),
        "--parquet-path", str(parquet_file),
        "--protocol-path", str(protocol_file),
        "--checkpoint-path", str(checkpoint_file),
        "--output-jsonl", str(out_jsonl),
        "--threads", "2",
        "--batch-size", "64",
    ]
    p = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    processes.append((w_id, p))
    print(f"  Worker {w_id} spawned (PID {p.pid}): Row Groups {s_rg} to {e_rg} -> {out_jsonl.name}")

# Monitor processes
active = list(processes)
while active:
    time.sleep(10)
    still_active = []
    total_records = 0
    for w_id, p in active:
        ret = p.poll()
        part_file = scratch_dir / f"c1_part_{w_id}.jsonl"
        count = 0
        if part_file.is_file():
            with open(part_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
        total_records += count

        if ret is None:
            still_active.append((w_id, p))
        else:
            if ret != 0:
                print(f"WARNING: Worker {w_id} exited with return code {ret}. Check scratch/worker_{w_id}.log.")

    active = still_active
    elapsed = time.time() - t0
    rate = total_records / elapsed if elapsed > 0 else 0
    eta = (71237 - total_records) / rate if rate > 0 else 0
    print(f"[Progress] Total evaluated: {total_records}/71237 ({rate:.1f} samples/s, Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

# Close log file descriptors
for fp in log_files:
    fp.close()

print("All workers completed! Running aggregator...")
subprocess.run([str(python_exe), str(aggregator_script)], check=True)
print(f"Evaluation pipeline completed in {time.time() - t0:.1f}s!")
