#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DAPL

mapfile -t pids < <(pgrep -f 'step1_finetune_latent_pipeline_All_split.py.*stage16e/finetune' || true)
if ((${#pids[@]})); then
  echo "Killing ${#pids[@]} stage16e finetune workers"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 5
fi

pgrep -f 'optimization_runner.py finetune.*stage16e' | xargs -r kill 2>/dev/null || true
pgrep -f 'run_round16_own_proto_context_stage16e.sh' | xargs -r kill 2>/dev/null || true
pgrep -f 'run_round16_continue_ed.sh' | xargs -r kill 2>/dev/null || true
sleep 2

python3 - <<'PY'
import pandas as pd
path = "result/optimization_runs/round16_bruteforce/manifests/stage16e_finetune_dispatch_manifest.csv"
df = pd.read_csv(path)
mask = df["status"] == "running"
n = int(mask.sum())
if n:
    df.loc[mask, "status"] = "pending"
    df.loc[mask, "start_time"] = ""
    df.to_csv(path, index=False)
    print(f"Reset {n} running -> pending")
vc = df["status"].value_counts().to_dict()
print(vc)
PY

echo "Stopped stage16e workers."
