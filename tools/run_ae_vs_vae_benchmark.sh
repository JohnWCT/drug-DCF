#!/usr/bin/env bash
# Benchmark AEwC vs VAEwC (exp_746 settings) for strict filter pass rate.
set -euo pipefail
cd /workspace/DAPL

OUT="result/benchmark_ae_vs_vae_exp746"
AE_DIR="${OUT}/aewc"
LOG="${OUT}/benchmark.log"
CONFIG="config/params_benchmark_aewc_exp746.json"
MAX_PARALLEL="${AE_MAX_PARALLEL:-2}"

mkdir -p "${OUT}" "${AE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "========== AE vs VAE BENCHMARK $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
echo "AE config: ${CONFIG} | out: ${AE_DIR} | parallel=${MAX_PARALLEL}"

# Run 3 AE jobs (control + 2 mild InfoNCE) — sequential batches if parallel>1
python3 pretrain_AEwC.py \
  --config "${CONFIG}" \
  --outfolder "${AE_DIR}" \
  --target_domain tcga \
  --overlap_tcga data/TCGA/PMID27354694_DR_OMICS_ad.csv \
  --batch-size 128

echo "=== Generating AE t-SNE plots ==="
for exp_dir in "${AE_DIR}"/exp_*; do
  [ -d "${exp_dir}" ] || continue
  if [ ! -f "${exp_dir}/tsne_gan_best.png" ]; then
    python3 plot_tsne_from_latent.py --exp_dir "${exp_dir}" || true
  fi
done

echo "=== Filter comparison (strict vs relaxed) ==="
python3 tools/compare_pretrain_filter_metrics.py \
  --aewc-dir "${AE_DIR}" \
  --vaewc-ids exp_746 \
  --round3-ids exp_005 \
  --out "${OUT}/filter_comparison.csv"

python3 - <<'PY'
import json, os
from visualize_vaewc_results import load_experiment_data
from tools.compare_pretrain_filter_metrics import STRICT_FILTER, apply_quality_filter
import pandas as pd

out = "result/benchmark_ae_vs_vae_exp746"
rows = []
for label, sub in [("VAE_exp746", "result/pretrain_vaewc/exp_746"), ("VAE_round3_exp005", "result/optimization_runs/vaewc_proto_infonce_round3_exp746/pretrain/exp_005")]:
    if os.path.isdir(sub):
        r = load_experiment_data(sub)
        r["model"] = label
        rows.append(r)
for d in sorted(os.listdir(f"{out}/aewc")):
    if d.startswith("exp_"):
        r = load_experiment_data(f"{out}/aewc/{d}")
        r["model"] = f"AE_{d}"
        rows.append(r)
df = pd.DataFrame(rows)
report = {
    "strict_thresholds": STRICT_FILTER["thresholds"],
    "models": [],
}
for _, r in df.iterrows():
    sub = pd.DataFrame([r])
    strict_ok = len(apply_quality_filter(sub, STRICT_FILTER)) == 1
    report["models"].append({
        "id": r.get("model", r.get("ID")),
        "lambda_proto": float(r.get("lambda_proto", 0)),
        "fid": float(r["fid"]),
        "wasserstein": float(r["wasserstein"]),
        "kmeans_ari": float(r["kmeans_ari"]),
        "kmeans_nmi": float(r["kmeans_nmi"]),
        "kmeans_calinski_harabasz": float(r["kmeans_calinski_harabasz"]),
        "best_gan_epoch": int(r.get("best_gan_epoch", 0) or 0),
        "pass_strict_filter": strict_ok,
    })
path = f"{out}/benchmark_report.json"
with open(path, "w") as f:
    json.dump(report, f, indent=2)
print("Wrote", path)
for m in report["models"]:
    tag = "PASS" if m["pass_strict_filter"] else "FAIL"
    print(f"  [{tag}] {m['id']}: fid={m['fid']:.2f} wass={m['wasserstein']:.3f} ari={m['kmeans_ari']:.3f}")
PY

echo "========== BENCHMARK DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
