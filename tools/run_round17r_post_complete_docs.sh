#!/usr/bin/env bash
# Wait for Round 17R C/D/F pipeline, refresh docs, and commit results.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

LOG="${LOG:-logs/round17r_17r_cdf.log}"
ROUND17R_ROOT="${ROUND17R_ROOT:-result/optimization_runs/round17r_18class}"

echo "[round17r-watcher] waiting for pipeline in ${LOG} ..."
while [[ ! -f "${LOG}" ]] || ! grep -q "ROUND17R STAGES 17R-C/D/F ALL DONE" "${LOG}"; do
  if [[ -f "${LOG}" ]] && grep -qE "stage-fail|ERROR: missing" "${LOG}"; then
    echo "[round17r-watcher] pipeline failed; see ${LOG}"
    exit 1
  fi
  sleep 300
done

echo "[round17r-watcher] pipeline done; updating docs ..."

python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("result/optimization_runs/round17r_18class")
settings = json.loads(Path("config/round17r_18class_focused_settings.json").read_text())
r13 = settings["references"]["round13_best"]

def load_top(stage):
    p = root / f"reports_stage17r_{stage}" / "round17r_top_candidates.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df if len(df) else None

def best_row(df):
    return df.sort_values("Average_TCGA_AUC_mean", ascending=False).iloc[0]

def manifest_status(stage):
    p = root / "manifests" / f"stage17r_{stage}_finetune_dispatch_manifest.csv"
    if not p.is_file():
        return None
    s = pd.read_csv(p)["status"].value_counts().to_dict()
    total = sum(s.values())
    ok = s.get("success", 0)
    return ok, total, s

lines = []
lines.append("# Round 17R Final Report（18-class-clean）\n")
lines.append(f"**Run:** `{root}`  ")
lines.append("**Pipeline:** Stage 17R-A → 17R-B → 17R-C → 17R-D → 17R-F  ")
lines.append("**Status:** ALL_DONE\n")

lines.append("## 整體完成度\n")
lines.append("| Stage | 狀態 |")
lines.append("|-------|------|")
for stage, label, key in [
    ("a", "17R-A feature smoke", None),
    ("b", "17R-B focused finetune", "b"),
    ("c", "17R-C hyperparameter refine", "c"),
    ("d", "17R-D 10-seed confirm", "d"),
    ("f", "17R-F prototype tSNE", None),
]:
    if key:
        st = manifest_status(key)
        if st:
            ok, total, counts = st
            lines.append(f"| {label} | ✅ **{ok}/{total}** |")
        else:
            lines.append(f"| {label} | ✅ 完成 |")
    else:
        if stage == "f":
            viz = root / "visualizations" / "prototype_tsne" / "r13_exp_008" / "prototype_tsne_coordinates.csv"
            lines.append(f"| {label} | {'✅ 完成' if viz.is_file() else '⏳ 待確認'} |")
        else:
            lines.append(f"| {label} | ✅ 完成 |")

for stage in ("b", "c", "d"):
    df = load_top(stage)
    if df is None:
        continue
    row = best_row(df)
    lines.append(f"\n## 17R-{stage.upper()} 最佳\n")
    lines.append(f"- model: `{row['model_id']}` / `{row['feature_mode']}`")
    lines.append(f"- Average_TCGA_AUC: **{row['Average_TCGA_AUC_mean']:.4f}**")
    if "Average_TCGA_AUC_std" in row and row["Average_TCGA_AUC_std"] > 0:
        lines.append(f"- std: ±{row['Average_TCGA_AUC_std']:.4f}")
    lines.append(f"- vs Round 13 ({r13:.4f}): **{row['Average_TCGA_AUC_mean'] - r13:+.4f}**")

df_d = load_top("d")
if df_d is not None:
    lines.append("\n## 17R-D Top-5（10-seed confirm）\n")
    lines.append("| Rank | Model | feature_mode | AUC mean ± std |")
    lines.append("|------|-------|--------------|----------------|")
    for i, (_, r) in enumerate(df_d.sort_values("Average_TCGA_AUC_mean", ascending=False).head(5).iterrows(), 1):
        std = r.get("Average_TCGA_AUC_std", 0) or 0
        lines.append(
            f"| {i} | `{r['model_key']}` | `{r['feature_mode']}` | {r['Average_TCGA_AUC_mean']:.4f} ± {std:.4f} |"
        )

lines.append("\n## 參考\n")
lines.append("- `docs/round17r_18class_dataset_sample_usage.md`")
lines.append("- `docs/round17_final_report.md`")
lines.append("\n---\n*Auto-updated after 17R-C/D/F pipeline completion.*\n")

Path("docs/round17r_18class_final_report.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote docs/round17r_18class_final_report.md")
PY

# Patch round17_final_report.md Round 17R table to ALL_DONE
python3 - <<'PY'
from pathlib import Path
import re
p = Path("docs/round17_final_report.md")
text = p.read_text(encoding="utf-8")
text = re.sub(
    r"\| 17R-C refine \| ⏳ 待跑.*\n\| 17R-D 10-seed confirm \| ⏳ 待跑.*\n\| 17R-F tSNE \| ⏳ 待跑.*\n",
    "| 17R-C refine | ✅ 完成 |\n| 17R-D 10-seed confirm | ✅ 完成 |\n| 17R-F tSNE | ✅ 完成 |\n",
    text,
)
p.write_text(text, encoding="utf-8")
print("patched docs/round17_final_report.md")
PY

git add docs/round17r_18class_final_report.md docs/round17_final_report.md \
  tools/run_round17r_stage17r_c_refine.sh \
  tools/run_round17r_stage17r_d_confirm.sh \
  tools/run_round17r_stage17r_cdf_pipeline.sh \
  tools/run_round17r_post_complete_docs.sh 2>/dev/null || true

git add docs/round17r_18class_final_report.md docs/round17_final_report.md \
  tools/run_round17r_stage17r_*.sh

if git diff --cached --quiet; then
  echo "[round17r-watcher] nothing to commit"
else
  git commit -m "$(cat <<'EOF'
Complete Round 17R stages 17R-C/D/F and refresh final reports.

Run hyperparameter refinement and 10-seed confirmation on 18-class-clean
candidates, export prototype tSNE, and auto-update consolidated documentation.
EOF
)"
  echo "[round17r-watcher] committed"
fi

echo "[round17r-watcher] done"
