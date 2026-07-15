
import json
from pathlib import Path

from tools.round19_stage19f_manifest_builder import build_checkpoint_inventory


ROOT = Path("result/optimization_runs/round19_factorial")


def test_proposal_inventory_has_all_unique_roles_and_15_members():
    proposal = json.loads((ROOT / "reports" / "round19_final_role_proposal.json").read_text())
    inventory = build_checkpoint_inventory(
        ROOT / "manifests" / "stage19d_manifest.csv",
        proposal,
    )
    assert inventory["source_candidate_id"].nunique() == 6
    assert len(inventory) == 90
    assert (inventory.groupby("source_candidate_id").size() == 15).all()
    assert inventory["checkpoint_path"].map(lambda p: Path(p).is_file()).all()
