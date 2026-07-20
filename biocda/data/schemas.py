"""Graph batch schema constants."""
from __future__ import annotations

REQUIRED_GRAPH_FIELDS = ("x", "edge_index")
OPTIONAL_ATOM_INDEX_FIELDS = (
    "model_atom_index",
    "original_atom_index",
    "rdkit_atom_index",
)
GRAPH_METADATA_FIELDS = ("drug_id",)
