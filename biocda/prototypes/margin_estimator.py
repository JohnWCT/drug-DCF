"""Source natural-radius margin estimator (Round 25).

prototype_upper_margin = P90(source minibatch centroid → EMA anchor distance)
prototype_lower_margin (S2b only) = P10 of the same distribution

Margins are estimated during warm-up, then frozen + hashed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


@dataclass
class MarginArtifact:
    metric: str
    estimator: str
    upper_percentile: float
    lower_percentile: Optional[float]
    minimum_cancer_observations: int
    fallback: str
    per_cancer_upper: Dict[str, float]
    per_cancer_lower: Dict[str, float]
    global_upper: float
    global_lower: Optional[float]
    n_observations: Dict[str, int]
    frozen: bool = True

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = asdict(self)
        blob["sha256"] = self.sha256()
        path.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return blob["sha256"]


class SourceRadiusMarginEstimator:
    """Collects source batch-centroid → EMA-anchor radii during warm-up."""

    def __init__(
        self,
        num_cancer_types: int,
        *,
        metric: str = "cosine",
        upper_percentile: float = 90.0,
        lower_percentile: Optional[float] = None,
        minimum_cancer_observations: int = 20,
        fallback: str = "global_median",
        cancer_names: Optional[Dict[int, str]] = None,
    ):
        self.num_cancer_types = int(num_cancer_types)
        self.metric = str(metric).lower()
        self.upper_percentile = float(upper_percentile)
        self.lower_percentile = (
            None if lower_percentile is None else float(lower_percentile)
        )
        self.minimum_cancer_observations = int(minimum_cancer_observations)
        self.fallback = str(fallback)
        self.cancer_names = cancer_names or {
            i: str(i) for i in range(self.num_cancer_types)
        }
        self._radii: Dict[int, List[float]] = {
            i: [] for i in range(self.num_cancer_types)
        }
        self._frozen_artifact: Optional[MarginArtifact] = None

    @torch.no_grad()
    def observe_batch(
        self,
        source_z: torch.Tensor,
        source_cancer_ids: torch.Tensor,
        source_anchors: torch.Tensor,
        initialized_mask: torch.Tensor,
        *,
        min_count: int = 2,
    ) -> None:
        if self._frozen_artifact is not None:
            raise RuntimeError("margin estimator is frozen; do not observe during formal training")
        source_z = source_z.detach()
        source_cancer_ids = source_cancer_ids.long().detach()
        source_anchors = source_anchors.detach()

        for class_id in range(self.num_cancer_types):
            if not bool(initialized_mask[class_id].item()):
                continue
            mask = source_cancer_ids == class_id
            n = int(mask.sum().item())
            if n < int(min_count):
                continue
            centroid = source_z[mask].mean(dim=0, keepdim=True)
            anchor = source_anchors[class_id].unsqueeze(0)
            if self.metric == "cosine":
                dist = float(_cosine_distance(centroid, anchor).item())
            else:
                dist = float((centroid - anchor).norm(p=2).item())
            self._radii[class_id].append(dist)

    def freeze(self) -> MarginArtifact:
        if self._frozen_artifact is not None:
            return self._frozen_artifact

        all_radii: List[float] = []
        for vals in self._radii.values():
            all_radii.extend(vals)
        if not all_radii:
            raise ValueError("no source radius observations collected during warm-up")

        all_t = torch.tensor(all_radii, dtype=torch.float64)
        if self.fallback == "global_median":
            global_upper = float(torch.median(all_t).item())
        else:
            global_upper = float(
                torch.quantile(all_t, self.upper_percentile / 100.0).item()
            )
        global_lower = None
        if self.lower_percentile is not None:
            global_lower = float(
                torch.quantile(all_t, self.lower_percentile / 100.0).item()
            )

        per_upper: Dict[str, float] = {}
        per_lower: Dict[str, float] = {}
        n_obs: Dict[str, int] = {}
        for class_id, vals in self._radii.items():
            name = self.cancer_names.get(class_id, str(class_id))
            n_obs[name] = len(vals)
            if len(vals) < self.minimum_cancer_observations:
                per_upper[name] = global_upper
                if global_lower is not None:
                    per_lower[name] = global_lower
                continue
            t = torch.tensor(vals, dtype=torch.float64)
            per_upper[name] = float(
                torch.quantile(t, self.upper_percentile / 100.0).item()
            )
            if self.lower_percentile is not None:
                per_lower[name] = float(
                    torch.quantile(t, self.lower_percentile / 100.0).item()
                )

        art = MarginArtifact(
            metric=self.metric,
            estimator="source_radius_percentile",
            upper_percentile=self.upper_percentile,
            lower_percentile=self.lower_percentile,
            minimum_cancer_observations=self.minimum_cancer_observations,
            fallback=self.fallback,
            per_cancer_upper=per_upper,
            per_cancer_lower=per_lower,
            global_upper=global_upper,
            global_lower=global_lower,
            n_observations=n_obs,
            frozen=True,
        )
        self._frozen_artifact = art
        return art

    def margins_tensor(
        self,
        cancer_ids: List[int],
        *,
        which: str = "upper",
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        art = self.freeze()
        vals = []
        for cid in cancer_ids:
            name = self.cancer_names.get(cid, str(cid))
            if which == "upper":
                vals.append(art.per_cancer_upper.get(name, art.global_upper))
            else:
                if art.global_lower is None:
                    raise ValueError("lower margins were not estimated")
                vals.append(art.per_cancer_lower.get(name, art.global_lower))
        return torch.tensor(vals, device=device, dtype=dtype)
