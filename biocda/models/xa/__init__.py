"""BioCDA-XA v2 package."""
from biocda.models.xa.factory import build_model, build_xa_v2
from biocda.models.xa.model import BioCDAXA

__all__ = ["BioCDAXA", "build_model", "build_xa_v2"]
