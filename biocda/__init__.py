"""BioCDA: Biological-Context-Guided Drug Response Prediction."""

from biocda.models.biocda_model import BioCDA
from biocda.models.model_factory import build_model
from biocda.models.outputs import BioCDAOutput

__all__ = ["BioCDA", "BioCDAOutput", "build_model"]

__version__ = "biocda-xa-v1"
