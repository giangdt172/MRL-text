"""FlagMRL for text embeddings; no dependency on the VLM project."""

from .loss import FlagMRLLoss, FlagProjector
from .objectives import FLAG_OBJECTIVES

__all__ = ["FlagMRLLoss", "FlagProjector", "FLAG_OBJECTIVES"]
