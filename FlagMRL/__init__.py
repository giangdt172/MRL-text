"""FlagMRL for text embeddings; no dependency on the VLM project."""

from .loss import FlagMRLLoss, FlagProjector
from .flag_mrl_trace_ratio import FlagMRLTraceRatioLoss

LOSS_TYPES = {"flag_mrl": FlagMRLLoss, "flag_mrl_trace_ratio": FlagMRLTraceRatioLoss}

__all__ = ["FlagMRLLoss", "FlagMRLTraceRatioLoss", "FlagProjector", "LOSS_TYPES"]
