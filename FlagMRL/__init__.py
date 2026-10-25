"""Spherical FlagMRL for text embeddings; no dependency on the VLM project."""

from .loss import FlagMRLLoss, FlagProjector

__all__ = ["FlagMRLLoss", "FlagProjector"]
