"""Chunked fused LM-head + cross-entropy."""

from .loss import fused_linear_cross_entropy, naive_linear_cross_entropy

__all__ = ["fused_linear_cross_entropy", "naive_linear_cross_entropy"]
