"""FlagMRL with pairwise nested trace ratio for the flag update only."""

from .loss import FlagMRLLoss


class FlagMRLTraceRatioLoss(FlagMRLLoss):
    """Share the encoder/MRL path; replace only the flag objective."""

    flag_objective = "trace_ratio"
