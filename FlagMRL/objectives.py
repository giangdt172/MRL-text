from . import spherical, trace_ratio


FLAG_OBJECTIVES = {
    "trace_ratio": trace_ratio.loss_and_grad,
    "spherical": spherical.loss_and_grad,
}


def get_flag_objective(name):
    try:
        return FLAG_OBJECTIVES[name]
    except KeyError:
        raise ValueError(
            f"Unknown flag objective {name!r}; choose from {', '.join(FLAG_OBJECTIVES)}."
        ) from None
