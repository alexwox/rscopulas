from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "plot_density",
    "plot_mean_excess",
    "plot_scatter",
    "plot_tail_concentration",
    "plot_tail_dependence",
    "plot_threshold_stability",
    "plot_vine_structure",
]


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised in environments without matplotlib
        raise ImportError(
            "matplotlib is required for rscopulas plotting support. "
            "Install it with `uv add 'rscopulas[viz]'` (in a uv project), or `uv pip install 'rscopulas[viz]'` / `matplotlib`."
        ) from exc
    return plt


def _resolve_ax(ax: Any):
    plt = _import_pyplot()
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def _as_float_matrix(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expected a 2D array")
    return array


def _validate_dims(dims: tuple[int, int], n_columns: int) -> tuple[int, int]:
    first = int(dims[0])
    second = int(dims[1])
    if first == second:
        raise ValueError("dims must reference two distinct columns")
    if first < 0 or second < 0 or first >= n_columns or second >= n_columns:
        raise ValueError("dims are out of bounds for the provided sample")
    return first, second


def _pretty_family_name(model: Any) -> str:
    family = str(getattr(model, "family", "copula"))
    return family.replace("_", " ").title()


def _vine_summary_lines(vine_model: Any, *, max_lines: int = 10) -> list[str]:
    lines: list[str] = []
    for tree in getattr(vine_model, "trees", []):
        lines.append(f"Tree {tree.level}")
        for edge in tree.edges:
            params = ", ".join(f"{value:.2f}" for value in edge.parameters) or "-"
            cond = ",".join(str(value) for value in edge.conditioning) or "-"
            lines.append(
                f"  {edge.conditioned[0]}-{edge.conditioned[1]} | {cond}  "
                f"{edge.family} {edge.rotation}  [{params}]"
            )
            if len(lines) >= max_lines:
                lines.append("  ...")
                return lines
    return lines


def plot_density(
    model: Any,
    *,
    ax: Any = None,
    grid_size: int = 100,
    levels: int = 10,
    clip_eps: float = 1e-12,
    cmap: str = "viridis",
):
    """Plot a bivariate copula density using the model's public API."""
    if not hasattr(model, "dim") or not hasattr(model, "log_pdf"):
        raise TypeError("plot_density() expects a copula model with `dim` and `log_pdf`.")
    if int(model.dim) != 2:
        raise ValueError("plot_density() only supports bivariate copulas with dim == 2.")

    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    axis = np.linspace(0.01, 0.99, grid_size, dtype=np.float64)
    u1, u2 = np.meshgrid(axis, axis)
    points = np.column_stack([u1.ravel(), u2.ravel()])
    from ._models import PairCopula

    log_density = (
        model.log_pdf(points[:, 0], points[:, 1], clip_eps=clip_eps)
        if isinstance(model, PairCopula)
        else model.log_pdf(points, clip_eps=clip_eps)
    )
    density = np.exp(log_density).reshape(grid_size, grid_size)

    fig, ax = _resolve_ax(ax)
    filled = ax.contourf(u1, u2, density, levels=max(int(levels), 2), cmap=cmap)
    ax.contour(u1, u2, density, levels=max(int(levels) // 2, 2), colors="white", linewidths=0.5)
    ax.set_title(f"{_pretty_family_name(model)} density")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(filled, ax=ax, shrink=0.85, label="density")
    return ax


def plot_scatter(
    *,
    sample: npt.ArrayLike | None = None,
    model: Any = None,
    n: int = 500,
    seed: int | None = None,
    ax: Any = None,
    alpha: float = 0.4,
    s: float = 12,
    dims: tuple[int, int] = (0, 1),
):
    """Plot pseudo-observations directly or sample from a model first."""
    if sample is None and model is None:
        raise ValueError("plot_scatter() requires either `sample` or `model`.")
    if sample is not None and model is not None:
        raise ValueError("plot_scatter() accepts either `sample` or `model`, not both.")

    values = _as_float_matrix(sample if sample is not None else model.sample(int(n), seed=seed))
    if values.shape[1] < 2:
        raise ValueError("scatter plots require at least two columns.")

    first, second = _validate_dims(dims, values.shape[1])
    fig, ax = _resolve_ax(ax)
    color = "#d62728" if model is not None else "#1f77b4"
    ax.scatter(values[:, first], values[:, second], s=s, alpha=alpha, color=color, edgecolors="none")
    ax.set_xlabel(f"u{first + 1}")
    ax.set_ylabel(f"u{second + 1}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    if model is not None:
        ax.set_title(f"{_pretty_family_name(model)} samples")
    else:
        ax.set_title("Observed pseudo-observations")
    return ax


def plot_vine_structure(vine_model: Any, *, ax: Any = None, annotate: bool = True):
    """Plot the structure matrix of a fitted vine model."""
    if not hasattr(vine_model, "structure_info") or not hasattr(vine_model, "trees"):
        raise TypeError("plot_vine_structure() expects a vine model with `structure_info` and `trees`.")

    structure_info = vine_model.structure_info
    matrix = np.asarray(structure_info.matrix)
    if matrix.ndim != 2:
        raise ValueError("vine structure matrices must be 2D")

    fig, ax = _resolve_ax(ax)
    image = ax.imshow(matrix, cmap="magma")
    kind = str(getattr(vine_model, "structure_kind", structure_info.kind)).upper()
    ax.set_title(f"{kind}-vine structure matrix")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="white")

        summary_lines = _vine_summary_lines(vine_model)
        if summary_lines:
            ax.text(
                1.05,
                1.0,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, shrink=0.85, label="node id")
    return ax


def plot_mean_excess(x: npt.ArrayLike, *, thresholds: npt.ArrayLike | None = None, ax: Any = None):
    """Mean residual life plot for choosing a generalized Pareto threshold.

    Look for the lowest threshold from which the curve is straight within the
    band (two standard errors). Above it the excesses can be modelled as
    generalized Pareto. The slope shows the tail type: rising for a heavy
    tail, flat for an exponential one, falling for a bounded one.
    """
    from ._margins import mean_excess

    result = mean_excess(x, thresholds)
    fig, ax = _resolve_ax(ax)
    ax.plot(result.threshold, result.mean_excess, color="#1f77b4")
    ax.fill_between(
        result.threshold,
        result.mean_excess - 2.0 * result.std_error,
        result.mean_excess + 2.0 * result.std_error,
        color="#1f77b4",
        alpha=0.2,
        label="+- 2 standard errors",
    )
    ax.set_xlabel("threshold")
    ax.set_ylabel("mean excess")
    ax.set_title("Mean residual life")
    ax.legend(loc="best")
    return ax


def plot_threshold_stability(
    x: npt.ArrayLike, *, thresholds: npt.ArrayLike | None = None, axes: Any = None
):
    """Generalized Pareto shape and modified scale against the threshold.

    Above a valid threshold both panels are flat within their bands (two
    standard errors). Returns the two axes ``(shape, modified scale)``.
    """
    from ._margins import threshold_stability

    result = threshold_stability(x, thresholds)
    if axes is None:
        plt = _import_pyplot()
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    shape_ax, scale_ax = axes
    for ax, values, errors, label in (
        (shape_ax, result.shape, result.shape_std_error, "shape"),
        (scale_ax, result.modified_scale, result.modified_scale_std_error, "modified scale"),
    ):
        ax.plot(result.threshold, values, color="#1f77b4")
        ax.fill_between(
            result.threshold,
            values - 2.0 * errors,
            values + 2.0 * errors,
            color="#1f77b4",
            alpha=0.2,
        )
        ax.set_xlabel("threshold")
        ax.set_ylabel(label)
        ax.set_title(f"Stability of the {label}")
    return shape_ax, scale_ax


def _pair_columns(
    data: npt.ArrayLike, dims: tuple[int, int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int, int]:
    values = _as_float_matrix(data)
    if values.shape[1] < 2:
        raise ValueError("tail plots require at least two columns.")
    first, second = _validate_dims(dims, values.shape[1])
    return (
        np.ascontiguousarray(values[:, first]),
        np.ascontiguousarray(values[:, second]),
        first,
        second,
    )


def plot_tail_concentration(
    data: npt.ArrayLike,
    *,
    model: Any = None,
    dims: tuple[int, int] = (0, 1),
    q: npt.ArrayLike | None = None,
    ax: Any = None,
):
    """Plot the empirical tail concentration function of two columns.

    The left half shows ``P(U2 <= q | U1 <= q)`` and the right half
    ``P(U2 > q | U1 > q)``; the ends of the curve approach the lower and
    upper tail-dependence coefficients. Pass a :class:`rscopulas.PairCopula`
    as ``model`` to overlay what the fitted family implies. A model curve
    that leaves the data at either end is the wrong family for that tail.
    """
    from ._tail import default_tail_levels, empirical_tail_concentration

    first_values, second_values, first, second = _pair_columns(data, dims)
    levels = default_tail_levels() if q is None else np.asarray(q, dtype=np.float64)
    observed = empirical_tail_concentration(first_values, second_values, q=levels)

    fig, ax = _resolve_ax(ax)
    ax.plot(observed.q, observed.values, color="#1f77b4", label="empirical")
    if model is not None:
        if not hasattr(model, "tail_concentration"):
            raise TypeError("plot_tail_concentration() expects `model` to be a PairCopula.")
        ax.plot(
            levels,
            model.tail_concentration(levels),
            color="#d62728",
            linestyle="--",
            label=_pretty_family_name(model),
        )
    ax.axvline(0.5, color="0.6", linewidth=0.8)
    ax.set_xlabel("q")
    ax.set_ylabel("tail concentration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Tail concentration of u{first + 1}, u{second + 1}")
    ax.legend(loc="lower center")
    return ax


def plot_tail_dependence(
    data: npt.ArrayLike,
    *,
    model: Any = None,
    dims: tuple[int, int] = (0, 1),
    k: npt.ArrayLike | None = None,
    ax: Any = None,
):
    """Plot the tail-dependence estimates against the number of tail points ``k``.

    Use it to choose ``k``: the estimates are noisy on the left and biased
    toward the body of the distribution on the right, and a flat stretch in
    between is the value to read. The dotted line marks the default
    ``k = floor(sqrt(n))``. Pass a :class:`rscopulas.PairCopula` as ``model``
    to draw its closed-form coefficients as horizontal lines.
    """
    from ._tail import default_tail_k, empirical_tail_dependence_curve

    first_values, second_values, first, second = _pair_columns(data, dims)
    curve = empirical_tail_dependence_curve(first_values, second_values, k=k)

    fig, ax = _resolve_ax(ax)
    ax.plot(curve.k, curve.lower, color="#1f77b4", label="lower")
    ax.plot(curve.k, curve.upper, color="#ff7f0e", label="upper")
    if model is not None:
        if not hasattr(model, "tail_dependence"):
            raise TypeError("plot_tail_dependence() expects `model` to be a PairCopula.")
        implied = model.tail_dependence()
        if implied is not None and hasattr(implied, "lower") and np.ndim(implied.lower) == 0:
            ax.axhline(implied.lower, color="#1f77b4", linestyle="--", label="model lower")
            ax.axhline(implied.upper, color="#ff7f0e", linestyle="--", label="model upper")
    ax.axvline(default_tail_k(first_values.shape[0]), color="0.5", linestyle=":", label="sqrt(n)")
    ax.set_xlabel("k (observations in each tail)")
    ax.set_ylabel("tail dependence")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Tail dependence of u{first + 1}, u{second + 1}")
    ax.legend(loc="best")
    return ax
