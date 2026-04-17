from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["plot_density", "plot_scatter", "plot_vine_structure"]


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised in environments without matplotlib
        raise ImportError(
            "matplotlib is required for rscopulas plotting support. "
            "Install it with `pip install rscopulas[viz]` or `pip install matplotlib`."
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
    density = np.exp(model.log_pdf(points, clip_eps=clip_eps)).reshape(grid_size, grid_size)

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
