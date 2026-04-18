"""Generate one figure per public copula model type (saved as PNG).

Run from repo root (after `maturin develop` and `pip install -e ".[viz]"`):

    PYTHONPATH=python python python/examples/copula_gallery.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rscopulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    HierarchicalArchimedeanCopula,
    PairCopula,
    StudentTCopula,
    VineCopula,
)
from rscopulas.plotting import plot_density, plot_scatter, plot_vine_structure


def output_dir() -> Path:
    path = Path(__file__).resolve().parent / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_density_scatter(
    model,
    suptitle: str,
    filename: str,
    *,
    n_sample: int = 900,
    seed: int = 42,
    grid_size: int = 96,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="semibold")

    plot_density(model, ax=axes[0], grid_size=grid_size, levels=16, cmap="viridis")
    axes[0].set_title("Density")

    plot_scatter(model=model, n=n_sample, seed=seed, ax=axes[1], alpha=0.35, s=11)
    axes[1].set_title("Samples")

    path = output_dir() / filename
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_vine_example() -> Path:
    corr = np.array(
        [
            [1.0, 0.62, 0.38],
            [0.62, 1.0, 0.28],
            [0.38, 0.28, 1.0],
        ],
        dtype=np.float64,
    )
    vine = VineCopula.gaussian_c_vine([0, 1, 2], corr)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    fig.suptitle("Vine copula (Gaussian C-vine, d=3)", fontsize=14, fontweight="semibold")

    plot_vine_structure(vine, ax=axes[0], annotate=True)
    axes[0].set_title("Structure matrix")

    plot_scatter(model=vine, n=800, seed=7, ax=axes[1], dims=(0, 1), alpha=0.38, s=12)
    axes[1].set_title(r"Samples: $(u_1, u_2)$")

    path = output_dir() / "gallery_vine.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_hac_example() -> Path:
    tree: dict = {
        "family": "gumbel",
        "theta": 1.35,
        "children": [
            0,
            1,
            {"family": "clayton", "theta": 1.8, "children": [2, 3]},
        ],
    }
    model = HierarchicalArchimedeanCopula.from_tree(tree)
    sample = model.sample(1000, seed=19)

    fig = plt.figure(figsize=(11.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.45])
    fig.suptitle("Hierarchical Archimedean copula", fontsize=14, fontweight="semibold")

    ax01 = fig.add_subplot(gs[0, 0])
    plot_scatter(sample=sample, ax=ax01, alpha=0.4, s=12)
    ax01.set_title(r"Pseudo-obs: $(u_1, u_2)$")

    ax02 = fig.add_subplot(gs[0, 1])
    plot_scatter(sample=sample, ax=ax02, dims=(0, 2), alpha=0.4, s=12)
    ax02.set_title(r"Pseudo-obs: $(u_1, u_3)$")

    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    summary = [
        f"dim = {model.dim}",
        f"leaf_order = {model.leaf_order}",
        f"families = {model.families}",
        f"parameters = {[round(p, 4) for p in model.parameters]}",
        "",
        "tree JSON:",
        json.dumps(model.tree, indent=2) if isinstance(model.tree, dict) else repr(model.tree),
    ]
    ax_txt.text(
        0.0,
        1.0,
        "\n".join(summary),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax_txt.transAxes,
    )

    path = output_dir() / "gallery_hierarchical_archimedean.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_pair_copula_example() -> Path:
    pair = PairCopula.from_spec("student_t", [0.55, 5.0])
    grid_size = 100
    axis = np.linspace(0.02, 0.98, grid_size, dtype=np.float64)
    u1, u2 = np.meshgrid(axis, axis)
    log_pdf = pair.log_pdf(u1.ravel(), u2.ravel())
    density = np.exp(log_pdf).reshape(grid_size, grid_size)

    # PairCopula has no `sample()`; use the matching bivariate Student t copula for a comparable scatter.
    ref = StudentTCopula.from_params(
        np.array([[1.0, 0.55], [0.55, 1.0]], dtype=np.float64),
        5.0,
    )
    scatter_u = ref.sample(700, seed=23)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    fig.suptitle(
        r"Pair-copula kernel: Student $t$ ($\rho=0.55$, $\nu=5$)",
        fontsize=14,
        fontweight="semibold",
    )

    filled = axes[0].contourf(u1, u2, density, levels=18, cmap="magma")
    axes[0].contour(u1, u2, density, levels=10, colors="white", linewidths=0.45, alpha=0.55)
    axes[0].set_title("Pair copula density")
    axes[0].set_xlabel("u1")
    axes[0].set_ylabel("u2")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(filled, ax=axes[0], shrink=0.85, label="density")

    axes[1].scatter(
        scatter_u[:, 0],
        scatter_u[:, 1],
        s=11,
        alpha=0.38,
        c="#2ca02c",
        edgecolors="none",
    )
    axes[1].set_title(r"Bivariate Student $t$ samples (same $\rho$, $\nu$)")
    axes[1].set_xlabel("u1")
    axes[1].set_ylabel("u2")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal", adjustable="box")

    path = output_dir() / "gallery_pair_copula.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    saved: list[Path] = []

    g = GaussianCopula.from_params(np.array([[1.0, 0.68], [0.68, 1.0]], dtype=np.float64))
    saved.append(save_density_scatter(g, "Gaussian copula", "gallery_gaussian.png", seed=1))

    st = StudentTCopula.from_params(
        np.array([[1.0, 0.52], [0.52, 1.0]], dtype=np.float64),
        6.0,
    )
    saved.append(save_density_scatter(st, "Student t copula", "gallery_student_t.png", seed=2))

    cl = ClaytonCopula.from_params(2, 2.2)
    saved.append(save_density_scatter(cl, "Clayton copula", "gallery_clayton.png", seed=3))

    fr = FrankCopula.from_params(2, 5.5)
    saved.append(save_density_scatter(fr, "Frank copula", "gallery_frank.png", seed=4))

    gu = GumbelCopula.from_params(2, 2.1)
    saved.append(save_density_scatter(gu, "Gumbel copula", "gallery_gumbel.png", seed=5))

    saved.append(save_vine_example())
    saved.append(save_hac_example())
    saved.append(save_pair_copula_example())

    print("Gallery figures written:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
