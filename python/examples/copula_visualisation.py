import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rscopulas import GaussianCopula, VineCopula
from rscopulas.plotting import plot_density, plot_scatter, plot_vine_structure


def normal_cdf(values: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def make_pseudo_observations(seed: int = 7, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    correlation = np.array(
        [
            [1.0, 0.72, 0.35],
            [0.72, 1.0, 0.55],
            [0.35, 0.55, 1.0],
        ],
        dtype=np.float64,
    )
    latent = rng.multivariate_normal(np.zeros(3), correlation, size=n)
    uniforms = normal_cdf(latent)
    return np.clip(uniforms, 1e-6, 1.0 - 1e-6)


def format_tree_lines(vine: VineCopula) -> list[str]:
    lines = []
    for tree in vine.trees:
        lines.append(f"Tree {tree.level}")
        for edge in tree.edges:
            params = ", ".join(f"{value:.2f}" for value in edge.parameters) or "-"
            cond = ",".join(str(value) for value in edge.conditioning) or "-"
            lines.append(
                "  "
                f"{edge.conditioned[0]}-{edge.conditioned[1]} | {cond}  "
                f"{edge.family} {edge.rotation}  params=[{params}]"
            )
    return lines


def main() -> None:
    data = make_pseudo_observations()

    gaussian_fit = GaussianCopula.fit(data[:, :2])
    gaussian_model = gaussian_fit.model

    vine_fit = VineCopula.fit_r(
        data,
        family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
        include_rotations=True,
        criterion="aic",
        truncation_level=1,
    )
    vine_model = vine_fit.model

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax_obs = fig.add_subplot(gs[0, 0])
    plot_scatter(sample=data[:, :2], ax=ax_obs, alpha=0.65, s=14)

    ax_sample = fig.add_subplot(gs[0, 1])
    plot_scatter(model=gaussian_model, n=1200, seed=11, ax=ax_sample, alpha=0.35, s=12)

    ax_density = fig.add_subplot(gs[0, 2])
    plot_density(gaussian_model, ax=ax_density, grid_size=140, levels=18)

    ax_corr = fig.add_subplot(gs[1, 0])
    corr_image = ax_corr.imshow(gaussian_model.correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax_corr.set_title("Fitted Gaussian correlation")
    ax_corr.set_xticks(range(2), ["u1", "u2"])
    ax_corr.set_yticks(range(2), ["u1", "u2"])
    for i in range(2):
        for j in range(2):
            ax_corr.text(j, i, f"{gaussian_model.correlation[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(corr_image, ax=ax_corr, shrink=0.82, label="rho")

    ax_vine = fig.add_subplot(gs[1, 1])
    plot_vine_structure(vine_model, ax=ax_vine, annotate=False)

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    tree_lines = "\n".join(format_tree_lines(vine_model))
    structure_matrix = vine_model.structure_info.matrix
    summary = "\n".join(
        [
            "rscopulas example",
            "",
            f"Gaussian AIC: {gaussian_fit.diagnostics.aic:.2f}",
            f"Gaussian BIC: {gaussian_fit.diagnostics.bic:.2f}",
            f"Vine AIC: {vine_fit.diagnostics.aic:.2f}",
            f"Vine BIC: {vine_fit.diagnostics.bic:.2f}",
            f"Vine order: {vine_model.order}",
            "",
            "Selected vine edges:",
            tree_lines,
        ]
    )
    ax_text.text(
        0.0,
        1.0,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.5,
    )

    fig.suptitle("Visual tour of fitted copulas with rscopulas", fontsize=17)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "copula_visualisation.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved visualisation to: {output_path}")
    print(f"Gaussian diagnostics: AIC={gaussian_fit.diagnostics.aic:.3f}, BIC={gaussian_fit.diagnostics.bic:.3f}")
    print(f"Vine diagnostics: AIC={vine_fit.diagnostics.aic:.3f}, BIC={vine_fit.diagnostics.bic:.3f}")
    print(f"Vine order: {vine_model.order}")
    print("Vine structure matrix:")
    print(structure_matrix)


if __name__ == "__main__":
    main()
