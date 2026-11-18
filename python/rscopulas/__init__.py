"""Copula modeling on validated pseudo-observations, backed by a Rust core.

Compute-bound calls (fitting, ``log_pdf``, sampling, Rosenblatt transforms)
release the GIL, so other Python threads keep running while Rust works.
Every model supports ``to_json``/``from_json``, ``pickle``, ``copy``,
``==``, and ``repr``.
"""

from importlib.metadata import PackageNotFoundError, version as _package_version

from ._models import (
    ClaytonCopula,
    FactorCopula,
    FactorFitDiagnostics,
    FactorFitResult,
    FitDiagnostics,
    FitResult,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    HierarchicalArchimedeanCopula,
    PairCopula,
    StudentTCopula,
    VineCopula,
    VineEdgeInfo,
    VineStructureInfo,
    VineTreeInfo,
)
from ._margins import (
    GeneralizedExtremeValue,
    GeneralizedPareto,
    GevFit,
    GpdFit,
    GpdTail,
    Margins,
    MeanExcess,
    SemiParametricMargin,
    ThresholdStability,
    block_maxima,
    mean_excess,
    threshold_stability,
)
from ._pseudo_obs import to_pseudo_obs
from ._tail import (
    TailConcentration,
    TailDependence,
    TailDependenceCurve,
    TailDependenceMatrix,
    default_tail_k,
    empirical_tail_concentration,
    empirical_tail_dependence,
    empirical_tail_dependence_curve,
    empirical_tail_dependence_matrix,
)
from ._rscopulas import (
    BackendError,
    InternalError,
    InvalidInputError,
    ModelFitError,
    NonPrefixConditioningError,
    NumericalError,
    RscopulasError,
)
from ._rscopulas import __version__ as _extension_version

try:
    __version__ = _package_version("rscopulas")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    # Fall back to the version compiled into the extension module.
    __version__ = str(_extension_version)

__all__ = [
    "BackendError",
    "ClaytonCopula",
    "FactorCopula",
    "FactorFitDiagnostics",
    "FactorFitResult",
    "FitDiagnostics",
    "FitResult",
    "FrankCopula",
    "GaussianCopula",
    "GeneralizedExtremeValue",
    "GeneralizedPareto",
    "GevFit",
    "GpdFit",
    "GpdTail",
    "GumbelCopula",
    "HierarchicalArchimedeanCopula",
    "InternalError",
    "InvalidInputError",
    "Margins",
    "MeanExcess",
    "ModelFitError",
    "NonPrefixConditioningError",
    "NumericalError",
    "PairCopula",
    "RscopulasError",
    "SemiParametricMargin",
    "StudentTCopula",
    "TailConcentration",
    "TailDependence",
    "TailDependenceCurve",
    "TailDependenceMatrix",
    "ThresholdStability",
    "VineCopula",
    "VineEdgeInfo",
    "VineStructureInfo",
    "VineTreeInfo",
    "block_maxima",
    "default_tail_k",
    "empirical_tail_concentration",
    "empirical_tail_dependence",
    "empirical_tail_dependence_curve",
    "empirical_tail_dependence_matrix",
    "mean_excess",
    "threshold_stability",
    "to_pseudo_obs",
]
