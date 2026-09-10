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
from ._pseudo_obs import to_pseudo_obs
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
    "GumbelCopula",
    "HierarchicalArchimedeanCopula",
    "InternalError",
    "InvalidInputError",
    "ModelFitError",
    "NonPrefixConditioningError",
    "NumericalError",
    "PairCopula",
    "RscopulasError",
    "StudentTCopula",
    "VineCopula",
    "VineEdgeInfo",
    "VineStructureInfo",
    "VineTreeInfo",
    "to_pseudo_obs",
]
