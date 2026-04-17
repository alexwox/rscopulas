from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from . import _rscopulas

ModelT = TypeVar("ModelT")


def _as_float_matrix(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expected a 2D array")
    return array


def _as_order(order: Sequence[int]) -> list[int]:
    return [int(value) for value in order]


def _family_set(family_set: Sequence[str] | None) -> list[str] | None:
    if family_set is None:
        return None
    return [str(family) for family in family_set]


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    loglik: float
    aic: float
    bic: float
    converged: bool
    n_iter: int

    @classmethod
    def _from_core(cls, diagnostics: Any) -> "FitDiagnostics":
        return cls(
            loglik=float(diagnostics.loglik),
            aic=float(diagnostics.aic),
            bic=float(diagnostics.bic),
            converged=bool(diagnostics.converged),
            n_iter=int(diagnostics.n_iter),
        )


@dataclass(frozen=True, slots=True)
class FitResult(Generic[ModelT]):
    model: ModelT
    diagnostics: FitDiagnostics


@dataclass(frozen=True, slots=True)
class VineStructureInfo:
    kind: str
    matrix: npt.NDArray[np.int_]
    truncation_level: int | None

    @classmethod
    def _from_core(cls, payload: dict[str, Any]) -> "VineStructureInfo":
        return cls(
            kind=str(payload["kind"]),
            matrix=np.asarray(payload["matrix"]).copy(),
            truncation_level=(
                None if payload["truncation_level"] is None else int(payload["truncation_level"])
            ),
        )


@dataclass(frozen=True, slots=True)
class VineEdgeInfo:
    tree: int
    conditioned: tuple[int, int]
    conditioning: list[int]
    family: str
    rotation: str
    parameters: tuple[float, ...]

    @classmethod
    def _from_core(cls, payload: dict[str, Any]) -> "VineEdgeInfo":
        return cls(
            tree=int(payload["tree"]),
            conditioned=(int(payload["conditioned"][0]), int(payload["conditioned"][1])),
            conditioning=[int(value) for value in payload["conditioning"]],
            family=str(payload["family"]),
            rotation=str(payload["rotation"]),
            parameters=tuple(float(value) for value in payload["parameters"]),
        )


@dataclass(frozen=True, slots=True)
class VineTreeInfo:
    level: int
    edges: list[VineEdgeInfo]

    @classmethod
    def _from_core(cls, payload: dict[str, Any]) -> "VineTreeInfo":
        return cls(
            level=int(payload["level"]),
            edges=[VineEdgeInfo._from_core(edge) for edge in payload["edges"]],
        )


class _BaseModel:
    def __init__(self, core_model: Any) -> None:
        self._core = core_model

    @classmethod
    def _fit_result(cls, payload: tuple[Any, Any]) -> FitResult[Any]:
        core_model, diagnostics = payload
        return FitResult(model=cls(core_model), diagnostics=FitDiagnostics._from_core(diagnostics))

    @property
    def dim(self) -> int:
        return int(self._core.dim)

    @property
    def family(self) -> str:
        return str(self._core.family)

    def log_pdf(
        self, data: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.log_pdf(_as_float_matrix(data), clip_eps=clip_eps))

    def sample(self, n: int, *, seed: int | None = None) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.sample(int(n), seed=seed))


class GaussianCopula(_BaseModel):
    @classmethod
    def from_params(cls, correlation: npt.ArrayLike) -> "GaussianCopula":
        return cls(_rscopulas._GaussianCopula.from_params(_as_float_matrix(correlation)))

    @classmethod
    def fit(
        cls, data: npt.ArrayLike, *, clip_eps: float = 1e-12, max_iter: int = 500
    ) -> FitResult["GaussianCopula"]:
        return cls._fit_result(
            _rscopulas._GaussianCopula.fit(
                _as_float_matrix(data), clip_eps=clip_eps, max_iter=max_iter
            )
        )

    @property
    def correlation(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.correlation).copy()


class StudentTCopula(_BaseModel):
    @classmethod
    def from_params(
        cls, correlation: npt.ArrayLike, degrees_of_freedom: float
    ) -> "StudentTCopula":
        return cls(
            _rscopulas._StudentTCopula.from_params(
                _as_float_matrix(correlation), degrees_of_freedom
            )
        )

    @classmethod
    def fit(
        cls, data: npt.ArrayLike, *, clip_eps: float = 1e-12, max_iter: int = 500
    ) -> FitResult["StudentTCopula"]:
        return cls._fit_result(
            _rscopulas._StudentTCopula.fit(
                _as_float_matrix(data), clip_eps=clip_eps, max_iter=max_iter
            )
        )

    @property
    def correlation(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.correlation).copy()

    @property
    def degrees_of_freedom(self) -> float:
        return float(self._core.degrees_of_freedom)


class ClaytonCopula(_BaseModel):
    @classmethod
    def from_params(cls, dim: int, theta: float) -> "ClaytonCopula":
        return cls(_rscopulas._ClaytonCopula.from_params(int(dim), float(theta)))

    @classmethod
    def fit(
        cls, data: npt.ArrayLike, *, clip_eps: float = 1e-12, max_iter: int = 500
    ) -> FitResult["ClaytonCopula"]:
        return cls._fit_result(
            _rscopulas._ClaytonCopula.fit(
                _as_float_matrix(data), clip_eps=clip_eps, max_iter=max_iter
            )
        )

    @property
    def theta(self) -> float:
        return float(self._core.theta)


class FrankCopula(_BaseModel):
    @classmethod
    def from_params(cls, dim: int, theta: float) -> "FrankCopula":
        return cls(_rscopulas._FrankCopula.from_params(int(dim), float(theta)))

    @classmethod
    def fit(
        cls, data: npt.ArrayLike, *, clip_eps: float = 1e-12, max_iter: int = 500
    ) -> FitResult["FrankCopula"]:
        return cls._fit_result(
            _rscopulas._FrankCopula.fit(
                _as_float_matrix(data), clip_eps=clip_eps, max_iter=max_iter
            )
        )

    @property
    def theta(self) -> float:
        return float(self._core.theta)


class GumbelCopula(_BaseModel):
    @classmethod
    def from_params(cls, dim: int, theta: float) -> "GumbelCopula":
        return cls(_rscopulas._GumbelCopula.from_params(int(dim), float(theta)))

    @classmethod
    def fit(
        cls, data: npt.ArrayLike, *, clip_eps: float = 1e-12, max_iter: int = 500
    ) -> FitResult["GumbelCopula"]:
        return cls._fit_result(
            _rscopulas._GumbelCopula.fit(
                _as_float_matrix(data), clip_eps=clip_eps, max_iter=max_iter
            )
        )

    @property
    def theta(self) -> float:
        return float(self._core.theta)


class VineCopula(_BaseModel):
    @classmethod
    def gaussian_c_vine(cls, order: Sequence[int], correlation: npt.ArrayLike) -> "VineCopula":
        return cls(
            _rscopulas._VineCopula.gaussian_c_vine(
                _as_order(order), _as_float_matrix(correlation)
            )
        )

    @classmethod
    def gaussian_d_vine(cls, order: Sequence[int], correlation: npt.ArrayLike) -> "VineCopula":
        return cls(
            _rscopulas._VineCopula.gaussian_d_vine(
                _as_order(order), _as_float_matrix(correlation)
            )
        )

    @classmethod
    def fit_c(
        cls,
        data: npt.ArrayLike,
        *,
        family_set: Sequence[str] | None = None,
        include_rotations: bool = True,
        criterion: str = "aic",
        truncation_level: int | None = None,
        independence_threshold: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
    ) -> FitResult["VineCopula"]:
        return cls._fit_result(
            _rscopulas._VineCopula.fit_c(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                clip_eps=clip_eps,
                max_iter=max_iter,
            )
        )

    @classmethod
    def fit_d(
        cls,
        data: npt.ArrayLike,
        *,
        family_set: Sequence[str] | None = None,
        include_rotations: bool = True,
        criterion: str = "aic",
        truncation_level: int | None = None,
        independence_threshold: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
    ) -> FitResult["VineCopula"]:
        return cls._fit_result(
            _rscopulas._VineCopula.fit_d(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                clip_eps=clip_eps,
                max_iter=max_iter,
            )
        )

    @classmethod
    def fit_r(
        cls,
        data: npt.ArrayLike,
        *,
        family_set: Sequence[str] | None = None,
        include_rotations: bool = True,
        criterion: str = "aic",
        truncation_level: int | None = None,
        independence_threshold: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
    ) -> FitResult["VineCopula"]:
        return cls._fit_result(
            _rscopulas._VineCopula.fit_r(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                clip_eps=clip_eps,
                max_iter=max_iter,
            )
        )

    @property
    def structure_kind(self) -> str:
        return str(self._core.structure_kind)

    @property
    def truncation_level(self) -> int | None:
        value = self._core.truncation_level
        return None if value is None else int(value)

    @property
    def order(self) -> list[int]:
        return [int(value) for value in self._core.order()]

    @property
    def pair_parameters(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.pair_parameters(), dtype=np.float64)

    @property
    def structure_info(self) -> VineStructureInfo:
        return VineStructureInfo._from_core(self._core.structure_info())

    @property
    def trees(self) -> list[VineTreeInfo]:
        return [VineTreeInfo._from_core(tree) for tree in self._core.trees()]
