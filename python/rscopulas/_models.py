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


def _as_float_vector(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("expected a 1D array")
    return array


def _family_set(family_set: Sequence[str] | None) -> list[str] | None:
    if family_set is None:
        return None
    return [str(family) for family in family_set]


def _parameter_values(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def _edge_parameters(edge: Any) -> list[float]:
    if isinstance(edge, VineEdgeInfo):
        return [float(value) for value in edge.parameters]
    if isinstance(edge, dict):
        raw = edge.get("parameters", edge.get("params", []))
        return _parameter_values(raw)
    raise TypeError("vine edges must be VineEdgeInfo instances or dictionaries")


def _serialize_pair_spec(spec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "family": str(spec["family"]),
        "rotation": str(spec.get("rotation", "R0")),
        "parameters": [float(value) for value in spec.get("parameters", spec.get("params", []))],
    }
    if payload["family"] == "khoudraji":
        payload["shape_1"] = float(spec["shape_1"])
        payload["shape_2"] = float(spec["shape_2"])
        payload["base_copula_1"] = _serialize_pair_spec(spec["base_copula_1"])
        payload["base_copula_2"] = _serialize_pair_spec(spec["base_copula_2"])
    return payload


def _serialize_vine_edge(edge: VineEdgeInfo | dict[str, Any]) -> dict[str, Any]:
    if isinstance(edge, VineEdgeInfo):
        payload = {
            "tree": edge.tree,
            "conditioned": (edge.conditioned[0], edge.conditioned[1]),
            "conditioning": list(edge.conditioning),
            "family": edge.family,
            "rotation": edge.rotation,
            "parameters": list(edge.parameters),
        }
        if edge.family == "khoudraji":
            payload["shape_1"] = edge.shape_1
            payload["shape_2"] = edge.shape_2
            payload["base_copula_1"] = dict(edge.base_copula_1 or {})
            payload["base_copula_2"] = dict(edge.base_copula_2 or {})
        return payload
    if isinstance(edge, dict):
        conditioned = edge["conditioned"]
        payload = {
            "tree": int(edge.get("tree", 0)),
            "conditioned": (int(conditioned[0]), int(conditioned[1])),
            "conditioning": [int(value) for value in edge.get("conditioning", [])],
            "family": str(edge["family"]),
            "rotation": str(edge.get("rotation", "R0")),
            "parameters": _edge_parameters(edge),
        }
        if payload["family"] == "khoudraji":
            payload["shape_1"] = float(edge["shape_1"])
            payload["shape_2"] = float(edge["shape_2"])
            payload["base_copula_1"] = _serialize_pair_spec(edge["base_copula_1"])
            payload["base_copula_2"] = _serialize_pair_spec(edge["base_copula_2"])
        return payload
    raise TypeError("vine edges must be VineEdgeInfo instances or dictionaries")


def _serialize_vine_tree(tree: VineTreeInfo | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tree, VineTreeInfo):
        return {
            "level": tree.level,
            "edges": [_serialize_vine_edge(edge) for edge in tree.edges],
        }
    if isinstance(tree, dict):
        return {
            "level": int(tree["level"]),
            "edges": [_serialize_vine_edge(edge) for edge in tree["edges"]],
        }
    raise TypeError("vine trees must be VineTreeInfo instances or dictionaries")


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
    shape_1: float | None = None
    shape_2: float | None = None
    base_copula_1: dict[str, Any] | None = None
    base_copula_2: dict[str, Any] | None = None

    @classmethod
    def _from_core(cls, payload: dict[str, Any]) -> "VineEdgeInfo":
        return cls(
            tree=int(payload["tree"]),
            conditioned=(int(payload["conditioned"][0]), int(payload["conditioned"][1])),
            conditioning=[int(value) for value in payload["conditioning"]],
            family=str(payload["family"]),
            rotation=str(payload["rotation"]),
            parameters=tuple(float(value) for value in payload["parameters"]),
            shape_1=None if payload.get("shape_1") is None else float(payload["shape_1"]),
            shape_2=None if payload.get("shape_2") is None else float(payload["shape_2"]),
            base_copula_1=None if payload.get("base_copula_1") is None else dict(payload["base_copula_1"]),
            base_copula_2=None if payload.get("base_copula_2") is None else dict(payload["base_copula_2"]),
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


class PairCopula:
    def __init__(self, core_model: Any) -> None:
        self._core = core_model

    @classmethod
    def from_spec(
        cls,
        family: str,
        parameters: Sequence[float] = (),
        *,
        rotation: str = "R0",
    ) -> "PairCopula":
        return cls(
            _rscopulas._PairCopula.from_spec(
                str(family),
                parameters=_parameter_values(parameters),
                rotation=str(rotation),
            )
        )

    @classmethod
    def from_khoudraji(
        cls,
        first_family: str,
        second_family: str,
        *,
        shape_1: float,
        shape_2: float,
        first_parameters: Sequence[float] = (),
        second_parameters: Sequence[float] = (),
        rotation: str = "R0",
        first_rotation: str = "R0",
        second_rotation: str = "R0",
    ) -> "PairCopula":
        return cls(
            _rscopulas._PairCopula.from_khoudraji(
                str(first_family),
                str(second_family),
                float(shape_1),
                float(shape_2),
                first_parameters=_parameter_values(first_parameters),
                second_parameters=_parameter_values(second_parameters),
                rotation=str(rotation),
                first_rotation=str(first_rotation),
                second_rotation=str(second_rotation),
            )
        )

    @property
    def dim(self) -> int:
        return int(self._core.dim)

    @property
    def family(self) -> str:
        return str(self._core.family)

    @property
    def rotation(self) -> str:
        return str(self._core.rotation)

    @property
    def parameters(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._core.parameters)

    @property
    def spec(self) -> dict[str, Any]:
        return dict(self._core.spec)

    def log_pdf(
        self, u1: npt.ArrayLike, u2: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._core.log_pdf(_as_float_vector(u1), _as_float_vector(u2), clip_eps=clip_eps)
        )

    def cond_first_given_second(
        self, u1: npt.ArrayLike, u2: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._core.cond_first_given_second(
                _as_float_vector(u1), _as_float_vector(u2), clip_eps=clip_eps
            )
        )

    def cond_second_given_first(
        self, u1: npt.ArrayLike, u2: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._core.cond_second_given_first(
                _as_float_vector(u1), _as_float_vector(u2), clip_eps=clip_eps
            )
        )

    def inv_first_given_second(
        self, p: npt.ArrayLike, u2: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._core.inv_first_given_second(
                _as_float_vector(p), _as_float_vector(u2), clip_eps=clip_eps
            )
        )

    def inv_second_given_first(
        self, u1: npt.ArrayLike, p: npt.ArrayLike, *, clip_eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._core.inv_second_given_first(
                _as_float_vector(u1), _as_float_vector(p), clip_eps=clip_eps
            )
        )


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


class HierarchicalArchimedeanCopula(_BaseModel):
    @classmethod
    def from_tree(cls, tree: int | dict[str, Any]) -> "HierarchicalArchimedeanCopula":
        return cls(_rscopulas._HierarchicalArchimedeanCopula.from_tree(tree))

    @classmethod
    def fit(
        cls,
        data: npt.ArrayLike,
        *,
        tree: int | dict[str, Any] | None = None,
        family_set: Sequence[str] | None = None,
        structure_method: str = "agglomerative_tau_then_collapse",
        fit_method: str = "recursive_mle",
        collapse_eps: float = 0.05,
        mc_samples: int = 256,
        allow_experimental: bool = True,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
    ) -> FitResult["HierarchicalArchimedeanCopula"]:
        return cls._fit_result(
            _rscopulas._HierarchicalArchimedeanCopula.fit(
                _as_float_matrix(data),
                tree=tree,
                family_set=_family_set(family_set),
                structure_method=structure_method,
                fit_method=fit_method,
                collapse_eps=collapse_eps,
                mc_samples=mc_samples,
                allow_experimental=allow_experimental,
                clip_eps=clip_eps,
                max_iter=max_iter,
            )
        )

    @property
    def is_exact(self) -> bool:
        return bool(self._core.is_exact)

    @property
    def exact_loglik(self) -> bool:
        return bool(self._core.exact_loglik)

    @property
    def used_smle(self) -> bool:
        return bool(self._core.used_smle)

    @property
    def mc_samples(self) -> int:
        return int(self._core.mc_samples)

    @property
    def structure_method(self) -> str:
        return str(self._core.structure_method)

    @property
    def fit_method(self) -> str:
        return str(self._core.fit_method)

    @property
    def tree(self) -> int | dict[str, Any]:
        return self._core.tree()

    @property
    def leaf_order(self) -> list[int]:
        return [int(value) for value in self._core.leaf_order()]

    @property
    def parameters(self) -> list[float]:
        return [float(value) for value in self._core.parameters()]

    @property
    def families(self) -> list[str]:
        return [str(value) for value in self._core.families()]


class VineCopula(_BaseModel):
    @classmethod
    def from_trees(
        cls,
        kind: str,
        trees: Sequence[VineTreeInfo | dict[str, Any]],
        *,
        truncation_level: int | None = None,
    ) -> "VineCopula":
        payload = [_serialize_vine_tree(tree) for tree in trees]
        return cls(_rscopulas._VineCopula.from_trees(str(kind), payload, truncation_level))

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
