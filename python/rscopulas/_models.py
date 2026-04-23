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
    def fit_tll(
        cls,
        u1: npt.ArrayLike,
        u2: npt.ArrayLike,
        *,
        method: str = "constant",
    ) -> "PairCopula":
        """Fit a nonparametric TLL (Transformation Local Likelihood) pair
        copula from pseudo-observations. The current implementation supports
        ``method='constant'`` only (Gaussian-kernel density estimation on
        Φ⁻¹-transformed inputs); ``'linear'`` and ``'quadratic'`` are reserved
        for future local-polynomial orders and currently raise an error.
        """
        return cls(
            _rscopulas._PairCopula.fit_tll(
                _as_float_vector(u1),
                _as_float_vector(u2),
                method=str(method),
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
        order: Sequence[int] | None = None,
    ) -> FitResult["VineCopula"]:
        """Fit a C-vine.

        Pass ``order`` to pin the variable order explicitly. Pattern for exact
        conditional sampling: place the column you intend to condition on at
        the **end** of ``order`` (e.g. ``order=[..., US10Y_YIELD_IDX]``). That
        column becomes ``variable_order[0]`` — the Rosenblatt anchor consumed
        by :meth:`sample_conditional`. For k > 1 conditioning variables, put
        them in the trailing positions in the order you want them to occupy
        ``variable_order[0:k]``; inspect :attr:`variable_order` after fitting
        to confirm the layout (the mapping from ``order`` to
        ``variable_order`` is not a simple reversal for C-vines).
        """
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
                order=None if order is None else _as_order(order),
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
        order: Sequence[int] | None = None,
    ) -> FitResult["VineCopula"]:
        """Fit a D-vine.

        Pass ``order`` to pin the variable path explicitly. For exact
        conditional sampling place the columns you intend to condition on at
        the **end** of ``order``: for a D-vine ``variable_order`` equals
        ``list(reversed(order))``, so ``order[-1]`` is the Rosenblatt anchor,
        ``order[-2]`` is the second Rosenblatt position, and so on.
        """
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
                order=None if order is None else _as_order(order),
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
    def variable_order(self) -> list[int]:
        """Diagonal ordering used by the Rosenblatt transform.

        ``variable_order[0]`` is the Rosenblatt anchor: the first variable
        simulated when traversing the fitted vine. To enable exact conditional
        sampling on a column X, fit the vine with ``fit_c(order=[X, ...])`` or
        ``fit_d(order=[X, ...])`` so that ``variable_order[0] == X``.
        """
        return [int(value) for value in self._core.variable_order()]

    @property
    def pair_parameters(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._core.pair_parameters(), dtype=np.float64)

    @property
    def structure_info(self) -> VineStructureInfo:
        return VineStructureInfo._from_core(self._core.structure_info())

    @property
    def trees(self) -> list[VineTreeInfo]:
        return [VineTreeInfo._from_core(tree) for tree in self._core.trees()]

    def rosenblatt(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Forward Rosenblatt transform ``U = F(V)``.

        Takes a matrix ``V`` of vine-distributed pseudo-observations (shape
        ``(n, d)``, indexed by original variable label) and returns the
        associated independent uniforms ``U`` with the same layout. For any
        fitted vine, ``inverse_rosenblatt(rosenblatt(V)) == V`` up to clip_eps.
        """
        return np.asarray(self._core.rosenblatt(_as_float_matrix(data)))

    def inverse_rosenblatt(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Inverse Rosenblatt transform ``V = F^{-1}(U)``.

        Takes a matrix ``U`` of independent uniforms (shape ``(n, d)``,
        indexed by original variable label) and returns a vine-distributed
        sample ``V`` with the same layout. This is the primitive behind
        :meth:`sample` and :meth:`sample_conditional`.
        """
        return np.asarray(self._core.inverse_rosenblatt(_as_float_matrix(data)))

    def sample_conditional(
        self,
        known: dict[int, npt.ArrayLike],
        n: int,
        *,
        seed: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Draw ``n`` samples from the vine conditional on known columns.

        Parameters
        ----------
        known
            Mapping ``{column_index: values}`` where ``values`` is a 1D array
            of length ``n`` in ``(0, 1)``. The provided column indices must
            form a prefix of :attr:`variable_order` — i.e. they must equal
            ``variable_order[0:k]`` as a set for some ``k``. Pin a variable
            there by fitting with ``fit_c(order=[X, ...])``.
        n
            Number of samples.
        seed
            RNG seed controlling the free (non-conditioned) columns.

        Returns
        -------
        V : ndarray of shape ``(n, dim)``
            Vine-distributed samples in original variable-label order, with
            ``V[:, col] == np.clip(known[col], eps, 1 - eps)`` for every
            supplied column.

        Raises
        ------
        NonPrefixConditioningError
            If the supplied known columns are not a diagonal prefix of
            ``variable_order``.
        """
        eps = 1e-12
        d = int(self._core.dim)
        if n <= 0:
            raise ValueError("n must be positive")

        known_columns = {int(col): _as_float_vector(values) for col, values in known.items()}
        if not known_columns:
            raise ValueError("sample_conditional requires at least one known column")
        for col, values in known_columns.items():
            if values.shape[0] != n:
                raise ValueError(
                    f"known column {col} has length {values.shape[0]}, expected {n}"
                )
            if col < 0 or col >= d:
                raise ValueError(f"known column {col} is out of range [0, {d})")

        k = len(known_columns)
        variable_order = self.variable_order
        prefix = variable_order[:k]
        if set(known_columns) != set(prefix):
            from ._rscopulas import NonPrefixConditioningError

            raise NonPrefixConditioningError(
                f"sample_conditional requires the known columns to match a prefix of "
                f"variable_order. Given known columns {sorted(known_columns)}, "
                f"variable_order[:{k}] is {list(prefix)}. "
                f"To fix, re-fit with fit_c(data, order=[..., *known_cols]) or "
                f"fit_d(data, order=[..., *known_cols]) so the conditioning "
                f"variables occupy the trailing positions of order and thus "
                f"the leading positions of variable_order."
            )

        rng = np.random.default_rng(seed)

        if k == 1:
            col = next(iter(known_columns))
            u = np.empty((n, d), dtype=np.float64)
            u[:, col] = np.clip(known_columns[col], eps, 1.0 - eps)
            free_mask = [var for var in range(d) if var != col]
            u[:, free_mask] = rng.uniform(size=(n, d - 1))
            return self.inverse_rosenblatt(u)

        v_partial = np.full((n, d), 0.5, dtype=np.float64)
        for col, values in known_columns.items():
            v_partial[:, col] = np.clip(values, eps, 1.0 - eps)

        u_fixed = np.asarray(self._core.rosenblatt_prefix(v_partial, int(k)))

        u = np.empty((n, d), dtype=np.float64)
        for idx, var in enumerate(variable_order):
            if idx < k:
                u[:, var] = u_fixed[:, idx]
            else:
                u[:, var] = rng.uniform(size=n)
        return self.inverse_rosenblatt(u)
