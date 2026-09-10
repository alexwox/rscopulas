from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from . import _rscopulas
from ._rscopulas import InvalidInputError, NonPrefixConditioningError

ModelT = TypeVar("ModelT")
_ModelT = TypeVar("_ModelT", bound="_SerializableMixin")

# Version tag embedded in pickled wrapper state.
_STATE_FORMAT = 1


def _as_float_matrix(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise InvalidInputError(f"expected a 2D array, got ndim={array.ndim}")
    return array


def _as_order(order: Sequence[int]) -> list[int]:
    return [int(value) for value in order]


def _as_float_vector(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 1:
        raise InvalidInputError(f"expected a 1D array, got ndim={array.ndim}")
    return array


def _as_count(value: Any, name: str = "n") -> int:
    """Validate a sample count up front: an ``int`` (not ``bool``) that is ``>= 1``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    count = int(value)
    if count < 1:
        raise InvalidInputError(f"{name} must be a positive integer, got {count}")
    return count


def _fmt(value: float) -> str:
    return f"{float(value):.4g}"


def _format_vector(values: Sequence[float], limit: int = 6) -> str:
    items = list(values)
    shown = ", ".join(_fmt(value) for value in items[:limit])
    return f"[{shown}, ...]" if len(items) > limit else f"[{shown}]"


def _format_matrix(matrix: npt.ArrayLike, limit: int = 3) -> str:
    array = np.asarray(matrix, dtype=np.float64)
    rows = ", ".join(_format_vector(row, limit) for row in array[:limit])
    return f"[{rows}, ...]" if array.shape[0] > limit else f"[{rows}]"


def _format_names(names: Sequence[str], limit: int = 6) -> str:
    items = [str(name) for name in names]
    shown = ", ".join(repr(name) for name in items[:limit])
    return f"[{shown}, ...]" if len(items) > limit else f"[{shown}]"


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
    if "state" in spec:
        payload["state"] = spec["state"]
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
        if edge.state is not None:
            payload["state"] = edge.state
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
        if "state" in edge:
            payload["state"] = edge["state"]
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
    likelihood_kind: str = "joint"

    @classmethod
    def _from_core(cls, diagnostics: Any) -> "FitDiagnostics":
        return cls(
            loglik=float(diagnostics.loglik),
            aic=float(diagnostics.aic),
            bic=float(diagnostics.bic),
            converged=bool(diagnostics.converged),
            n_iter=int(diagnostics.n_iter),
            likelihood_kind=str(diagnostics.likelihood_kind),
        )


@dataclass(frozen=True, slots=True)
class FitResult(Generic[ModelT]):
    model: ModelT
    diagnostics: FitDiagnostics


@dataclass(frozen=True, slots=True)
class FactorFitDiagnostics:
    """Diagnostics returned by :meth:`FactorCopula.fit`.

    Extends :class:`FitDiagnostics` with delta-method standard errors for
    every polished link parameter. Entries are in the flat parameter layout
    produced by walking the fitted links in input-column order and
    concatenating their free parameters (skipping Independence, TLL,
    Khoudraji, and the ν block of Student-t, which are held fixed during
    the polish).
    """

    loglik: float
    aic: float
    bic: float
    converged: bool
    n_iter: int
    std_errors: tuple[float, ...]
    likelihood_kind: str = "joint"

    @classmethod
    def _build(cls, diagnostics: Any, std_errors: Any) -> "FactorFitDiagnostics":
        return cls(
            loglik=float(diagnostics.loglik),
            aic=float(diagnostics.aic),
            bic=float(diagnostics.bic),
            converged=bool(diagnostics.converged),
            n_iter=int(diagnostics.n_iter),
            likelihood_kind=str(diagnostics.likelihood_kind),
            std_errors=tuple(float(value) for value in std_errors),
        )


@dataclass(frozen=True, slots=True)
class FactorFitResult:
    model: "FactorCopula"
    diagnostics: FactorFitDiagnostics


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
    state: dict[str, Any] | None = None

    @classmethod
    def _from_core(cls, payload: dict[str, Any]) -> "VineEdgeInfo":
        return cls(
            tree=int(payload["tree"]),
            conditioned=(int(payload["conditioned"][0]), int(payload["conditioned"][1])),
            conditioning=[int(value) for value in payload["conditioning"]],
            family=str(payload["family"]),
            rotation=str(payload["rotation"]),
            parameters=tuple(float(value) for value in payload["parameters"]),
            state=payload.get("state"),
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


class _SerializableMixin:
    """Serialization, copying, equality, and repr shared by every model.

    Subclasses set ``_core_cls`` to the compiled class that backs them and
    override ``_repr_fields`` to describe their key parameters.
    """

    _core_cls: ClassVar[Any]
    _core: Any

    def __init__(self, core_model: Any) -> None:
        self._core = core_model

    @classmethod
    def from_json(cls: type[_ModelT], payload: str) -> _ModelT:
        """Rebuild a model from :meth:`to_json` output.

        Payloads are validated on load: malformed or out-of-range state
        raises :class:`InvalidInputError`. Vine payloads carry a
        ``format_version`` and unversioned payloads are rejected.
        """
        return cls(cls._core_cls.from_json(str(payload)))

    def to_json(self) -> str:
        """JSON serialization of the model.

        The layout mirrors the Rust ``serde`` representation and pairs with
        :meth:`from_json`; compare payloads only across matching rscopulas
        minor versions.
        """
        return str(self._core.to_json())

    def __getstate__(self) -> dict[str, Any]:
        return {"format": _STATE_FORMAT, "model": self.to_json()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or state.get("format") != _STATE_FORMAT
            or "model" not in state
        ):
            raise InvalidInputError(f"unsupported pickled state for {type(self).__name__}")
        self._core = type(self)._core_cls.from_json(str(state["model"]))

    def __copy__(self: _ModelT) -> _ModelT:
        return type(self)(self._core.__copy__())

    def __deepcopy__(self: _ModelT, memo: dict[int, Any]) -> _ModelT:
        return type(self)(self._core.__deepcopy__(memo))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return bool(self._core == other._core)

    # Models compare by value, so they are deliberately unhashable.
    __hash__ = None  # type: ignore[assignment]

    def _repr_fields(self) -> list[tuple[str, str]]:
        return []

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={value}" for name, value in self._repr_fields())
        return f"{type(self).__name__}({fields})"


class _BaseModel(_SerializableMixin):
    @classmethod
    def _fit_result(cls: type[_ModelT], payload: tuple[Any, Any]) -> FitResult[_ModelT]:
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
        """Draw ``n`` pseudo-observations from the model.

        ``n`` must be a positive ``int``. ``seed`` must be ``None`` or an
        integer in ``[0, 2**64)``; a given seed reproduces the same draw
        regardless of NumPy's global random state. The draw runs with the
        GIL released.
        """
        return np.asarray(self._core.sample(_as_count(n), seed=seed))

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [("family", repr(self.family)), ("dim", str(self.dim))]


class PairCopula(_SerializableMixin):
    _core_cls = _rscopulas._PairCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [
            ("family", repr(self.family)),
            ("rotation", repr(self.rotation)),
            ("parameters", _format_vector(self.parameters)),
            ("dim", str(self.dim)),
        ]

    @classmethod
    def from_spec(
        cls,
        family: str,
        parameters: Sequence[float] = (),
        *,
        rotation: str = "R0",
        state: dict[str, Any] | None = None,
    ) -> "PairCopula":
        return cls(
            _rscopulas._PairCopula.from_spec(
                str(family),
                parameters=_parameter_values(parameters),
                rotation=str(rotation),
                state=state,
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
        copula using ``constant``, ``linear``, or ``quadratic`` local likelihood.
        The fitted grid is normalized to uniform margins; ``spec["state"]``
        preserves it for reconstruction with :meth:`from_spec`.
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
    _core_cls = _rscopulas._GaussianCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [*super()._repr_fields(), ("correlation", _format_matrix(self.correlation))]

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
    _core_cls = _rscopulas._StudentTCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [
            *super()._repr_fields(),
            ("degrees_of_freedom", _fmt(self.degrees_of_freedom)),
            ("correlation", _format_matrix(self.correlation)),
        ]

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
    _core_cls = _rscopulas._ClaytonCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [*super()._repr_fields(), ("theta", _fmt(self.theta))]

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
    _core_cls = _rscopulas._FrankCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [*super()._repr_fields(), ("theta", _fmt(self.theta))]

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
    _core_cls = _rscopulas._GumbelCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [*super()._repr_fields(), ("theta", _fmt(self.theta))]

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
    _core_cls = _rscopulas._HierarchicalArchimedeanCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [
            *super()._repr_fields(),
            ("families", _format_names(self.families)),
            ("parameters", _format_vector(self.parameters)),
        ]

    def composite_log_pdf(self, data: npt.ArrayLike, *, clip_eps: float = 1e-12) -> npt.NDArray[np.float64]:
        """Pairwise composite score, which is not a normalized joint density.

        Nested HACs currently support this score and sampling. ``log_pdf``
        requires an exact, flat HAC. Full/simulated MLE methods are unsupported.
        """
        return np.asarray(self._core.composite_log_pdf(_as_float_matrix(data), clip_eps=clip_eps))

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
        fit_method: str = "composite_mle",
        collapse_eps: float = 0.05,
        mc_samples: int = 0,
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
    _core_cls = _rscopulas._VineCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        families = [edge.family for tree in self.trees for edge in tree.edges]
        return [
            *super()._repr_fields(),
            ("kind", repr(self.structure_kind)),
            ("truncation_level", str(self.truncation_level)),
            ("families", _format_names(families)),
        ]

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
        independence_test_level: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
        order: Sequence[int] | None = None,
        tree_algorithm: str = "kruskal",
        tree_criterion: str = "tau",
        select_trunc_lvl: bool = False,
        rng_seed: int | None = None,
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

        ``tree_algorithm`` is accepted only as ``"kruskal"`` for C-vines —
        the star-tree shape is fixed by the vine family, so tree-search
        choices do not apply. ``family_set`` (``None`` delegates to the core
        default set), ``independence_test_level``, ``tree_criterion``,
        ``select_trunc_lvl``, and ``rng_seed`` behave exactly as documented
        on :meth:`fit_r`; ``criterion`` can be ``"mbicv"`` or
        ``"mbicv:<psi0>"`` to drive auto-truncation.
        """
        return cls._fit_result(
            _rscopulas._VineCopula.fit_c(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                independence_test_level=independence_test_level,
                clip_eps=clip_eps,
                max_iter=max_iter,
                order=None if order is None else _as_order(order),
                tree_algorithm=tree_algorithm,
                tree_criterion=tree_criterion,
                select_trunc_lvl=select_trunc_lvl,
                rng_seed=rng_seed,
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
        independence_test_level: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
        order: Sequence[int] | None = None,
        tree_algorithm: str = "kruskal",
        tree_criterion: str = "tau",
        select_trunc_lvl: bool = False,
        rng_seed: int | None = None,
    ) -> FitResult["VineCopula"]:
        """Fit a D-vine.

        Pass ``order`` to pin the variable path explicitly. For exact
        conditional sampling place the columns you intend to condition on at
        the **end** of ``order``: for a D-vine ``variable_order`` equals
        ``list(reversed(order))``, so ``order[-1]`` is the Rosenblatt anchor,
        ``order[-2]`` is the second Rosenblatt position, and so on.

        ``tree_algorithm`` is accepted only as ``"kruskal"`` for D-vines
        (path structure is fixed). ``family_set`` (``None`` delegates to the
        core default set), ``independence_test_level``, and the other
        keyword arguments match :meth:`fit_r`.
        """
        return cls._fit_result(
            _rscopulas._VineCopula.fit_d(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                independence_test_level=independence_test_level,
                clip_eps=clip_eps,
                max_iter=max_iter,
                order=None if order is None else _as_order(order),
                tree_algorithm=tree_algorithm,
                tree_criterion=tree_criterion,
                select_trunc_lvl=select_trunc_lvl,
                rng_seed=rng_seed,
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
        independence_test_level: float | None = None,
        clip_eps: float = 1e-12,
        max_iter: int = 500,
        tree_algorithm: str = "kruskal",
        tree_criterion: str = "tau",
        select_trunc_lvl: bool = False,
        rng_seed: int | None = None,
    ) -> FitResult["VineCopula"]:
        """Fit an R-vine with Dissmann-style spanning-tree selection.

        ``family_set``
            Candidate pair families as strings: ``independence``,
            ``gaussian``, ``student_t``, ``clayton``, ``frank``, ``gumbel``,
            ``joe``, ``bb1``, ``bb6``, ``bb7``, ``bb8``, ``tawn1``, ``tawn2``,
            ``tll``, and ``khoudraji``. ``None`` (default) delegates to the
            core default set: ``independence``, ``gaussian``, ``student_t``,
            ``clayton``, ``frank``, ``gumbel``, ``joe``, ``bb1``, ``bb7``.
            ``khoudraji`` and ``tll`` are opt-in because they dominate fit
            time.

        Options matching pyvinecopulib's ``FitControlsVinecop``:

        ``criterion``
            ``"aic"`` (default), ``"bic"``, or ``"mbicv"`` / ``"mbicv:<psi0>"``.
            mBICV is Nagler, Bumann & Czado's modified vine-BIC — required
            for ``select_trunc_lvl=True``.
        ``tree_algorithm``
            ``"kruskal"`` (default), ``"prim"``, ``"random_weighted"``, or
            ``"random_unweighted"``. Wilson-based random algorithms use
            ``rng_seed`` for reproducibility.
        ``tree_criterion``
            ``"tau"`` (default, Kendall's τ), ``"rho"`` (Spearman's ρ), or
            ``"hoeffding"`` (Hoeffding's D — picks up non-monotone pair
            dependence that the rank correlations miss).
        ``select_trunc_lvl``
            When ``True`` and ``criterion="mbicv"``, truncation depth is
            chosen automatically by walking back from the full fit and
            dropping trees whose mBICV contribution is positive. A manual
            ``truncation_level`` becomes an upper cap on the auto-selected
            depth (matching vinecopulib's semantics).
        ``independence_test_level``
            Optional significance level (for example ``0.05``) for the
            asymptotic Kendall-τ independence test run on every edge before
            family selection; edges that do not reject independence receive
            the independence copula. ``None`` (default) disables the test.
            ``independence_threshold`` keeps its raw cut-off semantics and
            both checks may be combined. Must lie in ``(0, 1)``.
        ``rng_seed``
            Seed for stochastic tree algorithms. Ignored for ``kruskal`` /
            ``prim``. ``None`` draws from the OS RNG; otherwise an integer
            in ``[0, 2**64)``.
        """
        return cls._fit_result(
            _rscopulas._VineCopula.fit_r(
                _as_float_matrix(data),
                family_set=_family_set(family_set),
                include_rotations=include_rotations,
                criterion=criterion,
                truncation_level=truncation_level,
                independence_threshold=independence_threshold,
                independence_test_level=independence_test_level,
                clip_eps=clip_eps,
                max_iter=max_iter,
                tree_algorithm=tree_algorithm,
                tree_criterion=tree_criterion,
                select_trunc_lvl=select_trunc_lvl,
                rng_seed=rng_seed,
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
        sampling on a column X, fit the vine with ``fit_c(order=[..., X])`` or
        ``fit_d(order=[..., X])`` so that ``variable_order[0] == X``.
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
            there by fitting with ``fit_c(order=[..., X])``.
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
        n = _as_count(n)

        known_columns = {int(col): _as_float_vector(values) for col, values in known.items()}
        if not known_columns:
            raise InvalidInputError("sample_conditional requires at least one known column")
        for col, values in known_columns.items():
            if values.shape[0] != n:
                raise InvalidInputError(
                    f"known column {col} has length {values.shape[0]}, expected {n}"
                )
            if col < 0 or col >= d:
                raise InvalidInputError(f"known column {col} is out of range [0, {d})")

        k = len(known_columns)
        variable_order = self.variable_order
        prefix = variable_order[:k]
        if set(known_columns) != set(prefix):
            raise NonPrefixConditioningError(
                f"sample_conditional requires the known columns to match a prefix of "
                f"variable_order. Given known columns {sorted(known_columns)}, "
                f"variable_order[:{k}] is {list(prefix)}. "
                f"To fix, re-fit with fit_c(data, order=[..., *known_cols]) or "
                f"fit_d(data, order=[..., *known_cols]) so the conditioning "
                f"variables occupy the trailing positions of order and thus "
                f"the leading positions of variable_order."
            )

        # Free columns come from the seeded Rust generator that `sample` also
        # uses, so a seed reproduces the draw regardless of NumPy's global
        # random state. Column j of `uniforms` feeds variable_order[k + j].
        uniforms = np.asarray(_rscopulas.uniform_matrix(n, d - k, seed))

        if k == 1:
            col = next(iter(known_columns))
            u = np.empty((n, d), dtype=np.float64)
            u[:, col] = np.clip(known_columns[col], eps, 1.0 - eps)
            u[:, variable_order[1:]] = uniforms
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
                u[:, var] = uniforms[:, idx - k]
        return self.inverse_rosenblatt(u)


class FactorCopula(_BaseModel):
    """Krupskii–Joe factor copula (single-factor ``Basic1F`` layout).

    The model joins ``d`` observed variables through a single latent factor
    ``V ~ U(0, 1)``. Each observed-to-factor link is an arbitrary bivariate
    pair-copula, so every family in :mod:`rscopulas.PairCopula`
    (Gaussian, Clayton, Frank, Gumbel, Joe, BB1/6/7/8, Tawn1/2, TLL,
    Khoudraji) can be used as a link.

    Gaussian links use the exact Gaussian joint density. Other links use
    normal-scale Gauss-Legendre integration with adaptive interval refinement.
    ``quadrature_rel_tol`` controls the estimated relative error, and
    ``quadrature_max_nodes`` caps integrand evaluations per row (defaults:
    1e-7 and 4096). Exceeding that budget raises a numerical error.
    ``adaptive_quadrature=False`` uses exactly ``quadrature_nodes`` with no
    accuracy guarantee. These settings are stored with the fitted model.

    Fitting initializes from a normal-score pseudo-latent, refines link fits,
    and optionally polishes the joint likelihood by coordinate ascent.
    ``converged`` reports the polish stopping condition; it is false when
    polishing is disabled. Standard errors use a numerical Hessian and can
    be NaN when information is not identifiable or evaluation fails.

    Example
    -------
    >>> from rscopulas import FactorCopula
    >>> fit = FactorCopula.fit(data, family_set=["gaussian", "clayton"])
    >>> log_density = fit.model.log_pdf(data)
    >>> sample = fit.model.sample(1000, seed=0)
    """

    _core_cls = _rscopulas._FactorCopula

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [
            *super()._repr_fields(),
            ("layout", repr(self.layout)),
            ("links", _format_names([str(link["family"]) for link in self.links])),
        ]

    @classmethod
    def from_links(
        cls,
        links: Sequence[dict[str, Any]],
        *,
        quadrature_nodes: int = 25,
        adaptive_quadrature: bool = True,
        quadrature_max_nodes: int = 4096,
        quadrature_rel_tol: float = 1e-7,
    ) -> "FactorCopula":
        """Build a factor copula directly from pre-specified link dicts.

        Each entry in ``links`` follows the same schema as
        :class:`VineEdgeInfo` / :class:`PairCopula` specs:
        ``{"family": str, "rotation": str, "parameters": [floats]}``.
        Khoudraji links require the usual ``base_copula_1 / shape_1 / …``
        keys; see the PairCopula documentation for the full shape.
        """
        payload = [_serialize_pair_spec(dict(link)) for link in links]
        return cls(
            _rscopulas._FactorCopula.from_links(
                payload, int(quadrature_nodes), adaptive_quadrature=adaptive_quadrature,
                quadrature_max_nodes=int(quadrature_max_nodes), quadrature_rel_tol=float(quadrature_rel_tol),
            )
        )

    @classmethod
    def fit(
        cls,
        data: npt.ArrayLike,
        *,
        family_set: Sequence[str] | None = None,
        include_rotations: bool = True,
        criterion: str = "aic",
        quadrature_nodes: int = 25,
        adaptive_quadrature: bool = True,
        quadrature_max_nodes: int = 4096,
        quadrature_rel_tol: float = 1e-7,
        refine_iterations: int = 2,
        joint_polish_cycles: int = 5,
        joint_polish_rel_tol: float = 1e-6,
        layout: str = "basic_1f",
        clip_eps: float = 1e-12,
        max_iter: int = 500,
    ) -> FactorFitResult:
        """Fit a factor copula to pseudo-observations.

        Parameters
        ----------
        data
            Pseudo-observation matrix of shape ``(n, d)`` with entries in
            ``(0, 1)``.
        family_set
            Candidate link families. Defaults to a conservative set
            (Independence, Gaussian, Clayton, Frank, Gumbel). Opt into BB*,
            Tawn*, TLL, or Khoudraji by naming them explicitly.
        include_rotations
            Whether rotated Archimedean links may be selected (R180 for
            positive-dependent lower-tail data, R90/R270 for negative tau).
        criterion
            ``"aic"`` or ``"bic"`` — used to pick between candidate link
            families at each observed variable.
        quadrature_nodes
            Minimum work in adaptive mode, or exact node count in fixed mode.
        adaptive_quadrature
            Refine the latent integral by estimated error (default True).
            False trades accuracy checks for a fixed amount of work.
        quadrature_max_nodes
            Hard cap on integrand evaluations per row, including refinement
            work. Applies throughout fitting and inference in adaptive mode.
        quadrature_rel_tol
            Target estimated relative error for adaptive integration.
        refine_iterations
            EM-style refinement passes after the initial sequential MLE.
            Two is the default and usually enough to correct the warm-start
            attenuation bias.
        joint_polish_cycles
            Coordinate-ascent sweeps of the final joint-MLE polish. Each
            sweep optimises every link's free parameters jointly against
            the true quadrature-integrated factor log-likelihood. Set to
            ``0`` to disable the polish (reproduces the pre-polish fit for
            benchmarking). Default ``5`` caps the number of sweeps;
            diagnostics report whether the stopping criterion was reached.
        joint_polish_rel_tol
            Relative-tolerance stop criterion for the polish sweep: any
            sweep that improves the log-likelihood by less than
            ``joint_polish_rel_tol * |loglik|`` ends the polish early.
        layout
            Factor layout. Only ``"basic_1f"`` is supported today.

        Returns
        -------
        FactorFitResult
            The fitted model plus factor-specific diagnostics (including
            delta-method standard errors for every polished parameter).
        """
        core_model, diagnostics, std_errors = _rscopulas._FactorCopula.fit(
            _as_float_matrix(data),
            family_set=_family_set(family_set),
            include_rotations=include_rotations,
            criterion=criterion,
            quadrature_nodes=int(quadrature_nodes),
            adaptive_quadrature=adaptive_quadrature,
            quadrature_max_nodes=int(quadrature_max_nodes),
            quadrature_rel_tol=float(quadrature_rel_tol),
            refine_iterations=int(refine_iterations),
            joint_polish_cycles=int(joint_polish_cycles),
            joint_polish_rel_tol=float(joint_polish_rel_tol),
            layout=layout,
            clip_eps=clip_eps,
            max_iter=max_iter,
        )
        return FactorFitResult(
            model=cls(core_model),
            diagnostics=FactorFitDiagnostics._build(diagnostics, std_errors),
        )

    @property
    def num_factors(self) -> int:
        """Number of latent factors. Always 1 for the ``Basic1F`` layout."""
        return int(self._core.num_factors)

    @property
    def quadrature_nodes(self) -> int:
        """Gauss–Legendre quadrature size used for log-density evaluation."""
        return int(self._core.quadrature_nodes)

    @property
    def adaptive_quadrature(self) -> bool:
        return bool(self._core.adaptive_quadrature)

    @property
    def quadrature_max_nodes(self) -> int:
        return int(self._core.quadrature_max_nodes)

    @property
    def quadrature_rel_tol(self) -> float:
        return float(self._core.quadrature_rel_tol)

    @property
    def layout(self) -> str:
        """Factor layout name, e.g. ``"basic_1f"``."""
        return str(self._core.layout)

    @property
    def links(self) -> list[dict[str, Any]]:
        """Per-variable link specifications as plain dictionaries.

        One entry per observed variable, in input-column order. Use these to
        inspect which families were selected, their rotations, and fitted
        parameters.
        """
        return [dict(link) for link in self._core.links()]
