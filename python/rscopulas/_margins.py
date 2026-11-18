"""Marginal models with extreme-value tails.

A copula describes dependence between uniforms. To go from raw data to a
joint model and back, each column also needs a distribution function (raw
value to uniform) and its inverse (uniform to raw value).
:func:`rscopulas.to_pseudo_obs` gives the first direction only, and ranks
say nothing beyond the largest observation.

:class:`Margins` gives both directions. Each column is modelled as::

         lower tail            body                upper tail
    generalized Pareto | interpolated ECDF | generalized Pareto

so a sample drawn from a fitted copula can be mapped back to the raw scale,
including levels more extreme than anything observed.

:class:`GeneralizedExtremeValue` is the companion model for block maxima,
with return levels. :func:`mean_excess` and :func:`threshold_stability` are
the standard diagnostics for choosing a tail threshold.

Sign convention: a positive ``shape`` is a heavy (power-law) tail, zero an
exponential-type tail, negative a tail with a finite end point.
``scipy.stats.genextreme`` uses ``c = -shape``; ``genpareto`` uses
``c = shape``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from . import _rscopulas
from ._models import _SerializableMixin
from ._rscopulas import InvalidInputError

__all__ = [
    "GeneralizedExtremeValue",
    "GeneralizedPareto",
    "GevFit",
    "GpdFit",
    "GpdTail",
    "Margins",
    "MeanExcess",
    "SemiParametricMargin",
    "ThresholdStability",
    "block_maxima",
    "mean_excess",
    "threshold_stability",
]

_Array = npt.NDArray[np.float64]


def _vector(data: npt.ArrayLike, name: str) -> _Array:
    try:
        array = np.ascontiguousarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be a 1-D vector, got ndim={array.ndim}")
    return array


def _matrix(data: npt.ArrayLike, name: str) -> _Array:
    try:
        array = np.ascontiguousarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 2:
        raise InvalidInputError(f"{name} must be a 2-D (n, d) matrix, got ndim={array.ndim}")
    return array


def _elementwise(function: Any, what: str, values: npt.ArrayLike, *parameters: float) -> Any:
    """Apply a compiled vector function to a scalar or an array of any shape."""
    array = np.asarray(values, dtype=np.float64)
    result = np.asarray(function(what, np.ascontiguousarray(array.ravel()), *parameters))
    return float(result[0]) if array.ndim == 0 else result.reshape(array.shape)


def _optional(value: float | None) -> float | None:
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Generalized Pareto
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralizedPareto:
    """Distribution of an excess ``y >= 0`` over a threshold.

    ``P(Y <= y) = 1 - (1 + shape * y / scale) ** (-1 / shape)``.
    """

    scale: float
    shape: float

    def __post_init__(self) -> None:
        # Validates the parameters in the core.
        _rscopulas.gpd_eval("cdf", np.zeros(1), float(self.scale), float(self.shape))

    def cdf(self, y: npt.ArrayLike) -> Any:
        return _elementwise(_rscopulas.gpd_eval, "cdf", y, self.scale, self.shape)

    def sf(self, y: npt.ArrayLike) -> Any:
        """Survival function ``P(Y > y)``, accurate far in the tail."""
        return _elementwise(_rscopulas.gpd_eval, "survival", y, self.scale, self.shape)

    def logpdf(self, y: npt.ArrayLike) -> Any:
        return _elementwise(_rscopulas.gpd_eval, "ln_pdf", y, self.scale, self.shape)

    def quantile(self, p: npt.ArrayLike) -> Any:
        """Quantiles for ``0 <= p < 1``."""
        return _elementwise(_rscopulas.gpd_eval, "quantile", p, self.scale, self.shape)

    @classmethod
    def fit(cls, excesses: npt.ArrayLike) -> "GpdFit":
        """Maximum-likelihood fit to excesses over a threshold.

        Pass ``x[x > u] - u``. The profile likelihood is searched over its
        whole domain, so no starting value is needed. The shape is
        restricted to ``>= -1``, where the likelihood is bounded.
        """
        scale, shape, loglik, covariance = _rscopulas.gpd_fit(_vector(excesses, "excesses"))
        return GpdFit(
            model=cls(float(scale), float(shape)),
            loglik=float(loglik),
            n_obs=int(np.size(excesses)),
            covariance=None if covariance is None else np.asarray(covariance),
        )


@dataclass(frozen=True, eq=False)
class GpdFit:
    """A fitted :class:`GeneralizedPareto`.

    ``covariance`` is the asymptotic covariance of ``(scale, shape)``, or
    ``None`` when the estimate lies on the boundary ``shape = -1``.
    """

    model: GeneralizedPareto
    loglik: float
    n_obs: int
    covariance: _Array | None

    @property
    def std_errors(self) -> _Array | None:
        """Standard errors of ``(scale, shape)``."""
        return None if self.covariance is None else np.sqrt(np.diag(self.covariance))


# ---------------------------------------------------------------------------
# Generalized extreme value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralizedExtremeValue:
    """Limit distribution of block maxima.

    ``P(M <= x) = exp(-(1 + shape * (x - location) / scale) ** (-1 / shape))``.
    """

    location: float
    scale: float
    shape: float

    def __post_init__(self) -> None:
        _rscopulas.gev_eval("cdf", np.zeros(1), *self._parameters())

    def _parameters(self) -> tuple[float, float, float]:
        return float(self.location), float(self.scale), float(self.shape)

    def cdf(self, x: npt.ArrayLike) -> Any:
        return _elementwise(_rscopulas.gev_eval, "cdf", x, *self._parameters())

    def logpdf(self, x: npt.ArrayLike) -> Any:
        return _elementwise(_rscopulas.gev_eval, "ln_pdf", x, *self._parameters())

    def quantile(self, p: npt.ArrayLike) -> Any:
        """Quantiles for ``0 < p < 1``."""
        return _elementwise(_rscopulas.gev_eval, "quantile", p, *self._parameters())

    def return_level(self, period: npt.ArrayLike) -> Any:
        """Level exceeded by a block maximum once every ``period`` blocks.

        With annual maxima, ``return_level(100)`` is the 100-year level: the
        ``1 - 1/period`` quantile. ``period`` must be greater than 1.
        """
        return _elementwise(_rscopulas.gev_eval, "return_level", period, *self._parameters())

    @classmethod
    def fit(cls, maxima: npt.ArrayLike) -> "GevFit":
        """Maximum-likelihood fit to block maxima (see :func:`block_maxima`)."""
        location, scale, shape, loglik, covariance = _rscopulas.gev_fit(_vector(maxima, "maxima"))
        return GevFit(
            model=cls(float(location), float(scale), float(shape)),
            loglik=float(loglik),
            n_obs=int(np.size(maxima)),
            covariance=None if covariance is None else np.asarray(covariance),
        )


@dataclass(frozen=True, eq=False)
class GevFit:
    """A fitted :class:`GeneralizedExtremeValue`.

    ``covariance`` is the asymptotic covariance of ``(location, scale,
    shape)``, or ``None`` when the information matrix is not positive
    definite.
    """

    model: GeneralizedExtremeValue
    loglik: float
    n_obs: int
    covariance: _Array | None

    @property
    def std_errors(self) -> _Array | None:
        """Standard errors of ``(location, scale, shape)``."""
        return None if self.covariance is None else np.sqrt(np.diag(self.covariance))

    def return_level_std_error(self, period: npt.ArrayLike) -> Any:
        """Delta-method standard error of ``model.return_level(period)``.

        The interval ``level +- 1.96 * std_error`` is symmetric, which is
        optimistic on the upper side for long periods and heavy tails.
        Returns ``None`` when the fit has no covariance.
        """
        if self.covariance is None:
            return None
        periods = np.asarray(period, dtype=np.float64)
        result = np.asarray(
            _rscopulas.gev_return_level_std_error(
                np.ascontiguousarray(periods.ravel()),
                *self.model._parameters(),
                np.ascontiguousarray(self.covariance),
            )
        )
        return float(result[0]) if periods.ndim == 0 else result.reshape(periods.shape)


def block_maxima(x: npt.ArrayLike, block_size: int) -> _Array:
    """Maxima of consecutive blocks of ``block_size`` observations.

    A final incomplete block is dropped.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise InvalidInputError(f"block_size must be an integer, got {block_size!r}")
    if block_size < 1:
        raise InvalidInputError("block_size must be at least 1")
    return np.asarray(_rscopulas.block_maxima(_vector(x, "x"), int(block_size)))


# ---------------------------------------------------------------------------
# Semi-parametric margins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpdTail:
    """One fitted tail of a margin.

    ``threshold`` is where the tail starts on the raw scale, ``probability``
    the mass beyond it, and ``gpd`` the distribution of the distance beyond
    the threshold, fitted to ``n_exceedances`` observations.
    """

    threshold: float
    probability: float
    gpd: GeneralizedPareto
    n_exceedances: int

    @classmethod
    def _from_core(cls, payload: dict[str, Any] | None) -> "GpdTail | None":
        if payload is None:
            return None
        return cls(
            threshold=float(payload["threshold"]),
            probability=float(payload["probability"]),
            gpd=GeneralizedPareto(float(payload["scale"]), float(payload["shape"])),
            n_exceedances=int(payload["n_exceedances"]),
        )


def _fit_core(
    data: _Array,
    lower_tail: float | None,
    upper_tail: float | None,
    lower_threshold: float | None,
    upper_threshold: float | None,
    min_exceedances: int,
) -> Any:
    if isinstance(min_exceedances, bool) or not isinstance(min_exceedances, (int, np.integer)):
        raise InvalidInputError(f"min_exceedances must be an integer, got {min_exceedances!r}")
    if min_exceedances < 0:
        raise InvalidInputError("min_exceedances must not be negative")
    return _rscopulas._Margins.fit(
        data,
        lower_tail=_optional(lower_tail),
        upper_tail=_optional(upper_tail),
        lower_threshold=_optional(lower_threshold),
        upper_threshold=_optional(upper_threshold),
        min_exceedances=int(min_exceedances),
    )


class _MarginBase(_SerializableMixin):
    _core_cls = _rscopulas._Margins


class SemiParametricMargin(_MarginBase):
    """Distribution function of one variable.

    The body is the empirical distribution function, linear between
    observations; each tail beyond its threshold is a generalized Pareto
    distribution. The pieces join continuously and the function is strictly
    increasing, so :meth:`quantile` inverts :meth:`cdf` exactly.
    """

    def __init__(self, core_model: Any) -> None:
        if int(core_model.dim) != 1:
            raise InvalidInputError("SemiParametricMargin holds exactly one column")
        super().__init__(core_model)

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        *,
        lower_tail: float | None = 0.1,
        upper_tail: float | None = 0.1,
        lower_threshold: float | None = None,
        upper_threshold: float | None = None,
        min_exceedances: int = 10,
    ) -> "SemiParametricMargin":
        """Fit the margin to a 1-D sample.

        Parameters
        ----------
        x
            Raw observations of one variable.
        lower_tail, upper_tail
            Fraction of the sample modelled as each tail, in ``(0, 0.5)``.
            ``None`` leaves that tail empirical: the distribution function is
            then flat beyond the sample on that side.
        lower_threshold, upper_threshold
            Start a tail at this raw-scale value instead of at a fraction. It
            is moved to the nearest observation on the body side. Takes
            precedence over the fraction.
        min_exceedances
            Fewest observations beyond a threshold for which a tail is fitted.
        """
        column = _vector(x, "x").reshape(-1, 1)
        return cls(
            _fit_core(
                column, lower_tail, upper_tail, lower_threshold, upper_threshold, min_exceedances
            )
        )

    def _info(self) -> dict[str, Any]:
        return self._core.tails()[0]

    @property
    def n_obs(self) -> int:
        return int(self._info()["n_obs"])

    @property
    def lower_tail(self) -> GpdTail | None:
        return GpdTail._from_core(self._info()["lower"])

    @property
    def upper_tail(self) -> GpdTail | None:
        return GpdTail._from_core(self._info()["upper"])

    def cdf(self, x: npt.ArrayLike) -> Any:
        """``P(X <= x)`` for a scalar or an array of raw values."""
        array = np.asarray(x, dtype=np.float64)
        result = np.asarray(self._core.cdf(np.ascontiguousarray(array.reshape(-1, 1))))
        return float(result[0, 0]) if array.ndim == 0 else result.reshape(array.shape)

    def quantile(self, p: npt.ArrayLike) -> Any:
        """Raw-scale quantiles for probabilities strictly inside ``(0, 1)``."""
        array = np.asarray(p, dtype=np.float64)
        result = np.asarray(self._core.quantile(np.ascontiguousarray(array.reshape(-1, 1))))
        return float(result[0, 0]) if array.ndim == 0 else result.reshape(array.shape)

    def return_level(self, period: float) -> float:
        """Level exceeded once every ``period`` observations on average.

        With daily data, ``return_level(365.25 * 100)`` is the 100-year
        level, provided the observations are independent.
        """
        return float(self._core.return_level(float(period))[0])

    def _repr_fields(self) -> list[tuple[str, str]]:
        def describe(tail: GpdTail | None) -> str:
            if tail is None:
                return "empirical"
            return f"GPD(threshold={tail.threshold:.6g}, shape={tail.gpd.shape:.4g})"

        return [
            ("n_obs", str(self.n_obs)),
            ("lower", describe(self.lower_tail)),
            ("upper", describe(self.upper_tail)),
        ]


class Margins(_MarginBase):
    """One :class:`SemiParametricMargin` per column of a data matrix.

    The full pipeline from raw data to simulated raw data::

        margins = Margins.fit(x)                    # x: raw (n, d)
        u = margins.cdf(x)                          # pseudo-observations
        model = VineCopula.fit_r(u).model           # any rscopulas model
        simulated = margins.quantile(model.sample(100_000, seed=1))
    """

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        *,
        lower_tail: float | None = 0.1,
        upper_tail: float | None = 0.1,
        min_exceedances: int = 10,
    ) -> "Margins":
        """Fit every column of a raw ``(n, d)`` matrix with the same settings.

        See :meth:`SemiParametricMargin.fit` for the parameters. To use
        different thresholds per column, fit the columns one by one and
        combine them with :meth:`from_margins`.
        """
        return cls(
            _fit_core(_matrix(x, "x"), lower_tail, upper_tail, None, None, min_exceedances)
        )

    @classmethod
    def from_margins(cls, margins: Sequence[SemiParametricMargin]) -> "Margins":
        """Combine margins fitted one at a time, in column order."""
        parts = list(margins)
        if not parts or not all(isinstance(part, SemiParametricMargin) for part in parts):
            raise InvalidInputError("from_margins expects a non-empty list of SemiParametricMargin")
        return cls(_rscopulas._Margins.concat([part._core for part in parts]))

    @property
    def dim(self) -> int:
        return int(self._core.dim)

    @property
    def margins(self) -> list[SemiParametricMargin]:
        """The margin of each column."""
        return [SemiParametricMargin(self._core.column(index)) for index in range(self.dim)]

    def cdf(self, x: npt.ArrayLike) -> _Array:
        """Raw ``(n, d)`` values to uniforms, ready for any copula model.

        Values are kept ``1e-12`` away from 0 and 1.
        """
        return np.asarray(self._core.cdf(_matrix(x, "x")))

    def quantile(self, u: npt.ArrayLike) -> _Array:
        """Uniforms ``(n, d)`` back to the raw scale.

        Use it on a sample drawn from a fitted copula. Values must lie
        strictly inside ``(0, 1)``.
        """
        return np.asarray(self._core.quantile(_matrix(u, "u")))

    def return_level(self, period: float) -> _Array:
        """The ``period``-observation return level of every column."""
        return np.asarray(self._core.return_level(float(period)), dtype=np.float64)

    def _repr_fields(self) -> list[tuple[str, str]]:
        return [("dim", str(self.dim))]


# ---------------------------------------------------------------------------
# Threshold diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MeanExcess:
    """Mean excess over each threshold, with its standard error."""

    threshold: _Array
    mean_excess: _Array
    std_error: _Array
    n_exceedances: npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ThresholdStability:
    """Generalized Pareto fit above each threshold.

    ``modified_scale`` is ``scale - shape * threshold``.
    """

    threshold: _Array
    shape: _Array
    shape_std_error: _Array
    scale: _Array
    modified_scale: _Array
    modified_scale_std_error: _Array
    n_exceedances: npt.NDArray[np.int64]


def default_thresholds(x: npt.ArrayLike, count: int = 40, min_exceedances: int = 30) -> _Array:
    """Candidate thresholds: sample quantiles from the median up to the level
    that leaves ``min_exceedances`` observations above it."""
    values = _vector(x, "x")
    if values.size <= 2 * min_exceedances:
        raise InvalidInputError(
            f"default thresholds need more than {2 * min_exceedances} observations; "
            "pass thresholds explicitly"
        )
    top = 1.0 - min_exceedances / values.size
    return np.unique(np.quantile(values, np.linspace(0.5, top, int(count))))


def mean_excess(x: npt.ArrayLike, thresholds: npt.ArrayLike | None = None) -> MeanExcess:
    """Mean residual life: the average excess over each threshold.

    Above a threshold from which the excesses are generalized Pareto, the
    mean excess is linear in the threshold. The lowest threshold from which
    the curve looks linear is a candidate. The slope is positive for a heavy
    tail, zero for an exponential tail, negative for a bounded one.

    ``thresholds`` defaults to 40 sample quantiles from the median up to the
    level that leaves 30 observations above it.
    """
    values = _vector(x, "x")
    levels = default_thresholds(values) if thresholds is None else _vector(thresholds, "thresholds")
    result = _rscopulas.mean_excess(values, levels)
    return MeanExcess(
        threshold=np.asarray(result["threshold"]),
        mean_excess=np.asarray(result["mean_excess"]),
        std_error=np.asarray(result["std_error"]),
        n_exceedances=np.asarray(result["n_exceedances"], dtype=np.int64),
    )


def threshold_stability(
    x: npt.ArrayLike, thresholds: npt.ArrayLike | None = None
) -> ThresholdStability:
    """Parameter stability: a generalized Pareto fit above each threshold.

    Above a valid threshold the shape and the modified scale stay constant
    up to sampling noise. The lowest threshold from which both are flat
    within their standard errors is a candidate.

    ``thresholds`` defaults as in :func:`mean_excess`.
    """
    values = _vector(x, "x")
    levels = default_thresholds(values) if thresholds is None else _vector(thresholds, "thresholds")
    result = _rscopulas.threshold_stability(values, levels)
    return ThresholdStability(
        threshold=np.asarray(result["threshold"]),
        shape=np.asarray(result["shape"]),
        shape_std_error=np.asarray(result["shape_std_error"]),
        scale=np.asarray(result["scale"]),
        modified_scale=np.asarray(result["modified_scale"]),
        modified_scale_std_error=np.asarray(result["modified_scale_std_error"]),
        n_exceedances=np.asarray(result["n_exceedances"], dtype=np.int64),
    )
