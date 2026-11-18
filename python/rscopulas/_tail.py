"""Tail-dependence diagnostics.

Tail dependence asks whether the extremes of two variables arrive together.
The lower and upper coefficients are the limits

    lower = P(U2 <= q | U1 <= q)   as q -> 0
    upper = P(U2 >  q | U1 >  q)   as q -> 1

The functions here estimate them from data. Fitted models report the values
they imply through ``model.tail_dependence()``, so the two can be compared.
Every estimator uses only the ranks of its input, so raw observations and
pseudo-observations give the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _rscopulas
from ._rscopulas import InvalidInputError

__all__ = [
    "TailConcentration",
    "TailDependence",
    "TailDependenceCurve",
    "TailDependenceMatrix",
    "default_tail_k",
    "empirical_tail_concentration",
    "empirical_tail_dependence",
    "empirical_tail_dependence_curve",
    "empirical_tail_dependence_matrix",
]


@dataclass(frozen=True)
class TailDependence:
    """Lower and upper tail-dependence coefficients of one pair."""

    lower: float
    upper: float


@dataclass(frozen=True, eq=False)
class TailDependenceMatrix:
    """Pairwise coefficients: two symmetric ``(d, d)`` arrays, unit diagonal."""

    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class TailDependenceCurve:
    """Estimates at each number of tail observations ``k``."""

    k: npt.NDArray[np.int64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class TailConcentration:
    """Tail concentration function ``values`` at the levels ``q``."""

    q: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]


def _vector(data: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    try:
        array = np.ascontiguousarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be a 1-D vector, got ndim={array.ndim}")
    return array


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise InvalidInputError(f"{name} must be an integer, got {value!r}")
    return int(value)


def default_tail_k(n: int) -> int:
    """Number of tail observations used when ``k`` is not given: ``floor(sqrt(n))``.

    A small ``k`` uses only genuine extremes but averages over few points; a
    large ``k`` is stable but reaches into the body of the distribution.
    """
    return max(1, int(np.floor(np.sqrt(_count(n, "n")))))


def default_tail_levels() -> npt.NDArray[np.float64]:
    """Levels used when ``q`` is not given: 0.01, 0.02, ..., 0.99."""
    return np.arange(1, 100, dtype=np.float64) / 100.0


def empirical_tail_dependence(
    u1: npt.ArrayLike, u2: npt.ArrayLike, *, k: int | None = None
) -> TailDependence:
    """Estimate both tail-dependence coefficients of a pair.

    Uses the ``k`` smallest and the ``k`` largest observations of each
    variable (Schmidt and Stadtmüller, 2006)::

        lower = #{i : R_i <= k     and S_i <= k    } / k
        upper = #{i : R_i >  n - k and S_i >  n - k} / k

    where ``R_i`` and ``S_i`` are the ranks of ``u1`` and ``u2``.

    Parameters
    ----------
    u1, u2
        Two 1-D samples of the same length. Raw data or pseudo-observations.
    k
        Tail observations to use, ``1 <= k <= n``. Defaults to
        :func:`default_tail_k`. Use :func:`empirical_tail_dependence_curve`
        to check that the estimate is stable around the chosen ``k``.
    """
    first, second = _vector(u1, "u1"), _vector(u2, "u2")
    count = default_tail_k(first.shape[0]) if k is None else _count(k, "k")
    if count < 0:
        raise InvalidInputError("k must lie between 1 and the number of observations")
    lower, upper = _rscopulas.empirical_tail_dependence(first, second, count)
    return TailDependence(float(lower), float(upper))


def empirical_tail_dependence_curve(
    u1: npt.ArrayLike, u2: npt.ArrayLike, *, k: npt.ArrayLike | None = None
) -> TailDependenceCurve:
    """Estimate the coefficients at many ``k`` to find a stable range.

    The estimates are noisy for small ``k`` and drift toward the body of the
    distribution for large ``k``. A plateau in between is the value to read.

    ``k`` defaults to ``1, 2, ..., n // 5``.
    """
    first, second = _vector(u1, "u1"), _vector(u2, "u2")
    if k is None:
        counts = np.arange(1, max(first.shape[0] // 5, 1) + 1, dtype=np.int64)
    else:
        raw = np.asarray(k)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise InvalidInputError("k must be a 1-D sequence of integers")
        if (raw < 1).any():
            raise InvalidInputError("k must lie between 1 and the number of observations")
        counts = raw.astype(np.int64)
    lower, upper = _rscopulas.empirical_tail_dependence_curve(first, second, counts.tolist())
    return TailDependenceCurve(counts, np.asarray(lower), np.asarray(upper))


def empirical_tail_dependence_matrix(
    data: npt.ArrayLike, *, k: int | None = None
) -> TailDependenceMatrix:
    """Estimate the coefficients for every pair of columns of ``data``.

    ``data`` is an ``(n, d)`` matrix of raw data or pseudo-observations.
    """
    try:
        matrix = np.ascontiguousarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"data must be numeric: {exc}") from exc
    if matrix.ndim != 2:
        raise InvalidInputError(f"data must be a 2-D (n, d) matrix, got ndim={matrix.ndim}")
    count = default_tail_k(matrix.shape[0]) if k is None else _count(k, "k")
    if count < 0:
        raise InvalidInputError("k must lie between 1 and the number of observations")
    lower, upper = _rscopulas.empirical_tail_dependence_matrix(matrix, count)
    return TailDependenceMatrix(np.asarray(lower), np.asarray(upper))


def empirical_tail_concentration(
    u1: npt.ArrayLike, u2: npt.ArrayLike, *, q: npt.ArrayLike | None = None
) -> TailConcentration:
    """Empirical tail concentration function.

    For ``q <= 0.5`` this is the share of observations with both variables
    below their ``q`` quantile, divided by ``q``; above ``0.5`` it is the
    share with both above, divided by ``1 - q``. Its ends approach the two
    tail-dependence coefficients. Compare with
    ``PairCopula.tail_concentration(q)`` of a fitted model to see whether the
    family follows the data into both tails.

    ``q`` defaults to ``0.01, 0.02, ..., 0.99``; every level must lie
    strictly inside ``(0, 1)``.
    """
    first, second = _vector(u1, "u1"), _vector(u2, "u2")
    levels = default_tail_levels() if q is None else _vector(q, "q")
    values = _rscopulas.empirical_tail_concentration(first, second, levels)
    return TailConcentration(levels, np.asarray(values))
