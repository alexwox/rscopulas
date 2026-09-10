"""Rank-based pseudo-observations.

Every model in :mod:`rscopulas` consumes pseudo-observations: values strictly
inside ``(0, 1)`` that represent where each observation falls in its
marginal distribution. :func:`to_pseudo_obs` builds them from raw data with
a rank transform, which makes no assumption about the marginals.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from ._rscopulas import InvalidInputError

__all__ = ["TiesMethod", "Scaling", "to_pseudo_obs"]

TiesMethod = Literal["average", "min", "max", "ordinal"]
Scaling = Literal["n+1", "n"]

_TIES_METHODS: tuple[str, ...] = ("average", "min", "max", "ordinal")
_SCALINGS: tuple[str, ...] = ("n+1", "n")


def _rank(values: npt.NDArray[np.float64], ties: str) -> npt.NDArray[np.float64]:
    """Return 1-based ranks for one column using the requested tie rule."""
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    if ties == "ordinal":
        # Stable sort makes equal values rank in order of appearance.
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        return ranks

    sorted_values = values[order]
    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=is_group_start[1:])
    starts = np.flatnonzero(is_group_start)  # first sorted position of each tie group
    ends = np.append(starts[1:], n)  # one past the last position of each group
    group = np.cumsum(is_group_start) - 1  # tie group of each sorted position
    if ties == "min":
        group_rank = starts + 1.0
    elif ties == "max":
        group_rank = ends.astype(np.float64)
    else:  # average
        group_rank = (starts + 1.0 + ends) / 2.0
    ranks[order] = group_rank[group]
    return ranks


def to_pseudo_obs(
    x: npt.ArrayLike,
    *,
    ties: TiesMethod = "average",
    scaling: Scaling = "n+1",
) -> npt.NDArray[np.float64]:
    """Transform raw observations into rank-based pseudo-observations.

    Each column of ``x`` is ranked independently and the ranks are rescaled
    so that every value lies strictly inside ``(0, 1)``, ready for
    ``fit``/``log_pdf`` on any :mod:`rscopulas` model.

    Parameters
    ----------
    x
        A 1-D vector of length ``n`` or a 2-D matrix of shape ``(n, d)``.
        Anything ``numpy.asarray`` can turn into ``float64`` is accepted.
    ties
        How equal values share ranks: ``"average"`` (default) assigns the
        mean of the tied ranks, ``"min"`` the smallest, ``"max"`` the
        largest, and ``"ordinal"`` breaks ties by order of appearance so
        every rank is distinct. These match ``scipy.stats.rankdata``.
    scaling
        ``"n+1"`` (default) maps rank ``r`` to ``r / (n + 1)``; ``"n"``
        maps it to ``(r - 0.5) / n``. Both keep values away from the
        boundary.

    Returns
    -------
    ndarray
        ``float64`` array with the same shape as ``x``.

    Raises
    ------
    InvalidInputError
        If ``x`` contains NaN, is empty, is not 1-D or 2-D, or if ``ties``
        or ``scaling`` is not one of the supported options.

    Examples
    --------
    >>> import numpy as np
    >>> from rscopulas import to_pseudo_obs
    >>> to_pseudo_obs(np.array([3.0, 1.0, 2.0]))
    array([0.75, 0.25, 0.5 ])
    >>> to_pseudo_obs(np.array([1.0, 2.0, 2.0, 3.0]), ties="min")
    array([0.2, 0.4, 0.4, 0.8])
    """
    ties_method = str(ties).strip().lower()
    if ties_method not in _TIES_METHODS:
        raise InvalidInputError(
            f"unsupported ties method '{ties}'; expected one of {', '.join(_TIES_METHODS)}"
        )
    scaling_method = str(scaling).strip().lower().replace(" ", "")
    if scaling_method not in _SCALINGS:
        raise InvalidInputError(
            f"unsupported scaling '{scaling}'; expected one of {', '.join(_SCALINGS)}"
        )

    try:
        array = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"to_pseudo_obs expects numeric input convertible to float64: {exc}"
        ) from exc
    if array.ndim not in (1, 2):
        raise InvalidInputError(
            "to_pseudo_obs expects a 1-D vector or a 2-D (n, d) matrix, "
            f"got an array with ndim={array.ndim}"
        )
    matrix = array.reshape(-1, 1) if array.ndim == 1 else array
    n = matrix.shape[0]
    if n == 0:
        raise InvalidInputError("to_pseudo_obs expects at least one observation")

    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        rows, cols = np.nonzero(nan_mask)
        location = f"row {int(rows[0])}" if array.ndim == 1 else f"row {int(rows[0])}, column {int(cols[0])}"
        raise InvalidInputError(
            f"to_pseudo_obs cannot rank NaN values; found {int(nan_mask.sum())} NaN "
            f"entries (first at {location}). Drop or impute missing values before ranking."
        )

    ranks = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        ranks[:, column] = _rank(matrix[:, column], ties_method)

    if scaling_method == "n+1":
        uniforms = ranks / (n + 1.0)
    else:
        uniforms = (ranks - 0.5) / n
    return uniforms[:, 0] if array.ndim == 1 else uniforms
