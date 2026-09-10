"""Compute-bound bindings release the GIL and are safe to share across threads."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rscopulas import GaussianCopula, VineCopula

HEAVY_FAMILIES = ["gaussian", "clayton", "frank", "gumbel", "joe", "bb1"]


def _corr(dim: int, rho: float = 0.6) -> np.ndarray:
    return np.array([[rho ** abs(i - j) for j in range(dim)] for i in range(dim)])


@pytest.fixture(scope="module")
def heavy_data() -> np.ndarray:
    return GaussianCopula.from_params(_corr(6)).sample(3000, seed=1)


@pytest.fixture(scope="module")
def vine_model(heavy_data: np.ndarray) -> VineCopula:
    return VineCopula.fit_r(heavy_data, family_set=["gaussian", "frank"]).model


def test_background_thread_progresses_during_fit(heavy_data: np.ndarray) -> None:
    """A pure-Python thread keeps ticking while the main thread fits a vine.

    While the GIL is held for a whole fit the ticker manages only one or two
    ticks in total (the measured baseline before the bindings released the
    GIL was 1 tick during a 0.09 s fit); with the GIL released it ticks about
    once per millisecond of fit time, even with coarse 15 ms timer slices.
    """
    ticks = 0
    stop = threading.Event()
    started = threading.Event()

    def ticker() -> None:
        nonlocal ticks
        started.set()
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    worker = threading.Thread(target=ticker, daemon=True)
    worker.start()
    started.wait()
    time.sleep(0.05)

    before = ticks
    start = time.perf_counter()
    fits = 0
    # Keep fitting until at least half a second of Rust work has elapsed so the
    # measurement is meaningful on fast machines too.
    while time.perf_counter() - start < 0.5:
        VineCopula.fit_r(heavy_data, family_set=HEAVY_FAMILIES, include_rotations=True)
        fits += 1
    elapsed = time.perf_counter() - start
    during = ticks - before
    stop.set()
    worker.join()

    assert fits >= 1
    assert during >= 20, f"background thread made only {during} ticks during {elapsed:.2f}s of fitting"
    assert during >= 10 * elapsed, f"only {during / elapsed:.0f} ticks/s during fitting"


def test_concurrent_log_pdf_matches_serial(vine_model: VineCopula, heavy_data: np.ndarray) -> None:
    expected = vine_model.log_pdf(heavy_data)
    results: list[np.ndarray | None] = [None] * 4
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            results[index] = vine_model.log_pdf(heavy_data)
        except BaseException as exc:  # noqa: BLE001 - surfaced through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    for result in results:
        assert result is not None
        np.testing.assert_array_equal(result, expected)


def test_concurrent_sampling_with_same_seed_is_identical(vine_model: VineCopula) -> None:
    expected = vine_model.sample(2000, seed=8)
    results: list[np.ndarray | None] = [None] * 4

    def worker(index: int) -> None:
        results[index] = vine_model.sample(2000, seed=8)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for result in results:
        assert result is not None
        np.testing.assert_array_equal(result, expected)


def test_concurrent_fits_match_serial(heavy_data: np.ndarray) -> None:
    serial = VineCopula.fit_r(heavy_data, family_set=["gaussian", "frank"])
    results: list[float | None] = [None] * 3

    def worker(index: int) -> None:
        fit = VineCopula.fit_r(heavy_data, family_set=["gaussian", "frank"])
        results[index] = fit.diagnostics.loglik

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [serial.diagnostics.loglik] * 3
