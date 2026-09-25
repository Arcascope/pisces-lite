"""Throughput-oriented packing for the JAX NUSTFT (``packing: "vectorized"``).

The default ``senpy`` packing (``senpy.jax_backend.pack_nustft_window_batches``)
discovers windows with a Python loop per window, copies each window into its
batch row by row, then runs every batch synchronously: the host blocks on
batch ``k`` before it starts building batch ``k + 1``. With small batches each
GPU call lasts a few milliseconds, so the device idles most of the time.

This path keeps the numerics -- the same senpy jitted transform, the same
window grid, the same ``(nfft_padded, source_width)`` bucketing, the same
host-side spectral surface -- and changes only how work reaches the device:

* window discovery and row gathers are vectorized NumPy;
* host buffers are built in float32, the dtype JAX would cast them to anyway;
* batches are larger (``rows_per_call``) and dispatched asynchronously, with a
  bounded number in flight (``max_in_flight``), so building the next batch on
  ``build_threads`` host threads overlaps the current transform;
* frequency bins above ``target_fs / 2`` are dropped on the device, before the
  device-to-host copy.

It uses senpy's private ``_nustft_window_batch_transform``, so it is tied to
the pinned senpy version.
"""
from __future__ import annotations

import collections
from functools import lru_cache
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Rows per device call. The senpy default of 128 (and inputs.json's 512)
# leaves a large GPU mostly idle; memory per row is ~0.2 MB of intermediates
# at eps=1e-6 and 512-sample windows.
DEFAULT_ROWS_PER_CALL = 8192
MAX_IN_FLIGHT = 3


def _next_power_of_two(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    out = np.ones_like(value)
    mask = value > 1
    out[mask] = np.left_shift(1, np.ceil(np.log2(value[mask])).astype(np.int64))
    # Guard against log2 rounding at exact powers of two.
    too_small = out < value
    out[too_small] <<= 1
    return out


def _window_starts(t_end: float, window_s: float, hop_s: float, median_fs: float) -> np.ndarray:
    """Reproduce ``start += hop_s`` accumulation from senpy's packer exactly."""
    limit = t_end + 1.0 / median_fs
    # Upper bound on the count, then trim with the same comparison the loop uses.
    n_max = int(max(0.0, (limit - window_s) / hop_s)) + 2
    starts = np.add.accumulate(np.concatenate([[0.0], np.full(n_max, hop_s)]))
    return starts[starts + window_s <= limit]


def _discover_windows(timestamps: np.ndarray, samples: np.ndarray, window_s: float, overlap_s: float):
    t = np.asarray(timestamps, dtype=np.float64)
    s = np.asarray(samples)
    if t.ndim != 1 or s.ndim != 2 or s.shape != (t.size, 3):
        raise ValueError("each recording must be a (timestamps[N], samples[N, 3]) pair")
    if t.size < 2:
        raise ValueError("each recording requires at least two timestamps")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(s)):
        raise ValueError("timestamps and samples must be finite")
    t = t - t[0]
    diffs = np.diff(t)
    if np.any(diffs < 0.0):
        raise ValueError("timestamps must be sorted")
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        raise ValueError("timestamps must contain at least one positive time step")
    median_fs = 1.0 / float(np.sort(positive)[positive.size // 2])
    nfft = int(window_s * median_fs)
    if nfft < 2:
        raise ValueError("window_s is too short for the observed sampling density")
    nfft_padded = int(_next_power_of_two(np.array([nfft]))[0])

    starts = _window_starts(float(t[-1]), window_s, window_s - overlap_s, median_fs)
    first = np.searchsorted(t, starts, side="left")
    last = np.searchsorted(t, starts + window_s, side="left")
    counts = last - first
    keep = counts >= 4
    window_indices = np.flatnonzero(keep)
    return dict(
        t=t,
        s=s,
        median_fs=median_fs,
        nfft_padded=nfft_padded,
        starts=starts[keep],
        first=first[keep],
        counts=counts[keep],
        widths=_next_power_of_two(counts[keep]),
        window_indices=window_indices,
        times=starts[keep] + window_s / 2.0,
    )


@lru_cache(maxsize=None)
def _trimmed_transform(nfft_padded: int, detrend: bool, eps: float, n_keep: int):
    """senpy's cached batch transform, followed by an on-device bin trim."""
    import jax
    from senpy import jax_backend as senpy_jax

    inner = senpy_jax._nustft_window_batch_transform(nfft_padded, detrend, eps)

    @jax.jit
    def transform(points, signals, valid, median_fs):
        return inner(points, signals, valid, median_fs)[..., :n_keep]

    return transform


def _padded_rows(n: int, rows_per_call: int) -> int:
    """Pad a partial batch to one of a few fixed row counts.

    Every distinct shape is an XLA compile (~0.1 s each); a short menu keeps
    that to a handful per run at the cost of some zero rows, which the GPU
    handles far faster than a compile.
    """
    for size in (rows_per_call // 8, rows_per_call // 2):
        if n <= size:
            return size
    return rows_per_call


def compute_packed_jax_recording_spectrograms(
    *,
    recordings: Sequence[Tuple[np.ndarray, Sequence[np.ndarray]]],
    window_s: float,
    overlap_s: float,
    target_fs: Optional[float],
    kind: str,
    detrend: bool,
    backend_kwargs: Optional[dict],
    rows_per_call: Optional[int] = None,
    max_in_flight: int = MAX_IN_FLIGHT,
    build_threads: int = 4,
):
    import senpy
    from pisces_lite.proc.processing import _spectral_surface

    kwargs = dict(backend_kwargs or {})
    eps = float(kwargs.get("eps", 1e-6))
    rows_per_call = int(rows_per_call or kwargs.get("rows_per_call", DEFAULT_ROWS_PER_CALL))

    # Pack channels in groups of three, exactly as pisces-lite does.
    groups = []  # (recording_index, channel_start, group_size, windows)
    for recording_index, (timestamps, signals) in enumerate(recordings):
        signals = [np.asarray(signal) for signal in signals]
        if not signals:
            raise ValueError("each JAX recording requires at least one signal")
        for channel_start in range(0, len(signals), 3):
            group = signals[channel_start : channel_start + 3]
            group_size = len(group)
            group.extend(np.zeros_like(group[0]) for _ in range(3 - group_size))
            windows = _discover_windows(timestamps, np.column_stack(group), window_s, overlap_s)
            windows["t_pad"] = np.append(windows["t"], 0.0)
            windows["sT_pad"] = np.concatenate(
                [windows["s"].T.astype(np.float32), np.zeros((3, 1), np.float32)], axis=1
            )
            groups.append((recording_index, channel_start, group_size, windows))

    # Bucket rows across all groups by (nfft_padded, source_width).
    buckets: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = collections.defaultdict(list)
    for group_index, (_, _, _, w) in enumerate(groups):
        for width in np.unique(w["widths"]):
            rows = np.flatnonzero(w["widths"] == width)
            buckets[(w["nfft_padded"], int(width))].append((group_index, rows))

    n_keep_by_nfft: Dict[int, int] = {}
    freqs_by_nfft: Dict[int, np.ndarray] = {}
    for nfft_padded in {key[0] for key in buckets}:
        freqs = np.arange(nfft_padded // 2 + 1, dtype=np.float64) / window_s
        if target_fs is not None:
            freqs = freqs[freqs <= target_fs / 2.0 + np.finfo(float).eps]
        n_keep_by_nfft[nfft_padded] = freqs.size
        freqs_by_nfft[nfft_padded] = freqs

    # Output buffers per group, filled in window order.
    coeff_out = [
        np.empty((w["first"].size, 3, n_keep_by_nfft[w["nfft_padded"]]), dtype=np.complex64)
        for (_, _, _, w) in groups
    ]

    def batches():
        for (nfft_padded, width), members in sorted(buckets.items()):
            # Flatten (group, row) pairs for this bucket, then chunk.
            gidx = np.concatenate([np.full(rows.size, g) for g, rows in members])
            ridx = np.concatenate([rows for _, rows in members])
            for c0 in range(0, gidx.size, rows_per_call):
                yield nfft_padded, width, gidx[c0 : c0 + rows_per_call], ridx[c0 : c0 + rows_per_call]

    def build(nfft_padded, width, gidx, ridx):
        n = gidx.size
        b = _padded_rows(n, rows_per_call)
        points = np.zeros((b, width), dtype=np.float32)
        sig = np.zeros((b, 3, width), dtype=np.float32)
        valid = np.zeros((b, width), dtype=bool)
        fs = np.ones(b, dtype=np.float32)
        lane = np.arange(width)
        for g in np.unique(gidx):
            sel = np.flatnonzero(gidx == g)
            w = groups[g][3]
            r = ridx[sel]
            m = lane[None, :] < w["counts"][r][:, None]
            # Padded lanes point at a trailing zero sample, so no masking pass
            # is needed on the gathered signals.
            idx = np.where(m, w["first"][r][:, None] + lane[None, :], w["t"].size)
            local = w["t_pad"][idx]
            local -= w["starts"][r][:, None]
            local *= 2.0 * np.pi / window_s
            local -= np.pi
            local[~m] = 0.0
            points[sel] = local
            for c in range(3):
                sig[sel, c] = w["sT_pad"][c][idx]
            valid[sel] = m
            fs[sel] = w["median_fs"]
        return points, sig, valid, fs

    in_flight: Deque = collections.deque()

    def drain_one():
        result, gidx, ridx = in_flight.popleft()
        host = np.asarray(result)[: gidx.size]
        for g in np.unique(gidx):
            sel = np.flatnonzero(gidx == g)
            coeff_out[g][ridx[sel]] = host[sel]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=build_threads) as pool:
        pending: Deque = collections.deque()
        work = iter(batches())
        for item in work:
            pending.append((item, pool.submit(build, *item)))
            if len(pending) >= build_threads * 2:
                break
        while pending:
            (nfft_padded, width, gidx, ridx), future = pending.popleft()
            nxt = next(work, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(build, *nxt)))
            fn = _trimmed_transform(nfft_padded, bool(detrend), eps, n_keep_by_nfft[nfft_padded])
            in_flight.append((fn(*future.result()), gidx, ridx))
            if len(in_flight) >= max_in_flight:
                drain_one()
    while in_flight:
        drain_one()

    results: List[List[Optional["senpy.SpectrogramResult"]]] = [
        [None] * len(signals) for _, signals in recordings
    ]
    for group_index, (recording_index, channel_start, group_size, w) in enumerate(groups):
        if w["first"].size == 0:
            raise ValueError(
                "JAX packed NUSTFT requires enough data for at least one window "
                f"in recording {recording_index}"
            )
        coeffs = coeff_out[group_index]
        for channel_index in range(group_size):
            results[recording_index][channel_start + channel_index] = senpy.SpectrogramResult(
                frequencies=freqs_by_nfft[w["nfft_padded"]],
                times=w["times"].astype(np.float64),
                Sxx=_spectral_surface(coeffs[:, channel_index, :], kind),
                kind=kind,
                method="jax_finufft_packed",
            )
    return [[spec for spec in recording if spec is not None] for recording in results]
