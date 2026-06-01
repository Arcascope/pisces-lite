"""ETC ("easy to classify wake") model and scoring.

The ETC model is a Gaussian smoother applied to the per-timestamp sum of
spectrogram frequencies. It turns a spectrogram into a per-timestamp
wake-likelihood score using nothing but numpy primitives, so it can be applied
quickly anywhere with no extra dependency.

The **ETC score** of a data set is the AUROC of this model's wake score against
ground-truth labels in the ``(sleep=1, wake=0)`` class mode -- i.e. how easy the
wake epochs are to pick out from a crude energy smoother alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ETCModel:
    """A Gaussian-smoother "model" for wake likelihood.

    Attributes:
        n_timestamps: Kernel length, measured in input timestamps. The kernel is
            centred at ``n_timestamps // 2``.
        kernel_width: Width of the kernel as the fraction of ``n_timestamps``
            spanned by +/-1 standard deviation (i.e. by the full ``2 * std``
            interval). ``0.5`` means two standard deviations cover half the
            kernel: for ``n_timestamps=70`` the centre is index 35 and
            ``std = floor(0.5 * 70 / 2) = 17`` (so +/-1 sigma lands on 18 / 52).
    """

    n_timestamps: int
    kernel_width: float = 0.5

    def __post_init__(self) -> None:
        if self.n_timestamps < 1:
            raise ValueError("n_timestamps must be >= 1")
        if not (0.0 < self.kernel_width):
            raise ValueError("kernel_width must be > 0")

    @property
    def std(self) -> int:
        """Kernel standard deviation in timestamps (>= 1, rounded down)."""
        return max(1, int(np.floor(self.kernel_width * self.n_timestamps / 2)))

    @property
    def kernel(self) -> np.ndarray:
        """The Gaussian kernel, sampled from the normal PDF and summing to 1."""
        center = self.n_timestamps // 2
        idx = np.arange(self.n_timestamps, dtype=float)
        weights = np.exp(-0.5 * ((idx - center) / self.std) ** 2)
        return weights / weights.sum()

    def wake_proba(self, spectrograms: np.ndarray) -> np.ndarray:
        """Per-timestamp wake-likelihood score in ``[0, 1]``.

        Args:
            spectrograms: ``(T, F)`` for a single recording or ``(B, T, F)`` for
                a batch.

        Returns:
            ``(T,)`` or ``(B, T)`` scores, each recording normalised so its max
            over the time axis is ``1.0``.
        """
        specs = np.asarray(spectrograms, dtype=float)
        single = specs.ndim == 2
        if single:
            specs = specs[None]
        if specs.ndim != 3:
            raise ValueError(
                f"expected spectrograms of shape (T, F) or (B, T, F); got {specs.shape}"
            )

        energy = specs.sum(axis=-1)  # (B, T): sum over frequencies
        kernel = self.kernel
        # sliding dot product of the kernel along the time axis, always returning
        # exactly T samples (np.convolve 'same' would return max(T, len(kernel)),
        # which breaks when a recording is shorter than the kernel).
        offset = (len(kernel) - 1) // 2
        smoothed = np.stack(
            [
                np.convolve(row, kernel, mode="full")[offset : offset + row.shape[0]]
                for row in energy
            ]
        )
        # normalise each recording to a max of 1.0 over the time axis
        maxes = smoothed.max(axis=1, keepdims=True)
        maxes = np.where(maxes == 0.0, 1.0, maxes)
        out = smoothed / maxes
        return out[0] if single else out


def auroc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity, ties averaged.

    ``y_true`` is binary with ``1`` the positive class. Returns ``nan`` if either
    class is empty.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    # average ranks (1-based) over tied score groups
    ranks_sorted = np.arange(1, len(sorted_scores) + 1, dtype=float)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def etc_score(model: ETCModel, spectrograms, labels, reduce: str = "macro") -> float:
    """ETC score: AUROC of ``model``'s wake score in the ``(sleep=1, wake=0)`` mode.

    Args:
        model: The :class:`ETCModel` to apply.
        spectrograms: One recording ``(T, F)``, a batch ``(B, T, F)``, or a list
            of per-recording ``(T_i, F)`` arrays.
        labels: Matching ground-truth labels: ``(T,)``, ``(B, T)``, or a list of
            ``(T_i,)`` arrays. Any value ``> 0`` is treated as sleep, ``0`` as
            wake, and negative values (padding) are ignored.
        reduce: How to combine recordings. ``"macro"`` (default) computes a
            per-recording AUROC and averages those numbers, so every recording
            is weighted equally. ``"micro"`` pools every valid timestamp into a
            single AUROC, so longer recordings count for more.

    Returns:
        AUROC with sleep as the positive class, so a model that smooths energy
        well scores near ``1.0``. Recordings with only one class present
        (AUROC undefined) are skipped under ``"macro"``; returns ``nan`` if no
        recording yields a defined AUROC.
    """
    if reduce not in ("macro", "micro"):
        raise ValueError(f"reduce must be 'macro' or 'micro'; got {reduce!r}")

    specs = _iter_recordings(spectrograms, recording_ndim=2)
    labs = _iter_recordings(labels, recording_ndim=1)

    per_record: list[float] = []
    pooled_scores: list[np.ndarray] = []
    pooled_truths: list[np.ndarray] = []
    for spec, lab in zip(specs, labs):
        wake = model.wake_proba(np.asarray(spec))
        sleep_score = np.ravel(1.0 - wake)  # sleep-likelihood
        truth = np.ravel(np.asarray(lab))
        valid = truth >= 0
        y_bin = (truth[valid] > 0).astype(int)  # sleep=1, wake=0
        s = sleep_score[valid]
        if reduce == "macro":
            per_record.append(auroc_numpy(y_bin, s))
        else:
            pooled_scores.append(s)
            pooled_truths.append(y_bin)

    if reduce == "macro":
        defined = [a for a in per_record if not np.isnan(a)]
        return float(np.mean(defined)) if defined else float("nan")

    score = np.concatenate(pooled_scores) if pooled_scores else np.empty(0)
    truth = np.concatenate(pooled_truths) if pooled_truths else np.empty(0)
    return auroc_numpy(truth, score)


def _iter_recordings(data, recording_ndim: int):
    """Yield each recording from a single array, a batch, or a list of arrays.

    ``recording_ndim`` is the rank of one recording (2 for ``(T, F)``
    spectrograms, 1 for ``(T,)`` labels). An ndarray of that rank is a single
    recording; one extra leading axis is a batch to iterate over.
    """
    if isinstance(data, np.ndarray):
        if data.ndim <= recording_ndim:
            yield data
        else:
            yield from data
    else:  # list / tuple / other iterable of recordings
        yield from data
