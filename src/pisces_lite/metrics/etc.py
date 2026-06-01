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


def etc_score(model: ETCModel, spectrograms, labels) -> float:
    """ETC score: AUROC of ``model``'s wake score in the ``(sleep=1, wake=0)`` mode.

    Args:
        model: The :class:`ETCModel` to apply.
        spectrograms: One recording ``(T, F)``, a batch ``(B, T, F)``, or a list
            of per-recording ``(T_i, F)`` arrays.
        labels: Matching ground-truth labels: ``(T,)``, ``(B, T)``, or a list of
            ``(T_i,)`` arrays. Any value ``> 0`` is treated as sleep, ``0`` as
            wake, and negative values (padding) are ignored.

    Returns:
        Pooled AUROC across every valid timestamp in the data set; the positive
        class is sleep, so a model that smooths energy well scores near ``1.0``.
    """
    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    specs = _iter_recordings(spectrograms, recording_ndim=2)
    labs = _iter_recordings(labels, recording_ndim=1)
    for spec, lab in zip(specs, labs):
        wake = model.wake_proba(np.asarray(spec))
        scores.append(np.ravel(1.0 - wake))  # sleep-likelihood
        truths.append(np.ravel(np.asarray(lab)))

    score = np.concatenate(scores) if scores else np.empty(0)
    truth = np.concatenate(truths) if truths else np.empty(0)
    valid = truth >= 0
    y_bin = (truth[valid] > 0).astype(int)  # sleep=1, wake=0
    return auroc_numpy(y_bin, score[valid])


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
