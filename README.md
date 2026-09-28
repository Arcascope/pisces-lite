# pisces-lite

ML-backend-agnostic core for wearable sleep data: dataset discovery and loading,
accelerometer-to-spectrogram processing, cross-validation splitting, and metrics
logging. It carries no model framework, though there is a JAX-accelerated option
for the `[jax]` extra components.

## Install

```bash
pip install pisces-lite
```

Python 3.12+. To do the processing yourself rather than read a prebuilt feature
cache, install an extra — see [Optional extras](#optional-extras). The `proc`
and `jax` extras require a Python version the compiled backend publishes wheels
for (currently 3.12–3.14):

```bash
pip install "pisces-lite[proc]"
```

## Getting started

```python
from pisces_lite.datasets import DataSetObject, load_subject

# Discover datasets laid out as <dataset>/cleaned_<feature>/<subject>.csv
data_sets = DataSetObject.find_data_sets("data/")
ds = data_sets["mydata"]

# Load one subject: (accelerometer, PSG) frames with standardised columns
subject = load_subject(ds, ds.ids[0])
```

Datasets are described by an optional `data_set.json` next to the data; it sets
per-feature timestamp units, accelerometer scaling, PSG stage mappings, CSV
delimiters, and subject-id patterns. Non-standard layouts can ship a
`pisces_lite_adapter.py` exposing `load_data_sets(root, **kwargs)`.

Configure the processing pipeline and turn raw accelerometer into spectrograms:

```python
from pisces_lite.proc import ProcessingConfig

config = ProcessingConfig.from_dict({"type": "nufft", "fs": 32.0})
X = config.apply(accel_array, origin=psg_start)   # (T, F) or (T, F, C) with spectral_channels
```

### Frames and the window grid

senpy puts spectral window `k` at `[origin + k·hop, origin + k·hop + window)`
(`hop` is `window_step_seconds`) and reports every window, marking the ones
with too few samples. Frame `j` of `X` is centred at
`(window/2 mod hop) + j·hop` after the origin, so for 10 s windows every 2 s
the frames are centred at 1, 3, 5, … s. Window `k` fills frame
`k + floor((window/2)/hop)`, and the leading frames, centred before any window
could be, hold padding. A frame with no window behind it — the leading
frames, or a window with too little data, usually an accelerometer dropout —
holds `SPECTROGRAM_PADDING_VALUE` in every bin and channel;
`pisces_lite.proc.frame_validity(X)` finds them.

`origin` is where window 0 starts, in the unit and on the clock of the
accelerometer timestamp column:

- **Offline, with PSG:** pass the PSG start, e.g. the first timestamp of the
  PSG frame `pisces_lite.datasets.align_trim` returns. Frame `j` is then
  centred inside epoch `j // frames_per_epoch`, whatever the accelerometer's
  first sample was.
- **Online, no PSG:** pass `origin="unix"`, or set `"grid_origin": "unix"` in
  the config, to start the grid at a whole number of hops since the Unix
  epoch, so every session shares one grid.
- **Neither:** `origin=None` anchors the grid at the first sample, as before.

`apply_many(arrays, origins=[...])` takes one origin per recording.

Normalization leaves padding frames out of every statistic and returns them
still holding `SPECTROGRAM_PADDING_VALUE`, so `frame_validity` works on
normalized features too.

### Gap mode (opt-in)

Setting `gap_max_excluded_frame_fraction` marks a PSG epoch as the gap label
(`PAD_CLASS_LABEL`, -2) once more than that fraction of its frames are
padding. It is off (`null`) by default, which leaves labels unchanged.

```python
config = ProcessingConfig.from_dict({
    "type": "nufft", "window_seconds": 10, "window_step_seconds": 2,
    "gap_max_excluded_frame_fraction": 0.5,
})
X = config.apply(accel_array, origin=psg_start)
labels = config.mask_gap_epochs(labels, X)      # epoch 0 starts at psg_start
```

The last epoch of a recording always misses the frames within half a window
of its end, since no window is centred there; the 0.5 threshold tolerates that.
The pieces are also available on their own: `pisces_lite.proc.frame_validity`
and `pisces_lite.datasets.mask_labels_by_frame_coverage`.

Split subjects and score folds:

```python
from pisces_lite.cv import CVConfig, iter_folds
from pisces_lite.metrics import MetricsConfig, MetricsLogger

cv = CVConfig(mode="loso", train_sets=["mydata"], test_sets=["mydata"])
for fold in iter_folds(cv, data_sets):
    ...  # train on fold.train_subjects, evaluate fold.test_subjects
```

## Optional extras

The base install never pulls a C++ toolchain. Add an extra only when you need
the processing pipeline itself.

| Extra | Install | What it adds | Use when |
|---|---|---|---|
| `proc` | `pip install "pisces-lite[proc]"` | `arcascope-senpy` (import name `senpy`; compiled pybind11 + finufft), the CPU `streaming`/`cpu` NUFFT backends | You are turning raw accelerometer data into spectrograms |
| `jax` | `pip install "pisces-lite[jax]"` | `arcascope-senpy[jax]` and JAX, adding the `jax` NUFFT backend | You want GPU-batched spectrogram extraction |
| `dev` | `pip install "pisces-lite[dev]"` | `pytest` | You are running the test suite |

Working from a prebuilt feature cache — a training or inference host that only
reads `cv`, `metrics`, `model_io`, and `ProcessingConfig` — needs no extra.
`pisces_lite.proc` resolves its pipeline classes lazily, so importing a
`ProcessingConfig` does not import `senpy`; touching a pipeline class without
the extra raises a clear error naming the extra to install.

### JAX backend options

The `jax` backend transforms every channel of every recording passed to
`apply_many` in one call to senpy's `jax_backend.compute_nustft_many`.
`nufft_backend_kwargs` are passed through to it:

| Option | Default | Meaning |
|---|---|---|
| `rows_per_call` | `8192` | Windows per device call. `batch_size` is its deprecated old name. |
| `max_in_flight` | `3` | Device calls queued before the host waits on the oldest. |
| `build_threads` | `4` | Host threads building batches ahead of the device. |
| `eps` | `1e-6` | NUFFT tolerance. |
