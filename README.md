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
X = config.apply(accel_array)   # (T, F) or (T, F, C) with spectral_channels
```

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
