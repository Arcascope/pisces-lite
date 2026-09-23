"""Access to the optional processing backend.

``pisces_lite.proc.processing`` imports senpy, a compiled extension carried by
the ``proc`` extra rather than by the base install. Every deferred route into
that module goes through :func:`load_processing`, so a missing extra reports
itself the same way wherever it is first touched.
"""
from __future__ import annotations

from types import ModuleType

_HINT = (
    'needs the processing backend, which is an optional dependency: install '
    '"pisces-lite[proc]" (or "pisces-lite[jax]" for the JAX NUFFT backend). '
    "Reading a ProcessingConfig or a prebuilt feature cache does not require it."
)


def load_processing(what: str) -> ModuleType:
    """Import ``pisces_lite.proc.processing``, or explain why it is unavailable.

    ``what`` names the caller, so the message points at the thing the user
    actually asked for rather than at an import line.
    """
    import importlib

    try:
        return importlib.import_module("pisces_lite.proc.processing")
    except ModuleNotFoundError as exc:
        if exc.name != "senpy":
            raise
        raise ModuleNotFoundError(f"{what} {_HINT}", name="senpy") from exc
