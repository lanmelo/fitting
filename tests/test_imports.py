"""Import-hygiene guards.

An import cycle only manifests when a particular module is imported *first*;
going through the package ``__init__`` can hide one, because whichever module
it happens to pull in first wins. These tests therefore import each submodule
on its own, in a fresh interpreter.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import fitting
from fitting import core

SUBMODULES = [
    "core",
    "fitting",
    "losses",
    "models",
    "results",
    "selectors",
    "utils",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports_standalone(name: str) -> None:
    """Each submodule must import on its own, in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import fitting.{name}"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
    )
    assert (
        proc.returncode == 0
    ), f"`import fitting.{name}` failed on its own:\n{proc.stderr}"


def test_public_api_is_importable() -> None:
    """Everything named in ``__all__`` must actually be present."""
    missing = [n for n in fitting.__all__ if not hasattr(fitting, n)]
    assert not missing, f"__all__ names nothing: {missing}"


def test_base_classes_live_with_their_subclasses() -> None:
    """A base class belongs beside its subclasses, not in a shared module."""
    assert fitting.CurveModel.__module__ == "fitting.models"
    assert fitting.Loss.__module__ == "fitting.losses"
    assert fitting.SingleExponentialInterval.__module__ == "fitting.models"
    assert fitting.Poisson.__module__ == "fitting.losses"


def test_core_holds_only_shared_vocabulary() -> None:
    """``core`` must not accumulate model- or loss-specific machinery."""
    public = {n for n in vars(core) if not n.startswith("_")}
    assert "CurveModel" not in public
    assert "Loss" not in public
