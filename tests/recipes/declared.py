"""Fixture tasks' images, as the catalog reads a task module's ``IMAGES``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from hls.recipes import catalog
from hls.recipes.catalog import ImageSet, Served
from hls.recipes.model import LayerImage

#: sample_task's layer, which fixture tasks build on. Named here rather than read from sample_task's
#: module, which a role in a fixture checkout cannot import.
SAMPLE_LAYER = "s-a-tf-5-13-1"
#: The lock every fixture server image installs: a shared one, so that its key is known wherever
#: the fixture's images are keyed, a fixture task having no directory under ``tasks/``.
SERVER_LOCK = "layer-cuda-setuptools.txt"


def server(task: str, role: str) -> LayerImage:
    """The server image of fixture task *task*'s *role*, named ``<task with dashes>-<role>``: a
    server layer on the cuda base that ``declare`` makes a shared one, since a fixture task has no
    directory to keep its own locks in."""
    return LayerImage(name=f"{task.replace('_', '-')}-{role}", parent="cuda", pip_lock=SERVER_LOCK, server=True)


def image_set(task: str, roles: Sequence[str], *, entrypoint: Sequence[str] = ("python", "/app/server.py"),
              runtime_env: Mapping[str, str] | None = None) -> ImageSet:
    """Task *task*'s images: on sample_task's layer, a server image for each of *roles*
    (``baseline``, ``reference``, ``member-<engine>``), the baseline starting with *entrypoint*
    and *runtime_env*."""
    baseline = None
    if "baseline" in roles:
        baseline = Served(image=server(task, "baseline"), entrypoint=list(entrypoint),
                          runtime_env=dict(runtime_env or {}))
    return ImageSet(layer=SAMPLE_LAYER, baseline=baseline,
                    reference=server(task, "reference") if "reference" in roles else None,
                    members={r.removeprefix("member-"): server(task, r) for r in roles if r.startswith("member-")})


class _Sets(dict):
    """Task to its images, which registers each fixture server image (``server``) among the
    shared server layers as it is set."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, sets: dict[str, ImageSet]) -> None:
        super().__init__()
        self._monkeypatch = monkeypatch
        for task, images in sets.items():
            self[task] = images

    def __setitem__(self, task: str, images: ImageSet) -> None:
        for decl in images.roles().values():
            if isinstance(decl, LayerImage) and decl.owner is None and decl.name not in catalog.SERVER_LAYERS:
                self._monkeypatch.setitem(catalog.SERVER_LAYERS, decl.name, decl)
                self._monkeypatch.setitem(catalog.SHARED, decl.name, decl)
        super().__setitem__(task, images)


def declare(monkeypatch: pytest.MonkeyPatch, sets: dict[str, ImageSet]) -> dict[str, ImageSet]:
    """Make ``catalog.image_set`` answer *sets* (task to its images) for the tasks it names and
    every other task as its module declares; return *sets*, which a test may change in place. A
    fixture server image (``server``) a set names is a shared server layer while the test runs."""
    real = catalog.image_set
    declared = _Sets(monkeypatch, sets)
    monkeypatch.setattr(catalog, "image_set", lambda task: declared[task] if task in declared else real(task))
    return declared
