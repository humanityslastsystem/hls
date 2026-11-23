"""The measured baseline's sources, loaded from ``images/baseline/app`` as the image lays them out.

The image holds its servers at ``/app`` and the module they import at
``/app/tasks/deepresearch/serving_config.py``, with ``/app`` on its path. No module of the task
imports them, so a test loads each file by its path and gives a server the image's
``serving_config`` under the name it imports it by."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

#: The image's ``/app`` in this checkout.
APP = Path(__file__).resolve().parents[1] / "images" / "baseline" / "app"
#: The name the servers import their serving knobs by.
SERVING_CONFIG = "tasks.deepresearch.serving_config"


def _load(name: str, path: Path) -> ModuleType:
    """The module of the file at *path*, run from its source, so no bytecode file is written
    beside what the image copies in."""
    module = ModuleType(name)
    module.__file__ = str(path)
    # Under its name while it runs and after: a class it defines finds its module by that name.
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def serving_config() -> ModuleType:
    """The image's ``serving_config``, a fresh module on each call."""
    return _load("deepresearch_baseline_serving_config", APP / "tasks" / "deepresearch" / "serving_config.py")


def server(file: str, stand_ins: dict[str, ModuleType] | None = None) -> tuple[ModuleType, ModuleType]:
    """``/app/<file>`` of the image and the ``serving_config`` it imported: ``SERVING_CONFIG``
    names the image's copy only while the server loads. *stand_ins* are modules the server
    imports that this interpreter lacks (the engine), by name, in place for the load alone."""
    named = {SERVING_CONFIG: serving_config(), **(stand_ins or {})}
    before = {name: sys.modules.get(name) for name in named}
    sys.modules.update(named)
    try:
        return _load(f"deepresearch_baseline_{Path(file).stem}", APP / file), named[SERVING_CONFIG]
    finally:
        for name, module in before.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module
