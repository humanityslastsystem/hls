"""A task's provisioning published in a store's cache, as a provision warmer leaves it, for the
tests that plan a build or an evaluation: the planner sizes the stages that copy the task's data
by what the cache holds (``hls.stages.provision.cached_provision``).

``provision_into`` publishes a small stand-in of each entry the task needs, under the keys its
provisioning resolves to: the weights' ``config.json``, each tenant's dev and test rows with the
rows entry's ``ids.json``, and both variants of a tool bundle, the dev one keyed by those ids.
A test that serves or drives the data runs the real provision instead."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hls.provision.model import Provisioned, tool_key, weights_key
from hls.provision.role import IDS_FILE, resolve
from hls.provision.rows import dev_digest, tenant_subpath
from hls.stages.provision import cached_provision
from hls.store.cache import ContentCache
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from tests.store.entries import populate

#: The stand-in rows entry's row ids.
DEV_IDS, TEST_IDS = ["d0"], ["t0"]


def provision_into(store_root: Path, task: Any) -> Provisioned:
    """Publish in the cache under *store_root* every entry the provisioning of *task* (a module's
    ``TASK``) needs that it does not hold yet, as stand-ins, and return the provision record the
    planner reads (``cached_provision``)."""
    cache = ContentCache(PosixStore(store_root))
    holder = Holder.here("test provision")
    resolved = resolve(task)

    def publish(kind: str, key: str, files: dict[str, str]) -> None:
        if cache.lookup(kind, key) is not None:
            return

        def fill(d: Path) -> None:
            for rel, text in files.items():
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(text)

        populate(cache, kind, key, fill, holder=holder)

    model = resolved.weights
    publish("weights", weights_key(model.repo, model.revision), {"config.json": "{}"})
    if resolved.rows_key is not None:
        rows = {f"{tenant_subpath(t)}/{split}/rows.json": "[]" for t in resolved.tenants for split in ("dev", "test")}
        publish("rows", resolved.rows_key, {**rows, IDS_FILE: json.dumps({"dev": DEV_IDS, "test": TEST_IDS})})
        bundle = resolved.tools
        if bundle is not None:
            for variant, digest in (("full", None), ("dev", dev_digest(DEV_IDS, TEST_IDS))):
                publish("tools", tool_key(bundle.name, bundle.identity, variant, digest), {"bundle.txt": variant})
    return cached_provision(task, cache)


def provision_task(store_root: Path, task_module: str, overrides: Mapping[str, str] | None = None) -> Provisioned:
    """``provision_into`` *store_root* for the ``TASK`` of *task_module* with the ``--task.*``
    overrides among *overrides* applied, the task a plan of a command given *overrides* reads the
    provision record of."""
    from hls.pipeline.commands import split_overrides, task_id_of
    from hls.taskref import TaskRef, load_task

    task_id_of(task_module)
    _, task_over = split_overrides(overrides or {})
    return provision_into(store_root, load_task(TaskRef(module=task_module, overrides=task_over)))


def provision_module(store_root: Path, task_module: str) -> Provisioned:
    """``provision_into`` *store_root* for the ``TASK`` of *task_module*, imported as the planner
    imports it, from the checkout, and loaded as the planner loads it (``load_task``)."""
    from hls.pipeline.commands import task_id_of
    from hls.taskref import TaskRef, load_task

    task_id_of(task_module)
    return provision_into(store_root, load_task(TaskRef(module=task_module)))
