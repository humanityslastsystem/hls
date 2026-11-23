"""Planning an export (``hls export``): finished instances' publishable files copied out of the
store after a secret scan of every file of each, on the site's CPU partition when it has one
(``hls.stages.export``).

The login node reads each named instance's latest plan, to check that it exists, has ended,
belongs to the export's one task and is a kind an export publishes; adds the gradings under each
named rollout or warmer; checks the destination; and states the bytes the instances hold, by a
walk of their directory entries that reads no file. The stage's time is a fixed allowance
(``EXPORT_SHAPE``). It reads the secrets file for its names, since the scan searches for every
value of it, and refuses to plan without one, or with one that holds no secret."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from hls.pipeline.checkout import code_state
from hls.pipeline.commands import Echo, checkout_root, known_task_ids, planning_settings, site_secrets, split_overrides
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, STAGE_NAME_PATTERNS
from hls.pipeline.plan import Plan, PlanStore
from hls.pipeline.planner import PlanError, StageDecl, build_plan, toolchain_here
from hls.spec.resources import ResourceShape
from hls.stages.export import ExportStage
from hls.stages.finalize import tree_files
from hls.stages.package import PackageStage
from hls.store.layout import (EXPORTS, GRADED_AREAS, GRADINGS_DIR, WARMERS, InstanceKey, InstanceLayout,
                              NotAnInstance, new_rollout_id)
from hls.store.posix import PosixStore
from hls.verify.answers import answers_area

#: The export stage's shape: its two roles one after the other, each with 8 GiB, and the
#: runner's default reserve beside them. No disk: it writes only to the destination on the shared
#: filesystem. Its time is a fixed allowance of an hour, the scan and the copy each running to
#: the stage's deadline.
EXPORT_SHAPE = ResourceShape(cpus=2, memory_gib=12, time_sec=3600)


def exported_key(instance: str) -> InstanceKey:
    """The parts of *instance*, the root key of an instance an export publishes: a rollout, a
    grading under a rollout or a warmer, a warmer, a credit or a calibration, anything but an
    export. Raise ValueError for any other key."""
    try:
        key = InstanceLayout.parse(instance)
    except NotAnInstance as e:
        raise ValueError(f"{instance} is not an instance an export publishes: {e.why}") from None
    if key.area == EXPORTS:
        raise ValueError(f"{instance} is not an instance an export publishes: an export publishes a rollout, a "
                         "grading, a package warmer, a credit or a calibration, not an export")
    return key


def check_instance(store: PosixStore, instance: str) -> Plan:
    """The latest plan of *instance*, which must be in the store, have ended with its records
    written (an outcome not marked ``records_left``, whose zero and records are not in the store
    until ``hls resume`` writes them) and be a kind an export publishes: a warmer that is not a
    grading must hold a package stage. Raise ValueError otherwise."""
    key = exported_key(instance)
    plan = PlanStore(store, InstanceLayout(instance)).latest()
    if plan is None:
        raise ValueError(f"no instance {instance} in the store {store.root}")
    if plan.outcome is None:
        raise ValueError(f"{instance} has not ended; an export copies only finished instances")
    if plan.outcome.records_left:
        raise ValueError(f"{instance} ended failed, but its records are left; hls resume writes them first")
    if key.area == WARMERS and key.grading_id is None and not any(s.kind == PackageStage.kind for s in plan.stages):
        raise ValueError(f"{instance} is a warmer with no package stage; an export publishes a package warmer, "
                         "not a provision, images or oracle fill one")
    return plan


def with_gradings(store: PosixStore, instances: Sequence[str]) -> list[str]:
    """*instances*, each rollout or warmer followed by every grading under it that has a plan, in
    id order, each named once. Raise ValueError for a grading that is not exportable
    (``check_instance``)."""
    out: list[str] = []
    for instance in instances:
        found = [instance]
        key = InstanceLayout.parse(instance)
        if key.area in GRADED_AREAS and key.grading_id is None:
            for name in sorted(store.list_children(f"{instance}/{GRADINGS_DIR}")):
                try:
                    grading = InstanceLayout(instance).grading(name)
                except ValueError:
                    continue
                if PlanStore(store, grading).revisions():
                    check_instance(store, grading.root)
                    found.append(grading.root)
        out += [i for i in found if i not in out]
    return out


def instance_bytes(store: PosixStore, instance: str) -> int:
    """The bytes of every regular file of *instance*, but its gradings', which are instances of
    their own, by a walk of its directory entries (``tree_files``)."""
    files, _ = tree_files(store.path(instance), skip=store.path(f"{instance}/{GRADINGS_DIR}"))
    return sum(p.lstat().st_size for p in files)



def check_destination(destination: Path, *, store_root: Path, checkout: Path) -> Path:
    """*destination*, resolved through its parent, if it is an absolute path that does not exist,
    whose parent directory does, and that lies outside the store and the checkout as this node
    resolves them. Raise ValueError otherwise."""
    if not destination.is_absolute():
        raise ValueError(f"--to {destination} must be an absolute path")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"--to {destination} exists; an export writes a directory of its own")
    parent = destination.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"--to {destination}: its parent directory {parent} does not exist")
    target = parent / destination.name
    for inside, what in ((store_root.resolve(), "the store"), (checkout.resolve(), "the checkout")):
        if target.is_relative_to(inside):
            raise ValueError(f"--to {destination} lies inside {what} ({inside}); an export goes outside the store "
                             "and the checkout")
    return target


def split_arguments(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """*tokens*, the words of ``hls export`` that are none of its own options, as the instances
    they name and the ``--<dotted>`` overrides among them, each with its value: ``--a.b=v``, or
    ``--a.b`` and the word after it. No instance's key starts with ``--``."""
    instances: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            flags.append(token)
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                flags.append(tokens[i])
        else:
            instances.append(token)
        i += 1
    return instances, flags


def plan_export(*, config: Path, overrides: Mapping[str, str], instances: Sequence[str], destination: Path,
                oracle: bool, job_name: str, echo: Echo = print) -> Plan:
    """The first plan of an export of *instances* (and the gradings under each rollout or warmer)
    to *destination*, with the task's reference answers when *oracle*; nothing is submitted. Raise
    ValueError for what cannot be planned, a PlanError among it for a site with no secrets file or
    one that holds no secret."""
    if not instances:
        raise ValueError("name at least one instance to export")
    settings_overrides, task_overrides = split_overrides(overrides)
    if task_overrides:
        raise ValueError(f"an export runs no task's code, so it takes no --task.* overrides "
                         f"({', '.join(sorted(task_overrides))})")
    settings = planning_settings(config, settings_overrides)
    tasks = sorted({exported_key(i).task for i in instances})
    if len(tasks) != 1:
        raise ValueError(f"an export holds one task's instances, and these are of {', '.join(tasks)}")
    [task] = tasks
    store = PosixStore(settings.site.store_root)
    for instance in instances:
        check_instance(store, instance)
    named = with_gradings(store, instances)
    code = code_state(checkout_root())
    target = check_destination(destination, store_root=settings.site.store_root, checkout=code.checkout)
    area = answers_area(task) if oracle else None
    if area is not None and not store.path(area).is_dir():
        raise ValueError(f"--oracle: {task} has no reference answers in the store ({area})")
    nbytes = sum(instance_bytes(store, i) for i in named)
    if area is not None:
        nbytes += sum(p.lstat().st_size for p in tree_files(store.path(area))[0])
    decl = StageDecl(name="export", kind=ExportStage.kind, shape=EXPORT_SHAPE,
                     params={"instances": named, "oracle": oracle, "destination": str(target)})
    secrets = site_secrets(settings, echo, [decl])
    # The export stage needs every value of the file, so a missing one was refused just now.
    assert secrets is not None
    if not secrets.secrets:
        raise PlanError(f"the secrets file at {settings.site.secrets_path} holds no secret: an export searches "
                        "every file for the values of the secrets file, and with none it would copy files no scan "
                        "had read")
    echo(f"export of {len(named)} instance{'' if len(named) == 1 else 's'} of {task}, "
         f"{nbytes / 2**30:.1f} GiB, to {target}")
    plan = build_plan(decls=[decl], task=task, pipeline="export",
                      layout=InstanceLayout.export(task, new_rollout_id()), settings=settings, code=code,
                      toolchain=toolchain_here(), job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=known_task_ids(), now=time.time(), secrets=secrets, partition="cpu",
                      known_patterns=STAGE_NAME_PATTERNS)
    return plan


