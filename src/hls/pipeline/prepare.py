"""What ``hls run`` does before it plans its rollout: it finds what a scored run needs and
measures what is missing, so a run of a task's default configuration needs no earlier command.

First the run's checks that need nothing measured (``commands.run_checks``: the arm, the upload
and skills folders, a committed checkout, the task's data and drive bounds), so a run that would
be refused is refused before anything is measured. Then, in order, each step only when the run
lacks what it makes:

1. ``provision``: the task's data in the cache (``commands.planned_provision`` needs it to size
   the rollout's copies), by the provision warmer, ``hls provision``;
2. the baseline credit, then the calibration, each when the task loads it
   (``hls.measured.needed_files``) and neither the store nor the checkout holds one that fits the
   task (``hls.measured.find``): ``hls baseline-credit`` with one drive, then ``hls calibrate``
   with its defaults, each of which writes what it measured into the store's measured area;
3. ``oracle fill``: the reference answers the rollout's verify reads, when a completed fill has
   not answered every test prompt, and the dev reference published
   beside the dev rows, when the task declares one and neither the store nor the task's defaults
   hold one that fits (``hls.verify.dev_reference``). A build-only run grades nothing, so only a
   missing dev reference runs the fill for it.

Each step is the command a user can run alone, planned with the run's configuration and overrides,
so what it makes is what the rollout then finds; it says in one line what it runs and why, then
runs to its end in this process (``commands.start``). The images each step needs are built by that
step's own images stage, as the rollout's are. A step that does not succeed ends
the command with its exit status, and no rollout is planned. A probe or preflight-only build
measures nothing: it is not scored (``prepare`` is not called for it)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from hls.measured import BASELINE_CREDIT_FILE, DEV_REFERENCE_FILE, MEASURES, WHAT, find, needed_files
from hls.pipeline import commands
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError
from hls.pipeline.preflight import AnswersMissing
from hls.recipes import catalog
from hls.stages.provision import NotProvisioned, cached_provision
from hls.store.cache import ContentCache
from hls.store.posix import PosixStore
from hls.taskref import TaskRef

#: How a step's plan is driven to its end: ``commands.start``'s signature.
Start = Callable[..., int]
#: Rows an oracle fill asks per chunk, as ``hls oracle fill`` asks by default.
ORACLE_CHUNK = 96


def _step(what: str, plan: Callable[[], Plan], overrides: Mapping[str, str], start: Start,
          echo: commands.Echo) -> int:
    """Plan step *what* with *plan* and drive it to its end with *start*; its exit status. A
    ValueError while planning, how a command refuses what it was given, is raised again naming the
    step, so ``hls run`` refuses in one line as the step's own command would. A step that ends
    without succeeding says so in one line on stderr, naming its instance and what to run."""
    try:
        made = plan()
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None
    status = start(made, overrides, echo=echo)
    if status:
        print(f"hls run: {what} did not succeed ({made.instance}), so no rollout is planned: hls resume "
              f"{made.instance} drives it to its end, then hls run plans the rollout", file=sys.stderr, flush=True)
    return status


def prepare(*, config: Path, overrides: Mapping[str, str], task_module: str, arm: str, arm_change: str | None,
            build_only: bool = False, echo: commands.Echo, start: Start | None = None,
            start_credit: Start | None = None) -> int:
    """Find or make what a scored ``hls run`` of *task_module* with arm *arm* needs, with the run's
    *config* and *overrides* (the module docstring's steps); 0 when the run may be planned, else the
    exit status of the step that did not succeed. *start* drives a step's plan to its end,
    ``commands.start`` unless given; *start_credit* drives a baseline credit's,
    ``hls.pipeline.credit.run_credit`` unless given. Raise ArmError for an arm the run refuses and
    ValueError for what cannot be planned."""
    from hls.pipeline import calibration, credit

    start = start or commands.start
    start_credit = start_credit or credit.run_credit
    settings_over, task_over = commands.split_overrides(overrides)
    settings = commands.planning_settings(config, settings_over)
    task_id = commands.task_id_of(task_module)
    task_ref = TaskRef(module=task_module, overrides=task_over)
    task = commands.task_to_plan(task_ref)
    store = PosixStore(settings.site.store_root)
    commands.run_checks(settings=settings, store=store, code=commands.code_state(commands.checkout_root()),
                        task_module=task_module, task=task, arm=arm, arm_change=arm_change, build_only=build_only,
                        stop_after=None)
    try:
        cached_provision(task, ContentCache(store))
    except NotProvisioned:
        echo(f"{task_id}: the cache lacks the task's data, so it is provisioned first (hls provision {task_module})")
        status = _step("hls provision", lambda: commands.plan_provision(
            config=config, overrides=overrides, task_module=task_module, weights=True, job_name="hls-provision",
            echo=echo), overrides, start, echo)
        if status:
            return status
    needed = needed_files(task)
    for name in needed:
        if name == DEV_REFERENCE_FILE:
            # The oracle fill produces it, below.
            continue
        found = find(name, task_ref=task_ref, task=task, task_id=task_id, checkout=commands.checkout_root(),
                     store=store)
        if found.pin is not None:
            continue
        echo(f"{task_id}: {found.said()}, so it is measured first ({commands.measuring_command(name, task_ref)})")
        if name == BASELINE_CREDIT_FILE:
            status = _step(MEASURES[name], lambda: credit.plan_credit(
                config=config, overrides=overrides, task_module=task_module, job_name="hls-credit", echo=echo),
                overrides, start_credit, echo)
        else:
            status = _step(MEASURES[name], lambda: calibration.plan_calibrate(
                config=config, overrides=overrides, task_module=task_module, job_name="hls-calibrate", echo=echo),
                overrides, start, echo)
        if status:
            return status
        if find(name, task_ref=task_ref, task=task, task_id=task_id, checkout=commands.checkout_root(),
                store=store).pin is None:
            raise PlanError(f"{MEASURES[name]} succeeded, but the store holds no {WHAT[name]} that fits {task_id}")
    why = []
    if DEV_REFERENCE_FILE in needed:
        dev = find(DEV_REFERENCE_FILE, task_ref=task_ref, task=task, task_id=task_id,
                   checkout=commands.checkout_root(), store=store)
        if dev.pin is None:
            why.append(dev.said())
    if not build_only:
        try:
            commands.checked_for_grading(store, task_ref=task_ref, task_id=task_id, task=task,
                                         catalog_key=catalog.catalog_task(task_module), site=settings.site)
        except AnswersMissing as e:
            why.append(f"its reference answers are not all in the store: {e}")
    if not why:
        return 0
    echo(f"{task_id}: {'; '.join(why)}; so the oracle fill runs first (hls oracle fill {task_module})")
    status = _step("hls oracle fill", lambda: commands.plan_oracle_fill(
        config=config, overrides=overrides, task_module=task_module, chunk=ORACLE_CHUNK,
        job_name="hls-oracle-fill", echo=echo), overrides, start, echo)
    if status:
        return status
    if DEV_REFERENCE_FILE in needed and find(DEV_REFERENCE_FILE, task_ref=task_ref, task=task, task_id=task_id,
                                             checkout=commands.checkout_root(), store=store).pin is None:
        raise PlanError(f"hls oracle fill succeeded, but the store holds no dev reference that fits {task_id}")
    return 0
