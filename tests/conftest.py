"""Shared pytest fixtures for the hls test suite."""
from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hls.cli import THREAD_POOL_VARIABLES
from hls.obs import logsetup
from hls.pipeline import stage
from hls.pipeline.orchestrator import Orchestrator
from hls.store.lock import LockLost
from hls.util import remove_tree
from tests.support import node


@pytest.fixture(autouse=True)
def hermetic_node(monkeypatch) -> None:
    """Every stage context reads the test node (``tests.support.node``), never this host: the
    runner's hard ``RLIMIT_NPROC``, the node's per-uid limits, the uid's and the node's tasks and
    the node's ``pid_max`` and kernel release. A test gives its own by patching the same names again."""
    monkeypatch.setattr(stage, "hard_limit", lambda: node.HARD)
    monkeypatch.setattr(stage, "node_limits", lambda: dict(node.UCOUNTS))
    monkeypatch.setattr(stage, "uid_tasks", lambda uid: node.UID_TASKS)
    monkeypatch.setattr(stage, "node_tasks", lambda: node.NODE_TASKS)
    monkeypatch.setattr(stage, "node_pid_max", lambda: node.PID_MAX)
    monkeypatch.setattr(stage, "kernel_release", lambda: node.KERNEL)


@pytest.fixture(autouse=True)
def _keep_thread_pool_sizes() -> Iterator[None]:
    """Put the thread-pool variables back as they were after every test. An orchestrator command
    run in the test's own process sizes them in its environment (``hls.cli``), and every later
    test's subprocesses would inherit them."""
    before = {name: os.environ.get(name) for name in THREAD_POOL_VARIABLES}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _close_orchestrators(monkeypatch):
    """Close every orchestrator a test made, so that no lease keep-alive thread and no console
    scrub outlives its test, and then hold the test to having left no console scrub registered:
    a scrub left behind would rewrite every later test's log lines. Orchestrators a test ran to
    the end are closed already, and one whose lease was taken over or lapsed on a test clock has
    nothing left to give up."""
    made: list[Orchestrator] = []
    init = Orchestrator.__init__

    def tracked(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Orchestrator, "__init__", tracked)
    yield
    for orch in made:
        with contextlib.suppress(LockLost):
            orch.close()
    left = list(logsetup._console_scrubs)
    logsetup._console_scrubs.clear()
    assert not left, f"{len(left)} console scrubs outlived the test's orchestrators: {left}"


@pytest.fixture
def warmed_servers(monkeypatch) -> Callable[[], None]:
    """Plans see each server image their images stage asks the cache about held, with the cuda
    base they build on, as on a cluster whose images warmers built every one: an entry is
    published in the store's cache for each such key a plan asks about, just before the cache is
    asked (``commands.cache_holds``). So a test of anything else plans no fill of them. Returns
    ``cold()``, which stops it: plans then see the cache as it is."""
    from hls.pipeline import commands
    from hls.recipes import catalog
    from tests.store.entries import hold_images

    real = commands.cache_holds

    def served(key: str) -> bool:
        name = key.rsplit("-", 1)[0]
        return name == "cuda" or name in catalog.SERVER_LAYERS

    def warming(settings):
        held = real(settings)

        def asked(keys):
            hold_images(settings.site.store_root, [key for key in keys if served(key)])
            return held(keys)

        return asked

    monkeypatch.setattr(commands, "cache_holds", warming)
    return lambda: monkeypatch.setattr(commands, "cache_holds", real)


@pytest.fixture
def short_dirs() -> Iterator[Callable[[], Path]]:
    """Makes fresh directories with short paths, for a test that binds a unix socket: the kernel
    takes at most 107 bytes of a socket's path, and a path under ``tmp_path`` can pass that when
    pytest is given a long ``--basetemp``. Each is removed when the test ends."""
    made: list[Path] = []

    def make() -> Path:
        made.append(Path(tempfile.mkdtemp(prefix="hls-")))
        return made[-1]

    yield make
    for path in made:
        remove_tree(path, ignore_errors=True)


@pytest.fixture
def short_dir(short_dirs: Callable[[], Path]) -> Path:
    """One directory from ``short_dirs``."""
    return short_dirs()


@pytest.fixture
def sample_taskspec_with_baseline():
    """A minimal, valid single-tenant TaskSpec with a Baseline attached.

    Mirrors the construction shape of ``tasks/sample_task/task.py`` (the smallest
    real single-tenant arena) but strips it down to the minimum the ``_check``
    validator requires: a workload + at least one verifier.
    """
    from hls.core.decode import Greedy
    from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
    from hls.reference.references import HFReference
    from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
    from hls.spec.baseline import Baseline
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.taskspec import TaskSpec
    from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
    from hls.verify import LetterMatchMetric, ReferenceRelativeVerifier

    model = ModelSpec(base_checkpoint="Qwen/Qwen2.5-0.5B-Instruct", arch="TX", chat_format="chatml")
    hardware = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
                         cpu_count=64, memory_gb=512, storage_gb=2048)
    contract = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
        transport="http")
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                         output_length_cap=ShortOutputCap(cap_tokens=4),
                         slo=NoSLO())
    verifiers = [ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                           reference=HFReference(decode=Greedy()), tol=0.0)]
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))

    return TaskSpec(id="sample_task_with_baseline", model=model, hardware=hardware,
                     contract=contract, workload=workload, verifiers=verifiers, scorer=scorer,
                     baseline=Baseline(image="ref-latest"))
