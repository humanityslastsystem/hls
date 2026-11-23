"""The node a test's stages run on. A stage context decides from what it reads of its node
(``hls.pipeline.stage``): a contestant step's process ceiling from the runner's hard
``RLIMIT_NPROC`` and the node's size, each sandbox's share of a per-uid count from the node's own
limits, and whether a contestant's end was starved from the uid's tasks, the node's tasks and its
``pid_max``. ``tests/conftest.py``'s ``hermetic_node`` gives every test the figures here in place of
this host's, so no stage decides by the machine the tests run on; a test gives its own by patching
the same names of ``hls.pipeline.stage``. The node's size is the launcher's
(``LocalLauncher.node_size``), ``tests.pipeline.fakes.NODE`` unless a test gives one.

A test whose own processes run under the ceiling the stage computes, as the agent build's sandbox
commands do through the real exec server and spawner, has this host's kernel set that ceiling,
and the kernel refuses one above the host's own hard limit: every command would then fail to
start. Such a test runs on the commands' node instead: the runner's hard limit ``COMMANDS_HARD``
holds the floor of up to ``COMMANDS_EVALUATIONS_MAX`` evaluations of the stage's size, and its
ceiling on a node of one evaluation, ``COMMANDS_CEILING``, is the largest it gives, under the hard limit of any machine the tests run on (31,633 on GitHub's ubuntu-24.04
runner). ``require_commands_ceiling`` fails such a test plainly on a host whose limit
is lower."""

from __future__ import annotations

import resource

import pytest

from hls.settings import RunSettings
from hls.substrate.nproc import FLOOR_PER_EVALUATION
from hls.substrate.ucounts import NAMESPACE_KINDS

#: The runner's hard ``RLIMIT_NPROC``: unlimited, so a contestant step's ceiling is the site's
#: cap (``site.nproc_max``) on a node of any size.
HARD = None
#: The node's own per-uid limits (``hls.substrate.ucounts.node_limits``), a large node's.
UCOUNTS = {**{f"max_{kind}_namespaces": 4_124_579 for kind in NAMESPACE_KINDS},
           "max_inotify_instances": 128, "max_inotify_watches": 1_048_576}
#: Every task of the uid on the node, whenever the stage counts them: far under any ceiling, so no
#: contestant step is refused its start or judged starved unless a test says so.
UID_TASKS = 400
#: Every task on the node, any user's, and the node's ``pid_max``.
NODE_TASKS, PID_MAX = 3_000, 4_194_304

#: The tasks kept for each evaluation's harness (``run.nproc_reserve``'s default).
RESERVE = RunSettings.model_fields["nproc_reserve"].default
#: The most evaluations of the stage's size a test's commands' node holds.
COMMANDS_EVALUATIONS_MAX = 4
#: The runner's hard limit on the commands' node: for each evaluation it may hold, the floor of
#: that evaluation's ceiling and its harness's reserve, 20,480 tasks.
COMMANDS_HARD = COMMANDS_EVALUATIONS_MAX * (FLOOR_PER_EVALUATION + RESERVE)
#: The process ceiling of a contestant step on a commands' node of one evaluation, the largest the
#: commands' node gives: the hard limit less that evaluation's reserve, 19,456 tasks.
COMMANDS_CEILING = COMMANDS_HARD - RESERVE
#: The node's kernel release (``hls.substrate.nproc.kernel_release``), one before 5.14: a stage
#: whose site sets no ``nproc_accounting`` counts the build sandbox's commands as ``uid``.
KERNEL = "4.18.0-553.el8_10.x86_64"


def require_commands_ceiling() -> None:
    """Fail the calling test when this host's hard ``RLIMIT_NPROC`` is under ``COMMANDS_CEILING``:
    its commands could not be given their ceiling, so none of them would start."""
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    if hard != resource.RLIM_INFINITY and hard < COMMANDS_CEILING:
        pytest.fail(f"this host's hard RLIMIT_NPROC of {hard} is under the process ceiling of {COMMANDS_CEILING} "
                    "the test's commands run under (tests.support.node.COMMANDS_CEILING)")
