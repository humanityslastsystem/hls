"""Coding-agent tasks as trajectory seeds, from Terminal-Bench 4.0.

One corpus row is one upstream task, and one task is one trajectory: the task's
``instruction.md`` seeds the conversation and the task's own container image is the
shell the model works in. The dataset carries no prompt of its own -- the framework
owns the system and instance templates, because they are the agent's design and a
change to them changes the traffic the gate was calibrated on.

``row_id`` is the task's upstream name rather than a positional index. It is the name
the task's image is keyed on and the name a removal list would use, and a positional
one would silently repoint at a different task if the upstream release were re-cut.

Why this corpus
---------------

The task measures a server under a long-horizon coding agent, so what matters is that
a trajectory keeps going: long prompts that grow turn over turn.

Terminal-Bench 4.0 sets every one of its 66 tasks an 8-hour agent timeout against a
median 4-hour expert time estimate, and publishes per-trial token totals measured
under the same mini-SWE-agent loop this task runs.

Depth inside a drive is a fact about the drive window rather than about these tasks. A
drive issues requests for 45 minutes, which at this stack's rate is 55 to 140
turns, so the deepest bands are out of reach inside a window whatever the tasks are.
That is what ``TerminalBenchDataset.deep_rows`` is for: a few rows resume a pre-rolled
conversation instead of starting at turn 0.

What a row carries and why
--------------------------

The instruction is the seed. The working directory is recorded because it is a
property of the task's image -- most are built around ``/app``, a few are not -- and a
container asked to start somewhere its image has no such directory fails to launch.
``source`` names the corpus the row came from, so a trace reads back without guessing.
The remaining fields are recorded so a trace can be read back without the checkout.

Where the corpus comes from
---------------------------

A local checkout of the upstream repository at a pinned tag, because it ships its
tasks as a git tree rather than a Hub dataset::

    git clone --depth 1 --branch v4.0.0 \\
        https://github.com/harbor-framework/terminal-bench

The organisation matters. ``laude-institute/terminal-bench`` is harbor's registry
entry for Terminal-Bench 2.0, a different repository with no tags at all, whose tasks
are laid out in an older format this reader cannot parse.

The checkout is checked rather than trusted. ``RELEASE_DIGEST`` pins what the task
definitions hash to, and ``corpus()`` refuses a checkout that does not match.

``corpus()`` reads ``tasks/`` out of the checkout. The repository's content is not
vendored here. The per-task container images are staged separately and are not read
here; ``task.py`` routes a session to its image by task name.
"""
from __future__ import annotations

import atexit
import functools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Iterator

from pydantic import Field, field_validator, model_validator

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.spec.datasets import Dataset, Split

#: The upstream release the corpus is read from, for a human reading a trace.
RELEASE_TAG = "v4.0.0"

#: The commit that tag points at, and what the corpus is actually identified by.
#:
#: A tag is a moving pointer. ``laude-institute/terminal-bench`` is harbor's registry
#: entry for Terminal-Bench 2.0 and ships its tasks in an older layout this reader
#: cannot parse. The corpus comes from ``harbor-framework/terminal-bench``, where
#: ``v4.0.0`` resolves here. A commit is what a checkout can be compared against, so
#: this rather than the tag is what reaches ``corpus_id``.
RELEASE_COMMIT = "452bf305c6daa62fc59061d22133a7cbc7c1572e"

#: The files whose contents a task's digest covers: what a corpus row is built from,
#: plus the file that says how the task's image should be built.
#:
#: Not the whole task directory. That is about half a gigabyte of image build data, and
#: hashing it at every provisioning on a slow shared filesystem would cost more than
#: the problem.
#:
#: **The Dockerfile pins the recipe, not the image.** A trajectory does not run this
#: file; it runs a ``.sif`` routed to by task name (``task.py``), the conversion of the
#: image the release built from the task's environment and published. This check catches
#: a checkout that is not the release. The image is pinned on its own, by the digest of
#: its published manifest (``RELEASE_IMAGES_FILE``).
#:
#: Also uncovered here: the data baked into an image, which that digest covers.
DIGESTED_FILES = ("task.toml", "instruction.md", "environment/Dockerfile")

#: Digest of the release's 66 task definitions, over ``DIGESTED_FILES``.
#:
#: Without it, a checkout at another commit, or one a local edit had touched, would
#: provision and grade without a word. The task count passes for any 66 directories.
#:
#: Over all 66 definitions rather than the admitted subset, so that changing an
#: exclusion list does not churn this pin. What it is for is noticing the release
#: moved; which tasks are excluded is pinned separately.
#:
#: Computed from a checkout at ``RELEASE_COMMIT``. Regenerate with
#: ``release_digest(tasks_dir())`` after a deliberate release change, never to make a
#: failure go away: a mismatch means the checkout is not the release this was
#: calibrated against.
RELEASE_DIGEST = "109a48afbe4adb9177633a82675ca68fe9d157760fc5b49882658efa8b25e040"

#: Where the per-task digests live. Generated data, so a file beside this module rather
#: than sixty-six hashes in it. Read only on the way to failing, to name which tasks
#: differ; the pin the corpus is gated on is ``RELEASE_DIGEST``, which is a declared
#: field and reaches the calibration's workload fingerprint as a file could not.
RELEASE_DIGESTS_FILE = Path(__file__).with_name("release_digests.json")

#: Where the digests of the release's published environment images live, one per task.
#: Generated data: the ``environment`` entries of the image manifest the release attaches
#: (``prebuilt-images.json``), which names, for every task, the image the release built for
#: linux/amd64 from the task's environment at ``RELEASE_COMMIT`` and published under a
#: digest. A session runs that image, converted to a ``.sif`` (``task.SANDBOX_IMAGES``); it
#: is pulled by this digest, so every site holds the same one.
RELEASE_IMAGES_FILE = Path(__file__).with_name("release_images.json")

#: Tasks the release ships. Asserted against the checkout so a truncated or wrong-tag
#: clone fails at provisioning rather than quietly yielding a short corpus.
RELEASE_TASK_COUNT = 66

#: Where the release is fetched from, at ``RELEASE_COMMIT``.
RELEASE_URL = "https://github.com/harbor-framework/terminal-bench"

#: Tasks whose container image could not be built, and which the corpus therefore does
#: not carry. Excluding them here rather than letting them fail at run time keeps a
#: trajectory's failure meaningful: every row that reaches a split has an image, so a
#: launch that fails is a fault to triage rather than a known gap.
#:
#: Under Apptainer 1.3 with rootless fakeroot, three fail converting their Docker
#: build to a ``.sif`` and five fail the build itself. A failed build leaves a partial
#: ``.sif`` behind, so a file on disk is not evidence the image is whole; the build
#: log is. A rebuild that fixes one is free to shorten the set.
NO_IMAGE = frozenset({
    # conversion failed
    "data-anonymization",
    "mp-checkpoint-consolidation",
    "risk-scorer-replay",
    # build failed, partial .sif left on disk
    "bun-sourcemap-leak",
    "coq-block-bound",
    "lake-temp-glm",
    "retro-console-soc",
    "wdm-design",
})

#: Tasks that declare a GPU, and which the corpus therefore does not carry.
#:
#: A separate set from ``NO_IMAGE`` rather than merged into it, because the reasons are
#: different and each one is worth reading on its own. A task here builds and launches
#: perfectly well; it simply cannot do the work it was set. The sandbox has no
#: accelerator to give it: this node's eight H200s belong to the server under
#: measurement, not to the shells driving it.
#:
#: Excluding them is about what the traffic is, not about whether a run survives. A
#: trajectory on a task that cannot succeed still produces turns, and those turns are
#: an agent flailing at an impossible problem rather than working a solvable one. That
#: is not the load this task exists to measure.
#:
#: Read off the release's own ``[environment]`` blocks, not guessed
#: from names: nothing in ``math-eval-grader`` says it wants a GPU.
NEEDS_ACCELERATOR = frozenset({
    "fp8-rmsnorm-gemm",
    "jax-speedrun-gpu",
    "math-eval-grader",
})

#: Fields a seed carries onto the envelope's metadata. ``task_name`` routes the sandbox
#: to the task's own image and ``workdir`` to the directory that image was built
#: around; the rest are recorded so a trace can be read back without the checkout.
#:
#: Every row fills all of them, so the corpus is one Arrow schema and a seed is the
#: same shape wherever in the release it came from.
TASK_NAME_FIELD = "task_name"
INSTRUCTION_FIELD = "instruction"
WORKDIR_FIELD = "workdir"
CATEGORY_FIELD = "category"
EXPERT_HOURS_FIELD = "expert_time_estimate_hours"
SOURCE_FIELD = "source"
#: Whether the task's own definition wants the sandbox to have a network. Carried on
#: the row rather than decided once for the run: Terminal-Bench leaves it unset and is
#: written expecting a network, and a corpus whose tasks declared otherwise would want
#: to say so per row rather than per run.
ALLOW_INTERNET_FIELD = "allow_internet"
#: Digest of the frozen conversation this task's trajectory resumes from, or "" for a
#: task that starts at turn 0. The framework loads the staged prefix by task name and
#: refuses one that hashes to anything else.
#:
#: The digest rather than the conversation, because rows are built where the corpus is
#: provisioned and prefixes are staged where the run happens: the row has to name what
#: it expects without being able to read it.
PREFIX_DIGEST_FIELD = "prefix_digest"
#: Turns this task's session serves, or 0 for the framework's own budget. Per task
#: because a turn at 600k of prompt costs far more than one at 5k, so a uniform budget
#: would let the deep trajectories consume the whole drive. Set for the deep rows and
#: left at 0 for the rest, whose trajectories end on their own.
TURN_BUDGET_FIELD = "turn_budget"

#: What this task's own definition says its container needs. Read off the release's
#: ``[environment]`` block rather than fixed for the run, because these span an
#: eightfold range in CPUs and an eightfold range in memory, and one figure for all of
#: them is either the largest -- which bounds how many trajectories run at once -- or
#: smaller, which starves the tasks that asked for more.
#:
#: Starving one is not a short trajectory. The container is OOM-killed mid-command and
#: the runtime's own error comes back as ordinary command output, so the model reads a
#: harness failure as its own command failing and keeps going.
#:
#: ``storage_mb`` is carried but not enforced: a session writes into a plain directory
#: under the overlay root, so it is the free space that root needs per trajectory
#: rather than a quota anything applies.
CPUS_FIELD = "cpus"
MEMORY_MB_FIELD = "memory_mb"
STORAGE_MB_FIELD = "storage_mb"
#: GPUs the task asks for. Recorded rather than honoured: this node's belong to the
#: server under measurement, so a task declaring one runs without it. On the row so a
#: trace says which sessions were in that position.
GPUS_FIELD = "gpus"

#: Sessions this row supplies, and the split it belongs in. Both are how the corpus
#: gets a deep session graded at all: only a pre-rolled row starts deep, a hashed
#: partition does not promise to put one of those in test, and one row is otherwise
#: one session.
DRAWS_FIELD = "draws"
SPLIT_FIELD = "split"

#: Sessions a deep row supplies. Two, because the deep rows are a thin sample of the
#: traffic the context-linear budget is built around and a second draw is the cheapest
#: way to widen it. Both draws resume the same frozen conversation, so they are repeat
#: samples of one trajectory rather than two trajectories; what differs is what the
#: server samples from that point on.
#:
#: A second draw costs a second container of the same image, so overlay disk scales
#: with sessions in flight rather than with rows.
DEEP_DRAWS = 2

#: Turns a deep row's session serves. Ten, because a resumed turn at 600k of prompt
#: costs far more than one at 5k: at the framework's own 250-turn ceiling a single deep
#: session would take the whole drive window.
#:
#: Applied to every deep row rather than declared per task, so filling ``deep_rows``
#: cannot leave a session on the ceiling by omission. ``turn_budgets`` still overrides
#: it for a row whose pre-roll cut a shallower band than planned.
DEEP_TURN_BUDGET = 10

#: What a row records for a task whose definition declares nothing, and what every
#: container is launched with: the seed carries neither field of a row to the launch
#: (``_seed_item``, ``task.sandbox_cpus``).
#:
#: The whole node has 64 CPUs and 2 TiB, but the eight GPUs and most of that memory
#: belong to the server under measurement, not to the shells driving it.
SANDBOX_CPUS = 4
SANDBOX_MEMORY_MB = 16384

#: The most a row may record for one trajectory's container, whatever its task
#: declares. A declaration above a cap is capped rather than refused: the task is in
#: the corpus on its instruction, not on its appetite.
#:
#: The node has 64 CPUs and about 2 TB, but thirty trajectories are alive at once and
#: the eight GPUs and most of that memory belong to the server under measurement. So
#: the ceiling is what a share of the node supports, not what the node holds: 16 CPUs
#: is the largest any admitted task asks for, 32 GiB the same, and 30 GiB of disk is
#: what the overlay root can hold thirty of.
#:
#: The defaults for the three ``TerminalBenchDataset.max_sandbox_*`` fields, which are
#: declared so the caps reach the calibration's workload fingerprint. They bound what
#: a row records, not what a container is launched with (``SANDBOX_CPUS``).
MAX_SANDBOX_CPUS = 16
MAX_SANDBOX_MEMORY_MB = 32768
#: Equal to ``SANDBOX_STORAGE_MB`` today, and a separate name because it does a
#: different job: that one is what a task with no declaration gets, this one is the
#: ceiling on what a task may declare. Moving one should not silently move the other.
MAX_SANDBOX_STORAGE_MB = 30720
#: Disk one trajectory's container may write, in MiB. Not a quota the runner enforces:
#: a session writes into a plain directory under the overlay root, so this is the free
#: space that root must have per concurrent trajectory.
SANDBOX_STORAGE_MB = 30720

#: Value ``source`` takes. One corpus, but the field stays: a trace reads back without
#: guessing where a row came from, and a second source would otherwise mean changing
#: the row schema rather than adding a value.
TERMINAL_BENCH_SOURCE = "terminal-bench"

#: Working directory assumed for a task whose Dockerfile names none. Upstream's own
#: default, and what the Singularity backend falls back to.
DEFAULT_WORKDIR = "/app"


class TerminalBenchError(RuntimeError):
    """The checkout is missing, incomplete, or not the pinned release."""


def release_images() -> tuple[str, dict[str, str]]:
    """The registry repository the release publishes its environment images in, and each
    task's image by the digest of its published manifest (``RELEASE_IMAGES_FILE``). Raise
    TerminalBenchError when the file is not of ``RELEASE_COMMIT``: the digests would then be
    of another release's images."""
    pinned = json.loads(RELEASE_IMAGES_FILE.read_text())
    if pinned["commit"] != RELEASE_COMMIT:
        raise TerminalBenchError(
            f"{RELEASE_IMAGES_FILE.name} pins the images of {pinned['commit'][:12]}, not of "
            f"the release at {RELEASE_COMMIT[:12]}")
    return str(pinned["repository"]), dict(pinned["images"])


def canonical_names(names) -> tuple[str, ...]:
    """A set of task names in the one order that stands for that set.

    The datasets here declare their exclusions as a field so the calibration's workload
    fingerprint sees them, and that fingerprint is a hash of canonical JSON. Canonical
    sorts keys but not sequences, so an unordered set written two ways would hash two
    ways and one spelling would refuse a calibration taken under the other."""
    return tuple(sorted(set(names)))


@functools.lru_cache(maxsize=1)
def tasks_dir() -> Path:
    """The ``tasks/`` directory of the release at ``RELEASE_COMMIT``, fetched from
    ``RELEASE_URL`` into a directory of this process's own the first time it is asked for, and
    removed when the process exits (``fetch_release``)."""
    root = Path(tempfile.mkdtemp(prefix="terminal-bench-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    fetch_release(root)
    return root / "tasks"


def fetch_release(root: Path, *, url: str = RELEASE_URL, commit: str = RELEASE_COMMIT) -> None:
    """Fetch commit *commit* of the repository at *url* alone into the empty directory *root*
    and check it out. Raise TerminalBenchError, with the end of git's own message, when any
    step fails."""
    for argv in (["init", "-q", str(root)],
                 ["-C", str(root), "fetch", "-q", "--depth", "1", url, commit],
                 ["-C", str(root), "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]):
        done = subprocess.run(["git", *argv], capture_output=True, text=True)
        if done.returncode != 0:
            tail = " ".join((done.stderr or done.stdout).split())[-300:]
            raise TerminalBenchError(f"fetching the release at {commit} from {url} failed: git {argv[-1]}: {tail}")


def dockerfile_workdir(dockerfile: Path) -> str:
    """The directory a task's image starts in.

    The last ``WORKDIR`` wins, which is what Docker itself does: several of these
    Dockerfiles set one early to build in and another at the end to run in, and only
    the last one is where the agent's shell lands.
    """
    workdir = DEFAULT_WORKDIR
    for line in dockerfile.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.upper().startswith("WORKDIR "):
            workdir = line.split(None, 1)[1].strip()
    return workdir


def warm_start_id(prefixes: dict, turn_budgets: dict, sessions: dict = {}) -> str:
    """The warm starts as a corpus identity fragment.

    Sorted, because a ``corpus_id`` that reordered would re-provision a corpus that
    had not changed."""
    cut = ",".join(f"{n}={d}" for n, d in sorted(prefixes.items()))
    budgets = ",".join(f"{n}={t}" for n, t in sorted(turn_budgets.items()))
    drawn = ",".join(f"{n}={k}" for n, k in sorted(sessions.items()))
    return f"prefixes={cut}|turn_budgets={budgets}|sessions={drawn}"


def session_fields(draws: int = 1, split: str = "") -> dict:
    """The two row fields that say how many sessions a row supplies and where it sits.

    Defaults are one session and a hashed split, which is every row outside
    ``deep_rows``."""
    return {DRAWS_FIELD: int(draws), SPLIT_FIELD: str(split)}


def warm_start_fields(name: str, prefixes: dict, turn_budgets: dict) -> dict:
    """The two row fields that say how a task's trajectory starts and how far it runs.

    Absent from either map means a cold start at turn 0 and the framework's own turn
    budget."""
    return {PREFIX_DIGEST_FIELD: str(prefixes.get(name, "")),
            TURN_BUDGET_FIELD: int(turn_budgets.get(name, 0))}


def task_digest(task_dir: Path) -> str:
    """One task's content digest, over ``DIGESTED_FILES``.

    A missing file hashes as absent rather than raising, so a truncated checkout
    reports a digest mismatch naming the task instead of a file-not-found from
    somewhere deep in provisioning."""
    import hashlib

    h = hashlib.sha256()
    for name in DIGESTED_FILES:
        path = task_dir / name
        h.update(name.encode())
        h.update(b"\0")
        h.update(path.read_bytes() if path.is_file() else b"<absent>")
        h.update(b"\0")
    return h.hexdigest()


def release_digest(root: Path) -> str:
    """The digest of every task definition in a checkout, in name order."""
    import hashlib

    h = hashlib.sha256()
    for d in sorted(x for x in root.iterdir() if (x / "task.toml").exists()):
        h.update(d.name.encode())
        h.update(b"\0")
        h.update(task_digest(d).encode())
        h.update(b"\0")
    return h.hexdigest()


def sandbox_fields(environment: dict, caps: tuple[int, int]) -> dict:
    """What this task's container is given, read off its own ``[environment]`` block.

    A definition with no block, or one missing a figure, takes the module default: the
    corpus should not hand the backend a container with no CPUs because a release was
    re-cut without a field.

    Capped at what the node has. A task is in the corpus on its instruction, not on its
    appetite, so one asking for more than the node holds runs on what the node can
    spare rather than being dropped."""
    max_cpus, max_memory, max_storage = caps
    return {
        CPUS_FIELD: min(int(environment.get(CPUS_FIELD) or SANDBOX_CPUS), max_cpus),
        MEMORY_MB_FIELD: min(
            int(environment.get(MEMORY_MB_FIELD) or SANDBOX_MEMORY_MB), max_memory),
        STORAGE_MB_FIELD: min(
            int(environment.get(STORAGE_MB_FIELD) or SANDBOX_STORAGE_MB), max_storage),
        GPUS_FIELD: int(environment.get(GPUS_FIELD) or 0),
    }


def _task_row(task_dir: Path, prefixes: dict, turn_budgets: dict,
              sessions: dict | None = None,
              caps: tuple[int, int, int] = (
                  MAX_SANDBOX_CPUS, MAX_SANDBOX_MEMORY_MB,
                  MAX_SANDBOX_STORAGE_MB)) -> dict:
    """One task's corpus row, read off its definition.

    ``sessions`` names the rows that supply more than one session. Those are the rows
    a pre-roll cuts a prefix for, and they are forced into test for the same reason:
    a deep row that hashed into dev would produce no scored deep traffic.

    ``caps`` bounds what the row may ask its sandbox for; see ``sandbox_fields``."""
    import tomllib

    instruction = task_dir / "instruction.md"
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not instruction.exists():
        raise TerminalBenchError(f"{task_dir.name} has no instruction.md")
    if not dockerfile.exists():
        raise TerminalBenchError(f"{task_dir.name} has no environment/Dockerfile")
    config = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = config.get("metadata") or {}
    return {
        **sandbox_fields(config.get("environment") or {}, caps),
        TASK_NAME_FIELD: task_dir.name,
        INSTRUCTION_FIELD: instruction.read_text(),
        WORKDIR_FIELD: dockerfile_workdir(dockerfile),
        CATEGORY_FIELD: str(metadata.get("category", "")),
        EXPERT_HOURS_FIELD: float(metadata.get("expert_time_estimate_hours") or 0.0),
        SOURCE_FIELD: TERMINAL_BENCH_SOURCE,
        # Every task in the release leaves this unset, and harbor's default is True.
        ALLOW_INTERNET_FIELD: True,
        **warm_start_fields(task_dir.name, prefixes, turn_budgets),
        **session_fields(draws=(sessions or {}).get(task_dir.name, 1),
                         split="test" if task_dir.name in (sessions or {}) else ""),
    }


def row_draws(row: dict) -> int:
    """Sessions this row supplies, at least one. A row without the field, or one that
    names a nonsense count, is one session."""
    try:
        return max(1, int(row.get(DRAWS_FIELD) or 1))
    except (TypeError, ValueError):
        return 1


def _seed_items(row: dict, idx: int, split_value: str,
                dataset_name: str) -> Iterator[DatasetItem]:
    """This row's trajectory seeds: one per draw.

    A row is one session unless it says otherwise. Where it says otherwise, each draw
    is a session of its own -- its own id, its own sandbox, its own trace -- resuming
    the same frozen conversation, so the draws are repeat samples of one trajectory."""
    for draw in range(row_draws(row)):
        yield _seed_item(row, idx, split_value, dataset_name, draw)


def _seed_item(row: dict, idx: int, split_value: str, dataset_name: str,
               draw: int = 0) -> DatasetItem:
    """One trajectory's seed, from a corpus row.

    Shared so that adding a corpus cannot change the shape of a seed: the framework
    reads the instruction, the task name and the working directory off the metadata,
    and would behave differently for two sources that rendered them differently.

    ``draw`` suffixes the request id, which becomes the session id and so the id of
    every turn under it. ``#`` and not ``.``: a turn id is ``<session>.<k>``, and the
    partition refuses a row id ending that way so the two can be told apart."""
    instruction = str(row.get(INSTRUCTION_FIELD, ""))
    task_name = str(row[TASK_NAME_FIELD])
    session_id = task_name if draw == 0 else f"{task_name}#{draw}"
    env = Envelope(
        request_id=session_id,
        contents=[TextPayload(text=instruction)],
        # No pre-rendered ``messages``: the framework renders the conversation,
        # so the prompts stay one definition rather than two that can drift.
        metadata={TASK_NAME_FIELD: task_name,
                  INSTRUCTION_FIELD: instruction,
                  WORKDIR_FIELD: str(row.get(WORKDIR_FIELD) or DEFAULT_WORKDIR),
                  CATEGORY_FIELD: str(row.get(CATEGORY_FIELD, "")),
                  EXPERT_HOURS_FIELD: row.get(EXPERT_HOURS_FIELD),
                  SOURCE_FIELD: str(row.get(SOURCE_FIELD, "")),
                  # Defaults True for a row without the field, which is what every
                  # Terminal-Bench task declares anyway.
                  ALLOW_INTERNET_FIELD: bool(row.get(ALLOW_INTERNET_FIELD, True)),
                  # A row without the warm-start fields starts cold, on the
                  # framework's own turn budget.
                  PREFIX_DIGEST_FIELD: str(row.get(PREFIX_DIGEST_FIELD, "")),
                  TURN_BUDGET_FIELD: int(row.get(TURN_BUDGET_FIELD) or 0)})
    # No ground truth. Upstream's answer is a shell script that has to run inside the
    # task's own container, which is not a thing a trace can carry or a verifier here
    # replays; correctness is the report-KL gate's job and whether the task was solved
    # is not scored.
    return DatasetItem(
        # The framework reads this as the session id, so it is the draw's and not the
        # row's: two draws sharing one would interleave in the trace and, worse, in
        # the sandbox, which is keyed on the session.
        request_id=session_id,
        turns=(env,),
        ground_truth_turns=(),
        metadata={"split": split_value, "index": idx, "dataset": dataset_name,
                  DRAWS_FIELD: draw})


def _mounted_rows(mount_dir, split: Split) -> list:
    """The rows provisioning attached for this split.

    Read rather than recomputed so the driver touches nothing outside its mount at run
    time, which is also why a missing file is reported as a provisioning gap."""
    path = Path(mount_dir) / split.value / "rows.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{split.value} split not mounted at {path}; provisioning did not "
            f"attach it for this role (TEST is withheld from the agent)")
    return json.loads(path.read_text())


class TerminalBenchDataset(Dataset):
    """Provisionable seeds for the mini-SWE-agent loop.

    Config rather than code, like every other dataset here: ``corpus()`` reads the
    checkout and ``SplitSpec`` carves the seeded dev/test partition, while ``items_at``
    reads the mounted ``rows.json`` so the driver reads nothing outside its mount at
    run time.
    """

    name: ClassVar[str] = "terminal_bench"

    #: The upstream release tag, for a human reading a trace. The checkout's git state
    #: is not inspected, so what actually catches a checkout at the wrong place is the
    #: digest below rather than either of these two labels.
    release: str = RELEASE_TAG

    #: The commit that tag pointed at when this corpus was pinned. A declared field
    #: rather than only the module constant, because which commit was graded is part
    #: of what a calibration was taken over, and only declared fields reach the
    #: workload fingerprint.
    commit: str = RELEASE_COMMIT

    #: The tasks with no image, as a declared field rather than a constant read from
    #: the module. Which tasks the corpus contains is part of what a calibration was
    #: taken over, and only declared fields reach the workload fingerprint
    #: (``hls.verify.calibration._traffic_spec`` serializes the dataset, and a module
    #: constant is invisible to that). ``NO_IMAGE`` is the default and the place the
    #: reasoning is written down.
    no_image: tuple[str, ...] = canonical_names(NO_IMAGE)

    #: Tasks whose sessions resume a pre-rolled conversation instead of starting at
    #: turn 0. Each is pinned to test and supplies ``DEEP_DRAWS`` sessions.
    #:
    #: Declared as data rather than built into a second dataset class, so naming a
    #: different three is an edit to this line.
    #:
    #: Why any row needs this at all is a fact about the drive window rather than about
    #: this corpus. A drive issues requests for 45 minutes, which at this stack's
    #: rate is 55 to 140 turns, so the deep range is out of reach inside a
    #: window whatever the tasks are. Terminal-Bench trajectories reach that range
    #: given enough turns, so what the pre-roll supplies is the head start, not depth
    #: the corpus lacks.
    #:
    #: Three rows, one per depth band, each a task whose trajectory deepens without
    #: command-less turns: rs-archive-clone holds the 600k band, ontology-kg-querying
    #: the 400k band and takens-embedding-lean the 200k band. An empty tuple would also
    #: be a coherent corpus: every row hashes, every session starts cold, and the split
    #: arithmetic in ``task.py`` reaches its session target either way.
    deep_rows: tuple[str, ...] = (
        "ontology-kg-querying", "rs-archive-clone", "takens-embedding-lean")

    #: Task name to the digest of the frozen conversation its session resumes from.
    #: Empty until a pre-roll has run; a row naming one is refused unless the staged
    #: file hashes to it, so this pins the conversation rather than the file.
    #:
    #: Separate from ``deep_rows`` because they answer different questions. Which rows
    #: are deep is a property of the corpus; whether their prefix has been cut yet is a
    #: property of the staging. A row in ``deep_rows`` with no digest here draws its
    #: sessions and starts them cold.
    #:
    #: Each is cut against GLM-5.3-Flash through the task's own loop at the
    #: contract's sampling (temperature 1, no truncation, no penalties), with the
    #: thinking block capped at 32,768 tokens and any reply that hit the cap drawn
    #: again, so every stored reply is closed and under the cap. Depth is the prompt
    #: length at the cut: 400,000 for ontology-kg-querying, 600,000 for
    #: rs-archive-clone and 200,000 for takens-embedding-lean.
    prefixes: dict[str, str] = {
        "ontology-kg-querying":
            "23aee821ca1c6ab4f86482c936035eee38381b03ceae1b2178078a23210ecebb",
        "rs-archive-clone":
            "350e2eb31507440fafac7ae534d52c8dfd6b0024c135783f917c3071da71ca98",
        "takens-embedding-lean":
            "35dc148e56f7fce8e405c773d48b10df8e439eb7e201890ec5f5704da5e77832",
    }

    #: What a row may ask its sandbox for, whatever its definition declares. Declared
    #: fields rather than module constants because only declared fields reach the
    #: calibration's workload fingerprint, and lowering either one starves the tasks
    #: that asked for more -- which changes the observations their commands return,
    #: and fails nothing while it does.
    #:
    #: A ceiling, not an allocation: almost every task asks for far less, and the
    #: corpus's own declarations total well under half what a flat budget would hand
    #: out.
    #:
    #: These are node facts sitting on a dataset, which is a layering smell and is
    #: recorded as one rather than argued away. The dataset owns them because the row
    #: is where a declaration is turned into what a session gets, and a cap applied
    #: anywhere else would leave the row saying one thing and the sandbox doing
    #: another.
    max_sandbox_cpus: int = MAX_SANDBOX_CPUS
    max_sandbox_memory_mb: int = MAX_SANDBOX_MEMORY_MB
    max_sandbox_storage_mb: int = MAX_SANDBOX_STORAGE_MB

    #: The release digest this corpus is pinned to; see ``RELEASE_DIGEST``. Declared
    #: rather than read from the constant so that a run against a different release
    #: cannot reuse a calibration taken on this one.
    #:
    #: A factory rather than a plain default so the value is read when an instance is
    #: built rather than when this class is defined. That is what lets a test point the
    #: pin at a checkout it invented, and it costs nothing here.
    corpus_digest: str = Field(default_factory=lambda: RELEASE_DIGEST)

    #: Tasks excluded because they declare an accelerator the sandbox cannot give.
    #: Declared for the same reason ``no_image`` is: which tasks the corpus contains is
    #: part of what a calibration was taken over. ``NEEDS_ACCELERATOR`` is the default
    #: and the place the reasoning is written down.
    needs_accelerator: tuple[str, ...] = canonical_names(NEEDS_ACCELERATOR)

    #: Task name to the turns its session serves, overriding the derived budget.
    #: Empty: a deep row takes ``DEEP_TURN_BUDGET`` and every other row takes the
    #: framework's own ceiling. Declared because how many turns a session serves is
    #: how much traffic it is, so a calibration cannot span two settings of it.
    turn_budgets: dict[str, int] = {}

    _canonical_no_image = field_validator("no_image")(canonical_names)
    _canonical_needs_accelerator = field_validator(
        "needs_accelerator")(canonical_names)
    _canonical_deep_rows = field_validator("deep_rows")(canonical_names)

    @model_validator(mode="after")
    def _pins_name_rows_the_corpus_will_carry(self):
        """Refuse a pin naming a row that will not be in the corpus, or not resume.

        Both failures are silent otherwise, and both move the fingerprint while
        changing nothing about the traffic.

        A prefix for a row outside ``deep_rows`` is worse than inert: every corpus row
        is offered the prefix map by name, so that row would resume a pre-rolled
        conversation while hashing like an ordinary row and drawing one session.

        A deep row the corpus drops for having no image is inert, and it breaks the
        split arithmetic besides: ``task.py`` sizes the hashed test count by
        subtracting ``pinned_test_rows()``, so a pinned row that never reaches the
        corpus makes that subtraction too large and the split comes up short.
        """
        dropped = sorted(set(self.deep_rows) & set(self.no_image))
        if dropped:
            raise ValueError(
                f"deep rows {dropped} have no image, so the corpus drops them; "
                f"remove them from deep_rows or from no_image")
        stray = sorted(set(self.prefixes) - set(self.deep_rows))
        if stray:
            raise ValueError(
                f"prefixes name {stray}, which does not resume: only a task in "
                f"deep_rows starts from a pre-rolled conversation")
        return self

    def usable_tasks(self) -> int:
        """Tasks this corpus can supply: the release less both exclusion sets."""
        return (RELEASE_TASK_COUNT - len(self.no_image)
                - len(self.needs_accelerator))

    def excluded(self) -> frozenset[str]:
        """Every task the corpus drops, whichever reason dropped it."""
        return frozenset(self.no_image) | frozenset(self.needs_accelerator)

    def pinned_test_rows(self) -> int:
        """Rows this corpus puts in test by declaration rather than through the hash.

        A split size has to know this: the partition holds a pinned row out of the
        ranking and adds it to test on top of the count it was asked for, so a test
        count sized as "every row not in dev" would ask the hash for more rows than
        the pool it draws from has, and the partition refuses the whole source.

        The deep rows are pinned because a pre-rolled conversation that landed in dev
        by hash would produce no scored deep traffic, which is the one thing cutting
        it was for."""
        return len(self.deep_rows)

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        """One trajectory seed per mounted row, in file order."""
        for idx, row in enumerate(_mounted_rows(mount_dir, split)):
            yield from _seed_items(row, idx, split.value, self.name)

    def row_id(self, row: dict) -> str:
        return str(row[TASK_NAME_FIELD])

    def forced_split(self, row: dict) -> str | None:
        """The split the row declares, or None for the hash."""
        return str(row.get(SPLIT_FIELD) or "") or None

    def draws(self, row: dict) -> int:
        return row_draws(row)

    def hub_revisions(self) -> list[tuple[str, str]]:
        """None: the corpus is read from a local checkout of the release, which ``corpus()``
        checks against ``corpus_digest`` before it reads a row, never from the hub."""
        return []

    def corpus(self):
        """Every task in the checkout the sandbox can run, as an Arrow dataset.

        The checkout is counted before anything is dropped, so a clone that is
        truncated or at the wrong tag fails here rather than provisioning a short
        corpus that would still partition and still run.
        """
        from datasets import Dataset as ArrowDataset

        root = tasks_dir()
        if not root.is_dir():
            raise TerminalBenchError(f"the release fetched from {RELEASE_URL} at {RELEASE_COMMIT} has no tasks/")
        found = sorted(d for d in root.iterdir() if (d / "task.toml").exists())
        if len(found) != RELEASE_TASK_COUNT:
            raise TerminalBenchError(
                f"{root} holds {len(found)} tasks; {self.release} ships "
                f"{RELEASE_TASK_COUNT}")
        self._check_release_digest(root, found)
        rows = [_task_row(d, self.prefixes, self.session_turn_budgets(),
                          self.session_draws(), self.sandbox_caps())
                for d in found]
        self._check_accelerator_list(rows)
        excluded = self.excluded()
        return ArrowDataset.from_list(
            [r for r in rows if r[TASK_NAME_FIELD] not in excluded])

    def _check_release_digest(self, root: Path, found: list[Path]) -> None:
        """Refuse a checkout whose task definitions are not the pinned release.

        The task count above passes for any 66 directories, so it catches a truncated
        clone and nothing else. This is what catches a checkout at another commit, or
        one a local edit touched.

        An empty pin refuses rather than skipping. A corpus provisioned with the check
        quietly off would look exactly like one that passed it.
        """
        if not self.corpus_digest:
            raise TerminalBenchError(
                "no corpus_digest pinned, so the checkout cannot be checked against "
                "the release; compute it with release_digest(tasks_dir())")
        got = release_digest(root)
        if got == self.corpus_digest:
            return
        # Recomputed per task only on the way to failing, because the name is what
        # makes the message worth reading and computing it always is wasted work.
        differing = sorted(
            d.name for d in found
            if task_digest(d) != self._pinned_task_digests().get(d.name))
        where = f"; tasks that differ: {differing}" if differing else ""
        raise TerminalBenchError(
            f"{root} does not match the pinned release: digest {got[:12]} but "
            f"corpus_digest is {self.corpus_digest[:12]}{where}. This checkout is not "
            f"{self.release} at {self.commit[:12]}")

    @staticmethod
    def _pinned_task_digests() -> dict[str, str]:
        """Per-task digests for the pinned release, or empty if the file is gone.

        Empty is honest rather than useless: the refusal above then names no task and
        says only that the release moved, which is still the right verdict. A missing
        or unreadable file must not turn a real mismatch into a crash about JSON."""
        try:
            return dict(json.loads(RELEASE_DIGESTS_FILE.read_text())["tasks"])
        except (OSError, ValueError, KeyError):
            return {}

    def _check_accelerator_list(self, rows: list[dict]) -> None:
        """Refuse a ``needs_accelerator`` list the checkout disagrees with.

        The list is a pin: it is a declared field so that what the corpus excluded
        reaches ``corpus_id`` and the calibration's workload fingerprint, which a set
        computed at read time could not do. A pin can go stale, and this one would go
        stale silently -- a release that gave an existing task a GPU would keep grading
        a trajectory that cannot do the work it was set, and one that dropped the
        requirement would keep a task out for a reason that had stopped being true.

        So the names are written down and checked rather than trusted. Checked here
        rather than in field validation because the declarations live in the checkout,
        and a dataset is constructed long before anything reads one. Both directions
        are named in the message, so a release that moved is a five-second diagnosis.
        """
        declares = {r[TASK_NAME_FIELD] for r in rows if r[GPUS_FIELD]}
        listed = set(self.needs_accelerator)
        if unlisted := sorted(declares - listed):
            raise TerminalBenchError(
                f"{unlisted} declare a GPU and are not in needs_accelerator; the "
                f"sandbox has no accelerator to give them, so add them or correct "
                f"the release pin")
        if stale := sorted(listed - declares):
            raise TerminalBenchError(
                f"{stale} no longer declare a GPU but are still excluded by "
                f"needs_accelerator; drop them from it so the corpus carries them")

    def session_draws(self) -> dict[str, int]:
        """Task name to the sessions its row supplies, for the rows that supply more
        than one. Derived from ``deep_rows`` rather than declared beside it: a second
        map would be a second place to forget."""
        return {name: DEEP_DRAWS for name in self.deep_rows}

    def sandbox_caps(self) -> tuple[int, int, int]:
        """The ceilings a row's declared CPUs, memory and disk are capped at."""
        return (self.max_sandbox_cpus, self.max_sandbox_memory_mb,
                self.max_sandbox_storage_mb)

    def session_turn_budgets(self) -> dict[str, int]:
        """Task name to the turns its session serves, for the rows that are bounded.

        Derived from ``deep_rows`` for the same reason the draw count is, and a
        declared ``turn_budgets`` entry wins over the derived one."""
        return {**{name: DEEP_TURN_BUDGET for name in self.deep_rows},
                **self.turn_budgets}

    def corpus_id(self) -> str:
        # The excluded set is part of the identity: re-including a task once its image
        # builds changes the corpus, and that has to re-provision rather than be read
        # from a cache keyed on the tag alone. So are the warm starts -- which rows
        # resume, from which conversation, how far each runs and how many sessions
        # each supplies -- since every one of them changes the traffic.
        return (f"terminal-bench/terminal-bench@{self.commit}"
                f"|tag={self.release}"
                f"|digest={self.corpus_digest}"
                f"|no_image={','.join(self.no_image)}"
                f"|needs_accelerator={','.join(self.needs_accelerator)}"
                f"|caps={self.max_sandbox_cpus},{self.max_sandbox_memory_mb},"
                f"{self.max_sandbox_storage_mb}"
                f"|{warm_start_id(self.prefixes, self.session_turn_budgets(), self.session_draws())}")


__all__ = ["ALLOW_INTERNET_FIELD", "CATEGORY_FIELD", "CPUS_FIELD", "DEEP_DRAWS",
           "DEEP_TURN_BUDGET", "DEFAULT_WORKDIR", "DRAWS_FIELD",
           "EXPERT_HOURS_FIELD", "GPUS_FIELD", "INSTRUCTION_FIELD",
           "MAX_SANDBOX_CPUS", "MAX_SANDBOX_MEMORY_MB",
           "MAX_SANDBOX_STORAGE_MB", "MEMORY_MB_FIELD",
           "NEEDS_ACCELERATOR", "NO_IMAGE",
           "PREFIX_DIGEST_FIELD", "RELEASE_COMMIT", "RELEASE_DIGEST", "RELEASE_IMAGES_FILE",
           "RELEASE_TAG", "RELEASE_TASK_COUNT", "release_digest", "release_images", "task_digest",
           "SANDBOX_CPUS", "SANDBOX_MEMORY_MB", "SANDBOX_STORAGE_MB",
           "RELEASE_URL", "SOURCE_FIELD", "SPLIT_FIELD", "STORAGE_MB_FIELD",
           "TASK_NAME_FIELD", "sandbox_fields",
           "TERMINAL_BENCH_SOURCE", "TURN_BUDGET_FIELD", "TerminalBenchDataset",
           "TerminalBenchError", "WORKDIR_FIELD", "canonical_names",
           "dockerfile_workdir", "fetch_release", "row_draws", "session_fields", "tasks_dir",
           "warm_start_fields", "warm_start_id"]
