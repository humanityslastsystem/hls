"""agentic_coding -- long-horizon coding-agent serving, graded by the report-KL sampling gate.

The harness owns a mini-SWE-agent loop (``framework``) and the shell it runs commands
in (``hls.spec.sandbox``); the artifact serves one reply per HTTP call and never
executes anything (Sec.9). One trajectory is one upstream coding task: the model reads
a task statement, then works through a real shell for as many turns as it takes, which
is the traffic this task exists to measure -- prompts that start long and grow, a reply
every turn, and a makespan that spans minutes.

One corpus. Terminal-Bench 4.0: all 66 tasks are given an 8-hour agent timeout
against a median expert time estimate of 4 hours, and its leaderboard publishes
per-trial token totals measured under this same agent loop.

With this loop the corpus reaches 674,326 prompt tokens at the graded tensor-parallel-8
shape, 64.3% of the declared context window, on ``rs-archive-clone`` at turn 140.

Depth inside a drive is a fact about the drive window rather than about the tasks. A
drive issues requests for 45 minutes (``DRIVE_WALL_CLOCK_SEC``), which at
this stack's observed rate is 55 to 140 turns, so the deepest bands are out of reach
inside a window whatever the corpus is. The pre-roll is what supplies the head start,
and ``TerminalBenchDataset.deep_rows`` names the rows that take it.

One caution for whoever chooses those rows. Terminal-Bench depth accrues fast, but 85 to
90% of it is the model's own reasoning fed back, which a turn producing no command
discards entirely. Depth resting on command output is the sturdier kind, so a row whose
turns run commands is a better pre-roll subject than one that thinks more.

What is graded
--------------

Correctness is the report-KL sampling verifier and nothing else. The server samples at
temperature 1 over the full vocabulary and reports what it sampled from; the reference
teacher-forces the same prefixes and the gate tests the divergence between the two,
the served tokens' rank uniformity, their log score, and the end-of-turn count.

Whether the task was solved is not graded and is not measured at all. Terminal-Bench
scores a task by running its verifier in a second container built from the task's own
``tests/`` image, which this harness does not build or run: what the gate reads is the
served tokens, and a trajectory that solves nothing serves exactly the same kind of
tokens as one that succeeds. ``metric`` reports trajectory shape instead -- turns
taken, commands run, and how each trajectory ended.

The score is speedup over the baseline serving stack in SLO-normalized per-turn
latency (``SloLatencyScorer``), so what a contestant competes on is how fast it serves
this traffic, with the gate deciding whether it served the right distribution while
doing so. Per turn rather than per trajectory because tool execution is 59.8% of a
coding agent's request time and none of it is the server's doing; scoring the
trajectory's wall clock would spend most of the measurement on the sandbox.

Hardware
--------

One node, eight H200 at 143,771 MiB each, NVLink. No collective leaves the box.

The checkpoint is 320B-A18B, 306 GiB in FP8. Its attention is hybrid: 11 of 45 layers
are NoPE MLA on a 512-wide latent, each picking 2,048 keys per query through a
DeepSeek-style sparse indexer; the other 34 are KDA linear attention. Note that vLLM
replicates the MLA latent on every tensor-parallel rank rather than sharding it.

The load
--------

Measured, by a concurrency ladder against the reference on the graded shape.
``MEASURED_CONCURRENCY`` carries the ladder and the criterion.

One measured number, and the rest follow from it. The admission cap is that number.
``TARGET_TEST_SESSIONS`` is sized above it so the slots stay busy, the test row count
is derived from the session count, dev is the remainder, and the arrival rate comes
from the session count and the window it has to start them in. Nothing here is a
second opinion about the same quantity.

Calibrate (one serving of the traffic per trusted stack). The pool is SGLang alone,
because vLLM is the pinned reference here and a stack cannot calibrate against itself:

    uv run hls calibrate tasks.agentic_coding.task --engines sglang

Run: uv run hls run tasks.agentic_coding.task
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

from hls.artifact.artifact import (
    AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation, EvaluationAllocation,
    HighestAvailable,
)
from hls.core.decode import Sample
from hls.core.model import AcceleratorSpec, Hardware, InterconnectSpec, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.prompt import ImplementerBrief
from hls.provision.sandbox_images import SandboxImage, SandboxImageSet
from hls.reference.references import VLLMReference
from hls.reference import service_harness as _SERVICE
from hls.recipes import pins
from hls.recipes.catalog import FLASHINFER_CUBINS, SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage, RegistryImage
from hls.scoring import SloLatencyScorer
from hls.spec.agent import AgentSpec
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.provision.sandbox_runtime import RUNTIME_TAG
from hls.spec.sandbox import SandboxToolRunner, default_env_factory
from hls.spec.taskspec import TaskSpec
from hls.spec.tools import DEVICE, HostPath, StoreRefused
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (
    ContextLinearLatencySLO, NoDelay, Poisson, ShortOutputCap, Workload,
)
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import AccessList, NetworkPolicy
from hls.verify import ReportFaithfulnessVerifier
from hls.verify.report import TOP_LOGPROBS_REQUESTED

from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding.datasets import (
    ALLOW_INTERNET_FIELD, CPUS_FIELD, DEEP_DRAWS, DEEP_TURN_BUDGET, MEMORY_MB_FIELD,
    PREFIX_DIGEST_FIELD, SANDBOX_CPUS, SANDBOX_MEMORY_MB,
    TASK_NAME_FIELD, WORKDIR_FIELD, TerminalBenchDataset, release_images,
)
from tasks.agentic_coding.framework import MiniSweFramework
from tasks.agentic_coding.metric import (
    ReportOnlyMetricVerifier, ReportOnlyTrajectoryShapeVerifier,
    TrajectoryServingMetrics,
)
from tasks.agentic_coding.mini_swe_loop import WORK_DIR
from tasks.agentic_coding.preroll import DEPTH_TARGETS

# -----------------------------------------------------------------------------
# MODEL. The published checkpoint is already FP8 (the BF16 variant carries the
# suffix), so the reference and the artifact pin the same weights and the gate is not
# measuring the quantization the task itself mandates.
MODEL_ID = "zai-org/GLM-5.3-Flash"
#: The commit the repo's main branch named while the task's calibration and baseline credit were
#: measured: its "Add evaluation results" commit, which main named until a later commit changed
#: the chat template. A provisioned snapshot is keyed by a commit, never a branch.
MODEL_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
MODEL = ModelSpec(base_checkpoint=MODEL_ID, revision=MODEL_REVISION, arch="MoE",
                  chat_format="glm")

#: The decode every server implements and the reference grades against: one draw per
#: position from the full-vocabulary softmax at temperature one, no truncation and no
#: penalty. Read by the oracle, the calibration pool, the baselines and the brief.
DECODE = Sample(temperature=1.0)

HARDWARE = Hardware(
    # One node of eight H200, NVLink-connected. nvidia-smi reports 143,771 MiB each,
    # so the node holds 1,123 GiB and no collective leaves the box.
    accelerator=AcceleratorSpec(kind="H200", count=8, memory_gb_per_device=141),
    interconnect=InterconnectSpec(kind="NVLink"),
    cpu_count=64, memory_gb=2011, storage_gb=4096,
)

# -----------------------------------------------------------------------------
# WORKLOAD.
#: Tokens a turn's first request may spend before the harness stops it and asks the
#: model to finish.
#:
#: Without a cap, 1 to 2.5% of this checkpoint's turns spend the whole budget inside the
#: thinking channel and emit no visible reply at all, each costing 1,100 to 1,500 seconds
#: of decode, and three such turns in a row end the trajectory.
#:
#: 32,768 comes off the distribution of turns that succeeded: p95 10,500, p99 22,500,
#: and 5 of 1,092 above 32k. So it stops the degenerate tail and leaves ordinary
#: thinking untouched.
#:
#: A capped reply is not thrown away. The loop appends the closing tag and sends a
#: second request asking the model to carry on from it, which is where REPLY_BUDGET
#: comes in. Both requests are served, scored and graded; the contract in the brief
#: says so.
THINK_CAP = 32_768

#: Tokens the continuation of a capped reply may spend.
#:
#: Smaller than the cap because it bounds a different thing: once the thought has been
#: closed for the model, what remains is the visible reply -- a tool call and a
#: sentence. Not much smaller, though: a continuation holds a whole command, a heredoc
#: among them, and 16,384 leaves room for a long one without leaving room for a second
#: runaway.
REPLY_BUDGET = 16_384
#: Context window the servers and the reference are built at. The checkpoint's full
#: declared window, and production coding agents reach it: prefix length runs to a
#: 99th percentile of 918k tokens (TraceLab, arXiv:2606.30560). Part of the workload
#: identity, so a calibration taken at one window is refused against a run at another.
MAX_CONTEXT_LEN = 1048576
#: Turns one trajectory may take. Upstream mini-SWE-agent's own ``step_limit``, kept
#: because Terminal-Bench sets none: it bounds an agent by wall clock instead, at 8
#: hours for every task, which is not a bound this harness can apply to a trajectory
#: whose speed is the thing being measured.
MAX_TURNS = 250
#: Concurrent requests the drive offers, and this task's operating point.
#:
#: Measured, by a concurrency ladder against the reference on the graded shape -- one
#: node, eight H200 at tensor parallel 8, exclusive -- taking the largest
#: level whose SLO-normalized latency still had headroom, which was a median ratio at
#: or below 0.7 and a p90 at or below 1.0:
#:
#:     C     req   median r   p90 r   med TTFT   med TPOT   prefix hit
#:      8    161      0.526   1.343     250 ms   10.37 ms        81.0%
#:     16    270      0.594   0.814     162 ms   11.84 ms        95.3%
#:     24    412      0.674   0.920     159 ms   13.48 ms        99.2%
#:     32    434      0.751   1.083     167 ms   15.01 ms        99.3%
#:     48    510      4.915  12.755   8,134 ms   15.21 ms        99.2%
#:
#: 24 is the pass, and the ladder is not monotone at either end for reasons that are
#: not capacity. C=8 fails on p90 with an 81% prefix hit against 99% higher up, so it
#: measures a cache filling rather than a server struggling. C=48 measures queueing
#: above the engine's 32-sequence limit, where requests wait outside the batch, which
#: is why its median jumps sevenfold while its decode rate barely moves.
#:
#: 16 also passed and is the wrong choice. The ladder exists to find the largest load
#: the reference carries inside its SLO, and running below it leaves the server
#: headroom it was measured not to need -- which makes the benchmark less
#: discriminating, since serving stacks differ under load and look alike under slack.
MEASURED_CONCURRENCY = 24

#: In-flight artifact submits. Tool waits are not admitted, so trajectories blocked on
#: a command overlap freely while the contestant's LLM batch stays bounded.
#:
#: The operating point itself, not a separate knob: a cap below it would keep the drive
#: from reaching the point the measurement picked.
#:
#: It also bounds the requests a scheduled stop can cut, about 2.4% of a drive at 24.
MAX_CONCURRENT_LLM = MEASURED_CONCURRENCY
#: True while the load is a stand-in rather than a measurement. The load is
#: ``TARGET_TEST_SESSIONS``, measured by a concurrency ladder against the reference on
#: the graded shape (``MEASURED_CONCURRENCY``); the row counts and the arrival rate are
#: derived from it.
PROFILE_PENDING = False

#: Per-turn latency budgets, in milliseconds: what production coding agents are
#: measured to deliver. Across 4,265 Claude Code and Codex sessions, TTFT ran 1.5 s at
#: 12k input tokens to 2.9 s at 185k, and decode held 50-61 tok/s (TraceLab,
#: arXiv:2606.30560). Published constants target chat-scale prompts and are several
#: times tighter than any stack achieves above 100k tokens, where this task's prompts
#: live; deriving the budget from measured latency instead follows Tempo
#: (arXiv:2504.20068).
#:
#: The TTFT budget is the straight line through those two points rather than a single
#: number, because they are two points and a flat budget can honour at most one of
#: them. A prompt grows by two orders of magnitude over a trajectory and prefill time
#: grows roughly with it, so one deadline would be generous on turn one and out of
#: reach at depth, and every deep turn would miss for a reason that says nothing about
#: how well the artifact serves.
#:
#: The line, rounded to whole numbers: 1,400 ms at a prompt of zero, 0.0080 ms per
#: prompt token. The two points put it at 1,403 ms and 0.0081; the rounding costs 20 ms
#: at 185k, which is 0.7% of a budget derived from two published figures in the first
#: place, and buys constants a reader can hold in their head.
#:
#: Above 185k the budget is an extrapolation of those two points, not a measurement.
#: That is most of this task's range -- the deep sessions resume at 200k, 400k and
#: 600k -- and it is a deliberate choice: the published figures stop at 185k, and a
#: line through two cited points is a number a reader can check, where one fitted from
#: this harness's own reference would be a number only this harness can produce.
TTFT_BUDGET_MS = 1400.0
TTFT_MS_PER_PREFILL_TOKEN = 0.0080
#: TPOT gets no such term. Decode can slow as the KV cache grows, but this checkpoint
#: is built so it should not: 34 of its 45 layers are KDA linear attention, whose state
#: is fixed-width, and the other 11 read 2,048 selected keys per query however long the
#: cache is. The cited 50-61 tok/s is likewise a rate rather than a length-dependent
#: one, so a flat budget is what those measurements support.
TPOT_BUDGET_MS = 20.0
SEED = 42
CONTAINER_PORT = 8000

# The workload itself is assembled further down, beside the drive's window: its
# arrival rate is derived from the number of test sessions and the window they have to
# start inside, and both of those are declared below.

# -----------------------------------------------------------------------------
# DATA.
#
#: Terminal-Bench 4.0, and nothing else: with this loop it reaches 64.3% of the
#: declared window.
#:
#: Depth inside a drive window comes from the pre-roll rather than from the corpus, for
#: a reason that is about the window: 45 minutes of issuing is 55 to 140 turns at this
#: stack's rate, which no corpus outruns. See ``TerminalBenchDataset.deep_rows``.
DATASET = TerminalBenchDataset()
#: Rows the corpus can supply: Terminal-Bench's 66 less the ones it excludes. A hard
#: ceiling on dev + test together, since the partition assigns each row to at
#: most one split.
#:
#: Asked of the dataset rather than recomputed from the module constants, so a corpus
#: built on a different pin cannot be sized by the default's numbers.
USABLE_TASKS = DATASET.usable_tasks()
#: Test sessions the drive offers. A session holds at most one request in flight, so
#: this is how many requests can be outstanding at once before the admission cap binds.
#:
#: Sized against ``MEASURED_CONCURRENCY``, where the ladder and the criterion are, and
#: it has to sit above it: with fewer sessions than admission slots the cap could never
#: bind and the drive would offer less than the point that was measured, whatever the
#: cap said. The margin is thin on purpose and cannot be trimmed: a session spends 93.6%
#: of its wall clock inside an LLM request (443 turns over five trajectories; the gap
#: between a reply landing and the next request is 1.06 s median), so 30 sessions offer
#: about 28 concurrent requests against 24 slots. Fewer sessions
#: would leave slots idle and the server below the measured point.
#:
#: These two are one operating point expressed twice, so the wiring test holds them
#: together: when they disagree the drive runs below the measurement and nothing says so.
TARGET_TEST_SESSIONS = 30

#: Extra sessions the pinned rows contribute beyond one each. Read off the corpus so
#: the arithmetic below recomputes if those rows leave it, rather than holding a
#: session count that quietly stopped being true.
EXTRA_PINNED_SESSIONS = sum(n - 1 for n in DATASET.session_draws().values())

#: Rows in the test split. Derived from the session count rather than the other way
#: round: the load is sessions, and rows are how many it takes to supply them.
TEST_INSTANCES = TARGET_TEST_SESSIONS - EXTRA_PINNED_SESSIONS
#: Rows in the dev split, which is now the remainder rather than the knob. Test takes
#: what its session count needs and dev has the rest; nothing reads dev's size.
DEV_INSTANCES = USABLE_TASKS - TEST_INSTANCES

#: Of those rows, the ones the corpus puts in test by declaring it: the deep rows,
#: pinned because a pre-rolled conversation that hashed into dev would be scored by
#: nobody.
PINNED_TEST_INSTANCES = DATASET.pinned_test_rows()
#: And the rest, which is what the partition is actually asked for.
#:
#: The two differ, and the difference is not cosmetic. A pinned row is held out of the
#: hash ranking and added to test on top of the count, so the pool the count is drawn
#: from is the corpus less the pinned rows. Asking for TEST_INSTANCES would ask the
#: hash for rows the pool does not have and the partition would refuse the whole
#: source. The split still ends up holding TEST_INSTANCES rows: these plus the pinned.
HASHED_TEST_INSTANCES = TEST_INSTANCES - PINNED_TEST_INSTANCES

#: Rows are not sessions. A row supplies one session unless it declares otherwise, and
#: a deep row declares ``DEEP_DRAWS``, so the test split runs more sessions than it has
#: rows. The counts above size the splits; what the drive offers is the sessions those
#: rows supply.
#:
#: Two costs come with a second draw. It is a second container of the same image, so
#: the overlay root needs its 30 GiB as well -- disk scales with sessions in flight,
#: not with rows. And both draws resume the same frozen conversation, so they are
#: repeat samples of one trajectory rather than two trajectories; what differs between
#: them is only what the server samples from the resume point on.
#: Equal to ``TARGET_TEST_SESSIONS`` by construction, since the row count above was
#: derived from it. Kept as its own name because it is what the drive offers and what
#: the arrival rate is computed against, and because the identity is worth being able
#: to assert rather than assume.
TEST_SESSIONS = TEST_INSTANCES + EXTRA_PINNED_SESSIONS
DATA_SOURCE = DataSource(
    dataset=DATASET,
    split=SplitSpec(dev=Count(value=DEV_INSTANCES),
                    test=Count(value=HASHED_TEST_INSTANCES)))
#: Small split for smoke and CI. Its test side is two hashed rows plus the pinned
#: three, for the same reason the full one's is: the pin is a property of the corpus,
#: so every split over this corpus carries those rows however small it asks to be.
SMOKE_DATA_SOURCE = DataSource(
    dataset=DATASET, split=SplitSpec(dev=Count(value=2), test=Count(value=2)))

# -----------------------------------------------------------------------------
# THE SANDBOX. One image per task, because that is the unit both corpora ship: a
# task's inputs, its tools and its installed dependencies are baked into its own
# environment, and there is no image that would serve two of them. The images are
# staged ahead of the run as Apptainer ``.sif`` files, one per task name and all in one
# store, because preparing a corpus of them takes hours and tens of gigabytes and
# neither belongs inside a measured drive.
#
# Each is the environment image the release published for its task, pulled by digest and
# converted, one per task name, into a single store the sandbox looks up by task name
# (``SANDBOX_IMAGES``).

#: The site store each sandbox path is bound from (``HostPath.store``): the site's
#: ``hls.toml`` names the directory on the node that holds each, and the drive binds it into
#: the driver's container at the path below. A site that names none of them cannot drive
#: this task.
IMAGE_STORE = "agentic-coding-images"
OVERLAY_STORE = "agentic-coding-overlays"
PREFIX_STORE = "agentic-coding-prefixes"
RUNTIME_STORE = "agentic-coding-runtime"

#: Where the staged images live inside the driver's container, one ``.sif`` per task name.
#: Staged ahead of the run: preparing them takes hours and tens of gigabytes, neither of
#: which belongs inside a measured drive.
SANDBOX_IMAGE_DIR = "/hls/sandboxes/images"

#: Where each trajectory's container keeps what it writes, one directory per session.
#: Apptainer's own writable layer is a tmpfs in its session directory, capped at 64 MiB
#: by ``sessiondir max size`` in apptainer.conf and not raisable without an admin, while
#: these tasks declare up to 30 GiB (``datasets.SANDBOX_STORAGE_MB``); a trajectory that
#: writes a real output file gets ENOSPC instead. An overlay directory is a real
#: filesystem and still leaves the image's own files visible underneath, which a bind
#: over the working directory would not.
#:
#: A plain directory, so nothing here reserves or caps a session's share. What the
#: filesystem behind it must have free is that 30 GiB per trajectory in flight, and a
#: run that outgrows it fails inside a container as a bare out-of-space error rather
#: than at launch.
SANDBOX_OVERLAY_DIR = "/hls/sandboxes/overlays"

#: Where the frozen conversations a trajectory can resume from are staged, one file
#: per task, each with its state directory beside it.
#:
#: Which one a task resumes from is the corpus's to declare, not this directory's --
#: the framework loads by task name and refuses a file that does not hash to the
#: declared digest. So a stale directory fails the run rather than changing it.
#:
#: Holds the three pre-rolled prefixes (``preroll.DEPTH_TARGETS``); every other task
#: starts at turn 0 and reads nothing from here.
SANDBOX_PREFIX_DIR = "/hls/sandboxes/prefixes"

#: The repository's copy of the three prefixes, each beside its state as one archive
#: (``prefix.to_repository``). ``hls provision`` places them in the prefix store
#: (``sandbox_place_prefixes``) and places none that does not hash to the digest the
#: corpus pins.
PREFIX_SOURCE_DIR = Path(__file__).resolve().parent / "prefixes"

#: Where the exec server's own interpreter is staged.
#:
#: Harbor runs a small HTTP server inside the sandbox to execute a turn's command, and
#: that server needs python with fastapi and uvicorn. Three of the images cannot supply
#: it: two ship a python without those packages and no usable pip, one ships no python
#: at all. Harbor's own fallback installs them, which needs a network, and every task
#: that declares itself offline gets no route out.
#:
#: So the interpreter is staged once, outside every image, and bound in read-only. Not
#: baked into the images: the corpus pins each task to a published image tag and checks
#: the checkout's declaration against that pin, and a modified ``.sif`` would not be what
#: the pin names.
#:
#: Read-only, and mounted at a path harbor already owns rather than anywhere on the
#: task's PATH, so nothing here is reachable as a tool. These are coding tasks whose
#: difficulty is what their image ships; handing a python to a task about porting C to
#: x86 assembly would change the task.
SANDBOX_RUNTIME_DIR = "/hls/sandboxes/runtime"
#: Where it is bound inside a trajectory's container. Harbor's bootstrap tries the
#: interpreter here after the image's own (``SandboxToolRunner.server_python``), so an image
#: that ships the server's dependencies keeps using its own and this bind is inert for it.
SANDBOX_RUNTIME_MOUNT = "/opt/harbor-server"

#: The device apptainer mounts a squashfs ``.sif`` through. A rootless install has no
#: privileged loop mount, so it goes through squashfuse, and without this the launch
#: fails with "fuse: device not found" -- ``--containall`` gives the driver's own
#: container a minimal /dev that does not include it.
FUSE_DEVICE = "/dev/fuse"


def _sandbox_images() -> SandboxImageSet:
    """The image of every task the corpus carries, as the release published it."""
    repository, digests = release_images()
    excluded = DATASET.excluded()
    return SandboxImageSet(
        store=IMAGE_STORE, repository=repository,
        images=tuple(SandboxImage(name=name, digest=digest)
                     for name, digest in sorted(digests.items()) if name not in excluded),
        # Fixed allowances for the stage of ``hls provision`` that pulls them. The 55 images
        # are 24.7 GB of compressed layers together and the largest is 3.8 GB. A pull holds
        # its layers, the tree it unpacks them to and the file it packs, all on node-local
        # disk, and packing uses every core it is given.
        #
        # Two at a time on 16 CPUs and 64 GiB: a pull of the largest image peaks at about
        # 30 GiB with its page cache, so two of the largest fit. The two largest are 7.4 GB
        # of layers together, so 80 GiB of disk holds them unpacked and packed at once with
        # room to spare.
        cpus=16, memory_gib=64, disk_gib=80, parallel=2,
        # Four hours for the stage and one for any single pull: a pull that takes longer is
        # ended and reported by name, and the others go on.
        time_sec=4 * 3600, pull_timeout_sec=3600)


#: The sandbox images, one per task the corpus carries: each the environment image the
#: release built from the task at the pinned commit and published, named by the digest of
#: its manifest (``datasets.RELEASE_IMAGES_FILE``). ``hls provision`` pulls the ones the
#: site's store lacks, as ``<task_name>.sif``. Nothing is built from the checkout's
#: Dockerfiles: most name their base image by tag and install unpinned packages, so a
#: build made later is another image, while the published one is the same bytes everywhere.
SANDBOX_IMAGES = _sandbox_images()


def sandbox_image(directory: str, context: dict) -> str:
    """The image one trajectory's container runs, by task name.

    The path is composed here rather than carried on the corpus row because where the
    images were staged is a property of the host that runs the sandbox, while the row
    is written on whichever host provisioned the data.

    ``directory`` is a parameter so that whoever binds this to the runner settles it,
    once, for every trajectory."""
    task_name = context.get(TASK_NAME_FIELD)
    if not task_name:
        raise KeyError(
            f"the sandbox needs a {TASK_NAME_FIELD} to pick an image; the seed's "
            "metadata carried none")
    return f"{directory}/{task_name}.sif"


def sandbox_allow_internet(context: dict) -> bool:
    """Whether one trajectory's container gets a network, per the task's own definition.

    Per session rather than per run, because a row declares its own. Terminal-Bench
    leaves ``allow_internet`` unset and its tasks expect a network, so a row without
    the field defaults to True."""
    return bool(context.get(ALLOW_INTERNET_FIELD, True))


def sandbox_cpus(context: dict) -> int:
    """CPUs one trajectory's container is given: ``SANDBOX_CPUS`` for every session.

    Read off the session's context, which is the seed's metadata. The seed
    (``datasets._seed_item``) carries neither a row's ``cpus`` nor its ``memory_mb``,
    so the row's capped figure (``datasets.sandbox_fields``) does not reach the launch
    and the constant is what the backend gets."""
    return int(context.get(CPUS_FIELD) or SANDBOX_CPUS)


def sandbox_memory_mb(context: dict) -> int:
    """Memory one trajectory's container is given: ``SANDBOX_MEMORY_MB`` for every
    session, for the reason ``sandbox_cpus`` gives.

    A container that passes it is OOM-killed mid-command, after which the runtime's own
    error comes back as ordinary command output and the model reads a harness failure
    as its own command failing."""
    return int(context.get(MEMORY_MB_FIELD) or SANDBOX_MEMORY_MB)


def sandbox_lower_overlays(directory: str, context: dict) -> tuple[str, ...]:
    """The state a session resumes into, or nothing for one that starts at turn 0.

    The overlay upper the pre-roll's own container wrote, mounted read-only under this
    session's fresh writable layer so it sees what the pre-roll left. Read-only is what
    lets the two draws of a task share one directory.

    Keyed on the task name, like the image, and present only for a row whose corpus
    declared a prefix. A missing directory is not an error here: the framework has
    already loaded that prefix and checked the directory against its manifest, so a
    session that gets this far has one."""
    if not context.get(PREFIX_DIGEST_FIELD):
        return ()
    state = prefix_format.state_path(directory, str(context[TASK_NAME_FIELD]))
    return (str(state),) if state.is_dir() else ()


def sandbox_row_entries(directory: str, context: dict) -> tuple[tuple[str, str, str], ...]:
    """What a deep session needs in the prefix store beside its image: its frozen
    conversation and the state its pre-roll left, named with the digest its row pins; nothing
    for a session that starts at turn 0."""
    digest = context.get(PREFIX_DIGEST_FIELD)
    if not digest:
        return ()
    name = str(context[TASK_NAME_FIELD])
    return ((str(prefix_format.prefix_path(directory, name)), "file", f"the prefix of {name} ({digest[:12]})"),
            (str(prefix_format.state_path(directory, name)), "dir", f"the state of {name} ({digest[:12]})"))


def sandbox_place_prefixes(source: Path, rows: list[dict], stores: Mapping[str, Path]) -> list[str]:
    """Copy each deep row's prefix and its state from *source*, the repository's
    ``PREFIX_SOURCE_DIR``, into the prefix store, checked against the digest the row pins
    (``prefix.place``); one line per prefix. A refusal is StoreRefused, in one line."""
    pins = {str(row[TASK_NAME_FIELD]): str(row[PREFIX_DIGEST_FIELD]) for row in rows if row.get(PREFIX_DIGEST_FIELD)}
    done = []
    for name, pin in sorted(pins.items()):
        try:
            placed = prefix_format.place(source, stores[PREFIX_STORE], name, pin)
        except prefix_format.PrefixError as e:
            raise StoreRefused(str(e)) from e
        done.append(f"{PREFIX_STORE}: the prefix and state of {name} ({pin[:12]}), {placed}")
    return done


def sandbox_workdir(context: dict) -> str:
    """The directory one trajectory's container starts in.

    Recorded on the row from the task's own Dockerfile. It has to be the directory that
    image was built around: the container backend passes it through verbatim, and a
    start in a directory the image does not have fails the launch."""
    return context.get(WORKDIR_FIELD) or WORK_DIR


#: Replacement containers one session may have before a dead sandbox ends its
#: trajectory, which the driver then drops (``hls.core.faults.SandboxGone``).
#:
#: On, because the reference loses exec servers on these rows: the connection to the
#: in-container exec server can die as an unhandled ``httpx.ReadError`` and never come
#: back, and the trajectory then goes on accumulating prompt depth with no command
#: running. A session is the traffic this task exists to generate, so the first move is
#: to get a container back.
#:
#: Five, which is generous: a session that runs out of restarts has a run environment
#: that is broken rather than unlucky, and it is dropped with every turn it did not
#: reach, so the budget must not be a hair-trigger. A container starts in about 13 seconds, so five
#: replacements are about a minute of a 45-minute window, and a replacement is built from
#: the session's own launch spec, so the overlay upper comes back and little is lost with
#: it.
#:
#: What it does not cover, and should not: a container that cannot survive at all.
#: Rebuilding one indefinitely spends the drive's wall clock and produces no turns,
#: which is worse than ending the session and saying why.
MAX_SANDBOX_RESTARTS = 5

#: Sandbox faults in a row that end a trajectory, when the runner is handing them to
#: the model rather than restarting.
#:
#: The backstop, and it has its own stop reason: left to run, a session against a dead
#: container spends its format-repair budget and is recorded as the model failing to
#: produce calls, which sends a reader to the parser rather than to the container.
#: Three because a trajectory can take a single fault and then run a hundred clean
#: turns, so a fault is not by itself a dead sandbox.
#:
#: Inert as this task ships, and kept anyway. A fault only reaches the loop's counter
#: as a failed command, and the runner returns one of those only when its restart
#: budget is zero; with the budget above at five it raises instead, the moment the
#: budget is spent. So under this configuration the counter never advances and this
#: number never fires. It is live the instant ``MAX_SANDBOX_RESTARTS`` goes to zero.
MAX_CONSECUTIVE_SANDBOX_FAULTS = 3

TOOL_RUNNER = SandboxToolRunner(
    # The runner is shipped to the driver by value, so a bound argument travels with it
    # where a lookup would run again at the far end.
    image=functools.partial(sandbox_image, SANDBOX_IMAGE_DIR),
    env_factory=default_env_factory,
    workdir=sandbox_workdir,
    # The pre-rolled state a deep session resumes into, read-only under its own
    # writable layer. Empty for a cold session; a pre-rolled one mounts its prefix.
    lower_overlays=functools.partial(sandbox_lower_overlays, SANDBOX_PREFIX_DIR),
    # What a plan checks the prefix store for, beside each row's image, and what
    # provisioning copies there from the repository.
    row_entries=functools.partial(sandbox_row_entries, SANDBOX_PREFIX_DIR),
    place_rows=functools.partial(sandbox_place_prefixes, PREFIX_SOURCE_DIR),
    # No setup step. The image is the environment: its inputs are already in place,
    # which is what made the per-task image worth staging.
    setup_command=None,
    # No scratch over the working directory. The runner's default puts an empty tmpfs
    # there, which is right for a sandbox whose workdir starts empty and wrong here:
    # most of these images ship the task's inputs at that path, and a tmpfs hides every
    # one of them. The container is already writable through apptainer's overlay, so
    # the scratch buys nothing this task needs.
    scratch_workdir=False,
    # What one trajectory's container is given: 4 CPUs and 16 GiB for every session
    # (``SANDBOX_CPUS``, ``SANDBOX_MEMORY_MB``). The resolvers read the session's
    # context, and the seed carries neither field of a row's declaration.
    #
    # Terminal-Bench's own 66 definitions, parsed per block, declare:
    #
    #   cpus        2 to 16.  58 of the 66 fit 4.
    #   memory_mb   4,096 to 32,768.  65 fit 16,384.
    #   storage_mb  4,096 to 1,024,000.  64 of the 66 fit 30,720.
    #
    # A container that passes its memory is OOM-killed mid-command, and every later
    # turn returns the runtime's own error as ordinary command output, so the model
    # reads a harness failure as its own commands failing and keeps going, with nothing
    # in the reply to say so.
    #
    # `takens-embedding-lean` asks for 1 TB of disk and gets the 30 GiB cap, because
    # thirty trajectories share one overlay root.
    #
    # The tasks that declare a GPU are not here at all. They are excluded from the
    # corpus (`datasets.NEEDS_ACCELERATOR`) rather than run without one: this node's
    # eight belong to the server under measurement, and an agent flailing at a problem
    # it cannot solve is not the load this task exists to measure. The row still
    # carries the declaration, so a future sandbox with an accelerator can re-admit
    # them by emptying that set.
    cpus=sandbox_cpus, memory_mb=sandbox_memory_mb,
    # Replace a container that has stopped executing commands rather than telling the
    # model its command failed. See MAX_SANDBOX_RESTARTS.
    max_sandbox_restarts=MAX_SANDBOX_RESTARTS,
    # The exec server's own interpreter, read-only, at the path harbor looks for one
    # after it has tried the image's. Every session gets it; an image that ships the
    # server's dependencies never reaches this path, so the bind is inert there. See
    # SANDBOX_RUNTIME_DIR for why it is bound rather than baked. The tree is the one
    # provisioning stages under the store for its pins (hls.provision.sandbox_runtime).
    mounts=[{"type": "bind", "source": f"{SANDBOX_RUNTIME_DIR}/{RUNTIME_TAG}",
             "target": SANDBOX_RUNTIME_MOUNT, "read_only": True}],
    server_python=f"{SANDBOX_RUNTIME_MOUNT}/bin/python3",
    # What the driver's own container has to be given. It runs under --containall,
    # which drops the bind paths apptainer.conf configures cluster-wide along with the
    # host home and tmp, so none of these is reachable unless it is named here: the
    # images, the overlay root the runner creates each session's directory under, the
    # exec server's interpreter, and the fuse device the runtime mounts a .sif through.
    # The prefix store is the framework's rather than this runner's, but this list is
    # the only declaration of what the driver's container gets bound, and the
    # framework reads a prefix from inside it.
    #
    # Every sandbox sees these binds as the driver has them, so the stores a drive only
    # reads are read-only and only the overlay root, which the runner writes each
    # session's directory under, and the device stay writable: a graded model that can
    # write the prefix store changes what every later drive of the task resumes from.
    host_paths=(HostPath(SANDBOX_IMAGE_DIR, read_only=True, store=IMAGE_STORE),
                HostPath(SANDBOX_OVERLAY_DIR, read_only=False, store=OVERLAY_STORE),
                HostPath(SANDBOX_PREFIX_DIR, read_only=True, store=PREFIX_STORE),
                HostPath(SANDBOX_RUNTIME_DIR, read_only=True, store=RUNTIME_STORE),
                HostPath(FUSE_DEVICE, read_only=False, store=DEVICE)),
    overlay_root=SANDBOX_OVERLAY_DIR,
    # Each row's own declaration rather than one setting for the run. A row that sets
    # allow_internet = false gets a container with no route out. Terminal-Bench leaves
    # it unset and takes the default of True, so its rows keep a network.
    allow_internet=sandbox_allow_internet,
)

# -----------------------------------------------------------------------------
# THE DRIVE'S WALL CLOCK. A drive is a bounded window rather than a fixed number of
# requests: a trajectory's length is the model's to choose, so a run sized in requests
# would take as long as the slowest server made it.
#
# Not a deadline a server can fail. Reaching either is a scheduled stop that records no
# fault -- unlike ``AgentSpec.drive_timeout_sec``, which keeps its fault semantics as a
# backstop for the case the scheduled stop does not fire.
#
# The brief reads both, so this is the one place the numbers are written down.

#: How long the drive issues requests for. A session that wants a turn after this stops
#: where it is, with the work it has done kept and scored.
DRIVE_WALL_CLOCK_SEC = 45 * 60
#: After the wall clock, the window in which open streams are allowed to finish. What
#: is still open at the end is cut and scored on what it streamed.
DRIVE_CUT_GRACE_SEC = 10 * 60

# These two are the whole of the drive's timing. The backstop -- ``drive_timeout_sec``,
# whose expiry IS a fault -- is not declared here and must not be: ``ScheduledStop``
# derives it as the two above plus ``BACKSTOP_MARGIN_SEC``, which comes to 3,420
# seconds against a planned ending at 3,300. Declaring it would be a third number to
# keep in step with the two that determine it, and the failure that number has when it
# drifts is not a mild one: a backstop at or before the planned ending cancels the
# trajectories the stop was about to cut cleanly and records every one as a fault, so a
# drive ending exactly as designed zeroes and reads as the contestant's failure.
# ``AgentSpec`` refuses that arrangement at import rather than correcting it.
#: How long a reply may go between two tokens before the request fails. Unlike the two
#: above this is a failure: the window closing says nothing about a server, but a
#: stream that delivered tokens and then stopped is one that stopped working.
#:
#: Armed only once a first token has arrived, so a long prefill is never charged here.
#: Set far above the longest gap a healthy server shows on this traffic -- a vLLM
#: preemption recomputing a 600k-token context is tens of seconds -- because it ends
#: the run rather than one request's credit. Every turn records its own longest gap,
#: so the reference drive says how much headroom this leaves.
STALL_TIMEOUT_SEC = 120

#: How long the drive has to get every test session started, as a share of the issuing
#: window. A third of it.
#:
#: Not the whole window, and the difference is what the run measures. A session that
#: arrives near the stop takes one shallow turn and is cut, which costs a full sandbox
#: and pulls the measured population toward exactly the depth this task exists not to
#: measure. Compressing the arrivals keeps every session deep enough to be worth its
#: container.
#:
#: What that buys in disk, since it is the other side of the trade: arrivals inside 900
#: seconds means close to ``TEST_SESSIONS`` sandboxes alive at once, and at roughly
#: 30 GiB of overlay each (``datasets.SANDBOX_STORAGE_MB``) that is about 1.3 TB held
#: at the same time rather than spread over the drive.
#:
#: The filesystem behind the site's overlay store has to hold that, and a run that
#: exhausts it fails inside a container as a bare out-of-space error partway through
#: rather than at launch.
ARRIVAL_WINDOW_SEC = DRIVE_WALL_CLOCK_SEC / 3.0

#: Session arrival rate, sessions per second.
#:
#: Derived, not chosen. The rate has one job, which is to get every test session
#: started inside the window above, so it follows from how many there are and how long
#: they have; ``Poisson.covering`` is that inverse, with headroom for the draw. A rate
#: picked independently can schedule arrivals past the issuing window. Those sessions
#: are never opened, and the drive scores part of its declared corpus and reports
#: nothing unusual, because every session it did open behaved.
#:
#: So the load knob stays the session count, as ``PROFILE_PENDING`` says: a split that
#: moves now carries the rate with it instead of leaving it behind.
POISSON_RATE_PER_SEC = Poisson.covering(
    TEST_SESSIONS, ARRIVAL_WINDOW_SEC).rate_per_sec

WORKLOAD = Workload(
    arrival=Poisson(rate_per_sec=POISSON_RATE_PER_SEC),
    # No synthetic think time. A deep-research task inserts one to stand in for the
    # latency a live web tool had; here the gap between turns is a real command
    # running in a real container, already inside the makespan.
    inter_turn_delay=NoDelay(),
    output_length_cap=ShortOutputCap(cap_tokens=THINK_CAP),
    # Per request, not per trajectory: r = max(TTFT/budget, mean TPOT/budget), so the
    # more stringent phase is what a turn is scored on. A trajectory therefore weighs
    # by its turn count, which a trajectory-makespan grain would avoid; the trade is
    # that tool wait, 59.8% of a coding agent's request time and none of it the server's
    # doing, does not dilute the measurement.
    # The TTFT budget grows with the prompt. A trajectory's first turn prefills a few
    # thousand tokens and a deep one several hundred thousand, so one flat deadline
    # would be generous at the start of a session and unreachable at the end.
    slo=ContextLinearLatencySLO(
        ttft_ms=TTFT_BUDGET_MS, tpot_ms=TPOT_BUDGET_MS,
        ttft_ms_per_prefill_token=TTFT_MS_PER_PREFILL_TOKEN),
)

AGENT = AgentSpec(
    framework=MiniSweFramework(
        max_context_len=MAX_CONTEXT_LEN, prefix_dir=SANDBOX_PREFIX_DIR,
        checkpoint=MODEL_ID,
        # A reply the cap cut is continued rather than lost; see THINK_CAP.
        reply_budget=REPLY_BUDGET,
        # The same two constants the runner gets, so the loop and the runner count one
        # budget. Both stay out of the fingerprint (``MiniSweFramework._FINGERPRINT_EXEMPT``).
        max_sandbox_restarts=MAX_SANDBOX_RESTARTS,
        max_consecutive_sandbox_faults=MAX_CONSECUTIVE_SANDBOX_FAULTS),
    tool_runner=TOOL_RUNNER,
    max_turns=MAX_TURNS,
    max_concurrent_llm=MAX_CONCURRENT_LLM,
    wall_clock_sec=DRIVE_WALL_CLOCK_SEC,
    cut_grace_sec=DRIVE_CUT_GRACE_SEC,
    stall_timeout_sec=STALL_TIMEOUT_SEC,
    # No tool assets. A task's data is inside its own image, so there is nothing left
    # for provisioning to stage beside it.
)

# -----------------------------------------------------------------------------
# CONTRACT. Streaming, so per-step TTFT and inter-token latency are measurable, and
# asking for the report the gate grades: the served token's log-probability and its
# top alternatives at every position, each with an integer id.
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http-sse",
    # sends_tools: every turn carries the bash tool definition, so the brief and
    # the sample request body show `tools` and `tool_choice: "none"` rather than
    # letting a contestant meet them for the first time in a graded run.
    protocol=OpenAIChatProtocol(
        return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED,
        # The sample shows two alternatives where a reply carries 21; the brief says so.
        states_sample_abbreviation=True,
        sends_tools=True,
        # GLM-5.3-Flash's own call syntax, from its chat template. The template
        # pre-opens <think>, so a reply closes the channel without opening it.
        tool_call_example=(
        "Let me look at the repository first.</think>I'll list the directory."
        "<tool_call>bash<arg_key>command</arg_key>"
        "<arg_value>ls -la</arg_value></tool_call>")),
)

# THE REFERENCE. vLLM rather than HF transformers, which is a departure from every
# other task here.
#
# Transformers does support this architecture -- Glm5Next has been registered since
# v5.16.1. What it does not have is a kernel. The indexer picks a different 2,048 keys
# for every query, which no flash-attention mask API can express, so
# Glm5NextPreTrainedModel sets _supports_flash_attn = False and leaves SDPA and eager.
# Both score every query against every key pool in one fp32 tensor -- 32 GiB at a 32k
# prefix, 512 GiB at 128k -- then build the selection mask densely over queries by keys
# on top of that. Prefixes here reach several hundred thousand tokens by the end of a
# long trajectory. vLLM implements the sparse attention directly and prefills it paged
# and chunked.
#
# What it costs: the calibration pool loses a member. kappa is measured from SGLang
# alone rather than from two stacks against a neutral third, so the tolerance encodes
# one stack's quirks.
#
# The decode is Sample, which this engine can score but not generate under. That is the
# intended asymmetry: the gate reads distributions and never asks it for tokens.
_ORACLE = VLLMReference(decode=DECODE, max_model_len=MAX_CONTEXT_LEN,
                        gpu_memory_utilization=0.90, tensor_parallel_size=8)

# VERIFIERS. The first is the only gate; the rest report.
#
#   1. ReportFaithfulnessVerifier -- the calibrated sampling gate. Needs a
#      calibration.json, the store's or the task's default in defaults/,
#      fingerprinted to this task's model, contract, workload, split and seed.
#   2. TrajectoryServingMetrics -- per-step and per-session serving telemetry.
#   3. TrajectoryShape -- turns taken, commands run, and how each trajectory ended.
#      The check that the traffic was the traffic intended.
#
# The gate completes a normal stop that served no end-of-turn token with the one the
# reference rates highest next (implied_end_of_turn), so a reply that stops one token
# short of its end-of-turn id is scored on where it stopped rather than failing the run.
VERIFIERS = [
    ReportFaithfulnessVerifier(reference=_ORACLE, implied_end_of_turn=True),
    ReportOnlyMetricVerifier(metric=TrajectoryServingMetrics()),
    ReportOnlyTrajectoryShapeVerifier(),
]

# The ceiling is measured from one trusted stack, sglang: `hls calibrate --engines
# sglang`. vLLM is the reference here, and a stack cannot calibrate against itself --
# its divergence from the reference would be zero and would set a ceiling no honest
# server could meet. The pool member itself is built by
# `hls.calibrate._member_engines` from this task's model, decode, contract and
# declared hardware; there is nothing for the task to declare.

SCORER = SloLatencyScorer()

#: What a grading holds: the whole node, a 1,911 GiB job on its 64 CPUs. The server gets the job
#: less the runner's reserve. The driver runs on the server's CPUs, its 768 GiB, which every
#: trajectory's sandbox is launched inside, oversubscribed on the server's memory: its own step
#: limit and its memory guard are those 768 GiB, and the job's limit bounds the two together.
#: The server writes to the brief's 4,096 GB of storage (3,814 GiB).
EVALUATION_ALLOCATION = EvaluationAllocation(cpus=64, job_memory_gib=1911, writable_gib=3814,
                                             driver_memory_gib=768, driver_memory_shared=True)
#: What a build holds: the whole node's 64 CPUs and 1,911 GiB.
BUILD_ALLOCATION = BuildAllocation(cpus=64, memory_gib=1911)

# DRIVE BOUNDS (``hls.drive.bounds``). The largest published trace of this task holds
# 4,935,885,306 bytes of event JSON, of which 524,461,711 are its requests; its largest reply
# is 54,743,852 bytes of JSON, at the 32,768-token cap with logprobs. At most
# ``MAX_CONCURRENT_LLM`` replies are in flight at once, so a drive holds at most that many of
# the largest reply unwritten. The requests are given room.
#
# The verify role holds the reference sharded over the node's eight GPUs, so it has the
# evaluation's job (``EVALUATION_ALLOCATION``, 1,911 GiB) less the runner's default 4 GiB
# reserve, not the one-GPU reference job's. Its peak is the reference engine's host memory, the
# checkpoint's 306 GiB and 1 GiB, and 26.3 bytes per byte of trace, the report gate's.
DRIVE_BOUNDS = bounds_for(CONTRACT, trace_bytes=4_935_885_306, held_reply_bytes=MAX_CONCURRENT_LLM * 54_743_852,
                          trace_request_bytes=640 * 2**20,
                          verify_peak=VerifyPeak(base_bytes=307 * 2**30, bytes_per_byte=26.3),
                          verify_memory_bytes=(1911 - 4) * 2**30)

#: The prompt depths the pre-rolled sessions resume at, as the brief states them. Read
#: off the pre-roll rather than restated, so the two cannot drift.
_DEEP_BANDS = (", ".join(f"{d:,}" for d in DEPTH_TARGETS[:-1])
               + f" and {DEPTH_TARGETS[-1]:,}")
#: How many sessions resume a pre-rolled conversation: every depth band is drawn
#: ``DEEP_DRAWS`` times.
_DEEP_SESSIONS = DEEP_DRAWS * len(DEPTH_TARGETS)

#: The vLLM commit the reference serves from: the vendor image ``VLLM_IMAGE`` pins by digest
#: is vLLM's build of this commit. Here rather than beside the image because the brief names it
#: (a contestant is scored against what this build reports, so "the reference" has to say
#: which build), and the brief is defined above the images.
VLLM_COMMIT = "a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b"

# The reference service's figures the brief states, measured on one 8xH200 node with the
# baseline image, this checkpoint and _ORACLE's engine. GPU memory
# per device is nvidia-smi's, awake and asleep; wake and sleep are the slower of two
# explicit cycles; the wake need is the service's own measurement (awake minus asleep, plus
# its 1 GiB margin); the score time is the reference bundle's long-32k-0 (32,604 prompt
# and 105 continuation tokens), awake. Startup is launch to asleep with a warm compile
# cache and the node-local weights copy in place: 498.9 s at most. The
# brief states the limit the stage holds startup to (READY_TIMEOUT_SEC) beside it, and the
# spread below at its measured maximum.
REFERENCE_SERVICE_AWAKE_GIB = 125.5
REFERENCE_SERVICE_ASLEEP_GIB = 13.9
# Host memory the service holds asleep: its pinned copy of the weights, 40.5 GiB on each of the
# 8 ranks for this checkpoint. It is in the same memory as the agent's session.
REFERENCE_SERVICE_ASLEEP_HOST_GIB = 324
REFERENCE_SERVICE_WAKE_NEED_GIB = 112.4
REFERENCE_SERVICE_WAKE_SEC = 4.1
REFERENCE_SERVICE_SLEEP_SEC = 3.5
REFERENCE_SERVICE_SCORE_32K_SEC = 2.5
REFERENCE_SERVICE_SCORE_32K_TOKENS = 32_604 + 105
REFERENCE_SERVICE_STARTUP_SEC = 500
# How far the reference's own scores move between identical calls, which the service
# reproduces because it runs the grader's path. Measured with the
# grader's exact engine, with and without sleep mode (the spread is the same): the
# reference bundle's sequences below 2,048 tokens (dev-00, cont-00) scored bit-identically
# on every pass; long-32k-0 (32,709 tokens) scored twice moved at sampled prompt positions
# by a median 0.16 nats with 19% of them over 1 nat, and at its 105 continuation positions
# by a median 0 and at most 0.62 nats. 2,048 is the indexer's selection width: a shorter
# sequence's queries select every key.
REFERENCE_SERVICE_EXACT_BELOW_TOKENS = 2048
REFERENCE_SERVICE_REPEAT_PROMPT_MEDIAN_NATS = 0.16
REFERENCE_SERVICE_REPEAT_PROMPT_OVER_1_NAT_PCT = 19
REFERENCE_SERVICE_REPEAT_CONT_MAX_NATS = 0.62

# The brief is a specification: every sentence states a rule, an interface or a number
# the contestant needs, with nothing explanatory around it. Numbers are read off the
# constants they describe so the brief cannot drift from the harness.
_BRIEF = ("""{% extends "brief" %}
{% block goal %}Serve `{{ model.base_checkpoint }}` under the workload below. The score is a
speedup over the reference stack, `exp(mean ln(1/r) - B)`, where r is a request's
SLO-normalized latency, defined here; higher is better.

**Traffic.** Each session is one coding-agent trajectory. On every turn the grader sends
the whole conversation so far as one chat request, executes the shell command in your
reply, and appends the command's output before the next turn. """ + f"{TARGET_TEST_SESSIONS}" + """ sessions arrive
as a Poisson process at the rate under Workload, set so that all """ + f"{TARGET_TEST_SESSIONS}" + """ arrive within the
first """ + f"{ARRIVAL_WINDOW_SEC / 60:g}" + """ minutes; sessions run concurrently, with at most """ + f"{MAX_CONCURRENT_LLM}" + """ requests in flight
at once. """ + f"{_DEEP_SESSIONS}" + """ of the sessions start from a pre-rolled conversation, """ + f"{DEEP_DRAWS}" + """ at each of
""" + _DEEP_BANDS + """ prompt tokens; those run at most """ + f"{DEEP_TURN_BUDGET}" + """ further turns.

**Two requests per turn.** A turn's first request sets `max_tokens` = """ + f"{THINK_CAP:,}" + """. If the
reply reaches that cap, the grader closes it with `</think>` (appended unless the reply
already contains it) and sends a second request with
`max_tokens` = """ + f"{REPLY_BUDGET:,}" + """, `continue_final_message: true`, `add_generation_prompt: false`,
and the first request's messages followed by the reply so far as the final assistant
message. A continuation that reaches its cap ends the turn. Both requests are scored and
verified as ordinary requests.

**Score.** For every request:

    budget = """ + f"{TTFT_BUDGET_MS:,.0f}" + """ ms + """ + f"{TTFT_MS_PER_PREFILL_TOKEN}" + """ ms x prompt_tokens
    r      = max( TTFT / budget , mean TPOT / """ + f"{TPOT_BUDGET_MS:,.0f}" + """ ms )
    credit = ln(1 / r)
    score  = exp( mean credit over all requests - B )

B is the mean credit the reference stack earned on this workload.

**Window.** Requests are issued for """ + f"{DRIVE_WALL_CLOCK_SEC // 60}" + """ minutes; at """ + f"{(DRIVE_WALL_CLOCK_SEC + DRIVE_CUT_GRACE_SEC) // 60}" + """ minutes every open stream
is cut. A request still streaming at the cut is scored as if one more token completed at
the cut instant: a request with no token yet is charged the whole wait as TTFT; one with
n tokens is charged the span from its first token to the cut over n intervals. That token
is not part of the reply and is not verified.

**Failure.** After a request's first token, """ + f"{STALL_TIMEOUT_SEC}" + """ seconds without a further token fails the
request. Time to first token has no limit. Any failed request, or any request without a
complete streaming response, scores the run 0.{% endblock %}
{# The window is stated here rather than declared as TaskSpec.max_context_len: that
   field enters the workload fingerprint, and the framework and the reference already run
   at this window (MiniSweFramework.max_context_len, the oracle's max_model_len). #}
{# The readiness probe the harness sends after /health
   (hls.drive.readiness.wait_inference_ready), which the base brief does not state. #}
{% block health %}{{ super() }}

After /health returns 200, the harness sends one probe request to the inference
endpoint below and repeats it until the response status is 2xx, 3xx or exactly 400.
Any other status (404, 405, 422, other 4xx, 5xx) or no response keeps the server not
ready, and the startup timeout fails the run.{% endblock %}
{% block model %}{{ super() }}
- Context window: """ + f"{MAX_CONTEXT_LEN:,}" + """ tokens; serve the model at this window.{% endblock %}
{# The scoring service shares the GPUs (## Reference Scoring Service), so the line says only
   which they are. #}
{% block accelerators %}The accelerators are `cuda:0` through `cuda:{{ hardware.accelerator_count - 1 }}`.{% endblock %}
{# The base note describes tenants whose rows are wrapped in on-disk prefix text. This
   task's rows are task records, and the prompts are built by the grader at run time. #}
{% block dev_reference_note %}Each row is one coding task record: `instruction` (the task
  statement the trajectory opens with), `task_name`, `category`, `workdir`, `source`,
  `expert_time_estimate_hours`, the sandbox fields `cpus`, `memory_mb`, `storage_mb`,
  `gpus`, `allow_internet`, and the draw fields `prefix_digest`, `turn_budget`, `draws`,
  `split`. Nothing else sits beside the rows. The served prompts are the trajectories the
  grader builds from those tasks at run time and are not on disk.{% endblock +%}
{# The base block says a turn is one request; here a turn is one or two (the
   continuation rule in Goal), and the output cap differs between them. #}
{% block workload %}Arrivals are sessions. A session's turns are generated as it runs, up to
{{ turn_bound }}, and each turn is one request or two under the continuation rule above, so a
session's request count is not known in advance.
- Sessions: {{ workload.sessions }}
- Turns: up to {{ turn_bound }} per session
- Output cap: """ + f"{THINK_CAP:,}" + """ tokens on a turn's first request, """ + f"{REPLY_BUDGET:,}" + """ on its continuation
- SLO: {{ workload.slo }}{% endblock %}
{# The base block states the gate qualitatively and in numbers but never says what the
   reference is. Named here because it decides what a faithful server has to match: the
   engine, the weights as shipped, and no precision claim beyond the gate. Unnamed, "the
   reference model" reads as a transformers bf16 forward pass.

   The second paragraph says how the reference distribution is obtained, because the
   comparison is not symmetric: the grader teacher-forces the served tokens through the
   reference's prefill path while the contestant's numbers come from whatever path it
   decoded with. A contestant who assumes both sides are decode-path numbers cannot tell
   its own deviation from the prefill/decode difference the ceiling already absorbs. #}
{% block decoding %}{{ super() }}

The reference is this checkpoint served by vLLM at commit `""" + VLLM_COMMIT[:8] + """` on the FP8
weights as shipped, decoding under the rules above. A `transformers` forward pass in bf16
is not the reference. Numerical precision and approximations are yours to choose; the
verification above is the only faithfulness constraint.

The reference distribution at a served position is computed by teacher-forcing your served
tokens through the reference's prefill path; your reported distribution is whatever path
produced your sample. Prefill and decode use different kernels in most stacks, including
the reference; the threshold was calibrated on a serving stack's sampled tokens scored this
way, so it absorbs that difference.{% endblock %}
{# The rule the Blocked list enforces, stated where the list renders: the list is a set of
   names a proxy can refuse, not the prohibition itself, and a contestant who reads only
   the list concludes that a framework published under an unlisted name is permitted. The
   audit it names reads the build's source bundle, which the package stage keeps for every
   graded build. #}
{% block package_policy %}{{ super() }}

Your server and its kernels are your own work on the installed toolchain: no third-party
LLM inference engine, serving framework, or attention, GEMM, MoE or linear-attention
kernel library may be installed, vendored or copied, whether or not `pip install`
refuses it.
Submissions are audited against this rule; the Blocked list names the ones already known
and does not bound it.{% endblock %}
{# The toolchain this task's sandbox holds, which its own sandbox image provides. #}
{% block toolchain %}Installed, with offline documentation at the paths given (relative to your working
directory):

### Python
`torch`, `transformers`, `accelerate`, `datasets`, `fastapi`, `uvicorn`, `pydantic`.

- torch, including `torch.compile`/Inductor and `torch.utils.cpp_extension` -- `references/torch/`

### C / C++
`gcc`/`g++` 13.3, `clang`, `make`, `cmake`, `ninja`, `pkg-config`, `python3-dev`.

`TORCH_CUDA_ARCH_LIST` is set for this GPU; torch's `torch/include` headers and vendored
pybind11 are present, so `torch.utils.cpp_extension.load()` and setuptools
`BuildExtension` build offline.

- `cpp_extension` usage -- `references/torch/`

### Rust
`rustc`, `cargo`.

- The Rust Reference -- `references/rust/reference/`
- The Rust Book -- `references/rust/book/`

### CUDA toolchain
Compilers and binary tools: `nvcc`, `ptxas`, `nvdisasm`, `cuobjdump`.

- PTX ISA -- `references/ptx/` (start at `references/ptx/ptx-isa.md`)
- CUDA Runtime API -- `references/cuda-runtime/`
- CUDA Driver API -- `references/cuda-driver/`
- CUDA Python (`cuda.bindings`, `cuda.core`) -- `references/cuda-python/`
- NVRTC, runtime kernel compilation -- `references/nvrtc/`

Kernel-authoring DSLs (`triton`, `cuda-tile`, `nvidia-cutlass-dsl` are installed):

- Triton, with runnable annotated tutorials -- `references/triton/`
- CUTLASS C++ and the CuTe Python DSL -- `references/cutlass/`
- cuTile -- `references/cutile/`

Math and primitive libraries:

- cuBLAS / cuBLASLt -- `references/cublas/`
- CUB, Thrust, libcu++ (CCCL) -- `references/cccl/`
- cuDNN -- `references/cudnn/`
- cuSPARSE / cuSOLVER / cuFFT -- `references/cusparse/`, `references/cusolver/`, `references/cufft/`
- NCCL is installed

Profiling and debugging: `nsys`, `ncu`, `compute-sanitizer`, `cuda-gdb`.

- Nsight Systems / Nsight Compute / NVTX guides, debugging tools, performance traps --
  `references/profiling/`
{% endblock %}
{# The in-build reference service (AgentProducerSpec.reference_service, launched by the
   substrate, service in hls/reference/service.py), stated only when the flag is set, since
   a build without it has no socket to offer. Rendered after the uploads it extends:
   the reference logprobs there are a fixed sample, this scores any sequence. The figures
   are the measured REFERENCE_SERVICE_* constants. #}
{% block uploads %}{{ super() }}{% if reference_service +%}
## Reference Scoring Service
During the build, and only then, a service scores token sequences with the grader's
reference: it teacher-forces `prompt_token_ids` followed by `continuation_token_ids`
through the reference named under ## Decoding, on its prefill path with prefix caching
off, which is how the grader scores your served tokens. It serves HTTP/1.0 on the unix
socket at `$""" + _SERVICE.SOCKET_ENV_VAR + """` (`""" + _SERVICE.SOCKET_PATH + """`), one request per
connection, one request at a time. It has no TCP port.

    GET  /status  -> {"state": "starting" | "asleep" | "waking" | "awake" | "sleeping" | "unloaded" | "failed",
                      "gpu_free_bytes": [int, ...], "wake_need_bytes": [int, ...], "memory_error": {...} | null,
                      "last_wake_sec": float, "idle_sleep_sec": """ + f"{_SERVICE.IDLE_SLEEP_SEC}" + """, ...}
    POST /score   {"prompt_token_ids": [int, ...], "continuation_token_ids": [int, ...],
                   "prompt_positions": [int, ...]}
                  -> {"continuation": [{"token_id": int, "logprob": float,
                                        "top""" + f"{TOP_LOGPROBS_REQUESTED}" + """": [[int, float], ...]}, ...],
                      "prompt": [{"position": int, "token_id": int, "logprob": float,
                                  "top""" + f"{TOP_LOGPROBS_REQUESTED}" + """": [[int, float], ...]}, ...],
                      "n_prompt": int, "n_continuation": int, "woke": bool, "score_sec": float}
    POST /sleep, POST /wake  -> the /status object

`continuation[j]` is the reference's distribution at `continuation_token_ids[j]` given
every token before it. `prompt[k]` is the same at the prompt token `prompt_positions[k]`;
`prompt_positions` is optional and each entry lies in [1, len(prompt_token_ids)).
Log-probabilities are at temperature 1 and unrounded; `top""" + f"{TOP_LOGPROBS_REQUESTED}" + """` holds the """ + f"{TOP_LOGPROBS_REQUESTED}" + """ most
probable token ids, ranked by log-probability, ties by id. A request with a token id
outside the checkpoint's vocabulary, more than """ + f"{MAX_CONTEXT_LEN - 1:,}" + """ tokens in total, or nothing
to score returns HTTP 400. A """ + f"{REFERENCE_SERVICE_SCORE_32K_TOKENS:,}" + """-token sequence scores in """ + f"{REFERENCE_SERVICE_SCORE_32K_SEC}" + """ s awake.

Scores are bit-reproducible for sequences under """ + f"{REFERENCE_SERVICE_EXACT_BELOW_TOKENS:,}" + """ tokens. Beyond that the
reference itself varies run to run: at prompt positions by a median of """ + f"{REFERENCE_SERVICE_REPEAT_PROMPT_MEDIAN_NATS}" + """ nats per
token (""" + f"{REFERENCE_SERVICE_REPEAT_PROMPT_OVER_1_NAT_PCT}" + """ percent of positions by more than 1 nat), at generated positions by a
median of 0 and at most """ + f"{REFERENCE_SERVICE_REPEAT_CONT_MAX_NATS}" + """; the grader's reference carries the same variation.

The service starts with your session and takes about """ + f"{REFERENCE_SERVICE_STARTUP_SEC}" + """ s to load the reference
(one that has not loaded within """ + f"{_SERVICE.READY_TIMEOUT_SEC:,.0f}" + """ s is stopped),
holding up to """ + f"{REFERENCE_SERVICE_AWAKE_GIB}" + """ GiB on each GPU while it does. Until it is asleep, `/status` reports
`""" + _SERVICE.STARTING_STATE + """`, and `/score`, `/wake` and `/sleep` return HTTP 503 with `state`. Leave the
GPUs unused until `/status` reports `""" + _SERVICE.READY_STATE + """`: a service that fails to load is stopped,
and its socket then refuses connections for the rest of the build.
Awake it holds """ + f"{REFERENCE_SERVICE_AWAKE_GIB}" + """ GiB on each GPU; asleep, """ + f"{REFERENCE_SERVICE_ASLEEP_GIB}" + """ GiB.
`/score` and `/wake` wake it in """ + f"{REFERENCE_SERVICE_WAKE_SEC}" + """ s. It goes back to sleep """ + f"{_SERVICE.IDLE_SLEEP_SEC}" + """ s after its
last `/score` or `/wake`, or on `/sleep` in """ + f"{REFERENCE_SERVICE_SLEEP_SEC}" + """ s. A wake needs
`wake_need_bytes` (""" + f"{REFERENCE_SERVICE_WAKE_NEED_GIB}" + """ GiB) free on every GPU; when any GPU has less, `/score`
and `/wake` return HTTP 507 with `gpu_free_bytes` and `wake_need_bytes` and score
nothing: stop or shrink your server and retry.

Your session and the service share the memory stated for your sandbox: asleep the service
keeps a copy of the weights there, about """ + f"{REFERENCE_SERVICE_ASLEEP_HOST_GIB}" + """ GiB, and when the two together pass that
memory, the kernel ends your session's largest process. A load, a sleep or a wake that memory
is short for changes nothing and returns HTTP 507 with `memory`, `gpu` or `host`; a load short
of GPU memory also returns `gpu_free_bytes` and `gpu_need_bytes`, and `/status` carries the
last shortage as `memory_error`: release that memory and retry. A load short of memory leaves
`/status` at `""" + _SERVICE.UNLOADED_STATE + """` until `POST /wake` loads it again.{% endif %}{% endblock %}""")


class SweAgentBrief(ImplementerBrief):
    """ImplementerBrief plus this task's framing: the agentic traffic, the deep
    sessions that resume mid-conversation, the context-linear latency budget, and the
    bounded window with its cut."""

    template_str: str | None = _BRIEF

# -----------------------------------------------------------------------------
# IMAGES. Each is built from this module's declarations and the locks in ``locks/``.
#
# The name the baseline credit records its server under (``defaults/baseline.json``).
BASELINE = Baseline(image="hls-baseline-agentic-coding")

#: vLLM's own image of its build at ``VLLM_COMMIT``, by digest. No vLLM release serves this
#: checkpoint: through v0.29.0 the model registry has no ``Glm5Next`` entry, and the generic
#: Transformers backend it falls back to expects a ``k_conv1d`` weight the checkpoint ships fused.
#: This build registers ``Glm5NextForConditionalGeneration``, which the checkpoint declares.
#: The digest is the image's multi-platform index; its packages live in the system
#: interpreter. Moving it means moving ``VLLM_COMMIT`` too, which the brief names.
VLLM_IMAGE = RegistryImage(
    name="agentic-coding-vllm",
    ref="vllm/vllm-openai@sha256:4cbfd34aac145fd1870381c030131c7f868fcad45448f401ecdb5fd4ed020b42",
    python="/usr/bin/python3", snapshot=pins.SNAPSHOT)

#: Apptainer's release .deb, which Ubuntu's archive does not carry: the driver starts every
#: trajectory's sandbox with Apptainer inside its own container. It also provides the
#: ``singularity`` name the container backend calls.
APPTAINER_DEB = {
    "https://github.com/apptainer/apptainer/releases/download/v1.5.3/apptainer_1.5.3_amd64.deb":
        "82b0bdddf459087d202383360b8318d526ad6826c748a2f669913cc6aef9ee40",
}

#: The layer the agent's sandbox builds on, on the cuda base: torch 2.11 with transformers
#: 5.17.0, which holds this checkpoint's architecture (registered in 5.16.1), and the kernel
#: toolchain the brief lists (``locks/sandbox.txt``). It adds uv's CUDA 13.0 wheel set to the
#: base's variables, so a torch-family package the agent installs is that build's.
_SANDBOX_LAYER = LayerImage(name="agentic-coding-tf-5-17-0", parent="cuda", pip_lock="sandbox.txt",
                            owner="agentic_coding", env={"UV_TORCH_BACKEND": "cu130"})
#: The baseline: the vendor image and the task's server, which reports integer token ids where
#: vLLM's own OpenAI server reports token strings. The image holds everything the server
#: imports, so the layer adds the server alone.
_BASELINE = LayerImage(
    name="agentic-coding-baseline", parent=VLLM_IMAGE.name, owner="agentic_coding", server=True,
    files={"/app/server.py": "vllm_server.py",
           "/app/tasks/agentic_coding/__init__.py": "__init__.py",
           "/app/tasks/agentic_coding/serving_config.py": "serving_config.py"},
    env={"PYTHONPATH": "/app"})
#: The reference: the same vendor image, so the gate's reference engine is the stack the
#: baseline serves with, and what the driver needs to start the trajectories' sandboxes:
#: Apptainer, ``python`` (the name harness roles start under), and harbor's distribution with
#: the packages it imports (``locks/reference.txt``). The driver runs the checkout's harbor,
#: first on its path; ``harbor/__init__.py`` reads its installed distribution's version.
_REFERENCE = LayerImage(
    name="agentic-coding-reference", parent=VLLM_IMAGE.name, owner="agentic_coding", server=True,
    apt_lock="apt-reference.txt", debs=APPTAINER_DEB, pip_lock="reference.txt",
    env={"UV_HTTP_TIMEOUT": "300"})
#: The SGLang calibration member, on the cuda base: SGLang 0.5.20, the first release that
#: serves this checkpoint, from ``locks/member-sglang.txt``, with FlashInfer's kernels fetched
#: at build time, since a served member downloads nothing. The member drives its own
#: trajectories, so it holds what the reference holds for the driver: Apptainer and harbor's
#: distribution with the packages it imports.
_SGLANG = LayerImage(
    name="agentic-coding-member-sglang", parent="cuda", owner="agentic_coding", server=True,
    apt_lock="apt-member-sglang.txt", debs=APPTAINER_DEB, pip_lock="member-sglang.txt",
    fetched=FLASHINFER_CUBINS["0.6.18"],
    env={**SERVER_ENV, "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"})
# The sandbox is the agent cap on the task's own layer, and a submission is packaged on it.
# Harness roles run on the reference image: the driver needs its Apptainer, and the reference
# engine its vLLM. The baseline is served with the entrypoint and variables below.
IMAGES = ImageSet(
    layer=_SANDBOX_LAYER.name, images=[_SANDBOX_LAYER, VLLM_IMAGE],
    package_on_sandbox=True, harness_on_reference=True,
    baseline=Served(image=_BASELINE, entrypoint=["python3", "/app/server.py"],
                    runtime_env={"PYTHONPATH": "/app"}),
    reference=_REFERENCE,
    members={"sglang": _SGLANG},
)

# The toolchain paragraph of the uploads section. The shared one (tasks.sample_task.task) also
# says the uploads are the only reference on disk, which is false here: the installed
# transformers holds this architecture's implementation, which the uploads section names.
_TOOLCHAIN_PARAGRAPH = (
    "## Toolchain Documentation\n"
    "You are offline. Documentation for every installed language, compiler and\n"
    "library is on disk -- `references/README.md` indexes it, and the per-library paths\n"
    "are listed under ## Build Instructions above."
)

# The uploads section: the toolchain paragraph, then the two references that exist
# on disk for this checkpoint in particular. The transformers implementation is named
# because it is there and contestants find it anyway; an
# on-disk file the brief does not name is found by luck, which is a difference between
# contestants the score should not carry. It is named as what it is, with its measured
# distance from the reference, so a contestant can judge how far the on-disk
# implementation is from the distribution it is graded against. The figure is the
# reference bundle's HF check: transformers 5.17.0, bf16 through the FP8
# checkpoint's dequantizing load, eager attention, against the reference bundle's forced
# rows at generated positions, 6.03e-3 +/- 7.2e-4 nats per token over all 36 prompts of up
# to 8,192 tokens (the 12 longer ones skipped). Prompt positions are not stated: there it
# sits 0.62 away, mostly on system and user text, which no served token is scored on.
#
# The second paragraph documents the reference logprob set staged under the same upload
# root; the file names, the two per-position distributions and the prompt sample named
# here are the contract the staged directory must meet. Both continuation distributions
# are shipped because the grader scores the forced one and a contestant comparing against
# decode-path numbers would be chasing a difference that is not its own. The prompt
# positions are shipped because most of this workload's tokens are prefill: a server whose
# prefill kernel is wrong fails the gate on its generated tokens with nothing on disk to
# localize it.
_UPLOAD_BRIEF = (
    _TOOLCHAIN_PARAGRAPH + "\n\n"
    "The installed `transformers` package contains an implementation of this architecture\n"
    "at `transformers/models/glm5_next/modeling_glm5_next.py`. Run in bf16 on this\n"
    "checkpoint, that implementation is 6.0e-3 nats per token from the reference at\n"
    "generated positions, measured over 36 prompts of up to 8,192 tokens. It is not the\n"
    "reference your distributions are measured against; that reference is named\n"
    "under ## Decoding.\n"
    "\n"
    "## Reference Logprobs\n"
    "`references/glm53-reference/` holds reference output for self-checking.\n"
    "`prompts.jsonl` carries one record per prompt, with its token ids and its text, at\n"
    "lengths from a few thousand to several hundred thousand tokens. `reference.jsonl`\n"
    "carries, per prompt, the reference's sampled continuation token ids and, at every\n"
    "position, two distributions: `decode`, streamed by the reference while it sampled\n"
    "that continuation, and `forced`, the reference's prefill path teacher-forced on the\n"
    "same continuation, which is the path the grader uses. Each gives the served token's\n"
    f"log-probability and the {TOP_LOGPROBS_REQUESTED} most probable alternatives with "
    "their token ids and\n"
    "log-probabilities, at temperature 1. The two differ at the level of kernel noise. The\n"
    "reference itself cannot be run in your sandbox. Each prompt also carries\n"
    f"`prompt_forced`, the reference's prefill-path top-{TOP_LOGPROBS_REQUESTED} at a "
    "recorded sample of prompt\n"
    "positions -- every position in the last 1,024 tokens of the prompt and every 64th\n"
    "position before that -- so a prefill kernel can be checked against the reference\n"
    "directly. `selfcheck.py` reads a file of that shape from your own server and computes\n"
    f"the KL statistic and the top-{TOP_LOGPROBS_REQUESTED} coverage the grader computes, "
    "against `forced` by\n"
    "default, reporting prefill and generated positions separately. `README.md` states the\n"
    "formats."
)

# The package names the build seal's index refuses at `pip install`, and that the brief's
# Blocked line names: the one list every agentic build denies.
_DENIED_PACKAGES = list(DENIED_PACKAGES)

#: The tree digest of the view the two upload folders make merged
#: (``hls.agent.uploads.merged_digest``): the toolchain library the other tasks pin by
#: ``_TOOLCHAIN_REFERENCES_SHA256`` and this task's own ``references/``, which no path of the
#: library shares.
UPLOAD_SHA256 = "fbaa5b5c29508379a8a0e6d333ac80a9f112837004f7ad2a827ae78f75166e27"

ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        reasoning_effort=HighestAvailable(),
        evaluation_allocation=EVALUATION_ALLOCATION,
        build_allocation=BUILD_ALLOCATION,
        soft_budget=Budget(wallclock_sec=int(12 * 3600), turns=None),
        hard_budget=Budget(wallclock_sec=int(12.5 * 3600), turns=None),
        # The toolchain documentation the brief's Build Instructions point at, uploaded
        # into the sandbox rather than baked (see tasks.sample_task.task), and the task's own
        # references/, which holds the reference output the brief's Reference Logprobs
        # section names, merged with it at references/.
        upload_dir=("../assets/references", "references"), upload_sha256=UPLOAD_SHA256,
        upload_dest="/workspace/references",
        upload_brief=_UPLOAD_BRIEF,
        # Air-gapped, with the package index reachable through the proxy for everything
        # but the serving frameworks the brief names as blocked. The brief renders its
        # Blocked line from this list, so what it states is what the proxy refuses.
        network=NetworkPolicy(
            domains=AccessList(mode="allow"),
            packages=AccessList(mode="deny", entries=_DENIED_PACKAGES)),
        # The grader's pinned vLLM reference, asleep beside the sandbox for scoring.
        reference_service=True,
    ),
    brief=SweAgentBrief())

TASK = TaskSpec(
    id="agentic_coding",
    model=MODEL,
    hardware=HARDWARE,
    workload=WORKLOAD,
    data_source=DATA_SOURCE,
    contract=CONTRACT,
    verifiers=VERIFIERS,
    scorer=SCORER,
    agent=AGENT,
    baseline=BASELINE,
    drive_bounds=DRIVE_BOUNDS,
    seed=SEED,
)

__all__ = ["AGENT", "ARRIVAL_WINDOW_SEC", "ARTIFACT",
           "BASELINE", "CONTRACT",
           "BUILD_ALLOCATION", "DATA_SOURCE", "DATASET", "DECODE", "DRIVE_BOUNDS", "EVALUATION_ALLOCATION",
           "DEV_INSTANCES", "FUSE_DEVICE",
           "HARDWARE", "IMAGE_STORE", "MAX_TURNS",
           "MAX_CONTEXT_LEN", "MODEL", "MODEL_ID",
           "OVERLAY_STORE", "POISSON_RATE_PER_SEC", "PREFIX_SOURCE_DIR", "PREFIX_STORE", "PROFILE_PENDING",
           "RUNTIME_STORE", "SANDBOX_IMAGES", "SANDBOX_IMAGE_DIR", "SANDBOX_OVERLAY_DIR",
           "SANDBOX_PREFIX_DIR", "SANDBOX_RUNTIME_DIR", "SCORER",
           "REPLY_BUDGET", "SMOKE_DATA_SOURCE", "THINK_CAP", "TASK", "TEST_INSTANCES", "TEST_SESSIONS",
           "TOOL_RUNNER", "UPLOAD_SHA256",
           "TTFT_BUDGET_MS", "TTFT_MS_PER_PREFILL_TOKEN", "TPOT_BUDGET_MS",
           "USABLE_TASKS",
           "VERIFIERS", "WORKLOAD", "sandbox_image",
           "sandbox_allow_internet", "sandbox_lower_overlays", "sandbox_place_prefixes", "sandbox_row_entries",
           "sandbox_cpus", "sandbox_memory_mb",
           "sandbox_workdir"]
