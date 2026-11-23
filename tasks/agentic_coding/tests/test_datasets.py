"""The corpus: what it reads off a checkout, what it drops, and the shape of the seed a
trajectory starts from."""
from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from hls.spec.datasets import Split
from tasks.agentic_coding.datasets import (
    CATEGORY_FIELD, CPUS_FIELD, DEEP_DRAWS, DEEP_TURN_BUDGET, DEFAULT_WORKDIR,
    GPUS_FIELD, INSTRUCTION_FIELD, MAX_SANDBOX_CPUS, MAX_SANDBOX_MEMORY_MB,
    MAX_SANDBOX_STORAGE_MB, MEMORY_MB_FIELD, NEEDS_ACCELERATOR, NO_IMAGE,
    PREFIX_DIGEST_FIELD, RELEASE_COMMIT, RELEASE_TASK_COUNT, SANDBOX_CPUS,
    SANDBOX_MEMORY_MB, SANDBOX_STORAGE_MB, SOURCE_FIELD, STORAGE_MB_FIELD, TASK_NAME_FIELD,
    TERMINAL_BENCH_SOURCE, TURN_BUDGET_FIELD, WORKDIR_FIELD,
    TerminalBenchDataset, TerminalBenchError, dockerfile_workdir,
    release_digest,
)

#: A name that is neither excluded nor special, used wherever a test needs one task.
A_TASK = "task-00"

#: Three more of the same, used wherever a test needs rows that resume a pre-rolled
#: conversation. Ordinary tasks: what makes a row deep is being named in ``deep_rows``,
#: not anything the checkout says about it.
DEEP_TASKS = ("task-01", "task-02", "task-03")

#: A prefix digest's shape. The content is never read here -- these tests are about
#: which rows carry a digest, not about what staging does with one.
A_DIGEST = "b" * 64


def write_task(root, name, *, instruction="Do the thing.", workdir="/app",
               gpus=0, environment=None):
    """One task directory shaped the way the release ships them.

    ``environment`` is the block the release states its container's needs in; pass ""
    for a definition that has none, which is what the corpus falls back on. ``gpus``
    is broken out because the corpus checks its exclusion list against it.

    Overwrites an existing one, so a test can restate a task the fixture already made
    without changing the corpus size."""
    if environment is None:
        environment = f"cpus = 2\nmemory_mb = 4096\nstorage_mb = 10240\ngpus = {gpus}\n"
    task = root / name
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        '[task]\nname = "terminal-bench/%s"\n'
        '[metadata]\ncategory = "Science"\nexpert_time_estimate_hours = 4.0\n' % name
        + (f"[environment]\n{environment}" if environment else ""))
    (task / "instruction.md").write_text(instruction)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM python:3.11\nWORKDIR {workdir}\n")
    return task


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A full-size checkout: the release's task count, including the excluded ones."""
    root = tmp_path / "tasks"
    root.mkdir()
    # Both excluded sets are written out, because the release ships them and the count
    # is asserted against the release. A fixture holding only what the corpus admits
    # would agree with a dataset that had forgotten to exclude anything.
    excluded = sorted(NO_IMAGE) + sorted(NEEDS_ACCELERATOR)
    # The rows the shipped pin resumes are written under their real names, because the
    # split tests partition the shipped dataset and a pinned row the checkout lacks
    # would make the pinned count disagree with the corpus.
    pinned = list(TerminalBenchDataset().deep_rows)
    names = excluded + pinned + [
        f"task-{i:02d}"
        for i in range(RELEASE_TASK_COUNT - len(excluded) - len(pinned))]
    for name in names:
        # The accelerator set is checked against what the checkout declares, so the
        # fixture has to declare it: a release where those tasks asked for no GPU
        # would be a different release, and the corpus says so.
        write_task(root, name, gpus=1 if name in NEEDS_ACCELERATOR else 0)
    from tasks.agentic_coding import datasets

    monkeypatch.setattr(datasets, "tasks_dir", lambda: root)
    # These tasks are invented, so they cannot carry the real release's digest. The
    # pin is repointed at what this checkout actually holds, which leaves the check
    # live -- a test that alters a task after this still trips it -- while not
    # requiring the fixture to reproduce 66 real task definitions.
    pin_digest(monkeypatch, root)
    return root


def pin_digest(monkeypatch, root):
    """Point the corpus digest at what ``root`` currently holds.

    Two places, because the task module builds its dataset once at import and that
    instance is older than any monkeypatch: the constant new instances read, and the
    shipped singleton the split tests partition."""
    import sys

    from tasks.agentic_coding import datasets

    digest = release_digest(root)
    monkeypatch.setattr(datasets, "RELEASE_DIGEST", digest)
    task_module = sys.modules.get("tasks.agentic_coding.task")
    if task_module is not None:
        monkeypatch.setattr(
            task_module, "DATASET",
            task_module.DATASET.model_copy(update={"corpus_digest": digest}))


# --- where the corpus is read from -----------------------------------------

def _release_repo(tmp_path):
    """A repository standing in for the release: one commit holding ``tasks/one/task.toml``,
    which a fetch may ask for by its id. Returns its path and the commit."""
    import subprocess

    repo = tmp_path / "release"
    (repo / "tasks" / "one").mkdir(parents=True)
    (repo / "tasks" / "one" / "task.toml").write_text("[metadata]\n")
    for argv in (["init", "-q"], ["config", "uploadpack.allowAnySHA1InWant", "true"], ["add", "."],
                 ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "release"]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    return repo, commit


def test_the_release_is_fetched_at_its_pinned_commit(tmp_path):
    from tasks.agentic_coding.datasets import fetch_release

    repo, commit = _release_repo(tmp_path)
    root = tmp_path / "checkout"
    fetch_release(root, url=str(repo), commit=commit)
    assert (root / "tasks" / "one" / "task.toml").read_text() == "[metadata]\n"


def test_a_release_that_cannot_be_fetched_is_refused_naming_the_commit(tmp_path):
    from tasks.agentic_coding.datasets import fetch_release

    repo, _ = _release_repo(tmp_path)
    with pytest.raises(TerminalBenchError, match=f"fetching the release at {'0' * 40} from {repo} failed"):
        fetch_release(tmp_path / "checkout", url=str(repo), commit="0" * 40)


def test_the_corpus_is_read_from_the_fetched_release(monkeypatch, tmp_path):
    from tasks.agentic_coding import datasets

    monkeypatch.setattr(datasets, "tasks_dir", lambda: tmp_path / "absent")
    with pytest.raises(TerminalBenchError, match="has no tasks/"):
        TerminalBenchDataset().corpus()


# --- the pin the checkout is checked against -------------------------------

def test_the_identity_is_the_commit_rather_than_the_tag():
    # A tag is a moving pointer. The commit is what a checkout can be compared
    # against, so that is what the corpus is identified by.
    identity = TerminalBenchDataset().corpus_id()
    assert RELEASE_COMMIT in identity
    assert len(RELEASE_COMMIT) == 40


def test_an_altered_task_is_refused(checkout):
    # The point of the digest: a checkout at another commit, or one a local edit had
    # touched, would otherwise provision and grade without a word.
    (checkout / A_TASK / "instruction.md").write_text("Do something else entirely.")
    with pytest.raises(TerminalBenchError, match="does not match"):
        TerminalBenchDataset().corpus()


def test_the_task_named_in_the_refusal_is_the_one_that_changed(checkout):
    # A digest over the whole release says only that something moved. Triage needs
    # the name, so the refusal recomputes per task and reports which.
    (checkout / A_TASK / "instruction.md").write_text("Changed.")
    with pytest.raises(TerminalBenchError, match=A_TASK):
        TerminalBenchDataset().corpus()


def test_an_unpinned_digest_refuses_rather_than_skipping(checkout):
    # An empty pin must not read as "nothing to check". That is the absence-as-a-value
    # shape: a corpus provisioned with the check silently off looks exactly like one
    # that passed it.
    with pytest.raises(TerminalBenchError, match="no corpus_digest"):
        TerminalBenchDataset(corpus_digest="").corpus()


def test_the_digest_covers_what_the_corpus_reads(checkout):
    # Stated as a test because the limit matters: a file the digest does not cover is
    # a file two checkouts can disagree about silently. These three are what a row is
    # built from and what says which image the task wants.
    from tasks.agentic_coding.datasets import DIGESTED_FILES, task_digest

    assert DIGESTED_FILES == ("task.toml", "instruction.md",
                              "environment/Dockerfile")
    before = task_digest(checkout / A_TASK)
    (checkout / A_TASK / "environment" / "Dockerfile").write_text(
        "FROM other\nWORKDIR /elsewhere\n")
    assert task_digest(checkout / A_TASK) != before


def test_a_file_the_digest_does_not_cover_does_not_move_it(checkout):
    # The control, and the honest limit. Image build data is not hashed -- the tree is
    # half a gigabyte and the staged image is built elsewhere anyway -- so a change
    # there passes. Anyone relying on this check should know that.
    from tasks.agentic_coding.datasets import task_digest

    before = task_digest(checkout / A_TASK)
    (checkout / A_TASK / "environment" / "payload.bin").write_bytes(b"new data")
    assert task_digest(checkout / A_TASK) == before


# --- what the corpus carries -----------------------------------------------

def test_the_corpus_is_the_release_less_the_tasks_with_no_image(checkout):
    corpus = TerminalBenchDataset().corpus()
    assert len(corpus) == RELEASE_TASK_COUNT - len(NO_IMAGE) - len(NEEDS_ACCELERATOR)
    names = {row[TASK_NAME_FIELD] for row in corpus}
    # Dropped here rather than left to fail at launch, so every task that reaches a
    # split has an image and a failed launch is a fault rather than a known gap.
    assert names.isdisjoint(NO_IMAGE)


def test_a_task_that_needs_an_accelerator_is_not_in_the_corpus(checkout):
    # Dropped for a different reason than NO_IMAGE, and kept a different set so each
    # reason stays legible: these tasks build fine, they just cannot do the work they
    # were set. The node's GPUs belong to the server under measurement.
    corpus = TerminalBenchDataset().corpus()
    names = {row[TASK_NAME_FIELD] for row in corpus}
    assert names.isdisjoint(NEEDS_ACCELERATOR)
    assert NEEDS_ACCELERATOR, "an empty set means the census was never run"
    # And the two reasons do not overlap, or the count below would double-subtract.
    assert set(NEEDS_ACCELERATOR).isdisjoint(NO_IMAGE)


def test_a_task_that_newly_declares_a_gpu_is_refused(checkout, monkeypatch):
    # The list is a pin, not a survey, so it cannot be allowed to go quietly stale: a
    # release that gave an existing task a GPU would otherwise keep grading it, and
    # the trajectory would be an agent failing work it cannot do.
    write_task(checkout, A_TASK, gpus=1)
    # Repointed so the digest check passes and the accelerator check is what trips;
    # otherwise this test would pass for the wrong reason.
    pin_digest(monkeypatch, checkout)
    with pytest.raises(TerminalBenchError, match="declare a GPU"):
        TerminalBenchDataset().corpus()


def test_a_listed_task_that_declares_no_gpu_is_refused(checkout, monkeypatch):
    # The other direction, and the reason it is worth its own test: a release that
    # dropped the GPU requirement would leave a task excluded for a reason that had
    # stopped being true, and nothing would say so.
    write_task(checkout, sorted(NEEDS_ACCELERATOR)[0], gpus=0)
    pin_digest(monkeypatch, checkout)
    with pytest.raises(TerminalBenchError, match="no longer declare"):
        TerminalBenchDataset().corpus()


def test_the_shipped_list_is_what_the_checkout_declares(checkout):
    # The control for both. With the fixture agreeing with the pin, the corpus builds.
    assert len(TerminalBenchDataset().corpus()) == (
        RELEASE_TASK_COUNT - len(NO_IMAGE) - len(NEEDS_ACCELERATOR))


def test_the_usable_count_subtracts_both_exclusions(checkout):
    dataset = TerminalBenchDataset()
    assert dataset.usable_tasks() == (
        RELEASE_TASK_COUNT - len(NO_IMAGE) - len(NEEDS_ACCELERATOR))
    assert len(dataset.corpus()) == dataset.usable_tasks()


def test_the_accelerator_set_is_part_of_the_corpus_identity():
    # Re-including one, should a sandbox ever get an accelerator, changes the corpus
    # and has to re-provision rather than be served from a cache keyed on the tag.
    identity = TerminalBenchDataset().corpus_id()
    assert all(name in identity for name in NEEDS_ACCELERATOR)


def test_a_short_checkout_is_refused(checkout):
    # A truncated clone, or one at another tag, would otherwise provision a corpus
    # that still partitions and still runs.
    next(checkout.iterdir()).rename(checkout / "moved-aside")
    (checkout / "moved-aside" / "task.toml").unlink()
    with pytest.raises(TerminalBenchError, match="ships"):
        TerminalBenchDataset().corpus()


def test_a_row_carries_what_the_sandbox_needs_to_route(checkout, monkeypatch):
    write_task(checkout, A_TASK, instruction="Catalogue the variants.",
               workdir="/workspace")
    pin_digest(monkeypatch, checkout)
    row = next(r for r in TerminalBenchDataset().corpus()
               if r[TASK_NAME_FIELD] == A_TASK)
    assert row[INSTRUCTION_FIELD] == "Catalogue the variants."
    assert row[WORKDIR_FIELD] == "/workspace"
    assert row[SOURCE_FIELD] == TERMINAL_BENCH_SOURCE


def test_a_row_carries_what_its_own_definition_asks_for(checkout, monkeypatch):
    # Read off the task rather than fixed for the run. These declarations span an
    # eightfold range, so one figure for all of them either bounds how many
    # trajectories run at once or starves the tasks that asked for more.
    write_task(checkout, A_TASK,
               environment="cpus = 8\nmemory_mb = 32768\nstorage_mb = 20480\ngpus = 0\n")
    pin_digest(monkeypatch, checkout)
    row = next(r for r in TerminalBenchDataset().corpus()
               if r[TASK_NAME_FIELD] == A_TASK)
    assert row[CPUS_FIELD] == 8
    assert row[MEMORY_MB_FIELD] == 32768
    assert row[STORAGE_MB_FIELD] == 20480
    assert row[GPUS_FIELD] == 0


def test_a_row_records_the_gpus_its_task_asked_for():
    # Read off the definition even though no admitted row can carry one, because this
    # is the field the exclusion list is checked against. Tested on the row builder
    # rather than through the corpus, since a task declaring a GPU never reaches it.
    from tasks.agentic_coding.datasets import sandbox_fields

    caps = TerminalBenchDataset().sandbox_caps()
    assert sandbox_fields({"gpus": 2}, caps)[GPUS_FIELD] == 2
    assert sandbox_fields({}, caps)[GPUS_FIELD] == 0


def test_a_definition_that_declares_nothing_falls_back(checkout, monkeypatch):
    # A release re-cut without the block, or a task that never had one, must still
    # launch rather than asking the backend for a container with no CPUs.
    write_task(checkout, A_TASK, environment="")
    pin_digest(monkeypatch, checkout)
    row = next(r for r in TerminalBenchDataset().corpus()
               if r[TASK_NAME_FIELD] == A_TASK)
    assert row[CPUS_FIELD] == SANDBOX_CPUS
    assert row[MEMORY_MB_FIELD] == SANDBOX_MEMORY_MB
    assert row[STORAGE_MB_FIELD] == SANDBOX_STORAGE_MB


def test_a_declaration_above_the_cap_is_capped(checkout, monkeypatch):
    # The caps bind on real rows rather than guarding against nothing: one admitted
    # task asks for a terabyte of disk, which a shared node cannot give it. Capped
    # rather than refused, because a task the corpus admits should run on what the
    # node can spare, and the comment in task.py names the ones that end up short.
    write_task(checkout, A_TASK,
               environment=f"cpus = {MAX_SANDBOX_CPUS * 2}\n"
                           f"memory_mb = {MAX_SANDBOX_MEMORY_MB * 2}\n"
                           f"storage_mb = {MAX_SANDBOX_STORAGE_MB * 2}\n")
    pin_digest(monkeypatch, checkout)
    row = next(r for r in TerminalBenchDataset().corpus()
               if r[TASK_NAME_FIELD] == A_TASK)
    assert row[CPUS_FIELD] == MAX_SANDBOX_CPUS
    assert row[MEMORY_MB_FIELD] == MAX_SANDBOX_MEMORY_MB
    assert row[STORAGE_MB_FIELD] == MAX_SANDBOX_STORAGE_MB


def test_a_declaration_at_or_under_the_cap_is_untouched(checkout, monkeypatch):
    # The control. A cap that quietly rewrote every row would pass the test above.
    write_task(checkout, A_TASK,
               environment=f"cpus = {MAX_SANDBOX_CPUS}\nmemory_mb = 4096\n"
                           f"storage_mb = 10240\n")
    pin_digest(monkeypatch, checkout)
    row = next(r for r in TerminalBenchDataset().corpus()
               if r[TASK_NAME_FIELD] == A_TASK)
    assert (row[CPUS_FIELD], row[MEMORY_MB_FIELD], row[STORAGE_MB_FIELD]) == (
        MAX_SANDBOX_CPUS, 4096, 10240)


def test_the_cap_is_declared_so_it_reaches_the_fingerprint():
    # Moving the cap changes what a starved task is given, which changes the
    # observations its commands return, and nothing fails when it does. So it is a
    # declared field rather than a module constant, since only declared fields are
    # serialized into the workload fingerprint.
    from hls.core.poly import dump_spec

    spec = dump_spec(TerminalBenchDataset())
    assert "max_sandbox_cpus" in spec
    assert "max_sandbox_memory_mb" in spec
    assert "max_sandbox_storage_mb" in spec
    moved = TerminalBenchDataset(max_sandbox_cpus=8)
    assert moved.corpus_id() != TerminalBenchDataset().corpus_id()


def test_the_row_id_is_the_task_name(checkout):
    dataset = TerminalBenchDataset()
    row = next(r for r in dataset.corpus() if r[TASK_NAME_FIELD] == A_TASK)
    assert dataset.row_id(row) == A_TASK


# --- the working directory read off the image ------------------------------

def test_the_last_workdir_wins(tmp_path):
    # Several release Dockerfiles set one directory to build in and another to run in;
    # only the last is where the agent's shell lands.
    path = tmp_path / "Dockerfile"
    path.write_text("FROM x\nWORKDIR /build\nRUN make\nWORKDIR /app/vllm\n")
    assert dockerfile_workdir(path) == "/app/vllm"


def test_a_dockerfile_with_no_workdir_falls_back(tmp_path):
    path = tmp_path / "Dockerfile"
    path.write_text("FROM x\nRUN make\n")
    assert dockerfile_workdir(path) == DEFAULT_WORKDIR


# --- the seed a trajectory starts from -------------------------------------

def test_a_seed_is_the_instruction_keyed_by_task_name(tmp_path):
    # The framework reads the task name and working directory off a seed's metadata to
    # pick the session's image.
    split_dir = tmp_path / Split.DEV.value
    split_dir.mkdir()
    (split_dir / "rows.json").write_text(
        '[{"task_name": "wdm-design", "instruction": "Design the filter.",'
        ' "workdir": "/task", "category": "Engineering",'
        ' "expert_time_estimate_hours": 6.0, "source": "terminal-bench"}]')
    item = next(TerminalBenchDataset().items_at(tmp_path, Split.DEV))

    assert item.request_id == "wdm-design"
    seed = item.turns[0]
    assert seed.first_of("text").text == "Design the filter."
    assert seed.metadata[TASK_NAME_FIELD] == "wdm-design"
    assert seed.metadata[WORKDIR_FIELD] == "/task"
    assert seed.metadata[SOURCE_FIELD] == TERMINAL_BENCH_SOURCE
    assert seed.metadata[CATEGORY_FIELD] == "Engineering"
    # The framework renders the conversation, so a seed must not arrive pre-rendered.
    assert "messages" not in seed.metadata
    # Upstream's answer is a script that only runs inside the task's own container, so
    # there is no ground truth a trace could carry.
    assert item.ground_truth_turns == ()


def test_an_unmounted_split_says_which_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="test split not mounted"):
        list(TerminalBenchDataset().items_at(tmp_path, Split.TEST))


# ===========================================================================
# The pins: what moves the identity, and what a pin is refused for
# ===========================================================================

def test_the_excluded_set_is_part_of_the_corpus_identity():
    # Re-including a task once its image builds changes the corpus, and that has to
    # re-provision rather than be served from a cache keyed on the release alone.
    identity = TerminalBenchDataset().corpus_id()
    assert all(name in identity for name in NO_IMAGE)


#: One changed pin per case, each a thing that changes what a trajectory runs against:
#: a shorter release, a dropped task, a different set of rows resuming, a different
#: conversation to resume from, a different number of turns to serve.
MOVED_PINS = {
    "release": TerminalBenchDataset(release="v3.0.0"),
    "exclusions": TerminalBenchDataset(no_image=("data-anonymization",)),
    "deep rows": TerminalBenchDataset(deep_rows=DEEP_TASKS, prefixes={}),
    "prefix": TerminalBenchDataset(deep_rows=DEEP_TASKS,
                                   prefixes={DEEP_TASKS[0]: A_DIGEST}),
    "turn budget": TerminalBenchDataset(turn_budgets={A_TASK: 5}),
}


@pytest.mark.parametrize("moved", MOVED_PINS.values(), ids=list(MOVED_PINS))
def test_the_identity_moves_when_a_pin_moves(moved):
    # A calibration is a measurement of one corpus; applying it to a run where the
    # corpus changed would be reading a number taken over different traffic.
    assert moved.corpus_id() != TerminalBenchDataset().corpus_id()


@pytest.mark.parametrize("moved", MOVED_PINS.values(), ids=list(MOVED_PINS))
def test_the_workload_fingerprint_moves_when_a_pin_moves(moved):
    # corpus_id is not enough on its own. It content-addresses provisioning, while a
    # calibration is gated on workload_fingerprint, which is a hash of the dataset's
    # *declared fields* -- so a pin kept in a module constant would re-provision and
    # still let a stale calibration read as valid. This is the test that the pins are
    # declared and not merely written down.
    from hls.verify.calibration import workload_fingerprint

    from tasks.agentic_coding.task import TASK

    moved_task = TASK.model_copy(
        update={"data_source": TASK.data_source.model_copy(
            update={"dataset": moved})})
    assert workload_fingerprint(moved_task) != workload_fingerprint(TASK)


def test_reordering_an_exclusion_list_does_not_move_the_fingerprint():
    # The fingerprint is canonical JSON, which sorts keys but not sequences, so an
    # unordered set written two ways would otherwise hash two ways and refuse a
    # calibration taken under the other spelling.
    shuffled = TerminalBenchDataset(no_image=tuple(reversed(sorted(NO_IMAGE))))
    assert shuffled.no_image == TerminalBenchDataset().no_image
    assert shuffled.corpus_id() == TerminalBenchDataset().corpus_id()


def test_reordering_the_deep_rows_does_not_move_the_fingerprint():
    # Same hazard as the exclusion list, and worth its own test because this one is a
    # hand-written list: whoever fills it in writes the names in whatever order the
    # pre-roll reported them.
    shuffled = TerminalBenchDataset(deep_rows=tuple(reversed(DEEP_TASKS)), prefixes={})
    assert shuffled.deep_rows == TerminalBenchDataset(deep_rows=DEEP_TASKS,
                                                      prefixes={}).deep_rows


def test_a_deep_row_can_be_given_a_prefix():
    dataset = TerminalBenchDataset(deep_rows=DEEP_TASKS,
                                   prefixes={DEEP_TASKS[0]: A_DIGEST})
    assert dataset.prefixes == {DEEP_TASKS[0]: A_DIGEST}


def test_a_prefix_for_a_row_that_does_not_resume_is_refused():
    # A typo, or a name dropped from deep_rows and left behind here, would otherwise
    # be applied anyway: every corpus row is offered the prefix map by name, so the
    # named row would resume a conversation while hashing like an ordinary row and
    # drawing one session. Silent, and the fingerprint would have moved.
    with pytest.raises(ValidationError, match="does not resume"):
        TerminalBenchDataset(prefixes={A_TASK: A_DIGEST})


def test_a_deep_row_with_no_image_is_refused():
    # A subtler cause, same failure: the row is named deep, but the corpus drops it
    # for having no image, so nothing ever loads its prefix. It also breaks the split
    # arithmetic, which sizes the hashed test count by subtracting the pinned rows --
    # a pinned row that never reaches the corpus makes that subtraction too large.
    dropped = sorted(NO_IMAGE)[0]
    with pytest.raises(ValidationError, match="no image"):
        TerminalBenchDataset(deep_rows=(dropped,), prefixes={})


# ===========================================================================
# A row that supplies several sessions
# ===========================================================================

def _rows_at(tmp_path, rows: str, split: Split = Split.TEST):
    """Mount ``rows`` as a split and return the seeds the corpus renders from it."""
    split_dir = tmp_path / split.value
    split_dir.mkdir(parents=True)
    (split_dir / "rows.json").write_text(rows)
    return list(TerminalBenchDataset().items_at(tmp_path, split))


ONE_ROW = ('[{{"task_name": "wdm-design", "instruction": "Design it.",'
           ' "workdir": "/app", "category": "Engineering",'
           ' "expert_time_estimate_hours": 0.0, "source": "terminal-bench"{extra}}}]')


def test_a_row_is_one_session_unless_it_says_otherwise(tmp_path):
    # Every row that is not deep.
    assert len(_rows_at(tmp_path, ONE_ROW.format(extra=""))) == 1


def test_a_row_that_declares_draws_supplies_that_many_sessions(tmp_path):
    assert len(_rows_at(tmp_path, ONE_ROW.format(extra=', "draws": 3'))) == 3


def test_each_draw_is_its_own_session(tmp_path):
    # Distinct request ids, because the framework takes one as the session id and
    # builds every turn id under it; two draws sharing one would interleave in the
    # trace and share a sandbox.
    items = _rows_at(tmp_path, ONE_ROW.format(extra=', "draws": 3'))
    assert [i.request_id for i in items] == [
        "wdm-design", "wdm-design#1", "wdm-design#2"]


def test_a_draws_id_cannot_be_read_as_a_turn_id(tmp_path):
    # A turn is `<session>.<k>`, and the partition refuses a row id ending that way
    # so the two can be told apart. `#` keeps that true for a draw suffix.
    items = _rows_at(tmp_path, ONE_ROW.format(extra=', "draws": 2'))
    assert not any(re.search(r"\.\d+$", i.request_id) for i in items)


def test_every_draw_carries_the_same_seed(tmp_path):
    # They resume one conversation, so they are repeat samples of one trajectory and
    # the only thing that differs is the id.
    items = _rows_at(tmp_path, ONE_ROW.format(extra=', "draws": 2'))
    first, second = (i.turns[0] for i in items)
    assert first.metadata[TASK_NAME_FIELD] == second.metadata[TASK_NAME_FIELD]
    assert first.first_of("text").text == second.first_of("text").text


def test_a_nonsense_draw_count_is_one_session(tmp_path):
    # A row without the field, or one carrying junk, must not render
    # zero sessions and quietly shrink the drive.
    for n, junk in enumerate(('"draws": 0', '"draws": -2', '"draws": "two"',
                              '"draws": null')):
        got = _rows_at(tmp_path / str(n), ONE_ROW.format(extra=", " + junk))
        assert len(got) == 1, junk


def test_a_deep_row_declares_its_draws_and_its_split(checkout):
    # What puts a deep session in a graded run at all: rows the hash does not promise
    # to place in test, each drawn more than once. Read off the corpus rather than off
    # a hand-written row, so what is under test is what the corpus actually emits.
    dataset = TerminalBenchDataset(deep_rows=DEEP_TASKS,
                                   prefixes={DEEP_TASKS[0]: A_DIGEST})
    row = next(r for r in dataset.corpus() if r[TASK_NAME_FIELD] == DEEP_TASKS[0])
    assert dataset.forced_split(row) == "test"
    assert dataset.draws(row) == DEEP_DRAWS
    assert row[PREFIX_DIGEST_FIELD] == A_DIGEST


def test_an_ordinary_row_declares_neither(checkout):
    dataset = TerminalBenchDataset(deep_rows=DEEP_TASKS, prefixes={})
    row = next(r for r in dataset.corpus() if r[TASK_NAME_FIELD] == A_TASK)
    assert dataset.forced_split(row) is None
    assert dataset.draws(row) == 1
    assert row[PREFIX_DIGEST_FIELD] == ""
    # And it runs to the framework's own ceiling, because its trajectory ends when the
    # work is done rather than when a budget cuts it off.
    assert row[TURN_BUDGET_FIELD] == 0


def test_a_deep_row_gets_the_resumed_turn_budget_without_being_given_one(checkout):
    # Derived from deep_rows rather than declared beside it, for the same reason the
    # draw count is: a second map is a second place to forget. Forgetting this one is
    # expensive -- a resumed session inherits the framework's 250-turn ceiling and
    # serves every one of those turns at 600k of prompt.
    dataset = TerminalBenchDataset(deep_rows=DEEP_TASKS, prefixes={})
    row = next(r for r in dataset.corpus() if r[TASK_NAME_FIELD] == DEEP_TASKS[0])
    assert row[TURN_BUDGET_FIELD] == DEEP_TURN_BUDGET


def test_a_declared_turn_budget_overrides_the_derived_one(checkout):
    # The field stays a field: a pre-roll that cut a shallower band than planned wants
    # to say so for that row without moving every other row's budget.
    dataset = TerminalBenchDataset(deep_rows=DEEP_TASKS, prefixes={},
                                   turn_budgets={DEEP_TASKS[0]: 4})
    rows = {r[TASK_NAME_FIELD]: r for r in dataset.corpus()}
    assert rows[DEEP_TASKS[0]][TURN_BUDGET_FIELD] == 4
    assert rows[DEEP_TASKS[1]][TURN_BUDGET_FIELD] == DEEP_TURN_BUDGET


# ===========================================================================
# The splits the task declares over the corpus
# ===========================================================================
#
# The partition is the step between a corpus and a run, and every test above stops at
# the corpus. A split sized against the corpus rather than against the pool the hash
# draws from passes those and fails to provision.

@pytest.fixture(scope="module")
def task():
    """The task module. Imported lazily: it reads no checkout at import, but it does
    pull in the harness, which the corpus tests above do not need."""
    import tasks.agentic_coding.task as m
    return m


def _partition(dataset, split):
    """The dev and test rows one split spec carves out of a corpus.

    Drives provisioning's own helpers rather than restating them, so what is under
    test is the path a run takes."""
    from hls.provision.rows import pinned_test, row_ids
    from hls.spec.datasource import partition_by_hash

    corpus = dataset.corpus()
    dev_idx, test_idx = partition_by_hash(
        row_ids(dataset, corpus), dev=split.dev, test=split.test,
        pinned_test=pinned_test(dataset, corpus))
    return corpus.select(dev_idx).to_list(), corpus.select(test_idx).to_list()


def test_the_shipped_split_partitions_the_corpus(task, checkout):
    # The test count is drawn from the corpus less the pinned rows,
    # so sizing it as "everything not in dev" asks the hash for rows the pool does not
    # have and the partition refuses the source outright -- a run that cannot start.
    dev, test = _partition(task.DATASET, task.DATA_SOURCE.split)
    assert len(dev) == task.DEV_INSTANCES
    assert len(test) == task.TEST_INSTANCES


def test_the_shipped_split_supplies_the_declared_sessions(task, checkout, tmp_path):
    # Rows are not sessions once a row is drawn more than once. What the drive offers
    # is the seeds, so count those rather than the rows.
    import json

    _, test = _partition(task.DATASET, task.DATA_SOURCE.split)
    (tmp_path / Split.TEST.value).mkdir()
    (tmp_path / Split.TEST.value / "rows.json").write_text(json.dumps(test))
    items = list(task.DATASET.items_at(tmp_path, Split.TEST))
    # The declared count and the seeds the split actually expands to agree. Counting
    # the rows would not catch a redraw that was declared and never rendered.
    assert len(items) == task.TEST_SESSIONS
    ids = [i.request_id for i in items]
    assert len(set(ids)) == len(ids)


def test_the_smoke_split_partitions_too(task, checkout):
    # It carries the pinned rows as well, so its test side is larger than it asks for.
    dev, test = _partition(task.DATASET, task.SMOKE_DATA_SOURCE.split)
    assert len(dev) == 2
    assert len(test) == 2 + task.PINNED_TEST_INSTANCES


def test_the_pinned_count_is_what_the_rows_actually_declare(task, checkout):
    # The split arithmetic is only right if this number is. Counted off the rows here
    # rather than trusted, so a corpus that changed which rows it pins without
    # changing the count fails here instead of failing to provision.
    from hls.provision.rows import pinned_test

    corpus = task.DATASET.corpus()
    assert task.DATASET.pinned_test_rows() == len(pinned_test(task.DATASET, corpus))


# --- the same arithmetic with the deep rows filled in -----------------------
#
# The tests above cover the shipped pin. These rebuild the task's own arithmetic over
# a corpus that names invented deep rows, so the pinning, the redraws and the split
# sizes are shown to hold for a pin other than the shipped one.

@pytest.fixture
def deep_dataset(checkout):
    """A dataset with the pin filled in.

    Built per test rather than at import, because the corpus digest it carries has to
    be the one the `checkout` fixture pointed at this checkout."""
    return TerminalBenchDataset(deep_rows=DEEP_TASKS, prefixes={})


@pytest.fixture
def deep_split(task, deep_dataset):
    """``task.py``'s split arithmetic, restated over ``deep_dataset``.

    Restated rather than imported because the shipped constants are sized for the
    shipped pin; what is under test is that the arithmetic holds when the pin is not
    empty, for the same session target."""
    from hls.spec.datasource import Count, SplitSpec

    extra = sum(n - 1 for n in deep_dataset.session_draws().values())
    test_rows = task.TARGET_TEST_SESSIONS - extra
    hashed = test_rows - deep_dataset.pinned_test_rows()
    dev_rows = deep_dataset.usable_tasks() - test_rows
    return SplitSpec(dev=Count(value=dev_rows), test=Count(value=hashed)), test_rows


def test_a_filled_pin_still_reaches_the_session_target(task, checkout, tmp_path,
                                                       deep_split, deep_dataset):
    import json
    from collections import Counter

    split, test_rows = deep_split
    dev, test = _partition(deep_dataset, split)
    assert len(test) == test_rows

    (tmp_path / Split.TEST.value).mkdir()
    (tmp_path / Split.TEST.value / "rows.json").write_text(json.dumps(test))
    items = list(deep_dataset.items_at(tmp_path, Split.TEST))
    # Fewer rows than sessions: the point of the redraw is that the two counts come
    # apart.
    assert len(items) == task.TARGET_TEST_SESSIONS
    assert len(items) > len(test)
    # And the deep rows really are redrawn, which is why they differ. Asserted against
    # the rows rather than against the declaration, so a corpus that stopped redrawing
    # them would not quietly agree with itself.
    per_task = Counter(i.turns[0].metadata[TASK_NAME_FIELD] for i in items)
    assert all(per_task[name] == DEEP_DRAWS for name in DEEP_TASKS), per_task


def test_the_deep_rows_all_reach_the_test_split(checkout, deep_split,
                                                deep_dataset):
    # Why the pin exists. These carry the only deep traffic, and the hash makes no
    # promise about where they land.
    split, _ = deep_split
    _, test = _partition(deep_dataset, split)
    names = [r[TASK_NAME_FIELD] for r in test]
    assert set(DEEP_TASKS) <= set(names)
    # And they lead it, because a row's place in the split is its arrival slot and a
    # session this slow wants to start early.
    assert sorted(names[:len(DEEP_TASKS)]) == sorted(DEEP_TASKS)


def test_a_filled_pin_is_counted_off_the_rows(checkout, deep_dataset):
    from hls.provision.rows import pinned_test

    corpus = deep_dataset.corpus()
    assert deep_dataset.pinned_test_rows() == len(pinned_test(deep_dataset, corpus))
    assert deep_dataset.pinned_test_rows() == len(DEEP_TASKS)
