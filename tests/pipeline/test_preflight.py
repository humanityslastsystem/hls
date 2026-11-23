"""The plan-time checks of scored commands: a committed checkout, the default baseline credit
and calibration a plan pins committed, an answer in the store's answers area to every question a completed oracle fill
recorded for the rows a grading's verify will ask about, and the arm's resolved record as a
build prints it."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from hls import cli
from hls.agent.uploads import tree_digest
from hls.arms import ArmRecord, Route
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.obs.logsetup import OWNED_LOGGERS, configure_logging
from hls.pipeline import commands, preflight
from hls.pipeline.checkout import READ_GIT_TIMEOUT_SEC, CodeState, code_state
from hls.pipeline.planner import PlanError
from hls.measured import MeasuredFile, MeasuredFiles, pinned_key
from hls.pipeline.preflight import (OracleCoverage, UploadFolder, arm_lines, check_committed_tree,
                                    check_defaults_committed, check_oracle_coverage, check_skills_folder,
                                    check_upload_folder)
from hls.pipeline.stage import StageFault
from hls.prompt import JinjaPromptTemplate
from hls.provision.role import Refused, resolve
from hls.provision.rows import tenant_subpath
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.workload import ShortOutputCap
from hls.stages import agent_build
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, blob_id, task_dir
from hls.verify.answers import StoreAnswers, answer_key, answers_area, questions_name
from hls.verify.calibration import CALIBRATION_FILE
from hls.verify.fill import FillRecord, caching_references, corpus_prompts, fill, question_scopes, question_shape
from hls.verify.run import may_deploy
from tasks.multimodal_asr.task import TASK as ASR
from tasks.multimodal_omni.task import TASK as OMNI
from tasks.multitenant_slo.task import TASK as MSLO
from tests.verify_stage import fixture
from tests.verify_stage.fixture import AskVerifier, GeneratorReference

# ---------------------------------------------------------------------------
# A committed tree
# ---------------------------------------------------------------------------


def _code(checkout: Path, dirty: str | None) -> CodeState:
    return CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=dirty)


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo* as a test's own user, whatever the machine's git configuration holds; a
    local repository may be cloned as a submodule."""
    config = ("-c", "user.name=preflight", "-c", "user.email=preflight@example.com", "-c", "commit.gpgsign=false",
              "-c", "protocol.file.allow=always", "-c", "maintenance.auto=false", "-c", "gc.auto=0")
    return subprocess.run(["git", *config, "-C", str(repo), *args], capture_output=True, check=True,
                          text=True).stdout.strip()


@pytest.fixture
def outer(tmp_path) -> Path:
    """A checkout with a harbor submodule, both committed and clean."""
    sub = tmp_path / "harbor-upstream"
    sub.mkdir()
    _git(sub, "init", "-q", "-b", "main")
    (sub / "h.py").write_text("y = 1\n")
    _git(sub, "add", "h.py")
    _git(sub, "commit", "-q", "-m", "harbor")
    root = tmp_path / "outer"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("x = 1\n")
    _git(root, "add", "a.py")
    _git(root, "submodule", "add", "-q", str(sub), "harbor")
    _git(root, "commit", "-q", "-m", "a checkout")
    return root


def test_a_dirty_checkout_is_refused_naming_it(tmp_path):
    with pytest.raises(PlanError, match=rf"the checkout {re.escape(str(tmp_path))} or its harbor submodule has "
                                        r"uncommitted changes \(dirty diff cccccccccccc\).*"
                                        r"runs from committed code; commit them"):
        check_committed_tree(_code(tmp_path, "c" * 64))


def test_a_clean_checkout_with_harbor_at_the_commit_its_head_pins_passes(outer):
    state = code_state(outer)
    assert state.dirty_diff_sha256 is None
    check_committed_tree(state)


def test_a_moved_harbor_pointer_is_refused_naming_both_commits(outer):
    # harbor checked out at a commit HEAD's pointer does not name: git status calls it modified,
    # and the dirty digest, which leaves submodules to the recorded harbor commit, does not.
    pinned = _git(outer, "rev-parse", "HEAD:harbor")
    (outer / "harbor" / "b.py").write_text("z = 1\n")
    _git(outer / "harbor", "add", "b.py")
    _git(outer / "harbor", "commit", "-q", "-m", "moved")
    state = code_state(outer)
    assert state.dirty_diff_sha256 is None and state.harbor_commit != pinned
    with pytest.raises(PlanError, match=rf"^the checkout {re.escape(str(outer.resolve()))} pins harbor at "
                                        rf"{pinned[:12]}, but its harbor submodule is at {state.harbor_commit[:12]}: a "
                                        r"scored instance runs from committed code; "
                                        r"commit the pointer, or run git submodule update$"):
        check_committed_tree(state)


@pytest.mark.parametrize("files", [{"a.py": "x = 1\n"}, {"a.py": "x = 1\n", "harbor/h.py": "y = 1\n"}],
                         ids=["no-harbor", "harbor-committed-as-files"])
def test_a_checkout_whose_head_pins_no_harbor_is_refused(tmp_path, files):
    root = tmp_path / "plain"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "no harbor submodule")
    with pytest.raises(PlanError, match=r"pins no harbor commit at HEAD, so the harbor commit b{12} is no committed "
                                        r"code$"):
        check_committed_tree(_code(root, None))


# ---------------------------------------------------------------------------
# The default baseline credit and calibration a plan pins
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git checkout holding one task package and its default baseline credit and calibration,
    committed, on the import path under a name no other test imports."""
    root = tmp_path / "checkout"
    package = f"preflight_{secrets.token_hex(4)}"
    (root / package).mkdir(parents=True)
    (root / package / "__init__.py").write_text("")
    (root / package / "task.py").write_text("TASK = None\n")
    defaults = root / package / "defaults"
    defaults.mkdir(parents=True)
    (defaults / "baseline.json").write_text('{"log_credit_mean": 0.5}\n')
    (defaults / CALIBRATION_FILE).write_text('{"kind": "flips"}\n')
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "a task")
    monkeypatch.syspath_prepend(str(root))
    return SimpleNamespace(root=root, package=package, module=f"{package}.task")


def _pinned(root: Path, path: str, source: str = "defaults") -> MeasuredFile:
    """A pin of the file at *path* under *root*, by the blob id of its bytes as they are now."""
    return MeasuredFile(source=source, path=path, blob=blob_id((root / path).read_bytes()))


def _files(repo, *, credit: bool = True, calibration: bool = True) -> MeasuredFiles:
    """The pins a plan of *repo*'s task makes of its default files as the checkout holds them."""
    base = f"{repo.package}/defaults"
    return MeasuredFiles(credit=_pinned(repo.root, f"{base}/baseline.json") if credit else None,
                         calibration=_pinned(repo.root, f"{base}/{CALIBRATION_FILE}") if calibration else None)


def test_default_files_committed_as_pinned_pass(repo):
    files = _files(repo)
    check_defaults_committed(repo.root, files)
    # The pin is the id git commits the file under, which the score role records.
    assert files.credit.blob == _git(repo.root, "rev-parse", f"HEAD:{repo.package}/defaults/baseline.json")


def test_a_plan_that_pins_nothing_passes(repo):
    check_defaults_committed(repo.root, MeasuredFiles())


def test_a_store_file_needs_no_commit(tmp_path):
    """A file of the store's measured area is pinned by its bytes alone: the checkout, which here
    has no commit at all, holds nothing of it."""
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    store = MeasuredFile(source="store", path=pinned_key("t", "baseline.json", "a" * 40), blob="a" * 40)
    check_defaults_committed(root, MeasuredFiles(credit=store))


def test_an_uncommitted_change_to_a_default_file_is_refused(repo):
    (repo.root / repo.package / "defaults" / "baseline.json").write_text('{"log_credit_mean": 0.6}\n')
    with pytest.raises(PlanError, match=rf"{repo.package}/defaults/baseline.json in the checkout is blob "
                                        r"[0-9a-f]{12}, not the [0-9a-f]{12} committed at HEAD: a scored instance "
                                        r"loads only a committed default$"):
        check_defaults_committed(repo.root, _files(repo, calibration=False))


def test_an_untracked_default_file_is_refused(repo):
    _git(repo.root, "rm", "-q", "--cached", f"{repo.package}/defaults/{CALIBRATION_FILE}")
    _git(repo.root, "commit", "-q", "-m", "untrack the calibration")
    with pytest.raises(PlanError, match=rf"{repo.package}/defaults/{CALIBRATION_FILE} is not committed at HEAD of "
                                        r".*: a scored instance loads only a committed default$"):
        check_defaults_committed(repo.root, _files(repo, credit=False))


def test_a_directory_committed_in_the_files_place_is_refused(repo):
    pin = _files(repo, calibration=False).credit
    _git(repo.root, "rm", "-q", f"{repo.package}/defaults/baseline.json")
    (repo.root / repo.package / "defaults" / "baseline.json").mkdir()
    (repo.root / repo.package / "defaults" / "baseline.json" / "part").write_text("{}")
    _git(repo.root, "add", ".")
    _git(repo.root, "commit", "-q", "-m", "a directory where the credit was")
    with pytest.raises(PlanError, match=rf"{repo.package}/defaults/baseline.json is not committed at HEAD of "):
        check_defaults_committed(repo.root, MeasuredFiles(credit=pin))


def test_a_checkout_with_no_commit_is_refused(repo, tmp_path):
    files = _files(repo, calibration=False)
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    with pytest.raises(PlanError, match=r"has no HEAD commit"):
        check_defaults_committed(root, files)


def test_a_checkout_named_through_a_link_is_the_checkout(repo, tmp_path):
    (tmp_path / "link").symlink_to(repo.root)
    check_defaults_committed(tmp_path / "link", _files(repo))


def test_a_git_command_that_does_not_finish_is_refused(repo, monkeypatch):
    files = _files(repo, calibration=False)

    def hangs(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(preflight.subprocess, "run", hangs)
    with pytest.raises(PlanError, match=rf"^git rev-parse in {re.escape(str(repo.root.resolve()))} did not finish in "
                                        rf"{READ_GIT_TIMEOUT_SEC} s$"):
        check_defaults_committed(repo.root, files)


# ---------------------------------------------------------------------------
# The folder a build uploads into the sandbox
# ---------------------------------------------------------------------------


@pytest.fixture
def docs(repo) -> SimpleNamespace:
    """Reference documents at ``assets/docs`` of the checkout, as a task's upload folder names
    them from its module's directory, and the producer that pins them by their tree digest."""
    folder = repo.root.resolve() / "assets" / "docs"
    (folder / "ptx").mkdir(parents=True)
    (folder / "ptx" / "isa.md").write_text("the ISA\n")
    (folder / "index.md").write_text("the index\n")
    producer = SimpleNamespace(upload_dir=("../assets/docs",), upload_sha256=tree_digest(folder), upload_dest=None)
    return SimpleNamespace(folder=folder, producer=producer)


def _upload(repo, producer) -> UploadFolder | None:
    return check_upload_folder(repo.root, repo.module, producer)


#: Each case of an upload folder the plan-time check and the build's start both judge.
UPLOAD_CASES = ["pinned", "a-link-on-the-way-inside", "absent", "a-link-on-the-way-outside", "outside",
                "a-changed-file", "an-extra-file", "a-link-inside"]


def _upload_case(repo, docs, tmp_path, case: str) -> SimpleNamespace:
    """The producer of upload folder case *case*, the folder made so."""
    producer, folder = docs.producer, docs.folder
    if case == "a-link-on-the-way-inside":
        folder.rename(repo.root.resolve() / "real-docs")
        folder.symlink_to(repo.root.resolve() / "real-docs")
    elif case == "absent":
        shutil.rmtree(folder)
    elif case == "a-link-on-the-way-outside":
        folder.rename(tmp_path / "elsewhere")
        folder.symlink_to(tmp_path / "elsewhere")
    elif case == "outside":
        shutil.copytree(folder, tmp_path / "elsewhere" / "docs")
        producer = SimpleNamespace(upload_dir=(str(tmp_path / "elsewhere" / "docs"),), upload_sha256=producer.upload_sha256,
                                   upload_dest=None)
    elif case == "a-changed-file":
        (folder / "ptx" / "isa.md").write_text("another ISA\n")
    elif case == "an-extra-file":
        (folder / "notes.md").write_text("mine\n")
    elif case == "a-link-inside":
        (folder / "elsewhere.md").symlink_to(folder / "index.md")
    return producer


def test_the_upload_folder_the_task_pins_passes(repo, docs):
    assert _upload(repo, docs.producer) == UploadFolder(paths=("assets/docs",), sha256=docs.producer.upload_sha256)


def test_an_upload_folder_named_through_a_link_inside_the_checkout_passes(repo, docs, tmp_path):
    producer = _upload_case(repo, docs, tmp_path, "a-link-on-the-way-inside")
    assert _upload(repo, producer) == UploadFolder(paths=("real-docs",), sha256=docs.producer.upload_sha256)


def test_a_task_with_no_upload_folder_needs_none(repo):
    assert _upload(repo, SimpleNamespace(upload_dir=None, upload_sha256=None)) is None


def test_an_absent_upload_folder_is_refused_naming_it_and_the_remedy(repo, docs, tmp_path):
    producer = _upload_case(repo, docs, tmp_path, "absent")
    with pytest.raises(PlanError, match=rf"^{re.escape(repo.module)} uploads \.\./assets/docs \(producer\.upload_dir\) "
                                        rf"into every build's sandbox, and {re.escape(str(docs.folder))} is not a "
                                        r"directory: copy in the reference documents the task pins by "
                                        r"producer\.upload_sha256, which are not committed$"):
        _upload(repo, producer)


def test_a_missing_toolchain_library_is_refused_with_the_command_that_assembles_it(tmp_path):
    """The folder the three tasks upload is the toolchain library; its refusal ends with the
    script's invocation. Another folder's refusal ends as the one above."""
    import importlib

    producer = importlib.import_module("tasks.multitenant_slo.task").ARTIFACT.producer
    with pytest.raises(PlanError, match=r"^tasks\.multitenant_slo\.task uploads \.\./assets/references "
                                        r"\(producer\.upload_dir\) into every build's sandbox, and .*: copy in the "
                                        r"reference documents the task pins by producer\.upload_sha256, which are not "
                                        r"committed; run uv run python scripts/assemble_references\.py$"):
        check_upload_folder(tmp_path, "tasks.multitenant_slo.task", producer)


@pytest.mark.parametrize("case, words", [
    ("a-link-on-the-way-outside", "elsewhere lies outside the checkout"),
    ("outside", "elsewhere/docs lies outside the checkout"),
    ("a-changed-file", "has tree digest"),
    ("an-extra-file", "has tree digest"),
    ("a-link-inside", r"elsewhere\.md is a link"),
])
def test_an_upload_folder_the_task_does_not_pin_is_refused(repo, docs, tmp_path, case, words):
    producer = _upload_case(repo, docs, tmp_path, case)
    with pytest.raises(PlanError, match=rf"into every build's sandbox, and .*{words}.*: copy in the reference"):
        _upload(repo, producer)


@pytest.mark.parametrize("case", UPLOAD_CASES)
def test_plan_time_and_the_builds_start_judge_an_upload_folder_alike(repo, docs, tmp_path, case):
    """Both judge it through ``hls.agent.uploads.checked_upload``; a rule either kept of its own
    would show here as a case one passes and the other refuses, or refuses for another reason."""
    producer = _upload_case(repo, docs, tmp_path, case)
    checkout = repo.root.resolve()
    try:
        found = _upload(repo, producer)
        planned = ("passes", [str(checkout / path) for path in found.paths], found.sha256)
    except PlanError as e:
        planned = ("refused", str(e), None)
    try:
        folders, digest = agent_build.upload_folder(producer, task_dir(TaskRef(module=repo.module)), checkout)
        started = ("passes", [str(folder) for folder in folders], digest)
    except StageFault as e:
        assert (e.fault.fault_class, e.fault.retryable) == ("infra", False)
        started = ("refused", e.fault.reason.removeprefix("producer.upload_dir: "), None)
    assert planned[0] == started[0] == ("passes" if case in ("pinned", "a-link-on-the-way-inside") else "refused")
    if planned[0] == "passes":
        assert planned[1:] == started[1:]
    else:
        assert f"into every build's sandbox, and {started[1]}: copy in" in planned[1]


def test_an_upload_destination_under_the_workspace_passes(repo, docs):
    producer = SimpleNamespace(**{**vars(docs.producer), "upload_dest": "/workspace/references"})
    assert _upload(repo, producer) == UploadFolder(paths=("assets/docs",), sha256=docs.producer.upload_sha256)


@pytest.mark.parametrize("dest", ["/etc/references", "/workspace/../etc/references", "/taskfiles"])
def test_an_upload_destination_the_sandbox_does_not_bind_is_refused(repo, docs, dest):
    """The build would refuse it when it starts (``hls.agent.uploads.upload_destination``), after
    the allocation was granted; the plan refuses it first."""
    producer = SimpleNamespace(**{**vars(docs.producer), "upload_dest": dest})
    with pytest.raises(PlanError, match=rf"^{re.escape(repo.module)} copies its upload folder to {re.escape(dest)} "
                                        rf"\(producer\.upload_dest\), and {re.escape(dest)} is neither under /task "
                                        r"nor under /workspace$"):
        _upload(repo, producer)


# ---------------------------------------------------------------------------
# The skills a build copies into the sandbox
# ---------------------------------------------------------------------------

#: Each case of a skills folder the plan-time check and the build's start both judge.
SKILL_CASES = ["a-folder-of-skills", "one-skill", "a-directory-that-is-no-skill", "no-skill-at-all", "absent",
               "outside"]


def _skill(root: Path, name: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n")


def _skills_case(repo, tmp_path, case: str) -> SimpleNamespace:
    """The producer of skills folder case *case*, named from the task module's directory as a
    task names it, the folder made so."""
    root = repo.root.resolve() / "assets" / "skills"
    root.mkdir(parents=True)
    value = "../assets/skills"
    if case == "a-folder-of-skills":
        _skill(root, "nsight")
        _skill(root, "cuda")
    elif case == "one-skill":
        _skill(root.parent, "cuda")
        value = "../assets/cuda"
    elif case == "a-directory-that-is-no-skill":
        _skill(root, "cuda")
        (root / "broken").mkdir()
    elif case == "no-skill-at-all":
        (root / "README.md").write_text("no skill here\n")
    elif case == "absent":
        root.rmdir()
    elif case == "outside":
        _skill(tmp_path / "elsewhere", "cuda")
        value = str(tmp_path / "elsewhere")
    return SimpleNamespace(skills_dir=value)


def test_a_task_with_no_skills_folder_copies_none(repo):
    assert check_skills_folder(repo.root, repo.module, SimpleNamespace(skills_dir=None)) == []


def test_the_skills_a_task_names_pass_by_name(repo, tmp_path):
    assert check_skills_folder(repo.root, repo.module, _skills_case(repo, tmp_path, "a-folder-of-skills")) == \
        ["cuda", "nsight"]


def test_a_skills_root_holding_a_directory_that_is_no_skill_is_refused_naming_the_folder_and_the_directory(
        repo, tmp_path):
    """The task's own folder is at fault: the plan refuses it before an allocation is asked for,
    naming the task, its field, the folder and each directory that holds no ``SKILL.md``."""
    root = repo.root.resolve() / "assets" / "skills"
    with pytest.raises(PlanError, match=rf"^{re.escape(repo.module)} names \.\./assets/skills as its skills "
                                        r"\(producer\.skills_dir\), which every build copies into its sandbox, and "
                                        rf"Skill root {re.escape(str(root))} contains child directories without "
                                        r"SKILL\.md: broken$"):
        check_skills_folder(repo.root, repo.module, _skills_case(repo, tmp_path, "a-directory-that-is-no-skill"))


@pytest.mark.parametrize("case", SKILL_CASES)
def test_plan_time_and_the_builds_start_judge_a_skills_folder_alike(repo, tmp_path, case):
    """Both judge it through ``hls.agent.uploads.task_skills``; a rule either kept of its own
    would show here as a case one passes and the other refuses, or refuses for another reason."""
    producer = _skills_case(repo, tmp_path, case)
    checkout = repo.root.resolve()
    try:
        planned = ("passes", check_skills_folder(repo.root, repo.module, producer))
    except PlanError as e:
        planned = ("refused", str(e))
    try:
        skills = agent_build.skill_folders(producer, task_dir(TaskRef(module=repo.module)), checkout)
        started = ("passes", [s.name for s in skills])
    except StageFault as e:
        assert (e.fault.fault_class, e.fault.retryable) == ("infra", False)
        started = ("refused", e.fault.reason.removeprefix("producer.skills_dir: "))
    assert planned[0] == started[0] == ("passes" if case in ("a-folder-of-skills", "one-skill") else "refused")
    if planned[0] == "passes":
        assert planned[1] == started[1] == (["cuda", "nsight"] if case == "a-folder-of-skills" else ["cuda"])
    else:
        assert planned[1].endswith(f"which every build copies into its sandbox, and {started[1]}")


# ---------------------------------------------------------------------------
# The answers a grading's verify will ask for
# ---------------------------------------------------------------------------

#: The commit every hub source of the tasks below is read at, so their provisioning resolves on
#: the login node as the provision stage resolves it.
COMMIT = "c" * 40
#: The GPU kind and the reference image's file digest a grading of those tasks verifies on.
GPU, IMAGE = "B200", "e" * 64
#: A single-turn test split, and one of two turns per session.
ROWS = [{"q": "1"}, {"q": "2"}, {"q": "3"}]
CHAT_ROWS = [{"q": "1", "ctx": "c", "qs": ["a", "b"], "as": ["x", "y"]}]
#: The fill's writer name, as the oracle_fill stage makes one.
FILLER = "warmers+preflight-answers+w1+oracle_fill+1"


def _corpus(*, multi_turn: bool = False, revision: str = COMMIT, template: str = "{{q}}") -> DataSource:
    if multi_turn:
        dataset = HFDataset(path="org/chat", revision=revision, source_split="test", id_column="q",
                            context_col="ctx", questions_col="qs", answers_col="as", answers_text_key=None)
    else:
        dataset = HFDataset(path="org/corp", revision=revision, source_split="test", id_column="q",
                            prompt=JinjaPromptTemplate(template=template), target=JinjaPromptTemplate(template="B"))
    return DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=1), test=Count(value=3)))


def _task(*, cap: int = 16, chat: bool = False, revision: str = COMMIT) -> TaskSpec:
    """Tenant t0, whose test rows a verifier asks a reference that caches its answers, under an
    output cap of *cap* tokens, its corpus read at *revision*; with *chat*, beside tenant chat,
    a corpus of two turns per session whose reference does not cache."""
    workload = fixture._workload().model_copy(update={"output_length_cap": ShortOutputCap(cap_tokens=cap)})
    tenants = [Tenant(id="t0", data_source=_corpus(revision=revision), workload=workload,
                      verifiers=[AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))])]
    if chat:
        tenants.append(Tenant(id="chat", data_source=_corpus(multi_turn=True), workload=workload.model_copy(
                                  update={"mean_turns_per_session": 2.0}),
                              verifiers=[AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=False))]))
    return TaskSpec(id="preflight-answers", model=ModelSpec(base_checkpoint="org/m", revision=COMMIT, arch="TX"),
                    hardware=Hardware(accelerator=AcceleratorSpec(kind=GPU, count=1)), contract=fixture.CONTRACT,
                    scorer=fixture.SCORER, tenants=tenants)


def _rows(root: Path, rows: dict[str, list[dict]]) -> Path:
    """Each tenant's test *rows* under *root* as the fill role's container sees them; never in
    the store, so a check that read them could not."""
    for tenant, held in rows.items():
        (root / tenant_subpath(tenant) / "test").mkdir(parents=True)
        (root / tenant_subpath(tenant) / "test" / "rows.json").write_text(json.dumps(held))
    return root


def _fill(store: PosixStore, task: TaskSpec, rows: Path, *, gpu_kind: str = GPU, image: str = IMAGE,
          first: int | None = None, writer: str = FILLER) -> FillRecord:
    """Run the oracle fill's own code (``hls.verify.fill``) over the test rows under *rows*, as
    its role runs it under the writer name *writer*, banking into the store's answers area under
    *gpu_kind* and *image*, and recording its question set for the rows key the task's
    provisioning resolves to, as the stage tells its role. With *first*, a fill that died after
    its first *first* prompts: it banked those and recorded no question set."""
    asked = {tenant for ref in caching_references(task) for tenant in ref.tenants}
    prompts = corpus_prompts(task, rows, tenants=asked)
    answers = StoreAnswers(PosixStore(store.path(answers_area(task.id))), gpu_kind=gpu_kind, image=image,
                           writer=writer, strict=True)
    complete = first is None
    return asyncio.run(fill(task, prompts if complete else prompts[:first], answers, chunk=2,
                            rows_key=resolve(task).rows_key if complete else None))


def _fingerprint(task: TaskSpec) -> str:
    [ref] = caching_references(task)
    return ref.reference.fingerprint(task)


def _key_dir(store: PosixStore, task: TaskSpec) -> Path:
    return store.path(answers_area(task.id)) / answer_key(_fingerprint(task), GPU, IMAGE)


def _question_set(task: TaskSpec) -> str:
    """The name of the question set a completed fill of *task* records."""
    [(_, scope)] = question_scopes(task).items()
    return questions_name(resolve(task).rows_key, question_shape(task, scope))


#: The task reference the coverage tests' gradings name: the module a refusal names the fill of. The
#: check imports nothing by it.
REF = TaskRef(module="tests.pipeline.covered_task")


def _cover(store: PosixStore, task: TaskSpec, **changes) -> OracleCoverage | None:
    return check_oracle_coverage(store, **{"task_ref": REF, "task_id": task.id, "task": task, "gpu_kind": GPU,
                                           "reference_sha256": IMAGE, **changes})


def _no_fill(task_id: str, where: str, name: str, module: str = REF.module) -> str:
    """The refusal of a grading of the task module *module* whose answers key holds no question set
    named *name*."""
    return (rf"^{task_id}'s verifiers keep reference answers in the store's answers area, and there is no completed "
            rf"fill for these rows: {re.escape(where)} holds no {re.escape(name)}, the question set an oracle fill "
            rf"writes once it has banked an answer to every one\. Run hls oracle fill "
            rf"{re.escape(module)} first$")


def test_a_grading_finds_an_answer_to_every_question_a_completed_fill_recorded(tmp_path):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}))
    # The check reads the question set, never the rows.
    shutil.rmtree(tmp_path / "rows")
    assert _cover(store, task) == OracleCoverage(rows=resolve(task).rows_key, fingerprints=(_fingerprint(task),),
                                                 prompts=3, files=2, writers=("warmers/preflight-answers/w1",))


def test_the_coverage_names_every_instance_whose_answers_files_it_read(tmp_path):
    """A fill that died banked answers before the one that completed: the grading reads the files
    of both, and names both instances; a writer no stage names is named as it is."""
    store, task = PosixStore(tmp_path / "store"), _task()
    rows = _rows(tmp_path / "rows", {"t0": ROWS})
    _fill(store, task, rows, first=1, writer="warmers+preflight-answers+w0+oracle_fill+1")
    _fill(store, task, rows, first=2, writer="by-hand")
    _fill(store, task, rows)
    assert _cover(store, task).writers == ("by-hand", "warmers/preflight-answers/w0", "warmers/preflight-answers/w1")


@pytest.mark.parametrize("first", [None, 2], ids=["no-fill", "a-fill-that-died"])
def test_a_grading_with_no_completed_fill_for_its_rows_is_refused(tmp_path, first):
    store, task = PosixStore(tmp_path / "store"), _task()
    if first is not None:
        _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}), first=first)
    where = f"oracle/{task.id}/{answer_key(_fingerprint(task), GPU, IMAGE)}"
    with pytest.raises(PlanError, match=_no_fill(task.id, where, _question_set(task))):
        _cover(store, task)


def test_a_question_with_no_answer_is_refused_naming_it(tmp_path):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}))
    # The second chunk's file, which held the answer to row 3, is gone.
    (_key_dir(store, task) / f"{FILLER}.000002.json").unlink()
    fp = _fingerprint(task)
    where = f"oracle/{task.id}/{answer_key(fp, GPU, IMAGE)}"
    with pytest.raises(PlanError, match=rf"^{task.id}'s verify would compute reference answers inside the grading: "
                                        rf"1 of the 3 questions a completed fill recorded for reference "
                                        rf"{re.escape(fp)} have no answer in {where} \(1 answers files\) for {GPU} "
                                        r"GPUs and reference image eeeeeeeeeeee, t0/3 first\. Run hls oracle fill "
                                        rf"{re.escape(REF.module)} first$"):
        _cover(store, task)


def test_the_answers_are_looked_up_in_the_area_of_the_plans_task_id(tmp_path):
    # The verify stage mounts the area of the id the plan names the task by, whatever id an
    # override gave the task it loads.
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}))
    where = f"oracle/renamed/{answer_key(_fingerprint(task), GPU, IMAGE)}"
    with pytest.raises(PlanError, match=_no_fill("renamed", where, _question_set(task))):
        _cover(store, task, task_id="renamed")


@pytest.mark.parametrize("fill, check", [
    ({"gpu_kind": "H100"}, {}),
    ({"image": "f" * 64}, {}),
    ({}, {"gpu_kind": "H100"}),
    ({}, {"reference_sha256": "f" * 64}),
], ids=["filled-on-another-gpu-kind", "filled-in-another-image", "graded-on-another-gpu-kind",
        "graded-in-another-image"])
def test_a_fill_for_another_gpu_kind_or_image_does_not_count(tmp_path, fill, check):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}), **fill)
    with pytest.raises(PlanError, match=r"there is no completed fill for these rows"):
        _cover(store, task, **check)


def test_an_edit_that_reshapes_every_prompt_of_the_same_rows_needs_its_own_fill(tmp_path):
    # The same rows and reference, so the same rows key and answer key; the output cap stamped
    # on every prompt differs, so no question digests as a filled one did.
    store, filled, graded = PosixStore(tmp_path / "store"), _task(cap=16), _task(cap=32)
    assert resolve(graded).rows_key == resolve(filled).rows_key and _fingerprint(graded) == _fingerprint(filled)
    rows = _rows(tmp_path / "rows", {"t0": ROWS})
    _fill(store, filled, rows)
    where = f"oracle/{graded.id}/{answer_key(_fingerprint(graded), GPU, IMAGE)}"
    with pytest.raises(PlanError, match=_no_fill(graded.id, where, _question_set(graded))):
        _cover(store, graded)
    # Its own fill records a set of its own beside the first.
    _fill(store, graded, rows, writer="warmers+preflight-answers+w2+oracle_fill+1")
    assert _cover(store, graded).prompts == 3 and _cover(store, filled).prompts == 3


def test_a_question_is_answered_only_under_its_own_prompt_digest(tmp_path):
    store, filled, graded = PosixStore(tmp_path / "store"), _task(cap=16), _task(cap=32)
    rows = _rows(tmp_path / "rows", {"t0": ROWS})
    _fill(store, graded, rows)
    # The answers to the graded prompts are gone; the same rows' answers under the other cap stay.
    for answers in _key_dir(store, graded).glob(f"{FILLER}.*.json"):
        answers.unlink()
    _fill(store, filled, rows)
    with pytest.raises(PlanError, match=r"3 of the 3 questions a completed fill recorded for reference .* have no "
                                        r"answer .* \(2 answers files\)"):
        _cover(store, graded)


@pytest.mark.parametrize("state", ["no-fill", "a-fill-that-died", "an-answers-file-lost", "a-completed-fill"])
def test_the_check_passes_exactly_when_a_verify_of_the_whole_split_loads_no_engine(tmp_path, state):
    store, task = PosixStore(tmp_path / "store"), _task()
    rows = _rows(tmp_path / "rows", {"t0": ROWS})
    if state != "no-fill":
        _fill(store, task, rows, first=2 if state == "a-fill-that-died" else None)
    if state == "an-answers-file-lost":
        (_key_dir(store, task) / f"{FILLER}.000001.json").unlink()
    # The prompts a drive of the whole test split sends, stamped as it stamps them.
    sent = corpus_prompts(task, rows)
    answers = StoreAnswers(PosixStore(store.path(answers_area(task.id))), gpu_kind=GPU, image=IMAGE,
                           writer="grading+verify+1")
    try:
        _cover(store, task)
        covered = True
    except PlanError:
        covered = False
    assert covered is (state == "a-completed-fill")
    assert may_deploy(task, sent, answers) is not covered


def test_answers_no_completed_fill_recorded_are_refused_though_a_verify_would_serve_them(tmp_path):
    # Every answer is there, banked by a fill told no rows key, as a verify banks its own: with
    # no question set the check cannot know they are the rows' every question, and refuses.
    store, task = PosixStore(tmp_path / "store"), _task()
    rows = _rows(tmp_path / "rows", {"t0": ROWS})
    answers = StoreAnswers(PosixStore(store.path(answers_area(task.id))), gpu_kind=GPU, image=IMAGE, writer=FILLER)
    asyncio.run(fill(task, corpus_prompts(task, rows), answers, chunk=2))
    assert may_deploy(task, corpus_prompts(task, rows), StoreAnswers(
        PosixStore(store.path(answers_area(task.id))), gpu_kind=GPU, image=IMAGE, writer="grading+verify+1")) is False
    with pytest.raises(PlanError, match=r"there is no completed fill for these rows"):
        _cover(store, task)


def test_only_the_rows_a_caching_reference_is_asked_about_count(tmp_path):
    # Tenant chat's rows run two turns and no reference caches its answers: the verify asks
    # the area nothing about them, and the fill records none of them.
    store, task = PosixStore(tmp_path / "store"), _task(chat=True)
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS, "chat": CHAT_ROWS}))
    assert _cover(store, task).prompts == 3


#: A value a refusal must never print.
_NEVER_SHOWN = "value-the-refusal-never-prints"


@pytest.mark.parametrize("text, errors", [
    ('{"questions": ', "json_invalid at the top level"),
    (f'{{"fingerprint": 7, "shape": "{_NEVER_SHOWN}"}}',
     "string_type fingerprint; missing gpu_kind (required by this version of hls); "
     "missing image (required by this version of hls)"),
], ids=["not-json", "not-a-set"])
def test_a_question_set_that_does_not_parse_is_refused_naming_its_errors_and_no_value(tmp_path, text, errors):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}))
    (_key_dir(store, task) / _question_set(task)).write_text(text)
    where = f"oracle/{task.id}/{answer_key(_fingerprint(task), GPU, IMAGE)}/{_question_set(task)}"
    with pytest.raises(PlanError) as refused:
        _cover(store, task)
    assert str(refused.value).startswith(f"{where} cannot be read as a question set: {errors}")
    assert _NEVER_SHOWN not in str(refused.value)


@pytest.mark.parametrize("text, errors", [
    (b'{"fingerprint": ', "json_invalid at the top level"),
    (b'{"fingerprint": 7, "gpu_kind": ["' + _NEVER_SHOWN.encode() + b'"]}',
     "string_type fingerprint; string_type gpu_kind; missing image"),
], ids=["not-json", "not-an-answers-file"])
def test_an_unreadable_answers_file_is_skipped_as_the_verify_role_skips_it_and_named(tmp_path, text, errors):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}))
    key = answer_key(_fingerprint(task), GPU, IMAGE)
    (_key_dir(store, task) / f"{FILLER}.000002.json").write_bytes(text)
    with pytest.raises(PlanError, match=r"1 of the 3 questions .* \(1 answers files\) .*; answers files the verify "
                                        rf"role skips as unreadable: 1, the first: skipping reference answers "
                                        rf"{key}/{re.escape(FILLER)}\.000002\.json, which does not parse "
                                        rf"\(ValidationError: {re.escape(errors)}") as refused:
        _cover(store, task)
    assert _NEVER_SHOWN not in str(refused.value)


def test_a_task_whose_provisioning_does_not_resolve_is_refused(tmp_path):
    with pytest.raises(PlanError, match=r"^preflight-answers's provisioning does not resolve: .*revision 'main' is "
                                        r"not a commit id"):
        _cover(PosixStore(tmp_path / "store"), _task(revision="main"))


def test_a_resolve_refusal_of_several_lines_is_refused_on_one(tmp_path, monkeypatch):
    def refuses(task):
        raise Refused("the task's provisioning does not validate:\n  rows_key\n    not a rows key")

    monkeypatch.setattr(preflight, "resolve", refuses)
    with pytest.raises(PlanError, match=r"^preflight-answers's provisioning does not resolve: the task's provisioning "
                                        r"does not validate: rows_key not a rows key$"):
        _cover(PosixStore(tmp_path / "store"), _task())


class _First(GeneratorReference):
    """The fake generator's reference under one name, so under one fingerprint."""


class _Second(GeneratorReference):
    """The same under another name, so under another fingerprint and another answer key."""


def _two_keys() -> TaskSpec:
    """Tenants a and b, each verified against a caching reference of its own fingerprint."""
    workload = fixture._workload().model_copy(update={"output_length_cap": ShortOutputCap(cap_tokens=16)})
    tenants = [Tenant(id="a", data_source=_corpus(), workload=workload,
                      verifiers=[AskVerifier(reference=_First(decode=Greedy(), cache=True))]),
               Tenant(id="b", data_source=_corpus(template="B: {{q}}"), workload=workload,
                      verifiers=[AskVerifier(reference=_Second(decode=Greedy(), cache=True))])]
    return TaskSpec(id="preflight-two-keys", model=ModelSpec(base_checkpoint="org/m", revision=COMMIT, arch="TX"),
                    hardware=Hardware(accelerator=AcceleratorSpec(kind=GPU, count=1)), contract=fixture.CONTRACT,
                    scorer=fixture.SCORER, tenants=tenants)


@pytest.mark.parametrize("lost", [None, "first", "second", "both"])
def test_every_answer_key_is_checked(tmp_path, lost):
    store, task = PosixStore(tmp_path / "store"), _two_keys()
    rows = _rows(tmp_path / "rows", {"a": ROWS, "b": ROWS})
    _fill(store, task, rows)
    keys = [answer_key(fingerprint, GPU, IMAGE) for fingerprint in question_scopes(task)]
    assert len(keys) == 2
    area = store.path(answers_area(task.id))
    for i in {None: [], "first": [0], "second": [1], "both": [0, 1]}[lost]:
        for answers in (area / keys[i]).glob(f"{FILLER}.*.json"):
            answers.unlink()
    reader = StoreAnswers(PosixStore(area), gpu_kind=GPU, image=IMAGE, writer="grading+verify+1")
    deploys = may_deploy(task, corpus_prompts(task, rows), reader)
    if lost is None:
        assert _cover(store, task).prompts == 6 and deploys is False
    else:
        with pytest.raises(PlanError, match=r"of the 3 questions a completed fill recorded for reference .* have no "
                                            r"answer"):
            _cover(store, task)
        assert deploys is True


@pytest.mark.parametrize("task, module", [(ASR, "tasks.multimodal_asr.task"), (OMNI, "tasks.multimodal_omni.task")],
                         ids=["asr", "omni"])
def test_a_grading_of_a_task_with_no_completed_fill_is_refused_naming_its_question_set(tmp_path, task, module):
    # The rows key and the question set's name come from the task's declaration alone, on the
    # login node.
    [(fingerprint, scope)] = question_scopes(task).items()
    kind = task.hardware.accelerator.kind
    where = f"oracle/{task.id}/{answer_key(fingerprint, kind, IMAGE)}"
    name = questions_name(resolve(task).rows_key, question_shape(task, scope))
    with pytest.raises(PlanError, match=_no_fill(task.id, where, name, module)):
        check_oracle_coverage(PosixStore(tmp_path / "store"), task_ref=TaskRef(module=module), task_id=task.id,
                              task=task, gpu_kind=kind, reference_sha256=IMAGE)


def test_a_task_without_a_caching_reference_needs_no_answers(tmp_path):
    assert caching_references(MSLO) == []
    assert check_oracle_coverage(PosixStore(tmp_path / "store"), task_ref=TaskRef(module="tasks.multitenant_slo.task"),
                                 task_id=MSLO.id, task=MSLO, gpu_kind=MSLO.hardware.accelerator.kind,
                                 reference_sha256=IMAGE) is None


def test_a_refusal_names_the_fill_with_the_gradings_task_overrides(tmp_path):
    """A fill run without the grading's task overrides answered other rows, so the refusal names
    the fill command with each of them, as a command line gives it."""
    store, task = PosixStore(tmp_path / "store"), _task()
    ref = TaskRef(module=REF.module, overrides={"task.data_source.split.test.value": 3, "task.seed": 7,
                                                "task.agent.name": "a b", "task.flag": True})
    with pytest.raises(PlanError, match=rf"Run hls oracle fill {re.escape(REF.module)} --task\.agent\.name 'a b' "
                                        r"--task\.data_source\.split\.test\.value 3 --task\.flag true "
                                        r"--task\.seed 7 first$"):
        _cover(store, task, task_ref=ref)


def test_a_reference_image_not_known_before_the_images_stage_is_left_to_the_caller(tmp_path):
    assert _cover(PosixStore(tmp_path / "store"), _task(), reference_sha256=None) is None


# ---------------------------------------------------------------------------
# A refusal, as a command shows it
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_logging():
    """The hls loggers as a command configures them (``configure_logging``: a handler on the
    command's stderr at INFO), put back after the test. Call it inside the test, once pytest's
    capture holds stderr."""
    owned = [logging.getLogger(name) for name in OWNED_LOGGERS]
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in owned]
    root_level = logging.getLogger().level
    yield lambda: configure_logging(verbose=False)
    for lg, (handlers, level, propagate) in zip(owned, saved, strict=True):
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    logging.getLogger().setLevel(root_level)


@pytest.mark.parametrize("refusal", ["dirty-checkout", "uncommitted-baseline", "no-completed-fill",
                                     "missing-answer"])
def test_a_refusal_is_one_line_on_stderr_and_submits_nothing(refusal, repo, tmp_path, monkeypatch, capsys,
                                                             cli_logging):
    store, task = PosixStore(tmp_path / "store"), _task()
    _fill(store, task, _rows(tmp_path / "rows", {"t0": ROWS}), first=2 if refusal == "no-completed-fill" else None)
    if refusal == "missing-answer":
        (_key_dir(store, task) / f"{FILLER}.000002.json").unlink()
    (repo.root / repo.package / "defaults" / "baseline.json").write_text('{"log_credit_mean": 0.6}\n')
    started = []
    monkeypatch.setattr(commands, "start", lambda *args, **kwargs: started.append(args) or 0)
    capsys.readouterr()
    cli_logging()

    def plan():
        if refusal == "dirty-checkout":
            check_committed_tree(_code(repo.root, "c" * 64))
        elif refusal == "uncommitted-baseline":
            check_defaults_committed(repo.root, _files(repo, calibration=False))
        else:
            _cover(store, task)
        return "a plan"

    with pytest.raises(typer.Exit) as ended:
        cli._plan_and_start("grade", plan, {})
    assert ended.value.exit_code == 1 and started == []
    out, err = capsys.readouterr()
    assert out == "" and len(err.splitlines()) == 1 and err.startswith("hls grade: "), err


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------

OPUS = ArmRecord(name="arm-a", harness="claude-code", model="claude-opus-5", route="anthropic",
                 context_window=1_000_000, output_limit=64_000, effort="max",
                 route_table=Route(credential="anthropic_relay", base_url="anthropic_relay_url", auth="bearer"))


def test_the_arm_lines_name_no_value():
    assert arm_lines(OPUS, soft_sec=10_800, hard_sec=12_600, arm_change=None) == [
        "arm arm-a: claude-code, model claude-opus-5, context window 1,000,000 tokens, output limit 64,000 tokens, "
        "effort max",
        "route anthropic: credential from secret anthropic_relay, base URL from secret anthropic_relay_url, "
        "auth bearer",
        "budget: told 10,800 s, held to 12,600 s",
    ]


def test_an_arm_on_its_harnesss_own_provider_with_no_soft_budget_says_so():
    k3 = ArmRecord(name="k3", harness="claude-code", model="k3", route="relay", context_window=200_000,
                   output_limit=64_000, route_table=Route(credential="relay_token", auth="bearer"))
    assert arm_lines(k3, soft_sec=None, hard_sec=3_300, arm_change="the relay raised its output cap") == [
        "arm k3: claude-code, model k3, context window 200,000 tokens, output limit 64,000 tokens, effort unset "
        "(the harness's default)",
        "route relay: credential from secret relay_token, the harness's own provider, auth bearer",
        "budget: told no budget, held to 3,300 s",
        "declared change from its last build: the relay raised its output cap",
    ]
