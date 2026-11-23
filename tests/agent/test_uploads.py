"""What agent_build copies into the sandbox from the task's own directory: the producer's uploaded
folder and its skills, each folder read from the task module's directory, copied by one host role
before the sandbox starts. The build's ``uploads`` step runs here alone: the copying role's
arguments are kept rather than started. ``tests/agent/test_stage.py`` pins the uploaded folder's
digest and the checkout it must lie in, and ``tests/pipeline/test_preflight.py`` that a scored
command's plan judges each folder by the same rule."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.agent.uploads import merged_digest, task_folder, tree_digest
from hls.agent.workspace import SKILLS, SandboxLayout
from hls.pipeline.stage import StageFault
from hls.stages import agent_build
from hls.taskref import TaskRef, task_dir

#: A task module of the agent tests; its directory holds ``build_task_files``.
MODULE = "tests.agent.build_task"
#: The checkout this repository's tests run from.
REPO = Path(__file__).resolve().parents[2]


class _Uploads(agent_build._Build):
    """The build of a checkout at *checkout* with no other context: its one copying role is
    recorded, not started."""

    def __init__(self, checkout: Path) -> None:
        self.params = SimpleNamespace(task=TaskRef(module=MODULE))
        self.ctx = SimpleNamespace(plan=SimpleNamespace(code=SimpleNamespace(checkout=checkout)))
        self.argv: list[str] | None = None

    def _host(self, name: str, argv: list[str], **kw) -> None:
        self.argv = argv

    def _wait(self, name: str) -> None:
        return None

    @staticmethod
    def _check(outcome: None) -> None:
        return None

    def copies(self) -> list[tuple[Path, Path]]:
        """Each (source, destination) pair the role was given, in order."""
        assert self.argv is not None
        pairs = self.argv[4:]
        return [(Path(src), Path(dst)) for src, dst in zip(pairs[::2], pairs[1::2])]


def _artifact(**producer) -> SimpleNamespace:
    """An artifact whose producer names *producer*'s folders, the uploaded one with its digest."""
    upload = producer.get("upload_dir")
    digest = merged_digest([Path(folder) for folder in upload]) if upload is not None else None
    return SimpleNamespace(producer=SimpleNamespace(**{"upload_dir": None, "upload_dest": None, "skills_dir": None,
                                                      "upload_sha256": digest, **producer}))


def _skill(root: Path, name: str) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n")
    return skill


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("notes\n")
    return docs


def test_a_relative_folder_is_read_from_the_task_modules_directory():
    base = task_dir(TaskRef(module=MODULE))
    assert task_folder("build_task_files", base, REPO) == (base / "build_task_files").resolve()


def test_an_absolute_folder_is_read_as_it_is(tmp_path):
    (tmp_path / "docs").mkdir()
    assert task_folder(str(tmp_path / "docs"), Path("/nowhere"), tmp_path) == tmp_path / "docs"


def test_a_skills_folder_that_is_no_directory_is_refused_and_not_retried(tmp_path):
    with pytest.raises(StageFault) as e:
        agent_build.skill_folders(SimpleNamespace(skills_dir="nope"), tmp_path, tmp_path)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"producer.skills_dir: {tmp_path / 'nope'} is not a directory"


def test_an_upload_without_a_destination_goes_to_the_default(tmp_path):
    docs = _docs(tmp_path)
    layout = SandboxLayout(tmp_path / "sandbox")
    build = _Uploads(tmp_path)
    assert build.uploads(layout, _artifact(upload_dir=(str(docs),))) == (None, tree_digest(docs))
    # UPLOAD_DEST, /task/files, is the host side of /task.
    assert build.copies() == [(docs, layout.task / "files")]


def test_an_upload_goes_to_the_destination_the_task_names(tmp_path):
    docs = _docs(tmp_path)
    layout = SandboxLayout(tmp_path / "sandbox")
    build = _Uploads(tmp_path)
    build.uploads(layout, _artifact(upload_dir=(str(docs),), upload_dest="/workspace/references"))
    assert build.copies() == [(docs, layout.workspace / "references")]


def test_several_folders_are_each_copied_into_the_one_destination(tmp_path):
    """A task's folders merge at its destination: one role copies each into it, in the task's
    order, and the digest the build records is the merged view's."""
    docs = _docs(tmp_path)
    own = tmp_path / "own"
    (own / "ref").mkdir(parents=True)
    (own / "ref" / "notes.txt").write_text("own\n")
    layout = SandboxLayout(tmp_path / "sandbox")
    build = _Uploads(tmp_path)
    assert build.uploads(layout, _artifact(upload_dir=(str(docs), str(own)), upload_dest="/workspace/references")) \
        == (None, merged_digest([docs, own]))
    assert build.copies() == [(docs, layout.workspace / "references"), (own, layout.workspace / "references")]


def test_two_folders_that_hold_the_same_path_are_refused_before_any_copy(tmp_path):
    docs = _docs(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "README.md").write_text("other\n")
    build = _Uploads(tmp_path)
    artifact = SimpleNamespace(producer=SimpleNamespace(upload_dir=(str(docs), str(other)), upload_dest=None,
                                                        skills_dir=None, upload_sha256="a" * 64))
    with pytest.raises(StageFault) as e:
        build.uploads(SandboxLayout(tmp_path / "sandbox"), artifact)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"producer.upload_dir: README.md is held by both {docs} and {other}"
    assert build.argv is None


@pytest.mark.parametrize("dest", ["/etc/references", "/workspace/../etc/references", "/taskfiles", "references",
                                  "/"])
def test_an_upload_destination_under_neither_the_task_nor_the_workspace_directory_is_refused_and_not_retried(
        tmp_path, dest):
    """The sandbox binds ``/task`` and ``/workspace`` for the agent's files; a copy anywhere else
    would never reach it, and one that climbs out with ``..`` would land outside the sandbox's
    directories on the host. The task's own field is at fault, so a retry would fail alike, and
    no role starts."""
    docs = _docs(tmp_path)
    build = _Uploads(tmp_path)
    with pytest.raises(StageFault) as e:
        build.uploads(SandboxLayout(tmp_path / "sandbox"), _artifact(upload_dir=(str(docs),), upload_dest=dest))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"producer.upload_dest: {dest} is neither under /task nor under /workspace"
    assert build.argv is None


def test_a_skills_root_is_copied_one_directory_per_skill(tmp_path):
    root = tmp_path / "skills"
    _skill(root, "nsight")
    _skill(root, "cuda")
    layout = SandboxLayout(tmp_path / "sandbox")
    build = _Uploads(tmp_path)
    assert build.uploads(layout, _artifact(skills_dir=str(root))) == (SKILLS, None)
    assert build.copies() == [(root / "cuda", layout.task / "skills" / "cuda"),
                              (root / "nsight", layout.task / "skills" / "nsight")]


def test_a_single_skill_is_copied_under_its_own_name(tmp_path):
    skill = _skill(tmp_path, "cuda")
    layout = SandboxLayout(tmp_path / "sandbox")
    build = _Uploads(tmp_path)
    assert build.uploads(layout, _artifact(skills_dir=str(skill))) == (SKILLS, None)
    assert build.copies() == [(skill, layout.task / "skills" / "cuda")]


def test_a_skills_root_with_directories_that_are_no_skills_is_refused_naming_them_and_not_retried(tmp_path):
    """The task's own folder is at fault, and a retry would read the same one: the refusal names
    the folder and each directory in it that holds no ``SKILL.md``, before any role starts."""
    root = tmp_path / "skills"
    _skill(root, "cuda")
    (root / "broken").mkdir()
    (root / "notes").mkdir()
    build = _Uploads(tmp_path)
    with pytest.raises(StageFault) as e:
        build.uploads(SandboxLayout(tmp_path / "sandbox"), _artifact(skills_dir=str(root)))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == (f"producer.skills_dir: Skill root {root} contains child directories without "
                                    "SKILL.md: broken, notes")
    assert build.argv is None


def test_a_skills_folder_holding_no_skill_is_refused_and_not_retried(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "README.md").write_text("no skill here\n")
    build = _Uploads(tmp_path)
    with pytest.raises(StageFault) as e:
        build.uploads(SandboxLayout(tmp_path / "sandbox"), _artifact(skills_dir=str(root)))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith(f"producer.skills_dir: Skill path {root} must be a skill directory")
    assert build.argv is None


def test_a_task_with_no_folders_starts_no_role(tmp_path):
    build = _Uploads(tmp_path)
    assert build.uploads(SandboxLayout(tmp_path / "sandbox"), _artifact()) == (None, None)
    assert build.argv is None
