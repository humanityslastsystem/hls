"""The exec server's runtime, staged into its sandbox store by provisioning
(``hls.provision.sandbox_runtime``): kept when it already holds the pinned packages, built beside
its place and renamed into it otherwise, and a tree that does not hold them moved aside, never
removed."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hls.provision import sandbox_runtime
from hls.provision.sandbox_runtime import RUNTIME_PACKAGES, RUNTIME_PYTHON, RUNTIME_TAG, RuntimeStageError, held, stage


def _python(path: Path, exit_code: int) -> None:
    """A stand-in ``bin/python3`` that answers every check and install with *exit_code*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    """A uv whose ``python install`` lays out a standalone tree whose python passes the check."""
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text('#!/bin/sh\n# uv python install --install-dir DIR VERSION\n'
                  'tree="$4/cpython-$5-linux"\nmkdir -p "$tree/bin" "$tree/lib/python3.12"\n'
                  'touch "$tree/lib/python3.12/EXTERNALLY-MANAGED"\n'
                  'printf "#!/bin/sh\\nexit 0\\n" > "$tree/bin/python3"\nchmod +x "$tree/bin/python3"\n')
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sandbox_runtime, "_uv", lambda: str(uv))
    return uv


def test_the_tag_names_the_python_and_each_pin():
    assert RUNTIME_TAG == f"cpython-{RUNTIME_PYTHON}-fastapi-{RUNTIME_PACKAGES['fastapi']}-uvicorn-" \
                          f"{RUNTIME_PACKAGES['uvicorn']}"


def test_a_tree_is_held_only_when_its_python_passes_the_check(tmp_path):
    assert not held(tmp_path / "absent")
    _python(tmp_path / "good" / "bin" / "python3", 0)
    _python(tmp_path / "bad" / "bin" / "python3", 1)
    assert held(tmp_path / "good") and not held(tmp_path / "bad")


def test_a_held_tree_is_kept(tmp_path, monkeypatch):
    root = tmp_path / "store" / RUNTIME_TAG
    _python(root / "bin" / "python3", 0)
    monkeypatch.setattr(sandbox_runtime, "_uv", lambda: pytest.fail("a held tree is not built again"))
    assert stage(root) == "held"


def test_a_tree_is_built_beside_its_place_and_renamed_into_it(tmp_path, fake_uv):
    root = tmp_path / "store" / RUNTIME_TAG
    assert stage(root) == "staged"
    assert held(root) and not list(root.glob("lib/python*/EXTERNALLY-MANAGED"))
    assert [p.name for p in root.parent.iterdir()] == [RUNTIME_TAG]


def test_a_tree_that_does_not_hold_the_pins_is_moved_aside_never_removed(tmp_path, fake_uv):
    root = tmp_path / "store" / RUNTIME_TAG
    _python(root / "bin" / "python3", 1)
    (root / "kept").write_text("x")
    assert stage(root) == "staged" and held(root)
    [aside] = [p for p in root.parent.iterdir() if p.name != RUNTIME_TAG]
    assert aside.name.startswith(f"{RUNTIME_TAG}.unusable-") and (aside / "kept").read_text() == "x"


def test_a_failed_step_is_named(tmp_path, monkeypatch):
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\necho 'no network' >&2\nexit 2\n")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sandbox_runtime, "_uv", lambda: str(uv))
    with pytest.raises(RuntimeStageError, match=f"uv python install {RUNTIME_PYTHON} failed: no network"):
        stage(tmp_path / "store" / RUNTIME_TAG)


def test_the_agentic_runner_makes_its_stores_the_overlay_root_and_its_runtime(tmp_path, monkeypatch):
    from tasks.agentic_coding.task import TOOL_RUNNER

    staged = []
    monkeypatch.setattr(sandbox_runtime, "stage", lambda root: staged.append(root) or "staged")
    stores = {name: tmp_path / name for name in ("agentic-coding-images", "agentic-coding-overlays",
                                                 "agentic-coding-prefixes", "agentic-coding-runtime")}
    done = TOOL_RUNNER.provision_stores(stores)
    assert all(path.is_dir() for path in stores.values())
    assert staged == [stores["agentic-coding-runtime"] / RUNTIME_TAG]
    assert "agentic-coding-runtime: the exec server's runtime, staged" in done
    # What the plan checks of the stores is what provisioning made, the images and prefixes aside.
    missing = [e for e in TOOL_RUNNER.store_entries([])
               if not ((stores[e.store] / e.path).is_dir() if e.kind == "dir" else (stores[e.store] / e.path).is_file())]
    assert [e.what for e in missing] == ["the exec server's interpreter"]


def test_the_provision_stage_hands_the_stores_role_each_store_and_the_rows_entry(tmp_path):
    """The stage runs the ``stores`` role with the site's directory of each store the task names
    and, when the task has rows, the published rows entry, whose rows the stores are made for."""
    import json
    from types import SimpleNamespace

    from hls.stages import provision as stage
    from hls.taskref import TaskRef

    emitted, ran = [], []
    ctx = SimpleNamespace(
        cache=object(), job_id=None,
        plan=SimpleNamespace(instance="i", settings=SimpleNamespace(sandbox_stores={"s": tmp_path / "s",
                                                                                    "other": tmp_path / "o"})),
        params=stage.ProvisionParams(task=TaskRef(module="tasks.agentic_coding.task"), sandbox_stores=("s",)),
        events=SimpleNamespace(emit=lambda kind, **fields: emitted.append((kind, fields))))
    attempt = stage._Provision(ctx)
    attempt.run_role = lambda name, *args: ran.append((name, args))
    attempt.stores(SimpleNamespace(path=tmp_path / "rows"))
    attempt.stores(None)
    with_rows, without = (args for _, args in ran)
    assert with_rows == ("stores", "--task-module", "tasks.agentic_coding.task", "--stores",
                         json.dumps({"s": str(tmp_path / "s")}), "--rows", str(tmp_path / "rows"))
    assert without == with_rows[:-2]
    assert emitted == [("sandbox_stores_provisioned", {"stores": ["s"]})] * 2


def test_the_provision_roles_stores_command_makes_what_the_task_needs(tmp_path):
    """The role run as the provision stage runs it, in a process of its own; the runtime store
    already holds a runtime, so nothing is fetched."""
    import json
    import subprocess
    import sys

    stores = {name: str(tmp_path / name) for name in ("agentic-coding-images", "agentic-coding-overlays",
                                                      "agentic-coding-prefixes", "agentic-coding-runtime")}
    _python(Path(stores["agentic-coding-runtime"]) / RUNTIME_TAG / "bin" / "python3", 0)
    done = subprocess.run([sys.executable, "-m", "hls.provision.role", "stores", "--task-module",
                           "tasks.agentic_coding.task", "--stores", json.dumps(stores)],
                          capture_output=True, text=True, timeout=300, cwd=Path(__file__).parents[2])
    assert done.returncode == 0, done.stderr
    assert all(Path(path).is_dir() for path in stores.values())
    assert "agentic-coding-runtime: the exec server's runtime, held" in done.stdout
