from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls import baseline_credit
from hls.baseline_credit import BASELINE_CREDIT_FILE, credited, read_baseline_credit
from hls.core.poly import dump_spec
from hls.measured import defaults_path
from hls.taskref import (MeasuredPaths, TaskRef, apply_task_overrides, blob_id, blob_sha, first_on_import_path,
                         load_task, read_task, task_dir)
from hls.verify import calibration
from hls.verify.calibration import CALIBRATION_FILE, calibrated, calibration_kind, read_calibration

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", ["tasks.sample_task.task", "tasks.multitenant_slo.task"])
def test_a_task_ref_applies_its_overrides_to_the_modules_task(module):
    """The task a reference loads is the module's ``TASK`` with exactly the overridden leaves
    changed, and nothing else of it."""
    ref = TaskRef(module=module, overrides={"task.seed": 7, "task.contract.port": 9001,
                                            "task.hardware.cpu_count": 12})
    task = load_task(ref)
    assert (task.seed, task.contract.port, task.hardware.cpu_count) == (7, 9001, 12)
    bare = importlib.import_module(module).TASK
    assert _differing_leaves(dump_spec(task), dump_spec(bare)) == {"seed", "contract.port", "hardware.cpu_count"}
    assert dump_spec(load_task(TaskRef(module=module))) == dump_spec(bare)


def _differing_leaves(a, b, path: str = "") -> set[str]:
    """The dotted paths of the leaves of two dumped trees that differ, a key one tree lacks among
    them."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {leaf for key in a.keys() | b.keys()
                for leaf in _differing_leaves(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return {leaf for i, (x, y) in enumerate(zip(a, b)) for leaf in _differing_leaves(x, y, f"{path}.{i}")}
    return set() if a == b else {path}


def test_no_overrides_is_the_module_task():
    mod = importlib.import_module("tasks.sample_task.task")
    assert dump_spec(load_task(TaskRef(module="tasks.sample_task.task"))) == dump_spec(mod.TASK)


def _defaults(ref: TaskRef) -> MeasuredPaths:
    """The task's default files, as a plan that pins them hands them to a role."""
    return MeasuredPaths(calibration=defaults_path(ref, CALIBRATION_FILE),
                         credit=defaults_path(ref, BASELINE_CREDIT_FILE))


def test_calibration_and_credit_are_injected_from_the_files_given():
    ref = TaskRef(module="tasks.multitenant_slo.task", overrides={"task.hardware.cpu_count": 12})
    files = _defaults(ref)
    expected = credited(calibrated(load_task(ref), files.calibration)[0], files.credit)[0]
    task = load_task(ref, calibration=True, credit=True, files=files)
    assert dump_spec(task) == dump_spec(expected)
    assert task.scorer.baseline_log_credit is not None
    assert dump_spec(load_task(ref)) != dump_spec(task)


@pytest.mark.parametrize("flag", ["calibration", "credit"])
def test_a_task_that_needs_a_file_none_was_given_for_is_refused(flag):
    """A load asked to inject a file the task needs, with none given, refuses as the reader refuses
    a missing file."""
    with pytest.raises(Exception, match="was given"):
        load_task(TaskRef(module="tasks.multitenant_slo.task"), **{flag: True})



@pytest.mark.parametrize("module", ["tasks.multitenant_slo.task", "tasks.multimodal_tts.task"],
                         ids=["flips", "report"])
def test_the_task_reads_each_file_once_and_names_the_bytes_it_parsed(tmp_path, monkeypatch, module):
    """The calibration and the baseline credit are each read once, and the load names the blob id of
    the bytes it parsed: a file of the same content in another layout is other bytes, and the id
    named is that of the bytes read, even once the file has been rewritten since."""
    defaults = defaults_path(TaskRef(module=module), CALIBRATION_FILE).parent
    for name in (CALIBRATION_FILE, BASELINE_CREDIT_FILE):
        (tmp_path / name).write_text(json.dumps(json.loads((defaults / name).read_text()), indent=7))
    files = MeasuredPaths(calibration=tmp_path / CALIBRATION_FILE, credit=tmp_path / BASELINE_CREDIT_FILE)
    reads: list[str] = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(self.name) if self.parent == tmp_path
                                                          else None) or real(self))
    parsed: dict[str, bytes] = {}

    def rewritten(reader, name):
        """*reader*, after which its file holds the same content in yet another layout."""
        def read(*args, **kwargs):
            value, data = reader(*args, **kwargs)
            parsed[name] = data
            (tmp_path / name).write_text(json.dumps(json.loads(data), indent=3))
            return value, data
        return read

    monkeypatch.setattr(calibration, "read_calibration", rewritten(calibration.read_calibration, CALIBRATION_FILE))
    monkeypatch.setattr(baseline_credit, "read_baseline_credit",
                        rewritten(baseline_credit.read_baseline_credit, BASELINE_CREDIT_FILE))
    loaded = read_task(TaskRef(module=module), calibration=True, credit=True, files=files)
    assert sorted(reads) == sorted([BASELINE_CREDIT_FILE, CALIBRATION_FILE])
    for name, read in ((CALIBRATION_FILE, loaded.calibration), (BASELINE_CREDIT_FILE, loaded.credit)):
        assert read.blob == blob_id(parsed[name]) != blob_sha(tmp_path / name)
        assert read.blob != blob_sha(defaults / name)
    # What they parsed to is the default files' content, which the task was given.
    kind = calibration_kind(loaded.task)
    assert loaded.calibration.value == read_calibration(defaults / CALIBRATION_FILE, loaded.task, kind)[0]
    assert loaded.credit.value == read_baseline_credit(defaults / BASELINE_CREDIT_FILE, loaded.task)[0]
    assert loaded.task.scorer.baseline_log_credit == loaded.credit.value.log_credit_mean
    assert load_task(TaskRef(module=module), calibration=True, credit=True,
                     files=_defaults(TaskRef(module=module))) == loaded.task


def test_a_task_given_neither_file_names_none():
    loaded = read_task(TaskRef(module="tasks.multitenant_slo.task"))
    assert (loaded.calibration, loaded.credit) == (None, None)
    # A task none of whose verifiers needs a calibration and whose scorer needs no credit reads none.
    loaded = read_task(TaskRef(module="tasks.sample_task.task"), calibration=True, credit=True)
    assert (loaded.calibration, loaded.credit) == (None, None)


def test_an_override_that_names_no_field_is_refused():
    ref = TaskRef(module="tasks.sample_task.task", overrides={"task.contract.no_such_field": 1})
    with pytest.raises(ValueError, match="task.contract.no_such_field"):
        load_task(ref)


@pytest.mark.parametrize("key", ["seed", "artifact.image", "task", "task.a..b"])
def test_applying_an_override_that_is_no_task_path_is_refused(key):
    """The one application every role shares takes only the paths a task reference holds, so
    none of a key's characters is dropped for a prefix it does not have."""
    task = importlib.import_module("tasks.sample_task.task").TASK
    with pytest.raises(ValueError, match=r"overrides must be task\.<field>"):
        apply_task_overrides(task, {key: 1})


@pytest.mark.parametrize("key, value", [("task.drive_bounds.trace_cap_bytes", 4 * 2**30),
                                        ("task.drive_bounds.trace_request_bytes", 2**30),
                                        ("task.drive_bounds", None)])
def test_a_task_override_may_not_set_the_drive_bounds(key, value):
    """A task's drive bounds limit what a scored drive may make the harness hold, so only the task
    declares them: an override of them is refused in a task reference, as a plan holds one, and
    in the application every role shares."""
    with pytest.raises(ValidationError, match="drive_bounds"):
        TaskRef(module="tasks.sample_task.task", overrides={key: value})
    task = importlib.import_module("tasks.sample_task.task").TASK
    with pytest.raises(ValueError, match="drive_bounds"):
        apply_task_overrides(task, {key: value})


@pytest.mark.parametrize("overrides", [{"artifact.image": "x"}, {"task": 1}, {"task.": 1}, {"task.a..b": 1}])
def test_overrides_address_fields_of_the_task(overrides):
    with pytest.raises(ValidationError, match="task"):
        TaskRef(module="tasks.sample_task.task", overrides=overrides)


@pytest.mark.parametrize("module", ["tasks/sample_task/task.py", "tasks.sample-task", "", ".tasks"])
def test_a_module_is_a_dotted_python_module_name(module):
    with pytest.raises(ValidationError):
        TaskRef(module=module)


def test_a_module_must_declare_a_task():
    with pytest.raises(ValueError, match="declares no TASK"):
        load_task(TaskRef(module="hls.settings"))


def test_task_dir_is_the_task_modules_own_directory(tmp_path, monkeypatch):
    assert task_dir(TaskRef(module="tasks.sample_task.task")) == REPO / "tasks" / "sample_task"
    (tmp_path / "nsonly").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="no __file__"):
        task_dir(TaskRef(module="nsonly"))


def test_blob_sha_is_the_id_git_commits_the_file_under(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"log_credit": 1.25}) + "\n")
    env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "add", "baseline.json"], check=True, env=env)
    staged = subprocess.run(["git", "-C", str(tmp_path), "ls-files", "-s", "baseline.json"], check=True,
                            capture_output=True, text=True, env=env).stdout.split()[1]
    assert blob_sha(path) == staged
    assert blob_sha(REPO / "tasks/multitenant_slo/defaults/baseline.json") == subprocess.run(
        ["git", "hash-object", str(REPO / "tasks/multitenant_slo/defaults/baseline.json")], check=True,
        capture_output=True, text=True, env=env).stdout.strip()


_PROBE = """
import sys
sys.path[:0] = [{src!r}, {repo!r}]
import importlib
importlib.import_module({module!r})
before = set(sys.modules)
from hls.measured import BASELINE_CREDIT_FILE, CALIBRATION_FILE, defaults_path
from hls.taskref import MeasuredPaths, TaskRef, load_task
ref = TaskRef(module={module!r})
files = MeasuredPaths(calibration=defaults_path(ref, CALIBRATION_FILE), credit=defaults_path(ref, BASELINE_CREDIT_FILE))
load_task(ref, calibration=True, credit=True, files=files)
heavy = ("torch", "transformers", "vllm", "sglang", "datasets", "huggingface_hub", "pyarrow", "scipy")
print(sorted({{m.split(".")[0] for m in set(sys.modules) - before}} & set(heavy)))
"""


@pytest.mark.parametrize("module", ["tasks.sample_task.task", "tasks.multitenant_slo.task"])
def test_loading_a_task_imports_nothing_heavy_beyond_the_module_itself(module):
    code = _PROBE.format(src=str(REPO / "src"), repo=str(REPO), module=module)
    out = subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


def test_a_checkout_goes_first_on_the_import_path_for_the_block_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", ["/a", "/b"])
    with first_on_import_path(tmp_path):
        assert sys.path == [str(tmp_path), "/a", "/b"]
    assert sys.path == ["/a", "/b"]
    # A checkout already on the path stays where it is, before the block, during it and after.
    monkeypatch.setattr(sys, "path", ["/a", str(tmp_path)])
    with first_on_import_path(tmp_path):
        assert sys.path == ["/a", str(tmp_path)]
    assert sys.path == ["/a", str(tmp_path)]
    # The path is restored when the block raises.
    monkeypatch.setattr(sys, "path", ["/a"])
    with pytest.raises(RuntimeError), first_on_import_path(tmp_path):
        raise RuntimeError("in the block")
    assert sys.path == ["/a"]
