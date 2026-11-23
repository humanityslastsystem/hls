"""The internal-reference check (``.github/scripts/check_internal_refs.py``): each class counts a
line that names something internal and leaves a line of the harness's own vocabulary alone, an
allowed token clears only itself, and a strict class with a line fails the run."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "check_internal_refs.py"
_spec = importlib.util.spec_from_file_location("check_internal_refs", SCRIPT)
refs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refs)

#: Lines that name something internal, by the class that counts them. Each is written in pieces,
#: so that no line of this file is one.
INTERNAL = [
    ("site", ("the drives ran on klo", "ne")),
    ("site", ("account = ", '"zl', 'ab"')),
    ("site", ('assert "--partition=gpu-h', '200" in args')),
    ("site", ('CPU_SITE = {"cpu_partition": "cpu-', 'g2"}')),
    ("site", ("using 4 processors on nodes node11 in partition gpu-l", "40")),
    ("node", ("the limit read on g30", "50")),
    ("node", ("a node such as n30", "12")),
    ("job_id", ("the peak of job 414", "11689")),
    ("job_id", ("/node/hls-4101", "9110/roles/artifact")),
    ("job_id", ("a second node (3454", "84) holds less")),
    ("job_id", ("sac", "ct MaxRSS 16.12G for the step")),
    ("date", ("the limit as read 2026-10", "-05")),
    ("date", ("the run of 10", "-05, which stopped early")),
    ("arm", ("the op", "us5 build of the task")),
    ("arm", ("the ki", "mik3 rollouts")),
    ("arm", ("the gl", "m53 arm's builds")),
    ("arm", ("the r2", "/medium cell")),
    ("arm", ("every build of round", " 2")),
    ("published", ("all 16 pub", "lished gradings held it")),
    ("published", ("the grant every pub", "lished build had")),
    ("published", ("regrades of five pub", "lished cells")),
    ("published", ("the commit the pub", "lished runs were provisioned at")),
    ("published", ("the allocation its task's pub", "lished builds ran in")),
    ("published", ("the cal", "112 runs")),
    ("published", ("rerun the sl", "ate")),
    ("published", ("the 1010 camp", "aign's gradings")),
    ("published", ("branch camp", "aign/next")),
    ("published", ("the rer", "oll-1010 gradings")),
    ("pr_issue", ("fixed in PR #", "112")),
    ("pr_issue", ("see is", "sue 148")),
    ("pr_issue", ("merged as #", "183")),
    ("pr_issue", ("url = https://github.com/sh", "g8/hls.git")),
    ("pr_issue", ("see github.com/sh", "g8/hls/pull/9")),
    ("commit", ("as of commit 9d6", "97f98a")),
    ("history", ("the driver no", " longer holds a drive's replies")),
    ("history", ("the split policy is no", " longer a dataset's concern")),
    ("history", ("the runner used", " to write the file twice")),
    ("history", ("a missing ``role`` used", " to raise KeyError")),
    ("history", ("TaskSpec no", " longer has the field")),
    ("history", ("`reference` no", " longer carries a model")),
    ("history", ("the file was previ", "ously read whole")),
    ("measured", ("the build took 67 minutes on tilli", "cum")),
    ("measured", ("the pack took less on the work", "station")),
    ("measured", ("both si", "tes add the first four")),
    ("measured", ("the limit of the smallest no", "de measured")),
    ("ruling", ("the rule R", "-made-up-name")),
]

#: Lines of the harness's own vocabulary or of a legitimate value, by the class that must not
#: count them.
LEGITIMATE = [
    ("site", "the site's partition and account"),
    ("site", 'assert "--partition=gpu-a" in args and "--gpus=h200:4" in args and "--partition=cpu-a" not in args'),
    ("site", '# gpu_gres = { H200 = "h200", L40 = "l40" }, cpu_partition = "cpu", account = "mylab"'),
    ("node", r'@pytest.mark.parametrize("out, ok", [("1000\n1000", True), ("0\n0", False)])'),
    ("node", r'msg = "exit 1\n2000 lines"'),
    ("job_id", 'store.write_bytes(f"{root}/orchestrator/gen-00000001/events.jsonl", b"{}")'),
    ("job_id", "subsample_seed: int = 20240915"),
    ("job_id", "pattern_1 = _run(20250622)"),
    ("job_id", 'assert uuids == ["GPU-10111213-1415-1617-1819-1a1b1c1d1e1f"]'),
    ("job_id", 'r = self.commands("sacct", "--noheader", "--parsable2", "--allocations")'),
    ("job_id", "_STEP_TOOL_TIMEOUT_SEC = {\"sacct\": 10.0, \"scancel\": 10.0, \"squeue\": 10.0}"),
    ("job_id", "#: of a role step's memory (sacct MaxRSS) reads for a multi-GB weights download."),
    ("job_id", 'JOB = "1234567"'),
    ("date", 'PROTOCOL = "2024-11-05"'),
    ("date", "the tokenizer's main since 2024-09-20"),
    ("date", 'msg = "2000-01-01 12:00:00 orchestrator gen 1: instance_ended"'),
    ("arm", 'KEY = "runs/t/r1/stages/agent_build/MANIFEST.json"'),
    ("arm", 'model = "kimi-k3"'),
    ("arm", "| agentic_coding | `zai-org/GLM-5.3-Flash` | the checkpoint the task serves |"),
    ("arm", '[arms.arm-a]'),
    ("published", "a published image, which holds no constraints file"),
    ("published", "the row of a published rollout with no rollout.json"),
    ("published", "whose largest published trace held trace_bytes of JSON"),
    ("published", "For an instance whose finalize published a grading record, say its hardware"),
    ("published", "The three published rollouts under root/TASK/"),
    ("published", "the viewer reads a store's warmers or a tree of published runs"),
    ("published", "Earlier published builds of the task, each with its arm"),
    ("published", "deepresearch and agentic_coding reroll on their published images"),
    ("published", "what a campaign has measured"),
    ("published", "a task that rerolls on its published images"),
    ("published", "its floor laid with a double course of slate over a bed"),
    ("pr_issue", "10. [Hardware + resources + placement](#10-hardware--resources--placement)"),
    ("pr_issue", 'GitSource("https://github.com/sh' 'g8/ptx-isa-markdown", "8ac5f8e150fe2693a8fd4348d129abaecfc725a6")'),
    ("commit", "the commit point writes its bundle to the store once"),
    ("history", "the cache no longer holds s-a, which it held when the plan was made"),
    ("history", "it waits no longer than the stage's deadline"),
    ("history", "raise Refused(f\"Job {job_id} is no longer pending execution\")"),
    ("history", "#: When the stage first saw the role's step started; None until then."),
    ("history", "# The artifact's deployed parameters, used to check the trace floor"),
    ("history", "the helper is used to keep the verdict readable"),
    ("history", "a PATH that names only the directory no longer finds them"),
    ("history", "once its pid no longer holds it with resident memory"),
    ("measured", "The agent's budget is measured on the runner's clock"),
    ("measured", "the Poisson tail at the same mean is the larger of the two and bounds it"),
    ("measured", "half as much again, for a slower node or a build that grew"),
    ("ruling", "an R-squared of 0.9, read from the X-Request-Id header's row"),
    ("ruling", "the r-made-up-name directory"),
]


def _line(pieces: tuple[str, ...]) -> str:
    return "".join(pieces)


@pytest.mark.parametrize("name, pieces", INTERNAL, ids=[f"{n}-{i}" for i, (n, _) in enumerate(INTERNAL)])
def test_a_line_that_names_something_internal_is_counted_in_its_class(name, pieces):
    assert name in refs.classes_of("src/hls/example.py", _line(pieces))


@pytest.mark.parametrize("name, line", LEGITIMATE, ids=[f"{n}-{i}" for i, (n, _) in enumerate(LEGITIMATE)])
def test_a_line_of_the_harness_own_vocabulary_is_not_counted(name, line):
    assert name not in refs.classes_of("src/hls/example.py", line)


def test_every_class_has_a_line_it_counts_and_one_it_does_not():
    assert {name for name, _ in INTERNAL} == set(refs.CLASSES) == {name for name, _ in LEGITIMATE}


def test_an_allowed_token_clears_its_class_in_its_file_alone():
    cls, path, token, _ = next(entry for entry in refs.ALLOW if entry[0] == "date")
    line = f"#: Node ({token}), with the npm it ships."
    assert cls not in refs.classes_of(path, line)
    assert cls in refs.classes_of("src/hls/example.py", line)
    # Another date on the line is still one.
    assert cls in refs.classes_of(path, line + _line((" Read again 2026-10", "-05.")))


@pytest.mark.parametrize("entry", refs.ALLOW, ids=[f"{e[0]}-{Path(e[1]).name}-{i}" for i, e in enumerate(refs.ALLOW)])
def test_every_allowed_token_is_in_its_file_matches_its_class_and_has_a_reason(entry):
    """An entry that clears nothing is deleted with the text it was written for."""
    cls, path, token, why = entry
    assert cls in refs.CLASSES and why
    lines = [line for line in (ROOT / path).read_text().splitlines() if token in line]
    assert lines, f"{path} holds no {token!r}"
    assert refs.PATTERNS[cls].search(token), f"{token!r} is not what the {cls} pattern matches"
    assert all(cls not in refs.classes_of(path, line) for line in lines)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """A git work tree at *tmp_path* that tracks *files*."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *files], check=True)
    return tmp_path


CLEAN = '"""A module."""\n\nJOB = "1234567"  # a made-up id\n'


def test_a_strict_class_with_a_line_fails_the_run_and_prints_the_line(tmp_path, monkeypatch, capsys):
    seeded = CLEAN + "# " + _line(("the drives ran on klo", "ne")) + "\n"
    monkeypatch.chdir(_tree(tmp_path, {"src/a.py": seeded, "docs/b.md": "plain text\n"}))
    assert refs.main(["--strict-classes", "site,node"]) == 1
    out = capsys.readouterr().out
    assert "1 lines of the strict classes (site, node):" in out
    assert "src/a.py:4: [site/comment] # " + _line(("the drives ran on klo", "ne")) in out
    # The line is of another class than these, and without a strict class it is only counted.
    assert refs.main(["--strict-classes", "node,job_id,ruling"]) == 0
    assert refs.main([]) == 0
    assert "strict classes" not in capsys.readouterr().out


@pytest.mark.parametrize("name, pieces", INTERNAL, ids=[f"{n}-{i}" for i, (n, _) in enumerate(INTERNAL)])
def test_strict_fails_on_a_line_of_any_class_in_any_tracked_text(name, pieces, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(_tree(tmp_path, {"src/a.py": CLEAN, "docs/b.md": f"plain text\n{_line(pieces)}\n"}))
    assert refs.main(["--strict"]) == 1
    assert f"docs/b.md:2: [{name}/other] {_line(pieces)}" in capsys.readouterr().out


def test_a_tree_with_no_line_of_a_strict_class_passes(tmp_path, monkeypatch, capsys):
    legitimate = "".join(f"{line}\n" for _, line in LEGITIMATE)
    monkeypatch.chdir(_tree(tmp_path, {"src/a.py": CLEAN, "notes.md": legitimate}))
    assert refs.main(["--strict-classes", ",".join(refs.CLASSES)]) == 0
    assert refs.main(["--strict"]) == 0
    assert "strict classes" not in capsys.readouterr().out


@pytest.mark.parametrize("value", ["sites", "site,,nodes", ""])
def test_a_strict_class_that_is_not_a_class_is_refused(value, capsys):
    with pytest.raises(SystemExit) as stop:
        refs.main(["--strict-classes", value])
    assert stop.value.code == 2 and "not a class" in capsys.readouterr().err


def test_the_lint_workflow_runs_the_check_strict_on_every_class():
    workflow = (ROOT / ".github" / "workflows" / "lint.yml").read_text()
    assert re.findall(r"run: (python \.github/scripts/check_internal_refs\.py.*)\n", workflow) == [
        "python .github/scripts/check_internal_refs.py --strict"]
