from __future__ import annotations

import ast
import multiprocessing as mp
from pathlib import Path

import pytest

from hls.store import layout as layout_module
from hls.store.layout import (AREAS, CALIBRATIONS, CREDITS, EXPORTS, GRADINGS_DIR, INDEX_DIR, ROLLOUTS, WARMERS,
                              InstanceKey, InstanceLayout, NotAnInstance, check_manifest_key, check_stage, task_index)
from hls.store.posix import PosixStore


def test_rollout_and_grading_paths():
    r = InstanceLayout.rollout("multitenant_slo", "20261002T000000Z-abc123")
    assert r.root == "runs/multitenant_slo/20261002T000000Z-abc123"
    assert r.plan == f"{r.root}/plan"
    assert r.plan_revision(3) == f"{r.root}/plan/rev-00000003.json"
    assert r.lease == f"{r.root}/lease"
    assert r.cancel_request == f"{r.root}/cancel-request.json"
    assert r.submission(0, 1) == f"{r.root}/allocations/0/submissions/1"
    assert r.allocation_status(0, 2) == f"{r.root}/allocations/0/submissions/2/status.json"
    assert r.orchestrator_file(2, "events.jsonl") == f"{r.root}/orchestrator/gen-00000002/events.jsonl"
    assert r.manifest("agent_build") == f"{r.root}/stages/agent_build/MANIFEST.json"
    assert r.attempt_file("verify", 2, "status.json") == f"{r.root}/stages/verify/attempts/2/status.json"
    g = r.grading("g-20261002T010000Z-def456")
    assert g.root == f"{r.root}/gradings/g-20261002T010000Z-def456"


def test_credit_and_calibration_roots():
    assert InstanceLayout.credit("tab_completion", "c1").root == "credits/tab_completion/c1"
    assert InstanceLayout.calibration("multitenant_slo", "k1").root == "calibrations/multitenant_slo/k1"


@pytest.mark.parametrize("bad", ["Agent", "1x", "a-b", "", "a/b", "verify\n"])
def test_check_stage_rejects(bad):
    with pytest.raises(ValueError):
        check_stage(bad)


def test_layout_manifest_keys_are_manifest_keys():
    r = InstanceLayout.rollout("t", "r")
    keys = [
        r.manifest("agent_build"),
        r.grading("g").manifest("verify"),
        InstanceLayout.credit("t", "c").manifest("drive"),
    ]
    assert [check_manifest_key(k) for k in keys] == keys
    with pytest.raises(ValueError):
        check_manifest_key(r.plan)


def test_bad_ids_rejected():
    with pytest.raises(ValueError):
        InstanceLayout.rollout("t", "../x")


@pytest.mark.parametrize(
    "build",
    [
        lambda: InstanceLayout.rollout("t", "r/x"),
        lambda: InstanceLayout.rollout("a/b", "r"),
        lambda: InstanceLayout.credit("t", "c/d"),
        lambda: InstanceLayout.calibration("t", "k/1"),
        lambda: InstanceLayout.rollout("t", "r").grading("g/h"),
        lambda: InstanceLayout.rollout("t", "r\n"),
    ],
)
def test_ids_with_a_slash_or_bad_segment_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_instances_are_disjoint():
    grading = InstanceLayout.rollout("t", "r").grading("g")
    with pytest.raises(ValueError):
        InstanceLayout.rollout("t", "r/gradings/g")
    assert grading.root == "runs/t/r/gradings/g"


def test_claim_attempt_counts_up(tmp_path):
    store = PosixStore(tmp_path)
    lay = InstanceLayout.rollout("t", "r")
    assert lay.attempts(store, "package") == []
    assert lay.claim_attempt(store, "package") == 1
    assert lay.claim_attempt(store, "package") == 2
    assert lay.attempts(store, "package") == [1, 2]


def _claim(root: str, q) -> None:
    q.put(InstanceLayout.rollout("t", "r").claim_attempt(PosixStore(Path(root)), "verify"))


def test_concurrent_claims_are_unique(tmp_path):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_claim, args=(str(tmp_path), q)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    got = sorted(q.get(timeout=5) for _ in procs)
    assert got == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "build",
    [
        lambda r: r.plan_revision(0),
        lambda r: r.submission(-1, 1),
        lambda r: r.submission(0, 0),
        lambda r: r.attempt("verify", 0),
        lambda r: r.orchestrator_file(0, "events.jsonl"),
        lambda r: r.submission(True, 1),
    ],
)
def test_bad_indices_are_rejected(build):
    with pytest.raises(ValueError):
        build(InstanceLayout.rollout("t", "r"))


def test_attempts_skip_names_that_are_not_numbers(tmp_path):
    store = PosixStore(tmp_path)
    lay = InstanceLayout.rollout("t", "r")
    lay.claim_attempt(store, "package")
    store.write_bytes(f"{lay.stage('package')}/attempts/notes/readme", b"")
    store.write_bytes(lay.attempt_file("package", 1, "logs/role.out"), b"x")
    assert lay.attempts(store, "package") == [1]


def test_attempts_are_positive_decimal_names_only(tmp_path):
    store = PosixStore(tmp_path)
    lay = InstanceLayout.rollout("t", "r")
    lay.claim_attempt(store, "package")
    # U+0661 is ARABIC-INDIC DIGIT ONE, which int() reads as 1.
    for stray in ("0", "007", "x", "\u0661"):
        (tmp_path / lay.stage("package") / "attempts" / stray).mkdir()
    assert lay.attempts(store, "package") == [1]
    assert lay.claim_attempt(store, "package") == 2


def test_parse_reads_back_every_root_a_constructor_builds():
    rollout, warmer = InstanceLayout.rollout("t", "r1"), InstanceLayout.warmer("t", "w1")
    built = {
        rollout: InstanceKey(ROLLOUTS, "t", "r1", None),
        InstanceLayout.credit("t", "c1"): InstanceKey(CREDITS, "t", "c1", None),
        InstanceLayout.calibration("t", "k1"): InstanceKey(CALIBRATIONS, "t", "k1", None),
        warmer: InstanceKey(WARMERS, "t", "w1", None),
        InstanceLayout.export("t", "x1"): InstanceKey(EXPORTS, "t", "x1", None),
        rollout.grading("g1"): InstanceKey(ROLLOUTS, "t", "r1", "g1"),
        warmer.grading("g1"): InstanceKey(WARMERS, "t", "w1", "g1"),
    }
    assert {layout: InstanceLayout.parse(layout.root) for layout in built} == built
    assert {key.area for key in built.values()} == set(AREAS)
    assert rollout.grading("g1").root == f"{rollout.root}/{GRADINGS_DIR}/g1"


@pytest.mark.parametrize("key", [
    "runs/t", "runs", "runs/t/r1/stages", "runs/t/r1/gradings", "runs/t/r1/gradings/g1/stages",
    "runs/t/r1/other/g1", "oracle/t/x1", "credits/t/c1/gradings/g1", "calibrations/t/k1/gradings/g1",
    "exports/t/x1/gradings/g1", "runs/t/r1/gradings/g1/gradings/g2", "/runs/t/r1", "runs/t/r1/", "runs/t/..",
    "runs/t/r 1", "runs/t/r1\n",
])
def test_parse_refuses_a_key_that_is_no_instances_root(key):
    with pytest.raises(NotAnInstance, match="is not an instance's root"):
        InstanceLayout.parse(key)


@pytest.mark.parametrize("key, why", [
    ("runs/t/index", "index directory"),
    ("warmers/t/index", "index directory"),
    ("warmers/t/index/gradings/g1", "index directory"),
    ("runs/t/r+1", r"holds \+"),
    ("credits/t/c+1", r"holds \+"),
    ("runs/t/r+1/gradings/g1", r"holds \+"),
])
def test_parse_refuses_an_id_that_could_name_another_instances_index_row(key, why):
    # A task's index directory lies beside its instances, and a grading's row joins its
    # instance's id and its own with "+".
    with pytest.raises(NotAnInstance, match=why) as refused:
        InstanceLayout.parse(key)
    assert refused.value.key == key and str(refused.value).startswith(f"{key} is not an instance's root: ")


def test_an_index_row_lies_in_its_tasks_index_directory_apart_from_every_other():
    rollout, warmer = InstanceLayout.rollout("t", "r1"), InstanceLayout.warmer("t", "w1")
    rows = {InstanceLayout.parse(layout.root).index_row
            for layout in (rollout, rollout.grading("g1"), warmer, warmer.grading("g1"),
                           InstanceLayout.rollout("t", "r2"), InstanceLayout.rollout("u", "r1"))}
    assert rows == {"runs/t/index/r1.json", "runs/t/index/r1+g1.json", "warmers/t/index/w1.json",
                    "warmers/t/index/w1+g1.json", "runs/t/index/r2.json", "runs/u/index/r1.json"}
    assert task_index(ROLLOUTS, "t") == f"runs/t/{INDEX_DIR}"
    for area, task in ((ROLLOUTS, "a/b"), ("oracle", "t")):
        with pytest.raises(ValueError):
            task_index(area, task)


def _module_strings(path: Path) -> set[str]:
    """Every string constant of the module at *path*, the literal parts of its f-strings among
    them."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def test_only_the_layout_names_the_store_areas_and_the_gradings_directory():
    # Every other module takes these names from the layout, so renaming one moves every reader.
    own = Path(layout_module.__file__)
    src = own.parents[1]
    names = {*AREAS, GRADINGS_DIR}
    typed = {str(path.relative_to(src)): sorted(names & _module_strings(path))
             for path in sorted(src.rglob("*.py")) if path != own}
    assert {path: found for path, found in typed.items() if found} == {}
