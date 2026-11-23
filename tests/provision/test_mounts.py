from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, tool_key, weights_key
from hls.provision.mounts import (OFFLINE_ENV, ROLE_UNITS, contestant_mount_problem, copied_entries, copies_gib,
                                  data_mounts)
from hls.substrate.container import Mount
from tests.provision.tasks import SHA

#: The ENTRY.json digest provision recorded for each entry, one per entry so a mix-up shows.
WEIGHTS_SHA, ROWS_SHA, FULL_SHA, DEV_SHA = "a" * 64, "b" * 64, "c" * 64, "d" * 64
ROWS = "rows-0123456789abcdef"
DEV = "0123456789ab"
#: The disk each entry's node-local copy takes, as provision recorded it; one size per entry, the
#: weights the size of multitenant_slo's.
WEIGHTS_BYTES, ROWS_BYTES, FULL_BYTES, DEV_BYTES = 55_600_000_000, 3 << 30, (5 << 30) + 1, 7 << 30


def _provisioned(*, tools: bool = True, weights: bool = True, rows: bool = True) -> Provisioned:
    tool_refs = [ToolRef(name="tiny", identity="v1", variant=variant, dev_digest=digest,
                         key=tool_key("tiny", "v1", variant, digest), entry_sha256=sha, disk_bytes=size)
                 for variant, digest, sha, size in (("full", None, FULL_SHA, FULL_BYTES), ("dev", DEV, DEV_SHA,
                                                                                           DEV_BYTES))] if tools else []
    return Provisioned(
        weights=WeightsRef(repo="org/model", revision=SHA, key=weights_key("org/model", SHA),
                           entry_sha256=WEIGHTS_SHA, disk_bytes=WEIGHTS_BYTES) if weights else None,
        rows=RowsRef(key=ROWS, entry_sha256=ROWS_SHA, tenants=["chat", "api"],
                     disk_bytes=ROWS_BYTES) if rows else None,
        tools=tool_refs)


class _Local:
    """Node-local copies at <root>/<kind>/<key>, noting which entries were asked for, each
    with the digest it was pinned to."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.asked: list[tuple[str, str, str]] = []

    def __call__(self, kind: str, key: str, entry_sha256: str) -> Path:
        self.asked.append((kind, key, entry_sha256))
        return self.root / kind / key


def _mounts(role: str, provisioned: Provisioned, local: _Local, *, weights: bool = True):
    """``data_mounts`` of *role*, its stage declaring the copies the role makes."""
    declared = copied_entries((role,), provisioned, weights=weights)
    return data_mounts(role, provisioned, local, declared=declared, weights=weights)


def _as_tuples(mounts) -> list[tuple[str, str, bool]]:
    return [(str(m.source), m.target, m.read_only) if isinstance(m, Mount)
            else (m["source"], m["target"], m["read_only"]) for m in mounts]


def _expected_mounts(root: Path) -> dict[str, list[tuple[str, str, bool]]]:
    """Every role's mounts of ``_provisioned()``'s entries, node-local under *root*, all
    read-only: the weights first, then each tenant's rows (dev before test), then the bundle."""
    weights = (str(root / "weights" / weights_key("org/model", SHA)), "/data/weights", True)
    tenants = root / "rows" / ROWS / "tenants"

    def rows(*splits: str) -> list[tuple[str, str, bool]]:
        return [(str(tenants / t / split), f"/data/tenants/{t}/{split}", True)
                for t in ("chat", "api") for split in splits]

    dev_bundle = (str(root / "tools" / tool_key("tiny", "v1", "dev", DEV)), "/data/tools/tiny", True)
    full_bundle = (str(root / "tools" / tool_key("tiny", "v1", "full")), "/data/tools/tiny", True)
    return {
        # The build sandbox and the agent see the dev rows and the dev bundle.
        "build": [weights, *rows("dev"), dev_bundle],
        "agent": [weights, *rows("dev"), dev_bundle],
        # The served artifact sees the dev rows its build sandbox had, and no bundle.
        "artifact": [weights, *rows("dev")],
        # The driver serves both splits and runs the tools on the full bundle.
        "driver": [weights, *rows("dev", "test"), full_bundle],
        "reference": [weights, full_bundle],
        # The oracle fill answers the test rows, and the dev rows for the dev reference.
        "oracle": [weights, *rows("dev", "test"), full_bundle],
        # A calibration's capture and score roles drive and key the test split's sessions.
        "calibration": [weights, *rows("test"), full_bundle],
    }


@pytest.mark.parametrize("role", sorted(ROLE_UNITS))
def test_each_role_mounts_exactly_its_entries_and_reads_the_weights_offline(tmp_path, role):
    mounts, env = _mounts(role, _provisioned(), _Local(tmp_path))
    assert _as_tuples(mounts) == _expected_mounts(tmp_path)[role]
    assert env == {**OFFLINE_ENV, "HLS_WEIGHTS_DIR": "/data/weights"}


def test_the_oracle_fill_mounts_what_the_reference_does_and_the_rows(tmp_path):
    prov = _provisioned()
    reference, ref_env = _mounts("reference", prov, _Local(tmp_path))
    oracle, env = _mounts("oracle", prov, _Local(tmp_path))
    rows = tmp_path / "rows" / ROWS / "tenants"
    assert env == ref_env
    # The weights, then the rows where every role has them, the dev rows for the dev reference
    # and the test rows for the answers, then the full tools.
    same = _as_tuples(reference)
    assert _as_tuples(oracle) == same[:1] + [
        (str(rows / t / split), f"/data/tenants/{t}/{split}", True)
        for t in ("chat", "api") for split in ("dev", "test")] + same[1:]


def test_the_calibration_roles_mount_what_the_oracle_fill_does_but_the_dev_rows(tmp_path):
    # A calibration's capture and score roles drive and key the test split's sessions.
    prov = _provisioned()
    calibration, env = _mounts("calibration", prov, _Local(tmp_path))
    oracle, oracle_env = _mounts("oracle", prov, _Local(tmp_path))
    no_dev = [m for m in _as_tuples(oracle) if not m[1].endswith("/dev")]
    assert (_as_tuples(calibration), env) == (no_dev, oracle_env)


def test_no_role_but_the_driver_the_oracle_fill_and_calibration_sees_test_rows_and_ids_never_mount(tmp_path):
    prov = _provisioned()
    for role in ROLE_UNITS:
        mounts, _ = _mounts(role, prov, _Local(tmp_path))
        test_targets = [m.target for m in mounts if m.target.endswith("/test")]
        assert bool(test_targets) == (role in ("driver", "oracle", "calibration"))
        assert all(m.read_only for m in mounts)
        assert all(m.source.name != "ids.json" and m.source != tmp_path / "rows" / ROWS for m in mounts)
        # A bundle built from the test rows reaches only the roles that may see them.
        full = [m for m in mounts if m.source.name == tool_key("tiny", "v1", "full")]
        assert bool(full) == (role in ("driver", "reference", "oracle", "calibration"))


def test_only_the_entries_a_role_mounts_are_copied_to_the_node_each_pinned_to_its_recorded_digest(tmp_path):
    prov = _provisioned()
    weights = ("weights", prov.weights.key, WEIGHTS_SHA)
    local = _Local(tmp_path)
    _mounts("reference", prov, local)
    assert local.asked == [weights, ("tools", "tiny-v1-full", FULL_SHA)]
    local = _Local(tmp_path)
    _mounts("artifact", prov, local)
    assert local.asked == [weights, ("rows", ROWS, ROWS_SHA)]
    local = _Local(tmp_path)
    _mounts("build", prov, local)
    assert local.asked == [weights, ("rows", ROWS, ROWS_SHA), ("tools", f"tiny-v1-dev-{DEV}", DEV_SHA)]
    local = _Local(tmp_path)
    _mounts("oracle", prov, local)
    assert local.asked == [weights, ("rows", ROWS, ROWS_SHA), ("tools", "tiny-v1-full", FULL_SHA)]


CONTESTANT_ROLES = ("agent", "artifact", "build")


def _where(root: Path, source: Path) -> tuple[str, str, PurePosixPath]:
    """The entry kind, key and path inside the entry of a mount source under *root*."""
    kind, key, *rest = source.relative_to(root).parts
    return kind, key, PurePosixPath(*rest)


@pytest.mark.parametrize("role", sorted(ROLE_UNITS))
def test_the_wall_at_container_start_admits_every_mount_a_contestant_role_gets_and_no_other(tmp_path, role):
    mounts, _ = _mounts(role, _provisioned(), _Local(tmp_path))
    refused = [m.target for m in mounts if contestant_mount_problem(*_where(tmp_path, m.source)) is not None]
    test_rows = ["/data/tenants/chat/test", "/data/tenants/api/test", "/data/tools/tiny"]
    expected = {"driver": test_rows, "reference": ["/data/tools/tiny"], "oracle": test_rows, "calibration": test_rows}
    assert refused == expected.get(role, [])
    assert (role in CONTESTANT_ROLES) == (role not in expected)


def test_a_task_without_rows_or_tools_mounts_only_the_weights(tmp_path):
    mounts, env = _mounts("driver", _provisioned(rows=False, tools=False), _Local(tmp_path))
    assert [m.target for m in mounts] == ["/data/weights"]
    assert env == {**OFFLINE_ENV, "HLS_WEIGHTS_DIR": "/data/weights"}


def test_a_role_that_needs_weights_that_were_not_provisioned_is_refused(tmp_path):
    with pytest.raises(ValueError, match="which were not provisioned"):
        data_mounts("artifact", _provisioned(weights=False), _Local(tmp_path), declared={})
    with pytest.raises(ValueError, match="unknown role"):
        data_mounts("sidecar", _provisioned(), _Local(tmp_path), declared={})


@pytest.mark.parametrize("roles, entries", [
    (("agent",), ("weights", "rows", "dev")),
    (("build",), ("weights", "rows", "dev")),
    (("artifact", "driver"), ("weights", "rows", "full")),
    (("reference",), ("weights", "full")),
    (("oracle",), ("weights", "rows", "full")),
], ids=["agent", "build", "drive", "reference", "oracle"])
def test_a_stages_copies_count_each_entry_its_roles_mount_once(tmp_path, roles, entries):
    """A stage's roles mount node-local copies of cache entries, each entry copied once
    (``materialize``); ``copied_entries`` names those copies with the bytes the provision record
    states of each (``disk_bytes``), and ``copies_gib`` is the disk they take, rounded up to a
    whole GiB. It names exactly the entries ``data_mounts`` asks the node for, given them."""
    provisioned = _provisioned()
    sizes = {"weights": WEIGHTS_BYTES, "rows": ROWS_BYTES, "full": FULL_BYTES, "dev": DEV_BYTES}
    copies = copied_entries(roles, provisioned)
    assert sorted(copies.values()) == sorted(sizes[e] for e in entries)
    assert copies_gib(copies) == -(-sum(sizes[e] for e in entries) // 2**30)
    local = _Local(tmp_path)
    for role in roles:
        data_mounts(role, provisioned, local, declared=copies)
    assert {f"{kind}/{key}" for kind, key, _ in local.asked} == set(copies)


def test_a_role_that_mounts_no_weights_does_not_count_them():
    provisioned = _provisioned()
    assert copied_entries(("reference",), provisioned, weights=False) == {
        f"tools/{tool_key('tiny', 'v1', 'full')}": FULL_BYTES}
    assert copied_entries(("agent",), _provisioned(tools=False, rows=False)) == {
        f"weights/{weights_key('org/model', SHA)}": WEIGHTS_BYTES}


def test_a_copy_its_stage_did_not_declare_room_for_is_refused_before_any_is_made(tmp_path):
    """The node's start check passed the disk of the copies the plan declared, so a role copies
    nothing when one of its entries is not among them, or takes more than declared (an entry
    published again since the plan)."""
    provisioned = _provisioned()
    declared = copied_entries(("reference",), provisioned)
    full = f"tools/{tool_key('tiny', 'v1', 'full')}"
    local = _Local(tmp_path)
    with pytest.raises(ValueError, match=re.escape(f"the plan made no room on the node for a copy of {full}: plan the "
                                                   "instance again")):
        data_mounts("reference", provisioned, local, declared={k: v for k, v in declared.items() if k != full})
    weights = f"weights/{weights_key('org/model', SHA)}"
    with pytest.raises(ValueError, match=re.escape(f"no room on the node for a copy of {weights}, {full}: plan")):
        data_mounts("reference", provisioned, local, declared={})
    smaller = {**declared, full: FULL_BYTES - 1}
    with pytest.raises(ValueError, match=re.escape(f"the copy of {full} takes {FULL_BYTES} bytes, more than the "
                                                   f"{FULL_BYTES - 1} the plan made room for")):
        data_mounts("reference", provisioned, local, declared=smaller)
    assert local.asked == []
    # A declaration with room to spare, and one naming entries other roles of the stage copy, admit it.
    roomy = {**copied_entries(ROLE_UNITS, provisioned), full: FULL_BYTES + 1}
    data_mounts("reference", provisioned, local, declared=roomy)
    assert [kind for kind, _, _ in local.asked] == ["weights", "tools"]
