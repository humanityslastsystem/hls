from __future__ import annotations

import pytest

from hls.provision.model import (ModelRevision, Provisioned, Resolved, RowsRef, ToolRef, ToolSource,
                                 WeightsRef, is_dev_tool_key, rows_key, tool_key, weights_key)
from hls.store.base import check_segment

SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "a" * 64


def test_a_weights_key_is_one_store_segment_and_names_the_commit():
    key = weights_key("Qwen/Qwen3.6-27B", SHA)
    assert key == f"Qwen+Qwen3.6-27B@{SHA}" and check_segment(key) == key
    assert weights_key("gpt2", SHA) == f"gpt2@{SHA}"


@pytest.mark.parametrize("repo, revision", [
    ("Qwen/Qwen3.6-27B", "main"),            # a branch name moves, so it never keys an entry
    ("Qwen/Qwen3.6-27B", SHA.upper()),
    ("Qwen/Qwen3.6-27B", SHA[:12]),
    ("a/b/c", SHA),
    ("../etc", SHA),
    ("org/name\n", SHA),
    ("org+name/x", SHA),
])
def test_bad_repos_and_revisions_are_refused(repo, revision):
    with pytest.raises(ValueError):
        weights_key(repo, revision)


def test_rows_and_tool_keys():
    assert rows_key("0123456789abcdef") == "rows-0123456789abcdef"
    with pytest.raises(ValueError):
        rows_key("short")
    with pytest.raises(ValueError, match="not a rows key"):
        rows_key("0123")                     # hex, but not the 16 digits of a digest
    with pytest.raises(ValueError, match="not a rows key"):
        rows_key("0123456789abcdef0")
    assert tool_key("browsecomp", "1234abcd", "full") == "browsecomp-1234abcd-full"
    assert tool_key("browsecomp", "1234abcd", "dev", "0123456789ab") == "browsecomp-1234abcd-dev-0123456789ab"
    with pytest.raises(ValueError, match="partition digest"):
        tool_key("browsecomp", "1234abcd", "dev")
    with pytest.raises(ValueError, match="partition digest"):
        tool_key("browsecomp", "1234abcd", "full", "0123456789ab")
    for bad in ("XYZ", "0123456789a", "0123456789abc", "0123456789AB"):
        with pytest.raises(ValueError, match="not a partition digest"):
            tool_key("browsecomp", "1234abcd", "dev", bad)
    with pytest.raises(ValueError):
        tool_key("browse/comp", "1234abcd", "full")          # one store segment each
    with pytest.raises(ValueError):
        tool_key("browsecomp", "../1234", "full")
    with pytest.raises(ValueError):
        tool_key("browsecomp", "", "full")                   # "browsecomp--full" is a segment, but names nothing


def test_a_weights_ref_must_carry_its_own_key():
    WeightsRef(repo="org/m", revision=SHA, key=weights_key("org/m", SHA), entry_sha256=DIGEST, disk_bytes=4096)
    with pytest.raises(ValueError, match="is not the key"):
        WeightsRef(repo="org/m", revision=SHA, key=weights_key("org/other", SHA), entry_sha256=DIGEST,
                   disk_bytes=4096)


def test_every_entry_of_a_provision_record_states_the_disk_its_copy_takes():
    """A record without it cannot size the stages that copy the entry; a negative one is no
    size."""
    with pytest.raises(ValueError, match="disk_bytes"):
        WeightsRef(repo="org/m", revision=SHA, key=weights_key("org/m", SHA), entry_sha256=DIGEST)
    with pytest.raises(ValueError, match="disk_bytes"):
        RowsRef(key="rows-" + "0" * 16, entry_sha256=DIGEST, tenants=["t0"], disk_bytes=-1)


def test_resolved_ties_rows_to_tenants_and_tools_to_rows():
    m = ModelRevision(repo="org/m", revision=SHA)
    Resolved(weights=m, rows_key=None, tenants=[], tools=None)
    Resolved(weights=m, rows_key="rows-" + "0" * 16, tenants=["t0"],
             tools=ToolSource(name="fake", identity="fake-v1"))
    with pytest.raises(ValueError, match="at least one tenant"):
        Resolved(weights=m, rows_key=None, tenants=["t0"], tools=None)
    with pytest.raises(ValueError, match="needs the rows"):
        Resolved(weights=m, rows_key=None, tenants=[], tools=ToolSource(name="fake", identity="v"))
    # A key that is not a rows key is refused even when it comes with tenants.
    for bad in ("bad", "rows-" + "0" * 15, "rows-" + "0" * 17, "rows-" + "G" * 16):
        with pytest.raises(ValueError, match="not a rows key"):
            Resolved(weights=m, rows_key=bad, tenants=["t0"], tools=None)


def _tool(variant, *, key=None, dev_digest=None, name="fake", identity="v1"):
    return ToolRef(name=name, identity=identity, variant=variant, dev_digest=dev_digest,
                   key=key or tool_key(name, identity, variant, dev_digest), entry_sha256=DIGEST, disk_bytes=4096)


def test_provisioned_tools_come_in_pairs_with_their_rows():
    rows = RowsRef(key="rows-" + "0" * 16, entry_sha256=DIGEST, tenants=["t0"], disk_bytes=4096)
    full = _tool("full")
    dev = _tool("dev", dev_digest="0123456789ab")
    assert (full.key, dev.key) == ("fake-v1-full", "fake-v1-dev-0123456789ab")
    Provisioned(weights=None, rows=rows, tools=[full, dev])
    with pytest.raises(ValueError, match="both variants"):
        Provisioned(weights=None, rows=rows, tools=[full])
    with pytest.raises(ValueError, match="needs the rows"):
        Provisioned(weights=None, rows=None, tools=[full, dev])


def test_a_tool_ref_must_carry_its_own_key():
    with pytest.raises(ValueError, match="is not the key"):          # a full key on a dev ref
        _tool("dev", key="fake-v1-full", dev_digest="0123456789ab")
    with pytest.raises(ValueError, match="is not the key"):          # another partition's digest
        _tool("dev", key="fake-v1-dev-aaaaaaaaaaaa", dev_digest="0123456789ab")
    with pytest.raises(ValueError, match="is not the key"):          # another bundle's key
        _tool("full", key="other-v1-full")
    with pytest.raises(ValueError, match="is not the key"):          # another identity's key
        _tool("full", key="fake-v2-full")
    with pytest.raises(ValueError, match="partition digest"):        # a dev ref names its partition
        _tool("dev", key="fake-v1-dev-0123456789ab")
    with pytest.raises(ValueError, match="partition digest"):        # a full ref names none
        ToolRef(name="fake", identity="v1", variant="full", dev_digest="0123456789ab",
                key="fake-v1-full", entry_sha256=DIGEST, disk_bytes=4096)


def test_a_dev_bundle_key_is_told_from_a_full_one_by_its_suffix():
    assert is_dev_tool_key(tool_key("tiny", "v1", "dev", "0123456789ab"))
    assert not is_dev_tool_key(tool_key("tiny", "v1", "full"))
    # A name or identity that looks like the other variant does not change what the suffix says.
    assert is_dev_tool_key(tool_key("a", "x-full", "dev", "0123456789ab"))
    assert not is_dev_tool_key(tool_key("a", "x-dev-0123456789ab", "full"))
    assert not is_dev_tool_key("dev-0123456789ab") and not is_dev_tool_key("tiny-v1-dev-0123456789")
