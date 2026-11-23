"""The prefixes' placement: provisioning copies each deep task's prefix and its state from the
repository into the prefix store, and places nothing that does not hash to the digest the corpus
pins (``prefix.place``, ``task.sandbox_place_prefixes``)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from hls.spec.tools import StoreRefused
from tasks.agentic_coding import prefix as pfx
from tasks.agentic_coding import task as task_mod
from tasks.agentic_coding.tests.test_prefix import a_prefix, stage_state

NAME = "git-to-zig"


def _built(manifest: dict) -> pfx.Prefix:
    return pfx.build(**{k: v for k, v in a_prefix().payload().items() if k not in ("digest", "state_manifest")},
                     state_manifest=manifest)


def _held(source: Path, entries, manifest: dict | None = None) -> pfx.Prefix:
    """A small prefix whose state holds *entries*, in *source* as the repository holds one: its
    file, and its state as the archive ``to_repository`` writes. *manifest* replaces the state's
    own manifest in the prefix, for a prefix that records something its archive does not hold."""
    state = stage_state(source.parent / "upper", NAME, entries)
    pfx.to_repository(state, pfx.archive_path(source, NAME))
    return _built(pfx.manifest_of(state) if manifest is None else manifest)


def source_prefix(source: Path, entries=(("src/main.zig", 12), ("bin/run", 3), (".wh.gone", 0))) -> pfx.Prefix:
    """A small prefix and its state in *source*, as the repository holds one, with a symlink."""
    state = stage_state(source.parent / "upper", NAME, entries)
    (state / "latest").symlink_to("bin/run")
    pfx.to_repository(state, pfx.archive_path(source, NAME))
    return _built(pfx.manifest_of(state))


def test_a_prefix_is_placed_once_and_then_held(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    pfx.write(source, prefix)
    assert pfx.place(source, store, NAME, prefix.digest) == "placed"
    assert pfx.load(store, NAME, expect_digest=prefix.digest) == prefix
    assert pfx.prefix_path(store, NAME).read_bytes() == pfx.prefix_path(source, NAME).read_bytes()
    assert os.readlink(pfx.state_path(store, NAME) / "latest") == "bin/run"
    assert pfx.place(source, store, NAME, prefix.digest) == "held"
    assert sorted(p.name for p in store.iterdir()) == [f"{NAME}.prefix.json", f"{NAME}.state"]


def test_a_file_that_does_not_hash_to_the_pin_is_refused_in_one_line_and_not_placed(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    pfx.write(source, prefix)
    pin = "ab" * 32
    with pytest.raises(pfx.PrefixError) as e:
        pfx.place(source, store, NAME, pin)
    message = str(e.value)
    assert "\n" not in message
    assert str(pfx.prefix_path(source, NAME)) in message and prefix.digest in message and pin in message
    assert not store.exists() or not any(store.iterdir())


def test_a_file_edited_after_it_was_cut_is_refused(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    path = pfx.write(source, prefix)
    payload = json.loads(path.read_text())
    payload["digest"] = "cd" * 32
    path.write_text(json.dumps(payload))
    with pytest.raises(pfx.PrefixError, match=f"carries the digest {'cd' * 32} but hashes to {prefix.digest}"):
        pfx.place(source, store, NAME, prefix.digest)
    assert not pfx.prefix_path(store, NAME).exists()


def test_a_missing_file_is_refused_naming_it(tmp_path):
    with pytest.raises(pfx.PrefixError, match=f"{NAME}.prefix.json does not exist"):
        pfx.place(tmp_path / "repo", tmp_path / "store", NAME, "ab" * 32)


def test_a_state_that_does_not_match_its_manifest_is_refused_and_not_placed(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    pfx.write(source, prefix)
    state = pfx.state_path(source.parent / "upper", NAME)
    (state / "src" / "extra.zig").write_text("x")
    pfx.to_repository(state, pfx.archive_path(source, NAME))
    with pytest.raises(pfx.PrefixError, match=r"unexpected \['src/extra.zig'\]") as e:
        pfx.place(source, store, NAME, prefix.digest)
    assert "\n" not in str(e.value)
    assert not store.exists() or not any(store.iterdir())


def test_what_the_store_held_under_the_name_is_moved_aside(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    pfx.write(source, prefix)
    stale = a_prefix(turns=99)
    pfx.write(store, stale)
    stage_state(store, NAME, (("old.txt", 1),))
    assert pfx.place(source, store, NAME, prefix.digest) == "placed"
    assert pfx.load(store, NAME, expect_digest=prefix.digest) == prefix
    aside = sorted(p.name for p in store.iterdir() if ".unusable-" in p.name)
    assert [n.split(".unusable-")[0] for n in aside] == [f"{NAME}.prefix.json", f"{NAME}.state"]


def test_a_replaced_directory_is_marked_again_on_the_copy(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    probe = tmp_path / "probe"
    probe.mkdir()
    try:
        os.setxattr(probe, pfx.OPAQUE_XATTR, b"y")
    except OSError:
        pytest.skip("this filesystem carries no extended attributes")
    state = stage_state(tmp_path / "upper", NAME, (("opt/tool", 4),))
    prefix = _held(source, (("opt/tool", 4),), {**pfx.manifest_of(state), "opt": pfx.OPAQUE_SIZE})
    pfx.write(source, prefix)
    assert pfx.place(source, store, NAME, prefix.digest) == "placed"
    assert pfx.manifest_of(pfx.state_path(store, NAME))["opt"] == pfx.OPAQUE_SIZE


def test_a_store_that_cannot_mark_a_replaced_directory_is_left_as_it_was(tmp_path, monkeypatch):
    source, store = tmp_path / "repo", tmp_path / "store"
    state = stage_state(tmp_path / "upper", NAME, (("opt/tool", 4),))
    prefix = _held(source, (("opt/tool", 4),), {**pfx.manifest_of(state), "opt": pfx.OPAQUE_SIZE})
    pfx.write(source, prefix)

    def refuse(*args):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(pfx.os, "setxattr", refuse)
    with pytest.raises(pfx.PrefixError, match="cannot be marked as replacing the directory below it"):
        pfx.place(source, store, NAME, prefix.digest)
    assert list(store.iterdir()) == []


def _whiteout(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mknod(path, stat.S_IFCHR, os.makedev(0, 0))
    except PermissionError:
        pytest.skip("this kernel lets no unprivileged user make a whiteout")


def test_a_state_with_whiteouts_and_empty_directories_is_made_again_from_its_manifest(tmp_path):
    """A pre-roll's upper as the overlay wrote it: a deleted file as a whiteout, a directory it
    made and left empty, a file, a symlink, a file-form ``.wh.`` marker and a cloned package's
    ``.git``. The repository holds it as the archive ``to_repository`` writes, and placement
    makes the same manifest again: the whiteout a 0:0 device, the directories there."""
    real = tmp_path / "upper" / f"{NAME}{pfx.STATE_SUFFIX}"
    (real / "src").mkdir(parents=True)
    (real / "src" / "main.zig").write_text("const x = 1;")
    (real / "build" / "cache" / "empty").mkdir(parents=True)
    (real / "src" / "link").symlink_to("main.zig")
    (real / ".wh.gone").touch()
    (real / "pkg" / ".git").mkdir(parents=True)
    (real / "pkg" / ".git" / "index.lock").touch()
    _whiteout(real / "src" / "deleted.zig")
    _whiteout(real / "removed-at-top")
    manifest = pfx.manifest_of(real)
    assert manifest["src/deleted.zig"] == manifest["build/cache/empty"] == pfx.MARKER_SIZE
    source, store = tmp_path / "repo", tmp_path / "store"
    pfx.to_repository(real, pfx.archive_path(source, NAME))
    with tarfile.open(pfx.archive_path(source, NAME)) as archive:
        members = {m.name: m for m in archive.getmembers()}
    assert members["src/deleted.zig"].isfile() and members["src/deleted.zig"].size == 0
    assert members["build/cache/empty"].isdir() and members["pkg/.git/index.lock"].isfile()
    prefix = pfx.build(**{k: v for k, v in a_prefix().payload().items() if k not in ("digest", "state_manifest")},
                       state_manifest=manifest)
    pfx.write(source, prefix)
    assert pfx.place(source, store, NAME, prefix.digest) == "placed"
    placed = pfx.state_path(store, NAME)
    assert pfx.manifest_of(placed) == manifest
    for whiteout in ("src/deleted.zig", "removed-at-top"):
        st = os.lstat(placed / whiteout)
        assert stat.S_ISCHR(st.st_mode) and st.st_rdev == os.makedev(0, 0), whiteout
    assert (placed / "build" / "cache" / "empty").is_dir()
    assert (placed / "pkg" / ".git" / "index.lock").is_file()
    assert pfx.load(store, NAME, expect_digest=prefix.digest) == prefix


def test_a_node_that_will_not_make_a_whiteout_places_nothing(tmp_path, monkeypatch):
    source, store = tmp_path / "repo", tmp_path / "store"
    entries = (("src/main.zig", 4), ("src/deleted.zig", 0))
    state = stage_state(tmp_path / "upper", NAME, entries)
    prefix = _held(source, entries, {**pfx.manifest_of(state), "src/deleted.zig": pfx.MARKER_SIZE})
    pfx.write(source, prefix)

    def refuse(*args):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(pfx.os, "mknod", refuse)
    with pytest.raises(pfx.PrefixError, match="deleted.zig cannot be made again as its manifest records it") as e:
        pfx.place(source, store, NAME, prefix.digest)
    assert "\n" not in str(e.value)
    assert list(store.iterdir()) == []


def test_a_state_holding_another_kind_of_entry_has_no_repository_form(tmp_path):
    real = tmp_path / "upper"
    real.mkdir()
    os.mkfifo(real / "pipe")
    with pytest.raises(pfx.PrefixError, match="neither a file, a directory, a symlink nor a whiteout"):
        pfx.to_repository(real, tmp_path / "repo" / "state.tar")


def test_a_state_archive_is_the_same_bytes_wherever_and_whenever_it_is_written(tmp_path):
    """Sorted entries, owner 0, no names, time 0: the committed archive changes only with the
    state."""
    entries = (("b/two", 2), ("a/one", 1), ("c", 0))
    first = stage_state(tmp_path / "one", NAME, entries)
    second = stage_state(tmp_path / "two", NAME, tuple(reversed(entries)))
    os.utime(second / "a" / "one", (1_000_000, 1_000_000))
    pfx.to_repository(first, tmp_path / "first.tar")
    pfx.to_repository(second, tmp_path / "second.tar")
    assert (tmp_path / "first.tar").read_bytes() == (tmp_path / "second.tar").read_bytes()
    with tarfile.open(tmp_path / "first.tar") as archive:
        members = archive.getmembers()
    assert [m.name for m in members] == ["a", "a/one", "b", "b/two", "c"]
    assert {(m.uid, m.gid, m.uname, m.gname, m.mtime) for m in members} == {(0, 0, "", "", 0)}


@pytest.mark.parametrize("name", ["/etc/passwd", "../outside"])
def test_an_archive_entry_outside_the_state_is_refused(tmp_path, name):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = _built({})
    pfx.write(source, prefix)
    with tarfile.open(pfx.archive_path(source, NAME), "w") as archive:
        archive.addfile(tarfile.TarInfo(name))
    with pytest.raises(pfx.PrefixError, match="below the archive's root"):
        pfx.place(source, store, NAME, prefix.digest)
    assert list(store.iterdir()) == []


def test_the_committed_prefixes_place_as_the_corpus_pins_them(tmp_path):
    """The task's own prefixes, as the repository holds them, place into a store and load against
    the digests the corpus pins, their state exactly as each manifest records it."""
    from tasks.agentic_coding.datasets import TerminalBenchDataset

    for name, pin in sorted(TerminalBenchDataset().prefixes.items()):
        assert pfx.place(task_mod.PREFIX_SOURCE_DIR, tmp_path / "store", name, pin) == "placed"
        assert pfx.load(tmp_path / "store", name, expect_digest=pin).digest == pin


def test_the_task_places_only_the_rows_that_pin_a_prefix(tmp_path):
    source, store = tmp_path / "repo", tmp_path / "store"
    prefix = source_prefix(source)
    pfx.write(source, prefix)
    rows = [{"task_name": "cold", "prefix_digest": ""}, {"task_name": NAME, "prefix_digest": prefix.digest}]
    stores = {task_mod.PREFIX_STORE: store}
    assert task_mod.sandbox_place_prefixes(source, rows, stores) == [
        f"agentic-coding-prefixes: the prefix and state of {NAME} ({prefix.digest[:12]}), placed"]
    with pytest.raises(StoreRefused, match=f"{NAME}.prefix.json hashes to {prefix.digest}, not {'ab' * 32}"):
        task_mod.sandbox_place_prefixes(source, [{"task_name": NAME, "prefix_digest": "ab" * 32}], stores)


def test_the_task_places_from_its_own_prefixes_directory():
    assert task_mod.PREFIX_SOURCE_DIR == Path(task_mod.__file__).resolve().parent / "prefixes"


def test_the_stores_role_refuses_a_prefix_it_cannot_place_as_pinned(tmp_path):
    """The role as the provision stage runs it, handed a rows entry with one deep row whose pin
    no file in the repository hashes to: it exits refused, one line naming the file."""
    from hls.provision.role import REFUSED_EXIT
    from hls.provision.sandbox_runtime import RUNTIME_TAG

    stores = {name: str(tmp_path / name) for name in (task_mod.IMAGE_STORE, task_mod.OVERLAY_STORE,
                                                      task_mod.PREFIX_STORE, task_mod.RUNTIME_STORE)}
    python = Path(stores[task_mod.RUNTIME_STORE]) / RUNTIME_TAG / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    entry = tmp_path / "rows"
    (entry / "tenants" / "t0" / "test").mkdir(parents=True)
    pin = "ab" * 32
    (entry / "tenants" / "t0" / "test" / "rows.json").write_text(
        json.dumps([{"task_name": "rs-archive-clone", "prefix_digest": pin}]))
    done = subprocess.run([sys.executable, "-m", "hls.provision.role", "stores", "--task-module",
                           "tasks.agentic_coding.task", "--stores", json.dumps(stores), "--rows", str(entry)],
                          capture_output=True, text=True, timeout=300, cwd=Path(__file__).parents[3])
    assert done.returncode == REFUSED_EXIT, done.stderr
    refusal = done.stderr.strip().splitlines()[-1]
    assert refusal.startswith("refused: ") and "rs-archive-clone.prefix.json" in refusal and pin in refusal
    assert not (Path(stores[task_mod.PREFIX_STORE]) / "rs-archive-clone.prefix.json").exists()
