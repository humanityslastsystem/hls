"""The wheel check: every wheel the wheel step left, read with bounded memory before the offline
build, and refused when it is not what it declares."""

from __future__ import annotations

import errno
import io
import os
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

from hls.package import wheel_check
from hls.package.wheel_check import METADATA_MAX_BYTES, WheelCheck, check_wheels, main
from tests.package.fake_tools import declare, lying_wheel, wheel

#: Errors a read of a wheel can end with, each with whether it is the submission's.
ERRNO_SIDES = [(errno.EACCES, True), (errno.EPERM, True), (errno.EIO, False), (errno.ESTALE, False),
               (errno.ENOSPC, False)]
ERRNO_IDS = ["EACCES", "EPERM", "EIO", "ESTALE", "ENOSPC"]


def _zip(entries: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=method) as z:
        for member, data in entries.items():
            z.writestr(member, data)
    return out.getvalue()


def _wheels(tmp_path: Path, **files: bytes) -> Path:
    """A wheels directory holding *files*, each wheel's file name stem to its bytes."""
    out = tmp_path / "wheels"
    out.mkdir(exist_ok=True)
    for stem, data in files.items():
        (out / f"{stem}-py3-none-any.whl").write_bytes(data)
    return out


def test_wheels_that_read_as_they_declare_pass(tmp_path):
    stored = _zip({"st/__init__.py": b"x = 1\n" * 1000, "st-1.dist-info/RECORD": b"st/__init__.py,,\n"},
                  zipfile.ZIP_STORED)
    wheels = _wheels(tmp_path, **{"subm-0.1": wheel("subm", "0.1", {"subm/big.py": "y = 2\n" * 100_000}),
                                  "st-1": stored})
    assert check_wheels(wheels) == WheelCheck()


def test_an_empty_wheels_directory_passes(tmp_path):
    assert check_wheels(_wheels(tmp_path)) == WheelCheck()


@pytest.mark.parametrize("method", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED], ids=["deflated", "stored"])
@pytest.mark.parametrize("declared, found", [(100, "more than the 100"),
                                             (20_000, "10000 bytes, not the 20000")],
                         ids=["fewer-than-it-holds", "more-than-it-holds"])
def test_an_entry_that_lies_about_its_size_is_refused_naming_the_wheel_and_entry(tmp_path, method, declared,
                                                                                 found):
    entries = {"liar/big.py": b"z" * 10_000, "liar-1.dist-info/RECORD": b""}
    data = declare(_zip(entries, method), "liar/big.py", size=declared)
    record = check_wheels(_wheels(tmp_path, **{"liar-1": data}))
    assert record.problem == (f"the entry 'liar/big.py' of the wheel liar-1-py3-none-any.whl unpacks to "
                              f"{found} bytes it declares")


def test_an_entry_that_lies_about_its_crc_is_refused(tmp_path):
    true_crc = zlib.crc32(b"z" * 10_000)
    data = declare(_zip({"liar/big.py": b"z" * 10_000}), "liar/big.py", crc=true_crc ^ 1)
    record = check_wheels(_wheels(tmp_path, **{"liar-1": data}))
    assert record.problem == (f"the entry 'liar/big.py' of the wheel liar-1-py3-none-any.whl unpacks to CRC "
                              f"{true_crc:08x}, not the {true_crc ^ 1:08x} it declares")


def test_a_bomb_is_read_one_byte_past_what_it_declares_and_no_further(tmp_path):
    """A gibibyte of zeros that declares a hundred bytes: the read stops at the 101st, so the
    check neither holds nor spends the gibibyte."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z, z.open("bomb/zeros.bin", "w") as w:
        for _ in range(1024):
            w.write(bytes(2**20))
    wheels = _wheels(tmp_path, **{"bomb-1": declare(out.getvalue(), "bomb/zeros.bin", size=100)})
    unpacked = []
    real = wheel_check._unpacked

    def unpacked_(z, info):
        unpacked.append(real(z, info))
        return unpacked[-1]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wheel_check, "_unpacked", unpacked_)
        record = check_wheels(wheels)
    assert "unpacks to more than the 100 bytes it declares" in (record.problem or "")
    assert unpacked[0][0] == 101


@pytest.mark.parametrize("size, refused", [(METADATA_MAX_BYTES, False), (METADATA_MAX_BYTES + 1, True)],
                         ids=["at-the-limit", "one-byte-past"])
def test_a_metadata_entry_past_its_limit_is_refused(tmp_path, size, refused):
    """pip reads a wheel's ``.dist-info`` entries whole, so none may declare more than 16 MiB."""
    data = _zip({"meta/__init__.py": b"", "meta-1.dist-info/RECORD": bytes(size)})
    record = check_wheels(_wheels(tmp_path, **{"meta-1": data}))
    assert METADATA_MAX_BYTES == 16 * 2**20
    refusal = (f"the entry 'meta-1.dist-info/RECORD' of the wheel meta-1-py3-none-any.whl declares {size} "
               f"bytes, more than the {METADATA_MAX_BYTES} a wheel's metadata may hold")
    assert record.problem == (refusal if refused else None)


def test_wheels_that_declare_more_than_the_install_may_add_are_refused_before_any_is_unpacked(tmp_path,
                                                                                             monkeypatch):
    """Seven entries over two wheels each declare almost 4 GiB, which their bytes do not hold:
    together they pass the 24 GiB the install may add at the seventh, and the check refuses them
    by what they declare, before it unpacks anything."""
    def faked(stem: str, count: int) -> bytes:
        data = _zip({f"{stem}/m{i}.so": f"# {stem} {i}\n".encode() for i in range(count)})
        for i in range(count):
            data = declare(data, f"{stem}/m{i}.so", size=0xFFFFFFFE)
        return data

    wheels = _wheels(tmp_path, **{"aaa-1": faked("aaa", 4), "bbb-1": faked("bbb", 3)})
    monkeypatch.setattr(wheel_check, "_unpacked", lambda z, info: pytest.fail("unpacked before the totals"))
    assert check_wheels(wheels).problem == (
        f"the wheels declare {7 * 0xFFFFFFFE} bytes in all by the entry 'bbb/m2.so' of "
        "bbb-1-py3-none-any.whl, more than the 24 GiB the install may add")


@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-byte-past"])
def test_the_install_limit_counts_every_wheels_entries_together(tmp_path, monkeypatch, over):
    wheels = _wheels(tmp_path, **{"aaa-1": _zip({"aaa/a.py": b"a" * 600}),
                                  "bbb-1": _zip({"bbb/b.py": b"b" * 400})})
    monkeypatch.setattr(wheel_check, "INSTALL_MAX_BYTES", 1000 - over)
    assert (check_wheels(wheels).problem is not None) is bool(over)


_WHOLE = _zip({"junk/a.py": b"a" * 1000, "junk-1.dist-info/RECORD": b"junk/a.py,,\n"})


@pytest.mark.parametrize("data", [b"not a zip at all", _WHOLE[:len(_WHOLE) // 2], b""],
                         ids=["not-a-zip", "truncated", "empty"])
def test_a_file_that_is_not_a_readable_zip_is_refused(tmp_path, data):
    record = check_wheels(_wheels(tmp_path, **{"junk-1": data}))
    assert record.problem == ("the wheel junk-1-py3-none-any.whl is not a zip this check can read: "
                              "BadZipFile: File is not a zip file")


def _broken(how: str) -> bytes:
    """A zip whose entry ``junk/a.py`` does not unpack: its compressed bytes broken, or marked
    encrypted."""
    data = bytearray(_zip({"junk/a.py": b"a" * 1000}))
    local, central = data.index(b"junk/a.py") - 30, data.rindex(b"junk/a.py") - 46
    if how == "broken":
        start = local + 30 + len(b"junk/a.py")
        data[start:start + 8] = b"\xff" * 8
    else:
        data[local + 6] |= 1
        data[central + 8] |= 1
    return bytes(data)


@pytest.mark.parametrize("how, kind", [("broken", "error"), ("encrypted", "RuntimeError")])
def test_an_entry_that_does_not_unpack_is_refused_naming_it(tmp_path, how, kind):
    record = check_wheels(_wheels(tmp_path, **{"junk-1": _broken(how)}))
    assert (record.problem or "").startswith(
        f"the entry 'junk/a.py' of the wheel junk-1-py3-none-any.whl cannot be unpacked: {kind}: ")


def _mixed(entries: dict[str, tuple[bytes, int]]) -> bytes:
    """A zip of *entries*, each member to its bytes and the method it is packed by."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for member, (data, method) in entries.items():
            z.writestr(member, data, compress_type=method)
    return out.getvalue()


def _never_unpacked(z, info):
    pytest.fail(f"{info.filename} was unpacked before every entry's method was checked")


def _packed_by(method: int) -> bytes:
    """A wheel whose module is deflated and whose ``pk/zeros.bin`` is packed by *method*."""
    return _mixed({"pk/__init__.py": (b"", zipfile.ZIP_DEFLATED), "pk/zeros.bin": (bytes(10_000), method)})


@pytest.mark.parametrize("data, method", [
    (_packed_by(zipfile.ZIP_BZIP2), "12 (bzip2)"),
    (_packed_by(zipfile.ZIP_LZMA), "14 (lzma)"),
    (declare(_packed_by(zipfile.ZIP_STORED), "pk/zeros.bin", method=99), "99 (unknown)"),
], ids=["bzip2", "lzma", "unknown"])
def test_an_entry_packed_other_than_stored_or_deflated_is_refused_before_anything_is_unpacked(
        tmp_path, monkeypatch, data, method):
    """zipfile bounds what one read decompresses only for deflate: a bzip2 or lzma entry unpacks
    a whole compressed chunk into memory at once, in this check and in pip's extraction alike, so
    an honest entry of gibibytes would run either out of memory. Every entry must be stored or
    deflated, checked with what the entries declare, before any is decompressed."""
    monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked)
    record = check_wheels(_wheels(tmp_path, **{"pk-1": data}))
    assert record.problem == (f"the entry 'pk/zeros.bin' of the wheel pk-1-py3-none-any.whl is packed by "
                              f"method {method}, not stored or deflated")


def test_a_wheel_of_stored_and_deflated_entries_passes(tmp_path):
    data = _mixed({"ok/__init__.py": (b"x = 1\n" * 100, zipfile.ZIP_STORED),
                   "ok/big.py": (b"y = 2\n" * 10_000, zipfile.ZIP_DEFLATED),
                   "ok-1.dist-info/RECORD": (b"ok/__init__.py,,\n", zipfile.ZIP_STORED)})
    assert check_wheels(_wheels(tmp_path, **{"ok-1": data})) == WheelCheck()


def test_an_entry_packed_otherwise_in_a_later_wheel_is_refused_before_the_first_is_unpacked(tmp_path,
                                                                                          monkeypatch):
    fine = _mixed({"aaa/__init__.py": (b"", zipfile.ZIP_STORED),
                   "aaa/m.py": (b"m = 1\n", zipfile.ZIP_DEFLATED)})
    packed = _mixed({"bbb/__init__.py": (b"", zipfile.ZIP_DEFLATED),
                     "bbb/z.bin": (bytes(1000), zipfile.ZIP_BZIP2)})
    monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked)
    record = check_wheels(_wheels(tmp_path, **{"aaa-1": fine, "bbb-1": packed}))
    assert record.problem == ("the entry 'bbb/z.bin' of the wheel bbb-1-py3-none-any.whl is packed by method "
                              "12 (bzip2), not stored or deflated")


def test_a_zip_whose_own_offsets_make_a_seek_invalid_is_refused_not_a_node_error(tmp_path):
    """A directory offset past the directory makes zipfile seek before the file's start, which
    the kernel refuses with EINVAL: the zip's doing, never the node's."""
    data = bytearray(_zip({"junk/a.py": b"a" * 1000}))
    end = data.rindex(b"PK\x05\x06")
    struct.pack_into("<L", data, end + 16, struct.unpack_from("<L", data, end + 16)[0] + 10_000)
    record = check_wheels(_wheels(tmp_path, **{"junk-1": bytes(data)}))
    assert record.node_error is None
    assert record.problem == ("the entry 'junk/a.py' of the wheel junk-1-py3-none-any.whl cannot be "
                              "unpacked: [Errno 22] Invalid argument")


def test_a_wheel_whose_list_of_entries_passes_its_limit_is_refused(tmp_path, monkeypatch):
    """zipfile reads a wheel's list of entries whole, so its size is bounded before it does."""
    data = _zip({f"many/m{i}.py": b"" for i in range(10)})
    monkeypatch.setattr(wheel_check, "DIRECTORY_MAX_BYTES", 100)
    record = check_wheels(_wheels(tmp_path, **{"many-1": data}))
    assert (record.problem or "").startswith("the wheel many-1-py3-none-any.whl lists its entries in ")
    assert (record.problem or "").endswith(" bytes, more than the 100 allowed")


@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-past"])
def test_a_list_of_entries_at_its_limit_passes_and_one_byte_past_it_is_refused(tmp_path, monkeypatch, over):
    """The bound is on the list's own size, as the zip's end record states it: a list of exactly
    ``DIRECTORY_MAX_BYTES`` passes, and one byte more is refused."""
    data = _zip({f"many/m{i}.py": b"" for i in range(10)})
    listed = zipfile._EndRecData(io.BytesIO(data))[zipfile._ECD_SIZE]
    monkeypatch.setattr(wheel_check, "DIRECTORY_MAX_BYTES", listed - over)
    record = check_wheels(_wheels(tmp_path, **{"many-1": data}))
    if over:
        assert record.problem == (f"the wheel many-1-py3-none-any.whl lists its entries in {listed} bytes, more than "
                                  f"the {listed - 1} allowed")
    else:
        assert record == WheelCheck()


def _paths_refusal(entry: str, wheel: str, cap: int) -> str:
    return (f"the wheels name more than the {cap} paths their install may write, by the entry {entry!r} of the wheel "
            f"{wheel}: every entry, each directory above one, and each script their entry points declare")


@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-past"])
def test_the_paths_every_wheel_names_are_counted_together_before_anything_is_unpacked(tmp_path, monkeypatch,
                                                                                     over):
    """Each entry is a path the install writes, and so is each directory above one, counted
    once across all the wheels: aaa's two entries and its two directories, then bbb's two
    entries, the directory it adds below aaa's and its own, are eight paths. The count is checked
    with what the entries declare, before any is decompressed."""
    wheels = _wheels(tmp_path, **{"aaa-1": _zip({"aaa/x/1.py": b"", "aaa/x/2.py": b""}),
                                  "bbb-1": _zip({"aaa/x/y/3.py": b"", "bbb/b.py": b""})})
    monkeypatch.setattr(wheel_check, "PATHS_MAX", 8 - over)
    if over:
        monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked)
        assert check_wheels(wheels).problem == _paths_refusal("bbb/b.py", "bbb-1-py3-none-any.whl", 7)
    else:
        assert check_wheels(wheels) == WheelCheck()


def test_a_directory_entry_and_one_named_twice_are_each_counted(tmp_path, monkeypatch):
    """Every entry counts, even one naming a path another entry named: pip writes each."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("d/", b"")
        z.writestr("d/f.py", b"")
        with pytest.warns(UserWarning, match="Duplicate name"):
            z.writestr("d/f.py", b"again")
    wheels = _wheels(tmp_path, **{"d-1": out.getvalue()})
    # Three entries and the directory d above two of them.
    monkeypatch.setattr(wheel_check, "PATHS_MAX", 3)
    assert check_wheels(wheels).problem == _paths_refusal("d/f.py", "d-1-py3-none-any.whl", 3)
    monkeypatch.setattr(wheel_check, "PATHS_MAX", 4)
    assert check_wheels(wheels) == WheelCheck()


@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-past"])
def test_each_script_the_entry_points_can_declare_is_a_path(tmp_path, monkeypatch, over):
    """pip writes a script for each console or GUI entry point, and each takes an "=": their
    count is in the entry points' bytes, read once they are proven what they declare."""
    points = "[console_scripts]\n" + "".join(f"s{i} = ep:main\n" for i in range(5)) + "[ep.plugins]\nx = ep:x\n"
    wheels = _wheels(tmp_path, **{"ep-1": _zip({"ep/__init__.py": b"", "ep-1.dist-info/entry_points.txt":
                                                points.encode()})})
    # Two entries, two directories, six "=".
    monkeypatch.setattr(wheel_check, "PATHS_MAX", 10 - over)
    expected = None if not over else _paths_refusal("ep-1.dist-info/entry_points.txt", "ep-1-py3-none-any.whl", 9)
    assert check_wheels(wheels).problem == expected


def test_scripts_of_entry_points_outside_a_dist_info_are_not_counted(tmp_path, monkeypatch):
    wheels = _wheels(tmp_path, **{"ep-1": _zip({"ep/entry_points.txt": b"a=b\n" * 100})})
    monkeypatch.setattr(wheel_check, "PATHS_MAX", 3)
    assert check_wheels(wheels) == WheelCheck()


@pytest.mark.parametrize("depth, refused", [(64, False), (65, True)], ids=["at-the-limit", "one-deeper"])
def test_an_entry_nested_deeper_than_a_submissions_tree_may_be_is_refused(tmp_path, monkeypatch, depth, refused):
    """Counting the directories above an entry costs a step per level, so an entry may be
    nested as deep as a path of the submission's tree, 64 levels, and no deeper."""
    member = "/".join(["d"] * (depth - 1) + ["f.py"])
    wheels = _wheels(tmp_path, **{"d-1": _zip({member: b""})})
    monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked if refused else wheel_check._unpacked)
    expected = (f"the entry {member!r} of the wheel d-1-py3-none-any.whl is nested {depth} levels deep, more than "
                "the 64 allowed")
    assert check_wheels(wheels).problem == (expected if refused else None)


@pytest.mark.parametrize("size, refused", [(255, False), (256, True)], ids=["at-the-limit", "one-byte-past"])
def test_an_entry_holding_a_name_no_filesystem_holds_is_refused(tmp_path, monkeypatch, size, refused):
    """A name over 255 bytes is one pip could not write, and the directories it names are counted
    by name, so it is refused before it is counted."""
    member = "\u00e9" * (size // 2) + "x" * (size % 2) + "/f.py"
    wheels = _wheels(tmp_path, **{"n-1": _zip({member: b""})})
    monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked if refused else wheel_check._unpacked)
    expected = (f"the entry {member!r} of the wheel n-1-py3-none-any.whl holds a name of {size} bytes, more than "
                "the 255 a filesystem holds")
    assert check_wheels(wheels).problem == (expected if refused else None)


def _module_refusal(member: str, wheel: str, size: int) -> str:
    return (f"the entry {member!r} of the wheel {wheel} declares {size} bytes, more than the {128 * 2**20} a "
            "module may hold: pip compiles each module it installs, held whole")


@pytest.mark.parametrize("member", ["mod/big.py", "mod-1.data/scripts/run.py", "mod-1.data/purelib/m.py"])
@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-byte-past"])
def test_a_module_past_its_limit_is_refused_before_anything_is_unpacked(tmp_path, monkeypatch, member, over):
    """pip compiles every module it installs to bytecode in the offline build, holding the
    module and its syntax tree whole, so no ``.py`` entry may declare more than 128 MiB; the
    largest published module is under 1 MiB."""
    size = 128 * 2**20 + over
    data = declare(_zip({member: b"x = 1\n"}), member, size=size)
    if over:
        monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked)
        assert check_wheels(_wheels(tmp_path, **{"mod-1": data})).problem == _module_refusal(
            member, "mod-1-py3-none-any.whl", size)
    else:
        # Past the declared pass, the entry's bytes are not what it declares.
        assert (check_wheels(_wheels(tmp_path, **{"mod-1": data})).problem or "").endswith(
            f"unpacks to 6 bytes, not the {size} bytes it declares")


@pytest.mark.parametrize("member", ["mod/big.so", "mod/big.pyi", "mod/big.py.txt", "mod/big.PY"])
def test_an_entry_pip_does_not_compile_may_declare_more(tmp_path, monkeypatch, member):
    size = 129 * 2**20
    data = declare(_zip({member: b"x = 1\n"}), member, size=size)
    assert (check_wheels(_wheels(tmp_path, **{"mod-1": data})).problem or "").endswith(
        f"unpacks to 6 bytes, not the {size} bytes it declares")


def test_a_reason_names_an_entry_by_its_first_200_characters(tmp_path, monkeypatch):
    member = "d/" + "m" * 250 + ".py"
    size = 129 * 2**20
    data = declare(_zip({member: b"x = 1\n"}), member, size=size)
    monkeypatch.setattr(wheel_check, "_unpacked", _never_unpacked)
    assert check_wheels(_wheels(tmp_path, **{"mod-1": data})).problem == _module_refusal(
        member[:200], "mod-1-py3-none-any.whl", size)


def test_the_paths_cap_is_half_a_million():
    """16 times the zip entries of a set of 107 wheels (31,236), about 13 times its paths with
    their directories and scripts added."""
    assert wheel_check.PATHS_MAX == 500_000


def test_a_wheels_list_of_entries_may_hold_sixteen_mib():
    """zipfile keeps an object per entry: at 16 MiB of the smallest entries it peaks near 200
    MB, and a real wheel's list is a few MB."""
    assert wheel_check.DIRECTORY_MAX_BYTES == 16 * 2**20


def test_a_wheel_its_permissions_keep_out_is_the_submissions(tmp_path):
    wheels = _wheels(tmp_path, **{"subm-0.1": wheel("subm", "0.1")})
    target = wheels / "subm-0.1-py3-none-any.whl"
    target.chmod(0)
    try:
        record = check_wheels(wheels)
    finally:
        target.chmod(0o644)
    assert (record.problem or "").startswith("the wheel subm-0.1-py3-none-any.whl cannot be read: [Errno 13]")


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_wheel_read_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                       submissions):
    """EACCES and EPERM come of the permissions the wheel step left; any other error opening a
    wheel is the node's, recorded with its errno for the stage to retry."""
    wheels = _wheels(tmp_path, **{"subm-0.1": wheel("subm", "0.1")})

    def open_(path, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(path))

    monkeypatch.setattr(wheel_check, "open", open_, raising=False)
    record = check_wheels(wheels)
    reason = f"[Errno {code}] {os.strerror(code)}: '{wheels / 'subm-0.1-py3-none-any.whl'}'"
    name = "subm-0.1-py3-none-any.whl"
    if submissions:
        assert record == WheelCheck(problem=f"the wheel {name} cannot be read: {reason}")
    else:
        assert record == WheelCheck(node_error=f"the wheel {name}: {reason}", node_errno=code)


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_an_entry_read_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                         submissions):
    """An error decompressing an entry names the entry when it is the submission's; any other
    is the node's, recorded with its errno."""
    wheels = _wheels(tmp_path, **{"subm-0.1": wheel("subm", "0.1")})

    def unpacked(z, info):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(wheel_check, "_unpacked", unpacked)
    record = check_wheels(wheels)
    reason = f"[Errno {code}] {os.strerror(code)}"
    name = "subm-0.1-py3-none-any.whl"
    if submissions:
        assert record == WheelCheck(problem=f"the entry 'subm/__init__.py' of the wheel {name} cannot be "
                                            f"unpacked: {reason}")
    else:
        assert record == WheelCheck(node_error=f"the wheel {name}: {reason}", node_errno=code)


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_wheels_directory_listing_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch,
                                                                                     code, submissions):
    wheels = _wheels(tmp_path)

    def listdir(path):
        raise OSError(code, os.strerror(code), str(path))

    monkeypatch.setattr(wheel_check.os, "listdir", listdir)
    record = check_wheels(wheels)
    reason = f"[Errno {code}] {os.strerror(code)}: '{wheels}'"
    if submissions:
        assert record == WheelCheck(problem=f"the wheels directory cannot be listed: {reason}")
    else:
        assert record == WheelCheck(node_error=f"the wheels directory: {reason}", node_errno=code)


@pytest.mark.parametrize("code, data_error", [(None, True), (errno.EINVAL, True), (errno.EACCES, True),
                                              (errno.EPERM, True), (errno.EIO, False), (errno.ESTALE, False),
                                              (errno.ENOSPC, False)])
def test_which_errors_reading_a_zip_are_the_submissions(code, data_error):
    error = OSError("Invalid data stream") if code is None else OSError(code, os.strerror(code))
    assert wheel_check.unreadable_zip(error) is data_error


def test_a_record_is_a_problem_or_a_node_error_and_a_node_error_has_its_errno():
    with pytest.raises(ValueError):
        WheelCheck(node_error="x")
    with pytest.raises(ValueError):
        WheelCheck(node_errno=5)
    with pytest.raises(ValueError):
        WheelCheck(problem="x", node_error="y", node_errno=5)


def test_the_program_writes_its_verdict_and_exits_zero(tmp_path):
    wheels = _wheels(tmp_path, **{"bomb-1.0": lying_wheel()})
    out = tmp_path / "check.json"
    assert main(["--wheels", str(wheels), "--out", str(out)]) == 0
    record = WheelCheck.model_validate_json(out.read_text())
    assert record.problem == ("the entry 'bomb/__init__.py' of the wheel bomb-1.0-py3-none-any.whl unpacks "
                              "to more than the 100 bytes it declares")
    assert sorted(os.listdir(tmp_path)) == ["check.json", "wheels"]


def test_the_check_reads_a_large_entry_a_chunk_at_a_time(tmp_path):
    """An honest entry of 1.5 GiB is read a chunk at a time: the check passes it in a process
    whose address space is held to 1 GiB."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z, z.open("big/zeros.bin", "w") as w:
        for _ in range(1536):
            w.write(bytes(2**20))
    wheels = _wheels(tmp_path, **{"big-1": out.getvalue()})
    record = tmp_path / "check.json"
    limit = ("import resource, runpy, sys; resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30)); "
             "sys.argv = ['wheel_check', '--wheels', sys.argv[1], '--out', sys.argv[2]]; "
             "runpy.run_module('hls.package.wheel_check', run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", limit, str(wheels), str(record)], capture_output=True,
                          text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert done.returncode == 0, done.stderr[-2000:]
    assert WheelCheck.model_validate_json(record.read_text()) == WheelCheck()
