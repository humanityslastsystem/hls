"""The integrity check: a submission image's system paths against its base's, byte for byte, and
which differences the submission's wheels account for."""

from __future__ import annotations

import errno
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from hls.package import integrity
from hls.package.integrity import (BUILD_RECORDS, DIFFERS_EXIT, HostTree, IntegrityRecord, check, destination,
                                   install_targets, script_names, through_links, write_target)
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from tests.package.fake_tools import BASE_TREE, lay_out, wheel

# Links of a base that a wheel's install can reach: a directory of the venv that leads into
# /usr, and a script of the venv whose target is missing.
LINKED_BASE = {**BASE_TREE, "usr/share": ["dir", 0o755], "opt/venv/share": ["link", "/usr/share"],
               "opt/venv/bin/tool": ["link", "/usr/bin/tool-real"]}


def _changed(tree: dict, changes: dict | None) -> dict:
    out = dict(tree)
    for path, entry in (changes or {}).items():
        if entry is None:
            for p in [p for p in out if p == path or p.startswith(path + "/")]:
                del out[p]
        else:
            out[path] = entry
    return out


def _trees(tmp_path: Path, changes: dict | None = None, *, base: dict = BASE_TREE) -> tuple[Path, Path]:
    """The base (*base*), and an image that is it with *changes* (path to an entry, None to remove)."""
    lay_out(tmp_path / "base", base)
    lay_out(tmp_path / "image", _changed(base, changes))
    return tmp_path / "base", tmp_path / "image"


#: The venv's site-packages, from an image's root.
SITE = "opt/venv/lib/python3.12/site-packages"


def _wheels(tmp_path: Path, *, installed: bool = True, **entries: dict[str, str]) -> Path:
    """A wheels directory with one wheel per keyword: its file name stem to its extra entries.
    With *installed*, each is installed in the image at ``tmp_path / "image"``, as pip leaves
    one: its ``.dist-info`` directory in the venv's site-packages."""
    out = tmp_path / "wheels"
    out.mkdir(exist_ok=True)
    for stem, extra in entries.items():
        name, version = stem.split("-")[:2]
        (out / f"{stem}-py3-none-any.whl").write_bytes(wheel(name, version, extra))
        if installed:
            (tmp_path / "image" / SITE / f"{name}-{version}.dist-info").mkdir()
    return out


def _check(base: Path, image: Path, wheels: Path) -> IntegrityRecord:
    """The check of the extracted trees *base* and *image*."""
    return check(HostTree(base), HostTree(image), wheels)


def _found(record: IntegrityRecord) -> list[tuple[str, str]]:
    return [(d.path, d.change) for d in record.listed]


def test_an_image_whose_system_paths_are_the_bases_has_no_difference(tmp_path):
    base, image = _trees(tmp_path, {"workspace": ["dir", 0o755], "opt/venv/bin/tool": ["file", "x", 0o755],
                                    ".singularity.d/runscript": ["file", "#!/bin/sh\nexec x\n", 0o755],
                                    ".singularity.d/Singularity": ["file", "Bootstrap: localimage\n", 0o644],
                                    ".singularity.d/labels.json": ["file", "{\"a\": 1}\n", 0o644],
                                    ".singularity.d/bootstrap_history": ["dir", 0o755],
                                    ".singularity.d/bootstrap_history/Singularity0": ["file", "Bootstrap: x\n", 0o644]})
    record = _check(base, image, _wheels(tmp_path))
    assert record.differences == 0 and record.listed == []
    # The lib names come from the trees themselves; /opt and /workspace are not compared.
    assert record.compared == [".singularity.d", "bin", "etc", "lib", "lib64", "sbin", "usr"]
    assert record.entries == sum(1 for p in BASE_TREE if p.split("/")[0] in record.compared)
    assert BUILD_RECORDS == {".singularity.d/runscript", ".singularity.d/Singularity",
                             ".singularity.d/labels.json"}


@pytest.mark.parametrize("changes, expected", [
    ({"usr/bin/sh": ["file", "#!evil\n", 0o755]}, [("/usr/bin/sh", "content")]),
    ({"usr/bin/sh": ["file", "#!dash\n", 0o4755]}, [("/usr/bin/sh", "mode")]),
    ({"usr/bin/python3": ["link", "/tmp/py"]}, [("/usr/bin/python3", "target")]),
    ({"bin": ["dir", 0o755], "bin/sh": ["file", "#!evil\n", 0o755]}, [("/bin", "type"), ("/bin/sh", "added")]),
    ({"etc/os-release": None}, [("/etc/os-release", "removed")]),
    ({"usr/share": ["dir", 0o755], "usr/share/x": ["file", "x", 0o644]},
     [("/usr/share", "added"), ("/usr/share/x", "added")]),
    ({"libx32": ["dir", 0o755]}, [("/libx32", "added")]),
    ({"sbin": ["link", "usr/bin"]}, [("/sbin", "target")]),
    ({".singularity.d/env/99-hook.sh": ["file", "exit 1\n", 0o644]}, [("/.singularity.d/env/99-hook.sh", "added")]),
    ({".singularity.d/actions/exec": ["file", "#!/bin/sh\nexit 1\n", 0o755]},
     [("/.singularity.d/actions/exec", "content")]),
], ids=["content", "mode", "link-target", "type", "removed", "added", "new-lib", "top-link", "env-script", "action"])
def test_every_kind_of_change_to_a_system_path_is_found(tmp_path, changes, expected):
    base, image = _trees(tmp_path, changes)
    record = _check(base, image, _wheels(tmp_path))
    assert _found(record) == expected
    assert record.differences == record.unattributed == len(expected)


def test_same_size_contents_are_compared_by_their_bytes(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!bash\n", 0o755]})
    assert _found(_check(base, image, _wheels(tmp_path))) == [("/usr/bin/sh", "content")]


def test_where_a_wheel_installs_in_the_venv(tmp_path):
    assert destination("subm/__init__.py", "subm") == "/opt/venv/lib/python3.12/site-packages/subm/__init__.py"
    assert destination("subm-0.1.data/scripts/run", "subm") == "/opt/venv/bin/run"
    assert destination("subm-0.1.data/data/bin/sh", "subm") == "/opt/venv/bin/sh"
    assert destination("subm-0.1.data/headers/x.h", "subm") == "/opt/venv/include/site/python3.12/subm/x.h"
    assert destination("subm-0.1.data/platlib/m.so", "subm") == "/opt/venv/lib/python3.12/site-packages/m.so"
    assert destination("subm-0.1.data/data/../../../usr/bin/sh", "subm") == "/usr/bin/sh"
    assert destination("subm/", "subm") is None


def test_a_destination_is_followed_through_the_links_the_base_holds(tmp_path):
    base, _ = _trees(tmp_path)
    assert through_links(HostTree(base), "/opt/venv/bin/python") == "/usr/bin/python3.12"
    assert through_links(HostTree(base), "/bin/sh") == "/usr/bin/sh"
    assert through_links(HostTree(base), "/lib64/ld-linux-x86-64.so.2") == "/usr/lib/libc.so.6"
    assert through_links(HostTree(base), "/opt/venv/bin/new") == "/opt/venv/bin/new"
    assert through_links(HostTree(base), "/opt/venv/bin/../../../usr/bin/sh") == "/usr/bin/sh"
    os.symlink("loop-b", base / "loop-a")
    os.symlink("loop-a", base / "loop-b")
    assert through_links(HostTree(base), "/loop-a/x") is None


def test_a_wheel_whose_data_lands_in_the_venv_accounts_for_no_system_path(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/bin/sh": "#!evil\n"}}))
    assert [d.wheels for d in record.listed] == [[]] and record.unattributed == 1


def test_a_wheel_that_climbs_out_of_its_scheme_accounts_for_what_it_wrote(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755],
                                    "usr/share": ["dir", 0o755], "usr/share/evil": ["dir", 0o755],
                                    "usr/share/evil/f": ["file", "x", 0o644]})
    wheels = _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n",
                                               "subm-0.1.data/data/../../../usr/share/evil/f": "x"},
                                  "httpx-0.27.0": {}})
    record = _check(base, image, wheels)
    assert record.differences == 4 and record.unattributed == 0
    assert {d.path: d.wheels for d in record.listed} == {p: ["subm-0.1-py3-none-any.whl"] for p in
                                                         ("/usr/bin/sh", "/usr/share", "/usr/share/evil",
                                                          "/usr/share/evil/f")}


def test_pip_replaces_a_last_link_that_leads_somewhere_and_writes_through_one_that_does_not(tmp_path):
    base, _ = _trees(tmp_path, base=LINKED_BASE)
    # The venv's python leads to /usr/bin/python3.12: pip removes the link and writes in its place.
    assert write_target(HostTree(base), "/opt/venv/bin", "python") == "/opt/venv/bin/python"
    # The venv's tool leads to a file the base lacks: pip's write creates it.
    assert write_target(HostTree(base), "/opt/venv/bin", "tool") == "/usr/bin/tool-real"
    assert write_target(HostTree(base), "/usr/bin", "new") == "/usr/bin/new"


def test_a_wheel_replacing_a_link_of_the_venv_accounts_for_nothing_the_link_leads_to(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/python3.12": ["file", "evil\n", 0o755]})
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/scripts/python": "evil\n"}}))
    assert [(d.path, d.wheels) for d in record.listed] == [("/usr/bin/python3.12", [])]


def test_a_wheel_writing_through_the_bases_links_accounts_for_where_the_writes_land(tmp_path):
    base, image = _trees(tmp_path, {"usr/share/evil": ["dir", 0o755], "usr/share/evil/f": ["file", "x", 0o644],
                                    "usr/bin/tool-real": ["file", "y", 0o755]}, base=LINKED_BASE)
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/share/evil/f": "x",
                                                                  "subm-0.1.data/scripts/tool": "y"}}))
    assert {d.path: d.wheels for d in record.listed} == {p: ["subm-0.1-py3-none-any.whl"] for p in
                                                         ("/usr/bin/tool-real", "/usr/share/evil",
                                                          "/usr/share/evil/f")}
    assert record.unattributed == 0


@pytest.mark.parametrize("changes, path", [
    ({"usr/share": ["dir", 0o700]}, "/usr/share"),
    ({"usr/share/evil": ["dir", 0o755], "usr/share/evil/f": ["link", "/etc/passwd"]}, "/usr/share/evil/f"),
    ({"usr/share/evil": ["dir", 0o755], "usr/share/evil/f": ["dir", 0o755]}, "/usr/share/evil/f"),
    ({"usr/share": None}, "/usr/share"),
    ({"usr/bin/tool-real": ["link", "/etc/passwd"]}, "/usr/bin/tool-real"),
    ({"usr/share": ["file", "z", 0o644]}, "/usr/share"),
], ids=["dir-above-mode", "link-where-a-file-goes", "dir-where-a-file-goes", "removed", "link-where-a-write-lands",
        "file-where-the-base-held-a-dir"])
def test_a_change_pip_cannot_make_where_a_wheel_writes_is_unaccounted(tmp_path, changes, path):
    base, image = _trees(tmp_path, changes, base=LINKED_BASE)
    # The wheel writes /usr/share/evil/f, /usr/bin/tool-real and, climbing out of its scheme, /usr/share.
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/share/evil/f": "x",
                                                                  "subm-0.1.data/scripts/tool": "y",
                                                                  "subm-0.1.data/data/../../../usr/share": "z"}}))
    [found] = [d for d in record.listed if d.path == path]
    assert found.wheels == []


def test_a_file_where_the_base_held_a_link_is_a_write_of_pips(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/python3": ["file", "evil\n", 0o755]})
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/../../../usr/bin/python3": "e"}}))
    assert [(d.path, d.change, d.wheels) for d in record.listed] == [
        ("/usr/bin/python3", "type", ["subm-0.1-py3-none-any.whl"])]


def test_entry_point_scripts_are_read_as_pip_reads_them(tmp_path):
    text = ("# comment\n[console_scripts]\n  serve = subm:main  \n../../../usr/bin/sh=subm:sh\nno-equals\n"
            "[gui_scripts]\n/etc/evil = subm:gui\n[subm.plugins]\nother = subm:x\n")
    assert script_names(text) == ["serve", "../../../usr/bin/sh", "/etc/evil"]


def test_a_script_an_entry_point_names_is_a_write_where_the_name_leads(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!/opt/venv/bin/python\n", 0o755],
                                    "etc/evil": ["file", "#!/opt/venv/bin/python\n", 0o755]})
    points = "[console_scripts]\n../../../usr/bin/sh = subm:main\n[gui_scripts]\n/etc/evil = subm:main\n"
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.dist-info/entry_points.txt": points}}))
    assert {d.path: d.wheels for d in record.listed} == {"/usr/bin/sh": ["subm-0.1-py3-none-any.whl"],
                                                         "/etc/evil": ["subm-0.1-py3-none-any.whl"]}
    targets = install_targets(tmp_path / "wheels", HostTree(base), HostTree(image))
    assert "/opt/venv/bin/sh" not in targets.files


def test_a_difference_no_wheel_accounts_for_counts_against_the_others(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755],
                                    "etc/passwd": ["file", "evil\n", 0o644]})
    record = _check(base, image, _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/../../../usr/bin/sh": "x"}}))
    assert record.differences == 2 and record.unattributed == 1


def test_a_file_among_the_wheels_that_is_not_a_wheel_accounts_for_nothing(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    wheels = _wheels(tmp_path)
    (wheels / "broken-1-py3-none-any.whl").write_text("not a zip")
    (wheels / "notes.txt").write_text("x")
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})
    assert _check(base, image, wheels).unattributed == 1


_CLIMBS = {"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n"}


def test_a_wheel_pip_did_not_install_accounts_for_nothing(tmp_path):
    """pip installs only the wheels the project needs; the image holds the ``.dist-info``
    directory of each it installed. A wheel it left out wrote nothing, so a difference only such
    a wheel names is no write of the submission's install."""
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    wheels = _wheels(tmp_path, installed=False, **{"subm-0.1": _CLIMBS})
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})
    assert [(d.path, d.wheels) for d in _check(base, image, wheels).listed] == [("/usr/bin/sh", [])]


def test_only_the_installed_wheels_account_for_what_they_name(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    _wheels(tmp_path, installed=False, **{"aaa-1": {"aaa-1.data/data/../../../usr/bin/sh": "x"}})
    wheels = _wheels(tmp_path, **{"subm-0.1": _CLIMBS})
    record = _check(base, image, wheels)
    assert [(d.path, d.wheels) for d in record.listed] == [("/usr/bin/sh", ["subm-0.1-py3-none-any.whl"])]


def test_a_wheel_whose_dist_info_the_base_held_was_not_installed(tmp_path):
    """pip finds a distribution the base holds at that version already installed and installs
    nothing of its wheel."""
    held = {**BASE_TREE, f"{SITE}/subm-0.1.dist-info": ["dir", 0o755]}
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]}, base=held)
    wheels = _wheels(tmp_path, installed=False, **{"subm-0.1": _CLIMBS})
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})


@pytest.mark.parametrize("extra", [{}, {"other-2.dist-info/METADATA": "Name: other\n"}],
                         ids=["none-of-its-own", "two"])
def test_a_wheel_pip_refuses_for_its_dist_info_directories_was_not_installed(tmp_path, extra):
    """pip installs a wheel only when it holds exactly one ``.dist-info`` directory."""
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    entries = {**_CLIMBS, **extra}
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        for member, text in entries.items():
            z.writestr(member, text)
        if extra:
            z.writestr("subm-0.1.dist-info/METADATA", "Name: subm\n")
    (wheels / "subm-0.1-py3-none-any.whl").write_bytes(data.getvalue())
    for name in ("subm-0.1.dist-info", "other-2.dist-info"):
        (image / SITE / name).mkdir()
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})


def test_a_dist_info_the_image_holds_as_a_link_is_no_install(tmp_path):
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    wheels = _wheels(tmp_path, installed=False, **{"subm-0.1": _CLIMBS})
    (image / SITE / "subm-0.1.dist-info").symlink_to(tmp_path)
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})


_ENTRY_POINTS = "junk-1.dist-info/entry_points.txt"


def _unreadable_entry_points(how: str) -> bytes:
    """A wheel whose list of entries reads but whose ``entry_points.txt`` does not, in the way
    *how* names: its compressed bytes broken, marked encrypted, or packed by a method zipfile
    does not know."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("junk/__init__.py", "")
        z.writestr(_ENTRY_POINTS, "[console_scripts]\nsh = junk:main\n" * 20)
    data = bytearray(out.getvalue())
    # The name follows the member's local header (30 bytes), then its central record (46).
    local, central = data.index(_ENTRY_POINTS.encode()) - 30, data.rindex(_ENTRY_POINTS.encode()) - 46
    if how == "broken":
        start = local + 30 + len(_ENTRY_POINTS)
        data[start:start + 8] = b"\xff" * 8
    elif how == "encrypted":
        data[local + 6] |= 1
        data[central + 8] |= 1
    else:
        data[local + 8:local + 10] = data[central + 10:central + 12] = (99).to_bytes(2, "little")
    return bytes(data)


@pytest.mark.parametrize("how", ["broken", "encrypted", "unknown-method"])
def test_a_wheel_whose_entry_points_cannot_be_read_accounts_for_nothing(tmp_path, how):
    """Only the wheel step wrote the wheels, and pip installs none it cannot read, so a wheel
    whose entry points do not read writes nothing here, whatever way the read fails: what the
    wheels hold never fails the check itself, which would be retried as the harness's."""
    base, image = _trees(tmp_path, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    wheels = _wheels(tmp_path)
    (wheels / "junk-1-py3-none-any.whl").write_bytes(_unreadable_entry_points(how))
    (image / SITE / "junk-1.dist-info").mkdir()
    with zipfile.ZipFile(wheels / "junk-1-py3-none-any.whl") as z:
        assert _ENTRY_POINTS in z.namelist()
    assert install_targets(wheels, HostTree(base), HostTree(image)) == ({}, {})
    assert _check(base, image, wheels).unattributed == 1


@pytest.mark.parametrize("code, submissions", [(errno.EACCES, True), (errno.EPERM, True),
                                               (errno.EINVAL, True), (None, True), (errno.EIO, False),
                                               (errno.ESTALE, False), (errno.ENOSPC, False)],
                         ids=["EACCES", "EPERM", "EINVAL", "no-errno", "EIO", "ESTALE", "ENOSPC"])
def test_a_wheel_read_error_writes_nothing_only_for_its_permissions(tmp_path, monkeypatch, code, submissions):
    """A wheel its permissions keep out, or that is not a zip zipfile can read (an offset of
    its own that makes a seek invalid, a decompressor's data error), is one the submission left
    unreadable, so it writes nothing here (``unreadable_zip``); any other error reading it is the
    node's, which fails the check itself, a fault the stage retries."""
    base, _ = _trees(tmp_path)
    wheels = _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/bin/x": "x"}})

    def contents(path, *args):
        raise OSError("Invalid data stream") if code is None else OSError(code, os.strerror(code), path)

    monkeypatch.setattr(integrity, "_wheel_contents", contents)
    if submissions:
        assert install_targets(wheels, HostTree(base), HostTree(tmp_path / "image")) == ({}, {})
    else:
        with pytest.raises(OSError, match=rf"\[Errno {code}\]"):
            install_targets(wheels, HostTree(base), HostTree(tmp_path / "image"))


def test_a_wheel_its_permissions_keep_out_writes_nothing(tmp_path):
    base, _ = _trees(tmp_path)
    wheels = _wheels(tmp_path, **{"subm-0.1": {"subm-0.1.data/data/bin/x": "x"}})
    (wheels / "subm-0.1-py3-none-any.whl").chmod(0)
    try:
        assert install_targets(wheels, HostTree(base), HostTree(tmp_path / "image")) == ({}, {})
    finally:
        (wheels / "subm-0.1-py3-none-any.whl").chmod(0o644)


def test_the_list_is_cut_but_the_counts_cover_every_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "MAX_LISTED", 2)
    changes = {f"etc/new{i}": ["file", "x", 0o644] for i in range(5)}
    base, image = _trees(tmp_path, changes)
    record = _check(base, image, _wheels(tmp_path))
    assert len(record.listed) == 2 and record.differences == record.unattributed == 5


def _main(*args: str) -> subprocess.CompletedProcess:
    """Run the program; it reports its record in ``outputs.json`` beside ``--out``."""
    report = Path(args[args.index("--out") + 1]).with_name("outputs.json")
    return subprocess.run([sys.executable, "-m", "hls.package.integrity", *args, "--outputs-out", str(report)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})


def _reported(out: Path) -> tuple[str, int]:
    record = (read_outputs_report(out.with_name("outputs.json")) or {})[out.name]
    return record.sha256, record.size


def test_the_program_exits_by_whether_the_system_paths_differ(tmp_path):
    base, image = _trees(tmp_path)
    out = tmp_path / "integrity.json"
    done = _main("--base", str(base), "--image", str(image), "--wheels", str(_wheels(tmp_path)), "--out", str(out))
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["differences"] == 0
    assert _reported(out) == sha256_file(out)
    (image / "etc" / "passwd").write_text("evil\n")
    done = _main("--base", str(base), "--image", str(image), "--wheels", str(tmp_path / "wheels"), "--out", str(out))
    assert done.returncode == DIFFERS_EXIT
    assert _reported(out) == sha256_file(out)
    assert IntegrityRecord.model_validate_json(out.read_text()).listed[0].path == "/etc/passwd"


def test_an_entry_the_check_cannot_read_fails_the_check_itself(tmp_path):
    base, image = _trees(tmp_path)
    (image / "etc" / "os-release").chmod(0)
    (base / "etc" / "os-release").chmod(0)
    out = tmp_path / "integrity.json"
    try:
        done = _main("--base", str(base), "--image", str(image), "--wheels", str(_wheels(tmp_path)), "--out", str(out))
    finally:
        for tree in (base, image):
            (tree / "etc" / "os-release").chmod(0o644)
    assert done.returncode not in (0, DIFFERS_EXIT) and "PermissionError" in done.stderr
    assert not out.exists()


def test_a_tree_that_is_not_an_extracted_image_is_refused(tmp_path):
    base, _ = _trees(tmp_path)
    done = _main("--base", str(base), "--image", str(tmp_path / "none"), "--wheels", str(_wheels(tmp_path)),
                 "--out", str(tmp_path / "o.json"))
    assert done.returncode not in (0, DIFFERS_EXIT) and "is not an extracted image" in done.stderr
