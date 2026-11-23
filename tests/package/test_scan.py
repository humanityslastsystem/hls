"""The scan for secret values in what a submission image and a source bundle hide from finalize's
scan by compressing it (``hls.package.scan``), read while it is plain: the checked-out tree, and the
files the offline install added to the image, searched for every secret's value in their bytes as
stored and in their paths in the image. ``package-scan.json`` names each file by its path in the
image, a value in it replaced by the secret's name, with the secret's name, never a value."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.package import scan
from hls.package.scan import FINDING_PATH_CHARS, FINDINGS_PER_SECRET, added_entries, tree_entries
from hls.recipes.model import VENV
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from hls.substrate.secret_scan import PackageScan, PackageScanArea, PackageScanFinding
from tests.package.fake_tools import BASE_TREE, lay_out

#: The venv's site-packages, from an image's root.
SITE = "opt/venv/lib/python3.12/site-packages"
KEY, TOKEN = "sk-relay-0123456789abcdef", "hf-token-0123456789"
VALUES = {"relay": KEY, "hf_token": TOKEN}


def _tree(root: Path, files: dict[str, str]) -> Path:
    tree = root / "tree"
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    return tree


def _install(tmp_path: Path, files: dict[str, str], *, image: dict | None = None) -> tuple[Path, Path]:
    """The base, and an image of it in which the install wrote *files* (path under site-packages to
    text), with *image* (path from the root to an entry, or None to leave it out) on top."""
    lay_out(tmp_path / "base", BASE_TREE)
    tree = {**BASE_TREE, f"{SITE}/subm-0.1.dist-info": ["dir", 0o755], f"{SITE}/subm": ["dir", 0o755]}
    tree.update({f"{SITE}/{path}": ["file", text, 0o644] for path, text in files.items()})
    for path, entry in (image or {}).items():
        if entry is None:
            del tree[path]
        else:
            tree[path] = entry
    lay_out(tmp_path / "image", tree)
    return tmp_path / "base", tmp_path / "image"


def test_every_file_of_the_tree_is_read_dotfiles_and_all_and_named_by_its_path_in_the_image(tmp_path):
    tree = _tree(tmp_path, {"server.py": "print('ok')\n", ".env": f"RELAY={KEY}\n", "conf/.hidden/t.txt": TOKEN})
    found = scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace")))
    assert found == PackageScan(secrets=["hf_token", "relay"], forms=["value", "host"], read=["tree"],
                                findings=[PackageScanFinding(path="/workspace/.env", secret="relay"),
                                          PackageScanFinding(path="/workspace/conf/.hidden/t.txt",
                                                             secret="hf_token")])


def test_a_value_in_a_path_is_found_and_the_record_names_its_secret_in_its_place(tmp_path):
    """A file named after a key holds the key in its path: the record names it with the secret's
    name where the value stood, so the record itself never holds a value."""
    tree = _tree(tmp_path, {f"keys/{KEY}.txt": "nothing here\n"})
    record = scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace")))
    assert record.findings == [PackageScanFinding(path="/workspace/keys/<redacted:relay>.txt", secret="relay")]
    assert KEY not in record.model_dump_json()


def test_a_path_is_recorded_to_a_bounded_length(tmp_path):
    """A record names each file by at most ``FINDING_PATH_CHARS`` characters of its path, so its size
    is bounded whatever path the install wrote; the tree's paths, within the checkout's limits, are
    never cut."""
    deep = "/".join(["d" * 200] * 6)
    longest = "/".join(["e" * 200] * 5) + "/" + "f" * 19
    tree = _tree(tmp_path, {f"{deep}/k.txt": KEY, longest: KEY})
    found = scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace"))).findings
    assert [len(f.path) for f in found] == [FINDING_PATH_CHARS, len("/workspace/") + 1024]
    assert found[0].path == f"/workspace/{deep}/k.txt"[:FINDING_PATH_CHARS] and FINDING_PATH_CHARS >= 1035


def test_a_value_is_found_however_the_reads_split_it(tmp_path):
    tree = _tree(tmp_path, {"blob.bin": "x" * (scan.CHUNK_BYTES - 5) + KEY})
    assert [f.path for f in scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace"))).findings] == [
        "/workspace/blob.bin"]


def test_a_file_is_read_as_stored_never_decompressed(tmp_path):
    import gzip

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "plain.gz").write_bytes(gzip.compress(KEY.encode()))
    (tree / "magic").write_bytes(b"\x1f\x8b" + KEY.encode())
    assert [f.path for f in scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace"))).findings] == [
        "/workspace/magic"]


def test_a_link_in_the_tree_is_never_followed(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text(KEY)
    tree = _tree(tmp_path, {"server.py": ""})
    os.symlink(outside, tree / "leak.txt")
    os.symlink(tmp_path, tree / "up")
    assert scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace"))).findings == []


def test_every_file_the_install_added_or_changed_under_the_venv_is_read_and_no_base_file(tmp_path):
    """What the install added is every file under the image's venv that the base does not hold, or
    holds with other bytes: the wheels' entries, and what pip wrote of its own, such as the
    bytecode it compiles. A file the base holds as it is, a key in it or not, is the base's."""
    base_key = f"{SITE}/base_mod.py"
    base, image = _install(tmp_path, {"subm/key.py": f"KEY = {KEY!r}\n",
                                      "subm/__pycache__/key.cpython-312.pyc": f"\x00\x00{TOKEN}"},
                           image={f"{SITE}/subm/__pycache__": ["dir", 0o755]})
    # A base file the image holds unchanged, and one the install rewrote with the same size.
    for root in (base, image):
        (root / base_key).write_text(f"KEY = {KEY!r}\n")
    (base / f"{SITE}/other.py").write_text("x" * len(f"y{KEY}"))
    (image / f"{SITE}/other.py").write_text(f"y{KEY}")
    found = scan.scan(VALUES, ("install", added_entries(base, image)))
    # In the order of their paths.
    assert found.findings == [PackageScanFinding(path=f"/{SITE}/other.py", secret="relay"),
                              PackageScanFinding(path=f"/{SITE}/subm/__pycache__/key.cpython-312.pyc",
                                                 secret="hf_token"),
                              PackageScanFinding(path=f"/{SITE}/subm/key.py", secret="relay")]


def test_a_key_whole_only_in_the_bytecode_pip_compiled_is_found(tmp_path):
    """A module may split its key across two literals the parser joins: the key is whole only in
    the .pyc pip compiles as it installs the wheel, which the image then compresses."""
    whl = tmp_path / "leaky-1.0-py3-none-any.whl"
    module = f'KEY = ("{KEY[:10]}"\n       "{KEY[10:]}")\n'
    assert KEY not in module
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("leaky/__init__.py", module)
        z.writestr("leaky-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: leaky\nVersion: 1.0\n")
        z.writestr("leaky-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\n"
                                               "Tag: py3-none-any\n")
        z.writestr("leaky-1.0.dist-info/RECORD", "")
    base, image = tmp_path / "base", tmp_path / "image"
    subprocess.run([sys.executable, "-m", "venv", str(base / VENV.lstrip("/"))], check=True, timeout=300)
    shutil.copytree(base, image, symlinks=True)
    venv = image / VENV.lstrip("/")
    subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "--no-index", str(whl)], check=True,
                   timeout=300)
    pycs = list(venv.rglob("leaky/__pycache__/*.pyc"))
    assert pycs and any(KEY.encode() in p.read_bytes() for p in pycs)
    found = scan.scan(VALUES, ("install", added_entries(base, image)))
    assert [f.path.rsplit("/", 2)[-2:] for f in found.findings] == [["__pycache__", pycs[0].name]]


def test_an_install_file_that_leads_out_of_the_image_is_never_read(tmp_path):
    """A file under the venv is read only when it is a regular file whose real path stays inside the
    extracted image."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "key.py").write_text(KEY)
    base, image = _install(tmp_path, {}, image={f"{SITE}/subm": ["link", str(host)],
                                                f"{SITE}/leak.py": ["link", str(host / "key.py")]})
    assert scan.scan(VALUES, ("install", added_entries(base, image))).findings == []


def test_a_secret_past_the_bound_is_recorded_over_the_cap_by_its_count_and_area_not_its_paths(tmp_path,
                                                                                                  monkeypatch):
    """A contestant can write a value into every file it makes: each secret's listed files are
    bounded, so the record finalize reads stays small however many files hold it, and past the
    bound the record names the area and how many of its files hold the value, never their paths,
    for finalize to keep the whole area out of any export."""
    monkeypatch.setattr(scan, "FINDINGS_PER_SECRET", 3)
    deep = "d" * 200
    tree = _tree(tmp_path, {**{f"{deep}/f{i:03}": KEY for i in range(200)}, "t": TOKEN})
    record = scan.scan(VALUES, ("tree", tree_entries(tree, "/workspace")))
    assert [f.path for f in record.findings if f.secret == "relay"] == [f"/workspace/{deep}/f{i:03}" for i in range(3)]
    assert [f.path for f in record.findings if f.secret == "hf_token"] == ["/workspace/t"]
    assert record.areas == [PackageScanArea(secret="relay", area="tree", count=200)]
    assert len(record.model_dump_json()) < 2000
    assert FINDINGS_PER_SECRET == 1000


def test_a_secret_at_its_bound_is_not_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "FINDINGS_PER_SECRET", 3)
    record = scan.scan(VALUES, ("tree", tree_entries(_tree(tmp_path, {f"f{i}": KEY for i in range(3)}), "/workspace")))
    assert len(record.findings) == 3 and record.areas == []


def test_the_trees_scan_extends_the_bundles_record_under_the_same_bound(tmp_path, monkeypatch):
    """The source bundle's objects are scanned first, by the checkout, into a record of areas; the
    tree is scanned into it: what each scan read adds up, and so does the bound."""
    monkeypatch.setattr(scan, "FINDINGS_PER_SECRET", 2)
    objects = PackageScan(secrets=["hf_token", "relay"], forms=["value", "host"], findings=[],
                          read=["bundle"], areas=[PackageScanArea(secret="relay", area="bundle", count=4)])
    record = scan.scan(VALUES, ("tree", tree_entries(_tree(tmp_path, {f"f{i}": KEY for i in range(3)}), "/workspace")),
                       extend=objects)
    assert record.read == ["bundle", "tree"] and [f.path for f in record.findings] == ["/workspace/f0", "/workspace/f1"]
    assert record.areas == [PackageScanArea(secret="relay", area="bundle", count=4),
                            PackageScanArea(secret="relay", area="tree", count=3)]


def test_a_record_of_other_secrets_is_never_extended(tmp_path):
    first = PackageScan(secrets=["relay"], forms=["value", "host"], findings=[], read=["bundle"])
    with pytest.raises(ValueError, match="searched .'relay'., not .'hf_token', 'relay'."):
        scan.scan(VALUES, ("tree", tree_entries(_tree(tmp_path, {"f": ""}), "/workspace")), extend=first)


def test_the_role_writes_the_record_with_names_only_and_reports_its_digest(tmp_path):
    tree = _tree(tmp_path / "t", {".env": KEY, "server.py": ""})
    (tmp_path / "i").mkdir()
    base, image = _install(tmp_path / "i", {"subm/key.py": TOKEN})
    for argv, expected, read in (
            (["--tree", str(tree), "--tree-at", "/workspace"], [PackageScanFinding(path="/workspace/.env",
                                                                                   secret="relay")], ["tree"]),
            (["--base", str(base), "--image", str(image)], [PackageScanFinding(path=f"/{SITE}/subm/key.py",
                                                                              secret="hf_token")], ["install"])):
        out, report = tmp_path / "outputs" / "package-scan.json", tmp_path / "outputs.json"
        out.parent.mkdir(exist_ok=True)
        env = {**os.environ, "relay": KEY, "hf_token": TOKEN}
        done = subprocess.run([sys.executable, "-m", "hls.package.scan", "--names", "relay,hf_token", *argv,
                               "--out", str(out), "--outputs-out", str(report)],
                              capture_output=True, text=True, env=env, timeout=120)
        assert done.returncode == 0, done.stderr
        data = out.read_bytes()
        assert KEY.encode() not in data and TOKEN.encode() not in data
        record = PackageScan.model_validate_json(data)
        assert (record.findings, record.read) == (expected, read)
        assert read_outputs_report(report)["package-scan.json"].sha256 == sha256_file(out)[0]
        assert KEY not in done.stdout and TOKEN not in done.stdout
        assert json.loads(done.stdout.splitlines()[-1]) == {"findings": 1, "areas": []}


def _role_findings(tmp_path: Path, argv: list[str]) -> list[tuple[str, str]]:
    """The findings, each its path and label, of the role run on *argv*, in a process of its own
    under a time limit, so a role that opened a pipe fails here instead of hanging."""
    out, report = tmp_path / "outputs" / "package-scan.json", tmp_path / "outputs.json"
    out.parent.mkdir(exist_ok=True)
    done = subprocess.run([sys.executable, "-m", "hls.package.scan", "--names", "relay,hf_token", *argv, "--out",
                           str(out), "--outputs-out", str(report)], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "relay": KEY, "hf_token": TOKEN})
    assert done.returncode == 0, done.stderr
    return [(f.path, f.secret) for f in PackageScan.model_validate_json(out.read_bytes()).findings]


def test_every_entry_the_install_added_is_searched_in_its_path_and_a_link_in_the_text_it_holds(tmp_path):
    """The image carries every link, pipe and empty directory the install left in the venv: a form
    in the path of any of them, or in the target a link holds, is found, though no link is followed
    and no pipe opened. A pipe is never the base's, even where the base holds one."""
    base, image = _install(tmp_path, {}, image={f"{SITE}/subm/to-key": ["link", f"/srv/{KEY}"],
                                                f"{SITE}/subm/{TOKEN}.lnk": ["link", "elsewhere"],
                                                f"{SITE}/{TOKEN}-empty": ["dir", 0o755]})
    os.mkfifo(image / SITE / "subm" / f"{KEY}.pipe")
    (base / SITE / "subm").mkdir()
    os.mkfifo(base / SITE / "subm" / f"{KEY}.pipe")
    assert _role_findings(tmp_path, ["--base", str(base), "--image", str(image)]) == [
        (f"/{SITE}/<redacted:hf_token>-empty", "hf_token"), (f"/{SITE}/subm/<redacted:hf_token>.lnk", "hf_token"),
        (f"/{SITE}/subm/<redacted:relay>.pipe", "relay"), (f"/{SITE}/subm/to-key", "relay")]


def test_every_entry_of_the_tree_is_searched_in_its_path_and_a_link_in_the_text_it_holds(tmp_path):
    """The checkout holds no link and git no pipe, but the tree's scan reads every entry by the same
    rule as the install's: nothing under the tree is passed over unsearched."""
    tree = _tree(tmp_path, {"server.py": ""})
    os.symlink(f"/srv/{KEY}", tree / "to-key")
    (tree / f"{TOKEN}-empty").mkdir()
    os.mkfifo(tree / f"{KEY}.pipe")
    assert _role_findings(tmp_path, ["--tree", str(tree), "--tree-at", "/workspace"]) == [
        ("/workspace/<redacted:hf_token>-empty", "hf_token"), ("/workspace/<redacted:relay>.pipe", "relay"),
        ("/workspace/to-key", "relay")]


def test_the_role_fails_on_a_tree_it_cannot_read(tmp_path):
    out, report = tmp_path / "package-scan.json", tmp_path / "outputs.json"
    done = subprocess.run([sys.executable, "-m", "hls.package.scan", "--names", "relay", "--tree",
                           str(tmp_path / "missing"), "--tree-at", "/workspace", "--out", str(out), "--outputs-out",
                           str(report)], capture_output=True, text=True, env={**os.environ, "relay": KEY}, timeout=120)
    assert done.returncode == 1 and not out.exists() and KEY not in done.stderr


# --- every form of each secret: a URL secret's host is searched too ----------

#: A route's base URL, a secret whose host is one of its forms, and the host.
URL, HOST = "https://Relay.Example:8443/v1/anthropic", "relay.example"
REGISTRY = {**VALUES, "relay_url": URL}


def test_the_record_states_every_form_it_searched(tmp_path):
    record = scan.scan(REGISTRY, ("tree", tree_entries(_tree(tmp_path, {"server.py": ""}), "/workspace")))
    assert (record.secrets, record.forms) == (["hf_token", "relay", "relay_url"], ["value", "host"])


@pytest.mark.parametrize("printed", [HOST, HOST.upper(), "Relay.Example"])
def test_a_host_in_a_tree_file_is_found_under_its_label(tmp_path, printed):
    tree = _tree(tmp_path, {"client.py": f"BASE = 'https://{printed}:8443'\n", "server.py": ""})
    assert scan.scan(REGISTRY, ("tree", tree_entries(tree, "/workspace"))).findings == [
        PackageScanFinding(path="/workspace/client.py", secret="relay_url.host")]


def test_a_host_in_an_installed_file_is_found_under_its_label(tmp_path):
    base, image = _install(tmp_path, {"subm/conf.py": f"HOST = {HOST.upper()!r}\n"})
    assert scan.scan(REGISTRY, ("install", added_entries(base, image))).findings == [
        PackageScanFinding(path=f"/{SITE}/subm/conf.py", secret="relay_url.host")]


@pytest.mark.parametrize("printed", [HOST, "RELAY.EXAMPLE"])
def test_a_path_holding_a_host_is_redacted_in_the_record(tmp_path, printed):
    """A path is recorded with every form of every secret in it replaced by its marker, the host in
    any case, so the record never holds a value or a host. The directory named after the host is
    an entry of its own, found in its path as the file below it is."""
    tree = _tree(tmp_path, {f"cache/{printed}/{KEY}.json": "{}\n"})
    record = scan.scan(REGISTRY, ("tree", tree_entries(tree, "/workspace")))
    folder = "/workspace/cache/<redacted:relay_url.host>"
    path = f"{folder}/<redacted:relay>.json"
    assert record.findings == [PackageScanFinding(path=folder, secret="relay_url.host"),
                               PackageScanFinding(path=path, secret="relay"),
                               PackageScanFinding(path=path, secret="relay_url.host")]
    assert HOST not in record.model_dump_json().lower() and KEY not in record.model_dump_json()


def test_each_label_is_bounded_on_its_own(tmp_path, monkeypatch):
    """A file that holds a URL holds its host too: the value's and the host's files are each listed
    up to the bound, and past it each is counted in its area."""
    monkeypatch.setattr(scan, "FINDINGS_PER_SECRET", 2)
    tree = _tree(tmp_path, {f"f{i}": URL for i in range(3)})
    record = scan.scan(REGISTRY, ("tree", tree_entries(tree, "/workspace")))
    assert [(f.path, f.secret) for f in record.findings] == [
        ("/workspace/f0", "relay_url"), ("/workspace/f0", "relay_url.host"),
        ("/workspace/f1", "relay_url"), ("/workspace/f1", "relay_url.host")]
    assert record.areas == [PackageScanArea(secret="relay_url", area="tree", count=3),
                            PackageScanArea(secret="relay_url.host", area="tree", count=3)]


def test_a_record_that_searched_other_forms_is_never_extended(tmp_path):
    values_only = PackageScan(secrets=sorted(REGISTRY), forms=["value"], findings=[], read=["bundle"])
    tree = _tree(tmp_path, {"f": ""})
    with pytest.raises(ValueError, match=r"searched the forms \['value'\], not \['value', 'host'\]"):
        scan.scan(REGISTRY, ("tree", tree_entries(tree, "/workspace")), extend=values_only)


def test_a_record_that_does_not_say_which_forms_it_searched_is_no_package_scan():
    old = b'{"secrets": ["relay"], "read": ["bundle", "tree"], "findings": [], "areas": []}'
    with pytest.raises(ValidationError, match="forms"):
        PackageScan.model_validate_json(old)


def test_the_role_searches_every_form_and_records_none(tmp_path):
    tree = _tree(tmp_path, {"client.py": f"BASE = {HOST.upper()!r}\n", f"{HOST}.txt": ""})
    out, report = tmp_path / "outputs" / "package-scan.json", tmp_path / "outputs.json"
    out.parent.mkdir()
    argv = ["--names", "relay_url,relay", "--tree", str(tree), "--tree-at", "/workspace", "--out", str(out),
            "--outputs-out", str(report)]
    done = subprocess.run([sys.executable, "-m", "hls.package.scan", *argv], capture_output=True, text=True,
                          env={**os.environ, "relay_url": URL, "relay": KEY}, timeout=120)
    assert done.returncode == 0, done.stderr
    record = PackageScan.model_validate_json(out.read_bytes())
    assert (record.secrets, record.forms) == (["relay", "relay_url"], ["value", "host"])
    assert record.findings == [PackageScanFinding(path="/workspace/client.py", secret="relay_url.host"),
                               PackageScanFinding(path="/workspace/<redacted:relay_url.host>.txt",
                                                  secret="relay_url.host")]
    assert HOST not in out.read_text().lower() and HOST not in (done.stdout + done.stderr).lower()
