"""An image read from its archive's layers (``hls.package.layers``): the digests from the pinned
manifest down to each layer's bytes, the OCI changeset rules, names that are data, and the proof
that an image is its base and one layer more. The package's integrity check and scan read an
image so on a substrate whose images are archives; each gives, for the same images, the record
it gives for their extracted trees."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from hls.package import integrity, layers, scan
from hls.package.integrity import HostTree, archive_trees, check
from hls.package.layers import LayerError, Meta, Tree, apply, base_and_image, installed_layer, open_image, read_layer
from tests.package.fake_tools import BASE_TREE, lay_out, wheel
from tests.package.oci import GZIP, PLAIN, diff, layer, write_image

SITE = "opt/venv/lib/python3.12/site-packages"
KEY, TOKEN = "sk-relay-0123456789abcdef", "hf-token-0123456789"
VALUES = {"relay": KEY, "hf_token": TOKEN}
#: A base whose venv links into /usr, and a script of whose venv leads nowhere.
LINKED_BASE = {**BASE_TREE, "usr/share": ["dir", 0o755], "opt/venv/share": ["link", "/usr/share"],
               "opt/venv/bin/tool": ["link", "/usr/bin/tool-real"]}


def _split(tree: dict[str, list]) -> list[bytes]:
    """*tree* as two layers, /usr in the first, so a tree is merged across layers."""
    usr = {p: e for p, e in tree.items() if p.split("/")[0] == "usr"}
    return [layer(usr), layer({p: e for p, e in tree.items() if p not in usr})]


def _images(tmp_path: Path, base: dict[str, list], image: dict[str, list], *, media_type: str = PLAIN,
            rewritten: tuple[str, ...] = ()) -> tuple[Path, str, Path, str]:
    """Archives of *base* and of *image*, the base's layers and the one layer that makes it *image*,
    with their manifests' bare hex digests. The layer also holds the entries *rewritten* as *image*
    holds them, as a commit's layer holds a file written again with the same bytes, or a directory
    a write below it touched."""
    below = _split(base)
    top = {**diff(base, image), **{path: image[path] for path in rewritten}}
    b = write_image(tmp_path / "base.tar", below, media_type=media_type)
    i = write_image(tmp_path / "image.tar", [*below, layer(top)], media_type=media_type)
    return tmp_path / "base.tar", b[7:], tmp_path / "image.tar", i[7:]


def _changed(tree: dict, changes: dict) -> dict:
    out = dict(tree)
    for path, entry in changes.items():
        if entry is None:
            for p in [p for p in out if p == path or p.startswith(path + "/")]:
                del out[p]
        else:
            out[path] = entry
    return out


def _wheels(tmp_path: Path, image: dict, **entries: dict[str, str]) -> Path:
    """A wheels directory with one wheel per keyword, each installed in *image*."""
    out = tmp_path / "wheels"
    out.mkdir()
    for stem, extra in entries.items():
        name, version = stem.split("-")[:2]
        (out / f"{stem}-py3-none-any.whl").write_bytes(wheel(name, version, extra))
        image[f"{SITE}/{name}-{version}.dist-info"] = ["dir", 0o755]
    return out


# The digests from the manifest down.

def test_an_image_is_read_by_the_manifest_it_was_pinned_by(tmp_path):
    digest = write_image(tmp_path / "a.tar", [layer({"etc": ["dir", 0o755]})])
    image = open_image(tmp_path / "a.tar", digest)
    assert [layer.media_type for layer in image.layers] == [PLAIN]
    with pytest.raises(LayerError, match="not the manifest"):
        open_image(tmp_path / "a.tar", "sha256:" + "0" * 64)


def test_a_layer_whose_bytes_are_not_its_diff_id_is_refused(tmp_path):
    data = layer({"etc": ["dir", 0o755]})
    digest = write_image(tmp_path / "a.tar", [data], diff_ids=["sha256:" + "1" * 64])
    image = open_image(tmp_path / "a.tar", digest)
    with pytest.raises(LayerError, match="is not the layer"):
        read_layer(image, image.layers[0])


def test_gzip_and_plain_layers_read_alike(tmp_path):
    tree = {"etc": ["dir", 0o755], "etc/passwd": ["file", "root\n", 0o644], "bin": ["link", "usr/bin"]}
    plain = open_image(tmp_path / "p.tar", write_image(tmp_path / "p.tar", [layer(tree)]))
    zipped = open_image(tmp_path / "g.tar", write_image(tmp_path / "g.tar", [layer(tree)], media_type=GZIP))
    assert read_layer(plain, plain.layers[0]) == read_layer(zipped, zipped.layers[0])


def test_a_layer_of_a_kind_not_read_is_refused(tmp_path):
    digest = write_image(tmp_path / "a.tar", [layer({})], media_type="application/vnd.oci.image.layer.v1.tar+zstd")
    with pytest.raises(LayerError, match="not read"):
        open_image(tmp_path / "a.tar", digest)


# Names are data.

def test_a_name_that_climbs_out_of_the_image_is_refused(tmp_path):
    info = tarfile.TarInfo("etc/../../host/passwd")
    info.size = 1
    digest = write_image(tmp_path / "a.tar", [layer({}, extra=(info,))])
    image = open_image(tmp_path / "a.tar", digest)
    with pytest.raises(LayerError, match="climbs"):
        read_layer(image, image.layers[0])


def test_a_hard_link_to_no_file_before_it_is_refused_and_one_to_a_file_is_that_file(tmp_path):
    link = tarfile.TarInfo("etc/hard")
    link.type, link.linkname = tarfile.LNKTYPE, "etc/passwd"
    digest = write_image(tmp_path / "a.tar", [layer({"etc": ["dir", 0o755], "etc/passwd": ["file", "x", 0o640]},
                                                    extra=(link,))])
    image = open_image(tmp_path / "a.tar", digest)
    changes = {c.path: c.meta for c in read_layer(image, image.layers[0])}
    assert changes["etc/hard"] == changes["etc/passwd"] == Meta("file", 0o640, 1, sha256=changes["etc/passwd"].sha256)
    link.linkname = "etc/none"
    digest = write_image(tmp_path / "b.tar", [layer({"etc": ["dir", 0o755]}, extra=(link,))])
    image = open_image(tmp_path / "b.tar", digest)
    with pytest.raises(LayerError, match="no file before it"):
        read_layer(image, image.layers[0])


def test_names_with_a_leading_slash_or_dot_name_the_same_paths(tmp_path):
    infos = []
    for name, kind in (("/abs", tarfile.DIRTYPE), ("./abs/f", tarfile.REGTYPE)):
        info = tarfile.TarInfo(name)
        info.type = kind
        infos.append(info)
    digest = write_image(tmp_path / "a.tar", [layer({}, extra=tuple(infos))])
    image = open_image(tmp_path / "a.tar", digest)
    assert [c.path for c in read_layer(image, image.layers[0])] == ["abs", "abs/f"]


# The changeset rules.

def test_whiteouts_remove_what_the_layers_below_hold_and_none_of_their_own_layers():
    tree = Tree()
    apply(tree, [layers.Change("a", Meta("directory", 0o755)), layers.Change("a/b", Meta("directory", 0o755)),
                 layers.Change("a/b/c", Meta("file", 0o644, 1, sha256="x")),
                 layers.Change("a/d", Meta("file", 0o644, 1, sha256="y"))])
    apply(tree, [layers.Change("a/b", None), layers.Change("a/e", Meta("file", 0o600, 0, sha256="z"))])
    assert sorted(tree.entries) == ["", "a", "a/d", "a/e"]
    apply(tree, [layers.Change("a", None, opaque=True), layers.Change("a/f", Meta("directory", 0o700))])
    assert sorted(tree.entries) == ["", "a", "a/f"] and tree.names["a"] == {"f"}


def test_a_non_directory_over_a_directory_replaces_it_with_everything_under_it():
    tree = Tree()
    apply(tree, [layers.Change("a", Meta("directory", 0o755)), layers.Change("a/b", Meta("file", 0o644, 0, sha256="x"))])
    apply(tree, [layers.Change("a", Meta("link", 0o777, target="/etc"))])
    assert sorted(tree.entries) == ["", "a"] and "a" not in tree.names
    apply(tree, [layers.Change("a", Meta("directory", 0o750)), layers.Change("a/c", Meta("directory", 0o755))])
    assert sorted(tree.entries) == ["", "a", "a/c"]


def test_an_entry_whose_directory_no_layer_holds_is_refused():
    with pytest.raises(LayerError, match="no layer holds a directory"):
        apply(Tree(), [layers.Change("a/b", Meta("file", 0o644, 0, sha256="x"))])
    tree = Tree()
    apply(tree, [layers.Change("a", Meta("link", 0o777, target="b"))])
    with pytest.raises(LayerError, match="no layer holds a directory"):
        apply(tree, [layers.Change("a/c", Meta("file", 0o644, 0, sha256="x"))])


# Base and one layer more.

def test_an_image_is_its_base_and_one_layer_more_by_diff_id_whatever_the_layers_compression(tmp_path):
    below = _split(BASE_TREE)
    base = open_image(tmp_path / "b.tar", write_image(tmp_path / "b.tar", below, media_type=GZIP))
    top = layer({"workspace": ["dir", 0o755]})
    image = open_image(tmp_path / "i.tar", write_image(tmp_path / "i.tar", [*below, top]))
    assert installed_layer(base, image) == image.layers[-1]
    before, after, changes = base_and_image(base, image)
    assert [c.path for c in changes] == ["workspace"]
    assert after.entries == {**before.entries, "workspace": Meta("directory", 0o755)}


@pytest.mark.parametrize("layered", [
    lambda below, top: [*below, top, top],
    lambda below, top: [below[0], top],
    lambda below, top: [below[1], below[0], top],
    lambda below, top: below,
], ids=["two-more", "one-less", "reordered", "none-more"])
def test_an_image_that_is_not_its_base_and_one_layer_more_is_refused(tmp_path, layered):
    below = _split(BASE_TREE)
    base = open_image(tmp_path / "b.tar", write_image(tmp_path / "b.tar", below))
    top = layer({"workspace": ["dir", 0o755]})
    image = open_image(tmp_path / "i.tar", write_image(tmp_path / "i.tar", layered(below, top)))
    with pytest.raises(LayerError, match="one layer more"):
        installed_layer(base, image)


# The checks read an archive as they read its extracted tree.

@pytest.mark.parametrize("base, changes, wheels", [
    (BASE_TREE, {"workspace": ["dir", 0o755], "opt/venv/bin/tool": ["file", "x", 0o755]}, {}),
    (BASE_TREE, {"usr/bin/sh": ["file", "#!evil\n", 0o755]}, {}),
    (BASE_TREE, {"usr/bin/sh": ["file", "#!dash\n", 0o4755]}, {}),
    (BASE_TREE, {"usr/bin/python3": ["link", "/tmp/py"]}, {}),
    (BASE_TREE, {"bin": ["dir", 0o755], "bin/sh": ["file", "#!evil\n", 0o755]}, {}),
    (BASE_TREE, {"etc/os-release": None}, {}),
    (BASE_TREE, {"usr/lib64": None}, {}),
    (BASE_TREE, {"libx32": ["dir", 0o755]}, {}),
    (BASE_TREE, {"usr/bin/sh": ["file", "#!evil\n", 0o755], "usr/share": ["dir", 0o755],
                 "usr/share/evil": ["dir", 0o755], "usr/share/evil/f": ["file", "x", 0o644]},
     {"subm-0.1": {"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n",
                   "subm-0.1.data/data/../../../usr/share/evil/f": "x"}, "httpx-0.27.0": {}}),
    (LINKED_BASE, {"usr/share/evil": ["dir", 0o755], "usr/share/evil/f": ["file", "x", 0o644],
                   "usr/bin/tool-real": ["file", "y", 0o755]},
     {"subm-0.1": {"subm-0.1.data/data/share/evil/f": "x", "subm-0.1.data/scripts/tool": "y"}}),
    (LINKED_BASE, {"usr/share": ["file", "z", 0o644]},
     {"subm-0.1": {"subm-0.1.data/data/../../../usr/share": "z"}}),
], ids=["same", "content", "mode", "link-target", "type", "removed", "removed-dir", "new-lib", "climbing-wheel",
        "through-links", "file-over-dir"])
def test_the_integrity_check_of_an_archive_is_the_check_of_its_extracted_tree(tmp_path, base, changes, wheels):
    image = _changed(base, changes)
    found = _wheels(tmp_path, image, **wheels)
    lay_out(tmp_path / "base-tree", base)
    lay_out(tmp_path / "image-tree", image)
    extracted = check(HostTree(tmp_path / "base-tree"), HostTree(tmp_path / "image-tree"), found)
    read = check(*archive_trees(*_images(tmp_path, base, image)), found)
    assert read == extracted
    assert read.entries > 10


def test_the_integrity_program_reads_two_archives(tmp_path, capsys):
    image = _changed(BASE_TREE, {"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, BASE_TREE, image, media_type=GZIP)
    (tmp_path / "wheels").mkdir()
    out = tmp_path / "integrity.json"
    status = integrity.main(["--base-archive", str(base_archive), "--base-manifest", base_manifest,
                             "--image-archive", str(image_archive), "--image-manifest", image_manifest,
                             "--wheels", str(tmp_path / "wheels"), "--out", str(out),
                             "--outputs-out", str(tmp_path / "outputs.json")])
    assert status == integrity.DIFFERS_EXIT
    assert [(d.path, d.change) for d in integrity.IntegrityRecord.model_validate_json(out.read_text()).listed] == [
        ("/usr/bin/sh", "content")]
    with pytest.raises(SystemExit):
        integrity.main(["--base", str(tmp_path), "--image-archive", str(image_archive), "--wheels", str(tmp_path),
                        "--out", str(out), "--outputs-out", str(tmp_path / "o.json")])


def _install_trees(files: dict[str, str], extra: dict | None = None) -> dict[str, list]:
    tree = {**BASE_TREE, f"{SITE}/subm-0.1.dist-info": ["dir", 0o755], f"{SITE}/subm": ["dir", 0o755],
            f"{SITE}/base_mod.py": ["file", f"KEY = {KEY!r}\n", 0o644]}
    tree.update({f"{SITE}/{path}": ["file", text, 0o644] for path, text in files.items()})
    tree.update(extra or {})
    return tree


@pytest.mark.parametrize("files, extra", [
    ({"subm/key.py": f"KEY = {KEY!r}\n", "subm/__pycache__/key.cpython-312.pyc": f"\x00\x00{TOKEN}"},
     {f"{SITE}/subm/__pycache__": ["dir", 0o755]}),
    ({"other.py": f"y{KEY}", f"{TOKEN}.py": "x"}, {}),
    ({}, {f"{SITE}/leak": ["link", f"/{KEY}/x"]}),
    ({"base_mod.py": f"KEY = {KEY!r}\n"}, {}),
], ids=["added", "changed-and-named", "link-target", "unchanged"])
def test_the_scan_of_an_archive_is_the_scan_of_its_extracted_tree(tmp_path, files, extra):
    base = {**BASE_TREE, f"{SITE}/base_mod.py": ["file", f"KEY = {KEY!r}\n", 0o644],
            f"{SITE}/other.py": ["file", "x" * len(f"y{KEY}"), 0o644]}
    image = _install_trees(files, extra)
    image[f"{SITE}/other.py"] = image.get(f"{SITE}/other.py", base[f"{SITE}/other.py"])
    lay_out(tmp_path / "base-tree", base)
    lay_out(tmp_path / "image-tree", image)
    extracted = scan.scan(VALUES, ("install", scan.added_entries(tmp_path / "base-tree", tmp_path / "image-tree")))
    # The layer holds the venv's directories and the base's module again, as a commit's does.
    rewritten = ("opt/venv", "opt/venv/lib", SITE, f"{SITE}/base_mod.py")
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, base, image, rewritten=rewritten)
    forms = scan.registry_forms(VALUES)
    entries = scan.layer_entries(open_image(base_archive, "sha256:" + base_manifest),
                                 open_image(image_archive, "sha256:" + image_manifest), forms)
    assert scan.scan(VALUES, ("install", entries)) == extracted
    assert [at for _, at, _ in entries] == [at for _, at, _ in scan.added_entries(tmp_path / "base-tree",
                                                                                     tmp_path / "image-tree")]


def test_the_scan_program_reads_two_archives(tmp_path, monkeypatch):
    base = {**BASE_TREE}
    image = _install_trees({"subm/key.py": f"KEY = {KEY!r}\n"})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, base, image, media_type=GZIP)
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "package-scan.json"
    assert scan.main(["--names", ",".join(VALUES), "--base-archive", str(base_archive), "--base-manifest",
                      base_manifest, "--image-archive", str(image_archive), "--image-manifest", image_manifest,
                      "--out", str(out), "--outputs-out", str(tmp_path / "outputs.json")]) == 0
    record = scan.PackageScan.model_validate_json(out.read_text())
    assert record.read == ["install"]
    assert sorted((f.path, f.secret) for f in record.findings) == [(f"/{SITE}/base_mod.py", "relay"),
                                                                   (f"/{SITE}/subm/key.py", "relay")]


def test_the_scan_refuses_an_image_whose_venv_is_no_directory(tmp_path):
    image = _changed(BASE_TREE, {"opt/venv": ["link", "/usr"]})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, BASE_TREE, image)
    with pytest.raises(ValueError, match="no directory at /opt/venv"):
        scan.layer_entries(open_image(base_archive, "sha256:" + base_manifest),
                           open_image(image_archive, "sha256:" + image_manifest), scan.registry_forms(VALUES))


def test_no_name_of_a_layer_is_joined_to_a_host_path(tmp_path, monkeypatch):
    """The reader writes nothing: every open of a file in a read is of the archive itself."""
    image = _install_trees({"subm/key.py": "x"})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, BASE_TREE, image)
    opened: list[str] = []
    real_open = io.open

    def watch(file, mode="r", *args, **kwargs):
        opened.append(f"{os.fspath(file)}:{mode}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", watch)
    archive_trees(base_archive, base_manifest, image_archive, image_manifest)
    assert {entry.rsplit(":", 1)[0] for entry in opened} <= {str(base_archive), str(image_archive)}
    assert all(entry.endswith(":rb") for entry in opened)


# The base read once: the integrity check leaves its tree for the scan.

def test_a_listing_holds_the_tree_it_was_written_from_and_only_for_its_manifest(tmp_path):
    base_archive, base_manifest, _, _ = _images(tmp_path, LINKED_BASE, LINKED_BASE)
    base = open_image(base_archive, "sha256:" + base_manifest)
    tree = Tree()
    for each in base.layers:
        apply(tree, read_layer(base, each))
    layers.write_listing(tmp_path / "listing.json", tree, base.manifest_digest)
    assert layers.read_listing(tmp_path / "listing.json", base.manifest_digest) == tree
    with pytest.raises(LayerError, match="is of the image"):
        layers.read_listing(tmp_path / "listing.json", "sha256:" + "0" * 64)
    (tmp_path / "cut.json").write_text((tmp_path / "listing.json").read_text()[:-10])
    with pytest.raises(LayerError, match="cannot be read"):
        layers.read_listing(tmp_path / "cut.json", base.manifest_digest)


def test_the_scan_reads_the_listing_the_integrity_check_left_and_finds_what_it_finds_from_the_layers(tmp_path,
                                                                                                    monkeypatch):
    image = _install_trees({"subm/key.py": f"KEY = {KEY!r}\n"})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, BASE_TREE, image)
    archives = ["--base-archive", str(base_archive), "--base-manifest", base_manifest,
                "--image-archive", str(image_archive), "--image-manifest", image_manifest]
    (tmp_path / "wheels").mkdir()
    listing = tmp_path / "integrity" / "base-listing.json"
    listing.parent.mkdir()
    assert integrity.main([*archives, "--base-listing-out", str(listing), "--wheels", str(tmp_path / "wheels"),
                           "--out", str(tmp_path / "integrity.json"), "--outputs-out", str(tmp_path / "i.json")]) == 0
    assert listing.is_file() and not listing.with_name(listing.name + ".partial").exists()
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)

    def scanned(*extra: str) -> list[tuple[str, str]]:
        out = tmp_path / "package-scan.json"
        assert scan.main(["--names", ",".join(VALUES), *archives, *extra, "--out", str(out),
                          "--outputs-out", str(tmp_path / "s.json")]) == 0
        return [(f.path, f.secret) for f in scan.PackageScan.model_validate_json(out.read_text()).findings]

    from_layers = scanned()
    # With the listing, no layer of the base is read: only the image's one.
    read: list[str] = []
    real = layers.read_layer

    def watched(image, layer, **kwargs):
        read.append(image.archive.name)
        return real(image, layer, **kwargs)

    monkeypatch.setattr(layers, "read_layer", watched)
    assert scanned("--base-listing", str(listing)) == from_layers == [(f"/{SITE}/base_mod.py", "relay"),
                                                                      (f"/{SITE}/subm/key.py", "relay")]
    assert read == ["image.tar"]


def test_the_scan_refuses_a_listing_of_another_base(tmp_path, monkeypatch, capsys):
    image = _install_trees({"subm/key.py": "x"})
    base_archive, base_manifest, image_archive, image_manifest = _images(tmp_path, BASE_TREE, image)
    layers.write_listing(tmp_path / "listing.json", Tree(), "sha256:" + "0" * 64)
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    assert scan.main(["--names", ",".join(VALUES), "--base-archive", str(base_archive), "--base-manifest",
                      base_manifest, "--base-listing", str(tmp_path / "listing.json"), "--image-archive",
                      str(image_archive), "--image-manifest", image_manifest, "--out", str(tmp_path / "s.json"),
                      "--outputs-out", str(tmp_path / "o.json")]) == 1
    assert "is of the image sha256:" + "0" * 64 in capsys.readouterr().err
    assert not (tmp_path / "s.json").exists()
