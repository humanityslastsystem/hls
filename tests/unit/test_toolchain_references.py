"""``scripts/assemble_references.py`` and ``hls.agent.toolchain_references.assemble``, which rebuild
tasks/assets/references from pinned public sources: their tables, the manifest the tasks' digest
is taken over, and the rules they build, swap in and check the folder by. Nothing here touches the
network."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
from pathlib import Path, PurePosixPath

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.agent.toolchain_references import assemble
from hls.agent.uploads import tree_digest
from hls.pipeline import commands

_SCRIPT = commands.checkout_root() / "scripts" / "assemble_references.py"
_spec = importlib.util.spec_from_file_location("assemble_references", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

UPLOADING = ("tasks.multitenant_slo.task", "tasks.tab_completion.task", "tasks.multimodal_tts.task")


def _library_path(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts and p.as_posix() == path


def test_the_manifest_is_the_library_the_tasks_pin():
    """Its lines are the ones the tasks' tree digest is taken over, so a file the command builds
    is checked against the file the tasks were run with, and the folder it puts in place is the
    one the tasks upload from."""
    pinned = {importlib.import_module(m).ARTIFACT.producer.upload_sha256 for m in UPLOADING}
    rows = assemble.read_manifest()
    assert len(rows) == 2565
    assert {assemble.manifest_digest(rows)} == pinned
    for module in UPLOADING:
        task_dir = PurePosixPath(*module.split(".")[:-1])
        [upload_dir] = importlib.import_module(module).ARTIFACT.producer.upload_dir
        assert os.path.normpath(task_dir / upload_dir) == assemble.LIBRARY_DIR.as_posix()
    assert all(_library_path(rel) and mode in (0o644, 0o755) and re.fullmatch(r"[0-9a-f]{64}", sha)
               for rel, (mode, sha) in rows.items())


def test_the_manifest_digest_is_the_tree_digest_of_the_files_it_lists(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.md").write_text("x\n")
    (tmp_path / "run.sh").write_text("#!/bin/sh\n")
    (tmp_path / "a" / "b" / "x.md").chmod(0o644)
    (tmp_path / "run.sh").chmod(0o755)
    rows = {"a/b/x.md": (0o644, hashlib.sha256(b"x\n").hexdigest()),
            "run.sh": (0o755, hashlib.sha256(b"#!/bin/sh\n").hexdigest())}
    assert assemble.manifest_digest(rows) == tree_digest(tmp_path)
    rows["run.sh"] = (0o644, rows["run.sh"][1])
    assert assemble.manifest_digest(rows) != tree_digest(tmp_path)


def test_the_sources_are_pinned_public_and_land_inside_the_library():
    copies = [c for s in assemble.GIT_SOURCES for c in s.copies]
    for source in assemble.GIT_SOURCES:
        assert source.repo.startswith("https://github.com/"), source.repo
        assert re.fullmatch(r"[0-9a-f]{40}", source.commit), source.commit
    for c in copies:
        assert _library_path(c.src) and _library_path(c.dest), c
        assert c.localize is None or _library_path(c.localize), c
    web = assemble.read_web()
    for rel, url, images in web:
        assert _library_path(rel) and url.startswith("https://"), (rel, url)
        assert images is None or _library_path(images), rel
        # An archive capture is named by its timestamp, so it serves the bytes it captured.
        assert "web.archive.org" not in url, url
    assert assemble.PTX_FIGURES.base.startswith("https://docs.nvidia.com/cuda/archive/")
    dests = [c.dest for c in copies] + [rel for rel, _, _ in web] + [assemble.PTX_FIGURES.images, "README.md"]
    assert len(dests) == len(set(dests))


def test_every_file_of_the_manifest_has_a_source_and_every_source_lands_one():
    """A file is the most specific source's whose destination holds it (triton's tutorials lie
    below its docs; the build refuses a path two sources write). A file no source lands, or a
    source that lands none, is a table out of step with the manifest."""
    covers = [(c.dest, False) for s in assemble.GIT_SOURCES for c in s.copies]
    covers += [(rel, True) for rel, _, _ in assemble.read_web()]
    covers += [(assemble.PTX_FIGURES.images, False), ("README.md", True)]
    landed = {dest: 0 for dest, _ in covers}
    for rel in assemble.read_manifest():
        under = [d for d, exact in covers if rel == d or (not exact and rel.startswith(d + "/"))]
        assert under, rel
        landed[max(under, key=len)] += 1
    assert [d for d, n in landed.items() if not n] == []


def test_remote_images_are_linked_to_the_local_images_directory():
    page = ('<link rel="preload" as="image" href="https://www.nvidia.com/content/dam/1x1-00000000.png">\n'
            '<img src="https://raw.githubusercontent.com/o/r/main/a/b/chart.png" alt="c" />\n'
            '<img src="_images/local.png">\n<a href="https://example.com/page.png">link</a>\n')
    assert assemble.localize_images(page, "cudnn/operations/Attention.html", "cudnn/_images") == (
        '<link rel="preload" as="image" href="../_images/1x1-00000000.png">\n'
        '<img src="../_images/chart.png" alt="c" />\n'
        '<img src="_images/local.png">\n<a href="https://example.com/page.png">link</a>\n')
    readme = "[![v](https://img.shields.io/pypi/v/pkg.svg?style=flat)](https://pypi.org/p/)\n![x](./y.png)\n"
    assert assemble.localize_images(readme, "cudnn/README.md", "cudnn/_images") == (
        "[![v](_images/pkg.svg)](https://pypi.org/p/)\n![x](./y.png)\n")
    # A markdown image is only read as one in a markdown file.
    assert assemble.localize_images("![v](https://a/b.svg)", "cudnn/x.html", "cudnn/_images") == "![v](https://a/b.svg)"


def test_figures_are_every_image_the_markdown_links_under_images(tmp_path):
    (tmp_path / "ptx" / "docs" / "ch9").mkdir(parents=True)
    (tmp_path / "ptx" / "docs" / "ch9" / "a.md").write_text("![f](../../_images/one.png) ![g](../../_images/two.png)\n")
    (tmp_path / "ptx" / "top.md").write_text("![f](_images/one.png) [ref](other.md) ![w](https://x/y.png)\n")
    spec = assemble.Figures(markdown="ptx", images="ptx/_images", base="https://host/_images/")
    assert assemble.figures(spec, tmp_path) == [("ptx/_images/one.png", "https://host/_images/one.png"),
                                                ("ptx/_images/two.png", "https://host/_images/two.png")]


def test_the_comparison_names_every_file_that_is_not_the_pinned_one(tmp_path, capsys):
    files = {"a/same.md": (b"same\n", 0o644), "a/changed.md": (b"new\n", 0o644), "a/mode.sh": (b"#!\n", 0o644),
             "b/extra.md": (b"extra\n", 0o644)}
    for rel, (data, mode) in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
        (tmp_path / rel).chmod(mode)
    manifest = {"a/same.md": (0o644, hashlib.sha256(b"same\n").hexdigest()),
                "a/changed.md": (0o644, hashlib.sha256(b"old\n").hexdigest()),
                "a/mode.sh": (0o755, hashlib.sha256(b"#!\n").hexdigest()),
                "b/gone.md": (0o644, hashlib.sha256(b"gone\n").hexdigest())}
    assert assemble.compare(tmp_path, manifest) is False
    out = capsys.readouterr().out
    assert "a/: 1 identical, 2 differing" in out and "b/: 1 missing, 1 extra" in out
    assert "differing: a/changed.md: 4 bytes" in out and "differing: a/mode.sh: mode 644, pinned 755" in out
    assert "missing: b/gone.md" in out and "extra: b/extra.md" in out
    exact = {"a/same.md": manifest["a/same.md"]}
    for rel in ("a/changed.md", "a/mode.sh", "b/extra.md"):
        (tmp_path / rel).unlink()
    assert assemble.compare(tmp_path, exact) is True


def _library(files: dict[str, bytes]) -> dict[str, tuple[int, str]]:
    """The manifest of a library holding *files*, each with mode 644."""
    return {rel: (0o644, hashlib.sha256(data).hexdigest()) for rel, data in files.items()}


class FakeAssemble:
    """Stands in for ``assemble``: writes *files* into the stage, or raises *error* after writing
    one, and records the cache it was given."""

    def __init__(self, files: dict[str, bytes], error: Exception | None = None) -> None:
        self.files, self.error, self.caches = files, error, []

    def __call__(self, stage, cache, *, echo):
        self.caches.append(cache)
        for rel, data in self.files.items():
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_bytes(data)
            (stage / rel).chmod(0o644)
            if self.error is not None:
                raise self.error


FILES = {"README.md": b"index\n", "a/x.md": b"x\n"}


def test_the_library_is_built_beside_the_destination_and_swapped_in(tmp_path):
    dest = tmp_path / "assets" / "references"
    dest.mkdir(parents=True)
    # A library this command wrote, with a file the pinned one lacks.
    (dest / "README.md").write_bytes(assemble.LIBRARY_README.read_bytes())
    (dest / "stale.md").write_text("old\n")
    build, lines = FakeAssemble(FILES), []
    assert assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=build,
                                     echo=lines.append) is True
    assert build.caches == [tmp_path / "cache"]
    assert tree_digest(dest) == assemble.manifest_digest(_library(FILES))
    assert sorted(p.name for p in dest.parent.iterdir()) == ["references"]
    assert lines[-1].endswith("matches the pinned digest")


def test_a_destination_that_has_the_pinned_digest_is_left_as_it_is(tmp_path):
    dest = tmp_path / "references"
    dest.mkdir()
    FakeAssemble(FILES)(dest, None, echo=None)
    build = FakeAssemble({})
    assert assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=build,
                                     echo=lambda line: None) is True
    assert build.caches == []


def test_a_library_that_cannot_be_built_leaves_the_destination_as_it_was(tmp_path):
    dest = tmp_path / "references"
    dest.mkdir()
    (dest / "kept.md").write_text("kept\n")
    build = FakeAssemble(FILES, error=assemble.AssemblyError("https://host/x cannot be downloaded"))
    with pytest.raises(assemble.AssemblyError):
        assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=build,
                                  echo=lambda line: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["references"]
    assert [p.name for p in dest.iterdir()] == ["kept.md"]


def test_a_library_with_another_digest_is_named_file_by_file(tmp_path):
    dest = tmp_path / "references"
    built = {**FILES, "a/x.md": b"changed upstream\n"}
    lines: list[str] = []
    assert assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=FakeAssemble(built),
                                     echo=lines.append) is False
    assert any(line.startswith("  differing: a/x.md:") for line in lines)
    assert "the tasks refuse this folder" in lines[-1]


def test_the_script_assembles_into_the_checkout_and_exits_by_the_result(monkeypatch, tmp_path):
    seen = {}

    def assemble_library(dest, cache, *, build, echo=print):
        seen.update(dest=dest, cache=cache, build=build)
        return seen.get("exact", True)

    monkeypatch.setattr(assemble, "assemble_library", assemble_library)
    assert script.main([]) == 0
    assert seen == {"dest": commands.checkout_root() / "tasks" / "assets" / "references",
                    "cache": assemble.DEFAULT_CACHE, "build": assemble.assemble}
    seen["exact"] = False
    assert script.main(["--dest", str(tmp_path / "lib"), "--cache", str(tmp_path / "cache")]) == 1
    assert (seen["dest"], seen["cache"]) == (tmp_path / "lib", tmp_path / "cache")


def test_the_script_refuses_a_library_it_cannot_assemble_in_one_line(monkeypatch, tmp_path, capsys):
    def fail(dest, cache, *, build, echo=print):
        raise assemble.AssemblyError("https://host/x cannot be downloaded: 404")

    monkeypatch.setattr(assemble, "assemble_library", fail)
    assert script.main(["--dest", str(tmp_path / "lib")]) == 1
    assert capsys.readouterr().err == ("assemble_references.py: cannot assemble the library: https://host/x cannot "
                                       "be downloaded: 404\n")


def test_hls_has_no_assemble_references_command():
    """The script is the only entry point."""
    assert CliRunner().invoke(cli.app, ["assemble-references"]).exit_code == 2


@pytest.mark.parametrize("contents", [{"thesis.md": b"mine\n"}, {"README.md": b"# my notes\n"}])
def test_a_destination_this_command_did_not_write_is_refused_and_left_as_it_was(tmp_path, contents):
    dest = tmp_path / "references"
    dest.mkdir()
    for rel, data in contents.items():
        (dest / rel).write_bytes(data)
    with pytest.raises(assemble.AssemblyError, match="is not a library this command wrote"):
        assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=_no_network,
                                  echo=lambda line: None)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == contents
    assert sorted(p.name for p in tmp_path.iterdir()) == ["references"]


def test_a_file_or_a_dangling_link_as_destination_is_refused(tmp_path):
    (tmp_path / "file").write_text("x")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    for dest in (tmp_path / "file", tmp_path / "dangling"):
        with pytest.raises(assemble.AssemblyError, match="is not a library this command wrote"):
            assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(FILES), build=_no_network,
                                      echo=lambda line: None)


def test_a_library_this_command_wrote_is_replaced(tmp_path):
    """One whose README.md is the library's index, though its other files differ."""
    dest = tmp_path / "references"
    dest.mkdir()
    (dest / "README.md").write_bytes(assemble.LIBRARY_README.read_bytes())
    (dest / "stale.md").write_text("old\n")
    files = {"README.md": assemble.LIBRARY_README.read_bytes(), "a/x.md": b"x\n"}
    assert assemble.assemble_library(dest, tmp_path / "cache", manifest=_library(files), build=FakeAssemble(files),
                                     echo=lambda line: None) is True
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == ["README.md", "a/x.md"]


def test_the_script_refuses_the_checkout_itself_as_destination(tmp_path, capsys):
    assert script.main(["--dest", str(commands.checkout_root()), "--cache", str(tmp_path / "cache")]) == 1
    assert capsys.readouterr().err.startswith(f"assemble_references.py: cannot assemble the library: "
                                              f"{commands.checkout_root()} is there and is not a library this "
                                              "command wrote")
    assert (commands.checkout_root() / "pyproject.toml").is_file()


def _no_network(*args, **kwargs):
    """Stands in for the build a test does not expect to be reached."""
    raise AssertionError(f"a test reached the library's build: {args}")


# NVIDIA's archive host and the captures of the cuDNN pages.

ARCHIVE_PAGE = "https://archive.docs.nvidia.com/deeplearning/cudnn/v1.28.0/developer/misc.html"
PAGE = ('<html><head>\n<meta charset="utf-8">\n  </head>\n<body>\n    <header id="pst-header">h</header>\n'
        '<img src="https://img.shields.io/badge/x.svg"></body></html>\n')
STYLE = ('<!-- archive-s3-banner-style:start --><style id="archive-s3-banner-style">.a{color:#76b900}</style>'
         '<!-- archive-s3-banner-style:end -->')
BANNER = ('<!-- archive-s3-banner:start --><aside class="archive-s3-banner">Archived</aside>'
          '<!-- archive-s3-banner:end -->')
WRAPPED = PAGE.replace("  </head>", f"  {STYLE}</head>").replace('<header id', f'{BANNER}<header id')


def test_the_archive_hosts_banners_are_removed_by_their_markers():
    assert WRAPPED != PAGE
    assert assemble.unwrap_archive_page(WRAPPED, ARCHIVE_PAGE) == PAGE


def test_a_page_of_another_host_or_without_both_markers_is_left_as_fetched():
    assert assemble.unwrap_archive_page(WRAPPED, ARCHIVE_PAGE.replace("archive.docs", "docs")) == WRAPPED
    # A start marker with no end marker after it removes nothing.
    unended = WRAPPED.replace("<!-- archive-s3-banner:end -->", "")
    assert assemble.unwrap_archive_page(unended, ARCHIVE_PAGE) == unended.replace(STYLE, "")
    assert assemble.unwrap_archive_page(PAGE, ARCHIVE_PAGE) == PAGE


def _downloads(monkeypatch, answers: dict[str, object]) -> list[str]:
    """Stands in for ``download``: each url answers its bytes, or raises the AssemblyError given.
    Returns the urls asked, in order; a url not listed fails the test."""
    asked = []

    def download(url):
        asked.append(url)
        answer = answers[url]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(assemble, "download", download)
    return asked


def _kept(cache: Path) -> list[str]:
    return sorted(p.name for p in (cache / "web").glob("*")) if (cache / "web").is_dir() else []


def test_an_archive_page_is_unwrapped_converted_and_kept_once_it_is_the_pinned_file(monkeypatch, tmp_path):
    rel = "cudnn/developer/misc.html"
    data = assemble.converted(rel, ARCHIVE_PAGE, WRAPPED.encode(), "cudnn/_images")
    assert data == assemble.localize_images(PAGE, rel, "cudnn/_images").encode()
    asked = _downloads(monkeypatch, {ARCHIVE_PAGE: WRAPPED.encode()})
    sha = hashlib.sha256(data).hexdigest()
    assert assemble.fetch(rel, ARCHIVE_PAGE, "cudnn/_images", tmp_path, sha) == data
    assert _kept(tmp_path) == [hashlib.sha256(ARCHIVE_PAGE.encode()).hexdigest()]
    # A second run reads it from the cache.
    assert assemble.fetch(rel, ARCHIVE_PAGE, "cudnn/_images", tmp_path, sha) == data and asked == [ARCHIVE_PAGE]
    # An image of the archive host is copied as downloaded, though its bytes are no text.
    png = b"\x89PNG\r\n\x1a\n\xff\xfe"
    assert assemble.converted("cudnn/_images/x.png", ARCHIVE_PAGE.replace("developer/misc.html", "_images/x.png"),
                              png, None) == png


def test_a_body_that_is_not_the_pinned_file_is_refused_and_not_kept(monkeypatch, tmp_path):
    """A banner the markers do not name, or another build of the page: refused in one line naming
    the page, the pinned digest and what the source gave, and nothing is cached."""
    rel = "cudnn/developer/misc.html"
    pinned = hashlib.sha256(assemble.converted(rel, ARCHIVE_PAGE, WRAPPED.encode(), "cudnn/_images")).hexdigest()
    other = WRAPPED.replace("archive-s3-banner:end", "banner:end").encode()
    got = hashlib.sha256(assemble.converted(rel, ARCHIVE_PAGE, other, "cudnn/_images")).hexdigest()
    _downloads(monkeypatch, {ARCHIVE_PAGE: other})
    with pytest.raises(assemble.AssemblyError, match=rf"^{re.escape(rel)} from {re.escape(ARCHIVE_PAGE)} has sha256 "
                                                     rf"{got}, not the pinned {pinned}$"):
        assemble.fetch(rel, ARCHIVE_PAGE, "cudnn/_images", tmp_path, pinned)
    assert _kept(tmp_path) == []


def test_a_kept_download_that_is_not_the_pinned_file_is_removed_and_fetched_again(monkeypatch, tmp_path):
    rel = "cudnn/developer/misc.html"
    data = assemble.converted(rel, ARCHIVE_PAGE, WRAPPED.encode(), "cudnn/_images")
    kept = tmp_path / "web" / hashlib.sha256(ARCHIVE_PAGE.encode()).hexdigest()
    kept.parent.mkdir()
    kept.write_bytes(b"another build of the page")
    asked = _downloads(monkeypatch, {ARCHIVE_PAGE: WRAPPED.encode()})
    assert assemble.fetch(rel, ARCHIVE_PAGE, "cudnn/_images", tmp_path, hashlib.sha256(data).hexdigest()) == data
    assert asked == [ARCHIVE_PAGE] and kept.read_bytes() == WRAPPED.encode()


def test_a_download_that_cannot_be_had_is_refused_in_one_line(monkeypatch, tmp_path):
    def refused(request, timeout):
        raise assemble.urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(assemble.urllib.request, "urlopen", refused)
    with pytest.raises(assemble.AssemblyError, match=rf"^{re.escape(ARCHIVE_PAGE)} cannot be downloaded: HTTP Error 404"):
        assemble.fetch("cudnn/developer/misc.html", ARCHIVE_PAGE, "cudnn/_images", tmp_path, "0" * 64)
    assert _kept(tmp_path) == []


def test_every_cudnn_file_is_the_v1_28_0_release_on_the_archive_host_or_named_here():
    """The cuDNN pages, images and logos are the v1.28.0 build at its versioned path on NVIDIA's
    archive host. The images the pages and cudnn-frontend's README show from elsewhere are named
    here: cudnn-frontend's benchmark figures at the commit of its v1.28.0 release, two badges,
    a badge of that release's version and a pixel."""
    root = f"https://{assemble.ARCHIVE_HOST}/deeplearning/cudnn/v1.28.0/"
    files = {rel: url for rel, url, _ in assemble.read_web() if rel.startswith("cudnn/")}
    pages = [rel for rel, _, images in assemble.read_web() if images]
    assert len(pages) == 76 and all(rel.startswith("cudnn/") and rel.endswith(".html") for rel in pages)
    elsewhere = {rel: url for rel, url in files.items() if url != root + rel.removeprefix("cudnn/")}
    frontend = "https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/91159779637b672a3f768738f5b30eee33f2180d/"
    assert all(url.startswith(frontend) for rel, url in elsewhere.items() if rel.endswith("_flops.png")
               or "top_left" in rel or "no_mask" in rel)
    assert sorted(url for url in elsewhere.values() if not url.startswith(frontend)) == [
        "https://img.shields.io/badge/License-Apache_2.0-blue.svg",
        "https://img.shields.io/badge/docs-nvidia.github.io-blue.svg",
        "https://img.shields.io/badge/pypi-v1.28.0-blue.svg",
        "https://www.nvidia.com/content/dam/1x1-00000000.png"]
    assert next(s.commit for s in assemble.GIT_SOURCES if s.repo.endswith("/cudnn-frontend")) == frontend.split("/")[-2]


def test_no_source_is_a_capture_and_every_other_one_is_versioned_pinned_or_named_here():
    """The CUDA library pages and figures are CUDA 13.3.1's; the CUDA site's three logos have no
    versioned copy."""
    pinned_at_commit = re.compile(r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/")
    urls = [url for _, url, _ in assemble.read_web()]
    assert not [url for url in urls if "web.archive.org" in url]
    other = sorted(url for url in urls if not url.startswith(f"https://{assemble.ARCHIVE_HOST}/deeplearning/cudnn/v1.28.0/")
                   and not url.startswith("https://docs.nvidia.com/cuda/archive/13.3.1/")
                   and not pinned_at_commit.match(url))
    assert other == ["https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png",
                     "https://docs.nvidia.com/cuda/_static/NVIDIA-LogoBlack.svg",
                     "https://docs.nvidia.com/cuda/_static/NVIDIA-LogoWhite.svg",
                     "https://img.shields.io/badge/License-Apache_2.0-blue.svg",
                     "https://img.shields.io/badge/docs-nvidia.github.io-blue.svg",
                     "https://img.shields.io/badge/pypi-v1.28.0-blue.svg",
                     "https://www.nvidia.com/content/dam/1x1-00000000.png"]
    assert assemble.PTX_FIGURES.base.startswith("https://docs.nvidia.com/cuda/archive/13.3.1/")
