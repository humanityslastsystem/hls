"""Assemble ``tasks/assets/references`` (``scripts/assemble_references.py``), the offline toolchain
documentation that multitenant_slo, tab_completion and multimodal_tts upload into every build's
sandbox, from public sources.

The folder is not committed, and the three tasks pin its tree digest
(``AgentProducerSpec.upload_sha256``, taken by ``hls.agent.uploads.tree_digest``). Each part of
it is fetched from a pinned source:
- ``GIT_SOURCES``: a public repository at a commit, and the paths of it copied in;
- ``PTX_FIGURES``: the figures the PTX skill's markdown links, from NVIDIA's archive of the
  CUDA 13.3.1 documentation;
- ``web.tsv`` beside this module: every other downloaded file, one line each, ``<path in the
  library>\\t<url>`` plus, for a cuDNN page, ``\\t<images directory>`` (``localize_images``):
  NVIDIA's CUDA library pages and their figures from its CUDA 13.3.1 documentation, the cuDNN
  v1.28.0 pages, images and logos from ``ARCHIVE_HOST``, the CUDA site's logos, two shields.io
  badges and a pixel of www.nvidia.com, and cudnn-frontend's benchmark images at the commit of
  its v1.28.0 release;
- ``library-README.md`` beside this module: the library's own index.

``manifest.tsv`` beside this module holds the path, permission bits and sha256 of every file of
the library the tasks pin, the lines its tree digest is taken over: a download is kept only when
it is the pinned file, and a source that has changed since is named.

The library is built in a staging directory beside its destination and swapped in whole once
every source has been copied, so a run that fails leaves the destination as it was; a
destination that already has the pinned digest is left alone. Since the swap replaces the
destination whole, a destination that holds anything but a library this command wrote (its
``README.md`` is ``library-README.md``) is refused; an empty or missing one is built. Clones and downloads are kept in a
cache directory, so a rerun fetches nothing new. At the end every file is compared with the
manifest, per top-level directory, and the destination's tree digest with the pinned one. No
credential is used: every source is public."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from urllib.parse import urlsplit

from hls.agent.uploads import TreeDigestError, tree_digest

DATA = Path(__file__).parent
MANIFEST = DATA / "manifest.tsv"
WEB = DATA / "web.tsv"
LIBRARY_README = DATA / "library-README.md"
#: Where the library lies below the checkout: the tasks' ``upload_dir``, ``../assets/references``.
LIBRARY_DIR = Path("tasks") / "assets" / "references"
#: The command, run from the checkout's root, that puts the library at ``LIBRARY_DIR``.
ASSEMBLE_COMMAND = "uv run python scripts/assemble_references.py"
DEFAULT_CACHE = Path.home() / ".cache" / "hls-references"

#: The User-Agent of every download.
USER_AGENT = "hls-assemble-references"
#: The host NVIDIA serves its retired documentation versions from. It wraps each page in an
#: announcement banner: a style block before ``</head>`` and a banner before the page header, each
#: between the literal start and end markers of ``ARCHIVE_BANNERS``. The manifest's hash is of the
#: page without them, so they are removed from a page of this host before it is converted.
ARCHIVE_HOST = "archive.docs.nvidia.com"
ARCHIVE_BANNERS = (("<!-- archive-s3-banner-style:start -->", "<!-- archive-s3-banner-style:end -->"),
                   ("<!-- archive-s3-banner:start -->", "<!-- archive-s3-banner:end -->"))
#: Attempts per download, and the wait before the n-th retry (n times this many seconds): a
#: documentation host drops a request now and then under load.
FETCH_ATTEMPTS = 6
FETCH_BACKOFF_SEC = 10
#: Downloads run at once.
FETCH_WORKERS = 8


class AssemblyError(RuntimeError):
    """A source that cannot be fetched or copied as the tables say."""


@dataclass(frozen=True)
class Copy:
    """One path of a repository copied into the library: the file *src* to the library path
    *dest*, or every file below the directory *src* to the same relative path below *dest*, less
    those whose path below *src* matches a glob of *exclude* (``fnmatch``, whose ``*`` also
    matches ``/``). *localize* is the library directory a file's remote images are linked to
    (``localize_images``), or None to copy the bytes as they are."""

    src: str
    dest: str
    exclude: tuple[str, ...] = ()
    localize: str | None = None


@dataclass(frozen=True)
class GitSource:
    """A public repository *repo* (an https URL) at the full commit id *commit*, and what the
    library copies from it."""

    repo: str
    commit: str
    copies: tuple[Copy, ...]


@dataclass(frozen=True)
class Figures:
    """Images a markdown tree links by name under one directory: every ``_images/<name>`` that a
    markdown file below the library directory *markdown* links is fetched from *base* + name into
    the library directory *images*."""

    markdown: str
    images: str
    base: str


# Each source is pinned to a commit of its default branch. What a Sphinx project needs only to
# build its site (conf.py, templates, extensions, stylesheets, makefiles) and the projects'
# contributor and meeting notes are left out.
GIT_SOURCES: tuple[GitSource, ...] = (
    # The PTX ISA, CUDA runtime and driver API references and the profiling guides, converted to
    # markdown, with their image links already pointing at a local _images/ (PTX_FIGURES).
    GitSource("https://github.com/shg8/ptx-isa-markdown", "8ac5f8e150fe2693a8fd4348d129abaecfc725a6", (
        Copy("cuda_skill/references/ptx-docs", "ptx/ptx-docs"),
        Copy("cuda_skill/references/ptx-isa.md", "ptx/ptx-isa.md"),
        Copy("cuda_skill/references/cuda-runtime-docs", "cuda-runtime/cuda-runtime-docs"),
        Copy("cuda_skill/references/cuda-runtime.md", "cuda-runtime/cuda-runtime.md"),
        Copy("cuda_skill/references/cuda-driver-docs", "cuda-driver/cuda-driver-docs"),
        Copy("cuda_skill/references/cuda-driver.md", "cuda-driver/cuda-driver.md"),
        Copy("cuda_skill/references/debugging-tools.md", "profiling/debugging-tools.md"),
        Copy("cuda_skill/references/ncu-guide.md", "profiling/ncu-guide.md"),
        Copy("cuda_skill/references/nsys-guide.md", "profiling/nsys-guide.md"),
        Copy("cuda_skill/references/nvtx-patterns.md", "profiling/nvtx-patterns.md"),
        Copy("cuda_skill/references/performance-traps.md", "profiling/performance-traps.md"),
    )),
    GitSource("https://github.com/NVIDIA/cccl", "6e0de672764482bb2d4c69e1456f9ccfed3a347a", (
        Copy("docs", "cccl", exclude=("conf.py", "_ext/*", "_static/*", "_templates/*", "contributors/*",
                                      "infrastructure/*", "maintainers/*")),
    )),
    GitSource("https://github.com/NVIDIA/cuda-python", "608cc5fbb053fdc862ae14cda18d09c9f0cae8cc", tuple(
        Copy(f"{package}/docs", f"cuda-python/{package}",
             exclude=("Makefile", "make.bat", "source/conf.py", "source/_templates/*"))
        for package in ("cuda_bindings", "cuda_core", "cuda_pathfinder", "cuda_python"))),
    GitSource("https://github.com/NVIDIA/cutile-python", "d329de36725a2553fa4e0774614033dac7600378", (
        Copy("docs/source", "cutile", exclude=("conf.py", "_ext/*", "_templates/*", "_static/.gitkeep",
                                               "_static/css/*", "*.license")),
    )),
    GitSource("https://github.com/NVIDIA/cutlass", "f94ec46f4f63f96003d6cfdf2014731e7672c281", (
        Copy("media/docs", "cutlass/docs"),
        Copy("media/images", "cutlass/images"),
        Copy("python/docs_src/source", "cutlass/pythonDSL-api", exclude=("conf.py", "_static/*", "_templates/*")),
    )),
    GitSource("https://github.com/triton-lang/triton", "84cae05df02548bf0bb0a286d65b4943a1acbb2f", (
        Copy("docs", "triton", exclude=("conf.py", "Makefile", "_templates/*", "meetups/*")),
        Copy("python/tutorials", "triton/tutorials"),
    )),
    GitSource("https://github.com/pytorch/pytorch", "1424eca99808db6c0081736579526f13ed20c87d", (
        Copy("docs/source", "torch/source", exclude=("conf.py", "_templates/*", "_static/css/*")),
    )),
    GitSource("https://github.com/rust-lang/reference", "bec6b5e6631be1e45ed905c9e058ffa7ecd9eb62", (
        Copy("src", "rust/reference"),
    )),
    GitSource("https://github.com/rust-lang/book", "917544888a55e4da7109bdba8c88c893c0da70f4", (
        Copy("src", "rust/book"),
    )),
    # The README of cudnn-frontend's v1.28.0 release, its remote images linked to the
    # cuDNN pages' _images/ (web.tsv); the library's cuDNN pages are the same release's.
    GitSource("https://github.com/NVIDIA/cudnn-frontend", "91159779637b672a3f768738f5b30eee33f2180d", (
        Copy("README.md", "cudnn/cudnn-frontend-README.md", localize="cudnn/_images"),
    )),
)

#: The PTX skill's markdown links its figures as ``_images/<name>``: ptx-isa-markdown's
#: materialize_images.py rewrote NVIDIA's links so, and keeps the downloaded folder out of the
#: repository. They are the PTX ISA documentation's figures as CUDA 13.3.1 ships them.
PTX_FIGURES = Figures(markdown="ptx", images="ptx/_images",
                      base="https://docs.nvidia.com/cuda/archive/13.3.1/parallel-thread-execution/_images/")

# A remote image a page links: a markdown image, an <img> and an image a <link> preloads.
_MARKDOWN_IMAGE = re.compile(r"(!\[[^\]]*\]\()(https?://[^)\s]+)(\))")
_HTML_IMAGES = (re.compile(r'(<img\b[^>]*?\bsrc=")(https?://[^"]+)(")'),
                re.compile(r'(<link\b[^>]*?\bas="image"[^>]*?\bhref=")(https?://[^"]+)(")'))
# A markdown link to a file of an _images/ directory, by a relative path.
_LOCAL_FIGURE = re.compile(r"\]\((?:\.\./|\./)*_images/([^)/\s]+)\)")


def localize_images(text: str, path: str, images: str) -> str:
    """*text*, the file at library path *path*, with each remote image it links (``https://...``
    in a markdown image of a ``.md`` file, an ``<img>`` or an image ``<link rel="preload">``)
    pointed at the file of the same name in the library directory *images*, by a path relative
    to the file's directory. The images themselves are listed in web.tsv, each stored under its
    URL's last path segment, so links of one name share one file."""

    def local(match: re.Match[str]) -> str:
        name = match.group(2).split("?", 1)[0].rsplit("/", 1)[-1]
        return match.group(1) + os.path.relpath(f"{images}/{name}", os.path.dirname(path) or ".") + match.group(3)

    for pattern in _HTML_IMAGES:
        text = pattern.sub(local, text)
    if path.endswith(".md"):
        text = _MARKDOWN_IMAGE.sub(local, text)
    return text


def unwrap_archive_page(text: str, url: str) -> str:
    """*text*, the page at *url*, without the banner blocks ``ARCHIVE_HOST`` wraps its pages in:
    for each pair of ``ARCHIVE_BANNERS``, the first start marker through the end marker after it.
    A page of another host, or a pair whose markers are not both there, is left as it is."""
    if urlsplit(url).hostname != ARCHIVE_HOST:
        return text
    for start_marker, end_marker in ARCHIVE_BANNERS:
        start = text.find(start_marker)
        end = text.find(end_marker, start) if start >= 0 else -1
        if end >= 0:
            text = text[:start] + text[end + len(end_marker):]
    return text


def read_manifest(path: Path = MANIFEST) -> dict[str, tuple[int, str]]:
    """The manifest's files: library path to (permission bits, sha256)."""
    rows = {}
    for line in path.read_text().splitlines():
        rel, mode, sha = line.split("\t")
        rows[rel] = (int(mode, 8), sha)
    return rows


def manifest_digest(rows: dict[str, tuple[int, str]]) -> str:
    """The tree digest of a folder holding exactly the files *rows* describes, computed as
    ``hls.agent.uploads.tree_digest`` computes it from the files."""
    lines = sorted(f"{rel}\0{mode:o}\0{sha}\n" for rel, (mode, sha) in rows.items())
    return hashlib.sha256("".join(lines).encode("utf-8", "surrogateescape")).hexdigest()


def read_web(path: Path = WEB) -> list[tuple[str, str, str | None]]:
    """web.tsv's lines: library path, url, and the images directory a page is localized to."""
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        rows.append((fields[0], fields[1], fields[2] if len(fields) > 2 else None))
    return rows


def _git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise AssemblyError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def checkout(source: GitSource, cache: Path) -> Path:
    """A working tree of *source*'s repository at its commit holding the paths it copies, under
    *cache*: a partial clone of the one commit (no history, and only the blobs of those paths),
    made once and reused."""
    name = re.sub(r"[^A-Za-z0-9]+", "-", source.repo.removeprefix("https://github.com/")).strip("-")
    tree = cache / "git" / f"{name}-{source.commit[:12]}"
    if not (tree / ".git").is_dir():
        tree.mkdir(parents=True, exist_ok=True)
        _git("init", "--quiet", str(tree))
    if "origin" not in _git("remote", cwd=tree).split():
        _git("remote", "add", "origin", source.repo, cwd=tree)
    _git("sparse-checkout", "set", "--no-cone", *(f"/{c.src}" for c in source.copies), cwd=tree)
    head = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "HEAD"], cwd=tree,
                          capture_output=True, text=True).stdout.strip()
    if head != source.commit:
        _git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", source.commit, cwd=tree)
    _git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--force", "--detach", source.commit, cwd=tree)
    return tree


def _remove(path: Path) -> None:
    """Remove *path*, a link or file as itself and a directory with what it holds, if it is there."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _write(stage: Path, rel: str, data: bytes, mode: int) -> None:
    path = stage / rel
    if path.exists():
        raise AssemblyError(f"{rel} is written by two sources")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def copy_git(source: GitSource, tree: Path, stage: Path) -> int:
    """Copy *source*'s paths from its working tree *tree* into *stage*, each file with mode 755
    when git records it executable and 644 otherwise; return how many files were copied."""
    copied = 0
    for c in source.copies:
        listing = _git("ls-tree", "-r", "-z", "--full-tree", source.commit, "--", c.src, cwd=tree)
        entries = [e.split("\t", 1) for e in listing.split("\0") if e]
        if not entries:
            raise AssemblyError(f"{source.repo}@{source.commit} has no {c.src}")
        for meta, path in entries:
            git_mode, kind, _oid = meta.split()
            below = PurePosixPath(path).relative_to(c.src).as_posix() if path != c.src else ""
            if below and any(fnmatch.fnmatchcase(below, glob) for glob in c.exclude):
                continue
            if kind != "blob" or git_mode not in ("100644", "100755"):
                raise AssemblyError(f"{source.repo}@{source.commit}: {path} is a {git_mode} {kind}, not a file")
            dest = f"{c.dest}/{below}" if below else c.dest
            data = (tree / path).read_bytes()
            if c.localize:
                data = localize_images(data.decode("utf-8"), dest, c.localize).encode("utf-8")
            _write(stage, dest, data, 0o755 if git_mode == "100755" else 0o644)
            copied += 1
    return copied


def download(url: str) -> bytes:
    """The bytes at *url*. A refusal other than 429 (too many requests) is final; any other
    failure is retried. Raise AssemblyError when the bytes cannot be had."""
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    data = gzip.decompress(data)
            return data
        except OSError as e:
            final = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429
            if final or attempt == FETCH_ATTEMPTS:
                raise AssemblyError(f"{url} cannot be downloaded: {e}") from e
            time.sleep(FETCH_BACKOFF_SEC * attempt)
    raise AssertionError("unreachable")


def fetch(rel: str, url: str, images: str | None, cache: Path, sha256: str) -> bytes:
    """The bytes library path *rel* holds, the file of sha256 *sha256*: *url*'s download,
    ``converted``. A download is kept under *cache* only once it is that file; a kept one that
    is not is removed and downloaded again. Raise AssemblyError, naming *rel* and both digests,
    when the download is another file."""
    kept = cache / "web" / hashlib.sha256(url.encode()).hexdigest()
    if kept.is_file():
        data = converted(rel, url, kept.read_bytes(), images)
        if hashlib.sha256(data).hexdigest() == sha256:
            return data
        kept.unlink()
    raw = download(url)
    data = converted(rel, url, raw, images)
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise AssemblyError(f"{rel} from {url} has sha256 {got}, not the pinned {sha256}")
    kept.parent.mkdir(parents=True, exist_ok=True)
    partial = kept.with_name(kept.name + f".{os.getpid()}.{threading.get_ident()}.partial")
    partial.write_bytes(raw)
    partial.replace(kept)
    return data


def figures(spec: Figures, stage: Path) -> list[tuple[str, str]]:
    """The images the markdown below *spec*.markdown in *stage* links under ``_images/``: their
    library path and the url they are fetched from."""
    names = set()
    for md in sorted((stage / spec.markdown).rglob("*.md")):
        names.update(_LOCAL_FIGURE.findall(md.read_text(encoding="utf-8")))
    return [(f"{spec.images}/{name}", spec.base + name) for name in sorted(names)]


def converted(rel: str, url: str, data: bytes, images: str | None) -> bytes:
    """The bytes library path *rel* holds, from *data* downloaded from *url*: an HTML page of
    ``ARCHIVE_HOST`` without its banners, and a page with an images directory *images* with its
    remote images linked to it (``localize_images``). Any other file is as downloaded."""
    if urlsplit(url).hostname == ARCHIVE_HOST and urlsplit(url).path.endswith(".html"):
        data = unwrap_archive_page(data.decode("utf-8"), url).encode("utf-8")
    if images:
        data = localize_images(data.decode("utf-8"), rel, images).encode("utf-8")
    return data


def assemble(stage: Path, cache: Path, echo: Callable[[str], None] = print) -> None:
    """Build the whole library into the empty directory *stage*. Raise AssemblyError when a
    download is not the file the manifest pins from any of its sources (``fetch``)."""
    for source in GIT_SOURCES:
        echo(f"{source.repo} @ {source.commit[:12]}")
        copied = copy_git(source, checkout(source, cache), stage)
        echo(f"  {copied} files")
    downloads = [(rel, url, None) for rel, url in figures(PTX_FIGURES, stage)] + read_web()
    manifest = read_manifest()
    echo(f"{len(downloads)} downloads")
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        fetched = pool.map(lambda row: fetch(row[0], row[1], row[2], cache, manifest.get(row[0], (0, ""))[1]),
                           downloads)
        for (rel, _url, _images), data in zip(downloads, fetched, strict=True):
            _write(stage, rel, data, 0o644)
    _write(stage, "README.md", LIBRARY_README.read_bytes(), 0o644)


def compare(folder: Path, manifest: dict[str, tuple[int, str]], echo: Callable[[str], None] = print) -> bool:
    """Print how *folder*'s files compare with *manifest*, per top-level directory: identical,
    differing (bytes or permission bits), missing and extra, naming each file that is not
    identical. Return whether every file is identical and none is missing or extra."""
    built = {}
    for path in folder.rglob("*"):
        if path.is_file() and not path.is_symlink():
            built[path.relative_to(folder).as_posix()] = (
                path.stat().st_mode & 0o7777, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
    groups: dict[str, dict[str, list[str]]] = {}
    for rel in sorted(set(manifest) | set(built)):
        group = groups.setdefault(rel.split("/", 1)[0] if "/" in rel else ".", {
            "identical": [], "differing": [], "missing": [], "extra": []})
        if rel not in built:
            group["missing"].append(rel)
        elif rel not in manifest:
            group["extra"].append(rel)
        elif built[rel][:2] == manifest[rel]:
            group["identical"].append(rel)
        else:
            mode, sha = manifest[rel]
            got_mode, got_sha, size = built[rel]
            why = [f"mode {got_mode:o}, pinned {mode:o}"] if got_mode != mode else []
            if got_sha != sha:
                why.append(f"{size} bytes, sha256 {got_sha[:12]} (pinned {sha[:12]})")
            group["differing"].append(f"{rel}: {'; '.join(why)}")
    exact = True
    for name, group in groups.items():
        counts = ", ".join(f"{len(v)} {k}" for k, v in group.items() if v)
        echo(f"{name + '/' if name != '.' else '(top)'}: {counts}")
        for kind in ("differing", "missing", "extra"):
            for line in group[kind]:
                exact = False
                echo(f"  {kind}: {line}")
    return exact


def destination_refusal(dest: Path) -> str | None:
    """Why *dest* may not be replaced by the library, or None: it is there, and is neither an empty
    directory nor a library this command wrote, whose ``README.md`` is ``library-README.md``."""
    if not (dest.exists() or dest.is_symlink()):
        return None
    if dest.is_dir():
        readme = dest / "README.md"
        if not any(dest.iterdir()) or (readme.is_file() and readme.read_bytes() == LIBRARY_README.read_bytes()):
            return None
    return (f"{dest} is there and is not a library this command wrote (its README.md is not the library's "
            "index), and the library replaces its destination whole; remove it or name another --dest")


def assemble_library(dest: Path, cache: Path = DEFAULT_CACHE, *, build: Callable[..., None],
                     manifest: dict[str, tuple[int, str]] | None = None,
                     echo: Callable[[str], None] = print) -> bool:
    """Put the library the manifest describes (*manifest*, ``manifest.tsv`` unless given) at
    *dest*, with *build* building it into a staging directory and keeping its clones and
    downloads under *cache*, as ``assemble`` does. Return True when *dest* already had its tree
    digest or now has it, and False when the library built and swapped in has another, after
    naming each file that differs. Raise AssemblyError, with *dest* left as it was, when
    ``destination_refusal`` names a reason or the library cannot be built."""
    manifest = read_manifest() if manifest is None else manifest
    pinned = manifest_digest(manifest)
    try:
        if tree_digest(dest) == pinned:
            echo(f"{dest} already has the pinned tree digest {pinned}; left as it is")
            return True
    except TreeDigestError:
        pass
    why = destination_refusal(dest)
    if why:
        raise AssemblyError(why)
    stage = dest.with_name(f".{dest.name}.assembling")
    _remove(stage)
    stage.mkdir(parents=True)
    try:
        build(stage, cache, echo=echo)
    except BaseException:
        _remove(stage)
        raise
    replaced = dest.with_name(f".{dest.name}.replaced")
    _remove(replaced)
    if dest.exists() or dest.is_symlink():
        dest.rename(replaced)
    stage.rename(dest)
    _remove(replaced)
    echo(f"\nassembled {dest}; compared with the manifest:")
    compare(dest, manifest, echo=echo)
    digest = tree_digest(dest)
    if digest == pinned:
        echo(f"\ntree digest {digest}: matches the pinned digest")
        return True
    echo(f"\ntree digest {digest}: does not match the pinned digest {pinned}; the tasks refuse this folder")
    return False
