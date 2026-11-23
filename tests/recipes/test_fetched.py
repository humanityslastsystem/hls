"""The files a server layer fetches (``hls.recipes.model.Fetched``): what its build renders, what
its lock may hold, the ``%post`` steps run under sh with stand-ins for curl and the package, and
the FlashInfer kernels the SGLang calibration members fetch."""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from hls.recipes import catalog, pins
from hls.recipes.model import Fetched, LayerImage, UvPin
from hls.recipes.render import LOCK_ROOT, files_for, lock_dir, recipe_key, render_def

FIXTURES = Path(__file__).with_name("fixtures")
UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
PARENT = Path("/node/cuda.sif")
STEP_CHECK = 'hls_rc=$?; [ "$hls_rc" -eq 0 ] || exit "$hls_rc"'
#: The files of the stand-in archive's directory, by their path under it.
KERNELS = {"a/one.bin": b"kernel one", "two.h": b"#define TWO 2\n"}
#: The stand-in archive's other members, which the image never holds.
OTHERS = {"kernels/__init__.py": b"", "kernels-1.2.3.dist-info/METADATA": b"Name: kernels\n"}


def _archive(files: dict[str, bytes] = KERNELS) -> bytes:
    """A wheel-shaped zip whose directory ``kernels/files`` holds *files*."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in {**OTHERS, **{f"kernels/files/{p}": d for p, d in files.items()}}.items():
            z.writestr(name, data)
    return out.getvalue()


def _lock(files: dict[str, bytes] = KERNELS) -> bytes:
    lines = [f"{hashlib.sha256(data).hexdigest()}  {path}" for path, data in sorted(files.items())]
    return ("# the stand-in package's kernels\n\n" + "\n".join(lines) + "\n").encode()


def _fetched(**over) -> Fetched:
    fields = dict(url="https://example.com/releases/v1.2.3/kernels-1.2.3-py3-none-any.whl",
                  sha256=hashlib.sha256(_archive()).hexdigest(), member="kernels/files", lock="fetch-kernels.txt",
                  into="/srv/kernels/v1", variable="KERNELS_DIR", loader="kernels.loader:load",
                  refuse_env="KERNELS_NO_DOWNLOAD")
    return Fetched(**{**fields, **over})


def _layer(**over) -> LayerImage:
    """A server layer that fetches ``_fetched(**over)`` and bakes the variable that names its files'
    directory."""
    fetched = _fetched(**over)
    return LayerImage(name="stack", parent="cuda", pip_lock="pip-layer.txt", server=True, fetched=fetched,
                      env={"PATH": "/opt/venv/bin:$PATH", fetched.variable: fetched.into})


def _locks(tmp_path: Path, lock: bytes | None = None) -> Path:
    """A copy of the fixture locks that also holds the fetch lock *lock*."""
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    (locks / "fetch-kernels.txt").write_bytes(_lock() if lock is None else lock)
    return locks


def _post(text: str) -> list[str]:
    """The stripped lines of the %post section of a .def."""
    lines = text.splitlines()
    body = []
    for line in lines[lines.index("%post") + 1:]:
        if line.startswith("%"):
            break
        body.append(line.strip())
    return [b for b in body if b]


def _steps(text: str) -> list[str]:
    """The %post steps of *text*, each its lines up to its status check."""
    steps, current = [], []
    for line in _post(text)[2:]:
        if line == STEP_CHECK:
            steps.append("\n".join(current))
            current = []
        else:
            current.append(line)
    assert current == []
    return steps


def test_a_layer_copies_its_fetch_lock_from_the_shared_directory_whichever_task_owns_it(tmp_path):
    locks = _locks(tmp_path)
    tasks = tmp_path / "tasks"
    (tasks / "t" / "locks").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "pip-layer.txt", tasks / "t" / "locks" / "pip-layer.txt")
    own = _layer().model_copy(update={"owner": "t"})
    assert files_for(own, locks, tasks) == {
        f"{LOCK_ROOT}/fetch-kernels.txt": locks / "fetch-kernels.txt",
        f"{LOCK_ROOT}/pip-layer.txt": tasks / "t" / "locks" / "pip-layer.txt"}
    assert files_for(_layer(), locks, tasks) == {
        f"{LOCK_ROOT}/fetch-kernels.txt": locks / "fetch-kernels.txt",
        f"{LOCK_ROOT}/pip-layer.txt": locks / "pip-layer.txt"}
    with pytest.raises(FileNotFoundError, match="fetch-kernels.txt"):
        render_def(own, parent_sif=PARENT, locks=FIXTURES, uv=UV, tasks_dir=tasks)


def test_the_archive_is_fetched_by_url_and_unpacked_once_it_matches_its_digest(tmp_path):
    fetched = _fetched()
    steps = _steps(render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV))
    start = steps.index('hls_fetch="$(mktemp -d /opt/hls/fetch.XXXXXX)"')
    # After the pip install: the check calls the package the lock installs.
    assert any(step.startswith("uv pip install") for step in steps[:start])
    assert steps[start + 1:start + 4] == [
        f'curl -fsSL --retry 5 -o "$hls_fetch/archive.zip" {fetched.url}',
        f'echo "{fetched.sha256}  $hls_fetch/archive.zip" | sha256sum -c -',
        '/usr/bin/python3 -m zipfile -e "$hls_fetch/archive.zip" "$hls_fetch/unpacked"']
    assert 'mv "$hls_fetch/unpacked/kernels/files" /srv/kernels/v1' in steps
    assert steps[-1] == 'rm -rf "$hls_fetch"'


def test_every_fetched_file_is_checked_against_the_lock_and_nothing_else_may_be_there(tmp_path):
    steps = _steps(render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV))
    sums = steps.index(f"sed -e '/^#/d' -e '/^$/d' {LOCK_ROOT}/fetch-kernels.txt > \"$hls_fetch/sums\"")
    assert steps[sums + 1:sums + 4] == ["cd /srv/kernels/v1", 'sha256sum -c --strict --quiet "$hls_fetch/sums"',
                                        "cd /"]
    assert steps[sums + 4].startswith('hls_found="$(find /srv/kernels/v1 ! -type d | wc -l)"')
    assert "holds $hls_found files, and the lock names $hls_locked" in steps[sums + 4]


def test_the_layers_environment_bakes_the_variable_and_the_fetch_steps_set_none(tmp_path):
    text = render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV)
    section = text.split("%environment\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["    export PATH=/opt/venv/bin:$PATH", "    export KERNELS_DIR=/srv/kernels/v1"]
    # One place bakes variables: no step appends to the file Apptainer sources after it.
    assert "APPTAINER_ENVIRONMENT" not in text


def test_the_build_calls_the_packages_loader_on_every_file_with_downloads_refused(tmp_path):
    steps = _steps(render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV))
    check = steps[-2]
    # %post does not source %environment, so the check names the directory itself, as the layer bakes it.
    assert check.startswith('env HOME="$hls_fetch" PYTHONDONTWRITEBYTECODE=1 KERNELS_DIR=/srv/kernels/v1 '
                            "KERNELS_NO_DOWNLOAD=1 /opt/venv/bin/python -c '")
    assert 'getattr(import_module("kernels.loader"), "load")' in check and check.endswith('\' "$hls_fetch/sums"')
    # The variable that refuses downloads is the check's alone.
    assert not any("KERNELS_NO_DOWNLOAD" in step for step in steps if step != check)


def test_no_fetch_step_names_a_path_in_the_hosts_tmp_or_a_build_argument(tmp_path):
    text = render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV)
    assert "/tmp" not in text and "{{" not in text


@pytest.mark.parametrize(("lock", "match"), [
    (b"# nothing\n\n", "names no file"),
    (b"a" * 64 + b" one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"A" * 64 + b"  one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 63 + b"  one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b"  /one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b"  a/../one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b"  one bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b"  one.bin \n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b'  one".bin\n', "not a sha256, two spaces and a relative path"),
    (b" " + b"a" * 64 + b"  one.bin\n", "not a sha256, two spaces and a relative path"),
    (b"a" * 64 + b"  one.bin\n" + b"b" * 64 + b"  one.bin\n", "names one.bin twice"),
    (b"a" * 64 + b"  a\n" + b"b" * 64 + b"  a/one.bin\n", "as a file and as a directory"),
])
def test_a_fetch_lock_line_that_is_not_a_digest_and_a_plain_relative_path_is_refused(tmp_path, lock, match):
    with pytest.raises(ValueError, match=match):
        render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path, lock), uv=UV)


def _key(decl: LayerImage, locks: Path) -> str:
    return recipe_key(decl, parent_key="cuda-0123456789abcdef", locks_dir=locks, uv=UV)


def test_a_key_follows_the_fetched_files_and_how_they_are_fetched(tmp_path):
    locks = _locks(tmp_path)
    before = _key(_layer(), locks)
    assert re.fullmatch(r"stack-[0-9a-f]{16}", before)
    for over in ({"url": "https://example.com/releases/v1.2.4/kernels-1.2.4-py3-none-any.whl"}, {"sha256": "b" * 64},
                 {"member": "kernels/other"}, {"into": "/srv/kernels/v2"}, {"variable": "KERNELS_HOME"},
                 {"loader": "kernels.loader:get"}, {"refuse_env": "KERNELS_OFFLINE"}):
        assert _key(_layer(**over), locks) != before, over
    (locks / "fetch-kernels.txt").write_bytes(_lock({**KERNELS, "three.h": b"3"}))
    assert _key(_layer(), locks) != before


# The fetch steps themselves, run under sh with stand-ins for curl and the package, to pin that
# each check ends the build.

LOADER = '''
import hashlib
import os
from pathlib import Path


def load(path, sha256):
    with open(os.environ["LOADER_LOG"], "a") as log:
        log.write("%s %s %s\\n" % (path, os.environ.get("KERNELS_NO_DOWNLOAD"), os.environ["HOME"]))
    file = Path(os.environ["KERNELS_DIR"]) / path
    if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == sha256:
        return file.read_bytes()
    if os.environ.get("KERNELS_NO_DOWNLOAD"):
        raise RuntimeError("refusing to download " + path)
    return b"downloaded"


def elsewhere(path, sha256):
    """A package that looks beside the directory the variable names."""
    os.environ["KERNELS_DIR"] = os.environ["KERNELS_DIR"] + "-other"
    return load(path, sha256)


def mute(path, sha256):
    return b""
'''


def _run_fetch(tmp_path: Path, *, served: bytes | None = None, lock: bytes | None = None,
               before: str = "", **over) -> subprocess.CompletedProcess:
    """Run a layer's fetch steps, from making the download directory to removing it, with the
    image paths moved under *tmp_path*: curl serves *served* (the stand-in archive unless given),
    the venv's python is this interpreter with the stand-in package on its path, and *before*
    runs first. The pinned digest is the stand-in archive's."""
    steps = _steps(render_def(_layer(**over), parent_sif=PARENT, locks=_locks(tmp_path, lock), uv=UV))
    start = next(i for i, step in enumerate(steps) if step.startswith("hls_fetch="))
    script = "\n".join(f"{step}\n{STEP_CHECK}" for step in steps[start:])
    for image, host in (("/opt/hls", "opt-hls"), ("/opt/venv", "venv"), ("/srv/kernels", "srv-kernels")):
        script = script.replace(image, str(tmp_path / host))
    script = script.replace("/usr/bin/python3", sys.executable)
    hls = tmp_path / "opt-hls"
    (hls / "locks").mkdir(parents=True)
    shutil.copyfile(tmp_path / "locks" / "fetch-kernels.txt", hls / "locks" / "fetch-kernels.txt")
    (tmp_path / "served").write_bytes(_archive() if served is None else served)
    bin_dir, venv_bin, site = tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "site" / "kernels"
    for d in (bin_dir, venv_bin, site):
        d.mkdir(parents=True)
    (bin_dir / "curl").write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && hls_out="$2"; shift; done\n'
                                  f'cp {tmp_path}/served "$hls_out"\n')
    (venv_bin / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for stub in (bin_dir / "curl", venv_bin / "python"):
        stub.chmod(0o755)
    (site / "__init__.py").write_text("")
    (site / "loader.py").write_text(LOADER)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(tmp_path / "site"),
           "LOADER_LOG": str(tmp_path / "loader.log")}
    return subprocess.run(["sh", "-c", f"set -eu\n{before}\n{script}\necho reached-the-end\n"],
                          capture_output=True, text=True, env=env)


def _placed(tmp_path: Path) -> dict[str, bytes]:
    root = tmp_path / "srv-kernels" / "v1"
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_the_fetched_files_land_in_their_directory_and_the_package_loads_each(tmp_path):
    r = _run_fetch(tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    # The archive's directory alone, not its other members.
    assert _placed(tmp_path) == KERNELS
    # The loader saw every file, with downloads refused and a home inside the download directory,
    # which is gone with the archive.
    calls = [line.split() for line in (tmp_path / "loader.log").read_text().splitlines()]
    assert [(path, refused) for path, refused, _ in calls] == [("a/one.bin", "1"), ("two.h", "1")]
    assert all(Path(home).parent == tmp_path / "opt-hls" and Path(home).name.startswith("fetch.")
               for _, _, home in calls)
    assert [p.name for p in (tmp_path / "opt-hls").iterdir()] == ["locks"]


def test_an_archive_that_does_not_match_its_digest_is_never_unpacked(tmp_path):
    r = _run_fetch(tmp_path, served=_archive({**KERNELS, "a/one.bin": b"another kernel"}))
    assert r.returncode != 0 and "reached-the-end" not in r.stdout
    assert not (tmp_path / "srv-kernels").exists() and not (tmp_path / "opt-hls" / "unpacked").exists()


def test_a_file_with_another_digest_than_the_lock_names_ends_the_build(tmp_path):
    r = _run_fetch(tmp_path, lock=_lock({**KERNELS, "two.h": b"#define TWO 3\n"}))
    assert r.returncode != 0 and "two.h: FAILED" in r.stdout + r.stderr
    assert "reached-the-end" not in r.stdout and not (tmp_path / "loader.log").exists()


def test_a_locked_file_the_archive_lacks_ends_the_build(tmp_path):
    r = _run_fetch(tmp_path, lock=_lock({**KERNELS, "three.h": b"3"}))
    assert r.returncode != 0 and "three.h" in r.stdout + r.stderr and "reached-the-end" not in r.stdout
    assert not (tmp_path / "loader.log").exists()


def test_a_file_the_lock_does_not_name_ends_the_build(tmp_path):
    r = _run_fetch(tmp_path, lock=_lock({"two.h": KERNELS["two.h"]}))
    assert r.returncode == 1 and "holds 2 files, and the lock names 1" in r.stderr
    assert "reached-the-end" not in r.stdout and not (tmp_path / "loader.log").exists()


def test_an_archives_by_products_are_removed_before_the_directory_is_checked(tmp_path):
    """An archive may hold the empty lock file its build left beside each file. They are none of
    the package's files: the lock names none, and the loader would take an empty file for a
    missing one."""
    left = {**KERNELS, "a/one.bin.lock": b"", "two.h.lock": b""}
    served = _archive(left)
    digest = hashlib.sha256(served).hexdigest()
    r = _run_fetch(tmp_path / "kept", served=served, sha256=digest)
    assert r.returncode == 1 and "holds 4 files, and the lock names 2" in r.stderr
    r = _run_fetch(tmp_path / "dropped", served=served, sha256=digest, dropped=".lock")
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    assert _placed(tmp_path / "dropped") == KERNELS
    calls = [line.split()[0] for line in (tmp_path / "dropped" / "loader.log").read_text().splitlines()]
    assert calls == ["a/one.bin", "two.h"]


def test_a_declaration_with_no_by_product_is_written_and_rendered_as_before(tmp_path):
    assert "dropped" not in _fetched().model_dump() and _fetched(dropped=".lock").model_dump()["dropped"] == ".lock"
    plain = render_def(_layer(), parent_sif=PARENT, locks=_locks(tmp_path), uv=UV)
    assert "-delete" not in plain
    with pytest.raises(ValueError, match="a file-name suffix such as .lock"):
        _fetched(dropped="*.lock")


def test_a_directory_that_is_already_there_ends_the_build(tmp_path):
    r = _run_fetch(tmp_path, before=f"mkdir -p {tmp_path}/srv-kernels/v1")
    assert r.returncode == 1 and "is there before its files are fetched" in r.stderr
    assert _placed(tmp_path) == {}


def test_a_package_that_would_still_download_ends_the_build(tmp_path):
    # The package looks somewhere other than the directory the variable names, finds nothing
    # there and would download.
    r = _run_fetch(tmp_path, loader="kernels.loader:elsewhere")
    assert r.returncode != 0 and "refusing to download a/one.bin" in r.stderr
    assert "reached-the-end" not in r.stdout


def test_a_loader_that_returns_nothing_ends_the_build(tmp_path):
    r = _run_fetch(tmp_path, loader="kernels.loader:mute")
    assert r.returncode != 0
    assert "the package would still download 2 files, the first: ['a/one.bin', 'two.h']" in r.stderr
    assert "reached-the-end" not in r.stdout


# The FlashInfer kernels of the catalog, and the SGLang members that fetch them.

#: The directories FlashInfer 0.6.14 keeps its kernels in and the sha256 of each one's
#: checksums.txt, as its ``flashinfer/artifacts.py`` names them (``ArtifactPath``,
#: ``CheckSumHash``) for x86_64.
FLASHINFER_MANIFESTS = {
    "0.6.14": {
        "158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen":
            "c2d9399b2537be785882354a4f9902ed6c03136c0ea341e201eac40c3923e1dc",
        "481dce07c89a216cbfd18cf39de49a82d40739a8/batched_gemm-dd6d23e-721ae60":
            "aa19cf2a37eed029eee5b3f96b37e069e4ab40f419b25ed7a3fd9526d8833bfb",
        "10f64528a1172dae8e29601a3b99ab9dc78d37be/gemm-91e0ba0-2710384":
            "f97f90f9ce1dab73eb3d7c90fca4bbd52687642dd87a79dd10b77d7802b25c33",
        "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm":
            "1a2a166839042dbd2a57f48051c82cd1ad032815927c753db269a4ed10d0ffbf",
        "801e770219613fbf088bc074c414732b26cc550d/fmha/cute-dsl/x86_64/sm_100a":
            "778738c3aa89872248fcfddd134b57ae516021471df992d4ba9b058ead546d56",
        "801e770219613fbf088bc074c414732b26cc550d/fmha/cute-dsl/x86_64/sm_103a":
            "f57abef4c65968c99e93faa051d9b98cf789c82c805bd3a177fb3f2a426dac4f",
        "801e770219613fbf088bc074c414732b26cc550d/fmha/cute-dsl/x86_64/sm_110a":
            "f2450d136221d7c355876140af860999fd5f5cdd16ffa4b06ff8b799c2106c29",
    },
    "0.6.18": {
        "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen":
            "d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3",
        "8ec29a98612c3670f9f28825d1ed19f09496073b/batched_gemm-fa419f4-31ee4e5":
            "011635d3c36756addcdc148eea90c984f2d7611ba375626aaaf466d355c80e50",
        "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/gemm-fa419f4-25754e6":
            "d0783b41d2be41a0f7583668faa1edd6b4f5b48f70e2d58b7a9dfe475f87cc5a",
        "7ec7ac40b9fd48172651b77ff2ebe20d79decc39/deep-gemm":
            "09e961d4e3852a6cf81b3482d0604c09dcb1f69c1b7936f535c9ee2f53335184",
        "5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_100a":
            "832c303bb9b386af590d3efc294681859829b91991975fd2e188a5d7dc30c461",
        "5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_103a":
            "57322c10ddbbe9072c7ded41e2856fdf9d4276fbd79ac4bc825af0cd78844da6",
        "5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_107a":
            "8480678539adf622f8395e875923471bce683be7158694556bd6c536eadeaa45",
        "5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/x86_64/sm_110a":
            "4f6f0f3a868f0e9171c8ab217e6d2a87fde46b02a9417d8f55f1a779c53fa9fb",
    },
}
#: How many files of each kind ``python -m flashinfer download-cubin`` of that version fetches.
FLASHINFER_FILES = {"0.6.14": {"cubin": 16062, "h": 34, "txt": 7}, "0.6.18": {"cubin": 42704, "h": 34, "txt": 8}}


def _fetch_lock(fetched: Fetched) -> dict[str, str]:
    """Path to sha256 of every file fetch lock *fetched* names."""
    lines = [line for line in (pins.LOCKS_DIR / fetched.lock).read_text().splitlines()
             if line and not line.startswith("#")]
    return {path: sha256 for sha256, path in (line.split("  ") for line in lines)}


def test_the_catalog_knows_the_kernels_of_the_flashinfer_versions_the_tests_do():
    assert set(catalog.FLASHINFER_CUBINS) == set(FLASHINFER_MANIFESTS) == set(FLASHINFER_FILES)


@pytest.mark.parametrize("version", sorted(FLASHINFER_MANIFESTS))
def test_a_flashinfer_kernel_lock_holds_the_manifests_flashinfer_names_and_their_kernels(version):
    fetched = catalog.FLASHINFER_CUBINS[version]
    assert f"/v{version}/flashinfer_cubin-{version}-py3-none-any.whl" in fetched.url
    assert (fetched.variable, fetched.into) == ("FLASHINFER_CUBIN_DIR", "/opt/flashinfer-cubins")
    locked = _fetch_lock(fetched)
    manifests = {path.removesuffix("/checksums.txt"): sha256 for path, sha256 in locked.items()
                 if path.endswith("/checksums.txt")}
    assert manifests == FLASHINFER_MANIFESTS[version]
    kinds: dict[str, int] = {}
    for path in locked:
        # Every file is in one of the directories FlashInfer names, so it has a manifest.
        assert any(path.startswith(directory + "/") for directory in manifests), path
        kinds[path.rsplit(".", 1)[-1]] = kinds.get(path.rsplit(".", 1)[-1], 0) + 1
    assert kinds == FLASHINFER_FILES[version]


def _task_layers() -> list[tuple[str, str, LayerImage]]:
    """Every server layer a task declares for a role, as (task, role, layer)."""
    out = []
    for task in catalog.task_keys():
        images = catalog.task_images(task)
        by_name = {d.name: d for d in images.decls}
        out += [(task, role, by_name[name]) for role, name in images.measured.items()
                if isinstance(by_name[name], LayerImage)]
    return out


def _pip_lock(decl: LayerImage) -> dict[str, str]:
    text = (lock_dir(decl, pins.LOCKS_DIR) / decl.pip_lock).read_text()
    return dict(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", text, re.MULTILINE))


def test_a_layer_that_fetches_flashinfer_kernels_installs_that_flashinfer_and_no_kernel_wheel():
    fetching = [(task, role, decl) for task, role, decl in _task_layers() if decl.fetched is not None]
    assert {(task, role) for task, role, _ in fetching} >= {("multitenant_slo", "member-sglang"),
                                                              ("tab_completion", "member-sglang")}
    for task, role, decl in fetching:
        [version] = [v for v, fetched in catalog.FLASHINFER_CUBINS.items() if fetched == decl.fetched]
        locked = _pip_lock(decl)
        assert locked.get("flashinfer-python") == version, (task, role)
        # FlashInfer reads an installed flashinfer-cubin before the variable the image sets.
        assert "flashinfer-cubin" not in locked, (task, role)


#: The engine packages of the SGLang members, at the versions their locks hold.
SGLANG_STACK = {"sglang": "0.5.16", "sglang-kernel": "0.4.5", "sgl-deep-gemm": "0.1.4.post1",
                "flashinfer-python": "0.6.14", "apache-tvm-ffi": "0.1.11", "triton": "3.6.0",
                "nvidia-cutlass-dsl": "4.6.0", "quack-kernels": "0.6.3", "tilelang": "0.1.11",
                "flash-attn-4": "4.0.0b19", "torch": "2.11.0+cu130", "torchaudio": "2.11.0+cu130",
                "torchvision": "0.26.0+cu130", "transformers": "5.12.1", "tokenizers": "0.23.0rc0",
                "numpy": "2.3.5", "numba": "0.65.1", "xgrammar": "0.2.1"}


@pytest.mark.parametrize("task", ["multitenant_slo", "tab_completion"])
def test_the_sglang_member_builds_from_its_tasks_lock_with_flashinfers_kernels(task):
    images = catalog.task_images(task)
    member = next(d for d in images.decls if d.name == images.measured["member-sglang"])
    assert member == LayerImage(name=f"{task.replace('_', '-')}-member-sglang", parent="cuda",
                                pip_lock="member-sglang.txt", owner=task, server=True,
                                fetched=catalog.FLASHINFER_CUBINS["0.6.14"],
                                env={**catalog.SERVER_ENV, "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"})
    locked = _pip_lock(member)
    assert {name: locked.get(name) for name in SGLANG_STACK} == SGLANG_STACK
    text = render_def(member, parent_sif=PARENT, locks=pins.LOCKS_DIR, uv=pins.UV)
    assert "    export FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubins\n" in text.split("%environment\n", 1)[1]
    check = _steps(text)[-2]
    assert 'getattr(import_module("flashinfer.jit.cubin_loader"), "get_artifact")' in check
    assert " FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubins FLASHINFER_NO_DOWNLOAD=1 " in check

