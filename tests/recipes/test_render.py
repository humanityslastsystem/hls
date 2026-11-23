from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import hls.recipes.render as render_module
from hls.recipes.model import BASE_ENVIRONMENT, BaseImage, CapImage, LayerImage, NodePin, UvPin
from hls.recipes.render import (
    LOCK_ROOT,
    NODE_PREFIX,
    NPM_PREFIX,
    PYPI_INDEX,
    TORCH_INDEX,
    VENV,
    check_source_builds,
    files_for,
    node_url,
    npm_manifest,
    recipe_key,
    render_def,
)

FIXTURES = Path(__file__).with_name("fixtures")
CUDA = "nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:" + "7" * 64
NVIDIA_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
BASE = BaseImage(name="s-a", cuda_ref=CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
                 nvidia_apt={"cuda-nsight-systems-13-0=2025.3.2.474-1": hashlib.sha256(
                     b"deb cuda-nsight-systems-13-0=2025.3.2.474-1").hexdigest()}, nvidia_repo=NVIDIA_REPO,
                 pip_lock="pip-base.txt", constraints=["torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"],
                 env={"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
LAYER = LayerImage(name="s-a-tf", parent="s-a", pip_lock="pip-layer.txt")
TTS = LayerImage(name="s-a-tts", parent="s-a", pip_lock="pip-sdist.txt",
                 source_builds={"sox==1.5.0": ["setuptools"]})
NODE_TOP = "node-v22.23.3-linux-x64"


#: A Node.js on the host, which runs the build's npm checks (JavaScript) in the stand-in Node.
HOST_NODE = shutil.which("node")
#: npm-install-checks as the stand-in Node's npm carries it: the rule npm decides with which lock
#: entries it installs, on linux/x64 with glibc.
INSTALL_CHECKS = """\
const env = { os: "linux", cpu: "x64", libc: "glibc" };
const checkList = (value, list) => {
  if (typeof list === "string") list = [list];
  if (list.length === 1 && list[0] === "any") return true;
  let negated = 0, match = false;
  for (const entry of list) {
    if (entry.charAt(0) === "!") { negated++; if (value === entry.slice(1)) return false; }
    else match = match || value === entry;
  }
  return match || negated === list.length;
};
exports.checkPlatform = (target) => {
  for (const field of ["os", "cpu", "libc"]) {
    if (target[field] && !checkList(env[field], target[field])) {
      throw Object.assign(new Error("Unsupported platform"), { code: "EBADPLATFORM" });
    }
  }
};
"""
#: ``npm ls --all --json --long`` as the stand-in npm answers it: every package under the
#: prefix's node_modules, by name, with its path and version.
NPM_LS = """\
import json, sys
from pathlib import Path
prefix = Path(sys.argv[sys.argv.index("--prefix") + 1])
found = {}
for manifest in sorted((prefix / "node_modules").glob("**/package.json")):
    rel = manifest.parent.relative_to(prefix / "node_modules")
    if len(rel.parts) <= (2 if rel.parts[0].startswith("@") else 1) and not rel.parts[0].startswith("."):
        version = json.loads(manifest.read_text())["version"]
        found[str(rel)] = {"path": str(manifest.parent), "version": version}
print(json.dumps({"name": "hls-agent-cli", "dependencies": found}))
"""


def _node_tarball(version: str = "v22.23.3") -> bytes:
    """A stand-in Node release tarball. Its node answers --version with *version*, runs an ``-e``
    script with the host's node, and runs any other script with sh, marking it as run by this
    node. Its npm logs its PATH and arguments to $HLS_NPM_LOG, runs $HLS_NPM_CI for ``ci`` and
    answers ``ls`` from the prefix's node_modules. Its npm-install-checks decides as npm does."""
    host = f'exec {HOST_NODE} "$@"' if HOST_NODE else 'echo "no node on the host" >&2; exit 127'
    node = ("#!/bin/sh\n"
            'case "$1" in\n'
            f"  --version) echo {version} ;;\n"
            f"  -e) {host} ;;\n"
            '  *) HLS_RAN_BY=pinned-node; export HLS_RAN_BY; exec sh "$@" ;;\n'
            "esac\n").encode()
    npm = ('#!/bin/sh\necho "PATH=$PATH ARGS=$*" >> "$HLS_NPM_LOG"\n'
           'case "$1" in\n'
           '  ci) exec sh "$HLS_NPM_CI" ;;\n'
           f'  ls) exec {shutil.which("python3")} -c {shlex.quote(NPM_LS)} "$@" ;;\n'
           "esac\nexit 0\n").encode()
    checks = f"{NODE_TOP}/lib/node_modules/npm/node_modules/npm-install-checks"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data, mode in ((f"{NODE_TOP}/bin/node", node, 0o755), (f"{NODE_TOP}/bin/npm", npm, 0o755),
                                 (f"{NODE_TOP}/include/node/node.h", b"", 0o644),
                                 (f"{NODE_TOP}/lib/node_modules/npm/package.json", b"{}", 0o644),
                                 (f"{checks}/package.json", b'{"main": "lib/index.js"}', 0o644),
                                 (f"{checks}/lib/index.js", INSTALL_CHECKS.encode(), 0o644),
                                 (f"{NODE_TOP}/share/man/man1/node.1", b"", 0o644),
                                 (f"{NODE_TOP}/README.md", b"", 0o644)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    # No time in the gzip header, so every call gives the same bytes.
    return gzip.compress(buf.getvalue(), mtime=0)


NODE = NodePin(version="22.23.3", sha256=hashlib.sha256(_node_tarball()).hexdigest())
CAP = CapImage(name="s-a-tf-agent", parent="s-a-tf", cap="agent", apt_lock="apt-agent.txt",
               npm={"@anthropic-ai/claude-code": "2.1.251", "@openai/codex": "0.146.0"},
               npm_lock="npm-agent.json", node=NODE)


def _section(text: str, name: str) -> list[str]:
    """The stripped lines of section %<name> of a .def."""
    lines = text.splitlines()
    start = lines.index(f"%{name}") + 1
    body = []
    for line in lines[start:]:
        if line.startswith("%"):
            break
        body.append(line)
    return [b.strip() for b in body if b.strip()]


def test_a_base_starts_from_the_digest_pinned_cuda_image_and_copies_its_locks():
    text = render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV)
    # Apptainer refuses a reference with both a tag and a digest; the digest alone pins the bytes,
    # and the tag stays readable in a label.
    assert text.startswith("Bootstrap: docker\nFrom: nvidia/cuda@sha256:" + "7" * 64 + "\n")
    assert _section(text, "labels") == [f"hls.cuda_ref {CUDA}"]
    assert _section(text, "files") == [f"{FIXTURES}/apt-base.txt {LOCK_ROOT}/apt-base.txt",
                                       f"{FIXTURES}/pip-base.txt {LOCK_ROOT}/pip-base.txt"]


def test_a_base_resolves_apt_from_the_snapshot_and_checks_every_exact_version():
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post")
    assert post[:2] == ["set -eu", "set +x"]
    joined = "\n".join(post)
    assert 'APT::Sandbox::User "root";' in joined and 'Acquire::Check-Valid-Until "false";' in joined
    assert "https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/" in joined
    # NVIDIA's packages install from the .deb files checked against their recorded digests.
    assert ("apt-get install -y --no-install-recommends "
            f"$(sed -e '/^#/d' -e '/^$/d' {LOCK_ROOT}/apt-base.txt) $hls_nvidia_debs") in joined
    assert ("for hls_pin in $(sed -e '/^#/d' -e '/^$/d' /opt/hls/locks/apt-base.txt) "
            "cuda-nsight-systems-13-0=2025.3.2.474-1; do") in joined
    assert "install ok installed $hls_want" in joined
    assert "--ignore-fakeroot-command" not in joined


def test_the_venv_and_the_harness_programs_in_the_image_run_on_its_one_system_interpreter(monkeypatch):
    """The base makes its venv from the image's system interpreter (``model.SYSTEM_PYTHON``), and
    the harness's programs in the sandbox run on the same one, whose path the in-sandbox protocol
    keeps a copy of (``SANDBOX_PYTHON``), since it imports nothing of the harness."""
    from hls.agent import exec_protocol
    from hls.recipes import model

    assert exec_protocol.SANDBOX_PYTHON == model.SYSTEM_PYTHON
    monkeypatch.setattr(model, "SYSTEM_PYTHON", "/opt/elsewhere/python3")
    post = "\n".join(_section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post"))
    assert "uv venv --python /opt/elsewhere/python3 /opt/venv" in post


def test_a_base_installs_uv_by_digest_and_the_pip_lock_by_hash_without_resolving():
    post = "\n".join(_section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post"))
    assert ("https://github.com/astral-sh/uv/releases/download/0.11.21/uv-x86_64-unknown-linux-gnu.tar.gz"
            in post)
    assert f'echo "{UV.sha256}  $hls_dl/uv.tar.gz" | sha256sum -c -' in post
    # No --seed: it would fetch pip unpinned; pip comes from the hashed lock like everything else.
    assert "uv venv --python /usr/bin/python3 /opt/venv" in post and "--seed" not in post
    install = next(line for line in post.splitlines() if line.startswith("uv pip install"))
    assert "--require-hashes --no-deps" in install and install.endswith(f"-r {LOCK_ROOT}/pip-base.txt")
    assert ("--index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple"
            in install)


def test_a_base_holds_torch_for_every_later_install():
    text = render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV)
    post = "\n".join(_section(text, "post"))
    assert "printf '%s\\n' torch==2.11.0+cu130 torchaudio==2.11.0+cu130 > /opt/hls/constraints.txt" in post
    assert _section(text, "environment") == [
        "export PATH=/opt/venv/bin:$PATH", "export VIRTUAL_ENV=/opt/venv",
        "export UV_CONSTRAINT=/opt/hls/constraints.txt", "export PIP_CONSTRAINT=/opt/hls/constraints.txt",
        "export HF_HUB_DISABLE_TELEMETRY=1", "export PYTHONUNBUFFERED=1"]
    # The variables the base sets itself are the model's, which refuses them as declared ones.
    assert _section(text, "environment")[:len(BASE_ENVIRONMENT)] == [
        f"export {name}={value}" for name, value in BASE_ENVIRONMENT.items()]
    assert len(_section(text, "environment")) == len(BASE_ENVIRONMENT) + len(BASE.env)
    assert "TORCH_CUDA_ARCH_LIST" not in text
    assert _section(text, "runscript") == ['exec "$@"']


def test_every_command_step_is_followed_by_its_status_check():
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post")[2:]
    check = 'hls_rc=$?; [ "$hls_rc" -eq 0 ] || exit "$hls_rc"'
    # Steps end at their check; a multi-line step is one step.
    steps, current = [], []
    for line in post:
        if line == check:
            steps.append(current)
            current = []
        else:
            current.append(line)
    assert current == [] and len(steps) >= 12


def _steps(text: str) -> list[str]:
    """The %post steps of *text*, each its lines up to its status check."""
    check = 'hls_rc=$?; [ "$hls_rc" -eq 0 ] || exit "$hls_rc"'
    steps, current = [], []
    for line in _section(text, "post")[2:]:
        if line == check:
            steps.append("\n".join(current))
            current = []
        else:
            current.append(line)
    assert current == []
    return steps


def test_each_install_removes_its_download_cache():
    # %post runs as root with HOME=/root and without %environment, so the caches would stay in
    # the image; --no-cache would stage the wheels in the host's /tmp instead.
    for decl, parent in ((BASE, None), (LAYER, Path("/node/s-a.sif"))):
        steps = _steps(render_def(decl, parent_sif=parent, locks=FIXTURES, uv=UV))
        installs = [i for i, step in enumerate(steps) if step.startswith("uv pip install")]
        assert installs and all(steps[i + 2] == "uv cache clean" for i in installs), steps
    # npm's cache, its logs among them, is a directory of the download that is removed after npm ci.
    steps = _steps(render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV))
    npm = [i for i, step in enumerate(steps) if f"{NODE_PREFIX}/bin/npm " in step]
    # cache add, ci and ls.
    assert len(npm) == 3 and all('--cache "$hls_npm/cache"' in steps[i] for i in npm), steps
    assert steps.index('rm -rf "$hls_npm"') == npm[-1] + 2


def test_no_def_names_a_path_in_the_hosts_tmp():
    # A build's /tmp is the host's, shared by concurrent builds and other users.
    for decl, parent in ((BASE, None), (LAYER, Path("/node/s-a.sif")), (CAP, Path("/node/s-a-tf.sif"))):
        assert "/tmp" not in render_def(decl, parent_sif=parent, locks=FIXTURES, uv=UV)


def _run_uv_download(tmp_path: Path, served: bytes) -> subprocess.CompletedProcess:
    """Run a base's uv steps, from making the download directory to removing it, with stand-ins
    for curl (which serves *served* and records where it wrote) and tar (which records that it
    ran), the image paths moved under *tmp_path*, and TMPDIR, the host's temporary directory, at
    *tmp_path*/host-tmp. The pin is the digest of b"uv release"."""
    uv = UvPin(version="0.11.21", sha256=hashlib.sha256(b"uv release").hexdigest())
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=uv), "post")
    start = next(i for i, line in enumerate(post) if "mktemp" in line and "uv-download" in line)
    end = next(i for i, line in enumerate(post) if line.startswith("uv venv"))
    script = "\n".join(post[start:end]).replace("/opt/hls", str(tmp_path / "opt-hls")).replace(
        "/usr/local/bin", str(tmp_path / "local-bin"))
    for name in ("opt-hls", "local-bin", "bin", "host-tmp"):
        (tmp_path / name).mkdir()
    (tmp_path / "served").write_bytes(served)
    bin_dir = tmp_path / "bin"
    _stub(bin_dir, "curl", 'while [ $# -gt 0 ]; do [ "$1" = -o ] && hls_out="$2"; shift; done\n'
                           f'printf %s "$hls_out" > {tmp_path}/curl-wrote\ncp {tmp_path}/served "$hls_out"\n')
    _stub(bin_dir, "tar", f'touch {tmp_path}/tar-ran\nmkdir -p "$4/uv-x86_64-unknown-linux-gnu"\n'
                          'printf uv > "$4/uv-x86_64-unknown-linux-gnu/uv"\n')
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "TMPDIR": str(tmp_path / "host-tmp")}
    return _sh(script + "\necho reached-the-end\n", env)


def test_uv_is_installed_from_a_private_download_that_matches_its_digest(tmp_path):
    r = _run_uv_download(tmp_path, b"uv release")
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    assert (tmp_path / "local-bin" / "uv").read_text() == "uv"
    # Downloaded into a fresh directory of the image's tree, never the host's temporary one.
    wrote = Path((tmp_path / "curl-wrote").read_text())
    assert wrote.parent.parent == tmp_path / "opt-hls" and wrote.parent.name.startswith("uv-download.")
    assert list((tmp_path / "opt-hls").iterdir()) == [] and list((tmp_path / "host-tmp").iterdir()) == []


def test_a_uv_download_that_does_not_match_its_digest_is_never_unpacked(tmp_path):
    r = _run_uv_download(tmp_path, b"something else")
    assert r.returncode != 0 and "reached-the-end" not in r.stdout
    assert not (tmp_path / "tar-ran").exists() and not (tmp_path / "local-bin" / "uv").exists()


def _install(decl, parent: Path | None = Path("/node/s-a.sif"), locks: Path = FIXTURES) -> str:
    """The pip install line of *decl*'s rendered %post."""
    post = _section(render_def(decl, parent_sif=parent, locks=locks, uv=UV), "post")
    return next(line for line in post if line.startswith("uv pip install"))


def test_every_pip_install_takes_wheels_only_unless_a_package_is_named():
    for decl, parent in ((BASE, None), (LAYER, Path("/node/s-a.sif"))):
        install = _install(decl, parent)
        assert " --only-binary :all: " in install
        assert "--no-binary" not in install and "--no-build-isolation" not in install


def test_a_named_source_build_builds_in_the_venv_without_isolation():
    # Without isolation uv resolves no build requirement: the build sees only the venv's packages,
    # which hashed locks installed.
    install = _install(TTS)
    assert " --only-binary :all: --no-binary sox --no-build-isolation-package sox " in install
    assert install.endswith(f"-r {LOCK_ROOT}/pip-sdist.txt") and "--build-constraints" not in install
    assert list(files_for(TTS, FIXTURES)) == [f"{LOCK_ROOT}/pip-sdist.txt"]
    both = TTS.model_copy(update={"source_builds": {"transformers==4.57.3": ["setuptools"],
                                                     "sox==1.5.0": ["setuptools"]}})
    assert (" --no-binary sox --no-build-isolation-package sox --no-binary transformers "
            "--no-build-isolation-package transformers ") in _install(both)


def test_a_source_build_needs_its_backends_installed_by_a_lock_below_it():
    check_source_builds([TTS, BASE], FIXTURES)
    # Names compare as indexes compare them.
    spelled = TTS.model_copy(update={"source_builds": {"sox==1.5.0": ["SetupTools"]}})
    check_source_builds([BASE, spelled], FIXTURES)
    # Backends from the parent layer's lock and from the base's, below it.
    on_layer = TTS.model_copy(update={"parent": "s-a-tf",
                                      "source_builds": {"sox==1.5.0": ["transformers", "setuptools"]}})
    check_source_builds([BASE, LAYER, on_layer], FIXTURES)


@pytest.mark.parametrize("decls, match", [
    # The base's lock does not install hatchling.
    ([BASE, TTS.model_copy(update={"source_builds": {"sox==1.5.0": ["setuptools", "hatchling"]}})],
     "sox==1.5.0 of image s-a-tts needs hatchling"),
    # The layer's own lock installs with the build, not before it.
    ([BASE, TTS.model_copy(update={"source_builds": {"sox==1.5.0": ["transformers"]}})],
     "needs transformers"),
    ([TTS], "unknown parent s-a"),
])
def test_a_source_build_whose_backend_no_lock_below_installs_is_refused(decls, match):
    with pytest.raises(ValueError, match=match):
        check_source_builds(decls, FIXTURES)


@pytest.mark.parametrize("name, content, match", [
    ("pip-sdist.txt", b"transformers==4.57.3 \\\n    --hash=sha256:" + b"4" * 64 + b"\n",
     "not in its pip lock"),
    ("pip-sdist.txt", b"sox==1.4.1 \\\n    --hash=sha256:" + b"3" * 64 + b"\n", "not in its pip lock"),
    ("pip-sdist.txt", b"sox==1.5.0\n", "sha256 hash"),
    ("pip-sdist.txt", b"sox==1.5.0 \\\n    --hash=md5:" + b"6" * 32 + b"\n", "sha256 hash"),
    ("pip-sdist.txt", b"sox>=1.5 \\\n    --hash=sha256:" + b"6" * 64 + b"\n", "sha256 hash"),
    ("pip-sdist.txt", b"sox==1.5.0 \\\n    --hash=sha256:" + b"6" * 64 + b" \\", "continued"),
    ("pip-sdist.txt", b"sox==1.5.0 \\\n    --hash=sha256:" + b"3" * 64 + b"\nnumpy==2.3.0\n", "sha256 hash"),
    ("pip-sdist.txt", b"sox==1.5.0 \\\n    --hash=sha256:" + b"3" * 64
     + b"\nSoX==1.5.0 \\\n    --hash=sha256:" + b"3" * 64 + b"\n", "twice"),
])
def test_render_refuses_a_source_build_its_locks_do_not_pin(tmp_path, name, content, match):
    locks = _locks_with(tmp_path, name, content)
    with pytest.raises(ValueError, match=match):
        render_def(TTS, parent_sif=Path("/node/s-a.sif"), locks=locks, uv=UV)
    with pytest.raises(ValueError, match=match):
        recipe_key(TTS, parent_key="s-a-0123456789abcdef", locks_dir=locks, uv=UV)


def test_an_unhashed_pip_lock_is_refused_for_a_base_too(tmp_path):
    locks = _locks_with(tmp_path, "pip-base.txt", b"torch==2.11.0+cu130\n")
    with pytest.raises(ValueError, match="sha256 hash"):
        render_def(BASE, parent_sif=None, locks=locks, uv=UV)


# The rendered install, run by the real uv offline against flat indexes, shows what uv itself does:
# refuse a package without a wheel that the declaration does not name, and build a named one with
# the venv's own backend, fetching nothing, or fail.

_BACKEND = '''
import os
import zipfile

VERSION = "{version}"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    fields = dict(line.split(": ", 1) for line in open("PKG-INFO").read().splitlines() if ": " in line)
    name, version = fields["Name"], fields["Version"]
    wheel = f"{{name}}-{{version}}-py3-none-any.whl"
    dist = f"{{name}}-{{version}}.dist-info"
    with zipfile.ZipFile(os.path.join(wheel_directory, wheel), "w") as z:
        z.writestr(f"{{name}}/__init__.py", f"BUILT_BY = {{VERSION!r}}\\n")
        z.writestr(f"{{dist}}/METADATA", f"Metadata-Version: 2.1\\nName: {{name}}\\nVersion: {{version}}\\n")
        z.writestr(f"{{dist}}/WHEEL",
                   "Wheel-Version: 1.0\\nGenerator: t\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
        z.writestr(f"{{dist}}/RECORD", "")
    return wheel
'''


def _wheel(index: Path, name: str, version: str, files: dict[str, str]) -> Path:
    index.mkdir(exist_ok=True)
    path = index / f"{name}-{version}-py3-none-any.whl"
    dist = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        for member, text in files.items():
            z.writestr(member, text)
        z.writestr(f"{dist}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        z.writestr(f"{dist}/WHEEL",
                   "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{dist}/RECORD", "")
    return path


def _sdist(index: Path, name: str) -> Path:
    """An sdist of *name* 1.0 whose build asks for the backend fakebe and for packaging."""
    path = index / f"{name}-1.0.tar.gz"
    members = {"PKG-INFO": f"Metadata-Version: 2.2\nName: {name}\nVersion: 1.0\n",
               "pyproject.toml": ('[build-system]\nrequires = ["fakebe", "packaging"]\n'
                                  'build-backend = "fakebe"\n')}
    with tarfile.open(path, "w:gz") as tar:
        for member, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-1.0/{member}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _run_install(tmp_path: Path, packages: list[str], source_builds: list[str], *, backend: bool = True,
                 ) -> tuple[subprocess.CompletedProcess, Path]:
    """Render a layer whose pip lock holds *packages* and which builds *source_builds* from source,
    and run its install with uv, offline, into a venv that holds the backend fakebe 1.0 (unless
    not *backend*). The layer's index also offers fakebe 2.0 and packaging, which nothing may take.
    Returns uv's result and the venv's site-packages."""
    index = tmp_path / "index"
    files = {"hasbin": _wheel(index, "hasbin", "1.0", {"hasbin/__init__.py": ""}),
             "onlysrc": _sdist(index, "onlysrc"), "alsosrc": _sdist(index, "alsosrc")}
    _wheel(index, "fakebe", "2.0", {"fakebe/__init__.py": _BACKEND.format(version="2.0")})
    _wheel(index, "packaging", "99.0", {"packaging/__init__.py": ""})
    base_index = tmp_path / "base-index"
    _wheel(base_index, "fakebe", "1.0", {"fakebe/__init__.py": _BACKEND.format(version="1.0")})
    locks = tmp_path / "locks"
    locks.mkdir()
    digests = {p: hashlib.sha256(files[p].read_bytes()).hexdigest() for p in packages}
    entries = [f"{p}==1.0 \\\n    --hash=sha256:{digests[p]}\n" for p in packages]
    (locks / "pip.txt").write_text("".join(entries))
    layer = LayerImage(name="t", parent="s-a", pip_lock="pip.txt",
                       source_builds={f"{p}==1.0": ["fakebe"] for p in source_builds})
    (tmp_path / "tmp").mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "TMPDIR": str(tmp_path / "tmp")}
    uv = ["--no-config", "--cache-dir", str(tmp_path / "cache")]
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--offline", "--python", sys.executable, str(venv), *uv], check=True,
                   capture_output=True, env=env)
    if backend:
        subprocess.run(["uv", "pip", "install", "--offline", "--no-index", "--find-links", str(base_index),
                        "--python", str(venv / "bin" / "python"), "fakebe==1.0", *uv],
                       check=True, capture_output=True, env=env)
    online = f"--index-strategy unsafe-best-match --index-url {TORCH_INDEX} --extra-index-url {PYPI_INDEX}"
    install = _install(layer, locks=locks)
    assert online in install
    install = install.replace(online, f"--offline --no-index --find-links {index}").replace(
        f"{VENV}/bin/python", f"{venv}/bin/python").replace(LOCK_ROOT, str(locks))
    done = subprocess.run([*shlex.split(install), *uv], capture_output=True, text=True, env=env, timeout=120)
    return done, next(venv.glob("lib/python*/site-packages"))


needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="runs the rendered install with uv")


@needs_uv
def test_a_named_source_build_uses_the_venvs_backend_and_fetches_nothing(tmp_path):
    r, site = _run_install(tmp_path, ["hasbin", "onlysrc"], ["onlysrc"])
    assert r.returncode == 0, r.stderr
    # Built by the venv's fakebe 1.0, not the index's 2.0; packaging, which the sdist also asks
    # for, was not fetched.
    assert (site / "onlysrc" / "__init__.py").read_text() == "BUILT_BY = '1.0'\n"
    assert sorted(p.name for p in site.glob("*.dist-info")) == [
        "fakebe-1.0.dist-info", "hasbin-1.0.dist-info", "onlysrc-1.0.dist-info"]


@needs_uv
def test_a_named_source_build_without_its_backend_fails_rather_than_fetching_it(tmp_path):
    r, site = _run_install(tmp_path, ["onlysrc"], ["onlysrc"], backend=False)
    assert r.returncode != 0 and "Failed to build" in r.stderr, r.stderr
    assert list(site.glob("*.dist-info")) == []


@needs_uv
@pytest.mark.parametrize("source_builds", [[], ["onlysrc"]])
def test_a_package_without_a_wheel_that_is_not_named_is_refused(tmp_path, source_builds):
    r, site = _run_install(tmp_path, ["hasbin", "onlysrc", "alsosrc"], source_builds)
    refused = "alsosrc" if source_builds else "onlysrc"
    assert r.returncode != 0, r.stderr
    assert f"{refused}==1.0 has no usable wheels" in " ".join(r.stderr.split()), r.stderr
    assert not (site / "hasbin").exists()


def test_a_layer_builds_on_its_parents_file_and_rechecks_the_held_versions():
    text = render_def(LAYER, parent_sif=Path("/node/s-a.sif"), locks=FIXTURES, uv=UV)
    assert text.startswith("Bootstrap: localimage\nFrom: /node/s-a.sif\n")
    post = "\n".join(_section(text, "post"))
    assert f"-r {LOCK_ROOT}/pip-layer.txt" in post and "/opt/hls/constraints.txt" in post
    # A child never writes %environment or %runscript: apptainer would replace the base's.
    assert "%environment" not in text and "%runscript" not in text
    assert "apt-get" not in post


#: The fixture lock's packages npm installs on linux/x64 (glibc), in lock order: both CLIs and their
#: linux-x64 builds, not the darwin one.
NPM_HERE = ["node_modules/@anthropic-ai/claude-code", "node_modules/@anthropic-ai/claude-code-linux-x64",
            "node_modules/@openai/codex", "node_modules/@openai/codex-linux-x64"]


def _hex(integrity: str) -> str:
    return base64.b64decode(integrity.removeprefix("sha512-")).hex()


def test_the_agent_cap_installs_its_clis_from_their_hashed_lock_and_checks_their_versions():
    text = render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV)
    steps = _steps(text)
    manifest = ('{"dependencies": {"@anthropic-ai/claude-code": "2.1.251", "@openai/codex": "0.146.0"}, '
                '"name": "hls-agent-cli", "private": true}')
    assert npm_manifest(CAP) == manifest
    assert f"printf '%s\\n' '{manifest}' > {NPM_PREFIX}/package.json" in steps
    assert not [step for step in steps if re.search(r"(^|/)npm install\b", step)]
    post = "\n".join(steps)
    assert "|| true" not in post
    assert "mkdir -p /workspace /logs/agent" in post
    assert f"{LOCK_ROOT}/apt-agent.txt" in post
    assert _section(text, "files")[0] == f"{FIXTURES}/npm-agent.json {NPM_PREFIX}/package-lock.json"


def test_the_build_fetches_each_package_npm_installs_here_and_checks_its_sha512_first():
    """npm treats an optional package whose integrity check or download fails as absent, and the
    CLIs' binaries live in optional platform packages; so the build checks them itself."""
    steps = _steps(render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV))
    lock = json.loads((FIXTURES / "npm-agent.json").read_text())["packages"]
    checks = []
    for n, key in enumerate(NPM_HERE, 1):
        fetch = f'curl -fsSL --retry 5 -o "$hls_npm/{n}.tgz" {lock[key]["resolved"]}'
        check = f'echo "{_hex(lock[key]["integrity"])}  $hls_npm/{n}.tgz" | sha512sum -c -'
        assert steps.index(fetch) + 1 == steps.index(check), key
        checks.append(steps.index(check))
    assert len(_hex(lock[NPM_HERE[0]]["integrity"])) == 128
    # Another platform's package is never fetched.
    assert "darwin" not in "\n".join(steps)
    # npm then installs offline, from a cache holding the checked files alone.
    files = " ".join(f'"$hls_npm/{n}.tgz"' for n in range(1, len(NPM_HERE) + 1))
    add = (f'PATH="{NODE_PREFIX}/bin:$PATH" {NODE_PREFIX}/bin/npm cache add --offline --no-update-notifier '
           f'--cache "$hls_npm/cache" {files}')
    ci = (f'PATH="{NODE_PREFIX}/bin:$PATH" {NODE_PREFIX}/bin/npm ci --prefix {NPM_PREFIX} --offline '
          '--cache "$hls_npm/cache" --engine-strict --no-fund --no-audit --no-update-notifier')
    assert max(checks) < steps.index(add) < steps.index(ci) < steps.index('rm -rf "$hls_npm"')


def test_npm_itself_says_which_packages_it_installs_here_and_which_it_installed():
    """Neither answer depends on the build's own platform filter: npm's install checks name the
    packages it installs here before anything is fetched, and npm ls names the ones it installed."""
    steps = _steps(render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV))
    checks = f"{NODE_PREFIX}/lib/node_modules/npm/node_modules/npm-install-checks"
    plan = next(i for i, s in enumerate(steps) if checks in s)
    assert steps[plan].startswith(f"{NODE_PREFIX}/bin/node -e ")
    assert steps[plan].endswith(f"{checks} {NPM_PREFIX}/package-lock.json " + " ".join(NPM_HERE))
    first_fetch = next(i for i, s in enumerate(steps) if s.startswith('curl -fsSL --retry 5 -o "$hls_npm/'))
    node_check = next(i for i, s in enumerate(steps) if "--version)\" || hls_node=absent" in s)
    assert node_check < plan < first_fetch
    ls = (f'PATH="{NODE_PREFIX}/bin:$PATH" {NODE_PREFIX}/bin/npm ls --prefix {NPM_PREFIX} --all --json '
          '--long --offline --cache "$hls_npm/cache" > "$hls_npm/ls.json"')
    installed = steps.index(ls) + 1
    assert steps[installed].startswith(f"{NODE_PREFIX}/bin/node -e ")
    assert steps[installed].endswith(f'"$hls_npm/ls.json" {NPM_PREFIX} ' + " ".join(
        f"{key}={version}" for key, version in INSTALLED.items()))
    ci = next(i for i, s in enumerate(steps) if f"{NODE_PREFIX}/bin/npm ci " in s)
    assert ci < steps.index(ls) < installed < steps.index('rm -rf "$hls_npm"')
    assert "for hls_cli in claude=2.1.251 codex=0.146.0; do" in "\n".join(steps)


#: Platform lists of one lock entry, and whether npm installs it on linux/x64 with glibc.
PLATFORMS = [
    ({"os": ["!win32"], "cpu": ["!arm64"]}, True),
    ({"os": ["!win32", "!darwin"], "cpu": ["x64"]}, True),
    ({"os": ["!linux"], "cpu": ["x64"]}, False),
    ({"os": ["linux", "!win32"], "cpu": ["!x64"]}, False),
    ({"os": ["darwin", "!win32"]}, False),
    ({"os": ["linux"], "cpu": ["x64"], "libc": ["!musl"]}, True),
    ({"os": ["linux"], "cpu": ["x64"], "libc": ["!glibc"]}, False),
    ({"os": ["linux"], "cpu": ["x64"], "libc": ["musl"]}, False),
    ({"os": "linux", "cpu": "x64"}, True),
    ({"os": "darwin"}, False),
    ({"os": ["any"], "cpu": ["any"]}, True),
    ({}, True),
]


@pytest.mark.parametrize("platform, here", PLATFORMS)
def test_a_package_is_fetched_exactly_when_npm_installs_it_here(tmp_path, platform, here):
    def change(lock):
        entry = _entry(lock, "@anthropic-ai/claude-code-darwin-arm64")
        del entry["os"], entry["cpu"]
        entry.update(platform)

    text = render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=_npm_lock_with(tmp_path, change), uv=UV)
    assert ("claude-code-darwin-arm64-2.1.251.tgz" in text) is here
    assert ("node_modules/@anthropic-ai/claude-code-darwin-arm64=2.1.251" in text) is here


def _npm_install_checks() -> Path | None:
    """npm-install-checks of the host node's own npm, if it has one."""
    if HOST_NODE is None:
        return None
    module = Path(HOST_NODE).resolve().parents[1] / "lib" / "node_modules" / "npm" / "node_modules" / \
        "npm-install-checks"
    return module if module.is_dir() else None


@pytest.mark.skipif(_npm_install_checks() is None, reason="needs a node with its npm on the host")
def test_the_platform_cases_are_the_ones_npm_itself_decides():
    """The expectations above are npm's own: its checkPlatform on linux/x64 with glibc."""
    script = ("const { checkPlatform } = require(process.argv[1]);"
              "for (const target of JSON.parse(process.argv[2])) {"
              " try { checkPlatform(target, false, { os: 'linux', cpu: 'x64', libc: 'glibc' });"
              " console.log('yes') }"
              " catch (e) { if (e.code !== 'EBADPLATFORM') throw e; console.log('no') } }")
    done = subprocess.run([HOST_NODE, "-e", script, str(_npm_install_checks()),
                           json.dumps([platform for platform, _ in PLATFORMS])],
                          capture_output=True, text=True, check=True)
    assert done.stdout.split() == ["yes" if here else "no" for _, here in PLATFORMS]


def test_the_agent_cap_keeps_ubuntus_node_and_runs_npm_and_its_clis_on_the_pinned_one():
    """The agent sees Ubuntu's /usr/bin/node and npm; the pinned Node,
    which claude-code 2.1.251 needs, lives in its own prefix."""
    assert [line.split("=")[0] for line in (FIXTURES / "apt-agent.txt").read_text().splitlines()
            if line and not line.startswith("#")] == ["tmux", "nodejs", "npm"]
    steps = _steps(render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV))
    post = "\n".join(steps)
    assert NODE_PREFIX == "/opt/hls/node"
    assert (f'tar -xzf "$hls_dl/node.tar.gz" -C {NODE_PREFIX} --strip-components=1 --no-same-owner '
            f"{NODE_TOP}/bin {NODE_TOP}/include {NODE_TOP}/lib {NODE_TOP}/share") in steps
    assert "-C /usr/local" not in post
    assert "/usr/local/bin/node" not in post and "/usr/local/bin/npm" not in post
    assert f'hls_node="$({NODE_PREFIX}/bin/node --version)" || hls_node=absent' in post
    # Every npm and node the build runs is the pinned one, never one found on PATH.
    assert not [s for s in steps if s.startswith(("npm ", "node "))]
    scripts = [s for s in steps if "require(process.argv[1])" in s]
    assert len(scripts) == 2 and all(s.startswith(f"{NODE_PREFIX}/bin/node -e ") for s in scripts)


def _npm_lock_with(tmp_path: Path, change) -> Path:
    """A copy of the fixture locks whose npm lock is the fixture's after *change* edits it."""
    lock = json.loads((FIXTURES / "npm-agent.json").read_text())
    change(lock)
    return _locks_with(tmp_path, "npm-agent.json", json.dumps(lock).encode())


def _entry(lock: dict, name: str) -> dict:
    return lock["packages"][f"node_modules/{name}"]


@pytest.mark.parametrize("change, match", [
    (lambda lock: _entry(lock, "@openai/codex-linux-x64").pop("integrity"), "sha512 integrity"),
    (lambda lock: _entry(lock, "@openai/codex").update(integrity="sha1-" + "A" * 27 + "="),
     "sha512 integrity"),
    (lambda lock: _entry(lock, "@openai/codex").update(resolved="http://registry.npmjs.org/x.tgz"), "https"),
    (lambda lock: _entry(lock, "@openai/codex").update(link=True), "link"),
    (lambda lock: lock["packages"].update({"vendor/x": {"version": "1.0.0"}}), "node_modules"),
    (lambda lock: lock["packages"][""]["dependencies"].pop("@openai/codex"), "depend on exactly"),
    (lambda lock: lock["packages"][""]["dependencies"].update({"left-pad": "1.3.0"}), "depend on exactly"),
    (lambda lock: lock["packages"][""].update(devDependencies={"left-pad": "1.3.0"}), "depend on exactly"),
    (lambda lock: _entry(lock, "@openai/codex").update(version="0.145.0"), "at 0.146.0"),
    (lambda lock: _entry(lock, "@openai/codex").pop("bin"), "no command"),
    (lambda lock: _entry(lock, "@openai/codex").update(bin={}), "no command"),
    (lambda lock: _entry(lock, "@openai/codex").update(bin={"co dex": "x.js"}), "command name"),
    (lambda lock: _entry(lock, "@openai/codex").update(bin={"claude": "x.js"}), "twice"),
    (lambda lock: lock.update(lockfileVersion=1), "version 2 or 3"),
    # Each CLI's platform packages: every optional dependency locked, at its exact version.
    (lambda lock: lock["packages"].pop("node_modules/@anthropic-ai/claude-code-linux-x64"),
     "does not lock @anthropic-ai/claude-code-linux-x64"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-linux-x64").update(version="2.1.250"),
     "locks @anthropic-ai/claude-code-linux-x64 as @anthropic-ai/claude-code-linux-x64@2.1.250, not 2.1.251"),
    (lambda lock: _entry(lock, "@openai/codex-linux-x64").update(version="0.145.0-linux-x64"),
     "not npm:@openai/codex@0.146.0-linux-x64"),
    (lambda lock: _entry(lock, "@openai/codex-linux-x64").update(name="@openai/codex-other"),
     "not npm:@openai/codex@0.146.0-linux-x64"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code")["optionalDependencies"].update(
        {"@anthropic-ai/claude-code-linux-x64": "^2.1.0"}), "not one exact version"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-linux-x64").update(os=["darwin"]),
     "no package for linux/x64"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-linux-x64").update(libc=["musl"]),
     "no package for linux/x64"),
    (lambda lock: _entry(lock, "@openai/codex-linux-x64").update(cpu=["!x64"]), "no package for linux/x64"),
    (lambda lock: _entry(lock, "@openai/codex").update(resolved="https://registry.npmjs.org/x y.tgz"),
     "plain https URL"),
    # Every entry is a plain path at one exact version: both reach the build's shell.
    (lambda lock: lock["packages"].update({"node_modules/Left Pad": _entry(lock, "@openai/codex")}),
     "not a plain package path at one exact version"),
    (lambda lock: lock["packages"].update({"node_modules/a/../b": _entry(lock, "@openai/codex")}),
     "not a plain package path at one exact version"),
    (lambda lock: lock["packages"].update({"node_modules/a/b": _entry(lock, "@openai/codex")}),
     "not a plain package path at one exact version"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-darwin-arm64").update(version="2.1"),
     "not a plain package path at one exact version"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-darwin-arm64").update(version="^2.1.251"),
     "not a plain package path at one exact version"),
    (lambda lock: _entry(lock, "@anthropic-ai/claude-code-darwin-arm64").update(version="2.1.251=x"),
     "not a plain package path at one exact version"),
])
def test_an_npm_lock_that_does_not_hash_and_pin_the_declared_clis_is_refused(tmp_path, change, match):
    locks = _npm_lock_with(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=locks, uv=UV)
    with pytest.raises(ValueError, match=match):
        recipe_key(CAP, parent_key="s-a-tf-0123456789abcdef", locks_dir=locks, uv=UV)


def test_a_missing_or_unreadable_npm_lock_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="npm-agent.json"):
        render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=_locks_without(tmp_path, "npm-agent.json"),
                   uv=UV)
    with pytest.raises(ValueError, match="not JSON"):
        render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"),
                   locks=_locks_with(tmp_path / "b", "npm-agent.json", b"{"), uv=UV)


def _locks_without(tmp_path: Path, name: str) -> Path:
    """A copy of the fixture locks without *name*."""
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    (locks / name).unlink()
    return locks


@pytest.mark.parametrize("decl, parent, match", [
    (BASE, Path("/node/x.sif"), "has no parent"),
    (LAYER, None, "needs its parent"),
])
def test_render_refuses_a_parent_that_does_not_fit(decl, parent, match):
    with pytest.raises(ValueError, match=match):
        render_def(decl, parent_sif=parent, locks=FIXTURES, uv=UV)


def test_render_refuses_a_missing_lock_and_a_path_with_whitespace(tmp_path):
    with pytest.raises(FileNotFoundError, match="apt-base.txt"):
        render_def(BASE, parent_sif=None, locks=tmp_path, uv=UV)
    spaced = tmp_path / "with space"
    shutil.copytree(FIXTURES, spaced)
    with pytest.raises(ValueError, match="whitespace"):
        render_def(BASE, parent_sif=None, locks=spaced, uv=UV)


def _locks_with(tmp_path: Path, name: str, content: bytes) -> Path:
    """A copy of the fixture locks with *name* holding *content*."""
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    (locks / name).write_bytes(content)
    return locks


# Every apt lock line reaches apt-get's arguments word-split, so only name=version may.
# sed drops only a line that starts with # or is empty, so an indented comment or a blank line of
# spaces would reach apt-get too.
@pytest.mark.parametrize("line", ["-oDpkg::Pre-Invoke::=touch /pwned", "curl", "curl=8.5.0 python3=3.12.3",
                                  "curl=8.5.0\r", " curl=8.5.0", "curl=*", "  # indented comment",
                                  "\t# tab", "   "])
def test_an_apt_lock_line_that_is_not_a_pin_is_refused(tmp_path, line):
    locks = _locks_with(tmp_path, "apt-base.txt", b"# lock\ncurl=8.5.0-2ubuntu10.6\n" + line.encode() + b"\n")
    with pytest.raises(ValueError, match="not an apt pin"):
        render_def(BASE, parent_sif=None, locks=locks, uv=UV)
    with pytest.raises(ValueError, match="not an apt pin"):
        recipe_key(BASE, parent_key=None, locks_dir=locks, uv=UV)


def test_an_apt_lock_that_pins_nothing_is_refused(tmp_path):
    locks = _locks_with(tmp_path, "apt-agent.txt", b"# nothing pinned\n\n")
    with pytest.raises(ValueError, match="pins nothing"):
        render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=locks, uv=UV)


def test_files_for_maps_container_paths_to_lock_files():
    assert files_for(CAP, FIXTURES) == {f"{NPM_PREFIX}/package-lock.json": FIXTURES / "npm-agent.json",
                                        f"{LOCK_ROOT}/apt-agent.txt": FIXTURES / "apt-agent.txt"}


def test_the_renderer_breaks_lines_only_at_newlines():
    # A value the model would refuse, built around it: the renderer's own split must not open a section.
    unchecked = BaseImage.model_construct(**{**dict(BASE), "env": {"X": "\x0c%post\x0ctouch /pwned #"}})
    text = render_def(unchecked, parent_sif=None, locks=FIXTURES, uv=UV)
    assert [line for line in text.split("\n") if line.strip().startswith("%post")] == ["%post"]


def test_a_def_never_carries_a_build_argument(tmp_path):
    # Apptainer substitutes "{{ name }}" anywhere in a .def.
    braced = tmp_path / "a{{b}}"
    shutil.copytree(FIXTURES, braced)
    with pytest.raises(ValueError, match="build argument"):
        render_def(BASE, parent_sif=None, locks=braced, uv=UV)


def test_env_values_are_shell_quoted():
    decl = BASE.model_copy(update={"env": {"A": "x y; $HOME `id`"}})
    text = render_def(decl, parent_sif=None, locks=FIXTURES, uv=UV)
    assert "export A='x y; $HOME `id`'" in _section(text, "environment")


@pytest.mark.parametrize("parent", ["/node/with space.sif", "/node/a.sif\nFrom: other.sif",
                                    "/node/a\x07.sif"])
def test_render_refuses_a_parent_path_a_def_cannot_carry(parent):
    with pytest.raises(ValueError, match="whitespace"):
        render_def(LAYER, parent_sif=Path(parent), locks=FIXTURES, uv=UV)


def test_rendering_is_deterministic():
    assert render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV) == render_def(
        BASE.model_copy(update={"env": dict(reversed(list(BASE.env.items())))}), parent_sif=None,
        locks=FIXTURES, uv=UV)


def _key(decl, locks=FIXTURES, parent="s-a-0123456789abcdef", uv=UV):
    return recipe_key(decl, parent_key=None if isinstance(decl, BaseImage) else parent, locks_dir=locks,
                      uv=uv)


def test_a_key_is_the_name_and_sixteen_hex_digits():
    assert re.fullmatch(r"s-a-[0-9a-f]{16}", _key(BASE))


def test_a_key_follows_every_input(tmp_path):
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    before = {"base": _key(BASE, locks), "layer": _key(LAYER, locks), "cap": _key(CAP, locks)}
    (locks / "pip-base.txt").write_text((locks / "pip-base.txt").read_text() + "# one more byte\n")
    assert _key(BASE, locks) != before["base"]
    (locks / "pip-layer.txt").write_text("transformers==5.13.2 \\\n    --hash=sha256:" + "3" * 64 + "\n")
    assert _key(LAYER, locks) != before["layer"]
    assert _key(LAYER, FIXTURES, parent="s-a-fedcba9876543210") != _key(LAYER, FIXTURES)
    assert _key(BASE, FIXTURES, uv=UV.model_copy(update={"sha256": "9" * 64})) != _key(BASE, FIXTURES)
    assert _key(BASE.model_copy(update={"env": {**BASE.env, "PYTHONUNBUFFERED": "0"}})) != _key(BASE)
    # Another CLI version needs a lock that pins it, and changes the key.
    newer = _npm_lock_with(tmp_path / "newer", lambda lock: (
        lock["packages"][""]["dependencies"].update({"@openai/codex": "0.147.0"}),
        _entry(lock, "@openai/codex").update(version="0.147.0")))
    assert _key(CAP.model_copy(update={"npm": {**CAP.npm, "@openai/codex": "0.147.0"}}), newer) != _key(CAP)
    # So is the Node the CLIs run on.
    assert _key(CAP.model_copy(update={"node": NODE.model_copy(update={"sha256": "9" * 64})})) != _key(CAP)
    # The source builds are part of the key.
    both = {"sox==1.5.0": ["setuptools"], "transformers==4.57.3": ["setuptools"]}
    assert _key(TTS.model_copy(update={"source_builds": both})) != _key(TTS)
    # So is the npm lock.
    cap = _key(CAP, locks)
    (locks / "npm-agent.json").write_text((locks / "npm-agent.json").read_text() + "\n")
    assert _key(CAP, locks) != cap


def test_a_key_does_not_depend_on_where_the_locks_or_the_parent_file_are(tmp_path):
    elsewhere = tmp_path / "deep" / "locks"
    shutil.copytree(FIXTURES, elsewhere)
    assert _key(BASE, elsewhere) == _key(BASE) and _key(LAYER, elsewhere) == _key(LAYER)


def test_a_key_refuses_a_parent_key_that_does_not_fit():
    with pytest.raises(ValueError, match="has no parent"):
        recipe_key(BASE, parent_key="x-0123456789abcdef", locks_dir=FIXTURES, uv=UV)
    with pytest.raises(ValueError, match="needs its parent"):
        recipe_key(LAYER, parent_key=None, locks_dir=FIXTURES, uv=UV)


# The %post script itself, run under sh with stand-ins for the tools, to pin that a failing
# step ends the build and that the version checks fail closed.


def _post_script(text: str, root: Path) -> str:
    """The %post body with the image paths moved under *root*."""
    body = "\n".join(_section(text, "post"))
    return body.replace("/opt/hls", str(root / "opt-hls")).replace("/opt/venv", str(root / "venv"))


def _sh(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env)


def _stub(bin_dir: Path, name: str, script: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(0o755)


def _run_layer_post(tmp_path: Path, *, apt_rc: int, dpkg_version: str | None,
                    held_version: str) -> subprocess.CompletedProcess:
    """Run a layer's %post with stand-ins: apt-get exits *apt_rc* on install, dpkg-query
    reports each package at its locked version (or at *dpkg_version*), and the venv's python
    sees torch at *held_version*."""
    layer = LayerImage(name="s-a-tools", parent="s-a", apt_lock="apt-agent.txt", pip_lock="pip-layer.txt")
    text = render_def(layer, parent_sif=Path("/node/s-a.sif"), locks=FIXTURES, uv=UV)
    locks = tmp_path / "opt-hls" / "locks"
    locks.mkdir(parents=True)
    for name in ("apt-agent.txt", "pip-layer.txt"):
        shutil.copy(FIXTURES / name, locks / name)
    (tmp_path / "opt-hls" / "constraints.txt").write_text("torch==2.11.0+cu130\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "apt-get", f'[ "$1" = install ] && exit {apt_rc}\nexit 0\n')
    locked = f'"$(grep "^$3=" {locks}/apt-agent.txt | cut -d= -f2-)"'
    version = f'"{dpkg_version}"' if dpkg_version else locked
    _stub(bin_dir, "dpkg-query", f"printf 'install ok installed %s' {version}\n")
    _stub(bin_dir, "uv", "exit 0\n")
    _stub(bin_dir, "rm", "exit 0\n")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    _stub(venv_bin, "python", f'exec {shutil.which("python3")} "$@"\n')
    dist = tmp_path / "site" / f"torch-{held_version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torch\nVersion: {held_version}\n")
    script = _post_script(text, tmp_path) + "\necho reached-the-end\n"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(tmp_path / "site")}
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env)


def test_the_post_script_runs_to_the_end_when_every_step_succeeds(tmp_path):
    r = _run_layer_post(tmp_path, apt_rc=0, dpkg_version=None, held_version="2.11.0+cu130")
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr


def test_a_failed_apt_install_ends_the_build(tmp_path):
    r = _run_layer_post(tmp_path, apt_rc=100, dpkg_version=None, held_version="2.11.0+cu130")
    assert r.returncode == 100 and "reached-the-end" not in r.stdout


def test_a_package_at_another_version_ends_the_build(tmp_path):
    r = _run_layer_post(tmp_path, apt_rc=0, dpkg_version="0.0-other", held_version="2.11.0+cu130")
    assert r.returncode == 1 and "missing or at another version" in r.stderr
    assert "reached-the-end" not in r.stdout


def test_a_layer_that_moves_a_held_package_ends_the_build(tmp_path):
    r = _run_layer_post(tmp_path, apt_rc=0, dpkg_version=None, held_version="2.13.0+cu130")
    assert r.returncode != 0 and "held packages at another version" in r.stderr
    assert "reached-the-end" not in r.stdout


def test_a_held_package_built_for_another_cuda_ends_the_build(tmp_path):
    # The local version is part of the hold: a +cu126 torch is not the +cu130 one.
    r = _run_layer_post(tmp_path, apt_rc=0, dpkg_version=None, held_version="2.11.0+cu126")
    assert r.returncode != 0 and "held packages at another version" in r.stderr


NSIGHT_PIN = "cuda-nsight-systems-13-0=2025.3.2.474-1"


def _run_base_apt(tmp_path: Path, *, deb: bytes | None = b"deb " + NSIGHT_PIN.encode(),
                  nsight_version: str = "2025.3.2.474-1") -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run a base's apt steps, from its apt-get update to its uv download, under sh with the image
    paths moved under *tmp_path* and stand-ins: ``apt-get download`` leaves *deb* as each named
    package's .deb (none for None), every apt-get call is logged with whether the preferences
    file exists yet, dpkg-query reports the locked packages at their versions and Nsight Systems
    at *nsight_version*, and rm removes only paths under *tmp_path*. Return the run and the log."""
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post")
    start = post.index("apt-get update")
    end = next(i for i, line in enumerate(post) if line.startswith("hls_dl="))
    script = "\n".join(post[start:end]).replace("/opt/hls", str(tmp_path / "opt-hls")).replace(
        "/etc/apt", str(tmp_path / "etc-apt"))
    locks = tmp_path / "opt-hls" / "locks"
    locks.mkdir(parents=True)
    (tmp_path / "etc-apt" / "preferences.d").mkdir(parents=True)
    shutil.copy(FIXTURES / "apt-base.txt", locks / "apt-base.txt")
    if deb is not None:
        (tmp_path / "deb").write_bytes(deb)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    prefs = tmp_path / "etc-apt" / "preferences.d" / "hls-nvidia"
    log = tmp_path / "apt.log"
    _stub(bin_dir, "apt-get", f'if [ -f {prefs} ]; then p=prefs; else p=no-prefs; fi\n'
                              f'echo "$p $*" >> {log}\n'
                              '[ "$1" = download ] || exit 0\nshift\n'
                              f'[ -f {tmp_path}/deb ] || exit 0\n'
                              'for pin in "$@"; do\n'
                              f'  cp {tmp_path}/deb "${{pin%%=*}}_${{pin#*=}}_amd64.deb"\n'
                              'done\n')
    _stub(bin_dir, "rm", f'case "$*" in *{tmp_path}/*) exec {shutil.which("rm")} "$@";; esac\nexit 0\n')
    _stub(bin_dir, "dpkg-query", f'v="$(grep "^$3=" {locks}/apt-base.txt | cut -d= -f2-)"\n'
                                 f'printf \'install ok installed %s\' "${{v:-{nsight_version}}}"\n')
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}"}
    r = _sh(script + "\necho reached-the-end\n", env)
    return r, log.read_text().splitlines() if log.exists() else []


def test_nvidia_packages_install_from_debs_checked_against_their_recorded_digests(tmp_path):
    r, log = _run_base_apt(tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    debs = next((tmp_path / "opt-hls").glob("nvidia-debs.*"), None)
    assert debs is None, "the checked .deb files are removed once installed"
    assert log[0] == "no-prefs update"
    assert log[1] == f"no-prefs download {NSIGHT_PIN}"
    # The install names the checked file, not the repository's version, and runs once the
    # preferences keep every version from NVIDIA's repository out.
    install = log[2].split()
    assert install[:5] == ["prefs", "install", "-y", "--no-install-recommends", "curl=8.5.0-2ubuntu10.6"]
    assert install[-1].startswith(str(tmp_path / "opt-hls" / "nvidia-debs."))
    assert install[-1].endswith("/cuda-nsight-systems-13-0_2025.3.2.474-1_amd64.deb")
    assert NSIGHT_PIN not in log[2]


def test_apt_preferences_hold_back_every_version_from_nvidias_repository(tmp_path):
    r, _ = _run_base_apt(tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "etc-apt" / "preferences.d" / "hls-nvidia").read_text() == (
        "Package: *\nPin: origin \"developer.download.nvidia.com\"\nPin-Priority: -1\n")


def test_a_nvidia_deb_that_does_not_match_its_recorded_digest_ends_the_build_before_installing(tmp_path):
    r, log = _run_base_apt(tmp_path, deb=b"republished under the same version")
    assert r.returncode != 0 and "reached-the-end" not in r.stdout
    assert "FAILED" in r.stdout + r.stderr
    assert [line for line in log if " install " in f" {line} "] == []


def test_a_nvidia_package_the_repository_no_longer_serves_ends_the_build(tmp_path):
    r, log = _run_base_apt(tmp_path, deb=None)
    assert r.returncode == 1 and "reached-the-end" not in r.stdout
    assert "apt-get download left no single .deb of cuda-nsight-systems-13-0" in r.stderr
    assert [line for line in log if " install " in f" {line} "] == []


def test_the_nvidia_pins_are_checked_like_the_locked_ones(tmp_path):
    r, _ = _run_base_apt(tmp_path, nsight_version="2025.1.1-1")
    assert r.returncode == 1 and "reached-the-end" not in r.stdout
    assert f"missing or at another version: {NSIGHT_PIN}" in r.stderr


def test_a_base_with_no_nvidia_packages_downloads_none_but_still_holds_the_repository_back():
    post = "\n".join(_section(render_def(BASE.model_copy(update={"nvidia_apt": {}}), parent_sif=None,
                                          locks=FIXTURES, uv=UV), "post"))
    assert "apt-get download" not in post and "nvidia-debs" not in post and "$hls_nvidia_debs" not in post
    assert "Pin-Priority: -1" in post
    bare = BASE.model_copy(update={"nvidia_apt": {}, "nvidia_repo": None})
    assert "hls-nvidia" not in render_def(bare, parent_sif=None, locks=FIXTURES, uv=UV)


def _cap_locks(tmp_path: Path, change=None) -> tuple[Path, dict[str, bytes]]:
    """A copy of the fixture locks whose npm lock, after *change* edits it, checks each package of
    NPM_HERE by the sha512 of a stand-in tarball, and those tarballs by their URL's last segment."""
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    lock = json.loads((locks / "npm-agent.json").read_text())
    if change is not None:
        change(lock)
    served = {}
    for key in NPM_HERE:
        data = f"tarball of {key}".encode()
        entry = lock["packages"][key]
        entry["integrity"] = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
        served[entry["resolved"].rsplit("/", 1)[1]] = data
    (locks / "npm-agent.json").write_text(json.dumps(lock))
    return locks, served


#: What the stand-in npm ci installs by default: every package npm installs here, at its version.
INSTALLED = {"node_modules/@anthropic-ai/claude-code": "2.1.251",
             "node_modules/@anthropic-ai/claude-code-linux-x64": "2.1.251",
             "node_modules/@openai/codex": "0.146.0",
             "node_modules/@openai/codex-linux-x64": "0.146.0-linux-x64"}


def _run_cap_post(tmp_path: Path, *, tamper: str | None = None, unreachable: str | None = None,
                  installed: dict[str, str] = INSTALLED, prints: dict[str, str] | None = None,
                  node: bytes | None = None, node_served: bytes | None = None, change=None,
                  ) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the agent cap's %post under sh with the image paths moved under *tmp_path* and
    stand-ins: apt-get does nothing, dpkg-query reports each apt pin installed, rm removes only
    paths under *tmp_path*, and curl serves the Node tarball the cap pins (*node*, by default
    the one NODE pins; *node_served* instead when given) and each npm package's stand-in tarball,
    but other bytes for the package whose URL ends in *tamper* and nothing for the one whose URL
    ends in *unreachable*. The Node tarball's npm
    installs *installed* (lock path to version) on ``ci``; claude.exe, a shell script, prints
    ``prints["claude"]`` and codex.js, a node script, prints ``prints["codex"]`` and which node ran
    it, and writes into $HOME. A node and npm of the system's own (Ubuntu's 18) come after the
    stand-ins on PATH. Return the run and the npm log."""
    if HOST_NODE is None:
        pytest.skip("the build's npm checks are JavaScript and need a node")
    prints = {"claude": "2.1.251 (Claude Code)", "codex": "codex-cli 0.146.0", **(prints or {})}
    locks, served = _cap_locks(tmp_path, change)
    node = _node_tarball() if node is None else node
    pin = NODE.model_copy(update={"sha256": hashlib.sha256(node).hexdigest()})
    cap = CAP.model_copy(update={"node": pin})
    text = render_def(cap, parent_sif=Path("/node/s-a-tf.sif"), locks=locks, uv=UV)
    script = "\n".join(_section(text, "post"))
    for path, name in (("/opt/hls", "opt-hls"), ("/usr/local", "usr-local"), ("/workspace", "workspace"),
                       ("/logs/agent", "logs-agent")):
        script = script.replace(path, str(tmp_path / name))
    (tmp_path / "opt-hls" / "locks").mkdir(parents=True)
    (tmp_path / "opt-hls" / "agent-cli").mkdir()
    # As %files copies it in.
    shutil.copy(locks / "npm-agent.json", tmp_path / "opt-hls" / "agent-cli" / "package-lock.json")
    shutil.copy(FIXTURES / "apt-agent.txt", tmp_path / "opt-hls" / "locks" / "apt-agent.txt")
    (tmp_path / "usr-local" / "bin").mkdir(parents=True)
    serve = tmp_path / "served"
    serve.mkdir()
    (serve / f"{NODE_TOP}.tar.gz").write_bytes(node if node_served is None else node_served)
    for name, data in served.items():
        if not unreachable or not name.endswith(unreachable):
            (serve / name).write_bytes(b"other bytes" if tamper and name.endswith(tamper) else data)
    modules = tmp_path / "opt-hls" / "agent-cli" / "node_modules"
    ci = [f"m={modules}"]
    for key, version in installed.items():
        sub = key.removeprefix("node_modules/")
        ci += [f'mkdir -p "$m/{sub}"', f"printf '{{\"version\": \"{version}\"}}' > \"$m/{sub}/package.json\""]
    ci += ['mkdir -p "$m/.bin" "$m/@anthropic-ai/claude-code/bin" "$m/@openai/codex/bin"',
           f"printf '#!/bin/sh\\necho \"{prints['claude']}\"\\n' "
           '> "$m/@anthropic-ai/claude-code/bin/claude.exe"',
           f"printf '#!/usr/bin/env node\\necho \"{prints['codex']} ran-by=$HLS_RAN_BY\"\\n"
           "mkdir -p \"$HOME/.codex\"\\n' > \"$m/@openai/codex/bin/codex.js\"",
           'chmod 0755 "$m/@anthropic-ai/claude-code/bin/claude.exe" "$m/@openai/codex/bin/codex.js"',
           'ln -s ../@anthropic-ai/claude-code/bin/claude.exe "$m/.bin/claude"',
           'ln -s ../@openai/codex/bin/codex.js "$m/.bin/codex"']
    (tmp_path / "npm-ci.sh").write_text("\n".join(ci) + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "apt-get", "exit 0\n")
    _stub(bin_dir, "rm", f'case "$*" in *{tmp_path}/*) exec {shutil.which("rm")} "$@";; esac\nexit 0\n')
    _stub(bin_dir, "curl", 'while [ $# -gt 0 ]; do\n'
                           '  case "$1" in -o) out="$2"; shift ;; https://*) url="$1" ;; esac\n'
                           '  shift\ndone\n'
                           f'[ -f "{serve}/${{url##*/}}" ] || exit 22\ncp "{serve}/${{url##*/}}" "$out"\n')
    _stub(bin_dir, "dpkg-query", 'printf \'install ok installed %s\' "$(grep "^$3=" '
                                 f'{FIXTURES}/apt-agent.txt | cut -d= -f2-)"\n')
    system = tmp_path / "system"
    system.mkdir()
    _stub(system, "node", 'echo "v18.19.1"\nexit 0\n')
    _stub(system, "npm", 'echo "the system npm ran: $*" >&2\nexit 1\n')
    (tmp_path / "root").mkdir()
    env = {"PATH": f"{bin_dir}:{system}:{os.environ['PATH']}", "HOME": str(tmp_path / "root"),
           "HLS_NPM_LOG": str(tmp_path / "npm.log"), "HLS_NPM_CI": str(tmp_path / "npm-ci.sh")}
    r = _sh(script + "\necho reached-the-end\n", env)
    log = tmp_path / "npm.log"
    return r, log.read_text().splitlines() if log.exists() else []


def test_the_agent_cap_installs_its_checked_packages_and_runs_its_clis_on_the_pinned_node(tmp_path):
    r, log = _run_cap_post(tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    pinned = tmp_path / "opt-hls" / "node" / "bin"
    # npm ran twice, both times on the pinned Node: cache add of the checked files, then ci offline.
    assert [line.split(" ARGS=")[1].split()[:2] for line in log] == [["cache", "add"], ["ci", "--prefix"],
                                                                     ["ls", "--prefix"]]
    assert all(line.startswith(f"PATH={pinned}:") for line in log)
    added = log[0].split(" ARGS=")[1].split()
    assert [Path(a).name for a in added if a.endswith(".tgz")] == ["1.tgz", "2.tgz", "3.tgz", "4.tgz"]
    local = tmp_path / "usr-local" / "bin"
    assert sorted(p.name for p in local.iterdir()) == ["claude", "codex"]
    modules = tmp_path / "opt-hls" / "agent-cli" / "node_modules"
    # A native or shell entry point is linked; a node script runs on the pinned Node.
    claude = modules / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    assert os.readlink(local / "claude") == str(claude)
    assert (local / "codex").read_text() == (
        f'#!/bin/sh\nexec {pinned}/node "{modules}/@openai/codex/bin/codex.js" "$@"\n')
    # The downloads, npm's cache and the CLIs' throwaway HOME are gone, and the version check left
    # nothing in the build's own HOME, though codex writes into whichever HOME it runs with.
    assert sorted(p.name for p in (tmp_path / "opt-hls").iterdir()) == ["agent-cli", "locks", "node"]
    assert list((tmp_path / "root").iterdir()) == []
    (tmp_path / "after").mkdir()
    ran = subprocess.run([str(local / "codex"), "--version"], capture_output=True, text=True,
                         env={"PATH": f"{tmp_path / 'system'}:/usr/bin:/bin",
                              "HOME": str(tmp_path / "after")})
    assert ran.stdout.strip() == "codex-cli 0.146.0 ran-by=pinned-node"
    assert [p.name for p in (tmp_path / "after").iterdir()] == [".codex"]
    assert (tmp_path / "workspace").is_dir() and (tmp_path / "logs-agent").is_dir()
    # Node's own tree lands in its prefix, its top-level files do not.
    assert sorted(p.name for p in (pinned.parent).iterdir()) == ["bin", "include", "lib", "share"]


@pytest.mark.parametrize("package", ["claude-code-linux-x64-2.1.251.tgz", "codex-0.146.0-linux-x64.tgz",
                                     "codex-0.146.0.tgz"])
def test_a_tampered_package_tarball_ends_the_build_before_npm_runs(tmp_path, package):
    r, log = _run_cap_post(tmp_path, tamper=package)
    assert r.returncode != 0 and "reached-the-end" not in r.stdout
    assert "FAILED" in r.stdout + r.stderr and log == []
    assert list((tmp_path / "usr-local" / "bin").iterdir()) == []


@pytest.mark.parametrize("package", ["claude-code-linux-x64-2.1.251.tgz", "codex-0.146.0-linux-x64.tgz"])
def test_an_unreachable_platform_package_ends_the_build_before_npm_runs(tmp_path, package):
    r, log = _run_cap_post(tmp_path, unreachable=package)
    assert r.returncode == 22 and "reached-the-end" not in r.stdout and log == []


@pytest.mark.parametrize("key, version, shown", [
    ("node_modules/@openai/codex-linux-x64", None,
     "missing node_modules/@openai/codex-linux-x64@0.146.0-linux-x64"),
    ("node_modules/@anthropic-ai/claude-code-linux-x64", "2.1.250",
     "missing node_modules/@anthropic-ai/claude-code-linux-x64@2.1.251"),
    ("node_modules/@anthropic-ai/claude-code-linux-x64", "2.1.250",
     "not checked node_modules/@anthropic-ai/claude-code-linux-x64@2.1.250"),
    ("node_modules/@anthropic-ai/claude-code-darwin-arm64", "2.1.251",
     "not checked node_modules/@anthropic-ai/claude-code-darwin-arm64@2.1.251"),
])
def test_npm_installing_other_than_the_checked_packages_ends_the_build(tmp_path, key, version, shown):
    """npm ls, not the build's filter, says what was installed: a package npm skipped, or one it
    installed that the build did not check, is named."""
    installed = dict(INSTALLED)
    if version is None:
        del installed[key]
    else:
        installed[key] = version
    r, _ = _run_cap_post(tmp_path, installed=installed)
    assert r.returncode == 1 and "reached-the-end" not in r.stdout
    assert "npm's installed packages differ from the checked ones:" in r.stderr and shown in r.stderr


def _with_musl(lock: dict) -> None:
    """Give claude-code its musl build for linux/x64 too, as the published package has it."""
    musl = "@anthropic-ai/claude-code-linux-x64-musl"
    lock["packages"]["node_modules/@anthropic-ai/claude-code"]["optionalDependencies"][musl] = "2.1.251"
    lock["packages"][f"node_modules/{musl}"] = {
        "version": "2.1.251",
        "resolved": f"https://registry.npmjs.org/{musl}/-/claude-code-linux-x64-musl-2.1.251.tgz",
        "integrity": "sha512-" + "F" * 86 + "==", "cpu": ["x64"], "optional": True, "os": ["linux"]}


@pytest.mark.parametrize("wrong, shown", [
    # The filter drops claude's glibc build, npm installs it here, and the musl build keeps the lock's
    # own check passing: offline, npm would skip the glibc build as a failed optional package.
    (lambda real, entry: real(entry) and "claude-code-linux-x64-2" not in entry.get("resolved", ""),
     "npm installs here, but the build does not fetch: node_modules/@anthropic-ai/claude-code-linux-x64"),
    (lambda real, entry: real(entry) or "darwin" in entry.get("os", []),
     "the build fetches, but npm does not install here: node_modules/@anthropic-ai/claude-code-darwin-arm64"),
])
def test_a_platform_filter_that_disagrees_with_npm_ends_the_build_before_fetching(tmp_path, monkeypatch,
                                                                                  wrong, shown):
    real = render_module._installs_here
    monkeypatch.setattr(render_module, "_installs_here", lambda entry: wrong(real, entry))
    r, log = _run_cap_post(tmp_path, change=_with_musl)
    assert r.returncode == 1 and "reached-the-end" not in r.stdout and log == []
    assert shown in r.stderr
    assert not list((tmp_path / "opt-hls").glob("npm-download.*"))


@pytest.mark.parametrize("cli, printed", [("codex", "codex-cli 0.145.0"), ("claude", "2.1.250 (Claude Code)"),
                                         ("claude", "12.1.251 (Claude Code)")])
def test_a_cli_that_prints_another_version_ends_the_build(tmp_path, cli, printed):
    r, _ = _run_cap_post(tmp_path, prints={cli: printed})
    assert r.returncode == 1 and "reached-the-end" not in r.stdout
    assert f"{cli} --version printed" in r.stderr


def test_the_agent_caps_node_comes_from_its_release_tarball_checked_by_digest():
    steps = _steps(render_def(CAP, parent_sif=Path("/node/s-a-tf.sif"), locks=FIXTURES, uv=UV))
    assert node_url(NODE) == "https://nodejs.org/dist/v22.23.3/node-v22.23.3-linux-x64.tar.gz"
    assert f"curl -fsSL --retry 5 -o \"$hls_dl/node.tar.gz\" {node_url(NODE)}" in steps
    assert f'echo "{NODE.sha256}  $hls_dl/node.tar.gz" | sha256sum -c -' in steps
    # Node is unpacked before npm runs, after the apt packages.
    first = {word: next(i for i, s in enumerate(steps) if s.startswith(word))
             for word in ("apt-get install", "curl", "tar", f'PATH="{NODE_PREFIX}/bin:$PATH"')}
    assert list(first.values()) == sorted(first.values())


def test_a_node_download_that_does_not_match_its_digest_is_never_unpacked(tmp_path):
    r, log = _run_cap_post(tmp_path, node_served=_node_tarball() + b"x")
    assert r.returncode != 0 and "reached-the-end" not in r.stdout and log == []
    pinned = tmp_path / "opt-hls" / "node"
    assert not pinned.exists() or list(pinned.iterdir()) == []


def test_a_node_that_is_not_the_pinned_version_ends_the_build(tmp_path):
    r, log = _run_cap_post(tmp_path, node=_node_tarball("v18.19.1"))
    assert r.returncode == 1 and "reached-the-end" not in r.stdout
    assert "node is v18.19.1, not v22.23.3" in r.stderr and log == []


def _run_base_sources(tmp_path: Path, sources: str, others: dict[str, str] | None = None,
                      ) -> tuple[subprocess.CompletedProcess, str]:
    """Run a base's %post up to its first ``apt-get update``, with the apt and image paths
    moved under *tmp_path*, *sources* as Ubuntu's sources file and *others* as further files
    under /etc/apt, by their path below it."""
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post")
    head = post[:post.index("apt-get update")]
    script = "\n".join(head).replace("/etc/apt", str(tmp_path / "etc-apt")).replace(
        "/opt/hls", str(tmp_path / "opt-hls")).replace("/usr/local/share/fonts", str(tmp_path / "fonts"))
    path = tmp_path / "etc-apt" / "sources.list.d" / "ubuntu.sources"
    path.parent.mkdir(parents=True)
    path.write_text(sources)
    for name, text in (others or {}).items():
        (tmp_path / "etc-apt" / name).write_text(text)
    r = subprocess.run(["sh", "-c", script + "\necho reached-the-end\n"], capture_output=True, text=True)
    return r, path.read_text()


def test_the_base_points_apt_at_the_snapshot(tmp_path):
    r, sources = _run_base_sources(tmp_path, "Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\n\n"
                                             "Types: deb\nURIs: http://security.ubuntu.com/ubuntu/\n")
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr
    assert sources.count("URIs: https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/") == 2
    apt_conf = (tmp_path / "etc-apt" / "apt.conf.d" / "99-hls").read_text()
    assert 'APT::Sandbox::User "root";' in apt_conf and 'Acquire::Retries "5";' in apt_conf


def test_a_source_the_rewrite_missed_ends_the_build(tmp_path):
    # An https archive line is not rewritten: the build must stop rather than resolve live.
    r, _ = _run_base_sources(tmp_path, "URIs: https://archive.ubuntu.com/ubuntu/\n\n"
                                       "URIs: http://security.ubuntu.com/ubuntu/\n")
    assert r.returncode == 1 and "reached-the-end" not in r.stdout


UBUNTU = "Types: deb\nURIs: http://archive.ubuntu.com/ubuntu/\nSuites: noble\n"
SNAPSHOT = "https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/"
CUDA_LIST = f"deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] {NVIDIA_REPO} /\n"


def test_the_declared_nvidia_repository_may_stay_beside_the_snapshot(tmp_path):
    r, _ = _run_base_sources(tmp_path, UBUNTU, {"sources.list": "# moved to ubuntu.sources\n",
                                                "sources.list.d/cuda-ubuntu2404-x86_64.list": CUDA_LIST})
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr


def test_a_folded_uris_field_may_name_the_declared_repository(tmp_path):
    # deb822 folds a field onto lines that start with whitespace; a later field's folded lines,
    # such as an inline key, are not URIs.
    folded = (f"Types: deb\nURIs: {SNAPSHOT}\n {NVIDIA_REPO}\nSuites: noble\nSigned-By:\n"
              " -----BEGIN PGP PUBLIC KEY BLOCK-----\n https://example.com/not-a-uri\n"
              " -----END PGP PUBLIC KEY BLOCK-----\n")
    r, _ = _run_base_sources(tmp_path, UBUNTU, {"sources.list.d/folded.sources": folded})
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr


@pytest.mark.parametrize("others", [
    {"sources.list.d/mirror.sources": "Types: deb\nURIs: http://mirror.example.com/ubuntu/\n"},
    {"sources.list.d/extra.list": "deb http://mirror.example.com/ubuntu/ noble main\n"},
    {"sources.list.d/extra.list": "deb [arch=amd64 signed-by=/k.gpg] https://example.com/repo/ noble main\n"},
    {"sources.list": "deb-src http://archive.example.com/ubuntu/ noble main\n"},
    {"sources.list.d/two.sources": f"URIs: {NVIDIA_REPO} https://example.com/repo/\n"},
    {"sources.list.d/folded.sources": f"Types: deb\nURIs: {SNAPSHOT}\n https://example.com/repo/\n"
                                      "Suites: noble\n"},
    {"sources.list.d/folded.sources": f"URIs: {SNAPSHOT}\n# a comment\n\t{NVIDIA_REPO} "
                                      "https://example.com/repo/\n"},
    {"sources.list.d/cuda.list": "deb https://example.com/repos/ubuntu2404/x86_64 /\n"},
])
def test_a_source_neither_the_snapshot_nor_the_declared_repository_ends_the_build(tmp_path, others):
    r, _ = _run_base_sources(tmp_path, UBUNTU, others)
    assert r.returncode == 1 and "reached-the-end" not in r.stdout and "example.com" in r.stderr


@pytest.mark.parametrize("uri", [f"{NVIDIA_REPO}/", NVIDIA_REPO.replace("x86_64", "sbsa"),
                                 NVIDIA_REPO.rstrip("/").replace("x86_64", "x86_6")])
def test_only_one_trailing_slash_is_the_same_repository(tmp_path, uri):
    r, _ = _run_base_sources(tmp_path, UBUNTU, {"sources.list.d/cuda.list": f"deb {uri} /\n"})
    assert r.returncode == 1 and "reached-the-end" not in r.stdout and uri in r.stderr


@pytest.mark.parametrize("line", [
    # The pinned CUDA image's own cuda.list names the repository without its trailing slash.
    f"deb {NVIDIA_REPO.rstrip('/')} /\n",
    f"deb [arch=amd64] {NVIDIA_REPO.rstrip('/')} /\n",
])
def test_the_declared_repository_is_the_same_with_or_without_its_trailing_slash(tmp_path, line):
    r, _ = _run_base_sources(tmp_path, UBUNTU, {"sources.list.d/cuda.list": line})
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr
    folded = f"Types: deb\nURIs: {SNAPSHOT.rstrip('/')}\n {NVIDIA_REPO.rstrip('/')}\nSuites: noble\n"
    r, _ = _run_base_sources(tmp_path / "deb822", UBUNTU, {"sources.list.d/folded.sources": folded})
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr


CUDA_IMAGE_APT = FIXTURES / "cuda-image-apt"


def test_the_pinned_cuda_images_own_apt_sources_pass_once_pointed_at_the_snapshot(tmp_path):
    """Its sources.list, ubuntu.sources and cuda.list, byte for byte as the image's layers hold
    them, through the base's steps up to its first apt-get update."""
    assert (CUDA_IMAGE_APT / "sources.list.d" / "cuda.list").read_bytes() == (
        b"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /\n")
    post = _section(render_def(BASE, parent_sif=None, locks=FIXTURES, uv=UV), "post")
    head = post[:post.index("apt-get update")]
    script = "\n".join(head).replace("/etc/apt", str(tmp_path / "etc-apt")).replace(
        "/opt/hls", str(tmp_path / "opt-hls")).replace("/usr/local/share/fonts", str(tmp_path / "fonts"))
    shutil.copytree(CUDA_IMAGE_APT, tmp_path / "etc-apt")
    r = subprocess.run(["sh", "-c", script + "\necho reached-the-end\n"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "reached-the-end", r.stderr
    sources = (tmp_path / "etc-apt" / "sources.list.d" / "ubuntu.sources").read_text()
    assert sources.count(f"URIs: {SNAPSHOT}\n") == 2 and "archive.ubuntu.com" not in sources


def test_sources_without_the_snapshot_end_the_build(tmp_path):
    r, _ = _run_base_sources(tmp_path, "# no sources\n")
    assert r.returncode == 1 and "reached-the-end" not in r.stdout


# A task's own images: a base with no pip lock, and a layer whose locks and files are its task's.

BARE = BASE.model_copy(update={"name": "cuda", "pip_lock": None, "constraints": []})
SERVER = LayerImage(name="t-baseline", parent="cuda", pip_lock="pip-layer.txt", owner="t", server=True,
                    files={"/app/server.py": "server.py"})


def _task_dir(root: Path) -> Path:
    """A tasks directory under *root* holding task ``t``'s lock and its server file."""
    (root / "t" / "locks").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "pip-layer.txt", root / "t" / "locks" / "pip-layer.txt")
    (root / "t" / "server.py").write_text("print('serving')\n")
    return root


def test_a_base_without_a_pip_lock_makes_the_venv_and_an_empty_constraints_file():
    text = render_def(BARE, parent_sif=None, locks=FIXTURES, uv=UV)
    post = "\n".join(_section(text, "post"))
    assert _section(text, "files") == [f"{FIXTURES}/apt-base.txt {LOCK_ROOT}/apt-base.txt"]
    assert "uv venv --python /usr/bin/python3 /opt/venv" in post
    assert ": > /opt/hls/constraints.txt" in post and "uv pip install" not in post


def test_a_tasks_own_layer_copies_its_locks_and_files_from_its_task(tmp_path):
    tasks = _task_dir(tmp_path / "tasks")
    assert files_for(SERVER, FIXTURES, tasks) == {
        "/app/server.py": tasks / "t" / "server.py", f"{LOCK_ROOT}/pip-layer.txt": tasks / "t" / "locks" / "pip-layer.txt"}
    text = render_def(SERVER, parent_sif=Path("/p/parent.sif"), locks=FIXTURES, uv=UV, tasks_dir=tasks)
    assert _section(text, "files") == [f"{tasks}/t/server.py /app/server.py",
                                       f"{tasks}/t/locks/pip-layer.txt {LOCK_ROOT}/pip-layer.txt"]
    with pytest.raises(FileNotFoundError, match="server.py"):
        render_def(SERVER, parent_sif=Path("/p/parent.sif"), locks=FIXTURES, uv=UV, tasks_dir=tmp_path / "none")


def test_a_tasks_own_layer_is_keyed_by_its_files_bytes_wherever_its_task_is(tmp_path):
    tasks = _task_dir(tmp_path / "tasks")
    elsewhere = _task_dir(tmp_path / "deep" / "tasks")

    def key(where: Path) -> str:
        return recipe_key(SERVER, parent_key="cuda-0123456789abcdef", locks_dir=FIXTURES, uv=UV, tasks_dir=where)

    before = key(tasks)
    assert before == key(elsewhere)
    (tasks / "t" / "server.py").write_text("print('serving faster')\n")
    assert key(tasks) != before
