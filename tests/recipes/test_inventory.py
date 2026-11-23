from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.recipes.inventory import (BRIEF_APT, BRIEF_PACKAGES, BRIEF_TOOLS, INVENTORY_TARGET, SCRIPT,
                                   Inventory, InventoryError, denied_in, missing_apt,
                                   missing_packages, missing_tools, parse_inventory, version_mismatches)

REPO = Path(__file__).resolve().parents[2]
TOOL_LINES = "\n".join(f"tool\t{t}\t{t} 1.0" for t in BRIEF_TOOLS)

# What the script prints in an S-A image, abridged by hand: the toolchain, torch held, an
# odd-cased distribution name, a tool that is absent.
S_A = f"""\
apt\tbuild-essential\t12.10ubuntu1
apt\tpython3-dev\t3.12.3-0ubuntu2
apt\tffmpeg\t7:6.1.1-3ubuntu5
apt\tlibsndfile1\t1.2.2-1ubuntu5
pip\ttorch\t2.11.0+cu130
pip\ttorchaudio\t2.11.0+cu130
pip\ttorchvision\t0.26.0+cu130
pip\ttriton\t3.6.0
pip\tNvidia_Cutlass_DSL\t4.2.1
pip\tcuda-tile\t1.0.0
pip\tcuda-python\t13.0.1
pip\tcuda-bindings\t13.0.1
pip\tcuda-core\t0.3.2
pip\ttransformers\t5.13.1
pip\taccelerate\t1.10.1
pip\tdatasets\t4.1.1
pip\tfastapi\t0.118.0
pip\tuvicorn\t0.37.0
pip\tpydantic\t2.11.9
{TOOL_LINES}
tool\tnode\t
tool\tclaude\t
env\tPATH
env\tPIP_CONSTRAINT
env\tUV_CONSTRAINT
env\tVIRTUAL_ENV
"""

# What it prints in a slim image: no venv, a system python with two distributions, no compilers.
SLIM = """\
apt\tlibc6\t2.39-0ubuntu8.6
apt\tpython3\t3.12.3-0ubuntu2
pip\tpip\t24.0
pip\tsetuptools\t68.1.2
tool\tgcc\t
tool\tpython3\tPython 3.12.3
tool\tnvcc\t
env\tPATH
"""


def test_an_s_a_inventory_parses_with_normalized_names_and_absent_tools():
    inv = parse_inventory(S_A)
    assert inv.pip["nvidia-cutlass-dsl"] == "4.2.1" and "Nvidia_Cutlass_DSL" not in inv.pip
    assert inv.apt["python3-dev"] == "3.12.3-0ubuntu2"
    assert inv.tools["nvcc"] == "nvcc 1.0" and inv.tools["node"] is None
    assert inv.env == ["PATH", "PIP_CONSTRAINT", "UV_CONSTRAINT", "VIRTUAL_ENV"]
    # Round trip through inventory.json, whose bytes do not depend on the script's order.
    assert Inventory.model_validate_json(inv.model_dump_json()) == inv
    assert [list(d) == sorted(d) for d in (inv.apt, inv.pip, inv.tools)] == [True, True, True]
    with pytest.raises(ValidationError, match="frozen"):
        inv.apt = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        Inventory.model_validate({**inv.model_dump(), "kernel": "6.8"})


def test_a_slim_inventory_parses():
    inv = parse_inventory(SLIM)
    assert inv.pip == {"pip": "24.0", "setuptools": "68.1.2"}
    assert inv.tools == {"gcc": None, "nvcc": None, "python3": "Python 3.12.3"}


@pytest.mark.parametrize("line, why", [
    ("deb\tlibc6\t2.39", "unknown record kind"),
    ("apt\tlibc6", "has 3 fields"),
    ("apt\tlibc6\t2.39\textra", "has 3 fields"),
    ("pip\ttorch\t", "needs a name and a version"),
    ("apt\t\t2.39", "needs a name and a version"),
    ("env\tPATH=/usr/bin", "the name of one variable"),
    ("env\tPATH\tx", "the name of one variable"),
    ("pip\tTyping_Extensions\t4.0\npip\ttyping-extensions\t4.1", "appears twice"),
    ("env\tPATH\nenv\tPATH", "appears twice"),
    # A second architecture's copy of a package reads as a repeat, which is refused.
    ("apt\tlibc6\t2.39\napt\tlibc6\t2.39", "appears twice"),
    ("tool\tgcc\t\ntool\tgcc\tgcc 13.3.0", "appears twice"),
    ("error\tthe image has no env", "an error record has 3 fields"),
    ("error\tsometimes\tpython: x", "deterministic or transient"),
    ("error\ttransient\t", "says what failed"),
])
def test_a_malformed_record_is_refused_naming_its_line(line, why):
    with pytest.raises(InventoryError, match=why) as caught:
        parse_inventory("apt\tok\t1\n" + line)
    assert "inventory line 2" in str(caught.value) or "inventory line 3" in str(caught.value)
    # The same output fails the same way again.
    assert caught.value.transient is False


# Error records as the script prints them after SLIM's 8 lines: what failed, and whether a
# second run may pass.
FAILED = ("error\tdeterministic\tpython: /x/a-1.dist-info has no name\n"
          "error\ttransient\tcuda-gdb --version timed out or was killed (exit 124)\n")


def test_strict_parsing_refuses_an_inventory_with_a_failed_probe():
    both = "line 9: python: /x/a-1.dist-info has no name; line 10: cuda-gdb"
    with pytest.raises(InventoryError, match=both) as caught:
        parse_inventory(SLIM + FAILED)
    # One failure that repeats makes a second run pointless.
    assert caught.value.transient is False
    with pytest.raises(InventoryError, match="line 9: cuda-gdb --version timed out") as caught:
        parse_inventory(SLIM + FAILED.split("\n", 1)[1])
    assert caught.value.transient is True


def test_a_denied_package_is_found_by_its_normalized_name():
    assert denied_in(parse_inventory(S_A), DENIED_PACKAGES) == []
    dirty = parse_inventory(S_A + "pip\tFlash_Attn\t2.8.3\npip\tVLLM\t0.22.1\n")
    assert denied_in(dirty, DENIED_PACKAGES) == ["flash-attn", "vllm"]
    # The deny list's own spelling is normalized too.
    assert denied_in(dirty, ["Flash.Attn"]) == ["flash-attn"]


def test_what_the_brief_promises_is_checked_against_the_inventory():
    inv = parse_inventory(S_A)
    assert missing_tools(inv, BRIEF_TOOLS) == []
    assert missing_packages(inv, BRIEF_PACKAGES) == []
    assert missing_apt(inv, BRIEF_APT) == []
    slim = parse_inventory(SLIM)
    assert missing_tools(slim, ["gcc", "python3", "nvcc", "cmake"]) == ["gcc", "nvcc", "cmake"]
    assert missing_packages(slim, ["torch", "SetupTools"]) == ["torch"]
    assert missing_packages(inv, ["Nvidia_Cutlass_DSL", "Cuda.Tile"]) == []
    assert missing_apt(slim, BRIEF_APT) == ["python3-dev"]


def test_torch_is_held_exactly_at_its_constraint():
    held = ["torch==2.11.0+cu130", "torchaudio==2.11.0+cu130", "torchvision==0.26.0+cu130"]
    assert version_mismatches(parse_inventory(S_A), held) == []
    moved = parse_inventory(S_A.replace("torch\t2.11.0+cu130", "torch\t2.13.0+cu130")
                            .replace("pip\ttorchvision\t0.26.0+cu130\n", ""))
    assert version_mismatches(moved, held) == [
        "torch is 2.13.0+cu130; the base holds it at 2.11.0+cu130",
        "torchvision is not installed; the base holds it at 0.26.0+cu130",
    ]
    # A local label is part of the version.
    plain = parse_inventory(S_A.replace("torch\t2.11.0+cu130", "torch\t2.11.0"))
    assert version_mismatches(plain, ["torch==2.11.0+cu130"]) == [
        "torch is 2.11.0; the base holds it at 2.11.0+cu130"]
    with pytest.raises(ValueError, match="not name==version"):
        version_mismatches(plain, ["torch>=2.11"])


def test_a_check_refuses_a_list_that_would_pass_it_vacuously():
    inv = parse_inventory(S_A + "pip\tvllm\t0.22.1\n")
    # A string reads as a list of its characters, and an empty deny list or constraint list
    # passes any image.
    with pytest.raises(TypeError, match="not the string 'vllm'"):
        denied_in(inv, "vllm")
    with pytest.raises(ValueError, match="the deny list is empty"):
        denied_in(inv, ())
    with pytest.raises(TypeError, match="not the string"):
        version_mismatches(inv, "torch==2.11.0+cu130")
    with pytest.raises(ValueError, match="the constraint list is empty"):
        version_mismatches(inv, [])
    for missing in (missing_tools, missing_packages, missing_apt):
        with pytest.raises(TypeError, match="not the string 'gcc'"):
            missing(inv, "gcc")
        # Nothing promised is nothing missing.
        assert missing(inv, []) == []


def test_a_held_name_is_normalized_and_a_version_pep_440_cannot_read_is_compared_as_text():
    inv = parse_inventory("pip\ttorch\t2.11.0+cu130\npip\tlocal-build\tsnapshot-a\n")
    assert version_mismatches(inv, ["Torch==2.11.0+cu130", "Local_Build==snapshot-a"]) == []
    # PEP 440 reads 2.11 as 2.11.0, and a constraint line's blanks are not part of it.
    assert version_mismatches(inv, ["torch==2.11+cu130", " torch==2.11.0+cu130 "]) == []
    assert version_mismatches(inv, ["local-build==snapshot-b"]) == [
        "local-build is snapshot-a; the base holds it at snapshot-b"]


def test_every_promise_is_in_the_brief_and_every_promised_tool_is_looked_for():
    brief = (REPO / "src/hls/prompts/brief.jinja").read_text()
    for name in (*BRIEF_TOOLS, *BRIEF_APT, *BRIEF_PACKAGES):
        # The brief names CUDA Python by its documentation directory and its two modules.
        spelled = {"cuda-python": "references/cuda-python/", "cuda-bindings": "`cuda.bindings`",
                   "cuda-core": "`cuda.core`"}.get(name, f"`{name}`")
        assert spelled in brief, name
    script = SCRIPT.read_text()
    looked_for = re.search(r'TOOLS="([^"]*)"', script).group(1).split()
    assert set(BRIEF_TOOLS) <= set(looked_for)
    assert INVENTORY_TARGET.startswith("/opt/")


# What the brief's toolchain section names in backticks that is not itself installed: parts of
# torch, an example file, and the arch-list variable, which no image sets.
NOT_INSTALLED = {"BuildExtension", "TORCH_CUDA_ARCH_LIST", "cpp_extension", "torch.compile",
                 "torch.utils.cpp_extension", "torch.utils.cpp_extension.load()", "torch/include",
                 "03-matrix-multiplication.py"}
# Modules the brief names, by the distribution that provides each.
PROVIDERS = {"cuda.bindings": "cuda-bindings", "cuda.core": "cuda-core"}


def test_the_brief_lists_hold_exactly_what_the_brief_promises():
    brief = (REPO / "src/hls/prompts/brief.jinja").read_text()
    start = brief.index("{% block toolchain %}")
    toolchain = brief[start:brief.index("{% endblock %}", start)]
    named = set(re.findall(r"`([^`]+)`", toolchain))
    promised = {PROVIDERS.get(n, n) for n in named - NOT_INSTALLED if not n.startswith("references/")}
    # CUDA Python itself is named by its documentation directory.
    assert "references/cuda-python/" in named
    promised.add("cuda-python")
    listed = [*BRIEF_TOOLS, *BRIEF_PACKAGES, *BRIEF_APT]
    assert len(listed) == len(set(listed)) and set(listed) == promised


def _run(path: str, tmp_path: Path, stdin: bytes = b"", **extra_env: str) -> str:
    """The script's output under *path*, decoded the way the images stage decodes it."""
    env = {"PATH": path, "HOME": str(tmp_path), "LANG": "C.UTF-8", **extra_env}
    done = subprocess.run(["/bin/sh", str(SCRIPT)], capture_output=True, input=stdin, env=env, timeout=300,
                          cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    return done.stdout.decode("utf-8", "replace")


def _bare_bin(tmp_path: Path) -> Path:
    """A bin directory holding only what the script itself needs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("tr", "sed", "cut", "sort", "env", "timeout"):
        real = subprocess.run(["sh", "-c", f"command -v {tool}"], capture_output=True, text=True)
        (bin_dir / tool).symlink_to(real.stdout.strip())
    return bin_dir


def _fake(bin_dir: Path, name: str, body: str) -> None:
    tool = bin_dir / name
    tool.write_text("#!/bin/sh\n" + body)
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)


def _dist(where: Path, name: str, version: str) -> None:
    info = where / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def _venv(tmp_path: Path) -> tuple[Path, Path]:
    """A venv standing in for the image's interpreter, and its site-packages."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    return venv, next(venv.glob("lib/python3*/site-packages"))


def test_the_script_runs_here_and_its_output_parses(tmp_path):
    out = _run(os.environ["PATH"], tmp_path, _HLS_PROBE="1")
    # This host's own packages, not the script, decide two outcomes, so they skip rather than fail.
    apt = Counter(line.split("\t")[1] for line in out.split("\n") if line.startswith("apt\t"))
    if repeated := sorted(name for name, count in apt.items() if count > 1):
        pytest.skip(f"this host has packages of a second architecture, which an inventory refuses: "
                    f"{repeated[:5]}")
    try:
        inv = parse_inventory(out)
    except InventoryError as e:
        pytest.skip(f"this host's own inventory has failed probes: {e}")
    looked_for = re.search(r'TOOLS="([^"]*)"', SCRIPT.read_text()).group(1).split()
    assert set(inv.tools) == set(looked_for)
    assert {"HOME", "LANG", "PATH", "_HLS_PROBE"} <= set(inv.env)


def test_the_script_never_fails_on_missing_tools_and_never_writes_a_tab_into_a_field(tmp_path):
    # A PATH holding only what the script itself needs, and one fake tool whose version line
    # carries a tab and a carriage return.
    bin_dir = _bare_bin(tmp_path)
    _fake(bin_dir, "cmake", "printf 'cmake version\\t3.28.3\\r\\nmore\\n'\n")
    inv = parse_inventory(_run(str(bin_dir), tmp_path))
    assert inv.tools["cmake"] == "cmake version 3.28.3 "
    assert inv.tools["gcc"] is None and inv.pip == {} and inv.apt == {}


def test_each_tool_s_version_line_is_picked_by_its_kind(tmp_path):
    bin_dir = _bare_bin(tmp_path)
    # The CUDA binary tools and ncu print a copyright year before the line that names the version.
    _fake(bin_dir, "nvcc", "printf 'nvcc: NVIDIA (R) Cuda compiler driver\\nCopyright (c) 2005-2025 NVIDIA\\n"
                           "Cuda compilation tools, release 13.0, V13.0.88\\n'\n")
    _fake(bin_dir, "ncu", "printf 'NVIDIA (R) Nsight Compute\\nCopyright (c) 2018-2025 NVIDIA\\n"
                          "Version 2025.3.1.0 (build 1)\\n'\n")
    # ffmpeg and tmux answer only their own flags.
    _fake(bin_dir, "ffmpeg", '[ "$1" = -version ] && echo "ffmpeg version 6.1.1"\n')
    _fake(bin_dir, "tmux", '[ "$1" = -V ] && echo "tmux 3.4"\n')
    # One prints its version on stderr, one a line past 200 characters, one no digit at all.
    _fake(bin_dir, "make", 'echo "GNU Make 4.3" >&2\n')
    _fake(bin_dir, "rustc", f"echo 'rustc 1.80.0 {'x' * 300}'\n")
    _fake(bin_dir, "pkg-config", f"echo\necho '{'y' * 300}'\n")
    tools = parse_inventory(_run(str(bin_dir), tmp_path)).tools
    assert tools["nvcc"] == "Cuda compilation tools, release 13.0, V13.0.88"
    assert tools["ncu"] == "Version 2025.3.1.0 (build 1)"
    assert tools["ffmpeg"] == "ffmpeg version 6.1.1" and tools["tmux"] == "tmux 3.4"
    assert tools["make"] == "GNU Make 4.3"
    assert tools["rustc"] == ("rustc 1.80.0 " + "x" * 300)[:200]
    assert tools["pkg-config"] == "y" * 200


def test_a_tool_that_does_not_run_counts_as_missing(tmp_path):
    bin_dir = _bare_bin(tmp_path)
    # A loader error, a nonzero exit after a version line, and two zero exits with no version.
    _fake(bin_dir, "ncu", "echo 'ncu: error while loading shared libraries: libcudart.so.13' >&2\nexit 127\n")
    _fake(bin_dir, "nvcc", "echo 'Cuda compilation tools, release 13.0, V13.0.88'\nexit 3\n")
    _fake(bin_dir, "ninja", "true\n")
    _fake(bin_dir, "cmake", "echo '   '\n")
    _fake(bin_dir, "gcc", "echo 'gcc (Ubuntu 13.3.0-6ubuntu2) 13.3.0'\n")
    inv = parse_inventory(_run(str(bin_dir), tmp_path))
    assert [inv.tools[t] for t in ("ncu", "nvcc", "ninja", "cmake")] == [None] * 4
    assert missing_tools(inv, ["gcc", "ncu", "nvcc", "ninja", "cmake"]) == ["ncu", "nvcc", "ninja", "cmake"]


def test_every_installed_dpkg_package_is_recorded_held_ones_too(tmp_path):
    # NVIDIA's CUDA images hold their cuBLAS and NCCL packages, which dpkg shows as "hi".
    # One marked for removal is still on disk. Removed, never installed and half-unpacked
    # packages are not installed.
    bin_dir = _bare_bin(tmp_path)
    _fake(bin_dir, "dpkg-query", "printf 'ii \\tpython3-dev\\t3.12.3-0ubuntu2\\n"
                                 "hi \\tlibnccl-dev\\t2.27.7-1\\nri \\tleaving\\t1.0\\n"
                                 "rc \\told-config\\t1.0\\nun \\tnever\\t\\niU \\thalf\\t2.0\\n'\n")
    inv = parse_inventory(_run(str(bin_dir), tmp_path))
    assert inv.apt == {"leaving": "1.0", "libnccl-dev": "2.27.7-1", "python3-dev": "3.12.3-0ubuntu2"}


def test_the_python_set_is_the_image_s_own_and_the_first_of_a_name_wins(tmp_path):
    # A .pth adds a directory after the venv's site-packages.
    venv, site = _venv(tmp_path)
    _dist(site, "Dup", "2.0")
    _dist(tmp_path / "later", "dup", "1.0")
    (site / "later.pth").write_text(f"{tmp_path / 'later'}\n")
    # Neither the working directory nor PYTHONPATH belongs to the image.
    _dist(tmp_path, "from-cwd", "1.0")
    _dist(tmp_path / "bound", "from-pythonpath", "1.0")
    bin_dir = _bare_bin(tmp_path)
    inv = parse_inventory(_run(f"{venv / 'bin'}:{bin_dir}", tmp_path, PYTHONPATH=str(tmp_path / "bound")))
    assert inv.pip == {"dup": "2.0"}


def test_each_distribution_s_record_follows_the_pip_rules(tmp_path):
    # Blanks around a name, no version, and a version past 200 characters.
    venv, site = _venv(tmp_path)
    for dirname, fields in [("spaced-1.dist-info", "Name:   Spaced  \nVersion: 1.0\n"),
                            ("nov-1.dist-info", "Name: nov\n"),
                            ("long-1.dist-info", f"Name: long\nVersion: 1.{'0' * 300}\n")]:
        (site / dirname).mkdir()
        (site / dirname / "METADATA").write_text("Metadata-Version: 2.1\n" + fields)
    inv = parse_inventory(_run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path))
    assert inv.pip == {"long": ("1." + "0" * 300)[:200], "nov": "unknown", "spaced": "1.0"}


def _unnamed(where: Path) -> str:
    return f"python: {where} has no name, or its metadata cannot be read"


def test_a_distribution_with_no_name_is_an_error_record(tmp_path):
    # One with metadata but no Name, and one with no metadata file at all.
    venv, site = _venv(tmp_path)
    _dist(site, "torch", "2.11.0+cu130")
    nameless, stale = site / "nameless-1.dist-info", site / "stale-1.dist-info"
    nameless.mkdir()
    (nameless / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 1.0\n")
    stale.mkdir()
    out = _run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path)
    with pytest.raises(InventoryError, match="has no name") as caught:
        parse_inventory(out)
    assert caught.value.transient is False
    assert all(_unnamed(d) in str(caught.value) for d in (nameless, stale))


# Root reads a mode-000 file or directory anyway.
_UNREADABLE = pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file or directory")


@_UNREADABLE
def test_a_distribution_whose_metadata_cannot_be_read_is_an_error_record(tmp_path):
    venv, site = _venv(tmp_path)
    _dist(site, "vllm", "0.22.1")
    metadata = site / "vllm-0.22.1.dist-info" / "METADATA"
    metadata.chmod(0)
    try:
        out = _run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path)
    finally:
        metadata.chmod(0o644)
    with pytest.raises(InventoryError) as caught:
        parse_inventory(out)
    assert caught.value.transient is False
    assert str(caught.value) == f"the script reported failed probes: line 1: {_unnamed(metadata.parent)}"


def test_a_metadata_finder_that_fails_is_an_error_record(tmp_path):
    # A .pth runs code even under -I; this one installs a finder whose listing raises.
    venv, site = _venv(tmp_path)
    (site / "broken_finder.py").write_text(
        "import sys\n\n\nclass Broken:\n    @staticmethod\n    def find_spec(*args, **kwargs):\n"
        "        return None\n\n    @staticmethod\n    def find_distributions(*args, **kwargs):\n"
        "        raise RuntimeError('the finder broke')\n\n\nsys.meta_path.append(Broken)\n")
    (site / "broken.pth").write_text("import broken_finder\n")
    with pytest.raises(InventoryError, match=r"python: RuntimeError\('the finder broke'\)") as caught:
        parse_inventory(_run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path))
    assert caught.value.transient is False


@_UNREADABLE
def test_a_sys_path_directory_that_cannot_be_listed_is_an_error_record(tmp_path):
    # importlib.metadata skips such a directory without a word; this one holds vllm. Its path is
    # long, and the error keeps all of it.
    venv, site = _venv(tmp_path)
    hidden = tmp_path / ("hidden-" + "d" * 150)
    _dist(hidden, "vllm", "0.22.1")
    (site / "hidden.pth").write_text(f"{hidden}\n")
    hidden.chmod(0)
    try:
        out = _run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path)
    finally:
        hidden.chmod(0o755)
    with pytest.raises(InventoryError, match=re.escape(f"python: {hidden} cannot be listed")) as caught:
        parse_inventory(out)
    assert caught.value.transient is False


def test_a_python_error_is_an_error_record_never_a_short_list(tmp_path):
    # Undecodable metadata in a directory sys.path reaches before the one holding vllm.
    venv, site = _venv(tmp_path)
    _dist(site, "torch", "2.11.0+cu130")
    poison, later = tmp_path / "poison", tmp_path / "later"
    (poison / "bad-1.dist-info").mkdir(parents=True)
    (poison / "bad-1.dist-info" / "METADATA").write_bytes(b"Name: \xff\xfe\nVersion: 1\n")
    _dist(later, "vllm", "0.22.1")
    (site / "paths.pth").write_text(f"{poison}\n{later}\n")
    path = f"{venv / 'bin'}:{_bare_bin(tmp_path)}"
    out = _run(path, tmp_path)
    with pytest.raises(InventoryError, match=r"failed probes: line \d+: python: .*UnicodeDecodeError") \
            as caught:
        parse_inventory(out)
    assert caught.value.transient is False
    assert str(caught.value).count("line ") == 1
    # With readable metadata the same image reaches vllm.
    (poison / "bad-1.dist-info" / "METADATA").write_text("Name: bad\nVersion: 1\n")
    assert denied_in(parse_inventory(_run(path, tmp_path)), DENIED_PACKAGES) == ["vllm"]


def test_an_interpreter_that_fails_is_an_error_record(tmp_path):
    # The venv's interpreter cannot load.
    venv, _ = _venv(tmp_path)
    (venv / "bin" / "python3").unlink()
    _fake(venv / "bin", "python3", "echo 'python3: error while loading shared libraries' >&2\nexit 127\n")
    with pytest.raises(InventoryError, match="line 1: the python inventory exited 127$") as caught:
        parse_inventory(_run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path))
    assert caught.value.transient is False


def test_a_dpkg_query_that_fails_is_an_error_record(tmp_path):
    bin_dir = _bare_bin(tmp_path)
    _fake(bin_dir, "dpkg-query", "printf 'ii \\tlibc6\\t2.39\\n'\n"
                                 "echo 'dpkg-query: error: failed to open package info file' >&2\nexit 2\n")
    with pytest.raises(InventoryError, match="line 2: dpkg-query exited 2$") as caught:
        parse_inventory(_run(str(bin_dir), tmp_path))
    assert caught.value.transient is False


def test_a_probe_that_timed_out_or_was_killed_is_a_transient_error_record(tmp_path):
    bin_dir = _bare_bin(tmp_path)
    # The exits timeout gives a probe it ended: 124 after the TERM, 137 after the KILL.
    _fake(bin_dir, "dpkg-query", "exit 124\n")
    _fake(bin_dir, "python3", "kill -KILL $$\n")
    _fake(bin_dir, "cuda-gdb", "kill -KILL $$\n")
    out = _run(str(bin_dir), tmp_path)
    with pytest.raises(InventoryError) as caught:
        parse_inventory(out)
    assert caught.value.transient is True
    assert re.fullmatch(r"the script reported failed probes: " + "; ".join(
        rf"line \d+: {re.escape(what)}" for what in (
            "dpkg-query timed out or was killed (exit 124)",
            "the python inventory timed out or was killed (exit 137)",
            "cuda-gdb --version timed out or was killed (exit 137)",
            "python3 --version timed out or was killed (exit 137)")), str(caught.value))
    # A failure that repeats, beside them, makes the refusal deterministic.
    (bin_dir / "env").unlink()
    with pytest.raises(InventoryError, match="the image has no env") as caught:
        parse_inventory(_run(str(bin_dir), tmp_path))
    assert caught.value.transient is False


@pytest.mark.parametrize("need", ["tr", "sed", "cut", "sort", "env"])
def test_a_missing_command_the_script_needs_is_an_error_record(tmp_path, need):
    bin_dir = _bare_bin(tmp_path)
    (bin_dir / need).unlink()
    with pytest.raises(InventoryError, match=f"failed probes: line 1: the image has no {need}$") as caught:
        parse_inventory(_run(str(bin_dir), tmp_path))
    assert caught.value.transient is False


def test_every_probe_is_bounded_and_no_tool_reads_the_script_s_stdin(tmp_path):
    bin_dir = _bare_bin(tmp_path)
    _fake(bin_dir, "dpkg-query", "printf 'ii \\tlibc6\\t2.39\\n'\n")
    _fake(bin_dir, "python3", "printf 'pip\\tpip\\t24.0\\n'\n")
    # A tool that reports whatever it can read on its stdin.
    _fake(bin_dir, "git", 'read -r line\necho "git version 2.43.0${line:+ read $line}"\n')
    # A timeout that logs how it was called, then runs the command unbounded.
    (bin_dir / "timeout").unlink()
    _fake(bin_dir, "timeout", 'printf "%s\\n" "$*" >> "$TIMEOUT_LOG"\nshift 3\nexec "$@"\n')
    log = tmp_path / "timeout.log"
    inv = parse_inventory(_run(str(bin_dir), tmp_path, stdin=b"leaked\n", TIMEOUT_LOG=str(log)))
    assert (inv.apt, inv.pip, inv.tools["git"]) == ({"libc6": "2.39"}, {"pip": "24.0"}, "git version 2.43.0")
    calls = log.read_text().splitlines()
    assert calls[0].startswith("-k 10 300 dpkg-query -W -f=")
    assert calls[1:] == ["-k 10 300 python3 -I -", "-k 10 60 git --version", "-k 10 60 python3 --version"]
    # Without timeout every probe still runs.
    (bin_dir / "timeout").unlink()
    inv = parse_inventory(_run(str(bin_dir), tmp_path, stdin=b"leaked\n"))
    assert (inv.apt, inv.pip, inv.tools["git"]) == ({"libc6": "2.39"}, {"pip": "24.0"}, "git version 2.43.0")


# Characters str.splitlines breaks a line on, besides the newline.
SEPARATORS = ["\x0b", "\x1c", "\x85", "\u2028"]


def test_no_separator_in_a_distribution_s_name_forges_a_record(tmp_path):
    # Data only: each name carries tabs and a line separator, set out to read as an apt and an
    # env record.
    venv, site = _venv(tmp_path)
    for i, sep in enumerate(SEPARATORS):
        info = site / f"forge{i}-1.dist-info"
        info.mkdir()
        name = f"okpkg{i}\t10{sep}apt\tforged{i}\t1{sep}env"
        (info / "METADATA").write_text(f"Name: {name}\nVersion: FORGED_{i}\n", encoding="utf-8")
    inv = parse_inventory(_run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path))
    assert inv.apt == {} and [e for e in inv.env if e.startswith("FORGED")] == []
    assert inv.pip == {f"okpkg{i} 10 apt forged{i} 1 env": f"FORGED_{i}" for i in range(len(SEPARATORS))}


def test_a_version_s_tabs_and_separators_become_spaces(tmp_path):
    venv, site = _venv(tmp_path)
    for i, sep in enumerate(SEPARATORS):
        info = site / f"sep{i}-1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: sep{i}\nVersion: 1.0{sep}local\tbuild\n", encoding="utf-8")
    inv = parse_inventory(_run(f"{venv / 'bin'}:{_bare_bin(tmp_path)}", tmp_path))
    assert inv.pip == {f"sep{i}": "1.0 local build" for i in range(len(SEPARATORS))}


@pytest.mark.parametrize("sep", ["\r", "\x0c", "\x1d", "\x1e", "\u2029", *SEPARATORS])
def test_the_parser_ends_a_record_only_at_a_newline(sep):
    # str.splitlines would read a second record, FORGED, out of this line.
    with pytest.raises(InventoryError, match="has 3 fields, not 4"):
        parse_inventory(f"pip\tok\t1.0{sep}env\tFORGED\n")
