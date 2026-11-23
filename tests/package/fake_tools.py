"""Stand-ins for the programs a package runs, on a machine without apptainer or a package
index: ``pip`` (inside the fake container) and ``apptainer``. ``apptainer build --fakeroot``
(the offline build) and ``build --sandbox`` (an extraction) of an image this module made are
faked here; ``exec``, and the extraction of any other file, are
``tests/substrate/fake_apptainer.py``'s. The fake container sees the host's filesystem, so
``exec`` hands it the ``--bind`` map (``FAKE_BINDS``) and the fake pip turns container paths
into host paths with it.

An image this module makes is a *tree image*: ``TREE_MAGIC`` and then a JSON object, path to
entry, for every entry of an image's root (``["dir", mode]``, ``["file", text, mode]``,
``["link", target]``). Extracting it lays exactly that tree out, so the integrity check sees
what a test planted. ``BASE_TREE`` is a small submission base laid out as the Ubuntu 24.04
bases are (``/bin``, ``/lib``, ``/lib64`` and ``/sbin`` link into ``/usr``), with the venv's
``python`` a link into ``/usr/bin``. With ``FAKE_TREE_GIB=<n>`` set, an extraction of a tree
image also writes ``<dir>.size`` beside the tree, a sparse file of *n* GiB, so the node's disk
holds an extracted image's apparent size; outside the tree, the integrity check, which hashes
the trees' files, never reads it.

``install(bin_dir, plan)`` writes both programs into *bin_dir*. Each reads the JSON file *plan*
on every call, so a test changes what they do between steps:
- either step's mode ``hang``: sleep ``hang_sec`` seconds (default 30), as a build hook that
  never ends would, ending at once when the process that started it is killed;
- ``dry_run_hold`` or ``wheel_hold``: a path. ``apptainer exec`` of that step's container waits
  until the path exists before anything of the container runs, so before its start marker, as
  a runtime slow to start a container would, and exits 1 at once when the process that started
  it is gone;
- ``dry_run_started`` or ``wheel_started``: a path that step's pip writes the time it started
  to (``time.time()``), before anything else it does;
- ``dry_run``: ``ok`` (after ``dry_run_sleep`` seconds, default none, write ``report`` to
  ``--report``, with ``@workspace`` replaced by the
  project directory's file URL) or ``fail`` (print ``dry_run_error`` and exit ``dry_run_exit``,
  default 1); ``report_raw`` is text written as the report instead, a lone surrogate standing
  for a byte that is not UTF-8; ``report_pad`` makes the written report that many bytes longer
  (sparse); ``seal_full`` names a directory, the step's seal's, that the dry run leaves unable
  to take a new file or a rename, as a full disk would, before it ends;
  ``report_locked`` takes every permission off it, ``report_dir_locked`` off its
  directory, and ``report_missing`` removes it;
  ``report_link`` (a path), ``report_fifo`` or ``report_dir`` replaces it with a symlink, a
  named pipe or a directory, as a build hook could; ``dry_run_fill_gib`` also writes a sparse
  file of that many GiB into the step's temporary directory, as a build that used its whole
  disk would, and ``dry_run_locked`` a directory there that neither it nor the one inside it
  lets its owner read, write or enter;
- ``wheel``: ``ok`` (a wheel, a real zip, per pin and one for the project in ``--wheel-dir``),
  ``link`` (the same plus a symlink to a host file, ``wheel_link_target`` or ``/etc/passwd``),
  ``dir`` (the same plus a directory named
  like a wheel, holding such a link), ``sdist`` (the same plus a source distribution),
  ``lying`` (the same, with ``lying_wheel()`` as the first pin's wheel: its module unpacks to
  more than it declares) or
  ``fail`` (print ``wheel_error`` and exit ``wheel_exit``, default 1); ``wheel_entries`` adds
  entries (member name to text) to the project's wheel; ``no_project_wheel`` leaves the
  project's wheel out; ``extra_wheels`` names more wheel files to write, each a real wheel of
  the name and version its file name starts with, as a build hook could; ``odd_files`` writes
  that many empty files ``x0``, ``x1`` and on; ``wheels_locked`` takes every
  permission off the project's wheel (``file``) or the wheels directory (``dir``) at the end;
- ``build``: ``ok``, ``fail-install`` (the offline install fails, by the status line the
  ``.def``'s ``%post`` prints) or ``fail-before`` (apptainer fails before ``%post``); whatever
  the mode, a ``%post`` check ``[ -f <path> ] || { echo "<line>" >&2; exit 1; }`` of a file the
  base's tree does not hold prints its line and ends the build before the install, and an
  install whose ``PIP_CONSTRAINT`` names such a file fails as pip does, by its status line;
  ``install_out`` and ``install_err`` are text the install prints on stdout and stderr before
  its status line, as pip echoes a wheel's metadata; any build first reads every file of its
  ``%files`` sources, as their copy does, and fails as apptainer does on one it cannot read; a
  build that succeeds writes a tree image of its base's tree with the ``.def``'s build records
  (the base's ``.def`` moved into the bootstrap history), ``/workspace``, the project in the
  venv and the ``.dist-info`` directory of each wheel it installed, every wheel but those
  ``not_installed`` names, then ``image_changes`` (path to an entry, or None to remove it): what the install
  wrote, as a test plants it; it reports ``installed_bytes`` (default 4096) as what the install
  added to the venv and, when that is a number past the ``.def``'s own check of it
  (``-le <limit> ]``), fails as apptainer does when ``%post`` fails, writing no image;
- ``print``: a line each pip call prints first, for the scrubbing test.
Each call appends what it saw to ``record`` (a JSON-lines file); the dry run records the names
of its environment's variables and the values of those a build that compiles reads."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import struct
import sys
import time
import zipfile
from pathlib import Path

_APPTAINER = Path(__file__).resolve().parents[1] / "substrate" / "fake_apptainer.py"
#: The first line of a tree image.
TREE_MAGIC = b"FAKE-TREE\n"
#: A small submission base.
BASE_TREE: dict[str, list] = {
    "bin": ["link", "usr/bin"], "sbin": ["link", "usr/sbin"], "lib": ["link", "usr/lib"],
    "lib64": ["link", "usr/lib64"], "usr": ["dir", 0o755], "usr/bin": ["dir", 0o755],
    "usr/bin/sh": ["file", "#!dash\n", 0o755], "usr/bin/python3.12": ["file", "python\n", 0o755],
    "usr/bin/python3": ["link", "python3.12"], "usr/sbin": ["dir", 0o755], "usr/lib": ["dir", 0o755],
    "usr/lib/libc.so.6": ["file", "libc\n", 0o755], "usr/lib64": ["dir", 0o755],
    "usr/lib64/ld-linux-x86-64.so.2": ["link", "../lib/libc.so.6"], "etc": ["dir", 0o755],
    "etc/os-release": ["file", "ID=ubuntu\n", 0o644], "etc/passwd": ["file", "root:x:0:0::/root:/bin/sh\n", 0o644],
    ".singularity.d": ["dir", 0o755], ".singularity.d/actions": ["dir", 0o755],
    ".singularity.d/actions/exec": ["file", "#!/bin/sh\nexec \"$@\"\n", 0o755],
    ".singularity.d/env": ["dir", 0o755],
    ".singularity.d/env/90-environment.sh": ["file", "export PATH=/opt/venv/bin:$PATH\n", 0o644],
    ".singularity.d/runscript": ["file", "#!/bin/sh\n", 0o755],
    ".singularity.d/Singularity": ["file", "Bootstrap: docker\n", 0o644],
    ".singularity.d/labels.json": ["file", "{}\n", 0o644],
    "opt": ["dir", 0o755], "opt/venv": ["dir", 0o755], "opt/venv/bin": ["dir", 0o755],
    "opt/venv/bin/python": ["link", "/usr/bin/python3"], "opt/venv/lib": ["dir", 0o755],
    "opt/venv/lib/python3.12": ["dir", 0o755], "opt/venv/lib/python3.12/site-packages": ["dir", 0o755],
    "opt/hls": ["dir", 0o755], "opt/hls/constraints.txt": ["file", "torch==2.11.0+cu130\n", 0o644],
}


def tree_image(tree: dict[str, list]) -> bytes:
    """The bytes of a tree image of *tree*."""
    return TREE_MAGIC + json.dumps(tree, sort_keys=True).encode()


def install(bin_dir: Path, plan: Path) -> None:
    """Write ``pip`` and ``apptainer`` into *bin_dir*, both reading *plan*."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    py, me = shlex.quote(sys.executable), shlex.quote(__file__)
    plan_env = f"FAKE_PACKAGE_PLAN={shlex.quote(str(plan))}"
    (bin_dir / "pip").write_text(f"#!/bin/sh\n{plan_env} exec {py} {me} pip \"$@\"\n")
    (bin_dir / "apptainer").write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = build ]; then shift; {plan_env} exec {py} {me} build \"$@\"; fi\n"
        f"{plan_env} exec {py} {me} exec \"$@\"\n")
    for name in ("pip", "apptainer"):
        (bin_dir / name).chmod(0o755)


def records(path: Path) -> list[dict]:
    """Everything the fakes recorded, in order."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _plan() -> dict:
    return json.loads(Path(os.environ["FAKE_PACKAGE_PLAN"]).read_text())


def _record(plan: dict, entry: dict) -> None:
    with open(plan["record"], "a") as f:
        f.write(json.dumps(entry) + "\n")


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _host(path: str) -> str:
    """The host path a container path *path* is bound from."""
    binds = json.loads(os.environ.get("FAKE_BINDS", "{}"))
    for target in sorted(binds, key=len, reverse=True):
        if path == target or path.startswith(target + "/"):
            return binds[target] + path[len(target):]
    return path


def _pip_step(argv: list[str]) -> str | None:
    """The pip step a container's command line *argv* runs (``dry_run`` or ``wheel``); None for
    any other container."""
    for i, arg in enumerate(argv[:-1]):
        if arg == "pip":
            return {"install": "dry_run", "wheel": "wheel"}.get(argv[i + 1])
    return None


def _held(gate: Path) -> bool:
    """Wait until *gate* exists, as a runtime slow to start its container would; False once the
    process that started this one, or the one that started that, is gone first."""
    parent = os.getppid()
    chain = (parent, _parent_of(parent))
    while not gate.exists():
        if (os.getppid(), _parent_of(parent)) != chain:
            return False
        time.sleep(0.05)
    return True


def _exec(argv: list[str]) -> int:
    """Hand the container its bind map, then run the fake ``apptainer exec``, once the step's
    hold (``<step>_hold``) is released when the plan names one."""
    hold = _plan().get(f"{_pip_step(argv)}_hold")
    if hold is not None and not _held(Path(hold)):
        return 1
    binds = {}
    for i, arg in enumerate(argv[:-1]):
        if arg == "--bind":
            source, target = argv[i + 1].split(":")[:2]
            binds[target] = source
    os.environ["APPTAINERENV_FAKE_BINDS"] = json.dumps(binds)
    os.execv(sys.executable, [sys.executable, str(_APPTAINER), *argv])
    return 0  # not reached


def wheel(name: str, version: str, entries: dict[str, str] | None = None) -> bytes:
    """A wheel of *name* *version*: a module, its dist-info, and *entries* (member to text)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(f"{name}/__init__.py", "")
        z.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        for member, text in (entries or {}).items():
            z.writestr(member, text)
    return out.getvalue()


def declare(data: bytes, member: str, *, size: int | None = None, crc: int | None = None,
            method: int | None = None) -> bytes:
    """The zip *data* with its entry *member* declaring *size* bytes, the CRC *crc* or the
    compression method *method* instead of its own, in its local header and its central record
    alike. The name must appear nowhere else in the zip."""
    out, name = bytearray(data), member.encode()
    # The name follows the entry's local header (30 bytes), then its central record (46).
    local, central = out.index(name) - 30, out.rindex(name) - 46
    if method is not None:
        struct.pack_into("<H", out, local + 8, method)
        struct.pack_into("<H", out, central + 10, method)
    if size is not None:
        struct.pack_into("<L", out, local + 22, size)
        struct.pack_into("<L", out, central + 24, size)
    if crc is not None:
        struct.pack_into("<L", out, local + 14, crc)
        struct.pack_into("<L", out, central + 16, crc)
    return bytes(out)


def lying_wheel() -> bytes:
    """A wheel ``bomb-1.0`` whose module unpacks to 100,000 bytes but declares 100."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("bomb/__init__.py", b"\0" * 100_000)
        z.writestr("bomb-1.0.dist-info/METADATA", "Name: bomb\nVersion: 1.0\n")
    return declare(out.getvalue(), "bomb/__init__.py", size=100)


def _parent_of(pid: int) -> int | None:
    """The parent of process *pid*; None once it is gone."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _hang(seconds: float) -> int:
    """Sleep as a build hook that never ends would, for *seconds* at most, and end at once when
    the process that started this one, or the one that started that, is gone: a killed step
    then leaves nothing running."""
    parent = os.getppid()
    chain, end = (parent, _parent_of(parent)), time.monotonic() + seconds
    while time.monotonic() < end and (os.getppid(), _parent_of(parent)) == chain:
        time.sleep(0.05)
    return 0


def _pip(argv: list[str]) -> int:
    began = time.time()
    plan = _plan()
    step = "dry_run" if argv[:2] == ["install", "--dry-run"] else "wheel"
    if plan.get(f"{step}_started"):
        Path(plan[f"{step}_started"]).write_text(repr(began))
    if plan.get("print"):
        print(plan["print"], file=sys.stderr)
    if plan.get(step) == "hang":
        return _hang(plan.get("hang_sec", 30))
    if argv[:2] == ["install", "--dry-run"]:
        workdir = argv[argv.index("--report") + 2]
        _record(plan, {"tool": "pip", "step": "dry_run", "cwd": os.getcwd(), "env": sorted(os.environ),
                       "build_env": {k: os.environ.get(k) for k in ("MAX_JOBS", "TORCH_CUDA_ARCH_LIST")},
                       "workspace": _host("/workspace")})
        if plan.get("dry_run_fill_gib"):
            with open(Path(_host("/build-tmp")) / "fill.bin", "wb") as f:
                f.truncate(int(plan["dry_run_fill_gib"] * 2**30))
        if plan.get("dry_run_locked"):
            locked = Path(_host("/build-tmp")) / "locked"
            (locked / "inner").mkdir(parents=True)
            (locked / "inner" / "file").write_text("x")
            for path in (locked / "inner", locked):
                path.chmod(0)
        if plan.get("seal_full"):
            # Leave the seal's directory unable to take a new file or a rename, as a full disk
            # would, and give its interceptor time to write a heartbeat.
            os.chmod(plan["seal_full"], 0o500)
            time.sleep(1)
        if plan.get("dry_run") == "fail":
            print(plan.get("dry_run_error", "ERROR: dependency resolution failed"), file=sys.stderr)
            return plan.get("dry_run_exit", 1)
        time.sleep(plan.get("dry_run_sleep", 0))
        report = json.dumps(plan["report"]).replace("@workspace", Path(workdir).as_uri())
        written = Path(_host(_option(argv, "--report")))
        if "report_raw" in plan:
            report = plan["report_raw"]
        written.write_bytes(report.encode(errors="surrogateescape"))
        if plan.get("report_locked"):
            written.chmod(0)
        if plan.get("report_missing"):
            written.unlink()
        if plan.get("report_dir_locked"):
            written.parent.chmod(0)
        if plan.get("report_pad"):
            with open(written, "r+b") as f:
                f.truncate(len(report.encode()) + plan["report_pad"])
        if plan.get("report_link") or plan.get("report_fifo") or plan.get("report_dir"):
            written.unlink()
            if plan.get("report_fifo"):
                os.mkfifo(written)
            elif plan.get("report_dir"):
                written.mkdir()
            else:
                written.symlink_to(plan["report_link"])
        return 0
    if argv[:2] == ["wheel", "--no-deps"]:
        out = Path(_host(_option(argv, "--wheel-dir")))
        pins = Path(_host(_option(argv, "-r"))).read_text().splitlines()
        _record(plan, {"tool": "pip", "step": "wheel", "pins": pins, "workspace": _host("/workspace")})
        if plan.get("wheel") == "fail":
            print(plan.get("wheel_error", "ERROR: Failed building wheel for subm"), file=sys.stderr)
            return plan.get("wheel_exit", 1)
        if not plan.get("no_project_wheel"):
            (out / "subm-0.1-py3-none-any.whl").write_bytes(wheel("subm", "0.1", plan.get("wheel_entries")))
        for extra in plan.get("extra_wheels", []):
            name, version = extra.split("-")[:2]
            (out / extra).write_bytes(wheel(name, version))
        for i in range(plan.get("odd_files", 0)):
            (out / f"x{i}").write_bytes(b"")
        if plan.get("wheel") == "link":
            os.symlink(plan.get("wheel_link_target", "/etc/passwd"), out / "passwd.whl")
        if plan.get("wheel") == "dir":
            (out / "nested.whl").mkdir()
            os.symlink("/etc/passwd", out / "nested.whl" / "passwd")
        if plan.get("wheel") == "sdist":
            (out / "evil-1.0.tar.gz").write_bytes(b"source")
        for pin in pins:
            name, _, version = pin.partition("==")
            (out / f"{name}-{version}-py3-none-any.whl").write_bytes(wheel(name, version))
        if plan.get("wheel") == "lying":
            name, _, version = pins[0].partition("==")
            (out / f"{name}-{version}-py3-none-any.whl").write_bytes(lying_wheel())
        if plan.get("wheels_locked") == "file":
            (out / "subm-0.1-py3-none-any.whl").chmod(0)
        elif plan.get("wheels_locked") == "dir":
            out.chmod(0)
        return 0
    print(f"fake pip: unexpected {argv}", file=sys.stderr)
    return 2


def _section(lines: list[str], name: str) -> list[str]:
    """The lines of the ``.def`` section *name*, without their indent."""
    start = lines.index(name) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("%")), len(lines))
    return [line.strip() for line in lines[start:end] if line.strip()]


def _read_tree(image: Path) -> dict[str, list]:
    data = image.read_bytes()
    return json.loads(data[len(TREE_MAGIC):]) if data.startswith(TREE_MAGIC) else {}


def _read_all(source: Path) -> None:
    """Read every file under *source*, as the build's ``%files`` copy does; raise OSError for a
    directory or a file it cannot read."""
    if source.is_file():
        source.read_bytes()
        return

    def fail(error: OSError) -> None:
        raise error

    for root, _, names in os.walk(source, onerror=fail):
        for name in names:
            (Path(root) / name).read_bytes()


def _build(argv: list[str]) -> int:
    plan = _plan()
    if argv[:1] != ["--fakeroot"] or len(argv) != 3:
        print(f"fake apptainer build: unexpected {argv}", file=sys.stderr)
        return 2
    out, definition = Path(argv[1]), Path(argv[2]).read_text()
    _record(plan, {"tool": "build", "def": definition, "env": {k: os.environ.get(k) for k in
                                                                 ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR")}})
    lines = definition.splitlines()
    parent = lines[1].removeprefix("From: ")
    files = _section(lines, "%files")
    if lines[0] != "Bootstrap: localimage" or not Path(parent).is_file():
        print(f"FATAL:   could not open image {parent}", file=sys.stderr)
        return 255
    if not all(Path(f.split()[0]).exists() for f in files):
        print("FATAL:   %files source missing", file=sys.stderr)
        return 255
    try:
        for f in files:
            _read_all(Path(f.split()[0]))
    except OSError as e:
        print(f"FATAL:   While performing build: copying files from host: {e}", file=sys.stderr)
        return 255
    mode = plan.get("build", "ok")
    if mode == "fail-before":
        print("FATAL:   While performing build: conveyor failed to get: no space left on device", file=sys.stderr)
        return 255
    tree = _read_tree(Path(parent))
    # %post's checks that a file is in the base end the build before the install, as a failed step
    # does, once the check has printed its line.
    for path, said in re.findall(r'^\s*\[ -f (\S+) \] \|\| \{ echo "([^"]*)" >&2; exit 1; \}$', definition,
                                 re.MULTILINE):
        if tree.get(path.lstrip("/"), [""])[0] != "file":
            print(said, file=sys.stderr)
            print("FATAL:   While performing build: while running engine: while running %post section: exit status 1",
                  file=sys.stderr)
            return 255
    # The status lines as the .def's own %post prints them.
    status = re.search(r'echo "(hls-offline-install-rc[^"$]*)\$hls_rc"', definition)
    size = re.search(r'echo "(hls-offline-install-bytes[^"$]*)\$\(\(', definition)
    if status is None or size is None:
        print("fake apptainer build: the .def prints no install status", file=sys.stderr)
        return 2
    if "echo hls-offline-install-begin" in definition:
        print("hls-offline-install-begin", flush=True)
    # What pip printed during the install, on each stream.
    for stream, key in ((sys.stdout, "install_out"), (sys.stderr, "install_err")):
        stream.flush()
        stream.buffer.write(plan.get(key, "").encode())
        stream.buffer.flush()
    if mode == "fail-install":
        print("ERROR: Could not find a version that satisfies the requirement httpx", file=sys.stderr)
        print(f"{status.group(1)}1")
        return 1
    # pip opens the constraints file the install names before it reads a wheel.
    held = re.search(r"PIP_CONSTRAINT=(\S+) ", definition)
    if held is not None and tree.get(held.group(1).lstrip("/"), [""])[0] != "file":
        print(f"ERROR: Could not open constraint file: [Errno 2] No such file or directory: '{held.group(1)}'",
              file=sys.stderr)
        print(f"{status.group(1)}1")
        return 255
    print("Successfully installed subm-0.1")
    print(f"{status.group(1)}0")
    added = plan.get("installed_bytes", 4096)
    print(f"{size.group(1)}{added}")
    limit = re.search(r"-le ([0-9]+) \] \|\| exit", definition)
    if isinstance(added, int) and limit is not None and added > int(limit.group(1)):
        print("FATAL:   While performing build: while running engine: exit status 3", file=sys.stderr)
        return 255
    # As apptainer does, the base's definition moves into the bootstrap history.
    history = ".singularity.d/bootstrap_history"
    kept = sum(1 for p in tree if p.startswith(history + "/"))
    tree.setdefault(history, ["dir", 0o755])
    tree[f"{history}/Singularity{kept}"] = tree[".singularity.d/Singularity"]
    runscript = "#!/bin/sh\n" + "\n".join(_section(lines, "%runscript")) + "\n"
    tree.update({".singularity.d/runscript": ["file", runscript, 0o755],
                 ".singularity.d/Singularity": ["file", definition, 0o644],
                 ".singularity.d/labels.json": ["file", '{"built": "fake"}\n', 0o644],
                 "workspace": ["dir", 0o755], "workspace/server.py": ["file", "print('serving')\n", 0o644],
                 "opt/venv/lib/python3.12/site-packages/subm": ["dir", 0o755],
                 "opt/venv/lib/python3.12/site-packages/subm/__init__.py": ["file", "", 0o644]})
    # pip leaves each wheel it installed its .dist-info directory in the venv.
    wheels = Path(next(f.split()[0] for f in files if f.split()[1] == "/wheels"))
    for found in sorted(wheels.glob("*.whl")):
        if found.name in plan.get("not_installed", []):
            continue
        with zipfile.ZipFile(found) as z:
            tops = {member.split("/")[0] for member in z.namelist()}
        for top in sorted(t for t in tops if t.endswith(".dist-info")):
            tree[f"opt/venv/lib/python3.12/site-packages/{top}"] = ["dir", 0o755]
    for path, entry in plan.get("image_changes", {}).items():
        if entry is None:
            for p in [p for p in tree if p == path or p.startswith(path + "/")]:
                del tree[p]
        else:
            tree[path] = entry
    out.write_bytes(tree_image(tree))
    return 0


def _extract(argv: list[str]) -> int:
    """``build --sandbox <dir> <image>``: lay a tree image out, or hand any other file to the
    fake apptainer's extraction."""
    if (len(argv) != 4 or os.path.lexists(argv[2]) or not Path(argv[3]).is_file()
            or not Path(argv[3]).read_bytes().startswith(TREE_MAGIC)):
        os.execv(sys.executable, [sys.executable, str(_APPTAINER), *argv])
    lay_out(Path(argv[2]), _read_tree(Path(argv[3])))
    tree_gib = os.environ.get("FAKE_TREE_GIB")
    if tree_gib:
        with open(f"{argv[2]}.size", "wb") as f:
            f.truncate(int(float(tree_gib) * 2**30))
    return 0


def lay_out(dest: Path, tree: dict[str, list]) -> None:
    """Make the directory *dest* hold exactly *tree*, as an extraction of its tree image would."""
    dest.mkdir()
    for path in sorted(tree):
        entry, target = tree[path], dest / path
        if entry[0] == "dir":
            target.mkdir(exist_ok=True)
        elif entry[0] == "file":
            target.write_text(entry[1])
            target.chmod(entry[2])
        else:
            os.symlink(entry[1], target)
    for path in sorted((p for p in tree if tree[p][0] == "dir"), reverse=True):
        (dest / path).chmod(tree[path][1])


if __name__ == "__main__":
    tool, rest = sys.argv[1], sys.argv[2:]
    if tool == "build" and rest[:1] == ["--sandbox"]:
        sys.exit(_extract(["build", *rest]))
    sys.exit({"pip": _pip, "build": _build, "exec": _exec}[tool](rest))
