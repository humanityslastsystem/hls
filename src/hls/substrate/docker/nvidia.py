"""The NVIDIA driver as a Docker role container is given it, with no NVIDIA Container Toolkit.

A container with GPUs is created with the runtime ``runc`` and these, and nothing else of the
driver's:
- devices: each granted GPU's ``/dev/nvidia<minor>`` and the driver's ``CONTROL_DEVICES`` (with
  ``/dev/nvidia-modeset`` where the host has it), which the container may read and write and not
  create; the container's device cgroup admits only those, so a GPU it was not given cannot be
  opened;
- the driver's user-space libraries bound read-only in ``LIB_DIR`` and its programs in ``BIN_DIR``,
  as ``NvidiaDriver`` lists them, the files the toolkit gave a container for the ``compute`` and
  ``utility`` capabilities. Each library is bound under every name ``ldconfig -p`` lists it by and
  its own file name, so the soname the loader asks for (``libcuda.so.1``) is there, and
  ``libcuda.so``, which a compiler links against, as the toolkit's links make it.

The gate (``gate.sh``) puts ``LIB_DIR`` on the role's ``LD_LIBRARY_PATH`` after the image's own
directories, with the image's CUDA forward-compatibility libraries (``COMPAT_DIR``) between them
when its ``libcuda`` is of a newer major version than the driver, the order the toolkit's ld cache
and its ``enable-cuda-compat`` hook gave them; and ``BIN_DIR`` last on the role's ``PATH``.

The scheduler reads the driver's files when it reads the host (``read_driver``) and records them
in the host's facts beside its nvidia-smi (``hls.substrate.local.inventory.HostFacts.driver``). A
library a GPU role cannot run without that ``ldconfig -p`` does not list, or a name the loader
asks for that resolves to another driver's file, refuses the host with one line
(``DriverUnusable``)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

#: The directory a GPU role's container holds the driver's files under.
DRIVER_ROOT = "/usr/local/nvidia"
#: The driver's libraries in a GPU role's container.
LIB_DIR = f"{DRIVER_ROOT}/lib64"
#: The driver's programs in a GPU role's container.
BIN_DIR = f"{DRIVER_ROOT}/bin"
#: The CUDA forward-compatibility libraries of an image, as NVIDIA's CUDA images hold them.
COMPAT_DIR = "/usr/local/cuda/compat"
#: The driver's libraries a GPU role is given, each by the name ``ldconfig -p`` lists it under
#: without a version: the NVIDIA Container Toolkit's for the ``utility`` and ``compute``
#: capabilities. A driver holds the ones of its own version, and a library it lacks is not bound.
LIBRARIES = (
    "libnvidia-ml.so", "libnvidia-cfg.so", "libnvidia-nscq.so",
    "libcuda.so", "libcudadebugger.so", "libnvidia-opencl.so", "libnvidia-gpucomp.so",
    "libnvidia-ptxjitcompiler.so", "libnvidia-fatbinaryloader.so", "libnvidia-allocator.so",
    "libnvidia-compiler.so", "libnvidia-pkcs11.so", "libnvidia-pkcs11-openssl3.so", "libnvidia-nvvm.so",
)
#: The libraries no GPU role runs without, CUDA's driver and NVML, each with the soname its users load.
REQUIRED_LIBRARIES = {"libcuda.so": "libcuda.so.1", "libnvidia-ml.so": "libnvidia-ml.so.1"}
#: The driver's programs a GPU role is given: the toolkit's for the same capabilities.
PROGRAMS = ("nvidia-smi", "nvidia-debugdump", "nvidia-persistenced", "nvidia-cuda-mps-control",
            "nvidia-cuda-mps-server")
#: The program no GPU role is given the driver without.
REQUIRED_PROGRAM = "nvidia-smi"
#: The driver's device nodes, beside its GPUs' own, that every GPU role is given.
CONTROL_DEVICES = ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools")
#: The driver's device node a GPU role is given where the host has it.
MODESET_DEVICE = "nvidia-modeset"
#: The ldconfig that lists the host's libraries.
LDCONFIG = "/sbin/ldconfig"
#: Seconds ``ldconfig -p`` may take; it reads one cache file.
LDCONFIG_TIMEOUT_SEC = 30
#: The ABI ``ldconfig -p`` names a host's own libraries by, after ``libc6``, by machine.
_ABI = {"x86_64": "x86-64", "aarch64": "AArch64"}
_ENTRY = re.compile(r"\s*(\S+) \(([^)]*)\) => (\S.*?)\s*")
_VERSION = re.compile(r"Kernel Module(?: for \S+)?\s+([0-9]+(?:\.[0-9]+)+)\s")


class DriverUnusable(RuntimeError):
    """The host's driver cannot be given to a GPU role as its files are; the message says why in
    one line."""


class DriverFile(BaseModel):
    """One file of the driver's, bound read-only into a GPU role's container."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The host's file, by its real path.
    path: Path
    #: The names the container sees it under, in ``LIB_DIR`` for a library, ``BIN_DIR`` for a program.
    names: tuple[str, ...] = Field(min_length=1)


class NvidiaDriver(BaseModel):
    """The running driver's files that every GPU role is given."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The version of the driver's kernel module, such as "570.211.01".
    version: str
    #: Its user-space libraries.
    libraries: tuple[DriverFile, ...]
    #: Its programs.
    programs: tuple[DriverFile, ...]

    def binds(self) -> list[tuple[Path, str]]:
        """Each file's host path with a path the container sees it at."""
        return ([(f.path, f"{LIB_DIR}/{name}") for f in self.libraries for name in f.names]
                + [(f.path, f"{BIN_DIR}/{name}") for f in self.programs for name in f.names])


def driver_version(proc: Path = Path("/proc")) -> str:
    """The version of the driver's loaded kernel module, from ``<proc>/driver/nvidia/version``.
    Raise DriverUnusable when it cannot be read."""
    path = proc / "driver" / "nvidia" / "version"
    try:
        found = _VERSION.search(path.read_text())
    except OSError as e:
        raise DriverUnusable(f"the NVIDIA driver's version cannot be read from {path}: {e.strerror}") from None
    if found is None:
        raise DriverUnusable(f"{path} names no NVIDIA kernel module version")
    return found[1]


def driver_libraries(listing: str, version: str, *, machine: str) -> tuple[DriverFile, ...]:
    """The libraries of ``LIBRARIES`` that *listing*, the output of ``ldconfig -p``, holds for
    driver *version* on a *machine* host, each the file named ``<library>.<version>`` with the
    names the listing gives it. A name the listing holds with another driver's version is a
    leftover of that driver and is passed over. Raise DriverUnusable when a library of
    ``REQUIRED_LIBRARIES`` or its soname is not listed for *version*, when a name the loader asks for
    (the library's, or its soname's ``<library>.<N>``) resolves to a file that is not the driver's,
    or when the driver's file of a library lies in two places."""
    abi = _ABI.get(machine)
    if abi is None:
        raise DriverUnusable(f"hls knows no ldconfig ABI of a {machine} host")
    entries: list[tuple[str, str]] = []
    for line in listing.splitlines():
        m = _ENTRY.fullmatch(line)
        if m is not None and [f.strip() for f in m[2].split(",")][:2] == ["libc6", abi]:
            entries.append((m[1], m[3]))
    out = []
    for library in LIBRARIES:
        own = f"{library}.{version}"
        found: dict[Path, set[str]] = {}
        for name, path in entries:
            if name != library and not name.startswith(library + "."):
                continue
            try:
                real: Path | None = Path(path).resolve(strict=True)
            except OSError:
                real = None
            if real is not None and real.name == own and real.is_file():
                found.setdefault(real, set()).add(name)
            elif name == library or name[len(library) + 1:].isdecimal():
                where = "a file that does not exist" if real is None else real.name
                raise DriverUnusable(f"{name} on this host is {path}, which resolves to {where}, not the running "
                                     f"NVIDIA driver's {own}")
        soname = REQUIRED_LIBRARIES.get(library)
        if not found:
            if soname is not None:
                raise DriverUnusable(f"ldconfig -p lists no {own}, the running NVIDIA driver's {library}")
            continue
        if len(found) > 1:
            raise DriverUnusable(f"the running NVIDIA driver's {own} lies in two places: "
                                 f"{', '.join(sorted(map(str, found)))}")
        ((real, names),) = found.items()
        if soname is not None and soname not in names:
            raise DriverUnusable(f"ldconfig -p lists no {soname}, the soname of the running NVIDIA driver's {own}")
        names.add(own)
        if library == "libcuda.so":
            names.add(library)
        out.append(DriverFile(path=real, names=tuple(sorted(names))))
    return tuple(out)


def driver_programs(which: Callable[[str], str | None] = shutil.which) -> tuple[DriverFile, ...]:
    """The programs of ``PROGRAMS`` that *which* finds, each by its real path. Raise
    DriverUnusable when it finds no ``REQUIRED_PROGRAM``."""
    out = []
    for name in PROGRAMS:
        found = which(name)
        if found is None:
            if name == REQUIRED_PROGRAM:
                raise DriverUnusable(f"there is no {name} on PATH")
            continue
        out.append(DriverFile(path=Path(found).resolve(), names=(name,)))
    return tuple(out)


def read_driver(*, proc: Path = Path("/proc"), run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                which: Callable[[str], str | None] = shutil.which, machine: str | None = None) -> NvidiaDriver:
    """The running driver's files on this host: its version from *proc*, its libraries from
    ``ldconfig -p`` and its programs from *which*. Raise DriverUnusable, saying why in one line,
    when they cannot be read or a GPU role could not run with them."""
    version = driver_version(proc)
    try:
        p = run([LDCONFIG, "-p"], capture_output=True, text=True, timeout=LDCONFIG_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise DriverUnusable(f"{LDCONFIG} -p could not be run: {e}") from None
    if p.returncode != 0:
        raise DriverUnusable(f"{LDCONFIG} -p exited {p.returncode}")
    return NvidiaDriver(version=version,
                        libraries=driver_libraries(p.stdout, version, machine=machine or os.uname().machine),
                        programs=driver_programs(which))


def driver_devices(minors: list[int], *, dev: Path = Path("/dev")) -> list[tuple[Path, str]]:
    """The device nodes of a container given the GPUs whose nodes have *minors*, each by its path
    under *dev* on the host with its path in the container: the GPUs' own, ``CONTROL_DEVICES`` and,
    where the host has it, ``MODESET_DEVICE``. Raise DriverUnusable when a node of
    ``CONTROL_DEVICES`` or a GPU's is not on the host."""
    names = [f"nvidia{m}" for m in minors] + list(CONTROL_DEVICES)
    out = []
    for name in names:
        if not (dev / name).exists():
            made = " (the host makes it when a CUDA program first runs there, or with nvidia-modprobe -u -c 0)" \
                if name.startswith("nvidia-uvm") else ""
            raise DriverUnusable(f"{dev / name} does not exist{made}")
        out.append((dev / name, f"/dev/{name}"))
    if (dev / MODESET_DEVICE).exists():
        out.append((dev / MODESET_DEVICE, f"/dev/{MODESET_DEVICE}"))
    return out
