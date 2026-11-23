"""The two pip steps of a package that run contestant code, and the pins between them.

The served image is the submission base plus the submission's ``pip install .``, so only what
that install adds needs wheels. ``pip wheel`` alone would resolve as if nothing were installed
and pull torch and the CUDA wheels, gigabytes of them, through the seal for every submission,
only for the offline install to keep the base's copies. So:
1. ``pip install --dry-run --report`` resolves the install against the submission base without
   installing anything. Its report lists what the base lacks or must replace;
2. ``missing_requirements`` turns the report into pins: an index pick as ``name==version``, a
   VCS checkout as ``name @ vcs+url@commit``, any other direct reference as ``name @ url``.
   The project itself is left out;
3. ``pip wheel --no-deps`` builds the project and exactly those pins.

Both pip steps run in sealed contestant containers on the submission base, because the
project's build hooks run there. A base built from the recipes holds torch at its constraint
(``PIP_CONSTRAINT`` of its environment), so a submission that needs another torch fails to
resolve, which is its own fault; a published base holds none. The base must carry ``pip`` for
these steps.

The report is written inside the contestant's container, where a build hook could have
rewritten it. So every pin is checked to be one requirement of the three shapes above, and a
line pip would read as an option (``--index-url``, ``-r``) can never reach the pins file."""

from __future__ import annotations

import re
from pathlib import Path

from hls.spec.resources import GpuKind

#: Seconds the two pip steps may run together before the one running is stopped
#: (``hls.stages.package.PipBudget``, ``contestant_verdict``): four hours. A sealed ``pip wheel``
#: of torch and its CUDA wheels can take most of an hour through a slow index, and both steps
#: must fit with room.
PIP_STEPS_SEC = 4 * 3600
#: The largest installation report of the dry run the package stage reads: far above any real
#: one's, a few hundred KB, and small enough that parsing the worst report a build hook can
#: write costs this process about 400 MB. The worst is arrays of one element nested as deep as
#: ``REPORT_MAX_DEPTH`` allows, about 90 bytes of objects for every 2 bytes of the report (380 MiB
#: on Python 3.12 and 3.14); empty arrays side by side cost about 210 MB. The step's
#: container wrote it, so the size is the submission's to keep.
REPORT_MAX_BYTES = 8 * 2**20
#: The deepest nesting of objects and arrays the dry run's report may hold, checked before it is
#: parsed (``hls.util.load_json``). pip's nests six deep, to an archive's hashes in an item's
#: ``download_info``; the rest is room for a newer pip. Without it the cost of a report within
#: ``REPORT_MAX_BYTES`` would depend on the interpreter: from Python 3.14 the parser recurses as
#: far as the C stack allows, and parses the deepest report the size allows, 4 million levels,
#: in about 960 MB under an unlimited stack.
REPORT_MAX_DEPTH = 32

#: Compute capability of each GPU kind, which becomes TORCH_CUDA_ARCH_LIST for a build that
#: compiles CUDA code; the base images never set it.
COMPUTE_CAPABILITY: dict[str, str] = {"A100": "8.0", "H100": "9.0", "H200": "9.0", "B200": "10.0", "L40": "8.9",
                                      "L40S": "8.9"}
#: Where the build steps keep pip's temporary files and cache, a node-local mount.
BUILD_TMP = "/build-tmp"
_PIP_NET = ["--retries", "5", "--timeout", "30"]
# A distribution name (PEP 508), a version, and a direct-reference URL with no whitespace.
_NAME = r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?"
_INDEX_PIN = re.compile(rf"{_NAME}==[A-Za-z0-9.+!_-]+")
_DIRECT_PIN = re.compile(rf"{_NAME} @ ((git|hg|svn|bzr)\+)?(https?|file|ssh)://[!-~]+")


def dry_run_command(workdir: str, report: str) -> list[str]:
    """Resolve the install of the project at *workdir* into *report*, installing nothing."""
    return ["pip", "install", "--dry-run", "--quiet", "--report", report, workdir, *_PIP_NET]


def wheel_command(workdir: str, pins_file: str, out: str) -> list[str]:
    """Build wheels of the project at *workdir* and of the pins in *pins_file* into *out*,
    without resolving dependencies again."""
    return ["pip", "wheel", "--no-deps", "--wheel-dir", out, workdir, "-r", pins_file, *_PIP_NET]


def check_pin(pin: str) -> str:
    """Return *pin* if it is one requirement of a shape ``missing_requirements`` writes, else
    raise ValueError."""
    if not (_INDEX_PIN.fullmatch(pin) or _DIRECT_PIN.fullmatch(pin)):
        raise ValueError(f"not a pin of one requirement: {pin!r}")
    return pin


def pin_name(pin: str) -> str:
    """The distribution *pin*, one of ``check_pin``'s shapes, names, as written."""
    found = re.match(_NAME, pin)
    if found is None:
        raise ValueError(f"not a pin of one requirement: {pin!r}")
    return found.group()


def missing_requirements(report: dict, project: str) -> list[str]:
    """The pins of what ``pip install <project>`` adds to the image the report was resolved in,
    from pip's installation report (format version 1). The project itself is left out, since the
    wheel step builds it from *project*. Raise ValueError when the report is not pip's shape or a
    pin is not one requirement."""
    project_uri = Path(project).as_uri()
    pins = []
    try:
        items = report["install"]
        if not isinstance(items, list):
            raise TypeError("install is not a list")
        for item in items:
            name, info = item["metadata"]["name"], item["download_info"]
            url, direct = info["url"], item["is_direct"]
            if not isinstance(url, str) or not isinstance(direct, bool):
                raise TypeError("url or is_direct has the wrong type")
            if url.rstrip("/") == project_uri:
                continue
            if not direct:
                pins.append(f"{name}=={item['metadata']['version']}")
            elif "vcs_info" in info:
                vcs = info["vcs_info"]
                pins.append(f"{name} @ {vcs['vcs']}+{url}@{vcs['commit_id']}")
            else:
                pins.append(f"{name} @ {url}")
    except (KeyError, TypeError) as e:
        raise ValueError(f"not a pip installation report: {e!r}") from e
    for pin in pins:
        check_pin(pin)
    return pins


def container_env(*, gpu_kind: GpuKind | None, cpus: int) -> dict[str, str]:
    """The variables of both build containers: pip's temporary files and cache on the node-local
    mount, no prompts, as many compile jobs as the stage has CPUs, and the compute capability of
    the run's GPU for a dependency that compiles CUDA code."""
    env = {"TMPDIR": BUILD_TMP, "PIP_CACHE_DIR": f"{BUILD_TMP}/pip-cache", "PIP_NO_INPUT": "1", "MAX_JOBS": str(cpus)}
    if gpu_kind is not None:
        env["TORCH_CUDA_ARCH_LIST"] = COMPUTE_CAPABILITY[gpu_kind]
    return env
