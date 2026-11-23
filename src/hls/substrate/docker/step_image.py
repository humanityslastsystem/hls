"""The step image: the image every harness role and seal gateway of a Docker host runs in, built
by the harness from its recipe, ``step.Dockerfile`` beside this module, and kept in the store's
cache like every image.

Its cache entry is ``images/step-<16 hex>``, keyed by the recipe's text (``step_key``), which pins
every input by digest or version, so the key changes exactly when the recipe does. The entry
holds the image archive (``images.IMAGE_FILE``), in the form both kinds of image store load, the
recipe (``images.RECIPE_FILE``) and the build's ``BuiltImage`` (``images.BUILT_FILE``).

The image is built before any allocation that runs roles in it is requested
(``ensure_step_image``, which the Docker substrate's scheduler calls before it submits): it needs no
GPU, and a build inside an allocation would spend the allocation's time. The build runs on the host,
in the submitting process, as a build role builds any image (``image_role.build_image``): a
BuildKit builder named after the recipe's key, so the next build removes one a killed build left,
held to ``STEP_BUILD_MEMORY_GIB`` and to the submitting process's CPUs, the ones the operator starts hls
on, which on a shared host narrow it to CPUs of its own. It needs the Docker CLI with
buildx on the host (``docker_cli_problem``), which the planner checks before any allocation is
requested. The cache's lock holds back every other submission that finds the entry missing until
it is published. A job's runner only finds the entry (``step_image_file``) and the launcher loads
its archive, checked against the sha256 the entry records (``launcher.load_step_image``).

The recipe is hls's own, so the privileged builder may build it; nothing a contestant or an agent
wrote is among its inputs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hls.recipes.render import text_key
from hls.store.cache import CacheCorrupt, ContentCache, write_file
from hls.store.digest import sha256_file
from hls.store.lock import Holder
from hls.substrate.docker.image_role import build_image, cpu_list
from hls.substrate.docker.images import BUILT_FILE, IMAGE_FILE, RECIPE_FILE

#: The step image's recipe.
RECIPE = Path(__file__).with_name("step.Dockerfile")
#: The name its cache key starts with.
STEP_NAME = "step"
#: Memory the step image's builder is held to, with no swap: a cold build of the recipe peaks well
#: under it.
STEP_BUILD_MEMORY_GIB = 4


def step_key() -> str:
    """The cache key of the step image built from ``RECIPE``: ``step-<16 hex>`` of its text."""
    return text_key(STEP_NAME, RECIPE.read_text(), {}, None)


def step_image_file(cache: ContentCache) -> tuple[Path, str] | None:
    """The step image's archive in *cache* and the sha256 its entry records; None without an entry.
    Raise CacheCorrupt for an entry that holds no archive."""
    entry = cache.lookup("images", step_key())
    if entry is None:
        return None
    sha256 = next((f.sha256 for f in entry.manifest.files if f.path == IMAGE_FILE), None)
    if sha256 is None:
        raise CacheCorrupt(f"cache entry images/{entry.key} holds no {IMAGE_FILE}")
    return entry.path / IMAGE_FILE, sha256


def docker_cli_problem(docker: str = "docker") -> str | None:
    """What this host lacks to build the step image, in one sentence; None when ``docker`` runs and
    has its buildx plugin."""
    for argv, what in (([docker, "--version"], f"the Docker CLI ({docker})"),
                       ([docker, "buildx", "version"], f"the Docker CLI's buildx plugin ({docker} buildx)")):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"this host has no working {what}, which builds the step image: {e}"
        if done.returncode != 0:
            return f"this host has no working {what}, which builds the step image: it exits {done.returncode}"
    return None


def ensure_step_image(cache: ContentCache, *, socket: Path, buildkit_image: str, work_root: Path,
                      docker: str = "docker") -> tuple[Path, str]:
    """The step image's archive in the store's *cache* and its sha256, built and published first when
    the cache holds no entry for the recipe. The build runs in a directory of its own under
    *work_root*, through the daemon on *socket*. Raise RuntimeError when the build fails, and as
    ``ContentCache.populate`` does."""
    found = step_image_file(cache)
    if found is not None:
        return found
    key = step_key()
    reported: dict[str, str] = {}

    def fill(staging: Path) -> None:
        work = work_root / f"hls-{key}-{os.getpid()}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            recipe = work / RECIPE_FILE
            text = RECIPE.read_text()
            recipe.write_text(text)
            config = work / "docker-config"
            config.mkdir(mode=0o700)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "DOCKER_CONFIG": str(config),
                   "DOCKER_HOST": f"unix://{socket}"}
            archive, digest, record = work / IMAGE_FILE, work / "image.sha256", work / BUILT_FILE
            status = build_image(docker=docker, buildkit_image=buildkit_image, builder=f"hls-{key}", recipe=recipe,
                                 contexts=(), out=archive, digest=digest, record=record, parent=None,
                                 memory=STEP_BUILD_MEMORY_GIB << 30, cpus=cpu_list(sorted(os.sched_getaffinity(0))),
                                 env=env)
            if status != 0:
                raise RuntimeError(f"the step image could not be built from {RECIPE}: the build exited {status}")
            shutil.copyfile(archive, staging / IMAGE_FILE)
            fd = os.open(staging / IMAGE_FILE, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            copied, _ = sha256_file(staging / IMAGE_FILE)
            built = digest.read_text().split()[0]
            if copied != built:
                raise RuntimeError(f"the step image's copy into the cache has sha256 {copied}, not the {built} built")
            reported[IMAGE_FILE] = copied
            reported[RECIPE_FILE] = write_file(staging / RECIPE_FILE, text.encode())
            reported[BUILT_FILE] = write_file(staging / BUILT_FILE, record.read_bytes())
        finally:
            shutil.rmtree(work, ignore_errors=True)

    cache.populate("images", key, fill, holder=Holder.here("step image"), digests=lambda: reported,
                   source={"image": STEP_NAME})
    found = step_image_file(cache)
    assert found is not None
    return found
