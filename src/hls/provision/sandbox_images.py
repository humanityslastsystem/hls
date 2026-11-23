"""A task's sandbox images: what a task declares, what a site store holds of them, and the role
program that pulls the ones it lacks.

A task whose sandboxes each start from their own image declares the set (``SandboxImageSet``):
the site store the images live in, the registry repository they are published in, each image's
name and the digest of its published manifest, and the fixed allowances of the stage that pulls
them. An image is never built here. It is the published image, named by digest, converted to one
Apptainer file::

    apptainer pull <name>.sif docker://<repository>@sha256:<digest>

The store holds ``<name>.sif`` and, beside it, the record ``<name>.json`` (``ImageRecord``): the
source the file was pulled from, with its digest, the file's sha256 and size, and the Apptainer
that converted it. The store holds an image (``held``) when its record names the declared source
and its file has the recorded size. A file with no record, or with a record of another source,
is not held: nothing says what it is. The plan of ``hls provision`` and the plan-time check of a
drive both ask ``missing_images``, so neither can pass a store the other refuses.

The role program, run by the ``sandbox_images`` stage as one host-side harness role::

    python -m hls.provision.sandbox_images --images FILE --store DIR --work DIR --report FILE

*FILE* holds the set as JSON. Images the store holds are left alone. The rest are pulled, at most
``SandboxImageSet.parallel`` at a time, each in its own directory under *DIR* (``--work``, on
node-local disk) with its own layer cache and unpacking directory, so two pulls share no state.
A finished file is hashed, copied into the store under a temporary name, synced and renamed, and
its record is written last: a file is in the store under its name only whole, and has a record
only once it is. A pull that fails, or does not end within ``pull_timeout_sec``, leaves nothing in
the store. A SIGTERM ends the pulls in flight, each then one that failed. Every image's outcome
goes to ``--report`` (``Report``). When any failed, the role ends with ``FAILED_EXIT`` and a last
line on stderr, ``failed: <name>: <why>; ...``.

A registry login is the images stage's: the user name in ``REGISTRY_USER_VAR`` and the token in
``REGISTRY_TOKEN_GRANT``, both or neither, read from the role's environment and handed to
apptainer under its own names in the pull's environment alone. No command line holds either."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from hls.settings import SandboxStoreName
from hls.store.base import check_segment
from hls.store.digest import CHUNK, Sha256Hex, sha256_file
from hls.substrate.apptainer.images import REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR, build_env
from hls.util import remove_tree

#: Exit status of the role when a sandbox image could not be pulled. Its last line on stderr is
#: then ``failed: <name>: <why>; ...``, and its report names every image's outcome.
FAILED_EXIT = 5
#: The most characters of a failed pull's reason the role reports.
WHY_MAX = 300
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REPOSITORY = re.compile(r"[a-z0-9][a-z0-9.-]*(?::[0-9]+)?(?:/[a-z0-9][a-z0-9._-]*)+")
_print_lock = threading.Lock()


def _digest(value: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"not a manifest digest (sha256:<64 hex>): {value!r}")
    return value


def _repository(value: str) -> str:
    if not _REPOSITORY.fullmatch(value):
        raise ValueError(f"not a registry repository (<host>/<path>): {value!r}")
    return value


class SandboxImage(BaseModel):
    """One sandbox image: its name in the store and the published manifest it is pulled from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The image's name: the store holds it as ``<name>.sif``.
    name: Annotated[str, AfterValidator(check_segment)]
    #: The digest of the published image's manifest, ``sha256:<64 hex>``.
    digest: Annotated[str, AfterValidator(_digest)]


class SandboxImageSet(BaseModel):
    """The sandbox images a task declares and the fixed allowances of the stage that pulls them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The site store (``Settings.sandbox_stores``) the images live in.
    store: SandboxStoreName
    #: The registry repository every image is published in, host first.
    repository: Annotated[str, AfterValidator(_repository)]
    #: The images, each name once.
    images: tuple[SandboxImage, ...]
    #: CPUs of the role that pulls: the conversions pack with every core they are given.
    cpus: int = Field(ge=1)
    #: Memory of that role, in GiB.
    memory_gib: int = Field(ge=1)
    #: Node-local disk the pulls in flight may hold, in GiB: each one's layers, its unpacked tree
    #: and its image file.
    disk_gib: int = Field(ge=1)
    #: Seconds the stage may take.
    time_sec: int = Field(gt=0)
    #: Pulls in flight at once.
    parallel: int = Field(ge=1)
    #: Seconds one pull may take before it is ended and reported as failed.
    pull_timeout_sec: int = Field(gt=0)

    @model_validator(mode="after")
    def _each_name_once(self) -> SandboxImageSet:
        names = [i.name for i in self.images]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise ValueError(f"sandbox images named twice: {', '.join(twice)}")
        if not names:
            raise ValueError("a sandbox image set names at least one image")
        return self

    def source(self, image: SandboxImage) -> str:
        """What *image* is pulled from: its published manifest, by digest."""
        return f"docker://{self.repository}@{image.digest}"


class ImageRecord(BaseModel):
    """The record beside a sandbox image's file in its store, ``<name>.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The image's name.
    name: Annotated[str, AfterValidator(check_segment)]
    #: What the file was pulled from (``SandboxImageSet.source``): the repository and the digest
    #: of the published manifest.
    source: str
    #: The sha256 of the image file.
    sha256: Sha256Hex
    #: The image file's size in bytes.
    size: int = Field(ge=0)
    #: The Apptainer that converted it, as ``apptainer --version`` prints it.
    apptainer: str


class Outcome(BaseModel):
    """What the role did about one image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The image's name.
    name: str
    #: ``found`` when the store held it, ``pulled`` when the role put it there, ``failed`` when
    #: it could not.
    status: Literal["found", "pulled", "failed"]
    #: The image's record in the store; None for one that failed.
    record: ImageRecord | None = None
    #: Why it failed, on one line; None otherwise.
    why: str | None = None
    #: Seconds the pull, the hash and the copy into the store took; 0 for one the store held.
    seconds: float = 0.0


class Report(BaseModel):
    """The role's report: every image's outcome, in the set's order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcomes: list[Outcome]

    def failed(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "failed"]


def failure_line(failed: Collection[Outcome]) -> str:
    """The one line that names each image of *failed* and why it failed."""
    return "; ".join(f"{o.name}: {o.why}" for o in failed)


def image_file(directory: Path, name: str) -> Path:
    """Where the store at *directory* holds image *name*'s file."""
    return directory / f"{check_segment(name)}.sif"


def record_file(directory: Path, name: str) -> Path:
    """Where the store at *directory* holds image *name*'s record."""
    return directory / f"{check_segment(name)}.json"


def read_record(directory: Path, name: str) -> ImageRecord | None:
    """Image *name*'s record in the store at *directory*; None when there is none that can be
    read as one."""
    try:
        return ImageRecord.model_validate_json(record_file(directory, name).read_bytes())
    except (OSError, ValidationError):
        return None


def held(directory: Path, name: str, source: str) -> ImageRecord | None:
    """The record of image *name* when the store at *directory* holds it as pulled from *source*:
    the record names the image and that source, and the image's file is a regular file of the
    recorded size. None otherwise. The file is not hashed: its sha256 is in the record for whoever
    checks one."""
    record = read_record(directory, name)
    if record is None or record.name != name or record.source != source:
        return None
    path = image_file(directory, name)
    try:
        found = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(found.st_mode) or found.st_size != record.size:
        return None
    return record


def missing_images(directory: Path, images: SandboxImageSet, names: Collection[str] | None = None) -> list[str]:
    """The images of *images* the store at *directory* does not hold (``held``), by name, in the
    set's order; with *names*, only those among them."""
    return [i.name for i in images.images
            if (names is None or i.name in names) and held(directory, i.name, images.source(i)) is None]


def pull_argv(out: Path, source: str, *, apptainer: str = "apptainer") -> list[str]:
    """The command that pulls the published image *source* and converts it to the image file
    *out*."""
    if not out.is_absolute():
        raise ValueError(f"the image file {out} must be an absolute path")
    return [apptainer, "pull", str(out), source]


def pull_env(work: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a pull that works in *work*: *base* with apptainer's layer cache and
    unpacking directory inside *work*, and the registry login, when *base* holds both halves,
    under apptainer's own names in place of the role's."""
    env = {k: v for k, v in base.items() if k not in (REGISTRY_USER_VAR, REGISTRY_TOKEN_GRANT)}
    env.update(build_env(cache_dir=work / "cache", tmp_dir=work / "tmp"))
    user, token = base.get(REGISTRY_USER_VAR), base.get(REGISTRY_TOKEN_GRANT)
    if user and token:
        env.update({"APPTAINER_DOCKER_USERNAME": user, "APPTAINER_DOCKER_PASSWORD": token})
    return env


def _one_line(text: str) -> str:
    return " ".join(text.split())[:WHY_MAX]


def _why(log: str, code: int) -> str:
    """Why a pull that exited *code* failed, from its output *log*: apptainer's last FATAL line,
    else its last line."""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    fatal = [line for line in lines if line.startswith("FATAL")]
    last = (fatal or lines or ["no output"])[-1]
    return _one_line(f"apptainer pull exited {code}: {last}")


def apptainer_version(apptainer: str) -> str:
    """What ``apptainer --version`` prints, on one line. Raise RuntimeError when it cannot run."""
    try:
        done = subprocess.run([apptainer, "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"{apptainer} --version could not run: {e}") from e
    if done.returncode != 0:
        raise RuntimeError(f"{apptainer} --version exited {done.returncode}: {_one_line(done.stderr)}")
    return _one_line(done.stdout)


def _copy_synced(source: Path, dest: Path) -> None:
    """Copy *source* to the new file *dest* and sync it."""
    with open(source, "rb") as src, open(dest, "xb") as out:
        while chunk := src.read(CHUNK):
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())


def _sync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(directory: Path, pulled: Path, record: ImageRecord) -> None:
    """Put the image file *pulled* into the store at *directory* under its name, then its
    *record*. The record of an earlier file goes first, so a record never describes another file;
    the file is copied under a temporary name, synced and renamed, so its name never holds a part
    of one; and the record is written last, the same way."""
    name = record.name
    record_path, image_path = record_file(directory, name), image_file(directory, name)
    record_path.unlink(missing_ok=True)
    partial = directory / f".{name}.sif.{os.getpid()}.partial"
    partial.unlink(missing_ok=True)
    try:
        _copy_synced(pulled, partial)
        os.replace(partial, image_path)
    finally:
        partial.unlink(missing_ok=True)
    partial_record = directory / f".{name}.json.{os.getpid()}.partial"
    partial_record.unlink(missing_ok=True)
    try:
        with open(partial_record, "xb") as f:
            f.write(record.model_dump_json(indent=1).encode() + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial_record, record_path)
    finally:
        partial_record.unlink(missing_ok=True)
    _sync_dir(directory)


class _Puller:
    """The pulls of one role."""

    def __init__(self, images: SandboxImageSet, store: Path, work: Path, *, apptainer: str,
                 env: Mapping[str, str]) -> None:
        self.images = images
        self.store = store
        self.work = work
        self.apptainer = apptainer
        self.env = env
        self.version = ""
        self._lock = threading.Lock()
        self._live: set[subprocess.Popen] = set()
        self._stopping = False

    def stop(self) -> None:
        """End every pull in flight and start no other: each is then reported as failed."""
        with self._lock:
            self._stopping = True
            live = list(self._live)
        for process in live:
            _signal_group(process)

    def one(self, image: SandboxImage) -> Outcome:
        """Leave *image* in the store, pulled unless it is held; never raises."""
        source = self.images.source(image)
        try:
            record = held(self.store, image.name, source)
            if record is not None:
                return Outcome(name=image.name, status="found", record=record)
            started = time.monotonic()
            record = self._pull(image, source)
            return Outcome(name=image.name, status="pulled", record=record,
                           seconds=round(time.monotonic() - started, 1))
        except _Failed as e:
            return Outcome(name=image.name, status="failed", why=str(e))
        except Exception as e:
            return Outcome(name=image.name, status="failed", why=_one_line(f"{type(e).__name__}: {e}"))

    def _pull(self, image: SandboxImage, source: str) -> ImageRecord:
        work = self.work / image.name
        remove_tree(work)
        (work / "tmp").mkdir(parents=True)
        (work / "cache").mkdir()
        out = work / f"{image.name}.sif"
        log_path = work / "pull.log"
        argv = pull_argv(out, source, apptainer=self.apptainer)
        try:
            with open(log_path, "wb") as log:
                with self._lock:
                    if self._stopping:
                        raise _Failed("the role was told to stop before the pull started")
                    # In its own session, so that the pull and everything it starts end together.
                    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               env=pull_env(work, self.env), start_new_session=True)
                    self._live.add(process)
                try:
                    code = process.wait(timeout=self.images.pull_timeout_sec)
                except subprocess.TimeoutExpired:
                    _signal_group(process)
                    process.wait()
                    self._show(image.name, log_path, "did not end")
                    raise _Failed(f"apptainer pull did not end in {self.images.pull_timeout_sec} s") from None
                finally:
                    with self._lock:
                        self._live.discard(process)
            text = self._show(image.name, log_path, f"exited {code}")
            if self._stopping:
                raise _Failed("the role was told to stop while the pull ran")
            if code != 0:
                raise _Failed(_why(text, code))
            if not out.is_file():
                raise _Failed("apptainer pull exited 0 and left no image file")
            sha256, size = sha256_file(out)
            record = ImageRecord(name=image.name, source=source, sha256=sha256, size=size, apptainer=self.version)
            publish(self.store, out, record)
            return record
        finally:
            remove_tree(work, ignore_errors=True)

    @staticmethod
    def _show(name: str, log_path: Path, ending: str) -> str:
        """Print the pull's output to the role's stderr under one header, whole, and return it."""
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        with _print_lock:
            print(f"== {name}: apptainer pull {ending}\n{text}", file=sys.stderr, flush=True)
        return text


class _Failed(Exception):
    """A pull failed; the message says why on one line."""


def _signal_group(process: subprocess.Popen) -> None:
    """Kill *process* and everything it started, which runs in its own session."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def pull_missing(images: SandboxImageSet, store: Path, work: Path, *, apptainer: str = "apptainer",
                 env: Mapping[str, str] | None = None, on_sigterm: bool = False) -> Report:
    """Leave every image of *images* in the store at *store*: those it holds as they are, the
    rest pulled, ``images.parallel`` at a time, each working in its own directory under *work*.
    One image's failure stops no other. With *on_sigterm*, a SIGTERM ends the pulls in flight and
    starts no other, and each of those is reported as failed. Raise RuntimeError when apptainer
    cannot run at all."""
    store, work = store.absolute(), work.absolute()
    store.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    puller = _Puller(images, store, work, apptainer=apptainer, env=os.environ if env is None else env)
    if missing_images(store, images):
        puller.version = apptainer_version(apptainer)
    before = signal.signal(signal.SIGTERM, lambda signum, frame: puller.stop()) if on_sigterm else None
    try:
        with ThreadPoolExecutor(max_workers=images.parallel) as pool:
            return Report(outcomes=list(pool.map(puller.one, images.images)))
    finally:
        if on_sigterm:
            signal.signal(signal.SIGTERM, before)


def write_report(path: Path, report: Report) -> None:
    """Write *report* to *path* whole: under a temporary name, then renamed."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(report.model_dump_json(indent=1))
    os.replace(partial, path)


def read_report(path: Path) -> Report | None:
    """The report a role wrote to *path*; None when there is none that can be read as one."""
    try:
        return Report.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.provision.sandbox_images")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--apptainer", default="apptainer")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    images = SandboxImageSet.model_validate(json.loads(args.images.read_text()))
    report = pull_missing(images, args.store, args.work, apptainer=args.apptainer, on_sigterm=True)
    write_report(args.report, report)
    failed = report.failed()
    if failed:
        print(f"failed: {failure_line(failed)}", file=sys.stderr)
        return FAILED_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
