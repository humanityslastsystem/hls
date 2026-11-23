"""A frozen conversation a trajectory resumes from, and the file it lives in.

A drive window is too short for a trajectory that starts at turn 0 to reach the deep
bands, so the corpus has to start some trajectories deep. A prefix is what a trajectory
starts from: the conversation an offline pre-roll built by driving the baseline from
turn 0, frozen at the turn its prompt crossed a depth target.

A prefix is half of a warm start. The other half is the filesystem the pre-roll left
behind -- fifty commands in, the model's next command refers to files it created and
builds it ran, and a fresh sandbox has none of them.

That half is a directory: the overlay upper layer the pre-roll's own container wrote,
staged beside this file. A resuming session mounts it read-only underneath its own
fresh writable layer, so it sees what the pre-roll left, deletions and replaced
directories included, and writes nothing back to it. Several sessions resuming the
same prefix therefore share one directory safely.

The task's prefixes and their state directories are in the repository, under
``prefixes/`` beside this file, and provisioning copies them into the prefix store
(``place``). ``state_manifest`` records what the directory held, so a prefix cannot be
addressed by its conversation alone, and a copy is placed only when it holds exactly
that. The overlay driver encodes a deletion as a character device on one filesystem and
as a ``.wh.`` file on another, and marks a replaced directory with an extended
attribute. The repository holds each state as one archive, the one ``to_repository``
writes, with each whiteout as an empty file of its name and no extended attributes, and
placement makes both again on the copy from the manifest. What keeps the encoding stable
across runs is the apptainer version pinned in ``task.py``.

No ``hls`` imports, for the reason ``mini_swe_loop`` has none: the same file is read by
the harness driver during a graded run, by the offline pre-roll that writes it, and
inside a build sandbox that has no harness to import from.

Why every field is here
-----------------------

The conversation alone does not resume a loop. A trajectory carries counters that
decide when it ends -- how many commands it has run, and how many replies in a row it
has failed to parse -- and a resumed trajectory that started them at zero would get a
fresh repair budget that a continuous one would not, which changes when the trajectory
stops and so changes the traffic. They are stored with the messages.

The digest is what the corpus declares, so a re-cut prefix cannot outlive a
calibration. It covers the checkpoint as well as the conversation: a prefix is that
checkpoint's own writing, and the same conversation in front of different weights is
off-distribution rather than merely stale.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: Filename extension for a stored prefix. One file per (task, depth).
PREFIX_SUFFIX = ".prefix.json"

#: Keys a stored prefix file carries, beside ``digest`` which is computed over them.
_FIELDS = ("task_name", "depth_tokens", "prefill_tokens", "turns", "commands_run",
           "consecutive_format_errors", "checkpoint", "image", "state_manifest",
           "messages")


class PrefixError(RuntimeError):
    """A prefix file is missing, malformed, or not the one that was declared."""


def digest_of(payload: dict) -> str:
    """The digest a prefix is addressed by: sha256 of its canonical JSON.

    Canonical, because the fingerprint that carries this is canonical JSON too and a
    prefix spelled two ways would otherwise address two ways. ``digest`` itself is
    excluded, since it cannot cover itself."""
    body = {k: payload[k] for k in _FIELDS}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Prefix:
    """One frozen conversation, with the loop state that resumes it."""

    #: The corpus task this prefix was cut from. Checked against the session's own
    #: task, so a mis-staged file cannot put one task's conversation in another's
    #: container.
    task_name: str
    #: The depth target this prefix was cut at, in prompt tokens. The band's name,
    #: not a measurement.
    depth_tokens: int
    #: What the prompt actually rendered to at the cut, counted harness-side through
    #: the checkpoint's tokenizer. The measurement, and what a loader checks the band
    #: against rather than trusting the name.
    prefill_tokens: int
    #: Turns the pre-roll served to reach here. Recorded so the diagnostics can tell
    #: a session's own turns from the depth it started at; the resumed loop counts its
    #: own turns from zero, because its turn budget is what it serves, not where it
    #: began.
    turns: int
    #: Commands the pre-roll's trajectory ran. Carried forward, so the shape
    #: diagnostics read one trajectory rather than a session that appeared mid-work
    #: having run nothing.
    commands_run: int
    #: Unparseable replies in a row at the cut. Carried forward, because it is a
    #: stopping rule: a prefix cut just after two format errors has one repair left,
    #: and a resumed loop that started at zero would take three more.
    consecutive_format_errors: int
    #: The checkpoint whose writing this conversation is.
    checkpoint: str
    #: The sandbox image the pre-roll ran, and a resuming session starts from. The
    #: base image, unchanged: the pre-roll's own writes are in the state directory
    #: beside this file rather than baked into a derivative.
    image: str
    #: What the pre-roll's overlay upper held, as ``path -> size`` over every entry,
    #: with a marker file or a replaced directory recorded as size -1. The state
    #: directory's contents, so the digest covers the filesystem a session resumes
    #: into and not only the conversation. Empty for a prefix with no state, which is
    #: a test fixture rather than anything a pass writes.
    state_manifest: dict
    #: The conversation exactly as it goes on the wire -- what ``wire_messages``
    #: emits, so an assistant turn carries its ``reasoning_content`` beside its
    #: ``content``. Byte-identical to what the artifact is sent and to what the
    #: oracle teacher-forces.
    messages: tuple[dict, ...]
    #: sha256 over everything above.
    digest: str

    def payload(self) -> dict:
        """This prefix as its file holds it."""
        return {"task_name": self.task_name, "depth_tokens": self.depth_tokens,
                "prefill_tokens": self.prefill_tokens, "turns": self.turns,
                "commands_run": self.commands_run,
                "consecutive_format_errors": self.consecutive_format_errors,
                "checkpoint": self.checkpoint, "image": self.image,
                "state_manifest": dict(self.state_manifest),
                "messages": [dict(m) for m in self.messages],
                "digest": self.digest}

    def check_opening(self, opening: list[dict]) -> None:
        """Refuse a prefix whose first messages are not what the templates render now.

        The system and instance messages are rendered from templates at turn 0, and a
        resumed trajectory installs the stored conversation instead of rendering them
        again. So an edit to either template would silently serve every resumed
        session a prompt whose opening disagrees with the one the pre-roll built, and
        with the one a cold session is served in the same drive. Nothing downstream
        would notice: the gate reads the served distribution, and both prompts are
        well-formed.

        Compared on role and content, which is all a rendered opening has."""
        stored = list(self.messages[:len(opening)])
        for i, (want, got) in enumerate(zip(opening, stored)):
            if want.get("role") == got.get("role") and \
                    want.get("content") == got.get("content"):
                continue
            raise PrefixError(
                f"prefix {self.task_name}@{self.depth_tokens} opens with a message "
                f"{i} the templates no longer render: stored {got.get('role')!r} "
                f"message of {len(str(got.get('content') or ''))} characters, "
                f"rendered {want.get('role')!r} message of "
                f"{len(str(want.get('content') or ''))}. The prefix was cut under "
                f"different templates, so re-cut it or restore them; a resumed "
                f"session would otherwise be served a different opening from a cold "
                f"one in the same drive.")


def from_payload(payload: dict) -> Prefix:
    """A ``Prefix`` from a file's parsed contents, refusing one that does not add up.

    Every check is on something a mis-staged or hand-edited file gets wrong, and each
    would otherwise surface as a trajectory that ran against the wrong container or
    the wrong weights rather than as a failed load."""
    missing = [k for k in (*_FIELDS, "digest") if k not in payload]
    if missing:
        raise PrefixError(f"prefix file is missing {missing}")
    found = digest_of(payload)
    if found != payload["digest"]:
        raise PrefixError(
            f"prefix {payload['task_name']!r} carries digest "
            f"{str(payload['digest'])[:12]}... but its contents hash to "
            f"{found[:12]}...; the file was edited after it was cut")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise PrefixError(
            f"prefix {payload['task_name']!r} holds {len(messages or [])} messages; a "
            f"conversation starts with a system and an instance message at least")
    return Prefix(task_name=str(payload["task_name"]),
                  depth_tokens=int(payload["depth_tokens"]),
                  prefill_tokens=int(payload["prefill_tokens"]),
                  turns=int(payload["turns"]),
                  commands_run=int(payload["commands_run"]),
                  consecutive_format_errors=int(payload["consecutive_format_errors"]),
                  checkpoint=str(payload["checkpoint"]),
                  image=str(payload["image"]),
                  state_manifest=dict(payload["state_manifest"]),
                  messages=tuple(dict(m) for m in messages),
                  digest=str(payload["digest"]))


def build(*, task_name: str, depth_tokens: int, prefill_tokens: int, turns: int,
          commands_run: int, consecutive_format_errors: int, checkpoint: str,
          image: str, messages: list[dict],
          state_manifest: dict | None = None) -> Prefix:
    """A ``Prefix`` with its digest computed. What the pre-roll writes."""
    payload = {"task_name": task_name, "depth_tokens": depth_tokens,
               "prefill_tokens": prefill_tokens, "turns": turns,
               "commands_run": commands_run,
               "consecutive_format_errors": consecutive_format_errors,
               "checkpoint": checkpoint, "image": image,
               "state_manifest": dict(state_manifest or {}),
               "messages": [dict(m) for m in messages]}
    return from_payload({**payload, "digest": digest_of(payload)})


#: Suffix of the directory holding a prefix's sandbox state, beside its file.
STATE_SUFFIX = ".state"

#: Size recorded for a directory. Its name carries the meaning and its contents do not,
#: so the manifest records that it was there rather than how big it was. A deletion
#: marker is an ordinary empty file and records its own size, 0.
MARKER_SIZE = -1

#: Size recorded for a directory that replaced one from the layer below rather than
#: merging with it. Distinct from an ordinary directory because the difference is the
#: whole of what it means: merged, the resumed container also sees the files the image
#: had there; replaced, it sees only what the pre-roll put there.
OPAQUE_SIZE = -2

#: The extended attribute the overlay driver marks a replaced directory with. It also
#: writes a ``.wh..wh..opq`` file beside it on some filesystems, which the manifest
#: catches as an ordinary entry, but the attribute is the marker that is always set and
#: a manifest that read only the file would miss the other configurations.
OPAQUE_XATTR = "user.fuseoverlayfs.opaque"


def _is_opaque(path: Path) -> bool:
    """Whether this directory replaced the one below it rather than merging.

    Reads the attribute where the platform has them and says no where it does not:
    a filesystem without extended attributes cannot be carrying an overlay upper in
    the first place, so there is nothing to miss."""
    try:
        return os.getxattr(path, OPAQUE_XATTR) in (b"y", b"y\x00")
    except (OSError, AttributeError):
        return False


def state_path(directory: str | Path, task_name: str) -> Path:
    """Where one task's sandbox state lives: a directory beside its prefix file.

    The overlay upper the pre-roll's container wrote, used in place rather than
    copied. See the module docstring for why a copy is not safe."""
    return Path(directory) / f"{task_name}{STATE_SUFFIX}"


def manifest_of(state_dir: str | Path) -> dict:
    """What a state directory holds, as ``relative path -> size``.

    Every entry, not only the files: a deletion is recorded by the *presence* of a
    marker whose contents are empty, so a manifest over file contents alone would miss
    exactly the difference between a prefix that deleted a file and one that did not.
    A directory takes ``MARKER_SIZE``, or ``OPAQUE_SIZE`` where it replaced the one
    below rather than merging with it, which is a difference in what the resumed
    container sees and so has to be part of the prefix's identity.

    Sizes rather than content hashes. This addresses the state, it does not verify it:
    a build tree is gigabytes, and hashing it on every load would cost more than the
    run it protects. What it catches is a directory staged short, staged from another
    task, or replaced -- which is what goes wrong with a staged artifact."""
    root = Path(state_dir)
    if not root.is_dir():
        return {}
    out: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_file() and not path.is_symlink():
            out[rel] = path.stat().st_size
        elif path.is_dir() and _is_opaque(path):
            out[rel] = OPAQUE_SIZE
        else:
            out[rel] = MARKER_SIZE
    return out


def prefix_path(directory: str | Path, task_name: str) -> Path:
    """Where one task's prefix is staged.

    One per task, because a task is assigned one depth band: the corpus declares which
    prefix each task carries, and a task with two would be two rows, which the split
    partition cannot keep together."""
    return Path(directory) / f"{task_name}{PREFIX_SUFFIX}"


def write(directory: str | Path, prefix: Prefix) -> Path:
    """Stage ``prefix`` and return where it landed."""
    path = prefix_path(directory, prefix.task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefix.payload(), sort_keys=True))
    return path


def load(directory: str | Path, task_name: str, *, expect_digest: str | None = None,
         expect_checkpoint: str | None = None) -> Prefix:
    """Read one task's staged prefix, refusing anything but the declared one.

    ``expect_digest`` is what the corpus pinned. A prefix that loaded without that
    check would let a re-cut file run under a fingerprint that had not moved, which is
    the defect the corpus's image pins already close.

    ``expect_checkpoint`` is the task's model. The digest covers it, so this only ever
    fires alongside a digest mismatch; it is here to say which of the two changed."""
    path = prefix_path(directory, task_name)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PrefixError(
            f"no prefix for {task_name!r} at {path}; the corpus declares one, so stage "
            f"it before the run") from exc
    except json.JSONDecodeError as exc:
        raise PrefixError(f"{path} is not valid JSON ({exc}); re-stage it") from exc
    prefix = from_payload(payload)
    if prefix.task_name != task_name:
        raise PrefixError(
            f"{path} holds a prefix cut from {prefix.task_name!r}, not {task_name!r}; "
            f"a session would resume one task's conversation in another's container")
    if expect_checkpoint is not None and prefix.checkpoint != expect_checkpoint:
        raise PrefixError(
            f"prefix {task_name!r} was cut against {prefix.checkpoint!r} but this task "
            f"serves {expect_checkpoint!r}; a conversation is the writing of the "
            f"weights that produced it, so re-cut it against this checkpoint")
    if expect_digest is not None and prefix.digest != expect_digest:
        raise PrefixError(
            f"prefix {task_name!r} hashes to {prefix.digest[:12]}... but the corpus "
            f"declares {expect_digest[:12]}...; the staged file is not the one this "
            f"task was calibrated over")
    found = manifest_of(state_path(directory, task_name))
    if found != prefix.state_manifest:
        missing = sorted(set(prefix.state_manifest) - set(found))[:5]
        extra = sorted(set(found) - set(prefix.state_manifest))[:5]
        raise PrefixError(
            f"the sandbox state staged for {task_name!r} is not the one this prefix "
            f"was cut with: {len(prefix.state_manifest)} entries expected, "
            f"{len(found)} found (missing {missing}, unexpected {extra}). A session "
            f"would resume a conversation about files that are not there")
    return prefix


def _source_prefix(source: Path, task_name: str, pin: str) -> tuple[Prefix, bytes]:
    """The prefix of *task_name* in *source* and its file's bytes, refused in one line unless it
    hashes to *pin*."""
    path = prefix_path(source, task_name)
    try:
        data = path.read_bytes()
        payload = json.loads(data)
    except FileNotFoundError as exc:
        raise PrefixError(f"{path} does not exist, and the corpus pins {pin} for {task_name}") from exc
    except json.JSONDecodeError as exc:
        raise PrefixError(f"{path} is not valid JSON ({exc})") from exc
    found = digest_of(payload) if isinstance(payload, dict) and all(k in payload for k in _FIELDS) else None
    if found != pin:
        raise PrefixError(f"{path} hashes to {found}, not {pin}, the digest the corpus pins for {task_name}; "
                          f"nothing of it is placed")
    if payload.get("digest") != pin:
        raise PrefixError(f"{path} carries the digest {payload.get('digest')} but hashes to {pin}; it was edited "
                          f"after it was cut, and nothing of it is placed")
    return from_payload(payload), data


def _mismatch(where: Path, found: dict, want: dict) -> str:
    """How the manifest *found* at *where* differs from *want*, on one line."""
    missing = sorted(set(want) - set(found))[:5]
    extra = sorted(set(found) - set(want))[:5]
    sized = sorted(k for k in set(want) & set(found) if want[k] != found[k])[:5]
    return (f"{where} holds {len(found)} entries where its prefix records {len(want)}: missing {missing}, "
            f"unexpected {extra}, of another size {sized}")


def _aside(path: Path, stamp: int) -> None:
    """Rename *path*, when it exists, out of the way: a placement never removes what it found."""
    if path.exists() or path.is_symlink():
        path.rename(path.with_name(f"{path.name}.unusable-{stamp}"))


#: The device number of an overlay whiteout: a character device 0:0 in an upper directory hides
#: the name from the layers below. The kernel lets any user make one (Linux 5.8 and later).
WHITEOUT_DEVICE = (0, 0)


def _is_whiteout(path: Path) -> bool:
    """Whether *path* is an overlay whiteout, a character device 0:0."""
    st = os.lstat(path)
    return stat.S_ISCHR(st.st_mode) and (os.major(st.st_rdev), os.minor(st.st_rdev)) == WHITEOUT_DEVICE


#: Suffix of the archive a repository holds a prefix's state in, beside its file: one
#: uncompressed tar of the state as ``to_repository`` writes it. A tree a commit held as it is
#: would lose what git does not track: a path with a ``.git`` component (a build tree holds the
#: ``.git`` of every package it cloned), and an empty directory.
ARCHIVE_SUFFIX = ".state.tar"


def archive_path(directory: str | Path, task_name: str) -> Path:
    """Where a repository holds one task's state archive: beside its prefix file."""
    return Path(directory) / f"{task_name}{ARCHIVE_SUFFIX}"


def _entry(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """*info* with nothing of the machine that wrote it: owner 0, no names, time 0."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def _held_entry(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    """*member* of a state archive, refused unless it is a file, a directory or a symlink named
    below the archive's root. A symlink keeps its target as written, absolute or not: an upper
    layer's links point into the container's own tree, and nothing is extracted through one,
    since ``to_repository`` never enters a link."""
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir() or member.issym()):
        raise PrefixError(f"{member.name} is no file, directory or symlink below the archive's root")
    return member


def to_repository(state: str | Path, out: str | Path) -> None:
    """Write the state directory *state* to the archive *out* in the form a repository holds:
    each whiteout as an empty file of its name, no extended attributes, and entries in sorted
    order with owner 0 and time 0, so the same state always writes the same bytes. ``place``
    makes the whiteouts and opaque marks again from the prefix's manifest, which tells a
    whiteout from a file by its size (``MARKER_SIZE`` against 0) and an opaque directory by
    ``OPAQUE_SIZE``. Raise PrefixError for any entry that is neither a file, a directory, a
    symlink nor a whiteout."""
    state, out = Path(state), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as archive:
        for path in sorted(state.rglob("*"), key=lambda p: p.relative_to(state).as_posix()):
            rel = path.relative_to(state).as_posix()
            if path.is_symlink() or path.is_dir():
                archive.addfile(_entry(archive.gettarinfo(path, rel)))
            elif path.is_file():
                with path.open("rb") as f:
                    archive.addfile(_entry(archive.gettarinfo(path, rel)), f)
            elif _is_whiteout(path):
                info = _entry(tarfile.TarInfo(rel))
                info.mode = 0o644
                archive.addfile(info)
            else:
                raise PrefixError(f"{path} is neither a file, a directory, a symlink nor a whiteout, so no "
                                  f"repository holds it")


def _rebuild(copy: Path, want: dict, task_name: str) -> None:
    """Make again in *copy*, the repository's form of a state copied (``to_repository``), what the
    repository could not hold, from the manifest *want*: each directory with no entry under it,
    each whiteout (an entry recorded as ``MARKER_SIZE`` that the copy holds as an empty file), and
    each opaque directory's mark. Raise PrefixError, in one line, when this node will not make
    one."""
    parents = {str(Path(rel).parent) for rel in want}
    for rel in sorted(want):
        size, path = want[rel], copy / rel
        if size >= 0 or rel in parents:
            continue
        try:
            if not path.exists() and not path.is_symlink():
                path.mkdir(parents=True)
            elif size == MARKER_SIZE and not path.is_symlink() and path.is_file() and path.stat().st_size == 0:
                path.unlink()
                os.mknod(path, stat.S_IFCHR, os.makedev(*WHITEOUT_DEVICE))
        except OSError as exc:
            raise PrefixError(f"{path} cannot be made again as its manifest records it ({exc}); nothing of "
                              f"{task_name} is placed") from exc
    for rel in sorted(rel for rel, size in want.items() if size == OPAQUE_SIZE):
        try:
            os.setxattr(copy / rel, OPAQUE_XATTR, b"y")
        except OSError as exc:
            raise PrefixError(f"{copy / rel} cannot be marked as replacing the directory below it ({exc}); "
                              f"nothing of {task_name} is placed") from exc


def place(source: str | Path, store: str | Path, task_name: str, pin: str) -> str:
    """Put the prefix of *task_name* and its state from *source*, the repository's copy, into
    *store*, the prefix store; return ``"held"`` when the store already holds the one the
    corpus pins, ``"placed"`` otherwise.

    Nothing that does not match is placed: the file in *source* must hash to *pin*, and the
    copy of its state must hold what the prefix's manifest records. Each refusal is a
    PrefixError of one line. The repository holds the state as the archive ``to_repository``
    writes (``archive_path``), which is extracted into the copy; the copy gets back from the
    manifest what that form drops (``_rebuild``): whiteouts and opaque marks, and an empty
    directory of a state with no archive. Both are made beside their place and renamed into it;
    what the store held under either name before is renamed aside, never removed."""
    source, store = Path(source), Path(store)
    try:
        load(store, task_name, expect_digest=pin)
        return "held"
    except PrefixError:
        pass
    prefix, data = _source_prefix(source, task_name, pin)
    want = dict(prefix.state_manifest)
    archive = archive_path(source, task_name)
    store.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    partial = Path(tempfile.mkdtemp(prefix=f".{task_name}.placing-", dir=store))
    try:
        copy = partial / f"{task_name}{STATE_SUFFIX}"
        copy.mkdir()
        if archive.is_file():
            with tarfile.open(archive) as held:
                held.extractall(copy, filter=_held_entry)
        _rebuild(copy, want, task_name)
        placed = manifest_of(copy)
        if placed != want:
            raise PrefixError(_mismatch(archive, placed, want))
        (partial / f"{task_name}{PREFIX_SUFFIX}").write_bytes(data)
        for name in (f"{task_name}{STATE_SUFFIX}", f"{task_name}{PREFIX_SUFFIX}"):
            _aside(store / name, stamp)
            (partial / name).rename(store / name)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return "placed"


__all__ = ["ARCHIVE_SUFFIX", "MARKER_SIZE", "OPAQUE_SIZE", "OPAQUE_XATTR", "PREFIX_SUFFIX",
           "STATE_SUFFIX", "WHITEOUT_DEVICE", "Prefix", "PrefixError",
           "archive_path", "build", "digest_of", "from_payload", "load", "manifest_of", "place",
           "prefix_path", "state_path", "to_repository", "write"]
