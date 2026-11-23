"""A stage's manifest: the one record that says the stage succeeded.

It is published once, by create-only write, after every output is in the store. It
names each output with its digest, the manifests and cache entries the stage read, the
code and images it ran, and the resources it declared and was granted. A later stage pins each input
by the manifest's store key and digest, so it can tell when an input is not the one
the plan expected. Keys rather than stage names identify inputs because a grading
reads manifests of its rollout, and after a rebuild both instances have a stage of
the same name."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from hls.spec.resources import GrantedResources, ResourceShape
from hls.store.base import AlreadyExists, NotFound, Store
from hls.store.cache import hash_file
from hls.store.digest import Sha256Hex, check_sha256_hex, sha256_bytes
from hls.store.layout import InstanceLayout, check_manifest_key, check_stage
from hls.util import validation_errors, write_in_place

#: Store key of a stage manifest, e.g. ``runs/<task>/<rollout>/stages/agent_build/MANIFEST.json``.
ManifestKey = Annotated[str, AfterValidator(check_manifest_key)]


class AlreadyPublished(Exception):
    """A stage's manifest already exists, so the stage already succeeded."""


class InputMismatch(Exception):
    """An input manifest is missing or is not the one the plan named."""


class OutputRef(BaseModel):
    """One output of a stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Store key of the output.
    key: str
    #: Hex SHA-256 of its bytes.
    sha256: Sha256Hex
    #: Size in bytes.
    size: int = Field(ge=0)


class ReportedOutput(BaseModel):
    """What a role reports of a stage output it wrote (``report_outputs``,
    ``report_written_outputs``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hex SHA-256 of the bytes on disk: read back once written, or hashed as the role wrote
    #: them and then synced.
    sha256: Sha256Hex
    #: Their size in bytes, which publish holds the file to.
    size: int = Field(ge=0)


#: A role's report of its outputs: each one's path under the outputs directory to its record.
_OUTPUTS_REPORT = TypeAdapter(dict[str, ReportedOutput])


def _output_name(root: Path, path: Path) -> str:
    """The "/"-separated path of the file *path* under *root*. Raise ValueError for a path that
    is not under *root*, ``..`` steps included."""
    rel = path.relative_to(root)
    if not rel.parts or ".." in rel.parts:
        raise ValueError(f"{path} is not a file under {root}")
    return rel.as_posix()


def _write_report(report: Path, records: Mapping[str, ReportedOutput]) -> None:
    """Write *records* into *report* in place, then cut it to their length (``write_in_place``).
    A report file the stage made larger beforehand (``hls.stages.deploy_drive``) so takes no new
    disk space, which a contestant on the same node may have used up."""
    write_in_place(report, _OUTPUTS_REPORT.dump_json(dict(sorted(records.items())), indent=1))


def report_outputs(report: Path, root: Path, *files: Path) -> None:
    """For a role that wrote the stage outputs *files*: report each one's sha256 and size, read
    back and synced (``hash_file``), keyed by its "/"-separated path under *root*, the attempt's
    outputs directory as the role sees it, in *report*, the place the stage gave it
    (``StageContext.outputs_report``). The stage's manifest records them, and the runner reads
    none of the files. Raise ValueError for a file outside *root*."""
    records = {}
    for f in files:
        name = _output_name(root, f)
        sha256, size = hash_file(f)
        records[name] = ReportedOutput(sha256=sha256, size=size)
    _write_report(report, records)


def report_written_outputs(report: Path, root: Path, written: Mapping[Path, ReportedOutput]) -> None:
    """``report_outputs`` for a role that hashed each output as it wrote it: report *written*,
    each output file's sha256 and size, keyed by its "/"-separated path under *root*, in
    *report*, without reading any file back. Its one rule: every record is the role's own
    digest of bytes it wrote itself, never a digest of a file it did not write, which only a
    read of the file could give. A role writing gigabytes reports them so without a second read.
    Raise ValueError for a file outside *root*, before anything is reported."""
    records = {_output_name(root, path): record for path, record in written.items()}
    _write_report(report, records)


def read_outputs_report(report: Path) -> dict[str, ReportedOutput] | None:
    """The report a role wrote to *report* with ``report_outputs``; None when there is none, or
    when it is not a map of records."""
    try:
        return _OUTPUTS_REPORT.validate_json(report.read_bytes())
    except (OSError, ValidationError):
        return None


class Provenance(BaseModel):
    """The code, images and config a stage ran with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: hls commit of the checkout.
    hls_commit: str
    #: harbor submodule commit.
    harbor_commit: str
    #: Digest of the uncommitted changes in the hls checkout and its harbor submodule: every
    #: path that differs from HEAD or is untracked and not ignored, with its file bytes and
    #: executable bit, its symlink target, or a deleted marker. None when both are clean.
    #: See ``hls.pipeline.checkout``.
    dirty_diff_sha256: Sha256Hex | None
    #: Role name to the hex SHA-256 of each image the stage launched.
    image_digests: dict[str, Sha256Hex]
    #: SHA-256 of the resolved settings recorded in the plan.
    config_sha256: Sha256Hex
    #: "<kind>/<key>" of each cache entry the stage copied to its node, to the SHA-256 of that
    #: entry's ENTRY.json, which names every file of the entry by its digest.
    cache_entries: dict[str, Sha256Hex] = Field(default_factory=dict)


class Resources(BaseModel):
    """What the stage declared and what the node gave it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The stage's declaration.
    declared: ResourceShape
    #: The runner's measurement of the grant.
    granted: GrantedResources


class Times(BaseModel):
    """When the stage's allocation was submitted and started, and when the stage ran, in
    POSIX seconds. Several stages can share one allocation, so queue time is
    ``allocation_started_at - submitted_at`` and the stage's work is
    ``ended_at - started_at``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Submission time of the allocation; None when unknown.
    submitted_at: float | None
    #: When the stage's allocation started; None when unknown.
    allocation_started_at: float | None
    #: Stage start.
    started_at: float
    #: Stage end.
    ended_at: float


class Manifest(BaseModel):
    """The success record of one stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Instance root key, e.g. ``runs/<task>/<rollout>``.
    instance: str
    stage: str
    #: The attempt whose outputs these are.
    attempt: int = Field(ge=1)
    #: Output name to reference.
    outputs: dict[str, OutputRef]
    #: Store key of each manifest the stage read, possibly another instance's, to the
    #: digest of that manifest's bytes.
    inputs: dict[ManifestKey, Sha256Hex]
    #: Code, images and config.
    provenance: Provenance
    #: Declared and granted resources.
    resources: Resources
    #: Submission, start and end times.
    times: Times


def canonical_json(model: BaseModel) -> bytes:
    """Bytes that depend only on the model's content, not on dict insertion order."""
    data = model.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def output_ref(store: Store, key: str, sha256: str) -> OutputRef:
    """Reference an output already written to *key* by the *sha256* its writer reported, with its
    size from the store's metadata. Nothing of it is read: the process that publishes a manifest
    moves no large data, so an output's digest comes from whoever wrote its bytes.
    Raise as ``Store.size`` does, and ValidationError for a digest that is not bare
    hex."""
    return OutputRef(key=key, sha256=sha256, size=store.size(key))


def publish_manifest(store: Store, layout: InstanceLayout, manifest: Manifest) -> str:
    """Publish *manifest* once and return its digest. Raise AlreadyPublished if the
    stage already succeeded."""
    check_stage(manifest.stage)
    if manifest.instance != layout.root:
        raise ValueError(f"manifest is for {manifest.instance}, not {layout.root}")
    data = canonical_json(manifest)
    try:
        store.create(layout.manifest(manifest.stage), data)
    except AlreadyExists as e:
        raise AlreadyPublished(f"{layout.root}: {manifest.stage}") from e
    return sha256_bytes(data)


class UnreadableRecord(ValueError):
    """Bytes in the store that are not the record a reader expects: other bytes than the manifest
    that published them recorded, or bytes that do not validate as the record's model."""


def checked_output(store: Store, ref: OutputRef) -> bytes:
    """The bytes of output *ref*, checked against the digest its manifest recorded. Raise
    UnreadableRecord when they are other bytes."""
    data = store.read_bytes(ref.key)
    digest = sha256_bytes(data)
    if digest != ref.sha256:
        raise UnreadableRecord(f"{ref.key} does not match its manifest: {digest} is not {ref.sha256}")
    return data


def parse_record[M: BaseModel](key: str, data: bytes, model: type[M]) -> M:
    """*data*, read at *key*, as a *model*. Raise UnreadableRecord saying where it does not
    validate, never what it holds."""
    try:
        return model.model_validate_json(data)
    except ValidationError as e:
        raise UnreadableRecord(f"{key} is not a {model.__name__}: {validation_errors(e)}") from None


def output_record[M: BaseModel](store: Store, ref: OutputRef, model: type[M]) -> M:
    """The *model* output *ref* holds, its bytes checked against its manifest (``checked_output``)
    before they are parsed (``parse_record``)."""
    return parse_record(ref.key, checked_output(store, ref), model)


def read_manifest(store: Store, layout: InstanceLayout, stage: str) -> tuple[Manifest, str] | None:
    """The published manifest of *stage* and its digest, or None if it has not
    succeeded."""
    try:
        data = store.read_bytes(layout.manifest(stage))
    except NotFound:
        return None
    return Manifest.model_validate_json(data), sha256_bytes(data)


def require_inputs(store: Store, expected: dict[ManifestKey, Sha256Hex]) -> dict[str, Manifest]:
    """Read every input manifest the plan names, keyed by manifest store key, and return
    them under the same keys. Each manifest's bytes are checked against the planned
    digest before they are parsed. Raise InputMismatch naming every missing, changed,
    unparseable or misplaced manifest at once. A key or digest that is malformed is a fault in the
    plan, and raises ValueError before anything is read."""
    for key, digest in expected.items():
        check_manifest_key(key)
        check_sha256_hex(digest)
    got: dict[str, Manifest] = {}
    problems = []
    for key, digest in sorted(expected.items()):
        try:
            raw = store.read_bytes(key)
        except NotFound:
            problems.append(f"{key}: not published")
            continue
        actual = sha256_bytes(raw)
        if actual != digest:
            problems.append(f"{key}: digest {actual} is not the planned {digest}")
            continue
        try:
            manifest = Manifest.model_validate_json(raw)
        except ValidationError as e:
            problems.append(f"{key}: not a valid manifest ({validation_errors(e)})")
            continue
        try:
            own_key = InstanceLayout(manifest.instance).manifest(manifest.stage)
        except ValueError as e:
            problems.append(f"{key}: manifest names an invalid instance or stage: {e}")
            continue
        if own_key != key:
            problems.append(f"{key}: holds the manifest of {own_key}")
            continue
        got[key] = manifest
    if problems:
        raise InputMismatch("; ".join(problems))
    return got
