"""How a stage reads what the stages it comes after published: their manifests among its inputs,
and their outputs, each checked against the digest its manifest recorded. Every input the plan
promised and the stage cannot read so is infra, not retried; so is a measured file the plan pins
(``hls.measured``) that is not the bytes it pinned."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hls.measured import PinMismatch, load_pinned, pinned_dev_files, write_pinned
from hls.pipeline.stage import StageContext, StageFault
from hls.store.base import Store
from hls.store.manifest import Manifest, OutputRef, UnreadableRecord, checked_output
from hls.taskref import MeasuredPaths, TaskRef


def input_manifest(ctx: StageContext, stage: str) -> Manifest:
    """The manifest of this instance's *stage* among the stage's inputs."""
    key = ctx.plan.layout.manifest(stage)
    manifest = ctx.inputs.get(key)
    if manifest is None:
        raise StageFault.infra(f"stage {ctx.stage.name} reads {key}, which is not among its inputs", retryable=False)
    return manifest


def output(manifest: Manifest, file: str) -> OutputRef:
    """The output of *manifest* that is its attempt's ``outputs/<file>``."""
    suffix = f"/outputs/{file}"
    ref = next((ref for ref in manifest.outputs.values() if ref.key.endswith(suffix)), None)
    if ref is None:
        raise StageFault.infra(f"the {manifest.stage} manifest has no output {file}", retryable=False)
    return ref


def read_checked(store: Store, ref: OutputRef) -> bytes:
    """The bytes of output *ref* in *store*, checked against the digest its manifest recorded
    (``checked_output``). Raise an infra fault that is not retried when they are other bytes."""
    try:
        return checked_output(store, ref)
    except UnreadableRecord as e:
        raise StageFault.infra(str(e), retryable=False) from None


def read_output(ctx: StageContext, ref: OutputRef) -> bytes:
    """The bytes of output *ref* in the stage's store (``read_checked``)."""
    return read_checked(ctx.store, ref)


def measured_copies(ctx: StageContext, directory: Path) -> MeasuredPaths:
    """Copies, in *directory*, of the measured files the plan pins (``hls.measured.write_pinned``),
    for a role to load. Raise an infra fault that is not retried when a file is gone or is not the
    bytes the plan pinned: another attempt reads the same file."""
    try:
        return write_pinned(ctx.plan.measured, directory, checkout=ctx.plan.code.checkout,
                            store_root=ctx.plan.store_root)
    except ValueError as e:
        raise StageFault.infra(str(e), retryable=False) from None


def measured_task(ctx: StageContext, ref: TaskRef, *, calibration: bool, credit: bool) -> Any:
    """The task *ref* names with the measured files the plan pins injected when asked, for the
    stage itself to load (``hls.measured.load_pinned``). Raise an infra fault that is not retried
    when a file is gone or is not the bytes the plan pinned: another attempt reads the same file."""
    try:
        return load_pinned(ref, ctx.plan.measured, calibration=calibration, credit=credit,
                           checkout=ctx.plan.code.checkout, store_root=ctx.plan.store_root)
    except PinMismatch as e:
        raise StageFault.infra(str(e), retryable=False) from None


def dev_files(ctx: StageContext) -> dict[str, dict[str, str]]:
    """The dev reference files the plan pins, by tenant id, each name to the text published beside
    the tenant's dev rows (``hls.measured.pinned_dev_files``); empty when it pins none. Raise an
    infra fault that is not retried when the file is gone or is not the bytes the plan pinned."""
    try:
        return pinned_dev_files(ctx.plan.measured, checkout=ctx.plan.code.checkout, store_root=ctx.plan.store_root)
    except ValueError as e:
        raise StageFault.infra(str(e), retryable=False) from None
