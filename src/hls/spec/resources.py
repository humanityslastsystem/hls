"""The resources a stage or role declares, and how declarations combine.

One declaration per stage feeds three things: the allocation request, the brief's
hardware section and the runner's check of what Slurm actually granted. So the brief
can never state hardware the run does not get."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

#: Accelerator kinds a shape may name, each with the pattern that finds it in the product
#: name nvidia-smi reports ("NVIDIA H200", "NVIDIA L40S").
GPU_KINDS: dict[str, re.Pattern[str]] = {
    "A100": re.compile(r"\bA100\b"),
    "H100": re.compile(r"\bH100\b"),
    "H200": re.compile(r"\bH200\b"),
    "B200": re.compile(r"\bB200\b"),
    "L40": re.compile(r"\bL40\b"),
    "L40S": re.compile(r"\bL40S\b"),
}


def check_gpu_kind(kind: str) -> str:
    """Return *kind* if it is one of ``GPU_KINDS``, else raise ValueError."""
    if kind not in GPU_KINDS:
        raise ValueError(f"unknown gpu kind {kind!r}; known kinds are {sorted(GPU_KINDS)}")
    return kind


def gpu_kind_of(product_name: str) -> str | None:
    """The kind a GPU's product name names, or None when it names no known kind or several."""
    found = [kind for kind, pattern in GPU_KINDS.items() if pattern.search(product_name)]
    return found[0] if len(found) == 1 else None


#: A GPU kind from ``GPU_KINDS``.
GpuKind = Annotated[str, AfterValidator(check_gpu_kind)]


class ResourceShape(BaseModel):
    """What one stage or role needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: GPUs, all of one kind.
    gpus: int = Field(default=0, ge=0)
    #: Accelerator kind from ``GPU_KINDS``; set exactly when gpus > 0.
    gpu_kind: GpuKind | None = None
    #: CPU cores.
    cpus: int = Field(ge=1)
    #: Memory limit in GiB.
    memory_gib: int = Field(ge=1)
    #: Free local disk in GiB the stage needs on the node.
    disk_gib: int = Field(default=0, ge=0)
    #: Wall-clock limit in seconds.
    time_sec: int = Field(gt=0)

    @model_validator(mode="after")
    def _kind_iff_gpus(self) -> ResourceShape:
        if (self.gpus > 0) != (self.gpu_kind is not None):
            raise ValueError("gpu_kind must be set exactly when gpus > 0")
        return self


class GrantedResources(BaseModel):
    """What the runner measured on the node it was given."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Visible GPUs.
    gpus: int = Field(ge=0)
    #: Kind of the visible GPUs; None when there are none, or when they are not all of one
    #: known kind.
    gpu_kind: GpuKind | None = None
    #: The product name of the visible GPUs as nvidia-smi reports it ("NVIDIA H200"), the
    #: distinct names sorted and joined by "; " when they differ; None when there are none, and in
    #: a grant measured without it.
    gpu_model: str | None = None
    #: CPUs the job may use (``nproc``).
    cpus: int = Field(ge=1)
    #: The node's CPU model, the ``model name`` of ``/proc/cpuinfo``, the distinct names sorted
    #: and joined by "; " when they differ; None when the node names none, and in a grant
    #: measured without it.
    cpu_model: str | None = None
    #: The job's cgroup memory limit in GiB.
    memory_gib: float = Field(gt=0)
    #: Free space on the node-local scratch in GiB.
    disk_free_gib: float = Field(ge=0)

    def hardware(self) -> HardwareRecord:
        """The hardware record of this grant."""
        return HardwareRecord(gpu_model=self.gpu_model, gpus=self.gpus, cpu_model=self.cpu_model, cpus=self.cpus,
                              memory_gib=self.memory_gib)


#: The fields of a hardware record a grading's is compared with its baseline credit's on:
#: the CPUs and memory are the allocation's shape, which the task declares.
COMPARED_HARDWARE = ("gpu_model", "gpus", "cpu_model")


class HardwareRecord(BaseModel):
    """What a run ran on, as the grant check measured it on the node: the hardware record every
    run's records state. A baseline credit carries ``BaselineHardware``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The GPUs' product name (``GrantedResources.gpu_model``); None for none.
    gpu_model: str | None
    #: The number of GPUs.
    gpus: int = Field(ge=0)
    #: The CPU model (``GrantedResources.cpu_model``); None when the node names none.
    cpu_model: str | None
    #: The CPUs the allocation may use.
    cpus: int = Field(ge=1)
    #: The allocation's memory limit in GiB.
    memory_gib: float = Field(gt=0)

    def differences(self, baseline: BaselineHardware) -> list[str]:
        """How this record differs from *baseline*'s on the fields of ``COMPARED_HARDWARE`` that
        *baseline* records, one "<field> <this> (baseline <baseline's>)" per field that differs, in
        that order; empty for the same hardware. A field *baseline* leaves blank is not compared."""
        return [f"{name} {getattr(self, name)} (baseline {getattr(baseline, name)})" for name in COMPARED_HARDWARE
                if getattr(baseline, name) is not None and getattr(self, name) != getattr(baseline, name)]


class BaselineHardware(BaseModel):
    """The hardware a baseline credit's drives ran on, as its file records it: the fields of a
    ``HardwareRecord``, any of them blank (None), which a grading's record is not compared on.
    A credit the harness measures records every field its drives' grants
    measured; a credit may record fewer, such as only the GPU model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The GPUs' product name, as ``GrantedResources.gpu_model`` reads it.
    gpu_model: str | None = None
    #: The number of GPUs.
    gpus: int | None = Field(default=None, ge=0)
    #: The CPU model, as ``GrantedResources.cpu_model`` reads it.
    cpu_model: str | None = None
    #: The CPUs the allocation could use.
    cpus: int | None = Field(default=None, ge=1)
    #: The allocation's memory limit in GiB.
    memory_gib: float | None = Field(default=None, gt=0)

    @classmethod
    def of(cls, record: HardwareRecord) -> BaselineHardware:
        """*record*, every field it measured recorded."""
        return cls(**record.model_dump())

    def describe(self) -> str:
        """The compared fields it records, for people: "<gpus> x <GPU model>, CPU <CPU model>",
        each part only when recorded."""
        gpus = f"{self.gpus} x " if self.gpus is not None else ""
        gpu = f"{gpus}{self.gpu_model or 'GPUs'}" if gpus or self.gpu_model else ""
        cpu = f"CPU {self.cpu_model}" if self.cpu_model else ""
        return ", ".join(part for part in (gpu, cpu) if part) or "no compared field recorded"


class ArtifactPlacement(BaseModel):
    """Where an evaluation's artifact step ran, as its step wrapper reported it
    (``hls.substrate.role_exec.placement``): what the drive records, and the harness conditions of
    the instance's records name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The CPUs the step could use, ascending.
    cpus: list[int]
    #: The NUMA nodes they lie on, ascending.
    numa_nodes: list[int]
    #: True when they lie on more than one node although one node holds as many.
    numa_split: bool


@dataclass(frozen=True)
class Mismatch:
    """One way a grant differs from its declaration."""

    #: Declared field name.
    field: str
    #: Declared value.
    declared: object
    #: Granted value.
    granted: object


def _one_kind(shapes: list[ResourceShape]) -> str | None:
    kinds = {s.gpu_kind for s in shapes if s.gpu_kind is not None}
    if len(kinds) > 1:
        raise ValueError(f"cannot combine shapes with different gpu kinds: {sorted(kinds)}")
    return kinds.pop() if kinds else None


def _nonempty(shapes: Iterable[ResourceShape]) -> list[ResourceShape]:
    out = list(shapes)
    if not out:
        raise ValueError("no shapes to combine")
    return out


def sequential(shapes: Iterable[ResourceShape]) -> ResourceShape:
    """The allocation that runs *shapes* one after another: the largest of each
    resource, and the sum of the times."""
    s = _nonempty(shapes)
    return ResourceShape(
        gpus=max(x.gpus for x in s), gpu_kind=_one_kind(s), cpus=max(x.cpus for x in s),
        memory_gib=max(x.memory_gib for x in s), disk_gib=max(x.disk_gib for x in s),
        time_sec=sum(x.time_sec for x in s))


def concurrent(shapes: Iterable[ResourceShape]) -> ResourceShape:
    """What *shapes* need while running at the same time: the sum of each resource,
    and the longest time."""
    s = _nonempty(shapes)
    return ResourceShape(
        gpus=sum(x.gpus for x in s), gpu_kind=_one_kind(s), cpus=sum(x.cpus for x in s),
        memory_gib=sum(x.memory_gib for x in s), disk_gib=sum(x.disk_gib for x in s),
        time_sec=max(x.time_sec for x in s))


def check_grant(declared: ResourceShape, granted: GrantedResources, *,
                memory_tolerance_gib: float = 1.0) -> list[Mismatch]:
    """Every difference between *declared* and *granted* that would make the brief
    false. GPUs, GPU kind and CPUs must match exactly, since a larger grant makes the
    brief's numbers wrong too. Memory must match within *memory_tolerance_gib*, and
    free disk must be at least the declared amount."""
    out = []
    for name, want, got in (("gpus", declared.gpus, granted.gpus),
                            ("gpu_kind", declared.gpu_kind, granted.gpu_kind),
                            ("cpus", declared.cpus, granted.cpus)):
        if want != got:
            out.append(Mismatch(name, want, got))
    if abs(granted.memory_gib - declared.memory_gib) > memory_tolerance_gib:
        out.append(Mismatch("memory_gib", declared.memory_gib, granted.memory_gib))
    if granted.disk_free_gib < declared.disk_gib:
        out.append(Mismatch("disk_gib", declared.disk_gib, granted.disk_free_gib))
    return out
