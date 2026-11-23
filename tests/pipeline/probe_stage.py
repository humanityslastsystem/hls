"""A stage for tests that reports what its context gave it: the names of its secrets and a
digest of each value, whether the scrubber knows them, and the cache entry and image it was
told to use. It can also start one role first. Importing this module registers it as kind
``probe``."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.spec.resources import ResourceShape


class ProbeParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: "<kind>/<key>@<ENTRY.json digest>" of a cache entry to materialize; None for none.
    entry: str | None = None
    #: A digest to record as the image of role "probe"; None for none.
    image: str | None = None
    #: True to fail with an exception whose message holds every secret value the stage got.
    leak: bool = False
    #: True to start one harness role, "probe-role", wait for it, and raise its stage fault.
    role: bool = False
    #: Bytes a harness role, "probe-writer", writes to output ``written.bin`` and reports the
    #: digest of, which the stage names as output "written" and never reads; None for none.
    role_output: str | None = None


@register
class ProbeStage(Stage):
    kind: ClassVar[str] = "probe"
    params_model: ClassVar[type[BaseModel]] = ProbeParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """At most its two harness roles, probe-role and probe-writer."""
        return 2

    def run(self, ctx: StageContext) -> dict[str, str]:
        p: ProbeParams = ctx.params
        if p.leak:
            raise RuntimeError(f"GET https://hub.example/x?token={','.join(ctx.secrets.values())} failed")
        if p.role:
            ctx.start_role("probe-role", [sys.executable, "-c", "pass"],
                           ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="harness")
            fault = stage_fault(ctx.wait_roles())
            if fault is not None:
                raise StageFault(fault)
        seen: dict[str, object] = {}
        outputs: dict[str, str] = {}
        if p.role_output is not None:
            out, report = ctx.output_path("written.bin"), ctx.outputs_report("probe-writer")
            code = ("import sys; from pathlib import Path; from hls.store.manifest import report_outputs; "
                    "Path(sys.argv[1]).write_bytes(sys.argv[3].encode()); "
                    "report_outputs(Path(sys.argv[2]), Path(sys.argv[1]).parent, Path(sys.argv[1]))")
            ctx.start_role("probe-writer", [sys.executable, "-c", code, str(out), str(report), p.role_output],
                           ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="harness")
            fault = stage_fault(ctx.wait_roles())
            if fault is not None:
                raise StageFault(fault)
            outputs["written"] = ctx.reported_output("written.bin", "probe-writer")
        if p.entry is not None:
            kind, pinned = p.entry.split("/", 1)
            key, digest = pinned.rsplit("@", 1)
            seen["entry_path"] = str(ctx.materialize(kind, key, digest))
        if p.image is not None:
            ctx.record_image("probe", p.image)
        seen |= {
            "names": sorted(ctx.secrets),
            "digests": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ctx.secrets.items()},
            "scrubbed": all(ctx.scrubber.scrub(v) != v for v in ctx.secrets.values()),
        }
        return {**outputs, "seen": ctx.write_output("seen.json", json.dumps(seen, sort_keys=True).encode())}
