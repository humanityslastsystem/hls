"""A stage for tests that needs the secrets registry and reports what its context gave it:
the names of its secrets, a digest of each value, whether the scrubber knows them, and whether
the runner handed it the registry.
Importing this module registers it as kind ``registry_probe``."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hls.pipeline.stage import Stage, StageContext, register
from hls.spec.resources import ResourceShape


class RegistryParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@register
class RegistryProbeStage(Stage):
    kind: ClassVar[str] = "registry_probe"
    params_model: ClassVar[type[BaseModel]] = RegistryParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    needs_secret_registry: ClassVar[bool] = True

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """It starts no role."""
        return 0

    def run(self, ctx: StageContext) -> dict[str, str]:
        seen = {
            "names": sorted(ctx.secrets),
            "digests": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ctx.secrets.items()},
            "scrubbed": all(ctx.scrubber.scrub(v) != v for v in ctx.secrets.values()),
            "registry": ctx.secret_registry,
        }
        return {"seen": ctx.write_output("seen.json", json.dumps(seen, sort_keys=True).encode())}
