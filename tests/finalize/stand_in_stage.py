"""A stage for tests that stands in for a stage kind these tests do not run (agent_build,
verify): it publishes no outputs, or raises the fault
planted for its stage name and attempt. Importing this module registers it as kind
``stand_in``, and its variants with the declarations of the stage they stand in for."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hls.pipeline.stage import Stage, StageContext, StageFault, register
from hls.spec.resources import ResourceShape
from hls.substrate.base import FaultRecord

#: The fault each attempt of a stage raises, by stage name, first attempt first; an attempt
#: past the list's end, or one whose entry is None, publishes. Planted by a test.
PLANTED: dict[str, list[FaultRecord | None]] = {}


class StandInParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@register
class StandInStage(Stage):
    kind: ClassVar[str] = "stand_in"
    params_model: ClassVar[type[BaseModel]] = StandInParams
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
        """It starts no role."""
        return 0

    def run(self, ctx: StageContext) -> dict[str, str]:
        faults = PLANTED.get(ctx.stage.name, [])
        fault = faults[ctx.attempt - 1] if ctx.attempt <= len(faults) else None
        if fault is not None:
            raise StageFault(fault)
        return {}


@register
class TranscriptStandInStage(StandInStage):
    """A stand-in whose outputs under ``transcript/`` an agent wrote, as agent_build's are, and
    whose ``notes.txt`` holds what a contestant said, written by the harness."""

    kind: ClassVar[str] = "transcript_stand_in"
    raw_outputs: ClassVar[tuple[str, ...]] = ("transcript/",)
    contestant_outputs: ClassVar[tuple[str, ...]] = ("notes.",)


@register
class AgentStandInStage(StandInStage):
    """A stand-in for a stage that hands its grants to the contestant it runs, as agent_build
    hands its arm's route credential and base URL to the agent."""

    kind: ClassVar[str] = "agent_stand_in"
    runs_contestant_code: ClassVar[bool] = True
    grants_reach_contestant: ClassVar[bool] = True
