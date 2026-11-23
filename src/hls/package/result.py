"""``package.json``: what the package stage publishes, built or not (``PackageResult``), and what
a grading reads of it (``PackageOutcome``), which a package an older or newer package stage
published still parses as.

An ``invalid_artifact`` is a result, not a fault: the stage publishes it like
a built image, a grading scores it zero, and it is never retried. Every failure that is not
the contestant's raises an infra fault instead and publishes nothing."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from hls.store.digest import Sha256Hex
from hls.substrate.nproc import ProcessCeiling
from hls.substrate.ucounts import ServedUcounts


class PackageOutcome(BaseModel):
    """What a grading reads of a ``package.json``: whether the submission built, the image it
    built and that image's digest, or why it did not, with the failed step's exit status and the
    end of its output. The other fields of the record are left unread, so a package an older or a
    newer package stage published, with fields this code lacks or does not know, still grades;
    the fields read must be there and agree with one another. The package stage writes the whole
    record, ``PackageResult``."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    #: ``built``, or ``invalid_artifact`` when the submission itself does not build.
    status: Literal["built", "invalid_artifact"]
    #: Store key of the submission image; set exactly when built.
    image: str | None = None
    #: Digest of the submission image's file; set exactly when built.
    sif_sha256: Sha256Hex | None = None
    #: Why the submission is invalid, for the contestant's record; set exactly when invalid.
    reason: str | None = None
    #: Exit status of the step that failed, when one did.
    returncode: int | None = None
    #: The end of that step's output, scrubbed of every secret value.
    log_tail: str | None = None
    #: The figures the build's brief stated for its graded server, as the build recorded them
    #: (``hls.stages.package.served_record``), which the grading seal writes; None for an operator's
    #: named bundle, which no brief stated figures for, for a build that recorded none, and in a
    #: record without the field: the grading computes the figures for each.
    served_ucounts: ServedUcounts | None = None
    #: The process ceiling the pip steps ran under (``StageContext.process_ceiling``); None when
    #: no pip step ran.
    process_ceiling: ProcessCeiling | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # The errors are typed by name, so a refusal can say which rule failed without its text.
        built = self.status == "built"
        if built != (self.image is not None and self.sif_sha256 is not None):
            raise PydanticCustomError("image_set_exactly_when_built",
                                      "a built package names its image and digest, and only a built one")
        if built == (self.reason is not None):
            raise PydanticCustomError("reason_set_exactly_when_invalid",
                                      "an invalid package says why, and only an invalid one")
        return self


class PackageResult(PackageOutcome):
    """The outcome of one package attempt, as the package stage writes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task the submission was built for.
    task: str
    #: Digest of the source bundle the package read; None when an agent_build bundle was refused
    #: for its size, unread, and when the build left no bundle (agent_build's ``no_submission``).
    bundle_sha256: Sha256Hex | None
    #: The commit of ``main`` that was built; None when the bundle has no ``main``, and when
    #: there is no bundle.
    commit: str | None
    #: The pins the wheel step built besides the project, in order.
    wheels: list[str] = Field(default_factory=list)
    #: As the outcome's, which the stage always states, None or not.
    served_ucounts: ServedUcounts | None

    @model_validator(mode="after")
    def _built_from_a_bundle(self) -> Self:
        if self.status == "built" and self.bundle_sha256 is None:
            raise PydanticCustomError("built_without_its_bundle", "a built package names the bundle it was built from")
        return self
