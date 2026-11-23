"""A task for the agent build's reference service tests: the build task's producer with the
service on, and the sample task graded against a vLLM reference, the engine the service builds."""

from hls.artifact.artifact import AgentProducedArtifact
from hls.reference.references import VLLMReference
from tasks.sample_task.task import TASK as SAMPLE_TASK
from tests.agent import build_task

__all__ = ["ARTIFACT", "TASK"]

TASK = SAMPLE_TASK.model_copy(update={"verifiers": [
    v.model_copy(update={"reference": VLLMReference(decode=v.reference.decode)}) for v in SAMPLE_TASK.verifiers]})
ARTIFACT = AgentProducedArtifact(
    producer=build_task.ARTIFACT.producer.model_copy(update={"reference_service": True}),
    brief=build_task.ARTIFACT.brief)
