"""A task for the agent build's tests: the sample task, with a producer that uploads a folder of
reference files into the workspace, as the production tasks upload theirs. The folder's digest is
taken from the folder itself, since its files' permission bits are the checkout's."""

from pathlib import Path

from hls.agent.uploads import tree_digest
from hls.artifact.artifact import AgentProducedArtifact, AgentProducerSpec, Budget, HighestAvailable
from tasks.sample_task.task import TASK

__all__ = ["ARTIFACT", "TASK"]

ARTIFACT = AgentProducedArtifact(producer=AgentProducerSpec(
    hard_budget=Budget(wallclock_sec=3600), soft_budget=Budget(wallclock_sec=1000), reasoning_effort=HighestAvailable(),
    upload_dir=("build_task_files",), upload_sha256=tree_digest(Path(__file__).parent / "build_task_files"),
    upload_dest="/workspace/references", upload_brief="## References\n\nThe task's reference notes."))
