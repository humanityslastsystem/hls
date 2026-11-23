"""sample_task-stream -- the SSE streaming variant of sample_task, a fixture of the reply-budget
tests (``tests/drive/test_reply_budget.py``).

Same arena as tasks.sample_task.task but the contract uses http-sse transport with
per-chunk token_ids, exercising the streaming-timing path end to end. A separate
task (not an env flag): streaming changes what is measured."""
from __future__ import annotations

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget,
                                   HighestAvailable)
from hls.spec.contracts import RequestResponseContract
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.taskspec import TaskSpec
from tasks.sample_task.task import (MODEL, HARDWARE, DATA_SOURCE, VERIFIERS, WORKLOAD, SCORER,
                                    SEED)

CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
    transport="http-sse", protocol=OpenAIChatProtocol())

TASK = TaskSpec(id="sample_task-stream", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
                workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                metadata={"theme": "T1", "tier": "Bronze", "transport": "http-sse"})

ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600, turns=40),
                               # As sample_task: the provider's top level, resolved at
                               # launch. A run refuses an unset effort, since the
                               # harnesses' own defaults differ and go unrecorded.
                               reasoning_effort=HighestAvailable()),
    serve_command=[])

__all__ = ["TASK", "ARTIFACT", "CONTRACT"]
