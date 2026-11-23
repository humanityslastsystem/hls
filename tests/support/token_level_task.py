"""Sample task -- token-level (teacher-forced) faithfulness verifier; live-agent contestant.

A test fixture (no catalog directory, so no command plans it): the sample_task arena with
`TokenFaithfulnessVerifier`, the calibrated token-level faithfulness gate, as its verifier.
The contestant's served token ids are teacher-forced through a pinned reference, and the run
fails if the pooled flip rate exceeds 2x the calibrated rate ceiling (q_max).
The calibration also carries the flip-count curve rule's per-level ceilings,
so both rules run here. The contract returns token_ids, so the agent's build brief (derived
from the contract) instructs the server to return choices[].token_ids. The provision tests
resolve it, and the calibration and brief tests build on it.

TokenFaithfulnessVerifier needs a calibration fingerprinted to this task's
model/contract/seed/dataset before it can grade; none is committed beside this module, so a
test that needs one makes its own.

It reuses sample_task's arena (scorer, seed) by import.
"""
from __future__ import annotations

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation,
                                   HighestAvailable)
from hls.core.decode import Greedy
from hls.reference.references import HFReference
from hls.prompt import JinjaPromptTemplate
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
from hls.verify import TokenFaithfulnessVerifier

# Reuse sample_task's arena (scorer/seed) by import. The
# DATASET/WORKLOAD are redefined here rather than reused: sample_task's arena is shaped for the
# letter-match verifier (a "single letter" prompt under a 4-token cap), which serves
# ~2 ultra-peaked tokens per request -- a token-level gate has nothing to measure
# there (the test split would total ~40 tokens, under any honest min_tokens floor,
# and trusted engines exhibit zero flips on such traffic, so calibration refuses
# with a degenerate all-zero ceiling). This
# variant re-shapes the same MMLU corpus into short free-form reasoning traffic:
# enough served tokens to clear the floor, and enough near-tie prose for the
# trusted-engine pool to show a nonzero bf16 noise floor.
from tasks.sample_task.task import SCORER, SEED
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec

# Task-local hardware, not sample_task's H200 declaration, because `hls calibrate`
# refuses to measure on anything but the declared accelerator, and the verifier
# refuses to score when the serving reference's device is not the calibrated
# kind -- so this declaration must name the GPU the demo actually runs on.
HARDWARE = Hardware(accelerator=AcceleratorSpec(
    kind="B200", count=1, memory_gb_per_device=180),
    cpu_count=64, memory_gb=512, storage_gb=2048)

# The MODEL differs from sample_task's Qwen2.5-0.5B for a related reason. cuDNN's SDPA
# decode kernel (torch 2.11-2.13 / cuDNN 9.19-9.20 / B200) returns an exact-zero
# output for any attention head whose logits are all below ln(FLT_MIN) ~ -87, and
# Qwen-family checkpoints carry many such "switched-off" heads, so stock torch kernel
# dispatch serves occasional wrong tokens on them. Llama-3.2-1B's heads never enter
# the defect's regime, so stock dispatch is safe here and the calibration pool's
# hf_default member covers it.
MODEL = ModelSpec(base_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
                  revision="9213176726f574b556790deb65791e0c5aa438b6", arch="TX", chat_format="llama3")

#: Per-request decoding budget: room for two or three reasoning sentences plus the
#: final answer line (vs sample_task's 4 -- see the import note above).
MAX_NEW_TOKENS = 64

DATASET = HFDataset(path="cais/mmlu", subset="abstract_algebra",
                    revision="c30699e8356da336a370243923dbaf21066bb9fe",
                    source_split="test",
                    prompt=JinjaPromptTemplate(
                        template="Answer the question. Explain your reasoning in two or "
                                 "three short sentences, then give the final line as "
                                 "'Answer: <letter>'.\n\n"
                                 "Question: {{ question }}\n"
                                 "A. {{ choices[0] }}\nB. {{ choices[1] }}\n"
                                 "C. {{ choices[2] }}\nD. {{ choices[3] }}"),
                    target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))
# The upstream corpus (abstract_algebra source test split) has 100 rows; dev+test=80
# leaves headroom. The eval always serves the held-out test partition: 64 rows x up
# to 64 tokens ~ 2-4k served tokens (the model stops early on some answers).
DATA_SOURCE = DataSource(
    dataset=DATASET,
    split=SplitSpec(dev=Count(value=16), test=Count(value=64)))

WORKLOAD = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=MAX_NEW_TOKENS),
                    slo=NoSLO())

# Contract always returns token_ids (mandatory) so the verifier can read the contestant's
# exact emitted ids (no re-tokenization). Plain http (unary) suffices --
# OpenAIChatProtocol.decode_response lifts choices[].token_ids into TextPayload.tokens; the
# build brief (rendered from this contract) tells the agent to return choices[].token_ids.
# Also enables top-k logprobs (with per-alternative token_id) -- the wire data
# TokenFaithfulnessVerifier's emitted-token consistency check (check_emit_consistency,
# left at its off default here) consumes.
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
    transport="http", protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=20))


# Oracle: the same model as the contestant, teacher-forced via greedy decode (held
# inside the verifier; loaded when the verify stage loads its engine).
_ORACLE = HFReference(decode=Greedy())
# Every measured parameter -- both rules' ceilings and the fail-closed trace floor --
# comes from calibration.json; only the 2x headroom is authored here. This demo's test
# split (64 rows x up to 64 tokens, a few thousand tokens) sits under the rate rule's
# ~5,000-token false-alarm floor for this checkpoint's q_max, so every verdict here
# carries "allow_underpowered_trace": true -- do not copy this flag onto a task whose
# verdict needs to be trusted.
VERIFIERS = [TokenFaithfulnessVerifier(
    reference=_ORACLE,
    allow_underpowered_trace=True,
)]

# Reuse sample_task's SEED so the split partitioning is intentional and identical.
TASK = TaskSpec(id="sample-token-level", model=MODEL, hardware=HARDWARE,
                contract=CONTRACT, workload=WORKLOAD, verifiers=VERIFIERS,
                scorer=SCORER, data_source=DATA_SOURCE, seed=SEED,
                metadata={"theme": "sample"})

# Default contestant: the live implementer agent (mirrors sample_task; pure config).
# The brief is rendered from CONTRACT, so the agent is told to return choices[].token_ids.
ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        hard_budget=Budget(wallclock_sec=3600, turns=40),
        soft_budget=Budget(wallclock_sec=1800),   # 30 min soft pacing budget (surfaced via the budget MCP)
        # As sample_task: the provider's top level, resolved at launch. A run refuses
        # an unset effort, since the harnesses' own defaults differ and go unrecorded.
        reasoning_effort=HighestAvailable(),
        # As sample_task: what a build holds, 8 CPUs and 180 GiB, whose writable disk cap
        # bounds the repository the agent commits.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180)),
    serve_command=[])

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "VERIFIERS"]
