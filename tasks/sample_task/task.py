"""sample_task -- the small public-model arena + the default (agentic) contestant.

One arena (TASK). The default contestant is the live implementer agent
(ARTIFACT = AgentProducedArtifact, pure config): the agent writes the inference
server, then it's committed and evaluated (``hls run tasks.sample_task.task --arm
<arm>``). Task modules export only declarative constants (no env reads, no
substrate); the images a task builds in and is driven from are the ones its ``IMAGES``
declares (``hls.recipes.catalog.ImageSet``).
"""

from __future__ import annotations

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation,
                                   EvaluationAllocation, HighestAvailable)
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import AccessList, NetworkPolicy
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.prompt import ImplementerBrief, JinjaPromptTemplate
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.taskspec import TaskSpec
from hls.verify import LetterMatchMetric, ReferenceRelativeVerifier
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload)
from hls.recipes.catalog import ImageSet

MAX_NEW_TOKENS = 4
SEED = 42

# The tiny public model, so a sample run is fast. The revision is the hub's main commit.
MODEL = ModelSpec(base_checkpoint="Qwen/Qwen2.5-0.5B-Instruct",
                  revision="7ae557604adf67be50417f59c2c2f167def9a775", arch="TX", chat_format="chatml")
# The model's weights are 1 GB in bf16, so one GPU, 8 CPUs and 32 GB of memory and of storage are
# ample for a server of it. A site with another GPU kind runs the task with
# `--task.hardware.accelerator.kind=<KIND>`; no memory per device is stated, since every kind
# holds the model.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1),
                    cpu_count=8, memory_gb=32, storage_gb=32)
DATASET = HFDataset(path="cais/mmlu", subset="abstract_algebra",
                    revision="c30699e8356da336a370243923dbaf21066bb9fe",
                    source_split="test",
                    prompt=JinjaPromptTemplate(
                        template="Answer with a single letter (A, B, C, or D).\n\n"
                                 "Question: {{ question }}\n"
                                 "A. {{ choices[0] }}\nB. {{ choices[1] }}\n"
                                 "C. {{ choices[2] }}\nD. {{ choices[3] }}\n\nAnswer:"),
                    target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))
# The dataset's identity + rendering live on DATASET; the dev/test split sizes and
# the weights checkpoint live on the DataSource (the seam resolution).
DATA_SOURCE = DataSource(
    dataset=DATASET,
    split=SplitSpec(dev=Count(value=5), test=Count(value=10)))

# Reference-relative verification: ReferenceRelativeVerifier runs the same oracle
# model over the artifact's inputs and compares the artifact's mean letter-accuracy
# against the oracle's, both scored versus the dataset ground truth. It requires the
# artifact's mean to be within `tol` of the oracle's. The oracle is held inside the
# verifier (loaded when the verify stage loads the verifier's engine).
# The OutputLengthCap is the single source of truth for the per-request decoding budget:
# the driver stamps it onto every request envelope, the wire protocol forwards it as
# `max_tokens` in the OpenAI request body, and the same value bounds the oracle's
# `generate(max_new_tokens=...)` call. Without this equality the reference-relative
# comparison runs the two sides under different caps and the verdict is meaningless.
WORKLOAD = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=MAX_NEW_TOKENS),
                    slo=NoSLO())
SCORER = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))

# Serving frameworks and pre-built kernels a build sandbox denies, so the contestant
# builds its serving engine on the image's torch / triton / CUDA toolkit and its own
# kernels rather than `pip install vllm`. This is the harness list
# (hls.egress.denied_packages), which the default producer network also denies. The
# text tasks that pass their own network (this one, tab_completion, multitenant_slo)
# import it from here.
_DENIED_SERVING_FRAMEWORKS = list(DENIED_PACKAGES)

# Offline toolchain documentation for every language and library a build image ships
# -- PTX, CUDA runtime/driver, CUDA Python, Triton, cuTile, CUTLASS/CuTe, CCCL,
# cuBLAS, cuDNN, NVRTC, cuSPARSE/cuSOLVER/cuFFT, torch, Rust, profiling. Uploaded
# rather than baked: ~105MB of docs do not belong in an image layer, and re-pinning a
# doc set should not mean rebuilding the image. Shared by every task that uploads
# this reference set (tab_completion, multitenant_slo, multimodal_tts) alongside their own
# `AgentProducerSpec(upload_dir=("../assets/references",),
# upload_dest="/workspace/references")`. The folder is not committed (git ignores
# tasks/assets/); README.md's walkthrough says where it comes from, and its own
# README.md indexes it.
#
# Landed in the working tree (not the /task/files staging path) so the agent greps it
# exactly as it greps its own source, and the brief can cite `references/<lib>/` as a
# relative path. `docs/` is left free so the agent has somewhere of its own to write.
_TOOLCHAIN_UPLOAD_BRIEF = (
    "## Toolchain Documentation\n"
    "You are offline. Documentation for every installed language, compiler and\n"
    "library is on disk -- `references/README.md` indexes it, and the per-library paths\n"
    "are listed under ## Build Instructions above. There is no other reference\n"
    "available to you, so consult these rather than relying on recall."
)
# The tree digest (hls.agent.uploads.tree_digest) of that reference set, which the folder is
# gitignored and too large to commit: the 2,565 files scripts/assemble_references.py builds from
# public sources (hls/agent/toolchain_references/manifest.tsv lists them), which the build checks
# before it uploads the folder.
_TOOLCHAIN_REFERENCES_SHA256 = "2458984914b96a16fbe591a48e412626599ea7a55782ae7b728d740c66636ca5"

CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
    transport="http")

VERIFIERS = [ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=HFReference(decode=Greedy()), tol=0.0)]

# No baseline: a goodput task is scored against its SoL, not against a measured server.
TASK = TaskSpec(id="sample_task", model=MODEL, hardware=HARDWARE,
                contract=CONTRACT, workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                # Its drives, which the drive and evaluate tests drive or plan, are kilobytes,
                # so its trace gets the least cap (hls.drive.bounds). Its
                # verify role keeps about 1.11 bytes per trace byte, as a reference-relative gate
                # does; its reference is the tests' own and holds no host memory to speak of.
                drive_bounds=bounds_for(CONTRACT, trace_bytes=0, held_reply_bytes=0, trace_request_bytes=2**20,
                                        verify_peak=VerifyPeak(base_bytes=0, bytes_per_byte=1.11)),
                metadata={"theme": "sample"})

# Default contestant: the live implementer agent (pure config). The arm names the model
# (`hls run --arm`). The agent ships a runnable repo (pyproject.toml + server.py), which the
# package stage builds the submission image from. serve_command is extra args only;
# host/port derive from CONTRACT.port (8000 default).
ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        # Hard budget (harness-enforced, invisible to the agent): wall-clock + turn caps.
        hard_budget=Budget(wallclock_sec=3600, turns=40),
        # Soft budget, which the brief states and the agent reads what is left of with
        # get_budget: 30 minutes, half the hard hour.
        soft_budget=Budget(wallclock_sec=1800),
        # The provider's top level for whatever harness and model an arm launches with,
        # resolved at launch from the harness's top_effort table and recorded in the
        # run's config.json. Never implicit: the harnesses' own defaults differ and go
        # unrecorded, and the run path refuses to launch without a value.
        reasoning_effort=HighestAvailable(),
        # What a grading holds with the task's GPU: 8 CPUs, 28 GiB for the server's job and 4 GiB
        # for the driver, which runs on the server's CPUs, and 32 GiB the server may write. The
        # harness's verify and package stages take 8 CPUs whatever a task declares, so fewer for
        # the server would not let a run fit a smaller machine. The verify role gets 16 GiB: its
        # reference is the same 1 GB model, beside a trace of at most 1 GiB.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=28, writable_gib=32, driver_memory_gib=4,
                                                   verify_memory_gib=16),
        # What a build holds with the task's GPU: 8 CPUs and 32 GiB. The sandbox's writable disk
        # cap is as much as its memory, and it bounds the repository the agent commits.
        build_allocation=BuildAllocation(cpus=8, memory_gib=32),
        # Producer egress policy: the agent reaches its model API through the arm's route and
        # the vendored-build pip wheels resolve through PyPI, behind the build seal. The
        # package policy sends pip through the seal's index, which refuses the denied names.
        network=NetworkPolicy(
            domains=AccessList(mode="allow", entries=[
                "pypi.org", "*.pypi.org", "files.pythonhosted.org", "pythonhosted.org"]),
            packages=AccessList(mode="deny", entries=_DENIED_SERVING_FRAMEWORKS)),
    ),
    brief=ImplementerBrief(),
    serve_command=[])

# The images the task runs on: the shared layer its sandbox, submission base and harness build on.
IMAGES = ImageSet(layer="s-a-tf-5-13-1")

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "SEED",
           "MODEL", "HARDWARE", "DATASET", "DATA_SOURCE", "VERIFIERS", "WORKLOAD", "SCORER"]
