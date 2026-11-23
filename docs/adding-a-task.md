# Adding a task

A task is a Python module that tells HLS what to benchmark: the model, the hardware, the
requests, the HTTP endpoint the server must expose, the correctness check, the score, the
agent's budget and the container images. Each task is a directory under `tasks/` that holds a
package with a `task.py`, and HLS loads it as the module `tasks.<directory>.task`.

The quickest start is a copy of the sample task:

```sh
cp -r tasks/sample_task tasks/my_task
```

Change the task's `id` in `tasks/my_task/task.py` and commit the directory. The task then runs
like any other (`--arm` names the agent and model from your `hls.toml`, as
[running.md](running.md) describes):

```sh
uv run hls run tasks.my_task.task --arm claude
```

HLS plans a scored run only from a checkout with no uncommitted change, so commit before you
run. It loads only the module named `task` in a task's directory, and the directory's name is
lowercase letters, digits and underscores, starting with a letter. A task module holds
declarations only: it reads no environment variable and starts nothing.

## What task.py declares

`tasks/sample_task/task.py` declares each part below as a module constant. It then combines
them into the three constants HLS reads: `TASK`, the task itself; `ARTIFACT`, what the agent
gets to build with; and `IMAGES`, the container images.

### Model and hardware

```python
MODEL = ModelSpec(base_checkpoint="Qwen/Qwen2.5-0.5B-Instruct",
                  revision="7ae557604adf67be50417f59c2c2f167def9a775", arch="TX", chat_format="chatml")
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1),
                    cpu_count=8, memory_gb=32, storage_gb=32)
```

`base_checkpoint` and `revision` name a Hugging Face repository at one commit. `revision` must
be a commit id; provisioning refuses a branch name. `hls provision` downloads the weights into
the store, the directory where HLS keeps images, weights, data and results. The agent's sandbox
and the served submission both see the weights read-only at `$HLS_WEIGHTS_DIR`. `arch` is the
model's architecture family (`TX`, `MoE`, `HYB` and others). `chat_format` is the chat format
named in the brief, the instructions HLS renders for the agent from the task's declarations.

`HARDWARE` is the machine the brief describes to the agent. Its GPU kind is one of `H200`,
`H100`, `B200`, `A100`, `L40` and `L40S`.

### Dataset and splits

```python
DATASET = HFDataset(path="cais/mmlu", subset="abstract_algebra",
                    revision="c30699e8356da336a370243923dbaf21066bb9fe", source_split="test",
                    prompt=JinjaPromptTemplate(
                        template="Answer with a single letter (A, B, C, or D).\n\n"
                                 "Question: {{ question }}\n"
                                 "A. {{ choices[0] }}\nB. {{ choices[1] }}\n"
                                 "C. {{ choices[2] }}\nD. {{ choices[3] }}\n\nAnswer:"),
                    target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))
DATA_SOURCE = DataSource(dataset=DATASET, split=SplitSpec(dev=Count(value=5), test=Count(value=10)))
```

The dataset renders each row into a prompt and a ground-truth target with Jinja templates. The
split sets how many rows the agent gets on disk in its sandbox (`dev`) and how many rows the
submission is scored on (`test`). Each is a `Count` of rows or a `Fraction` of the dataset. The
agent never sees the test rows. HLS uses them when it grades a submission: a grading serves the
submission, sends it the test requests, checks the replies and computes the score.

### Traffic and latency budget

```python
WORKLOAD = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=4), slo=NoSLO())
```

The workload describes the traffic. Requests come in sessions, and each turn of a session is
one request.

- `arrival` says when sessions arrive: `AllAtOnce`, `Poisson`, `Gamma` and the others in
  `hls.spec.workload`.
- `inter_turn_delay` says how long a session waits between its turns.
- `output_length_cap` is the most tokens a reply may hold. HLS sends it as `max_tokens` on every
  request.
- `slo` is the latency budget, one of these:

| SLO | Budget |
|---|---|
| `NoSLO()` | None. For a goodput task. |
| `StreamingLatencySLO(ttft_ms=, tpot_ms=)` | Time to first token and time per output token, per request. Needs a streaming contract. |
| `ContextLinearLatencySLO(ttft_ms=, tpot_ms=, ttft_ms_per_prefill_token=)` | As `StreamingLatencySLO`, with a time-to-first-token budget that grows by `ttft_ms_per_prefill_token` for each prompt token. |
| `RequestLatencySLO(target_ms=)` | Time from request to complete reply. |
| `TrajectoryMakespanSLO(target_sec=)` | Time from a session's first request to its last reply. |

A task that mixes several kinds of traffic declares `tenants` in place of the single
`workload`, `verifiers` and `data_source`. `tenants` is a list of `Tenant`, each with its own
`id`, data source, workload and verifiers. `tasks/multitenant_slo/task.py` is an example.

### The HTTP endpoint

```python
CONTRACT = RequestResponseContract(path="/v1/chat/completions",
                                   input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}), transport="http")
```

The contract names the endpoint the submission serves and the protocol its requests and
replies follow. [wire-protocol.md](wire-protocol.md) lists the protocols and what each requires
of a reply.

### Correctness checks

```python
VERIFIERS = [ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                       reference=HFReference(decode=Greedy()), tol=0.0)]
```

The verifiers are the gate: the correctness check a run must pass to score above zero. A run
passes only when every verifier passes. `hls.verify` provides these:

| Verifier | Passes when |
|---|---|
| `MetricVerifier(metric, min_value)` | The submission's mean metric against the dataset's targets reaches `min_value`. |
| `ReferenceRelativeVerifier(metric, reference, tol)` | The submission's mean metric is within `tol` of the reference's on the same requests. With `min_ratio` in place of `tol`, it is at least that fraction of the reference's. |
| `TokenFaithfulnessVerifier(reference, rate_multiplier)` | Under greedy decoding, the served tokens stay within the calibrated ceilings when the reference scores them. Needs a calibration and a protocol with token ids. |
| `ReportFaithfulnessVerifier(reference)` | Under sampling, the distributions the server reports match the reference's, and the served tokens are draws from them. Needs a calibration and a protocol with logprobs. |
| `CodecFidelityVerifier(reference)` | The streamed audio matches the reference codec's decode of the codec codes the server reports, in length and in log-mel PSNR. Needs a protocol that returns codec codes. |

`hls.verify` also holds the metrics, such as `LetterMatchMetric`, `ContainsMatchMetric`,
`CoqaF1Metric`, `WERMetric` and `MixedModalityMatchMetric`.

The reference is HLS's own run of the model from the same checkpoint:
`HFReference(decode=Greedy())` or `HFReference(decode=Sample(temperature=1.0))`.
`VLLMReference` runs it on vLLM: it answers faster for a metric verifier, and it fits a context
of hundreds of thousands of tokens.

A calibrated verifier reads its ceilings from the task's calibration, which
[Measuring the baseline credit and the calibration](#measuring-the-baseline-credit-and-the-calibration)
covers. The brief tells the agent to decode the way a calibrated verifier's reference does, and
greedily when the task has no calibrated verifier.

### How the run is scored

```python
SCORER = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
```

`GoodputScorer` scores goodput: correctly answered requests per second, divided by the ceiling
that `sol` sets (here a constant 1 request per second). `SloLatencyScorer()` scores the speedup over a baseline server in latency relative
to the SLO, and needs [a baseline](#declaring-a-baseline-server). [scoring.md](scoring.md) has
the formulas.

### Combining the parts into TASK

```python
TASK = TaskSpec(id="sample_task", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
                workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                drive_bounds=bounds_for(CONTRACT, trace_bytes=0, held_reply_bytes=0,
                                        trace_request_bytes=2**20,
                                        verify_peak=VerifyPeak(base_bytes=0, bytes_per_byte=1.11)))
```

`id` names the task in the store and in `[resources.<task id>.<stage>]` of `hls.toml`. It may
differ from the directory's name: `tasks/multimodal_asr` holds the task `multimodal-asr`.

`drive_bounds` is required, and HLS refuses to grade a task without it. A drive is one pass of
the test requests through the server, and its trace is the record of every request and reply.
`drive_bounds` caps the size of that trace:

| Argument of `bounds_for` | What it is |
|---|---|
| `trace_bytes` | The size of the largest trace a drive of the task has written. 0 gives the smallest cap, 1 GiB. |
| `held_reply_bytes` | The most reply bytes the drive held at once, received but not yet written to the trace. |
| `trace_request_bytes` | The bytes of all the drive's requests. |
| `verify_peak` | The memory the verify stage (the step of a grading that checks the replies) holds: `base_bytes`, plus `bytes_per_byte` for each byte of trace. |

HLS sets the cap to at least four times `trace_bytes`, rounded up to whole GiB, unless the verify
stage's memory can hold less. Once a drive's trace reaches the cap, every further reply fails as
a failed request.

### What the agent gets (ARTIFACT)

```python
ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        soft_budget=Budget(wallclock_sec=1800),
        hard_budget=Budget(wallclock_sec=3600, turns=40),
        reasoning_effort=HighestAvailable(),
        build_allocation=BuildAllocation(cpus=8, memory_gib=32),
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=28, writable_gib=32,
                                                   driver_memory_gib=4, verify_memory_gib=16),
        network=NetworkPolicy(
            domains=AccessList(mode="allow", entries=["pypi.org", "*.pypi.org",
                                                      "files.pythonhosted.org", "pythonhosted.org"]),
            packages=AccessList(mode="deny", entries=list(DENIED_PACKAGES)))),
    brief=ImplementerBrief(),
    serve_command=[])
```

`ARTIFACT` describes the agent's build: its budgets, its machine, its network and its brief, and
the machine its submission is graded on.

The soft budget is the time the brief tells the agent it has, and the agent can ask how much of
it is left. The hard budget is when HLS stops the agent: `wallclock_sec` is required, and
`turns` caps the agent's turns. `reasoning_effort=HighestAvailable()` asks for the highest
effort level the provider offers for the arm's agent and model; `ProviderLevel(level=...)`
names one level instead.

`build_allocation` is the sandbox's CPUs and memory, beside the task's GPUs. The sandbox may
write as much disk as it has memory.

`evaluation_allocation` is what a grading gives the submission in place of `HARDWARE`: the
server's CPUs (`cpus`), the memory of its job (`job_memory_gib`), the disk it may write
(`writable_gib`) and the memory of the driver, the HLS process that sends the requests
(`driver_memory_gib`). `verify_memory_gib` sets the memory of the grading's verify stage, for a
task whose reference is small. Without it HLS gives that stage 180 GiB.

`network` lists the hosts the sandbox may reach and the packages its package index refuses.
`DENIED_PACKAGES` is HLS's list of serving engines and prebuilt kernels. The agent reaches its
model's API whatever the list says. Without a `network`, the sandbox reaches only the package
index, which refuses `DENIED_PACKAGES`. A task that sets its own `network` denies that list
too, as the sample does.

`upload_dir` names folders the build copies into the agent's workspace, merged at
`upload_dest` (`/task/files` by default). It needs `upload_sha256`, the folders' digest as
`hls.agent.uploads.merged_digest` computes it, and `upload_brief`, the text that tells the
agent what the folders hold. Four shipped tasks upload the offline toolchain documentation this
way, from `tasks/assets/references`, which `uv run python scripts/assemble_references.py`
assembles.

`ImplementerBrief()` renders the brief from the task's declarations. To change a section, pass
`template_str`: a Jinja template that starts with `{% extends "brief" %}` and overrides a block
(`goal`, `model`, `hardware`, `contract`, `decoding` or `build_instructions`).
`tasks/multimodal_asr/task.py` overrides its goal this way.

`serve_command` holds extra arguments the submission's server starts with, after `--host` and
`--port`.

## Declaring a baseline server

A task scored with `SloLatencyScorer` declares the serving stack it is scored against, such as
stock vLLM:

```python
BASELINE = Baseline(image="hls-baseline-my-task",
                    serve_command=["--served-model-name", "served-model", "--dtype", "bfloat16"])
```

Pass it to the task as `TaskSpec(..., baseline=BASELINE)`. `image` is the baseline's name in the
records; the image HLS serves is `IMAGES.baseline` ([below](#container-images)).
`serve_command` holds the flags appended to the server's start command.

The baseline credit B is the baseline's own latency measure on the task's workload, and a
submission's score is its speedup over it ([scoring.md](scoring.md)). Both `image` and
`serve_command` are part of the credit's identity, so after a change to either, HLS measures
the credit again.

## Container images

`IMAGES` declares the container images the task runs on, and HLS builds each one from a pinned
recipe. The sample needs one line:

```python
IMAGES = ImageSet(layer="s-a-tf-5-13-1")
```

`layer` is the shared image that the agent's sandbox, the submission and HLS's own containers
build on. It holds the toolchain the brief lists (PyTorch, the compilers, the CUDA toolkit) with one
version of `transformers`. `hls.recipes.catalog.LAYERS` lists the layers.

A task with a baseline also declares the images its own servers run on. These are server
images: they may hold a serving engine, and no agent works in one.

```python
from hls.recipes.catalog import SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage

_VLLM = LayerImage(name="my-task-baseline", parent="cuda", pip_lock="baseline.txt",
                   owner="my_task", server=True, env=SERVER_ENV)
_REFERENCE = LayerImage(name="my-task-reference", parent="cuda", pip_lock="reference.txt",
                        owner="my_task", server=True, env=SERVER_ENV)
IMAGES = ImageSet(
    layer="s-a-tf-5-13-1",
    baseline=Served(image=_VLLM,
                    entrypoint=["sh", "-c", 'exec vllm serve "$HLS_WEIGHTS_DIR" "$@"', "vllm-baseline"]),
    reference=_REFERENCE,
    members={"vllm": _VLLM},
)
```

| Field | Image |
|---|---|
| `baseline` | The baseline server and the command that starts it. HLS appends `--host`, `--port` and the task's `Baseline.serve_command`. |
| `reference` | Where the gate's reference runs. Without it, the reference runs on HLS's own image, which builds on the task's layer. |
| `members` | The serving stacks a calibration serves, by engine name. |

Each server image installs from a hashed lock in the task's own directory. List the packages,
each pinned to a version, in `tasks/my_task/locks/baseline.in`, then compile
`tasks/my_task/locks/baseline.txt` from it:

```sh
uv run hls images lock-pip layer baseline --task my_task --server
```

A run builds the images it lacks. To build them ahead of the first run:

```sh
uv run hls images build my_task
```

## Other files in the task directory

| Path | What it is |
|---|---|
| `__init__.py` | Makes the directory a package. |
| `locks/*.in`, `locks/*.txt` | The pip inputs and hashed locks of the task's own images. |
| `defaults/baseline.json` | The baseline credit shipped with the task. |
| `defaults/calibration.json` | The calibration shipped with the task. |
| `defaults/dev_reference.json` | The reference's answers on the dev split, which the agent is shown. |
| `tests/` | The task's own tests, run by path: `uv run pytest tasks/my_task/tests`. |

The files under `defaults/` are measurements. A run uses the store's copy when the store holds
one that fits the task, and the shipped copy otherwise.

## Measuring the baseline credit and the calibration

A task scored against a baseline needs its baseline credit, and a task with a calibrated
verifier needs a calibration. The calibration records how far trusted serving stacks (vLLM and
SGLang) sit from the reference on the task's workload, and the verifier reads its ceilings from
it. `hls run` measures whatever a task lacks before the agent starts building. To measure ahead
of time, or again after the workload changes:

```sh
uv run hls provision tasks.my_task.task
uv run hls grade tasks.my_task.task --measured baseline
uv run hls baseline-credit tasks.my_task.task --drives 3
uv run hls calibrate tasks.my_task.task --engines vllm
uv run hls oracle fill tasks.my_task.task
```

| Command | What it does |
|---|---|
| `hls provision` | Downloads the weights, renders the dataset rows and fetches any tool bundles into the store. |
| `hls grade --measured baseline` | Serves the baseline as if it were a submission, sends it the workload and runs the gate on it. The baseline must pass the task's own gate before its credit means anything. |
| `hls baseline-credit` | Serves the baseline, sends it the workload `--drives` times and writes `baseline.json` into the store under `measured/<task id>/`. |
| `hls calibrate` | Serves the workload on each engine named in `--engines`, has the reference score each one and writes `calibration.json` beside `baseline.json`. Only a task with a calibrated verifier needs it. |
| `hls oracle fill` | Computes the answers of a reference declared with `cache=True` into the store, so gradings read them there. For a task whose data source declares a dev reference, it also writes `dev_reference.json` beside `baseline.json` when neither the store nor `defaults/` holds one that fits. |

`hls capped-requests tasks.my_task.task --from calibrations/<task id>/<id>` lists the sessions
in which a calibration engine (vLLM, unless `--engine` names another) ran to the output cap.
To leave those rows out of the task, list them in its data source's `removed_rows`.

To ship a measurement with the task, copy the file from the store into
`tasks/my_task/defaults/` and commit it.

## Checklist before the first run

- `tasks/<directory>/__init__.py` and `task.py` exist, and `task.py` defines `TASK`, `ARTIFACT`
  and `IMAGES`.
- `TASK.drive_bounds` is set.
- The module imports. When HLS loads a task, it checks the contract against the verifiers and
  the SLO and names any mismatch.
- `uv run hls images build <directory>` builds every image.
- A task scored against a baseline declares `Baseline` and `IMAGES.baseline`, and
  `hls grade --measured baseline` passes the gate.
- `hls baseline-credit` and, for a calibrated verifier, `hls calibrate` have written their
  files, and the ones you ship are under `defaults/`.
- The rendered brief reads correctly. `uv run python -m tests.instances.golden` writes it to
  `tests/instances/golden/<directory>.brief.md`, beside the task's other recorded facts, and
  `uv run pytest tests/instances` catches later changes to them.
- `uv run hls run tasks.<directory>.task --arm <arm>` ends with a score, and the transcript in
  `hls view` shows the agent testing its server on the dev split.
