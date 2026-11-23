# Tasks

Each task asks an agent to build an inference server for one model, one GPU setup and one
workload. The agent works in a sandbox that has the task's GPUs. The sandbox reaches only the
agent's model API and a Python package index, and that index refuses existing serving engines
(vLLM, SGLang, TensorRT and the like), so the agent writes its own engine on PyTorch, Triton and
the CUDA toolchain. When the agent finishes or its time runs out, HLS deploys the server the
agent committed in a fresh container and sends the task's workload through it. It checks the
replies against a reference implementation of the model and scores how fast the server
answered.

The correctness check is called the gate, and a run that fails it scores zero. Every task except
`sample_task` is scored as a speedup over its baseline: a production serving stack (vLLM or
vLLM-Omni) driven with the same workload. [scoring.md](scoring.md) says what the score means.

A task is named by its Python module, `tasks.<directory>.task`. Run one with an arm from your
`hls.toml`, where an arm is the agent that builds the server:

```sh
uv run hls run tasks.sample_task.task --arm claude
```

[running.md](running.md) describes arms, what the command does and where the results are.

| Module | Model | The agent builds | A run needs: GPUs, CPUs, memory, local disk | Baseline |
|---|---|---|---|---|
| `tasks.sample_task.task` | `Qwen/Qwen2.5-0.5B-Instruct` | A chat server for ten multiple-choice questions | 1 GPU, 8 CPUs, 68 GiB, 225 GiB | None; the score is correct answers per second |
| `tasks.multitenant_slo.task` | `Qwen/Qwen3.6-27B` | A streaming chat server for four tenants with different latency budgets | 1 H200, 8 CPUs, 196 GiB, 330 GiB | vLLM |
| `tasks.tab_completion.task` | `JetBrains/Mellum2-12B-A2.5B-Base` | A streaming code completion server | 1 H200, 8 CPUs, 196 GiB, 330 GiB | vLLM |
| `tasks.multimodal_asr.task` | `openai/whisper-large-v3` | A speech transcription server | 1 H200, 8 CPUs, 196 GiB, 326 GiB | vLLM |
| `tasks.multimodal_tts.task` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | A streaming text-to-speech server | 1 H200, 8 CPUs, 272 GiB, 500 GiB | vLLM-Omni |
| `tasks.multimodal_omni.task` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | A chat server that answers questions about images and translates speech | 2 H200, 8 CPUs, 196 GiB, 328 GiB | vLLM-Omni |
| `tasks.deepresearch.task` | `google/gemma-4-31B-it` | A streaming chat server for a research agent that calls search tools | 2 H200, 16 CPUs, 248 GiB, 324 GiB | vLLM |
| `tasks.agentic_coding.task` | `zai-org/GLM-5.3-Flash` | A streaming chat server for a coding agent that works in a shell | 8 H200, 64 CPUs, 1,911 GiB, 3,997 GiB | vLLM |

## What a run needs

A run of `hls run` holds the resources in the table in one allocation, from its first step to
its last. An allocation is one Slurm job, or one job of HLS's own scheduler on a Docker host. Its
GPUs, CPUs and memory are the most that any one stage of the run uses; a stage is one step of the
run, such as the agent's build or the timed drive of the workload. The local disk is the free
space it needs under `site.node_scratch`.

Every task except `agentic_coding` runs on either a Slurm cluster with Apptainer or a Docker
host. `agentic_coding` needs a Slurm cluster ([agentic_coding](#agentic_coding)).

To change what one stage asks for, add a `[resources.<task id>.<stage>]` table to `hls.toml`
(see [configuration.md](configuration.md)). The task id is the directory name, except for the
three `multimodal` tasks, whose ids have hyphens in place of underscores (`multimodal-asr`).

## How long the agent has

Every task except `sample_task` tells the agent it has 12 hours of wall-clock time and stops it
30 minutes after that. `sample_task` tells the agent it has 30 minutes and stops it at one hour.
For one run, `--soft-sec` (the time the agent is told) and `--hard-sec` (when HLS stops it) on
`hls run` replace both.

## Toolchain documentation for four tasks

`multitenant_slo`, `tab_completion`, `multimodal_tts` and `agentic_coding` upload offline
documentation of the toolchain into the agent's workspace. Assemble it once per checkout, into
`tasks/assets/references`, before the first run of any of them:

```sh
uv run python scripts/assemble_references.py
```

## The task description the agent reads

HLS writes each task's description for the agent from the task module. This description, the
brief, is the only statement of the task the agent sees. Each task's brief is in the repository
as `tests/instances/golden/<directory>.brief.md`. Read it for the exact request and reply
formats, the decoding rules and the limits of the sandbox.

## sample_task

`sample_task` is a small task for a first run, and the worked example of
[adding-a-task.md](adding-a-task.md). The agent builds a chat server for `Qwen/Qwen2.5-0.5B-Instruct` and has 30 minutes.

The server answers `POST /v1/chat/completions`. The workload is ten multiple-choice questions
from `cais/mmlu` (abstract algebra), all sent at once, each capped at 4 output tokens.

The gate passes when the server's letter accuracy is at least that of a greedy Hugging Face
decode of the same model on the same questions. The score is the number of correctly answered
requests per second; this task has no baseline.

```sh
uv run hls run tasks.sample_task.task --arm claude
```

A run needs one GPU, 8 CPUs, 68 GiB of memory and 225 GiB free on `site.node_scratch`. The task
declares an H200. On a site with another GPU, give its kind after the task module, one of
`A100`, `H100`, `B200`, `L40` and `L40S`:

```sh
uv run hls run tasks.sample_task.task --arm claude --task.hardware.accelerator.kind=L40S
```

## multitenant_slo

The agent builds a streaming chat server for `Qwen/Qwen3.6-27B` on one H200. The server decodes
greedily and takes traffic from four tenants at once. Each tenant is one kind of client with its
own dataset:

- open-ended chat (`allenai/WildChat-1M`);
- short function-calling requests (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`);
- math reasoning (`madrylab/gsm8k-platinum`);
- long multi-turn sessions (`openai/mrcr`).

Sessions arrive on a fixed schedule, 1.11 requests per second in total. Every turn of a session
is its own request, carrying the conversation so far. Each tenant has its own budget for the time
to first token (TTFT), from 300 to 1,000 ms, and every tenant has 50 ms per output token (TPOT).

The gate feeds the exact token ids the server returned through a bf16 Hugging Face reference of
the model. The run fails when too many of them are tokens the reference ranks below its own first
choice.

```sh
uv run hls run tasks.multitenant_slo.task --arm claude
```

## tab_completion

The agent builds a code completion server for `JetBrains/Mellum2-12B-A2.5B-Base` on one H200.
The server answers `POST /v1/completions` with streamed fill-in-the-middle completions, decoded
greedily and capped at 256 tokens. The prompt arrives assembled, with the model's `<fim_prefix>`,
`<fim_suffix>` and `<fim_middle>` markers in it.

Four tenants send the load. Two replay editing sessions from `gonglinyuan/real_fim_eval`: a
developer accepts a completion and types a little, and the editor sends nearly the same prompt
with a longer prefix. The other two send single algorithmic completions from
`gonglinyuan/safim`. The load is 24.6 requests per second in total. The budgets are 200 to 600 ms
to the first token and 10 ms per token after it.

The gate is the token check of `multitenant_slo`, and it also requires each reply to stop where
the reference stops.

```sh
uv run hls run tasks.tab_completion.task --arm claude
```

## multimodal_asr

The agent builds a speech transcription server for `openai/whisper-large-v3` on one H200. The
server answers `POST /v1/audio/transcriptions`. Each request carries one English speech clip as a
base64 WAV, and the reply is its transcript. The workload sends 2,400 LibriSpeech clips
(`hf-audio/open-asr-leaderboard`) at 52 requests per second. Each request has a budget of 6
seconds from sending it to the complete reply.

The gate compares transcription accuracy, which is 1 minus the word error rate. The server's
mean accuracy must be at least 98% of what a greedy bf16 Hugging Face decode of the same clips
scores.

```sh
uv run hls run tasks.multimodal_asr.task --arm claude
```

## multimodal_tts

The agent builds a streaming text-to-speech server for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` on
one H200. The server answers `POST /v1/audio/speech` with a stream of audio frames for one
sentence, in one fixed voice, sampled at temperature 1. After the audio, each reply sends the
codec codes the audio was decoded from, and the probabilities the server sampled them from.

The workload sends 2,400 sentences from `mythicinfinity/libritts` at 2 requests per second, with
no warm-up requests. Each request has a budget of 10 seconds to its last audio frame.

The gate has two parts. The reported probabilities must match the reference model's on the same
prefix, and the served codes must be draws from them. The audio must match the reference codec's
decode of the same codes.

```sh
uv run hls run tasks.multimodal_tts.task --arm claude
```

## multimodal_omni

The agent builds a chat server for `Qwen/Qwen3-Omni-30B-A3B-Instruct` on two H200s. The server
answers `POST /v1/chat/completions` for two kinds of request, decoded greedily. Half carry an
image and a question about it (`lmms-lab/VQAv2`). The other half carry French speech to
translate into English (FLEURS). The workload sends 4,600 requests at 30 per second, and each has
a budget of 15 seconds from sending it to the complete reply.

The gate compares the server with a bf16 Hugging Face decode of the same requests. Word F1 on
the image answers and sentence BLEU on the translations must each reach 98% of the reference's.

```sh
uv run hls run tasks.multimodal_omni.task --arm claude
```

## deepresearch

The agent builds a streaming chat server for `google/gemma-4-31B-it` on one H200. HLS runs a
research agent on the served model, and the run's second H200 runs the retriever its searches
use.

The research agent answers 50 questions, one trajectory per question. On each turn it sends a
streamed chat completion with three tool definitions (`web_search`, `web_open`, `web_find`). It
runs the tool call in the reply against a BrowseComp-Plus corpus, then sends the result back with
the conversation so far. A trajectory takes about 18 completions at the median. Prompts grow to
tens of thousands of tokens and share a long prefix from turn to turn. Each trajectory is scored
on its makespan, the time from its first request to its last reply, relative to 1,800 seconds.

The server samples at temperature 1. For every token it reports the token's log-probability and
the 21 most probable alternatives. The gate compares those reported distributions with the
reference model's on the same prefix, and checks that the served tokens are draws from them.

```sh
uv run hls run tasks.deepresearch.task --arm claude
```

## agentic_coding

The agent builds a streaming chat server for `zai-org/GLM-5.3-Flash` on one node of eight H200s,
at the model's full context window of 1,048,576 tokens. HLS runs a coding agent on the served
model. Each of 30 sessions is one Terminal-Bench 4.0 task. On every turn the coding agent sends
the whole conversation as a streamed chat completion. HLS runs the shell command in the reply in
that session's container, and the command's output joins the conversation.

Sessions arrive within the first 15 minutes. At most 24 requests are in flight at once, and HLS
issues requests for 45 minutes. Six sessions start from a saved conversation of 200,000 to
600,000 prompt tokens. Each request has a TTFT budget of 1,400 ms plus 0.008 ms per prompt token,
and a budget of 20 ms per output token.

The server samples at temperature 1 and reports log-probabilities with 21 alternatives. The gate
is the distribution check of `deepresearch`, against the model as vLLM serves it on its FP8
weights. Whether the coding tasks get solved is not scored.
[tasks/agentic_coding/README.md](../tasks/agentic_coding/README.md) describes the task in detail.

### What agentic_coding needs before its first run

The task needs a Slurm site whose nodes have Apptainer and `/dev/fuse`. On a Docker site,
planning refuses it.

The task keeps four directories outside the store, called sandbox stores. The
`[sandbox_stores]` table of `hls.toml` names the directory behind each one (see
[configuration.md](configuration.md)):

| Store | Holds | Filled by |
|---|---|---|
| `agentic-coding-images` | One container image per Terminal-Bench task: 55 images, about 30 GB, on the shared filesystem | `hls provision` |
| `agentic-coding-runtime` | The Python runtime that runs the command server inside the containers | `hls provision` |
| `agentic-coding-overlays` | The containers' writes, with 30 GiB free per session in flight | `hls provision` |
| `agentic-coding-prefixes` | The saved conversation and container state of each of the three tasks whose sessions start from a saved conversation | `hls provision` |

Provision the task before its first `hls run` or `hls grade`:

```sh
uv run hls provision tasks.agentic_coding.task
uv run hls run tasks.agentic_coding.task --arm claude
```

`hls provision` fetches Terminal-Bench 4.0 at its pinned commit. It pulls the 55 images the
benchmark publishes with Apptainer, each by its digest. The node it runs on needs a route to
github.com and docker.io, `git`, `uv`, Apptainer and 80 GiB of node-local scratch. Running the
command again pulls only the images the store lacks. The store holds each image as `<task>.sif`
with a record `<task>.json` beside it, and a `.sif` without its record is pulled again. The three
saved conversations ship in the repository under `tasks/agentic_coding/prefixes/`. Provisioning
checks them against the digests the task pins and copies them into their store.

Docker Hub limits anonymous pulls. To pull as a logged-in user, set `site.registry_username` and,
in the secrets file, grant `REGISTRY_PASSWORD` to the stage `sandbox_images` (see
[configuration.md](configuration.md)).
