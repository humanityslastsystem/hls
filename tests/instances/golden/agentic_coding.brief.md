# Task Description

## Goal
Serve `zai-org/GLM-5.3-Flash` under the workload below. The score is a
speedup over the reference stack, `exp(mean ln(1/r) - B)`, where r is a request's
SLO-normalized latency, defined here; higher is better.

**Traffic.** Each session is one coding-agent trajectory. On every turn the grader sends
the whole conversation so far as one chat request, executes the shell command in your
reply, and appends the command's output before the next turn. 30 sessions arrive
as a Poisson process at the rate under Workload, set so that all 30 arrive within the
first 15 minutes; sessions run concurrently, with at most 24 requests in flight
at once. 6 of the sessions start from a pre-rolled conversation, 2 at each of
200,000, 400,000 and 600,000 prompt tokens; those run at most 10 further turns.

**Two requests per turn.** A turn's first request sets `max_tokens` = 32,768. If the
reply reaches that cap, the grader closes it with `</think>` (appended unless the reply
already contains it) and sends a second request with
`max_tokens` = 16,384, `continue_final_message: true`, `add_generation_prompt: false`,
and the first request's messages followed by the reply so far as the final assistant
message. A continuation that reaches its cap ends the turn. Both requests are scored and
verified as ordinary requests.

**Score.** For every request:

    budget = 1,400 ms + 0.008 ms x prompt_tokens
    r      = max( TTFT / budget , mean TPOT / 20 ms )
    credit = ln(1 / r)
    score  = exp( mean credit over all requests - B )

B is the mean credit the reference stack earned on this workload.

**Window.** Requests are issued for 45 minutes; at 55 minutes every open stream
is cut. A request still streaming at the cut is scored as if one more token completed at
the cut instant: a request with no token yet is charged the whole wait as TTFT; one with
n tokens is charged the span from its first token to the cut over n intervals. That token
is not part of the reply and is not verified.

**Failure.** After a request's first token, 120 seconds without a further token fails the
request. Time to first token has no limit. Any failed request, or any request without a
complete streaming response, scores the run 0.
## Model
- Checkpoint: `zai-org/GLM-5.3-Flash`
- Chat format: `glm`
- Context window: 1,048,576 tokens; serve the model at this window.
## Hardware Envelope
8xH200, 141 GB/device, NVLink
2040 GB host RAM
## Workload
Arrivals are sessions. A session's turns are generated as it runs, up to
250, and each turn is one request or two under the continuation rule above, so a
session's request count is not known in advance.
- Sessions: poisson arrivals, 0.05017 sessions/s
- Turns: up to 250 per session
- Output cap: 32,768 tokens on a turn's first request, 16,384 on its continuation
- SLO: TTFT < 1400ms + 0.008ms per prompt token, TPOT < 20ms (per request)
## Network Contract
### GET /health
Return 200 **only** when the model is fully loaded and the inference endpoint can
serve a request immediately. Return non-200 (e.g. 503) at any point before that.
**If you load the model in a background thread, your /health must block on load
completion before returning 200.** Returning 200 early (before the model is ready)
violates the substrate's readiness contract: the harness stops polling /health
the moment it sees 200 and immediately sends inference traffic.

After /health returns 200, the harness sends one probe request to the inference
endpoint below and repeats it until the response status is 2xx, 3xx or exactly 400.
Any other status (404, 405, 422, other 4xx, 5xx) or no response keeps the server not
ready, and the startup timeout fails the run.

### POST /v1/chat/completions
Accept OpenAI chat/completions requests and return chat completion responses.
The harness sends `max_tokens` on every request body; treat it as a binding
per-request output cap. Do not substitute a server-side default when the
request supplies `max_tokens`.

Return a `token_ids` array on each choice (`choices[].token_ids`): the integer
vocabulary IDs of the exact tokens you emitted (NOT a re-tokenization of the
text). This is mandatory on every response.

The same faithfulness applies to the text: `message.content` must be the exact text
of the tokens in `token_ids`, with nothing removed. Do NOT run a reasoning parser or
a tool-call parser -- both move text out of `content` into a separate field, and the
harness reads `content`. Every request sets `tool_choice: "none"`:
render the tool definitions through the chat template and leave the model's tool call
in `content`, in the checkpoint's own syntax. A reply whose `content`
is missing text that its own `token_ids` carry fails the run.
Support streaming: when the request sets `"stream": true`, respond with an OpenAI
SSE stream (`data: {chunk}` lines, `chat.completion.chunk` objects with `delta`,
terminated by `data: [DONE]`). Include the per-chunk `token_ids` array for that
chunk's tokens.
Return logprobs per the OpenAI spec (`choices[].logprobs.content[]`): one entry per
token you generated, in the order you generated them, carrying that token's own
`logprob` and a `top_logprobs` array holding the top 21 alternatives at that
position -- the 21 most probable tokens, whether or not you emitted one of them.
In addition to OpenAI's decoded `token` string, every logprob entry and every alternative MUST carry an integer `token_id`
field (the vocabulary ID of that token); the verifier keys on these IDs.
A reply that omits the logprobs, reports fewer entries than it emitted tokens,
reports fewer than 21 alternatives at a position or omits a `token_id` fails verification.
Request:
```json
{
  "model": "served-model",
  "messages": [
    {
      "role": "user",
      "content": "<prompt>"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "<tool-name>",
        "description": "<what it does>",
        "parameters": {
          "type": "object",
          "properties": {
            "<arg>": {
              "type": "string"
            }
          },
          "required": [
            "<arg>"
          ]
        }
      }
    }
  ],
  "tool_choice": "none",
  "return_token_ids": true,
  "logprobs": true,
  "top_logprobs": 21
}
```
Response (each `top_logprobs` array is abbreviated to 2 entries below; a conforming reply carries 21 at every position):
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Let me look at the repository first.</think>I'll list the directory.<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>   <-- the whole generation, verbatim"
      },
      "finish_reason": "stop",
      "token_ids": [
        1234,
        5678
      ],
      "logprobs": {
        "content": [
          {
            "token": "<tok>",
            "token_id": 1234,
            "logprob": -0.1,
            "bytes": [
              60,
              116
            ],
            "top_logprobs": [
              {
                "token": "<tok>",
                "token_id": 1234,
                "logprob": -0.1
              },
              {
                "token": "<alt>",
                "token_id": 5678,
                "logprob": -1.5
              }
            ]
          }
        ]
      }
    }
  ],
  "usage": {
    "completion_tokens": 2
  }
}
```

## Decoding
**Decoding.** Sample at temperature 1. At each position, apply
the softmax to the model's logits over the full vocabulary and draw the next token from that
distribution. No top-p, top-k or min-p truncation; no repetition, frequency or presence
penalty; no logit bias. The chat-templated prompt already carries the model's special tokens,
so do not add another BOS. A reply ends when the model emits one of its end-of-turn tokens
(`finish_reason` `stop`, with that token as the last token id served) or when `max_tokens`
tokens have been served or the context window is full (`finish_reason` `length`).

For every generated token, report its log-probability and the most probable alternatives at
that position with their log-probabilities and token ids, as the Network Contract above
specifies. The reported values must be the distribution you sampled from.

**Verification.** For every served token, the reference model is run on the same prefix and
the KL divergence from your reported distribution to the reference's is averaged over all
graded tokens. The run fails if this mean exceeds 1.27e-01 nats per token.
Your served tokens must be faithful draws from the distribution you report; this is tested
statistically, and a server that reports one distribution and samples from another fails.
The reported alternatives must be the most probable tokens: the run fails if more than
0.47 percent of your served tokens are not among the alternatives you reported at
their position.

The reference is this checkpoint served by vLLM at commit `a8d1aa9c` on the FP8
weights as shipped, decoding under the rules above. A `transformers` forward pass in bf16
is not the reference. Numerical precision and approximations are yours to choose; the
verification above is the only faithfulness constraint.

The reference distribution at a served position is computed by teacher-forcing your served
tokens through the reference's prefill path; your reported distribution is whatever path
produced your sample. Prefill and decode use different kernels in most stacks, including
the reference; the threshold was calibrated on a serving stack's sampled tokens scored this
way, so it absorbs that difference.

## Build Instructions
Write your server as `/workspace/server.py`, run as
`python /workspace/server.py --host <H> --port <P>`. It is built and evaluated OFFLINE.
Implement it in any of the following languages and toolchains: `/workspace/server.py` need only be the entrypoint, and
may exec or bind whatever you build.

- Weights: a local checkpoint (config + safetensors + tokenizer) mounted read-only at
  `$HLS_WEIGHTS_DIR`. The accelerators are `cuda:0` through `cuda:7`.
- Data: a development split is on disk at `/data/tenants/t0/dev/rows.json`; the scored test split is withheld. Each row is one coding task record: `instruction` (the task
  statement the trajectory opens with), `task_name`, `category`, `workdir`, `source`,
  `expert_time_estimate_hours`, the sandbox fields `cpus`, `memory_mb`, `storage_mb`,
  `gpus`, `allow_internet`, and the draw fields `prefix_digest`, `turn_budget`, `draws`,
  `split`. Nothing else sits beside the rows. The served prompts are the trajectories the
  grader builds from those tasks at run time and are not on disk.
- `pip install`: any package except the blocked ones is installable.
- Blocked (`pip install` refuses these): `vllm`, `vllm-cpu`, `vllm-tpu`, `vllm-rocm`, `nano-vllm`, `sglang`, `sglang-router`, `lmdeploy`, `turbomind`, `tensorrt-llm`, `text-generation-inference`, `optimum-nvidia`, `llama-cpp-python`, `ctransformers`, `exllama`, `exllamav2`, `exllamav3`, `ktransformers`, `kt-kernel`, `lightllm`, `aphrodite-engine`, `mlc-llm`, `mlc-ai`, `mlc-ai-nightly`, `mlc-llm-cu121`, `mlc-llm-cu122`, `mlc-llm-cu124`, `mlc-llm-cu128`, `mlc-llm-nightly-cu121`, `mlc-llm-nightly-cu122`, `mlc-llm-nightly-cu124`, `mlc-llm-nightly-cu128`, `mlc-llm-nightly-cu130`, `mlc-ai-cu121`, `mlc-ai-cu122`, `mlc-ai-nightly-cu121`, `mlc-ai-nightly-cu122`, `mlc-ai-nightly-cu124`, `mlc-ai-nightly-cu128`, `mlc-ai-nightly-cu130`, `deepspeed`, `deepspeed-mii`, `deepspeed-kernels`, `mistral-inference`, `openllm`, `ai-dynamo`, `ai-dynamo-runtime`, `lmcache`, `lightseq`, `fastertransformer`, `ipex-llm`, `petals`, `medusa-llm`, `airllm`, `tokenspeed`, `tokenspeed-kernel`, `tokenspeed-scheduler`, `tokenspeed-triton`, `tokenspeed-deepgemm`, `tokenspeed-flashmla`, `tokenspeed-mla`, `tokenspeed-nightly`, `tokenspeed-proton`, `vllm-omni`, `sglang-omni`, `ctranslate2`, `faster-whisper`, `whisperx`, `whisper-ctranslate2`, `insanely-fast-whisper`, `whisper-s2t`, `wyoming-faster-whisper`, `faster-whisper-server`, `whisper-live`, `whisperlivekit`, `whisper-cpp-python`, `pywhispercpp`, `whispercpp`, `sherpa-onnx`, `funasr`, `tensorrt`, `tensorrt-bindings`, `tensorrt-libs`, `tensorrt-cu11`, `tensorrt-cu11-bindings`, `tensorrt-cu11-libs`, `tensorrt-cu12`, `tensorrt-cu12-bindings`, `tensorrt-cu12-libs`, `tensorrt-cu13`, `tensorrt-cu13-bindings`, `tensorrt-cu13-libs`, `tensorrt-lean`, `tensorrt-lean-bindings`, `tensorrt-lean-libs`, `tensorrt-lean-cu11`, `tensorrt-lean-cu11-bindings`, `tensorrt-lean-cu11-libs`, `tensorrt-lean-cu12`, `tensorrt-lean-cu12-bindings`, `tensorrt-lean-cu12-libs`, `tensorrt-lean-cu13`, `tensorrt-lean-cu13-bindings`, `tensorrt-lean-cu13-libs`, `tensorrt-dispatch`, `tensorrt-dispatch-bindings`, `tensorrt-dispatch-libs`, `tensorrt-dispatch-cu11`, `tensorrt-dispatch-cu11-bindings`, `tensorrt-dispatch-cu11-libs`, `tensorrt-dispatch-cu12`, `tensorrt-dispatch-cu12-bindings`, `tensorrt-dispatch-cu12-libs`, `tensorrt-dispatch-cu13`, `tensorrt-dispatch-cu13-bindings`, `tensorrt-dispatch-cu13-libs`, `tensorrt-rtx`, `tensorrt-rtx-cu12`, `nvidia-tensorrt`, `torch-tensorrt`, `sgl-kernel`, `sglang-kernel`, `vllm-flash-attn`, `flashinfer-cubin`, `flashinfer-jit-cache`, `kernels`, `kernels-community`, `hf-kernels`, `flash-attn`, `flash-attn-3`, `flash-attn3`, `flash-attn-4`, `flash-attn-cute`, `flash-attn-wheels`, `flash-attn-triton`, `flash-attention`, `flashinfer`, `flashinfer-python`, `xformers`, `flash-mla`, `sageattention`, `ring-flash-attn`, `fla-core`, `flash-linear-attention`, `transformer-engine`, `transformer-engine-cu12`, `transformer-engine-cu13`, `transformer-engine-torch`, `transformer-engine-jax`, `deep-gemm`, `auto-gptq`, `autoawq`, `autoawq-kernels`, `gptqmodel`, `marlin`, `marlin-kernels`, `bitblas`, `gemlite`, `flute-kernel`, `humming-kernels`, `fbgemm-gpu`, `fbgemm-gpu-genai`, `fbgemm-gpu-nightly`, `fbgemm-gpu-genai-nightly`, `sonic-moe`, `megablocks`, `grouped-gemm`, `moe-kernels`, `deep-ep`, `triton-kernels`, `liger-kernel`, `quack-kernels`, `unsloth`, `unsloth-zoo`, `causal-conv1d`, `mamba-ssm`.

Your server and its kernels are your own work on the installed toolchain: no third-party
LLM inference engine, serving framework, or attention, GEMM, MoE or linear-attention
kernel library may be installed, vendored or copied, whether or not `pip install`
refuses it.
Submissions are audited against this rule; the Blocked list names the ones already known
and does not bound it.

Installed, with offline documentation at the paths given (relative to your working
directory):

### Python
`torch`, `transformers`, `accelerate`, `datasets`, `fastapi`, `uvicorn`, `pydantic`.

- torch, including `torch.compile`/Inductor and `torch.utils.cpp_extension` -- `references/torch/`

### C / C++
`gcc`/`g++` 13.3, `clang`, `make`, `cmake`, `ninja`, `pkg-config`, `python3-dev`.

`TORCH_CUDA_ARCH_LIST` is set for this GPU; torch's `torch/include` headers and vendored
pybind11 are present, so `torch.utils.cpp_extension.load()` and setuptools
`BuildExtension` build offline.

- `cpp_extension` usage -- `references/torch/`

### Rust
`rustc`, `cargo`.

- The Rust Reference -- `references/rust/reference/`
- The Rust Book -- `references/rust/book/`

### CUDA toolchain
Compilers and binary tools: `nvcc`, `ptxas`, `nvdisasm`, `cuobjdump`.

- PTX ISA -- `references/ptx/` (start at `references/ptx/ptx-isa.md`)
- CUDA Runtime API -- `references/cuda-runtime/`
- CUDA Driver API -- `references/cuda-driver/`
- CUDA Python (`cuda.bindings`, `cuda.core`) -- `references/cuda-python/`
- NVRTC, runtime kernel compilation -- `references/nvrtc/`

Kernel-authoring DSLs (`triton`, `cuda-tile`, `nvidia-cutlass-dsl` are installed):

- Triton, with runnable annotated tutorials -- `references/triton/`
- CUTLASS C++ and the CuTe Python DSL -- `references/cutlass/`
- cuTile -- `references/cutile/`

Math and primitive libraries:

- cuBLAS / cuBLASLt -- `references/cublas/`
- CUB, Thrust, libcu++ (CCCL) -- `references/cccl/`
- cuDNN -- `references/cudnn/`
- cuSPARSE / cuSOLVER / cuFFT -- `references/cusparse/`, `references/cusolver/`, `references/cufft/`
- NCCL is installed

Profiling and debugging: `nsys`, `ncu`, `compute-sanitizer`, `cuda-gdb`.

- Nsight Systems / Nsight Compute / NVTX guides, debugging tools, performance traps --
  `references/profiling/`

## Toolchain Documentation
You are offline. Documentation for every installed language, compiler and
library is on disk -- `references/README.md` indexes it, and the per-library paths
are listed under ## Build Instructions above.

The installed `transformers` package contains an implementation of this architecture
at `transformers/models/glm5_next/modeling_glm5_next.py`. Run in bf16 on this
checkpoint, that implementation is 6.0e-3 nats per token from the reference at
generated positions, measured over 36 prompts of up to 8,192 tokens. It is not the
reference your distributions are measured against; that reference is named
under ## Decoding.

## Reference Logprobs
`references/glm53-reference/` holds reference output for self-checking.
`prompts.jsonl` carries one record per prompt, with its token ids and its text, at
lengths from a few thousand to several hundred thousand tokens. `reference.jsonl`
carries, per prompt, the reference's sampled continuation token ids and, at every
position, two distributions: `decode`, streamed by the reference while it sampled
that continuation, and `forced`, the reference's prefill path teacher-forced on the
same continuation, which is the path the grader uses. Each gives the served token's
log-probability and the 21 most probable alternatives with their token ids and
log-probabilities, at temperature 1. The two differ at the level of kernel noise. The
reference itself cannot be run in your sandbox. Each prompt also carries
`prompt_forced`, the reference's prefill-path top-21 at a recorded sample of prompt
positions -- every position in the last 1,024 tokens of the prompt and every 64th
position before that -- so a prefill kernel can be checked against the reference
directly. `selfcheck.py` reads a file of that shape from your own server and computes
the KL statistic and the top-21 coverage the grader computes, against `forced` by
default, reporting prefill and generated positions separately. `README.md` states the
formats.

These files are available at `/workspace/references` inside your sandbox.

## Reference Scoring Service
During the build, and only then, a service scores token sequences with the grader's
reference: it teacher-forces `prompt_token_ids` followed by `continuation_token_ids`
through the reference named under ## Decoding, on its prefill path with prefix caching
off, which is how the grader scores your served tokens. It serves HTTP/1.0 on the unix
socket at `$HLS_REFERENCE_SOCKET` (`/run/hls-reference/ref.sock`), one request per
connection, one request at a time. It has no TCP port.

    GET  /status  -> {"state": "starting" | "asleep" | "waking" | "awake" | "sleeping" | "unloaded" | "failed",
                      "gpu_free_bytes": [int, ...], "wake_need_bytes": [int, ...], "memory_error": {...} | null,
                      "last_wake_sec": float, "idle_sleep_sec": 120, ...}
    POST /score   {"prompt_token_ids": [int, ...], "continuation_token_ids": [int, ...],
                   "prompt_positions": [int, ...]}
                  -> {"continuation": [{"token_id": int, "logprob": float,
                                        "top21": [[int, float], ...]}, ...],
                      "prompt": [{"position": int, "token_id": int, "logprob": float,
                                  "top21": [[int, float], ...]}, ...],
                      "n_prompt": int, "n_continuation": int, "woke": bool, "score_sec": float}
    POST /sleep, POST /wake  -> the /status object

`continuation[j]` is the reference's distribution at `continuation_token_ids[j]` given
every token before it. `prompt[k]` is the same at the prompt token `prompt_positions[k]`;
`prompt_positions` is optional and each entry lies in [1, len(prompt_token_ids)).
Log-probabilities are at temperature 1 and unrounded; `top21` holds the 21 most
probable token ids, ranked by log-probability, ties by id. A request with a token id
outside the checkpoint's vocabulary, more than 1,048,575 tokens in total, or nothing
to score returns HTTP 400. A 32,709-token sequence scores in 2.5 s awake.

Scores are bit-reproducible for sequences under 2,048 tokens. Beyond that the
reference itself varies run to run: at prompt positions by a median of 0.16 nats per
token (19 percent of positions by more than 1 nat), at generated positions by a
median of 0 and at most 0.62; the grader's reference carries the same variation.

The service starts with your session and takes about 500 s to load the reference
(one that has not loaded within 1,800 s is stopped),
holding up to 125.5 GiB on each GPU while it does. Until it is asleep, `/status` reports
`starting`, and `/score`, `/wake` and `/sleep` return HTTP 503 with `state`. Leave the
GPUs unused until `/status` reports `asleep`: a service that fails to load is stopped,
and its socket then refuses connections for the rest of the build.
Awake it holds 125.5 GiB on each GPU; asleep, 13.9 GiB.
`/score` and `/wake` wake it in 4.1 s. It goes back to sleep 120 s after its
last `/score` or `/wake`, or on `/sleep` in 3.5 s. A wake needs
`wake_need_bytes` (112.4 GiB) free on every GPU; when any GPU has less, `/score`
and `/wake` return HTTP 507 with `gpu_free_bytes` and `wake_need_bytes` and score
nothing: stop or shrink your server and retry.

Your session and the service share the memory stated for your sandbox: asleep the service
keeps a copy of the weights there, about 324 GiB, and when the two together pass that
memory, the kernel ends your session's largest process. A load, a sleep or a wake that memory
is short for changes nothing and returns HTTP 507 with `memory`, `gpu` or `host`; a load short
of GPU memory also returns `gpu_free_bytes` and `gpu_need_bytes`, and `/status` carries the
last shortage as `memory_error`: release that memory and retry. A load short of memory leaves
`/status` at `unloaded` until `POST /wake` loads it again.
## Submission
You MUST work in a git repository at the repo root. Only the latest commit on `main` is evaluated; uncommitted or unmerged work is discarded. Merge to `main` the moment you have a correct server that performs better than your last commit, and again after every such improvement -- not only at the end.
The repo must contain:
- `pyproject.toml` declaring every dependency (the harness installs with `pip install .`; undeclared packages are absent at eval time).
- `server.py`, launched as `python server.py`.
The harness builds your submission with `pip install .` then `python server.py` in a fresh copy of this environment.

## Packaging limits
When your build ends, the last commit on `main` is bundled and built offline into the image your server is graded in. The harness then has at least 1 hour to collect your repository, at least 1,850 seconds of it to bundle it; a repository it cannot bundle in that time is not collected, and your build is not graded. Past any of these limits your build ends `over_limit`, with nothing gradable, and scores zero:
- the repository's objects, history and unreachable ones included, and the bundle of `main` stay within your sandbox's writable space, and what rebuilding its deltas reads within its bound, as the budget section below states;
- the repository's tree objects, history and unreachable ones included, hold at most 128 MiB together;
- the tree of `main`'s last commit lists at most 200,000 paths and 200,000 directories, each counted wherever it appears (a directory the tree names twice counts twice, with all it holds), and none of its paths, a directory's included, is longer than 1,024 bytes;
- the repository holds at most 5,000,000 objects, none larger than 4 GiB;
- the repository's objects lie in at most 10 object directories, its own and those its alternates name, and in at most 500 packs among them; those directories and its refs hold at most 6,000,000 files and directories together;
- a commit, tree or tag is at most 64 MiB, and an object stored as a delta at most 512 MiB.

Your build also ends `unreadable`, with nothing gradable, and scores zero, when the commit point cannot read the repository in your workspace:
- the workspace's `.git` is a symbolic link;
- the workspace's `.git` is a file that points at a repository elsewhere;
- the workspace's `.git` is a linked worktree's directory;
- the repository keeps its refs in reftable format;
- the repository names its objects with SHA-256;
- the repository's objects, refs or alternates, or a link among them, lead outside the workspace;
- symbolic links among the repository's objects or refs loop;
- the repository's objects or refs hold anything but directories, regular files and links;
- a file of the repository holds what git never writes: a ref, `packed-refs` or alternates file of more than 16 MiB or with a line of more than 64 KiB, an object or pack index git cannot read, or a delta chain deeper than 4,095;
- `main` is a symbolic ref to something that is not a ref, is reached through more than 5 symbolic refs, or holds no object name;
- `main`'s history has an object that is missing or corrupt, or `main` names something that is not a commit;
- `main`'s tree names a file or symbolic link whose object is not a blob but a tree, a commit or a tag, which git's plumbing can write and no checkout can read;
- a file of the repository refuses to be read, by the permissions you set.

Each file of your package is written byte for byte as `main`'s tree holds it: the conversions a `.gitattributes` asks git for (`text`, `eol`, `ident`, `filter` and `working-tree-encoding`) are not applied, though git applies them to the files in your sandbox.

Past any of these limits your submission is an invalid package and scores zero:
- the bundle of `main` reads back as git reads it: no object `main`'s history names is missing from it, of another kind, or stored under another object's name (a loose object's file name or a pack's index can store one so), and git can inflate every object in it and apply every delta;
- the committed tree holds at most 4 GiB of files; a path is at most 64 levels deep, each name in it at most 255 bytes, and every path is UTF-8 and holds no name git refuses to check out, such as `..` or one NTFS reads as `.git` (`.git`, `.GIT.`, `git~1`);
- the tree holds no symlink;
- `pyproject.toml` holds at most 1 MiB;
- installing your wheels adds at most 24 GiB to the image's Python environment, and their entries declare at most that much together;
- pip's two steps, a dry run that resolves your dependencies and then the build of the wheels, both running your project's build hooks, take at most 4 hours together, counted from each step's command's start, unless the package index failed meanwhile;
- each pip step fetches at most 10,000 pages and files from the package index, redirects included, and no dependency's index page is larger than 64 MiB;
- the dry run's report of what it would install holds at most 8 MiB and nests no deeper than 32 levels;
- the wheel step leaves only wheel files: your project's wheel and at most one wheel of each dependency the dry run resolved;
- your wheels together install at most 500,000 paths (each file, each directory above one and each script their entry points declare), nest no entry deeper than 64 levels, have no name in an entry's path longer than 255 bytes, hold no Python source file larger than 128 MiB and no `.dist-info` entry larger than 16 MiB, and none lists its entries in more than 16 MiB;
- every wheel is a zip each of whose entries is packed by `store` or `deflate` and unpacks to the size and checksum it declares.

Each pip step may write at most 1763 GiB to the node's disk, its copy of your tree, its temporary files and its wheels together, and hold at most 32 GiB of memory; a step past either is stopped, and the submission scores zero.

## Your budget
You have 12 hours of wall-clock time for this build. Call `get_budget` (the `budget` MCP server) periodically to see how much remains, and plan your remaining work to stay within it. Leave headroom to commit working code before you run out.

Your sandbox has 1900 GiB of memory, of which the harness's own processes may hold up to 4 GiB, and 64 CPUs. Its `/dev/shm` holds up to 1900 GiB, and what is there counts in that memory. You can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your session's logs share 1911 GiB of writable space, separate from your memory, and hold at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep. The objects your repository's `.git` holds, its history and unreachable objects among them, may come to at most those 1911 GiB together uncompressed, and the bundle of main the harness makes of them to at most 1911 GiB; past either, the build has no submission and scores zero. Every object git stores as a delta counts with what rebuilding it reads: itself, its delta, and the same for the object it is a delta on, down its chain. Those may come to at most 64 GiB together; past that, the build has no submission and scores zero. Your commands can start a process or a thread only while they hold fewer than 131,072 of them together; past that a fork fails with EAGAIN. Your commands, the agent's own among them, may hold at most 127 inotify instances, 1,047,552 inotify watches and 999,968 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another. When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending processes does not bring the memory back under it, the sandbox ends. A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on what you last committed on main; past the files or the nesting, it has run out of its writable space, and so it has when one of its processes holds open files the harness cannot list, as a process running a program from a file it cannot read does. Your commands run with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.

When your server is graded, it runs with 8 H200 GPUs, 64 CPUs, 1907 GiB of memory and 3814 GiB of writable space, which holds at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep; past any of those the server is stopped, and the grading scores zero. Build it for those. `/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and `/engine-cache`. A process of it that holds open files the harness cannot list counts as past its writable space. It and every process it starts may hold at most 128 inotify instances, 1,048,576 inotify watches and 1,048,560 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another.

When your server is graded, the harness records each request and its reply in the drive's trace as the drive goes, and the trace may hold at most 22 GiB of JSON: once what it has recorded, with each request it has sent and not yet recorded counted at the JSON the trace will record of that request, and each reply it has received and not yet recorded counted at the most JSON that reply's bytes can become, reaches that, every further reply fails as your server's failed request. One reply may carry at most 256 MiB, and a longer one fails; so does a reply in a content coding other than identity, since the harness asks for uncompressed replies.
