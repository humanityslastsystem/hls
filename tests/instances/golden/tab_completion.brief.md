# Task Description

## Goal
Build a hyperoptimized fill-in-the-middle code-completion server for
`JetBrains/Mellum2-12B-A2.5B-Base` under the workload below. Optimize until you reach the
physical limits of the hardware or exhaust your budget. The grader drives all tenants
concurrently against one instance of your server.

**Your score is SLO-normalized latency.** Each tenant declares its own TTFT and TPOT budgets
(listed per tenant under ## Workload). For every request:

    r      = max( TTFT / that tenant's TTFT budget , mean TPOT / that tenant's TPOT budget )
    credit = log(1 / r)          # natural log

    score  = exp( mean credit over every request - baseline credit )   higher is better.

Sessions arrive open-loop on a schedule you do not control. The commit tenants send
typing sessions: a developer accepts a completion, types a little, and the editor sends
nearly the same prompt again with a longer prefix. Each step is a separate request, sent
and scored on its own, and a later step's prompt repeats the earlier one's prefix.

**Your implementation must not fail any request.** Every request must receive a complete
streaming response. Correctness is checked at the token level: the exact token ids you
serve are teacher-forced through a bf16 reference implementation of this model --
HuggingFace transformers, batch size 1, MATH SDPA backend, cuDNN attention disabled,
the model class resolved from the checkpoint's declared architectures, scored in
4,096-token chunks against a KV cache. The run fails if more than
0.9517% of your served tokens have a lower reference log-probability than the
reference's most likely token, or if the number of such tokens exceeds a
calibrated ceiling at any of several severity levels, where severity is that
log-probability gap (the ceilings scale with the run's total token count). Exact
ties do not count: a token equally probable under the reference is not a deviation.

The thresholds are calibrated on fast production-grade serving stacks. What fails is
genuine deviation: quantized weights, altered algorithms, modified sampling. The gate is
pass/fail with no credit for margin, so latency spent buying faithfulness headroom below
the ceiling is pure loss.

Serve greedy: the argmax of the raw logits, with none of the checkpoint's
generation_config defaults applied. The reference also scores where each reply stops:
a reply that ends where the reference would have kept going is a deviation like any
other.

That check is one pooled measurement over every served token of the whole run, not one
per tenant. To reproduce it on the dev split, draw your sample in the per-tenant request
proportions the workload states and let each tenant generate under its output cap:
the tenants' individual rates differ severalfold, so a sample drawn in different
proportions, or a small one, can easily read 50% higher or lower than the number you
are graded on.
## Model
- Checkpoint: `JetBrains/Mellum2-12B-A2.5B-Base`
- Mellum2 Base, a fill-in-the-middle code-completion model from JetBrains: a mixture of
  experts, 12B weights with 2.5B active per token, no chat template. Every prompt
  arrives pre-assembled with its sentinels (`<fim_prefix>`, `<fim_suffix>`,
  `<fim_middle>`); never apply a chat template to it.
## Hardware Envelope
1xH200, 141 GB/device
181 GB host RAM
## Workload
Arrivals are sessions; each turn of a session is one request. Offered load over all
tenants: 24.61 requests/s, an upper bound, since a session can end before its
last turn.
### Tenant commit_add
- Dataset: gonglinyuan/real_fim_eval
- Sessions: poisson arrivals, 1.139 sessions/s
- Requests: 5.4 turns/session, 6.153 requests/s
- Between turns: gamma think-time (shape=1, scale=14.427, cap 60s)
- Per-request output cap: 256 tokens
- SLO: TTFT < 600ms, TPOT < 10ms (per request)

### Tenant commit_edit
- Dataset: gonglinyuan/real_fim_eval
- Sessions: poisson arrivals, 1.161 sessions/s
- Requests: 5.3 turns/session, 6.153 requests/s
- Between turns: gamma think-time (shape=1, scale=14.427, cap 60s)
- Per-request output cap: 256 tokens
- SLO: TTFT < 600ms, TPOT < 10ms (per request)

### Tenant algo_block
- Dataset: gonglinyuan/safim
- Sessions: poisson arrivals, 6.153 sessions/s
- Requests: 1 turn/session, 6.153 requests/s
- Per-request output cap: 256 tokens
- SLO: TTFT < 250ms, TPOT < 10ms (per request)

### Tenant algo_control
- Dataset: gonglinyuan/safim
- Sessions: poisson arrivals, 6.153 sessions/s
- Requests: 1 turn/session, 6.153 requests/s
- Per-request output cap: 256 tokens
- SLO: TTFT < 200ms, TPOT < 10ms (per request)
## Network Contract
### GET /health
Return 200 **only** when the model is fully loaded and the inference endpoint can
serve a request immediately. Return non-200 (e.g. 503) at any point before that.
**If you load the model in a background thread, your /health must block on load
completion before returning 200.** Returning 200 early (before the model is ready)
violates the substrate's readiness contract: the harness stops polling /health
the moment it sees 200 and immediately sends inference traffic.

### POST /v1/completions
Accept OpenAI text-completion requests and return text-completion responses. The
request carries a raw `prompt` string that is already the exact model input: feed it
to the model verbatim, with no chat template, no role markup and no added special
tokens. The harness sends `max_tokens` on every request body; treat it as a binding
per-request output cap. Do not substitute a server-side default when the request
supplies `max_tokens`.

Return a `token_ids` array on each choice (`choices[].token_ids`): the integer
vocabulary IDs of the exact tokens you emitted (NOT a re-tokenization of the text).
This is mandatory on every response.
Every request carries `stop_token_ids` [0]: generation ends when the
model emits any of them. `token_ids` must include that end-of-turn token when the
reply ended on one, even though it is not part of the returned `text`. A reply whose
first generated token is one of them is one token, not zero. A response with an
empty `token_ids` array is unverifiable and fails the run.
Support streaming: when the request sets `"stream": true`, respond with an OpenAI
SSE stream (`data: {chunk}` lines of `text_completion` objects whose `choices[].text`
holds that chunk's text, terminated by `data: [DONE]`). Include the per-chunk
`token_ids` array for that chunk's tokens.
Request:
```json
{
  "model": "served-model",
  "prompt": "<prompt>",
  "return_token_ids": true,
  "stop_token_ids": [
    0
  ]
}
```
Response:
```json
{
  "choices": [
    {
      "text": "<completion>",
      "finish_reason": "stop",
      "token_ids": [
        1234,
        5678
      ]
    }
  ],
  "usage": {
    "completion_tokens": 2
  }
}
```

Sizing: prompts run long, and a request must be served at whatever length it arrives --
refusing or truncating one is a failed request. Size the server for a context window of 32,768 tokens. No request can exceed the longest prompt plus the 256-token output cap, and the checkpoint's own window is larger still.

The `prompt` string is the complete model input. Tokenize it with
`add_special_tokens=False`: the sentinels are already in the string, and the reference
tokenizes the same way. The request's `stop_token_ids` is `<|endoftext|>`, id 0, the
checkpoint's only end-of-text token.

## Decoding
**Decoding.** Decode greedily (argmax, no sampling). Feed the prompt
verbatim with no added BOS; stop at `<|endoftext|>`, id 0, keep that id in `token_ids`,
and never put it in the response text.

## Build Instructions
Write your server as `/workspace/server.py`, run as
`python /workspace/server.py --host <H> --port <P>`. It is built and evaluated OFFLINE.
Implement it in any of the following languages and toolchains: `/workspace/server.py` need only be the entrypoint, and
may exec or bind whatever you build.

- Weights: a local checkpoint (config + safetensors + tokenizer) mounted read-only at
  `$HLS_WEIGHTS_DIR`. The accelerator is `cuda:0`.
- Data: a development split is on disk at `/data/tenants/<tenant>/dev/rows.json`, one per tenant (ids as listed under ## Workload); the scored test split is withheld. Beside every tenant's dev rows, `fim_prompt.txt` gives the
  template a served prompt is assembled from, and how a commit row becomes a typing
  session; the rows alone do not show either.
- `pip install`: any package except the blocked ones is installable.
- Blocked (`pip install` refuses these): `vllm`, `vllm-cpu`, `vllm-tpu`, `vllm-rocm`, `nano-vllm`, `sglang`, `sglang-router`, `lmdeploy`, `turbomind`, `tensorrt-llm`, `text-generation-inference`, `optimum-nvidia`, `llama-cpp-python`, `ctransformers`, `exllama`, `exllamav2`, `exllamav3`, `ktransformers`, `kt-kernel`, `lightllm`, `aphrodite-engine`, `mlc-llm`, `mlc-ai`, `mlc-ai-nightly`, `mlc-llm-cu121`, `mlc-llm-cu122`, `mlc-llm-cu124`, `mlc-llm-cu128`, `mlc-llm-nightly-cu121`, `mlc-llm-nightly-cu122`, `mlc-llm-nightly-cu124`, `mlc-llm-nightly-cu128`, `mlc-llm-nightly-cu130`, `mlc-ai-cu121`, `mlc-ai-cu122`, `mlc-ai-nightly-cu121`, `mlc-ai-nightly-cu122`, `mlc-ai-nightly-cu124`, `mlc-ai-nightly-cu128`, `mlc-ai-nightly-cu130`, `deepspeed`, `deepspeed-mii`, `deepspeed-kernels`, `mistral-inference`, `openllm`, `ai-dynamo`, `ai-dynamo-runtime`, `lmcache`, `lightseq`, `fastertransformer`, `ipex-llm`, `petals`, `medusa-llm`, `airllm`, `tokenspeed`, `tokenspeed-kernel`, `tokenspeed-scheduler`, `tokenspeed-triton`, `tokenspeed-deepgemm`, `tokenspeed-flashmla`, `tokenspeed-mla`, `tokenspeed-nightly`, `tokenspeed-proton`, `vllm-omni`, `sglang-omni`, `ctranslate2`, `faster-whisper`, `whisperx`, `whisper-ctranslate2`, `insanely-fast-whisper`, `whisper-s2t`, `wyoming-faster-whisper`, `faster-whisper-server`, `whisper-live`, `whisperlivekit`, `whisper-cpp-python`, `pywhispercpp`, `whispercpp`, `sherpa-onnx`, `funasr`, `tensorrt`, `tensorrt-bindings`, `tensorrt-libs`, `tensorrt-cu11`, `tensorrt-cu11-bindings`, `tensorrt-cu11-libs`, `tensorrt-cu12`, `tensorrt-cu12-bindings`, `tensorrt-cu12-libs`, `tensorrt-cu13`, `tensorrt-cu13-bindings`, `tensorrt-cu13-libs`, `tensorrt-lean`, `tensorrt-lean-bindings`, `tensorrt-lean-libs`, `tensorrt-lean-cu11`, `tensorrt-lean-cu11-bindings`, `tensorrt-lean-cu11-libs`, `tensorrt-lean-cu12`, `tensorrt-lean-cu12-bindings`, `tensorrt-lean-cu12-libs`, `tensorrt-lean-cu13`, `tensorrt-lean-cu13-bindings`, `tensorrt-lean-cu13-libs`, `tensorrt-dispatch`, `tensorrt-dispatch-bindings`, `tensorrt-dispatch-libs`, `tensorrt-dispatch-cu11`, `tensorrt-dispatch-cu11-bindings`, `tensorrt-dispatch-cu11-libs`, `tensorrt-dispatch-cu12`, `tensorrt-dispatch-cu12-bindings`, `tensorrt-dispatch-cu12-libs`, `tensorrt-dispatch-cu13`, `tensorrt-dispatch-cu13-bindings`, `tensorrt-dispatch-cu13-libs`, `tensorrt-rtx`, `tensorrt-rtx-cu12`, `nvidia-tensorrt`, `torch-tensorrt`, `sgl-kernel`, `sglang-kernel`, `vllm-flash-attn`, `flashinfer-cubin`, `flashinfer-jit-cache`, `kernels`, `kernels-community`, `hf-kernels`, `flash-attn`, `flash-attn-3`, `flash-attn3`, `flash-attn-4`, `flash-attn-cute`, `flash-attn-wheels`, `flash-attn-triton`, `flash-attention`, `flashinfer`, `flashinfer-python`, `xformers`, `flash-mla`, `sageattention`, `ring-flash-attn`, `fla-core`, `flash-linear-attention`, `transformer-engine`, `transformer-engine-cu12`, `transformer-engine-cu13`, `transformer-engine-torch`, `transformer-engine-jax`, `deep-gemm`, `auto-gptq`, `autoawq`, `autoawq-kernels`, `gptqmodel`, `marlin`, `marlin-kernels`, `bitblas`, `gemlite`, `flute-kernel`, `humming-kernels`, `fbgemm-gpu`, `fbgemm-gpu-genai`, `fbgemm-gpu-nightly`, `fbgemm-gpu-genai-nightly`, `sonic-moe`, `megablocks`, `grouped-gemm`, `moe-kernels`, `deep-ep`, `triton-kernels`, `liger-kernel`, `quack-kernels`, `unsloth`, `unsloth-zoo`, `causal-conv1d`, `mamba-ssm`.

Everything below is already installed. You are OFFLINE, so the documentation paths given
are the whole of what is readable -- there is no other reference available to you. Paths
are relative to your working directory.

### Python
`torch`, `transformers`, `accelerate`, `datasets`, `fastapi`, `uvicorn`, `pydantic`.

- torch, including `torch.compile`/Inductor and `torch.utils.cpp_extension` -- `references/torch/`

### C / C++
`gcc`/`g++` 13.3, `clang`, `make`, `cmake`, `ninja`, `pkg-config`, `python3-dev`.

PyTorch C++/CUDA extensions build here: torch ships its own `torch/include` headers and a
vendored pybind11, `TORCH_CUDA_ARCH_LIST` is already set for this GPU, and both
`torch.utils.cpp_extension.load()` (JIT) and setuptools `BuildExtension` work offline.

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

- Triton, with 27 runnable annotated tutorials -- `references/triton/`
  (`references/triton/tutorials/06-fused-attention.py` is FlashAttention,
  `03-matrix-multiplication.py` a tiled GEMM)
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
are listed under ## Build Instructions above. There is no other reference
available to you, so consult these rather than relying on recall.

These files are available at `/workspace/references` inside your sandbox.

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

Each pip step may write at most 32 GiB to the node's disk, its copy of your tree, its temporary files and its wheels together, and hold at most 32 GiB of memory; a step past either is stopped, and the submission scores zero.

`torch` 2.11.0+cu130, `torchaudio` 2.11.0+cu130 and `torchvision` 0.26.0+cu130 are held when your server is built: an install that asks for another version of one of them fails there. Your sandbox sets the same constraints by default (`PIP_CONSTRAINT`, `UV_CONSTRAINT`).

## Your budget
You have 12 hours of wall-clock time for this build. Call `get_budget` (the `budget` MCP server) periodically to see how much remains, and plan your remaining work to stay within it. Leave headroom to commit working code before you run out.

Your sandbox has 169 GiB of memory, of which the harness's own processes may hold up to 4 GiB, and 8 CPUs. Its `/dev/shm` holds up to 169 GiB, and what is there counts in that memory. You can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your session's logs share 180 GiB of writable space, separate from your memory, and hold at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep. The objects your repository's `.git` holds, its history and unreachable objects among them, may come to at most those 180 GiB together uncompressed, and the bundle of main the harness makes of them to at most 180 GiB; past either, the build has no submission and scores zero. Every object git stores as a delta counts with what rebuilding it reads: itself, its delta, and the same for the object it is a delta on, down its chain. Those may come to at most 64 GiB together; past that, the build has no submission and scores zero. Your commands can start a process or a thread only while they hold fewer than 131,072 of them together; past that a fork fails with EAGAIN. Your commands, the agent's own among them, may hold at most 15 inotify instances, 130,048 inotify watches and 124,968 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another. When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending processes does not bring the memory back under it, the sandbox ends. A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on what you last committed on main; past the files or the nesting, it has run out of its writable space, and so it has when one of its processes holds open files the harness cannot list, as a process running a program from a file it cannot read does. Your commands run with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.

When your server is graded, it runs with 1 H200 GPU, 8 CPUs, 176 GiB of memory and 180 GiB of writable space, which holds at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep; past any of those the server is stopped, and the grading scores zero. Build it for those. `/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and `/engine-cache`. A process of it that holds open files the harness cannot list counts as past its writable space. It and every process it starts may hold at most 16 inotify instances, 131,072 inotify watches and 131,056 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another.

When your server is graded, the harness records each request and its reply in the drive's trace as the drive goes, and the trace may hold at most 6 GiB of JSON: once what it has recorded, with each request it has sent and not yet recorded counted at the JSON the trace will record of that request, and each reply it has received and not yet recorded counted at the most JSON that reply's bytes can become, reaches that, every further reply fails as your server's failed request. One reply may carry at most 256 MiB, and a longer one fails; so does a reply in a content coding other than identity, since the harness asks for uncompressed replies.
