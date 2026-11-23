# Task Description

## Goal
Serve `Qwen/Qwen3-Omni-30B-A3B-Instruct` under the workload below. Optimize until
you reach the physical limits of the hardware or exhaust your budget.

**Your score is your speedup over a production serving framework in SLO-normalized
latency.** For every request, against the per-request SLO listed under ## Workload:

    r = (time from request sent to response complete) / that SLO

    score = exp( mean over requests of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling. A score of 1.0 means you matched the reference stack,
2.0 means a typical request took half as long. Meeting every SLO is not the finish line:
the score keeps rising as latency falls, so there is no point at which further speed
stops paying.

Requests arrive open-loop on a Poisson schedule you do not control, at a fixed rate.
Serving faster does not earn you more requests -- it earns you a better score on the ones
you get. Every request is weighted equally, so a slow tail costs as much as it would in
production.

The traffic is an even mix: half the requests carry an image with a question about it,
half carry French speech to translate into English. Both go to the same endpoint and are
scored on one scale, so a design that serves one modality well and queues the other
behind it pays for the queue.

**Your implementation must not fail any request.** One failed request zeroes the run,
which means shedding or refusing load is never a way to protect your latency.

Correctness is scored against a bf16 HuggingFace reference decode of this same
checkpoint, answering the same requests. Each modality is gated separately and both must
hold: word F1 against the reference answer on the image arm, sentence BLEU against the
reference translation on the speech arm. Each must reach 98% of what that reference
scores on the same run. Those bars are on your mean over the whole run, not on any single
request, and they are pass/fail with no credit for margin.

`oracle_dev.json` beside the dev rows carries the reference's answer and score for every
dev row, tagged with the arm it belongs to, plus a mean per arm. The bars are fractions of
those numbers, so this is what to compare your own dev scores against. Read them as the
levels to aim at rather than proof you clear the bars: the dev split is 64 rows and it
divides between the two arms, leaving about thirty on each, which is far too few to
separate a ratio of 0.98 from one of 1.00. A dev score near a bar does not settle the
graded run either way.
## Model
- Checkpoint: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Chat format: `qwen3-omni`
## Hardware Envelope
2xH200, 141 GB/device
181 GB host RAM
## Workload
Arrivals are sessions; each turn of a session is one request. Offered load over all
tenants: 30 requests/s, an upper bound, since a session can end before its
last turn.
- Sessions: poisson arrivals, 30 sessions/s
- Requests: 1 turn/session, 30 requests/s
- Per-request output cap: 256 tokens
- SLO: JCT < 15000ms (per request)
## Network Contract
### GET /health
Return 200 **only** when the model is fully loaded and the inference endpoint can
serve a request immediately. Return non-200 (e.g. 503) at any point before that.
**If you load the model in a background thread, your /health must block on load
completion before returning 200.** Returning 200 early (before the model is ready)
violates the substrate's readiness contract: the harness stops polling /health
the moment it sees 200 and immediately sends inference traffic.

### POST /v1/chat/completions
Accept OpenAI multimodal chat/completions requests: each message's ``content`` is a
list of typed parts -- ``{"type": "text", ...}``, ``{"type": "image_url", "image_url":
{"url": "data:image/...;base64,..."}}``, and ``{"type": "input_audio", "input_audio":
{"data": "<base64 WAV>", "format": "wav"}}``. A request may carry an image + a question
or a speech clip; its text part says what to do with the input. Decode greedily and
return a normal chat completion whose message content is the text answer.

Request:
```json
{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "<prompt>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,<...>"
          }
        }
      ]
    }
  ],
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```
Response:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "<answer>"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "completion_tokens": 2
  }
}
```


## Decoding
**Decoding.** Decode greedily (argmax, no sampling). The chat-templated prompt already
carries the model's special tokens, so do not add another BOS; strip special tokens from the output,
stopping at the end-of-turn token and never emitting the raw end token (e.g. `<|im_end|>`) in the
response text.

## Build Instructions
Write your server as `/workspace/server.py`, run as
`python /workspace/server.py --host <H> --port <P>`. It is built and evaluated OFFLINE.
Implement it in any of the following languages and toolchains: `/workspace/server.py` need only be the entrypoint, and
may exec or bind whatever you build.

- Weights: a local checkpoint (config + safetensors + tokenizer) mounted read-only at
  `$HLS_WEIGHTS_DIR`. The accelerators are `cuda:0` through `cuda:1`; all 2 are yours.
- Data: a development split is on disk at `/data/tenants/t0/dev/rows.json`; the scored test split is withheld. `oracle_dev.json` sits beside those rows with the
reference's answer and score for each of them, tagged with the arm it belongs to.
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

Your sandbox has 169 GiB of memory, of which the harness's own processes may hold up to 4 GiB, and 8 CPUs. Its `/dev/shm` holds up to 169 GiB, and what is there counts in that memory. You can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your session's logs share 180 GiB of writable space, separate from your memory, and hold at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep. The objects your repository's `.git` holds, its history and unreachable objects among them, may come to at most those 180 GiB together uncompressed, and the bundle of main the harness makes of them to at most 180 GiB; past either, the build has no submission and scores zero. Every object git stores as a delta counts with what rebuilding it reads: itself, its delta, and the same for the object it is a delta on, down its chain. Those may come to at most 64 GiB together; past that, the build has no submission and scores zero. Your commands can start a process or a thread only while they hold fewer than 131,072 of them together; past that a fork fails with EAGAIN. Your commands, the agent's own among them, may hold at most 31 inotify instances, 261,120 inotify watches and 249,968 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another. When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending processes does not bring the memory back under it, the sandbox ends. A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on what you last committed on main; past the files or the nesting, it has run out of its writable space, and so it has when one of its processes holds open files the harness cannot list, as a process running a program from a file it cannot read does. Your commands run with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.

When your server is graded, it runs with 2 H200 GPUs, 8 CPUs, 176 GiB of memory and 180 GiB of writable space, which holds at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep; past any of those the server is stopped, and the grading scores zero. Build it for those. `/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and `/engine-cache`. A process of it that holds open files the harness cannot list counts as past its writable space. It and every process it starts may hold at most 32 inotify instances, 262,144 inotify watches and 262,128 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another.

When your server is graded, the harness records each request and its reply in the drive's trace as the drive goes, and the trace may hold at most 3 GiB of JSON: once what it has recorded, with each request it has sent and not yet recorded counted at the JSON the trace will record of that request, and each reply it has received and not yet recorded counted at the most JSON that reply's bytes can become, reaches that, every further reply fails as your server's failed request. This cap is what the harness's verification of the trace can hold in memory, which is less than it would otherwise allow. One reply may carry at most 256 MiB, and a longer one fails; so does a reply in a content coding other than identity, since the harness asks for uncompressed replies.
