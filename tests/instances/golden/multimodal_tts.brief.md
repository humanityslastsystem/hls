# Task Description

## Goal
Serve `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` under the workload below. Optimize until
you reach the physical limits of the hardware or exhaust your budget.

**Your score is your speedup over a production serving framework in SLO-normalized
latency.** For every request, against the per-request SLO listed under ## Workload:

    r = (time from request sent to the last audio frame) / that SLO

    score = exp( mean over requests of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling. A score of 1.0 means you matched the production
framework, 2.0 means a typical utterance took half as long. The SLO is on the complete utterance,
not on first audio. Requests arrive open-loop on a Poisson schedule you do not control,
at a fixed rate.

**Your implementation must not fail any request.** One failed request zeroes the run.
The scored test split is 2,400 sentences, driven once and cold: no warm-up requests are
sent, so your first requests are timed like the rest. Every reply must end on end of
speech (2150) within the output cap under ## Workload; a reply that reaches the cap
without ending fails.

**Correctness is faithfulness to this checkpoint's own decode.** All 16 code groups are
sampled, so you are not graded by comparing your codes against a reference's. You are
graded on what you report: at every position, the log-probability you gave the code you
served and that position's top 65 alternatives, under the distribution you actually
sampled from. Two checks read the whole run. The first reads the report. For every served
code, the reference model is run on the same prefix and the KL divergence from your reported
distribution to the reference's is averaged over all graded positions; the run fails if this
mean exceeds 2.79e-03 nats per position.
The codes you served must be faithful draws from the distributions you report; this is
tested statistically, and a server that reports one distribution and samples from another
fails. The reported alternatives must be the most probable codes: the run fails if more
than 68.85 percent of your served codes are not among the
alternatives you reported at their position, counting only those above your
64th-largest reported value (the last two set the cut-off). Ending utterances
early is tested too: how often you end where the reference gives end of speech little
probability is held to what the reference expects.

The second check reads the audio: the 16 groups of codes you sent are decoded by the reference codec
decoder, and any utterance whose log-mel PSNR against that decode falls below
29 dB, or whose length differs by more than 2 codec frames, fails.
The log-mel analysis is:

    24 kHz, n_fft 1024, hop 256, 80 mels, centred Hann window
    POWER spectrogram (power=2.0), HTK mel scale, UNNORMALIZED filters (norm=None)
    floored at 1e-05, then base-10 log

and the PSNR's peak is the reference log-mel's own range (maximum minus minimum) over the
frames both utterances share.

The report's two ceilings are calibrated on a trusted serving stack of this checkpoint;
the audio floor sits 3 dB under the worst honest decoder variant measured. What fails is genuine
deviation: other weights, a quantization, another temperature, top-k or penalty, a report
of a distribution you did not sample from. Both checks are pass/fail with no credit for
margin -- a run that passes at 90% of a ceiling scores identically to one at 10% -- so
latency spent buying faithfulness headroom below a ceiling is pure loss. A reply whose
codes frame or report is missing, duplicated or malformed fails.


## Model
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Chat format: `qwen3-tts`
## Hardware Envelope
1xH200, 141 GB/device
245 GB host RAM
## Workload
Arrivals are sessions; each turn of a session is one request. Offered load over all
tenants: 2 requests/s, an upper bound, since a session can end before its
last turn.
- Sessions: poisson arrivals, 2 sessions/s
- Requests: 1 turn/session, 2 requests/s
- Per-request output cap: 1280 tokens
- SLO: JCT < 10000ms (per request)
## Network Contract
### GET /health
Return 200 **only** when the model is fully loaded and the inference endpoint can
serve a request immediately. Return non-200 (e.g. 503) at any point before that.
**If you load the model in a background thread, your /health must block on load
completion before returning 200.** Returning 200 early (before the model is ready)
violates the substrate's readiness contract: the harness stops polling /health
the moment it sees 200 and immediately sends inference traffic.

### POST /v1/audio/speech
Accept a JSON text-to-speech request and stream the synthesized audio back as an SSE
stream: when ``"stream": true``, respond with ``data: {frame}`` lines (each frame a
base64 PCM16 mono WAV chunk under ``audio``), terminated by ``data: [DONE]``. You may stream frames as they are decoded or send them
together; only the arrival of the last audio frame is timed.

Each frame is a complete WAV file, its own RIFF header followed by mono PCM16 samples at
the rate pinned under ## Decoding, base64-encoded; the frames concatenated in order are
the utterance that is graded.

Request body (JSON):
```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "<text>",
  "voice": "default",
  "stream": true,
  "return_token_ids": false,
  "return_codec_codes": true,
  "return_logprobs": true,
  "top_logprobs": 65
}
```
`voice` is always `default`; the speaker to synthesize in is the one pinned under
## Decoding, whatever this field carries.

An empty `input` is a readiness probe, sent after your `/health` returns 200 and before
any real traffic. Answer it immediately without synthesizing, with a 2xx (an empty stream
is fine) or a 400. Those two count as ready; any other status, including 404, 405 and
422, means not ready yet, and the probe is repeated every 2 s until startup times out.

Per-frame SSE payload:
```json
{
  "audio": "<b64 wav frame>"
}
```
After the last audio frame and before `data: [DONE]`, send exactly one further frame
carrying the codec codes the audio was decoded from, and no audio:
```json
{"codec_codes": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "codec_report": [[[-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]], [-0.412535, [[0, -0.412535], [17, -1.884102]]]]], "eos_code": 2150, "eos_report": [-0.412535, [[2150, -0.412535], [17, -1.884102]]]}
```
`codec_codes` has one row per codec frame in generation order, 16 integer ids per row
(one per codebook group), each in `[0, 2048)`, with no end-of-speech row.
The example above is abbreviated twice: it shows one codec frame where a reply sends one
row per frame of audio, and two alternatives per position where a reply sends 65.

`codec_report` covers exactly the same positions: `codec_report[f][g]` describes the one
whose served code is `codec_codes[f][g]`. Each is `[lp, [[id, lp], ...]]` -- the
log-probability your sampler gave the code you served there, then the 65 most probable
alternatives as `[id, log-probability]` pairs, whether or not you served one of them, all
under exactly the distribution you sampled from.

`eos_report` is the same for the end-of-speech decision, the codebook-0 position after
your last frame. `eos_code` is the id you served there, which for this checkpoint is 2150.
It is a position on codebook 0's vocabulary, not a seventeenth group. Alternatives at any
codebook-0 position may name ids above the codebook.

Send the alternatives in whatever order your stack produces; do not deduplicate or
reorder them.
This frame is the record of what your model chose, and correctness is graded on it; a
reply that omits it, sends it more than once, sends audio after it, or sends rows of
another shape fails. It is not charged as latency: the per-request clock stops at your
last audio frame.


## Decoding
**Decoding.** Every request is answered by `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`'s
own shipped weights, under this decode, with no substitute model in the path:

- Speaker `Ryan`, language `English`, and an empty style instruction.
- Non-streaming text mode: the whole sentence is in the talker's prefix before the first
  frame is generated.
- All 16 code groups sampled from the plain softmax at
  temperature 1.0: no top-k, no top-p, no repetition penalty.
- On codebook 0, the talker's special ids (2048 through 3071, except end of speech,
  2150) are masked at every step, and end of speech is masked at frames 0 and 1.
- Pass every one of these explicitly. The checkpoint's `generation_config` supplies
  temperature 0.9, top-k 50 and a repetition penalty of 1.05 to anything you leave unset.
- Report, at every position, the log-probability you gave the code you served and that
  position's top 65 alternatives, under the plain softmax at temperature
  1.0 after those masks, so a masked id carries no probability.
- Emit the codec decoder's own 24 kHz waveform unmodified: no resampling, no
  trimming, no gain, no time-stretching, no denoising. How you run the decoder is yours.
  Decoding in chunks with left context is allowed, but it does move the waveform: the
  worst chunked variant measured sits 3 dB above the audio floor.

The per-request output cap under ## Workload counts codec frames, not text tokens: one
frame is 16 codes and 80 ms of audio.

## Build Instructions
Write your server as `/workspace/server.py`, run as
`python /workspace/server.py --host <H> --port <P>`. It is built and evaluated OFFLINE.
`/workspace/server.py` need only be the entrypoint, and may exec or bind whatever you build.

- Weights: a local checkpoint (config + safetensors + tokenizer) mounted read-only at
  `$HLS_WEIGHTS_DIR`. The accelerator is `cuda:0`, with 8 CPU cores and 245 GB of host
  RAM beside it.
- Data: a development split of 64 sentences is on disk at `/data/tenants/t0/dev/rows.json`; the
  scored test split is withheld. Each row's `text_normalized` is the sentence to say,
  verbatim, and `source_index` identifies it; nothing else sits beside the rows.
- `pip install`: any package except the blocked ones is installable.
- Blocked (`pip install` refuses these): `vllm`, `vllm-cpu`, `vllm-tpu`, `vllm-rocm`, `nano-vllm`, `sglang`, `sglang-router`, `lmdeploy`, `turbomind`, `tensorrt-llm`, `text-generation-inference`, `optimum-nvidia`, `llama-cpp-python`, `ctransformers`, `exllama`, `exllamav2`, `exllamav3`, `ktransformers`, `kt-kernel`, `lightllm`, `aphrodite-engine`, `mlc-llm`, `mlc-ai`, `mlc-ai-nightly`, `mlc-llm-cu121`, `mlc-llm-cu122`, `mlc-llm-cu124`, `mlc-llm-cu128`, `mlc-llm-nightly-cu121`, `mlc-llm-nightly-cu122`, `mlc-llm-nightly-cu124`, `mlc-llm-nightly-cu128`, `mlc-llm-nightly-cu130`, `mlc-ai-cu121`, `mlc-ai-cu122`, `mlc-ai-nightly-cu121`, `mlc-ai-nightly-cu122`, `mlc-ai-nightly-cu124`, `mlc-ai-nightly-cu128`, `mlc-ai-nightly-cu130`, `deepspeed`, `deepspeed-mii`, `deepspeed-kernels`, `mistral-inference`, `openllm`, `ai-dynamo`, `ai-dynamo-runtime`, `lmcache`, `lightseq`, `fastertransformer`, `ipex-llm`, `petals`, `medusa-llm`, `airllm`, `tokenspeed`, `tokenspeed-kernel`, `tokenspeed-scheduler`, `tokenspeed-triton`, `tokenspeed-deepgemm`, `tokenspeed-flashmla`, `tokenspeed-mla`, `tokenspeed-nightly`, `tokenspeed-proton`, `vllm-omni`, `sglang-omni`, `ctranslate2`, `faster-whisper`, `whisperx`, `whisper-ctranslate2`, `insanely-fast-whisper`, `whisper-s2t`, `wyoming-faster-whisper`, `faster-whisper-server`, `whisper-live`, `whisperlivekit`, `whisper-cpp-python`, `pywhispercpp`, `whispercpp`, `sherpa-onnx`, `funasr`, `tensorrt`, `tensorrt-bindings`, `tensorrt-libs`, `tensorrt-cu11`, `tensorrt-cu11-bindings`, `tensorrt-cu11-libs`, `tensorrt-cu12`, `tensorrt-cu12-bindings`, `tensorrt-cu12-libs`, `tensorrt-cu13`, `tensorrt-cu13-bindings`, `tensorrt-cu13-libs`, `tensorrt-lean`, `tensorrt-lean-bindings`, `tensorrt-lean-libs`, `tensorrt-lean-cu11`, `tensorrt-lean-cu11-bindings`, `tensorrt-lean-cu11-libs`, `tensorrt-lean-cu12`, `tensorrt-lean-cu12-bindings`, `tensorrt-lean-cu12-libs`, `tensorrt-lean-cu13`, `tensorrt-lean-cu13-bindings`, `tensorrt-lean-cu13-libs`, `tensorrt-dispatch`, `tensorrt-dispatch-bindings`, `tensorrt-dispatch-libs`, `tensorrt-dispatch-cu11`, `tensorrt-dispatch-cu11-bindings`, `tensorrt-dispatch-cu11-libs`, `tensorrt-dispatch-cu12`, `tensorrt-dispatch-cu12-bindings`, `tensorrt-dispatch-cu12-libs`, `tensorrt-dispatch-cu13`, `tensorrt-dispatch-cu13-bindings`, `tensorrt-dispatch-cu13-libs`, `tensorrt-rtx`, `tensorrt-rtx-cu12`, `nvidia-tensorrt`, `torch-tensorrt`, `sgl-kernel`, `sglang-kernel`, `vllm-flash-attn`, `flashinfer-cubin`, `flashinfer-jit-cache`, `kernels`, `kernels-community`, `hf-kernels`, `flash-attn`, `flash-attn-3`, `flash-attn3`, `flash-attn-4`, `flash-attn-cute`, `flash-attn-wheels`, `flash-attn-triton`, `flash-attention`, `flashinfer`, `flashinfer-python`, `xformers`, `flash-mla`, `sageattention`, `ring-flash-attn`, `fla-core`, `flash-linear-attention`, `transformer-engine`, `transformer-engine-cu12`, `transformer-engine-cu13`, `transformer-engine-torch`, `transformer-engine-jax`, `deep-gemm`, `auto-gptq`, `autoawq`, `autoawq-kernels`, `gptqmodel`, `marlin`, `marlin-kernels`, `bitblas`, `gemlite`, `flute-kernel`, `humming-kernels`, `fbgemm-gpu`, `fbgemm-gpu-genai`, `fbgemm-gpu-nightly`, `fbgemm-gpu-genai-nightly`, `sonic-moe`, `megablocks`, `grouped-gemm`, `moe-kernels`, `deep-ep`, `triton-kernels`, `liger-kernel`, `quack-kernels`, `unsloth`, `unsloth-zoo`, `causal-conv1d`, `mamba-ssm`.
- Also installed for this task: `qwen-tts` 0.1.1, the checkpoint's reference
  implementation, with the packages it depends on; `torchaudio`, `numpy`, `numba`,
  `soundfile`; and `ffmpeg`, `sox`, `libsndfile` on the system.

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

Each pip step may write at most 92 GiB to the node's disk, its copy of your tree, its temporary files and its wheels together, and hold at most 32 GiB of memory; a step past either is stopped, and the submission scores zero.

`torch` 2.11.0+cu130, `torchaudio` 2.11.0+cu130 and `torchvision` 0.26.0+cu130 are held when your server is built: an install that asks for another version of one of them fails there. Your sandbox sets the same constraints by default (`PIP_CONSTRAINT`, `UV_CONSTRAINT`).

## Your budget
You have 12 hours of wall-clock time for this build. Call `get_budget` (the `budget` MCP server) periodically to see how much remains, and plan your remaining work to stay within it. Leave headroom to commit working code before you run out.

Your sandbox has 229 GiB of memory, of which the harness's own processes may hold up to 4 GiB, and 8 CPUs. Its `/dev/shm` holds up to 229 GiB, and what is there counts in that memory. You can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your session's logs share 240 GiB of writable space, separate from your memory, and hold at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep. The objects your repository's `.git` holds, its history and unreachable objects among them, may come to at most those 240 GiB together uncompressed, and the bundle of main the harness makes of them to at most 240 GiB; past either, the build has no submission and scores zero. Every object git stores as a delta counts with what rebuilding it reads: itself, its delta, and the same for the object it is a delta on, down its chain. Those may come to at most 64 GiB together; past that, the build has no submission and scores zero. Your commands can start a process or a thread only while they hold fewer than 131,072 of them together; past that a fork fails with EAGAIN. Your commands, the agent's own among them, may hold at most 15 inotify instances, 130,048 inotify watches and 124,968 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another. When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending processes does not bring the memory back under it, the sandbox ends. A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on what you last committed on main; past the files or the nesting, it has run out of its writable space, and so it has when one of its processes holds open files the harness cannot list, as a process running a program from a file it cannot read does. Your commands run with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.

When your server is graded, it runs with 1 H200 GPU, 8 CPUs, 236 GiB of memory and 240 GiB of writable space, which holds at most 2,000,000 files, directories and links, with no directory nested more than 256 levels deep; past any of those the server is stopped, and the grading scores zero. Build it for those. `/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and `/engine-cache`. A process of it that holds open files the harness cannot list counts as past its writable space. It and every process it starts may hold at most 16 inotify instances, 131,072 inotify watches and 131,056 namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at once; past that the kernel refuses another.

When your server is graded, the harness records each request and its reply in the drive's trace as the drive goes, and the trace may hold at most 61 GiB of JSON: once what it has recorded, with each request it has sent and not yet recorded counted at the JSON the trace will record of that request, and each reply it has received and not yet recorded counted at the most JSON that reply's bytes can become, reaches that, every further reply fails as your server's failed request. This cap is what the harness's verification of the trace can hold in memory, which is less than it would otherwise allow. One reply may carry at most 256 MiB, and a longer one fails; so does a reply in a content coding other than identity, since the harness asks for uncompressed replies.
