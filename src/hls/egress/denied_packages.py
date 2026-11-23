"""The package names every agentic build's pull-through index refuses.

A contestant builds its server from the image's torch, triton and CUDA toolkit plus kernels
it writes itself. This list is what keeps that true: the build seal's index refuses these
names at ``pip install``, and the declared-dependency pre-scan refuses a ``pyproject.toml``
that names one. Every agentic producer denies it: ``_default_producer_network`` for a task
that sets no network, and each task with an explicit network passes it too.

It is overcomplete on purpose. The policy matches exact PEP 503-normalized names with no
wildcards (``hls.egress.policy.AccessList.permits``), so each variant wheel is spelled out,
and a name that is not published today costs nothing.

Groups:

  1. Text-model serving frameworks and inference engines, with their own runtimes and
     launchers.
  2. TokenSpeed. A checkpoint's README can advertise it as the model's inference engine,
     so a contestant meets it in vendor documentation. Its kernels, scheduler and backend
     wheels publish separately and install on their own.
  3. Omni-modal serving frameworks. vllm-omni is the stack the speech tasks' baselines are
     measured on, so installing it reproduces the baseline rather than building a server.
  4. Speech recognition engines, most of them Whisper on CTranslate2, whisper.cpp or
     ONNX Runtime.
  5. TensorRT, which compiles a model into its own inference engine, in every CUDA and
     lean/dispatch/RTX wheel it ships as, plus the torch front end.
  6. A framework's kernels published under a name the framework's own is not, and the
     kernel hubs that fetch pre-built kernels for anything on request.
  7. Attention kernels: FlashAttention in each of its published forms (FA3 is
     ``flash-attn-3``, FA4 the CuTe ``flash-attn-4``), FlashInfer, xFormers, MLA,
     linear attention, and Transformer Engine's fused attention.
  8. GEMM kernels, most of them quantized: GPTQ, AWQ, Marlin (``marlin`` is the name the
     reference repo installs under), BitBLAS, FBGEMM and others.
  9. Mixture-of-experts kernels: grouped GEMMs, fused expert layers and the expert
     all-to-all.
 10. Norm, fused-layer and state-space kernels.

Compilers, DSLs and the toolchain stay installable: triton, tilelang, nvidia-cutlass,
nvidia-cutlass-dsl, cuda-python, torch, transformers.
"""

from __future__ import annotations

DENIED_PACKAGES: tuple[str, ...] = (
    # 1
    "vllm", "vllm-cpu", "vllm-tpu", "vllm-rocm", "nano-vllm",
    "sglang", "sglang-router",
    "lmdeploy", "turbomind",
    "tensorrt-llm", "text-generation-inference", "optimum-nvidia",
    "llama-cpp-python", "ctransformers",
    "exllama", "exllamav2", "exllamav3",
    "ktransformers", "kt-kernel", "lightllm", "aphrodite-engine",
    "mlc-llm", "mlc-ai", "mlc-ai-nightly",
    "mlc-llm-cu121", "mlc-llm-cu122", "mlc-llm-cu124", "mlc-llm-cu128",
    "mlc-llm-nightly-cu121", "mlc-llm-nightly-cu122", "mlc-llm-nightly-cu124",
    "mlc-llm-nightly-cu128", "mlc-llm-nightly-cu130",
    "mlc-ai-cu121", "mlc-ai-cu122",
    "mlc-ai-nightly-cu121", "mlc-ai-nightly-cu122", "mlc-ai-nightly-cu124",
    "mlc-ai-nightly-cu128", "mlc-ai-nightly-cu130",
    "deepspeed", "deepspeed-mii", "deepspeed-kernels",
    "mistral-inference", "openllm", "ai-dynamo", "ai-dynamo-runtime", "lmcache",
    "lightseq", "fastertransformer", "ipex-llm", "petals", "medusa-llm", "airllm",
    # 2
    "tokenspeed", "tokenspeed-kernel", "tokenspeed-scheduler", "tokenspeed-triton",
    "tokenspeed-deepgemm", "tokenspeed-flashmla", "tokenspeed-mla", "tokenspeed-nightly",
    "tokenspeed-proton",
    # 3
    "vllm-omni", "sglang-omni",
    # 4
    "ctranslate2", "faster-whisper", "whisperx", "whisper-ctranslate2",
    "insanely-fast-whisper", "whisper-s2t", "wyoming-faster-whisper",
    "faster-whisper-server", "whisper-live", "whisperlivekit", "whisper-cpp-python",
    "pywhispercpp", "whispercpp", "sherpa-onnx", "funasr",
    # 5
    "tensorrt", "tensorrt-bindings", "tensorrt-libs",
    "tensorrt-cu11", "tensorrt-cu11-bindings", "tensorrt-cu11-libs",
    "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs",
    "tensorrt-cu13", "tensorrt-cu13-bindings", "tensorrt-cu13-libs",
    "tensorrt-lean", "tensorrt-lean-bindings", "tensorrt-lean-libs",
    "tensorrt-lean-cu11", "tensorrt-lean-cu11-bindings", "tensorrt-lean-cu11-libs",
    "tensorrt-lean-cu12", "tensorrt-lean-cu12-bindings", "tensorrt-lean-cu12-libs",
    "tensorrt-lean-cu13", "tensorrt-lean-cu13-bindings", "tensorrt-lean-cu13-libs",
    "tensorrt-dispatch", "tensorrt-dispatch-bindings", "tensorrt-dispatch-libs",
    "tensorrt-dispatch-cu11", "tensorrt-dispatch-cu11-bindings",
    "tensorrt-dispatch-cu11-libs",
    "tensorrt-dispatch-cu12", "tensorrt-dispatch-cu12-bindings",
    "tensorrt-dispatch-cu12-libs",
    "tensorrt-dispatch-cu13", "tensorrt-dispatch-cu13-bindings",
    "tensorrt-dispatch-cu13-libs",
    "tensorrt-rtx", "tensorrt-rtx-cu12", "nvidia-tensorrt", "torch-tensorrt",
    # 6
    "sgl-kernel", "sglang-kernel", "vllm-flash-attn", "flashinfer-cubin",
    "flashinfer-jit-cache", "kernels", "kernels-community", "hf-kernels",
    # 7
    "flash-attn", "flash-attn-3", "flash-attn3", "flash-attn-4", "flash-attn-cute",
    "flash-attn-wheels", "flash-attn-triton", "flash-attention",
    "flashinfer", "flashinfer-python", "xformers", "flash-mla", "sageattention",
    "ring-flash-attn", "fla-core", "flash-linear-attention",
    "transformer-engine", "transformer-engine-cu12", "transformer-engine-cu13",
    "transformer-engine-torch", "transformer-engine-jax",
    # 8
    "deep-gemm", "auto-gptq", "autoawq", "autoawq-kernels", "gptqmodel",
    "marlin", "marlin-kernels", "bitblas", "gemlite", "flute-kernel", "humming-kernels",
    "fbgemm-gpu", "fbgemm-gpu-genai", "fbgemm-gpu-nightly", "fbgemm-gpu-genai-nightly",
    # 9
    "sonic-moe", "megablocks", "grouped-gemm", "moe-kernels", "deep-ep", "triton-kernels",
    # 10
    "liger-kernel", "quack-kernels", "unsloth", "unsloth-zoo", "causal-conv1d",
    "mamba-ssm",
)
