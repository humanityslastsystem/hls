"""Where a serving stack keeps what it builds or tunes at load, and the variables that move each such
directory into the engine cache.

A calibration's roles and a served artifact run their engines with an engine cache mounted at
``/engine-cache`` (``hls.stages.engine_cache.engine_cache_env``). An engine's owner publishes what
its role built there as the entry every later role of that engine is seeded from
(``hls.stages.engine_cache``), dropping the cache's ``ENGINE_CACHE_TMP`` directory first. So a build
or tuning product is reused only when its directory lies in the engine cache, outside
``ENGINE_CACHE_TMP``; one left in the role's home is built again at every load.

``ENGINE_CACHE_SURVEY`` is every directory the engine packages of the stacks in
``SURVEYED_STACKS`` write such products to, each with the lines that read it, and the variable of
``ENGINE_CACHE_VARIABLES`` that moves it, or why none does. A citation names its file under the
image's site-packages, then the letters of the stacks it holds for; one with no letter holds for
every stack that carries the package. A member image a task declares holds the engine packages
of one surveyed stack, at its versions, in the locks of its image's chain."""

from __future__ import annotations

from dataclasses import dataclass

#: The engine cache's directory for temporary files, which ``TMPDIR`` points at and an engine's owner
#: removes before it publishes the cache.
ENGINE_CACHE_TMP = "tmp"

#: Each variable that moves a surveyed directory into the engine cache, to the path under the engine
#: cache it points at; the empty path is the engine cache itself.
ENGINE_CACHE_VARIABLES: dict[str, str] = {
    "VLLM_CACHE_ROOT": "vllm",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",
    # FlashInfer appends its own ".cache/flashinfer/..." to this base.
    "FLASHINFER_WORKSPACE_BASE": "",
    "TMPDIR": ENGINE_CACHE_TMP,
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
    "SGLANG_CACHE_DIR": "sglang",
    "SGLANG_JIT_CACHE_DIR": "sglang/jit",
    "SGLANG_DG_CACHE_DIR": "deep_gemm",
    "CUTE_DSL_CACHE_DIR": "cute-dsl",
    "QUACK_CACHE_DIR": "quack",
    "TILELANG_CACHE_DIR": "tilelang",
    "HUMMING_CACHE_DIR": "humming/cache",
    "HUMMING_TMP_DIR": "humming/tmp",
    "OUTLINES_CACHE_DIR": "outlines",
    "CUDA_CACHE_PATH": "nv",
}

#: The engine packages of each stack the survey read, by its letter, at the versions it read:
#: ``V`` the vLLM members, ``S`` the SGLang 0.5.16 members, ``A`` the SGLang 0.5.20 member.
SURVEYED_STACKS: dict[str, dict[str, str]] = {
    "V": {"vllm": "0.22.1", "flashinfer-python": "0.6.11.post2", "apache-tvm-ffi": "0.1.9", "triton": "3.6.0",
          "nvidia-cutlass-dsl": "4.5.2", "quack-kernels": "0.5.0", "tilelang": "0.1.9", "torch": "2.11.0+cu130"},
    "S": {"sglang": "0.5.16", "sglang-kernel": "0.4.5", "sgl-deep-gemm": "0.1.4.post1", "flashinfer-python": "0.6.14",
          "apache-tvm-ffi": "0.1.11", "triton": "3.6.0", "nvidia-cutlass-dsl": "4.6.0", "quack-kernels": "0.6.3",
          "tilelang": "0.1.11", "torch": "2.11.0+cu130"},
    "A": {"sglang": "0.5.20", "sglang-kernel": "0.4.7", "sgl-deep-gemm": "0.2.0", "flashinfer-python": "0.6.18",
          "apache-tvm-ffi": "0.1.11", "triton": "3.7.1", "nvidia-cutlass-dsl": "4.6.2", "quack-kernels": "0.6.4",
          "tilelang": "0.1.12", "torch": "2.13.0+cu130"},
}


@dataclass(frozen=True)
class EngineCacheDir:
    """A directory an engine's packages write to at load or warmup, as the survey found it."""

    #: What is kept there, and where it is by default.
    holds: str
    #: The lines that read its place, by file under site-packages and image.
    source: str
    #: The variable of ``ENGINE_CACHE_VARIABLES`` that moves it into the engine cache; None for one
    #: left where it is.
    moved_by: str | None
    #: Why it is left where it is; None for one that is moved.
    left_because: str | None = None


#: Every directory the survey found; see the module docstring.
ENGINE_CACHE_SURVEY: tuple[EngineCacheDir, ...] = (
    EngineCacheDir(
        "vLLM's compile caches: its torch.compile cache, the Inductor and Triton caches its compiler "
        "sends there, and its DeepGEMM JIT cache; by default ~/.cache/vllm, under XDG_CACHE_HOME",
        "vllm/envs.py:286-289, 653-656; vllm/compilation/compiler_interface.py:477-480; "
        "vllm/utils/deep_gemm.py:210-213 (V)",
        "VLLM_CACHE_ROOT"),
    EngineCacheDir(
        "TorchInductor's compiled kernels and Helion's autotuning results under them; by default under "
        "the temporary directory",
        "torch/_inductor/runtime/cache_dir_utils.py:14-27; helion/autotuner/local_cache.py:99-102 (S); "
        "SGLang sets it only when unset, sglang/srt/environ.py:1927 (A)",
        "TORCHINDUCTOR_CACHE_DIR"),
    EngineCacheDir(
        "Triton's compiled kernels, and tokenspeed-triton's, which reads the same variable; by default "
        "~/.triton/cache",
        "triton/knobs.py:343, 347, 353 (V, S); tokenspeed_triton/knobs.py:344, 348",
        "TRITON_CACHE_DIR"),
    EngineCacheDir(
        "FlashInfer's JIT-built kernels, their sources, object files and build records, and its CuTe DSL "
        "modules; by default under ~/.cache/flashinfer",
        "flashinfer/jit/env.py:52 (V, S), :59 (A); flashinfer/jit/cute_dsl_core.py:113 (A)",
        "FLASHINFER_WORKSPACE_BASE"),
    EngineCacheDir(
        "the extensions tvm-ffi builds, SGLang 0.5.16's JIT kernels among them; by default "
        "~/.cache/tvm-ffi",
        "tvm_ffi/cpp/extension.py:625, 850 (V), :682, 933 (S, A); tvm_ffi/_optional_torch_c_dlpack.py:122; "
        "sglang/jit_kernel/utils/compile.py:244 (S)",
        "TVM_FFI_CACHE_DIR"),
    EngineCacheDir(
        "SGLang 0.5.20's JIT kernels; by default ~/.cache/sglang/jit",
        "sglang/kernels/jit/utils/compile/cache.py:301-303; sglang/srt/environ.py:1171 (A)",
        "SGLANG_JIT_CACHE_DIR"),
    EngineCacheDir(
        "SGLang's own cache: the FlashInfer autotune results it keeps, its torch.compile cache with the "
        "Inductor and Triton caches it sends there, the FA3 kernel lockfile, its custom all-reduce's "
        "peer-access results, and in 0.5.20 its CuTe AOT cache and per-layer embedding files; by "
        "default ~/.cache/sglang",
        "sglang/srt/environ.py:1135 (S), :1164, 1168, 311 (A); "
        "sglang/srt/model_executor/runner/flashinfer_autotune.py:132; sglang/srt/compilation/backend.py:405; "
        "sglang/srt/compilation/compiler_interface.py:196-201; sglang/jit_kernel/flash_attention_v3.py:42; "
        "sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:266 (S)",
        "SGLANG_CACHE_DIR"),
    EngineCacheDir(
        "DeepGEMM's JIT kernels under SGLang, which copies this variable into DG_JIT_CACHE_DIR over any "
        "value that has; by default ~/.cache/deep_gemm",
        "sglang/srt/environ.py:675 (S), :1154 (A); sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:39-41 "
        "(S), :46 (A)",
        "SGLANG_DG_CACHE_DIR"),
    EngineCacheDir(
        "the CuTe DSL's file cache of the kernels it compiles, for FlashInfer, FlashAttention 4, QuACK and "
        "SGLang; by default $TMPDIR/<user>/cutlass_python_cache, in the directory a stage drops",
        "nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/cache_helpers.py:71-73 (V); "
        "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/cache_helpers.py:71-73 (S, A); "
        "flashinfer/data/cutlass/python/CuTeDSL/cutlass/base_dsl/cache_helpers.py:70-71",
        "CUTE_DSL_CACHE_DIR"),
    EngineCacheDir(
        "QuACK's compiled objects (by default under the temporary directory) and its autotuning results "
        "(by default ~/.quack/cache), which share the variable",
        "quack/cache/__init__.py:95; quack/cache/jit.py:49-52; quack/autotuner.py:61-72 (V); "
        "quack/cache/__init__.py:39; quack/cache/jit.py:48-51; quack/autotuner.py:35-46 (S)",
        "QUACK_CACHE_DIR"),
    EngineCacheDir(
        "TileLang's kernel cache and its temporaries; by default ~/.tilelang/cache",
        "tilelang/env.py:262-263 (V), :311-312 (S), :337 (A)",
        "TILELANG_CACHE_DIR"),
    EngineCacheDir(
        "the modules Humming builds, which it keeps under its temporary directory; by default "
        "~/.humming/tmp",
        "humming/utils/jit.py:102-106, 121 (V), :111-116, 130 (S), :109-112 (A)",
        "HUMMING_TMP_DIR"),
    EngineCacheDir(
        "Humming's cache; by default ~/.humming/cache",
        "humming/utils/jit.py:112-115 (V), :121-124 (S)",
        "HUMMING_CACHE_DIR"),
    EngineCacheDir(
        "Outlines' cache of compiled grammars, which vLLM also reads first; by default ~/.cache/outlines",
        "outlines/caching.py:53-54 (S, A); vllm/v1/structured_output/utils.py:156-161 (V)",
        "OUTLINES_CACHE_DIR"),
    EngineCacheDir(
        "the CUDA driver's cache of the PTX it compiles for the GPU at load, keyed by driver and GPU, so an "
        "entry from another driver is a miss; by default ~/.nv/ComputeCache",
        "the CUDA programming guide's environment variables; SGLang sets it only when unset, "
        "sglang/srt/environ.py:1928 (A)",
        "CUDA_CACHE_PATH"),
    EngineCacheDir(
        "the C++ extensions torch builds; by default ~/.cache/torch_extensions, under XDG_CACHE_HOME",
        "torch/utils/cpp_extension.py:419, 2681; torch/_appdirs.py:349",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "Numba's cache of compiled functions; by default under ~/.cache, under XDG_CACHE_HOME",
        "numba/misc/appdirs.py:303; numba/core/caching.py:251, 341",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "cuda-python's program cache; by default ~/.cache/cuda-python, under XDG_CACHE_HOME",
        "cuda/core/cu13/utils/_program_cache/_file_stream.py:79-80",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "cuTile's compiled kernels; by default under ~/.cache, under XDG_CACHE_HOME",
        "cuda/tile/_context.py:96-98 (V); cutile_cache/_env.py:14-16 (S)",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "the cuDNN frontend's compiled GEMM kernels; by default under ~/.cache, under XDG_CACHE_HOME",
        "cudnn/gemm/frost/compiler.py:2155-2159 (S)",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "SGLang 0.5.20's expert packs; by default under ~/.cache, under XDG_CACHE_HOME",
        "sglang/srt/model_loader/expert_pack_runtime.py:29 (A)",
        "XDG_CACHE_HOME"),
    EngineCacheDir(
        "temporary files: nvcc's and ninja's intermediates and whatever else asks for the temporary "
        "directory, kept off the container's small session tmpfs and dropped before a stage publishes",
        "tempfile.gettempdir, which every package above that falls back to the temporary directory reads",
        "TMPDIR"),
    EngineCacheDir(
        "TileLang's JIT workspace, at ~/.cache/tilelang",
        "tilelang/jit/env.py:43-48",
        None,
        "only HOME would move it, and nothing reads the names that file defines for it, nor makes the "
        "directory, so nothing is written there; TileLang's kernels go to TILELANG_CACHE_DIR"),
    EngineCacheDir(
        "FlashAttention's CuTe DSL cache of its kernels",
        "flash_attn/cute/cache_utils.py:34, 39 (S, A); sglang/jit_kernel/flash_attn/cute/cache_utils.py:35, 41 "
        "(S); vllm/vllm_flash_attn/cute/cache_utils.py:34, 39 (V); "
        "flashinfer/cute_dsl/sparse/bsa_utils/cache_utils.py:52 (A)",
        None,
        "it is off unless FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1, so it writes nothing; the kernels are "
        "kept by the CuTe DSL's file cache"),
    EngineCacheDir(
        "FlashInfer's prebuilt cubins, FLASHINFER_CUBIN_DIR",
        "flashinfer/jit/env.py:89 (V, S), :74 (A)",
        None,
        "each image sets it to the read-only /opt/flashinfer-cubins it ships, which moving it would hide"),
    EngineCacheDir(
        "FlashInfer's sparse MLA autotune results, FLASHINFER_AUTOTUNE_DIR",
        "flashinfer/mla/_sparse_mla_sm120.py:1018, 1036 (S), :1040, 1058 (A)",
        None,
        "only an SM120 GPU runs it"),
    EngineCacheDir(
        "FlashInfer's knob cache for expert-parallel MoE, FLASHINFER_MOE_EP_KNOB_CACHE",
        "flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/knob_cache.py:61-66 (A)",
        None,
        "only FlashInfer's expert-parallel MoE kernels write it, and no member runs expert parallelism"),
    EngineCacheDir(
        "the kernels hub's download cache, KERNELS_CACHE",
        "kernels/utils.py:153 (S)",
        None,
        "it holds prebuilt kernels fetched from the Hub, and the roles run offline (HF_HUB_OFFLINE)"),
    EngineCacheDir(
        "vLLM's configuration root, VLLM_CONFIG_ROOT",
        "vllm/envs.py:36 (V)",
        None,
        "it holds configuration and usage statistics, not build products"),
    EngineCacheDir(
        "the metrics files of a server's processes, PROMETHEUS_MULTIPROC_DIR",
        "sglang/srt/utils/common.py:2295-2299 (S); vllm/v1/metrics/prometheus.py:26 (V)",
        None,
        "each server makes a fresh temporary directory for them"),
    EngineCacheDir(
        "Triton's home, TRITON_HOME",
        "triton/knobs.py:343, 353 (V, S)",
        None,
        "it holds Triton's cache only when TRITON_CACHE_DIR is unset, and its dump and override "
        "directories only when debugging turns them on"),
    EngineCacheDir(
        "FlashInfer's trace-apply solutions, at ~/.cache/flashinfer/trace_apply",
        "flashinfer/trace_apply/loaders/python.py:44 (S, A)",
        None,
        "it holds solutions a user supplies, read and never written at load"),
    EngineCacheDir(
        "CCCL's cooperative-algorithms cache, at ~/.cache/cccl",
        "flashinfer/data/cccl/python/cuda_cccl/cuda/coop/_caching.py:11",
        None,
        "it is source data FlashInfer ships, not a package anything imports"),
    EngineCacheDir(
        "SGLang's diffusion caches, SGLANG_DIFFUSION_CACHE_ROOT",
        "sglang/multimodal_gen/envs.py:20 (S)",
        None,
        "only SGLang's diffusion runtime writes them, which no member runs"),
)
