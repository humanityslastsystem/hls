# Offline toolchain references

Every library and language the build sandbox ships, documented here. This tree is
read-only reference material; `docs/` in the workspace is yours to write in. The sandbox is
air-gapped, so this is the whole of what is readable -- there is no network fallback.

Formats are upstream-as-authored (`.md` / `.rst`), except the five NVIDIA math libraries
and cuDNN, which upstream publishes only as HTML. Nothing here is machine-converted: a bad
conversion reads worse than tag noise under `grep`.

| directory | what | format |
| --------- | ---- | ------ |
| `ptx/` | PTX ISA reference (`ptx-docs/`), plus `ptx-isa.md` overview and 247 figures | 406 md |
| `cuda-runtime/` | CUDA Runtime API (`cuda-runtime-docs/`) + `cuda-runtime.md` overview | 105 md |
| `cuda-driver/` | CUDA Driver API (`cuda-driver-docs/`) + `cuda-driver.md` overview | 130 md |
| `cuda-python/` | CUDA Python bindings: `cuda_bindings`, `cuda_core`, `cuda_pathfinder` | 161 rst |
| `cutlass/` | CUTLASS C++ (`cpp/`, `operators/`) and the CuTe Python DSL (`pythonDSL/`, `pythonDSL-api/`) | 65 rst + 38 md |
| `cccl/` | CUDA C++ Core Libraries: CUB, Thrust, libcu++ | 427 rst |
| `cublas/` | cuBLAS + cuBLASLt reference | html |
| `cudnn/` | cuDNN developer guide + API + fused-attention/GEMM recipes; `cudnn-frontend-README.md` | 76 html + 1 md |
| `nvrtc/` | NVRTC runtime compilation | html |
| `cusparse/`, `cusolver/`, `cufft/` | cuSPARSE / cuSOLVER / cuFFT references | html |
| `triton/` | Triton docs and 27 runnable annotated tutorials (`tutorials/`) | 26 rst + 27 py + 20 md |
| `cutile/` | cuTile Python | 17 rst |
| `torch/` | PyTorch docs, incl. `torch.compile`/Inductor notes and `cpp_extension` | 247 md |
| `rust/` | The Rust Reference (`reference/`) and The Rust Book (`book/`) | 234 md |
| `profiling/` | Nsight Systems / Nsight Compute / NVTX guides, debugging tools, performance traps | 5 md |

Highest-value starting points, by task:

- writing a Triton kernel -> `triton/tutorials/` (`06-fused-attention.py` is FlashAttention,
  `03-matrix-multiplication.py` a tiled GEMM)
- writing raw PTX or reading `ptxas` output -> `ptx/ptx-isa.md`, then `ptx/ptx-docs/`
- CuTe / CUTLASS kernels -> `cutlass/pythonDSL/`, `cutlass/cpp/cute/`
- device-wide scan / reduce / sort primitives -> `cccl/` (CUB)
- calling cuBLASLt directly instead of `torch.matmul` -> `cublas/`
- JIT-compiling kernels at runtime -> `nvrtc/`, `cuda-python/cuda_core/`
- profiling -> `profiling/nsys-guide.md`, `profiling/ncu-guide.md`

## Provenance

Upstream pins and the re-fetch recipe for every directory are in
`tasks/assets/README.md`. The PTX figures are materialized on the host and committed
because the sandbox cannot reach the network: all 267 image links resolve to
`ptx/_images/` (verified 0 broken).
