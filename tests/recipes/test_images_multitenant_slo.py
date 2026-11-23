"""multitenant_slo's server images bake the environment their measured images baked."""

from __future__ import annotations

import pytest

from hls.recipes.catalog import task_images
from hls.recipes.model import CONSTRAINTS
from hls.recipes.render import baked_environment

#: Each server image's baked environment: what its measured image baked.
BAKED = {
    "multitenant-slo-baseline": {"PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv", "UV_TORCH_BACKEND": "cu130",
        "TORCH_CUDA_ARCH_LIST": "9.0", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
        "UV_NO_CACHE": "1", "VLLM_DO_NOT_TRACK": "1"},
    "multitenant-slo-reference": {"PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv", "UV_TORCH_BACKEND": "cu130",
        "TORCH_CUDA_ARCH_LIST": "9.0", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
        "UV_NO_CACHE": "1"},
    "multitenant-slo-member-sglang": {"PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv",
        "UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1", "UV_NO_CACHE": "1", "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"},
}


@pytest.mark.parametrize("image", sorted(BAKED))
def test_a_server_image_bakes_its_measured_environment(image):
    # The shared base adds the constraint variables, naming its empty constraints file, and vLLM's
    # tracking switch, which the measured images that hold no vLLM did not set.
    shared = {"UV_CONSTRAINT": CONSTRAINTS, "PIP_CONSTRAINT": CONSTRAINTS, "VLLM_DO_NOT_TRACK": "1"}
    assert baked_environment(image, task_images("multitenant_slo").decls) == {**BAKED[image], **shared}
