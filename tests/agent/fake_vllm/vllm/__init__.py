"""A stand-in for vLLM's ``LLM`` and ``SamplingParams``, as far as the in-build reference service
(``hls.reference.service``) uses them, for the build stage's tests: no GPU, no weights.

``FAKE_VLLM_PLAN`` names a JSON file read when an ``LLM`` is made: ``load`` is ``ok`` (the default),
``fail`` (the constructor raises), ``oom`` (it raises CUDA's out-of-memory error), ``hang`` (it never
returns) or ``exit`` (the process exits 3),
after ``load_sec`` seconds; ``record``, when set, is a file the constructor writes its keyword
arguments to. ``generate`` gives every position of the prompt a row that names its token."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def _plan() -> dict:
    path = os.environ.get("FAKE_VLLM_PLAN")
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


class SamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class LLM:
    def __init__(self, *, model: str, **kwargs) -> None:
        plan = _plan()
        time.sleep(float(plan.get("load_sec", 0)))
        load = plan.get("load", "ok")
        if load == "fail":
            raise RuntimeError("fake vllm: no such architecture in the registry")
        if load == "oom":
            raise RuntimeError("fake vllm: CUDA out of memory. Tried to allocate 20.00 GiB")
        if load == "hang":
            while True:
                time.sleep(1)
        if load == "exit":
            os._exit(3)
        if plan.get("record"):
            Path(plan["record"]).write_text(json.dumps({"model": model, **kwargs}))
        config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=kwargs.get("max_model_len") or 4096))
        self.llm_engine = SimpleNamespace(vllm_config=config, engine_core=SimpleNamespace(shutdown=lambda: None))
        self.slept = 0

    def generate(self, prompt: dict, sampling_params: SamplingParams, use_tqdm: bool = False) -> list:
        ids = prompt["prompt_token_ids"]
        rows = [None] + [{t: SimpleNamespace(logprob=-1.0), (t + 1) % 1000: SimpleNamespace(logprob=-2.0)}
                         for t in ids[1:]]
        return [SimpleNamespace(prompt_logprobs=rows)]

    def sleep(self, level: int = 1) -> None:
        self.slept += 1

    def wake_up(self) -> None:
        pass
