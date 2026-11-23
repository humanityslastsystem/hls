"""The ToolAssets seam on AgentSpec."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from hls.spec.agent import AgentSpec, ToolAssets
from hls.spec.tools import LocalCallableToolRunner, ToolRunner, ToolSpec
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.unit.test_trajectory_driver import _agent


class FakeAssets(ToolAssets):
    name = "fake"
    memory_mb = 1024
    gpu = True

    def identity(self) -> str:
        return "fake-v1"

    def hub_revisions(self) -> list[tuple[str, str]]:
        return []

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            '{"identity": "fake-v1", "variant": "%s", "dev": %d}' % (variant, len(dev_row_ids)))


def test_agent_spec_carries_optional_assets():
    base = _agent()
    assert base.assets is None
    with_assets = replace(base, assets=FakeAssets())
    assert isinstance(with_assets, AgentSpec)
    assert with_assets.assets.name == "fake" and with_assets.assets.identity() == "fake-v1"
    assert isinstance(with_assets.assets.build.__self__, ToolAssets)
    assert Path("/data/tools/fake").name == with_assets.assets.name


def test_tool_runner_warmup_defaults_to_a_noop_and_local_runner_calls_its_hook():
    calls = []

    async def warm():
        calls.append(1)

    plain = LocalCallableToolRunner(specs=[ToolSpec(name="t", description="d")],
                                    callables={"t": lambda: {}})
    hooked = LocalCallableToolRunner(specs=[ToolSpec(name="t", description="d")],
                                     callables={"t": lambda: {}}, warmup=warm)
    assert isinstance(plain, ToolRunner)
    asyncio.run(plain.warmup())
    asyncio.run(hooked.warmup())
    assert calls == [1]


def test_driver_for_sizes_the_trajectory_driver_from_its_assets():
    """The driver role holds what the tool assets ask for: their bundle's memory and, when the
    tool encodes on an accelerator, one GPU of the task's kind. Without assets it asks for
    nothing beyond the role's own shape."""
    from hls.core.engine import ResourceRequest
    from hls.spec.drivers import TrajectorySessionDriver, driver_for

    plain = driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC, gpu_kind="H200")
    assert isinstance(plain, TrajectorySessionDriver)
    assert plain.resources == ResourceRequest(gpu_count=0)

    with_assets = driver_for(replace(_agent(), assets=FakeAssets()), request_timeout_sec=REQUEST_TIMEOUT_SEC,
                             gpu_kind="H200")
    assert with_assets.resources == ResourceRequest(gpu_count=1, gpu_kind="H200", memory_mb=1024)


def test_trajectory_driver_warms_the_tool_runner_and_framework_before_its_clock():
    """Both pre-clock loads run after the artifact warmup and before the first send.

    A framework that loads anything -- a tokenizer, for one that counts its own
    prompts -- has the same problem the tool runner's corpus does: loaded lazily, the
    seconds land in whichever trajectory happened to need it first."""
    from hls.spec.agent import AgentFramework
    from tests.unit.test_trajectory_driver import (
        _ListDataset, _RecordingClient, _TwoCallFramework, _drive, _driver, _workload,
    )

    order = []

    class _Client(_RecordingClient):
        async def warmup(self):
            order.append("artifact")

        async def submit(self, env, *, attempt=None):
            order.append("send")
            return await super().submit(env, attempt=attempt)

    async def warm():
        order.append("tool")

    class _WarmingFramework(_TwoCallFramework):
        async def warmup(self):
            order.append("framework")

    # The ABC's default loads nothing, so a framework that declares no warmup is
    # untouched by the driver awaiting one.
    assert asyncio.run(AgentFramework.warmup(_TwoCallFramework())) is None

    runner = LocalCallableToolRunner(specs=[ToolSpec(name="lookup", description="d", json_schema={})],
                                     callables={"lookup": lambda q: {"text": f"found {q}"}},
                                     warmup=warm)
    driver = _driver(_WarmingFramework(), _ListDataset(n=1), _workload(),
                     tool_runner=runner)
    _drive(driver, client=_Client())
    assert order[:3] == ["artifact", "tool", "framework"] and "send" in order[3:]


def test_agent_payload_carries_the_assets_identity_only_when_declared():
    from hls.verify.calibration import agent_payload

    assert "assets" not in agent_payload(_agent())
    assert agent_payload(replace(_agent(), assets=FakeAssets()))["assets"] == "fake-v1"


def test_zero_memory_assets_ask_for_no_memory_on_either_slot():
    """Zero is "no ask", not "zero megabytes". Both the driver slot and a pool member's
    leave memory to the substrate, and the one rule lives on the assets."""
    from hls.calibrate import POOL_ENGINE_NAMES, member_engines
    from hls.spec.drivers import driver_for
    from tests.unit.test_calibrate import _agentic_task, _token_task

    class _NoMemory(FakeAssets):
        memory_mb = 0

    assert _NoMemory().memory_request() is None

    req = driver_for(replace(_agent(), assets=_NoMemory()), request_timeout_sec=REQUEST_TIMEOUT_SEC,
                     gpu_kind="H200").resources
    assert req.gpu_count == 1 and req.memory_mb is None

    task = _token_task(agent=replace(_agentic_task().agent, assets=_NoMemory()))
    for name in POOL_ENGINE_NAMES:
        assert member_engines(task)[name]().resources.memory_mb is None


def test_driver_resources_take_memory_without_a_gpu():
    """Assets that only load a bundle get the memory and no card."""
    from hls.spec.drivers import driver_for

    class _MemoryOnly(FakeAssets):
        memory_mb = 512
        gpu = False

    req = driver_for(replace(_agent(), assets=_MemoryOnly()), request_timeout_sec=REQUEST_TIMEOUT_SEC,
                     gpu_kind="H200").resources
    assert req.gpu_count == 0 and req.gpu_kind is None and req.memory_mb == 512

