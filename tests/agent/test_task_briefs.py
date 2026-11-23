"""What the brief the agent_build stage writes tells an agent about two tasks: the decode the
TTS gate prices, the report gates' calibrated numbers, the packages its build may not install,
and the multitenant SLO task's goal before its calibration is loaded. A brief is part of its
task's definition, so these hold on the brief the stage renders (``agent_build._Build.brief``),
with the producer's network as the plan gives the stage."""

from __future__ import annotations

import pytest

from hls.pipeline.pipelines import producer_network
from hls.util import remove_tree
from tests.agent import test_stage
from tests.unit.calibration_fixtures import make_report_calibration


@pytest.fixture
def render(tmp_path):
    """The brief the agent_build stage writes for a task and its artifact. One stage per test,
    its network the first artifact's producer's, so a test renders artifacts of one producer;
    the test module's teardown closes its context."""
    built: list = []

    def brief(task, artifact) -> str:
        if not built:
            params = test_stage._params(network=producer_network(artifact.producer))
            built.append(test_stage.briefing(test_stage._ctx(tmp_path, params)))
        return built[0].brief(task, artifact)

    return brief


@pytest.fixture(autouse=True)
def closed_contexts():
    """Close the contexts the test made, as the agent_build stage's own tests do."""
    yield
    while test_stage._CONTEXTS:
        test_stage._CONTEXTS.pop().close()
    while test_stage._SCRATCHES:
        remove_tree(test_stage._SCRATCHES.pop(), ignore_errors=True)


def test_the_tts_brief_states_the_decode_the_gate_prices(render):
    """A contestant decodes what the brief says. If the brief and the reference drift,
    every honest server is graded against a distribution it was never told to sample
    from, and the gate reads that as a deviation.

    The temperature is spliced from the reference, so it cannot drift. The rest is prose
    and is checked as prose: the three things that must not be applied, the two
    structural rules that must, and the checkpoint defaults a server has to override."""
    from tasks.multimodal_tts import task as t

    decoding = render(t.TASK, t.ARTIFACT).split("## Decoding", 1)[1].split("## Build", 1)[0]

    ref = t.VERIFIERS[0].reference
    assert f"temperature {ref.temperature}: no top-k" in decoding
    assert "no top-k, no top-p, no repetition penalty" in decoding
    assert "special ids (2048 through 3071, except end of speech," in decoding
    assert "end of speech is masked at frames 0 and 1" in decoding
    assert "after those masks, so a masked id carries no probability" in decoding
    # The values a server inherits by saying nothing, which is the reason the brief has
    # to tell it to say something.
    assert "temperature 0.9, top-k 50 and a repetition penalty of 1.05" in decoding
    # Top-k 50 appears nowhere else, so the brief never states it as an instruction.
    assert "top-k 50," not in decoding.replace(
        "temperature 0.9, top-k 50 and a repetition penalty of 1.05", "")


def test_the_tts_brief_states_the_report_gates_calibrated_numbers(render):
    """The divergence ceiling and the served-outside limit are stated as numbers once a
    calibration is loaded, as the default brief states them, and as the shared
    qualitative fallback before one is."""
    from tasks.multimodal_tts import task as t

    bare = render(t.TASK, t.ARTIFACT)
    assert "mean exceeds a ceiling calibrated on trusted serving stacks" in bare
    assert "than a fraction calibrated on trusted serving stacks of your served" in bare

    # One engine measured with no spread, so its upper limits are the readable numbers.
    cal = make_report_calibration(t.TASK, engines={"vllm": {"mean_kl": 9.5e-4, "se_kl": 0.0,
                                                            "outside_rate": 0.34, "se_outside": 0.0}})
    assert (cal.kappa, cal.outside_ceiling) == (9.5e-4, 0.34)
    gate = t.VERIFIERS[0].model_copy(update={"calibration": cal})
    task = t.TASK.model_copy(update={"verifiers": [gate, *t.TASK.verifiers[1:]]})
    brief = render(task, t.ARTIFACT)
    assert "mean exceeds 1.90e-03 nats per position" in brief
    assert "than 68.00 percent of your served codes" in brief


def test_the_tts_brief_blocks_exactly_what_the_package_egress_denies(render):
    """This task replaces the shared build_instructions block, so its Blocked line has to
    come from the producer's deny list as the shared one does. A hand-written line names
    whatever its author remembered, and the brief becomes a request rather than a rule.
    vllm-omni is the production framework the score is measured against."""
    from tasks.multimodal_tts import task as t

    packages = t.ARTIFACT.producer.network.packages
    assert packages.mode == "deny" and "vllm-omni" in packages.entries
    build = render(t.TASK, t.ARTIFACT).split("## Build Instructions", 1)[1]
    blocked = [line for line in build.splitlines() if line.startswith("- Blocked")]
    assert blocked == ["- Blocked (`pip install` refuses these): "
                       + ", ".join(f"`{p}`" for p in packages.entries) + "."]
    assert "- `pip install`: any package except the blocked ones is installable." in build


def test_the_slo_brief_states_its_goal_without_a_calibration(render):
    """Before its calibration is loaded, the multitenant SLO task's goal names the two rules
    qualitatively, with the reference-stack definition in the gate sentence itself, and
    states no calibrated threshold."""
    from tasks.multitenant_slo.task import ARTIFACT, TASK

    # Whitespace-normalized: the template hard-wraps, and where it wraps is not part of
    # the contract being pinned.
    brief = " ".join(render(TASK, ARTIFACT).split())
    assert "exceeds the thresholds" in brief
    assert "HuggingFace transformers, batch size 1, MATH SDPA backend" in brief
    # No calibrated number leaks before calibration: the gate sentence renders no "<pct>% of
    # your served tokens" threshold. (The dev-split guidance's generic "50% higher or lower"
    # is deliberate prose, not a calibrated figure.)
    goal_span = brief.split("## Model")[0].split("must not fail any request")[1]
    assert "% of your served tokens" not in goal_span
    assert "0.3930" not in goal_span and "0.26" not in goal_span
