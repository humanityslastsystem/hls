"""The implementer brief and its sections, rendered as the agent_build stage renders them
(``tests.unit.brief_task.render_brief``), and the prompts the agent harnesses send."""

from __future__ import annotations

import inspect

import pytest

from hls.prompt import ImplementerBrief, budget_brief_section, ralph_goal_prompt, reentry_prompt
from hls.spec.contracts import RequestResponseContract
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
from tests.unit.brief_task import TASK, render_brief


class _TS:
    """Minimal TaskSpec stand-in carrying only what the brief reads."""
    class _M:
        def prompt_fields(self):
            return {"fingerprint": "ckpt@TX", "base_checkpoint": "ckpt",
                    "arch": "TX", "chat_format": "chatml"}
    class _HW:
        def prompt_fields(self):
            return {"accelerator_kind": "B200", "accelerator_count": 1,
                    "memory_gb_per_device": 0, "interconnect": None,
                    "cpu_count": 0, "memory_gb": 0, "storage_gb": 0}
    id = "example-task"
    model = _M()
    hardware = _HW()
    agent = None
    contract = RequestResponseContract(path="/v1/chat/completions",
                                       input_modalities=frozenset({"text"}),
                                       output_modalities=frozenset({"text"}))
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=ShortOutputCap(cap_tokens=64),
                        slo=NoSLO())

    def effective_tenants(self):
        return [Tenant(id="t0", workload=self.workload, verifiers=[])]


def test_default_brief_renders_fields_and_openai_section():
    brief = render_brief(_TS())[0]
    assert "/workspace/server.py" in brief
    assert "POST /v1/chat/completions" in brief   # endpoint line from protocol_section
    assert "messages" in brief                    # OpenAI request example
    assert "Chat format: `chatml`" in brief     # model block rendered (actionable)
    assert "ckpt@TX" not in brief                 # fingerprint dropped -- unactionable to the model
    assert "B200" in brief                        # hardware.accelerator_kind


def test_model_block_omits_chat_template_arguments_when_no_pin():
    class _TSEmptyPin(_TS):
        """``ModelSpec.prompt_fields`` always emits the key, empty when unpinned; the
        bare ``_TS`` omits it, so both shapes are exercised."""
        class _M:
            def prompt_fields(self):
                return {"fingerprint": "ckpt@TX", "base_checkpoint": "ckpt",
                        "arch": "TX", "chat_format": "chatml",
                        "chat_template_kwargs": {}}
        model = _M()

    for taskspec in (_TS(), _TSEmptyPin()):
        brief = render_brief(taskspec)[0]
        assert "Chat template arguments" not in brief
        # The "Chat format" line is immediately followed by the next heading, with no
        # stray blank line or partial pin content left behind by the conditional.
        lines = brief.splitlines()
        idx = lines.index("- Chat format: `chatml`")
        assert lines[idx + 1] == "## Hardware Envelope"


def test_model_block_states_the_chat_template_pin():
    class _TSPinned(_TS):
        class _M:
            def prompt_fields(self):
                return {"fingerprint": "ckpt@TX", "base_checkpoint": "ckpt",
                        "arch": "TX", "chat_format": "chatml",
                        "chat_template_kwargs": {"enable_thinking": True}}
        model = _M()

    brief = render_brief(_TSPinned())[0]
    # The pin line must be its own distinct line right after "Chat format", not glued
    # onto its end (trim_blocks eats the newline after "{% if %}" unless it is made
    # explicit with {{ "\n" }}).
    lines = brief.splitlines()
    idx = lines.index("- Chat format: `chatml`")
    assert lines[idx + 1] == (
        "- Chat template arguments: `enable_thinking=true` -- pass these to "
        "`apply_chat_template` on every request. The reference renders every prompt "
        "with them, so a prompt rendered without them is graded against a different "
        "distribution and fails correctness."
    )


def test_streaming_contract_surfaces_sse_instructions():
    from hls.spec.contracts import RequestResponseContract

    class _TSStream(_TS):
        contract = RequestResponseContract(
            path="/v1/chat/completions", transport="http-sse",
            input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))

    brief = render_brief(_TSStream())[0]
    assert "[DONE]" in brief


def test_task_overrides_a_single_block_via_extends():
    child = ('{% extends "brief" %}'
             '{% block hardware %}CUSTOM-HW-XYZ{% endblock %}')
    brief = render_brief(_TS(), ImplementerBrief(template_str=child))[0]
    assert "CUSTOM-HW-XYZ" in brief   # overridden block content present
    assert "B200" not in brief        # default hardware block replaced
    assert "Chat format: `chatml`" in brief   # other blocks (model) still render from the base


@pytest.mark.parametrize("method", [ImplementerBrief.render_with_provenance, ImplementerBrief._context],
                         ids=["render_with_provenance", "_context"])
def test_every_argument_of_the_brief_is_required(method):
    """The caller gives every section the brief states, so none renders from a default the
    agent_build stage did not choose."""
    parameters = list(inspect.signature(method).parameters.values())[1:]
    assert parameters and all(p.kind is p.KEYWORD_ONLY for p in parameters), parameters
    assert [p.name for p in parameters if p.default is not p.empty] == []


#: A child template stating the two names the base brief leaves to a task.
_AGENTIC_CHILD = ('{% extends "brief" %}{% block workload %}up to {{ turn_bound }} turns'
                  '{% if reference_service %}; SERVICE{% endif %}{% endblock %}')


def test_a_task_template_states_the_turn_bound_of_its_agent():
    """The bound is the agent's ``max_turns``, the one the trajectory driver holds every session
    to. A task without an agent has none, and the base brief states it for no task."""
    class _Agent:
        max_turns = 250

    class _TSAgentic(_TS):
        agent = _Agent()

    assert "up to 250 turns" in render_brief(_TSAgentic(), ImplementerBrief(template_str=_AGENTIC_CHILD))[0]
    assert "up to None turns" in render_brief(_TS(), ImplementerBrief(template_str=_AGENTIC_CHILD))[0]
    assert render_brief(_TSAgentic())[0] == render_brief(_TS())[0]


def test_a_task_template_states_the_reference_service_only_when_the_build_runs_it():
    """The flag is the producer's, which the agent_build stage gives the brief; the base brief
    states nothing of it either way."""
    child = ImplementerBrief(template_str=_AGENTIC_CHILD)
    assert "; SERVICE" in render_brief(_TS(), child, reference_service=True)[0]
    assert "SERVICE" not in render_brief(_TS(), child, reference_service=False)[0]
    assert render_brief(_TS(), reference_service=True)[0] == render_brief(_TS(), reference_service=False)[0]


def test_brief_includes_submission_section_when_provided():
    text, _ = render_brief(TASK, submission_section="SUBMIT VIA GIT REPO XYZ")
    assert "## Submission" in text
    assert "SUBMIT VIA GIT REPO XYZ" in text


def test_brief_omits_an_empty_submission_section():
    text, _ = render_brief(TASK, submission_section="")
    assert "## Submission" not in text


def test_brief_includes_budget_section_when_provided():
    section = budget_brief_section(43200)
    text, _ = render_brief(_TS(), budget_section=section)
    assert "## Your budget" in text
    for banned in ("soft", "advisory", "not enforced", "unenforced"):
        assert banned not in text.lower().split("## your budget")[1][:400]


#: The section's sentences after the wall budget.
_GET_BUDGET = ("Call `get_budget` (the `budget` MCP server) periodically to see how much remains, "
               "and plan your remaining work to stay within it. Leave headroom to commit working "
               "code before you run out.")


@pytest.mark.parametrize("soft_sec, text", [
    pytest.param(43200, "You have 12 hours of wall-clock time for this build. Call `get_budget` (the "
                 "`budget` MCP server) periodically to see how much remains, and plan your remaining "
                 "work to stay within it. Leave headroom to commit working code before you run out.",
                 id="12h"),
    pytest.param(2700, "You have 45 minutes of wall-clock time for this build. Call `get_budget` (the "
                 "`budget` MCP server) periodically to see how much remains, and plan your remaining "
                 "work to stay within it. Leave headroom to commit working code before you run out.",
                 id="45min"),
    pytest.param(3600, f"You have 1 hour of wall-clock time for this build. {_GET_BUDGET}", id="1h"),
    pytest.param(5400, f"You have 1.5 hours of wall-clock time for this build. {_GET_BUDGET}", id="90min"),
])
def test_the_budget_section_states_the_wall_budget_and_the_tool(soft_sec, text):
    assert budget_brief_section(soft_sec) == text


def test_brief_omits_an_empty_budget_section():
    text, _ = render_brief(_TS(), budget_section="")
    assert "## Your budget" not in text


def test_uploads_brief_section_composes_task_text_and_dest():
    from hls.prompt import UPLOAD_DEST, uploads_brief_section

    body = uploads_brief_section("The folder holds the PTX ISA and cuBLAS docs.")
    assert body.startswith("The folder holds the PTX ISA and cuBLAS docs.")
    assert UPLOAD_DEST in body   # the harness names the in-sandbox location
    # No text -> no section (blank and None alike).
    assert uploads_brief_section(None) == ""
    assert uploads_brief_section("   ") == ""


def test_brief_includes_uploads_section_when_given():
    from hls.prompt import UPLOAD_DEST, uploads_brief_section

    # The task authors the WHOLE section, heading included -- the harness adds no
    # "## " title of its own, only the in-sandbox location line.
    section = uploads_brief_section("## CUDA Documentation\nPTX ISA reference docs.")
    text, prov = render_brief(_TS(), uploads_section=section)
    assert "## CUDA Documentation" in text          # the task's own heading, verbatim
    assert "PTX ISA reference docs." in text
    assert UPLOAD_DEST in text
    # Renders after Build Instructions.
    assert text.index("## Build Instructions") < text.index("## CUDA Documentation")
    # Provenance records the block like every other section, sourced to the task.
    uploads = {s.block: s for s in prov.sections}["uploads"]
    assert "PTX ISA reference docs." in uploads.rendered
    assert uploads.source == "task"


def test_brief_omits_an_empty_uploads_section():
    text, prov = render_brief(_TS(), uploads_section="")
    uploads = {s.block: s for s in prov.sections}["uploads"]
    assert uploads.rendered == "" and uploads.source == "base"


def test_brief_states_the_configured_destination():
    """The agent is told where the files are; a stale default here sends it to an empty
    directory and the whole upload is invisible to it."""
    from hls.prompt import UPLOAD_DEST, uploads_brief_section

    text = uploads_brief_section("## Refs", "/workspace/references")
    assert "/workspace/references" in text
    assert UPLOAD_DEST not in text


def test_brief_falls_back_to_the_default_destination():
    from hls.prompt import UPLOAD_DEST, uploads_brief_section

    assert UPLOAD_DEST in uploads_brief_section("## Refs", None)


def test_brief_renders_workload_section():
    """The default brief includes a ## Workload section with arrival, cap, and SLO."""
    brief = render_brief(_TS())[0]
    assert "## Workload" in brief
    assert "all sessions dispatched at once (offline batch)" in brief
    assert "64" in brief          # ShortOutputCap(cap_tokens=64) -> output_cap
    assert "none (no latency SLO)" in brief


def test_task_can_append_to_workload_block():
    """A task template using {{ super() }} inside the workload block appends
    extra lines while keeping the default arrival/SLO lines intact."""
    child = ('{% extends "brief" %}'
             '{% block workload %}{{ super() }}\n- EXTRA-NOTE{% endblock %}')
    brief = render_brief(_TS(), ImplementerBrief(template_str=child))[0]
    assert "all sessions dispatched at once (offline batch)" in brief  # super() preserved
    assert "EXTRA-NOTE" in brief                                        # appended line present


def test_multi_tenant_brief_renders_all_tenants():
    """Multi-tenant task (>1 tenant) renders per-tenant workload subsections."""
    from hls.spec.workload import Gamma, StreamingLatencySLO, ShortOutputCap, ConstantInterTurn, Poisson, Workload

    class _TSMulti(_TS):
        workload = None   # multi-tenant tasks have no top-level workload

        def effective_tenants(self):
            wl_a = Workload(
                arrival=Poisson(rate_per_sec=2.0),
                inter_turn_delay=ConstantInterTurn(gap=1.0),
                output_length_cap=ShortOutputCap(cap_tokens=128),
                slo=StreamingLatencySLO(ttft_ms=3000.0, tpot_ms=80.0),
            )
            wl_b = Workload(
                arrival=Gamma(shape=0.5, scale=2.0),
                inter_turn_delay=ConstantInterTurn(gap=0.0),
                output_length_cap=ShortOutputCap(cap_tokens=128),
                slo=StreamingLatencySLO(ttft_ms=2000.0, tpot_ms=60.0),
            )
            return [
                Tenant(id="corp-a", workload=wl_a, verifiers=[]),
                Tenant(id="corp-b", workload=wl_b, verifiers=[]),
            ]

    brief = render_brief(_TSMulti())[0]
    assert "### Tenant corp-a" in brief
    assert "### Tenant corp-b" in brief
    # The arrival line is stated in sessions and converted to requests, and the sum
    # over tenants heads the section.
    assert "- Sessions: poisson arrivals, 2 sessions/s" in brief
    assert "- Requests: 1 turn/session, 2 requests/s" in brief
    assert "Arrivals are sessions; each turn of a session is one request." in brief
    assert "TTFT < 3000ms, TPOT < 80ms (per request)" in brief
    assert "TTFT < 2000ms, TPOT < 60ms (per request)" in brief


# ---------------------------------------------------------------------------
# `verification`: the calibrated token-level thresholds a task's brief can state
# ---------------------------------------------------------------------------


def _calibration(q_max: float = 0.004, **overrides):
    """A minimal valid Calibration whose worst engine sits exactly at ``q_max`` -- the
    number the brief's threshold is derived from."""
    from tests.unit.calibration_fixtures import make_calibration

    return make_calibration(engines={"eng": {"rates": [q_max] * 3}}, **overrides)


def _token_verifier(**kwargs):
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import TokenFaithfulnessVerifier

    return TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()), **kwargs)


def _ts_with(*verifier_lists):
    """A taskspec stand-in whose tenants carry the given verifier lists."""
    class _TSVerified(_TS):
        workload = None if len(verifier_lists) > 1 else _TS.workload

        def effective_tenants(self):
            wl = _TS.workload
            return [Tenant(id=f"t{i}", workload=wl, verifiers=list(vs))
                    for i, vs in enumerate(verifier_lists)]

    return _TSVerified()


def test_verification_is_none_without_a_calibration_requiring_verifier():
    from hls.prompt import verification_fields

    assert verification_fields(_TS()) is None


def test_verification_is_none_until_the_calibration_is_injected():
    """A task loaded without its calibration (``load_task``'s default) renders the same
    template with no numbers rather than crashing."""
    from hls.prompt import verification_fields

    assert verification_fields(_ts_with([_token_verifier()])) is None


def test_verification_reports_the_injected_thresholds():
    from hls.prompt import verification_fields

    cal = _calibration(q_max=0.004)
    v = verification_fields(_ts_with([_token_verifier(calibration=cal)]))
    assert v["flip_rate_threshold"] == 2.0 * 0.004      # rate_multiplier * q_max
    assert v["flip_rate_pct"] == "0.8000"               # 0.008 of served tokens
    assert v["q_max"] == 0.004
    assert v["rate_multiplier"] == 2.0
    assert v["multiplier_x"] == "2"
    # The floor is derived, not authored: where the rate rule first meets its
    # false-alarm budget at this q_max and this headroom.
    from hls.verify.poisson import rate_rule_min_tokens
    assert v["min_tokens"] == rate_rule_min_tokens(
        cal.q_max, cal.delta_per_rung, 2.0)


def test_verification_reports_the_calibrated_curve_levels():
    """The curve check's level grid is stated as numbers (its per-level ceilings depend
    on the graded trace's length, so the grid is the invariant worth stating)."""
    from hls.prompt import verification_fields

    cal = _calibration()
    v = verification_fields(_ts_with([_token_verifier(calibration=cal)]))
    assert v["flip_curve_levels"] == list(cal.curve_side.levels_nats)


def test_verification_states_the_one_pooled_gate():
    """The token-level gate is pooled: it grades every tenant's events as one
    population and returns one verdict, so several tenants declaring it are declaring
    the same gate. The brief states that gate, and the multiplier it reports is the one
    the threshold was computed from."""
    from hls.prompt import verification_fields

    cal = _calibration()
    v = verification_fields(_ts_with(
        [_token_verifier(calibration=cal, rate_multiplier=2.0)],
        [_token_verifier(calibration=cal, rate_multiplier=2.0)],
    ))
    assert v["flip_rate_threshold"] == 2.0 * 0.004
    from hls.verify.poisson import rate_rule_min_tokens
    assert v["min_tokens"] == rate_rule_min_tokens(cal.q_max, cal.delta_per_rung, 2.0)
    assert (v["rate_multiplier"], v["multiplier_x"]) == (2.0, "2")


def test_verification_states_the_enforced_threshold_not_a_truncation():
    """The stated threshold is the ENFORCED one to 4 significant figures, rounded
    to nearest, never floored, so the brief never states a gate stricter than it is.
    The live case is the shipped multitenant_slo gate: 1.5x the calibration's
    worst clean engine rate, which the brief must state as 0.3930, not 0.393 -- the
    trailing zero is the fourth figure, so the formatting may not strip it."""
    from hls.prompt import verification_fields

    cal = _calibration(q_max=0.0026201847191499356)
    v = verification_fields(_ts_with(
        [_token_verifier(calibration=cal, rate_multiplier=1.5)]))
    assert v["flip_rate_pct"] == "0.3930"
    # And at 4 figures the stated number is the enforced one for any purpose an
    # agent could put it to (relative error < 1e-3, in either direction).
    assert abs(float(v["flip_rate_pct"]) / 100 - v["flip_rate_threshold"]) \
        <= 1e-3 * v["flip_rate_threshold"]

    # No trailing zero to keep: all four figures are significant digits.
    cal = _calibration(q_max=0.002602880006309802)
    v = verification_fields(_ts_with(
        [_token_verifier(calibration=cal, rate_multiplier=1.5)]))
    assert v["flip_rate_pct"] == "0.3904"

    cal = _calibration(q_max=0.000123456)
    v = verification_fields(_ts_with([_token_verifier(calibration=cal)]))
    assert v["flip_rate_pct"] == "0.02469"  # 0.0246912%, nearest at 4 figures


_VERIFICATION_TEMPLATE = (
    '{% extends "brief" %}{% block goal %}'
    '{% if verification %}FAILS above {{ verification.flip_rate_pct }}% of tokens '
    '({{ verification.multiplier_x }}x the floor), '
    'min {{ "{:,}".format(verification.min_tokens) }} tokens'
    '{% else %}a calibrated noise-floor envelope{% endif %}{% endblock %}'
)


def test_brief_renders_the_calibrated_numbers_in_an_overridden_block():
    cal = _calibration(q_max=0.004)
    template = ImplementerBrief(template_str=_VERIFICATION_TEMPLATE)
    text = render_brief(_ts_with([_token_verifier(calibration=cal)]), template)[0]
    # 0.8000% = 2x the fixture's 0.4% floor; 4,375 is the trace length below which the
    # rate rule loses its false-alarm guarantee at that headroom
    # (hls.verify.poisson.rate_rule_min_tokens), not an authored number.
    assert "FAILS above 0.8000% of tokens (2x the floor), min 4,375 tokens" in text
    assert "a calibrated noise-floor envelope" not in text


def test_brief_falls_back_to_prose_when_no_calibration_is_injected():
    template = ImplementerBrief(template_str=_VERIFICATION_TEMPLATE)
    text = render_brief(_ts_with([_token_verifier()]), template)[0]
    assert "a calibrated noise-floor envelope" in text
    assert "FAILS above" not in text


def test_brief_withholds_the_numbers_when_state_thresholds_is_off():
    """state_thresholds=False renders the qualitative fallback branch even with a
    calibration injected: the run is still gated by the calibrated numbers; only
    the brief withholds them."""
    cal = _calibration(q_max=0.004)
    withheld = ImplementerBrief(template_str=_VERIFICATION_TEMPLATE, state_thresholds=False)
    text = render_brief(_ts_with([_token_verifier(calibration=cal)]), withheld)[0]
    assert "a calibrated noise-floor envelope" in text
    assert "FAILS above" not in text


def test_single_tenant_brief_byte_identical_after_multi_tenant_support():
    """The workload section for a single-tenant task renders as a dash list (no
    per-tenant headers, no dataset line).
    """
    brief = render_brief(_TS())[0]
    # No per-tenant subsection headers.
    assert "### Tenant" not in brief
    # The exact dash-list lines.
    assert "- Sessions: all sessions dispatched at once (offline batch)" in brief
    # No rate, so no request line and no aggregate sentence.
    assert "- Requests:" not in brief
    assert "Offered load" not in brief
    assert "- Per-request output cap: 64 tokens" in brief
    assert "- SLO: none (no latency SLO)" in brief


# ---------------------------------------------------------------------------
# The decoding paragraph: the decode a contestant's server has to implement
# ---------------------------------------------------------------------------

#: The first sentence of the plain-greedy decoding paragraph.
_PLAIN_DECODING = "**Decoding.** Decode greedily (argmax, no sampling)."


def _penalized_verifier(penalty: float, **kwargs):
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import TokenFaithfulnessVerifier

    return TokenFaithfulnessVerifier(
        reference=HFReference(decode=Greedy(frequency_penalty=penalty)), **kwargs)


def _sampling_verifier(temperature: float, **kwargs):
    """A calibrated verifier whose reference samples. The brief reads the decode off the
    reference, not off the verifier class, so the token gate stands in for the sampling
    one here."""
    from hls.core.decode import Sample
    from hls.reference.references import HFReference
    from hls.verify import TokenFaithfulnessVerifier

    return TokenFaithfulnessVerifier(
        reference=HFReference(decode=Sample(temperature=temperature)), **kwargs)


def test_a_plain_greedy_task_states_the_decoding_paragraph_unchanged():
    """A task with no penalty states the plain-greedy paragraph to the letter."""
    brief = render_brief(_TS())[0]
    assert _PLAIN_DECODING in brief
    assert "frequency penalty" not in brief


def test_a_penalized_task_states_the_penalty_in_the_decoding_paragraph():
    """A server built to plain greedy against a penalized reference fails the gate at
    every position the penalty moved, so the brief has to say the penalty is there and
    how to apply it."""
    ts = _ts_with([_penalized_verifier(0.1)])
    brief = render_brief(ts)[0]

    assert "**Decoding.** Decode greedily after a frequency penalty of 0.1:" in brief
    assert _PLAIN_DECODING not in brief, "the two variants must not both render"
    # The three facts a server has to get right, and the two engines it will be built on.
    assert "Count only this request's output tokens, never the prompt" in brief
    assert "in fp32" in brief
    assert "`frequency_penalty=0.1`" in brief
    assert "`temperature=0`" in brief
    # The float renders as a decimal, not as 0.100000 or a repr.
    assert "0.100000" not in brief


def test_the_penalty_is_stated_before_any_calibration_exists():
    """The preview and export paths render the brief without injecting a calibration.
    The decode is a property of the taskspec, so it must still be stated there --
    otherwise a contestant reading an exported brief builds the wrong server."""
    ts = _ts_with([_penalized_verifier(0.1, calibration=None)])
    brief = render_brief(ts)[0]
    assert "frequency penalty of 0.1" in brief


def test_the_decode_comes_from_the_taskspec_alone():
    from hls.prompt import reference_decode

    assert reference_decode(_TS())["kind"] == "greedy"
    assert reference_decode(_ts_with([_token_verifier()]))["frequency_penalty"] == 0.0
    assert reference_decode(_ts_with([_penalized_verifier(0.1)])) == {
        "kind": "greedy", "frequency_penalty": 0.1, "temperature": 0.0}
    assert reference_decode(_ts_with([_sampling_verifier(1.0)])) == {
        "kind": "sample", "frequency_penalty": 0.0, "temperature": 1.0}


def test_calibrated_verifiers_disagreeing_on_the_decode_are_refused():
    """A brief states one decode. Two decodes have no single paragraph to render, so
    the render fails loud rather than picking one."""
    import pytest

    from hls.prompt import reference_decode

    ts = _ts_with([_penalized_verifier(0.1)], [_penalized_verifier(0.2)])
    with pytest.raises(ValueError, match="(?i)different"):
        reference_decode(ts)

    ts = _ts_with([_penalized_verifier(0.1)], [_sampling_verifier(1.0)])
    with pytest.raises(ValueError, match="(?i)different"):
        reference_decode(ts)


def test_a_sampling_task_states_the_sampling_paragraph():
    """Sampling is a different contract from greedy: the server draws rather than takes
    the argmax, and it reports what it drew from. A brief that rendered the greedy
    paragraph would send every contestant into a gate they cannot pass."""
    ts = _ts_with([_sampling_verifier(1.0)])
    brief = render_brief(ts)[0]

    assert "**Decoding.** Sample at temperature 1." in brief
    assert _PLAIN_DECODING not in brief, "the two variants must not both render"
    assert "frequency penalty of" not in brief
    # The four things a sampling server has to get right.
    assert "draw the next token from that" in brief
    assert "No top-p, top-k or min-p truncation" in brief
    assert "The reported values must be the distribution you sampled from." in brief
    assert "with that token as the last token id served" in " ".join(brief.split())
    assert "(`finish_reason` `length`)" in brief
    # The float renders as a decimal, not as 1.000000 or a repr.
    assert "1.000000" not in brief


def test_the_sampling_paragraph_says_what_verification_does():
    """A contestant who does not know that both the report and the tokens are graded can
    believe that reporting a prettier distribution than it sampled from is free. Without
    an injected calibration the ceiling is stated qualitatively."""
    ts = _ts_with([_sampling_verifier(1.0)])
    brief = render_brief(ts)[0]

    assert "**Verification.**" in brief
    assert "averaged over all\ngraded tokens" in brief
    assert "exceeds a ceiling calibrated on trusted serving stacks" in brief
    assert "faithful draws from the distribution you report" in brief


def test_a_calibrated_sampling_gate_states_its_ceiling():
    """With a report calibration injected, the brief states the number the run is held
    to: the divergence ceiling, the calibrated constant times the declared tolerance."""
    from types import SimpleNamespace

    from hls.prompt import verification_fields

    cal = SimpleNamespace(kappa=3.5e-4, outside_ceiling=0.0013)
    verifier = SimpleNamespace(needs_calibration=True, tolerance=2.0, calibration=cal)
    ts = SimpleNamespace(effective_tenants=lambda: [SimpleNamespace(verifiers=[verifier])])
    fields = verification_fields(ts)
    assert fields["kind"] == "report"
    assert fields["kl_ceiling"] == 7e-4
    assert fields["kl_ceiling_str"] == "7.00e-04"
    # The served-outside limit, the other calibrated constant times the same tolerance,
    # stated as a percentage of served tokens.
    assert abs(fields["outside_limit"] - 0.0026) < 1e-12
    assert fields["outside_limit_pct_str"] == "0.26"


def test_a_decode_with_no_brief_prose_is_refused():
    """Rendering greedy prose for an unrecognized strategy would describe a decode no
    server should implement, so a new decode config has to bring its own paragraph."""
    import pytest

    from hls.core.decode import DecodeConfig
    from hls.prompt import reference_decode

    class _Beam(DecodeConfig):
        def fingerprint(self) -> str:
            return "beam"

    ts = _ts_with([_token_verifier()])
    ts.effective_tenants()[0].verifiers[0].reference.__dict__["decode"] = _Beam()
    with pytest.raises(ValueError, match="(?i)no brief prose"):
        reference_decode(ts)


def test_the_sandbox_section_states_the_sandbox_roles_own_limits():
    from hls.prompt import sandbox_brief_section

    text = sandbox_brief_section(memory_gib=174, harness_gib=4.0, shm_gib=174, cpus=8, disk_gib=180,
                                 chain_gib=64, writable=("/workspace", "/tmp", "/var/tmp", "$HOME"),
                                 command_nproc=131_072, nproc_accounting="uid", ucounts=COMMANDS,
                                 grade_after_sandbox_oom=True)
    assert text == ("Your sandbox has 174 GiB of memory, of which the harness's own processes may hold up to 4 GiB, "
                    "and 8 CPUs. Its `/dev/shm` holds up to 174 GiB, and what is there counts in that memory. You "
                    "can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your session's "
                    "logs share 180 GiB of writable space, separate from your memory, and hold at most 2,000,000 "
                    "files, directories and links, with no directory nested more than 256 levels deep. The objects "
                    "your repository's `.git` holds, its history and unreachable objects among them, may come to at "
                    "most those 180 GiB together uncompressed, and the bundle of main the harness makes of them to at "
                    "most 180 GiB; past either, the build has no submission and scores zero. Every object git stores "
                    "as a delta counts with what rebuilding it reads: itself, its delta, and the same for the object "
                    "it is a delta on, down its chain. Those may come to at most 64 GiB together; past that, the build "
                    "has no submission and scores zero. Your commands "
                    "can start a process or a thread only while the node holds fewer than 131,072 of them for the "
                    "user they run as, the harness's and other builds' included; past that a fork fails with EAGAIN. "
                    "Your commands, the agent's own among them, may hold at most 15 inotify instances, 130,048 "
                    "inotify watches and 1,160,904 namespaces of each kind (user, pid, mnt) at once; past that the "
                    "kernel refuses another. When your sandbox's processes pass its memory, one of them is ended "
                    "(the largest of your commands' processes, or the one the kernel's out-of-memory killer picks) "
                    "and the build goes on; when ending processes does not bring the memory back under it, the "
                    "sandbox ends. A sandbox that ends so, or that writes past its writable space, ends the build, "
                    "which is graded on what you last committed on main; past the files or the nesting, it has run "
                    "out of its writable space, and so it has when one of its processes holds open files the harness "
                    "cannot list, as a process running a program from a file it cannot read does. Your commands run "
                    "with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.")
    # Where the kernel counts the commands' own tasks against their ceiling, that is what it says.
    own = sandbox_brief_section(memory_gib=174, harness_gib=4.0, shm_gib=174, cpus=8, disk_gib=180,
                                chain_gib=64, writable=("/workspace",), command_nproc=131_072,
                                nproc_accounting="user_namespace", ucounts={}, grade_after_sandbox_oom=True)
    assert ("Your commands can start a process or a thread only while they hold fewer than 131,072 of them "
            "together; past that a fork fails with EAGAIN. When your sandbox's processes pass") in own


#: The sandbox section's sentences on a sandbox's end past its memory or its writable space, by
#: whether a sandbox's end on its memory is graded (``run.grade_after_sandbox_oom``).
ENDS_GRADED = ("A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on "
               "what you last committed on main; past the files or the nesting, it has run out of its writable space")
ENDS_ZERO = ("A sandbox that ends so ends the build, which then scores zero: nothing you committed is graded. A "
             "sandbox that writes past its writable space ends the build, which is graded on what you last "
             "committed on main; past the files or the nesting, it has run out of its writable space")


@pytest.mark.parametrize("graded", [True, False], ids=["graded", "zero"])
def test_the_sandbox_section_states_the_rule_for_a_sandbox_that_ends_on_its_memory(graded):
    """The sentence after "the sandbox ends" states the rule the build's plan
    applies to a sandbox's end on its memory, graded on the last commit or scored zero with nothing
    graded; a write past the writable space is graded on the last commit either way. Nothing else
    of the section differs."""
    from hls.prompt import sandbox_brief_section

    def section(rule: bool) -> str:
        return sandbox_brief_section(memory_gib=174, harness_gib=4.0, shm_gib=174, cpus=8, disk_gib=180, chain_gib=64,
                                     writable=("/workspace",), command_nproc=131_072, nproc_accounting="uid",
                                     ucounts={}, grade_after_sandbox_oom=rule)

    text = section(graded)
    stated, other = (ENDS_GRADED, ENDS_ZERO) if graded else (ENDS_ZERO, ENDS_GRADED)
    assert f"the sandbox ends. {stated}, and so it has when one of its processes holds" in text
    assert other not in text
    assert text.replace(stated, other) == section(not graded)


#: A node's commands' limits for eight 1-GPU builds, three namespace kinds of them.
COMMANDS = {"max_user_namespaces": 1_160_904, "max_pid_namespaces": 1_160_904, "max_mnt_namespaces": 1_160_904,
            "max_inotify_instances": 15, "max_inotify_watches": 130_048}


def test_the_per_uid_limits_read_as_one_sentence_naming_each_count():
    from hls.prompt import ucounts_sentence

    assert ucounts_sentence("X", {}) == ""
    assert ucounts_sentence("X", {"max_user_namespaces": 8, "max_net_namespaces": 4}) == (
        "X may hold at most 8 user namespaces and 4 net namespaces at once; past that the kernel refuses another.")
    assert ucounts_sentence("X", {"max_inotify_watches": 1024}) == (
        "X may hold at most 1,024 inotify watches at once; past that the kernel refuses another.")


def test_the_sandbox_sections_entries_and_nesting_are_the_disk_watchs_bounds(monkeypatch):
    """The disk watch counts a tree of more entries, or nested deeper, than it walks as past the
    sandbox's cap (``hls.substrate.memory_watch``); the section reads both bounds from it when it
    renders."""
    from hls.prompt import sandbox_brief_section
    from hls.substrate import memory_watch

    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_ENTRIES", 1_234)
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_DEPTH", 7)
    text = sandbox_brief_section(memory_gib=174, harness_gib=4.0, shm_gib=174, cpus=8, disk_gib=180, chain_gib=64,
                                 writable=("/workspace",), command_nproc=131_072, nproc_accounting="uid", ucounts={},
                                 grade_after_sandbox_oom=True)
    assert "hold at most 1,234 files, directories and links, with no directory nested more than 7 levels deep." in text


def test_the_sandbox_section_names_the_error_the_commands_filter_refuses_with(monkeypatch):
    """The commands' seccomp filter refuses ``prctl(PR_SET_DUMPABLE, 0)`` with
    ``hls.agent.spawner.DUMPABLE_REFUSAL``; the section names that error as it renders."""
    import errno

    from hls.agent import spawner
    from hls.prompt import sandbox_brief_section

    monkeypatch.setattr(spawner, "DUMPABLE_REFUSAL", errno.EACCES)
    text = sandbox_brief_section(memory_gib=174, harness_gib=4.0, shm_gib=174, cpus=8, disk_gib=180, chain_gib=64,
                                 writable=("/workspace",), command_nproc=131_072, nproc_accounting="uid", ucounts={},
                                 grade_after_sandbox_oom=True)
    assert text.endswith("`prctl(PR_SET_DUMPABLE, 0)` fails with EACCES.")


def test_the_serving_section_states_the_graded_servers_figures(monkeypatch):
    from hls.prompt import serving_brief_section
    from hls.substrate import memory_watch

    text = serving_brief_section(gpus=1, gpu_kind="H200", cpus=8, memory_gib=176, writable_gib=180,
                                 served_writable=("/tmp", "/var/tmp", "$HOME", "/engine-cache"),
                                 ucounts={"max_inotify_instances": 16, "max_user_namespaces": 1_160_920})
    assert text == ("When your server is graded, it runs with 1 H200 GPU, 8 CPUs, 176 GiB of memory and 180 GiB of "
                    f"writable space, which holds at most {memory_watch.DISK_WALK_MAX_ENTRIES:,} files, directories "
                    f"and links, with no directory nested more than {memory_watch.DISK_WALK_MAX_DEPTH} levels deep; "
                    "past any of those the server is stopped, and the grading scores zero. Build it for those. "
                    "`/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and "
                    "`/engine-cache`. A process of it that holds open files the harness cannot list counts as "
                    "past its writable space. It and every process it starts may hold at most 16 inotify "
                    "instances and 1,160,920 namespaces of each kind (user) at once; past that the kernel refuses "
                    "another.")
    two = serving_brief_section(gpus=2, gpu_kind="H200", cpus=16, memory_gib=244, writable_gib=180,
                                served_writable=("/tmp",), ucounts={})
    assert "with 2 H200 GPUs, 16 CPUs" in two and "write only to `/tmp`." in two
    # The walk's bounds are read from the watch's constants when the section renders.
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_ENTRIES", 4_321)
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_DEPTH", 9)
    patched = serving_brief_section(gpus=1, gpu_kind="H200", cpus=8, memory_gib=176, writable_gib=180,
                                    served_writable=("/tmp",), ucounts={})
    assert "holds at most 4,321 files, directories and links, with no directory nested more than 9 levels" in patched


def test_the_drive_bounds_section_states_the_trace_cap_and_the_reply_cap():
    from hls.prompt import drive_bounds_brief_section

    text = drive_bounds_brief_section(trace_cap_bytes=6 * 2**30, reply_max_bytes=256 * 2**20, verify_limited=False)
    assert text == ("When your server is graded, the harness records each request and its reply in the drive's "
                    "trace as the drive goes, and the trace may hold at most 6 GiB of JSON: once what it has "
                    "recorded, with each request it has sent and not yet recorded counted at the JSON the trace will "
                    "record of that request, and each reply it has received and not yet recorded counted at the most "
                    "JSON that reply's bytes can become, reaches that, every further reply fails as your server's "
                    "failed request. One reply may carry at most 256 MiB, and a longer one fails; so does a reply in a "
                    "content coding other than identity, since the harness asks for uncompressed replies.")
    assert "at most 1,234,567 bytes of JSON" in drive_bounds_brief_section(trace_cap_bytes=1_234_567,
                                                                            reply_max_bytes=2**20, verify_limited=False)


def test_a_cap_the_verification_sets_is_said_to_be_so():
    """A task whose cap is what its verify role can hold, rather than the drives'
    margin (``hls.drive.bounds.bounds_for``), says so beside the cap."""
    from hls.prompt import drive_bounds_brief_section

    limited = drive_bounds_brief_section(trace_cap_bytes=3 * 2**30, reply_max_bytes=256 * 2**20, verify_limited=True)
    ruled = drive_bounds_brief_section(trace_cap_bytes=3 * 2**30, reply_max_bytes=256 * 2**20, verify_limited=False)
    sentence = (" This cap is what the harness's verification of the trace can hold in memory, which is less than "
                "it would otherwise allow.")
    assert limited == ruled.replace("failed request. One", f"failed request.{sentence} One")
    assert sentence not in ruled


# ---------------------------------------------------------------------------
# The prompts the agent harnesses send beside the brief
# ---------------------------------------------------------------------------


def test_goal_prompt_is_a_slash_command_at_message_start():
    # Commands are recognized only at message start; goal text is capped at 4000.
    goal = ralph_goal_prompt()
    assert goal.startswith("/goal ")
    assert len(goal) - len("/goal ") < 4000
    assert "\n" not in goal


def test_goal_prompt_states_the_budget_condition():
    # The only transcript-provable form of "budget exhausted" is the tool's output.
    assert "get_budget" in ralph_goal_prompt()


def test_goal_prompt_is_argv_safe_for_the_agents_pkill_patterns():
    # The goal rides argv; it must never match the agent's own `pkill -f`
    # housekeeping patterns, which target its dev-server filename.
    assert "server.py" not in ralph_goal_prompt()


def test_reentry_prompt_is_one_plain_sentence():
    # Not a /goal: the kickoff's goal is part of the resumed session.
    assert reentry_prompt() == "Your session ended, but its goal is not met."
