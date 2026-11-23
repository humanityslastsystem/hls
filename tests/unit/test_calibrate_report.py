"""Tests for the report (sampling-gate) calibration pipeline in hls.calibrate.

Everything here runs on the torch-free config host: no GPU, no docker. The pool member
and the pinned reference come through the calibration roles' ``deploy`` seam, composed as
the pipeline composes them (``tests.unit.calibration_campaign``), as they do for the
token-level campaign, so the whole procedure -- the fixed serving count, the
capture round trip with reports, the scoring pass through the gate's own code, the
precision check and the ceiling it sets -- is exercised without a substrate.

The reports the doubles serve are built to make the measurement exact rather than
realistic. A position whose alternatives all carry the same log-probability has no entry
strictly above the keep threshold, so its partition is one atom holding all the mass:
the divergence there is exactly minus the reference's log tail mass, and a test can
state the ceiling it expects. ``_peaked_entry`` covers the ordinary shape beside it.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math

import pytest

from hls.core.decode import Greedy, Sample
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.reference.references import HFReference
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Split
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.verify import ReportFaithfulnessVerifier, TokenFaithfulnessVerifier
from hls.verify.calibration import (
    DEFAULT_EOT_EXPECTED_MAX,
    UPPER_Z,
    CalibrationError,
    ReportCalibration,
)
from hls.verify.report import (
    EOT_EPS_GRID,
    PAD_ID,
    TOP_LOGPROBS_KEPT,
    TOP_LOGPROBS_REQUESTED,
)

from hls.calibrate import (DEFAULT_REPORT_SERVINGS, CaptureStore, ReportDrawStats, RowIndex, capture_items,
                           load_sessions, report_measure, report_verifier, score_report_capture,
                           write_calibration)
from tests.unit.calibration_campaign import in_process_campaign
from tests.unit.test_calibrate import (
    _MEMBER_VERSION,
    _data_source,
    _hardware,
    _make_deploy,
    _model,
    _provisioned,
    _scorer,
    _workload,
)

NEG_INF = float("-inf")

#: The checkpoint's one end-of-turn id, as the reference double reports it. A reply that
#: ends on it and says so satisfies the ending rule.
_EOT_ID = 999


# ---- task and doubles -------------------------------------------------------

def _report_contract() -> RequestResponseContract:
    """The sampling contract: token ids plus a report of the contracted width."""
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol(return_logprobs=True,
                                    top_logprobs=TOP_LOGPROBS_REQUESTED))


def _report_task(**overrides) -> TaskSpec:
    """A single-tenant task whose calibrated verifier is the sampling gate."""
    kwargs = dict(
        id="R", model=_model(), hardware=_hardware(), contract=_report_contract(),
        workload=_workload(), data_source=_data_source(),
        verifiers=[ReportFaithfulnessVerifier(
            reference=HFReference(decode=Sample(temperature=1.0)))],
        scorer=_scorer())
    kwargs.update(overrides)
    return TaskSpec(**kwargs)


def _token_task() -> TaskSpec:
    """The token-level task, for the options that do not apply to it."""
    return TaskSpec(
        id="T", model=_model(), hardware=_hardware(),
        contract=RequestResponseContract(
            path="/v1/chat/completions", input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}), protocol=OpenAIChatProtocol()),
        workload=_workload(), data_source=_data_source(),
        verifiers=[TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))],
        scorer=_scorer())


def _flat_entry(served: int) -> TokenLogprob:
    """One position's report with every alternative equally probable: the partition
    keeps nothing, so all the mass is the tail atom."""
    lp = math.log(1.0 / TOP_LOGPROBS_REQUESTED)
    return TokenLogprob(token="x", token_id=served, logprob=lp,
                        top_logprobs=[TopLogprob(token_id=1000 + i, logprob=lp)
                                      for i in range(TOP_LOGPROBS_REQUESTED)])


def _peaked_entry(served: int) -> TokenLogprob:
    """One position's report concentrated on the token it served: 0.9 on that id and
    the rest spread equally, so the partition keeps exactly that one atom."""
    head, rest = math.log(0.9), math.log(0.005)
    alts = [TopLogprob(token_id=served, logprob=head)]
    alts += [TopLogprob(token_id=2000 + i, logprob=rest)
             for i in range(TOP_LOGPROBS_REQUESTED - 1)]
    return TokenLogprob(token="x", token_id=served, logprob=head, top_logprobs=alts)


def _reply_payload(n_tokens: int, *, peaked: bool = False, first_id: int = 10,
                   finish_reason: str = "stop",
                   logprobs: bool = True) -> TextPayload:
    """One served reply: ``n_tokens`` ids from ``first_id`` up, ending on the
    checkpoint's end-of-turn id, with the report of whichever shape the test wants."""
    tokens = [first_id + i for i in range(n_tokens - 1)] + [_EOT_ID]
    entry = _peaked_entry if peaked else _flat_entry
    return TextPayload(text="x", tokens=tokens, finish_reason=finish_reason,
                       logprobs=[entry(t) for t in tokens] if logprobs else None)


class _ReportMember:
    """Pool-member double for a sampled serving: every reply carries a report.

    Each serving answers the first ``replies`` sessions it is handed with a reply of
    ``tokens`` ids, stamped the way the drive loop stamps them and naming real rows, so
    the capture store can key them against the task. ``no_report_on`` names replies that
    come back with no report at all, which is the contract violation a member with the
    wrong serving flags would produce."""

    def __init__(self, *, replies: int = 3, tokens: int = 4, peaked: bool = False,
                 first_id: int = 10, no_report_on: frozenset[int] = frozenset()):
        self.replies = replies
        self.tokens = tokens
        self.peaked = peaked
        #: The id every reply of this member starts with, so a reference double can
        #: tell one member's captures from another's.
        self.first_id = first_id
        self.no_report_on = no_report_on
        #: one (sessions, seed, max_sessions, agent) per serving driven.
        self.drives: list[tuple] = []

    async def serving_version(self) -> str:
        return _MEMBER_VERSION

    async def serve_traffic(self, sessions, *, seed, request_timeout_sec, max_sessions=None,
                            drive_timeout_sec=None,
                            agent=None, max_context_len=None):
        self.drives.append((sessions, seed, max_sessions, agent))
        tenant, items = sessions[0]
        cap = tenant.workload.output_length_cap.max_tokens()
        events = []
        for i in range(self.replies):
            rid = f"{items[i].request_id}.0"
            payload = _reply_payload(self.tokens, peaked=self.peaked,
                                     first_id=self.first_id,
                                     logprobs=i not in self.no_report_on)
            events.append(TraceEvent(
                request_id=rid, session_id=items[i].request_id, turn_index=0,
                input=Envelope(
                    request_id=rid,
                    metadata={"messages": [{"role": "user", "content": f"q{i}"}],
                              "tenant_id": tenant.id, "max_tokens": cap}),
                response=ArtifactResponse(output=Envelope(request_id=rid,
                                                          contents=[payload])),
                t_send=0.0, t_recv=1.0))
        return Trace(events=events)


_STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"},
          "device_name": "B200"}


class _ReportReference:
    """Pinned-reference double: answers ``reference_logprobs`` at the ids each item's
    report names.

    The log tail mass it reports is ``tail``, or ``tail_of(item, index)`` when a test
    needs it to vary -- by the reply (which is what gives the pooled standard error
    something to measure) or by the member (told apart by the ids its replies start
    with). The log-probability at every kept id is ``head``; a padded slot gathers
    minus infinity, as the real engine's masked gather does."""

    def __init__(self, *, tail: float = -0.1, tail_of=None,
                 head: float = math.log(0.9), max_context: int | None = 8192,
                 eot_price: float = 1e-6, eot_price_of=None, cut: bool = False):
        self.tail_of = tail_of or (lambda item, index: tail)
        self.head = head
        self.max_context = max_context
        #: What it prices ending the turn at, at a position where the model is not
        #: ending one -- ``eot_price``, or ``eot_price_of(item, position)`` when a test
        #: needs the prices to differ within a reply, which is what gives the threshold
        #: search something to choose between. Every reply of the member double ends on
        #: the end-of-turn id, and that last position is priced at 0.9 instead, so no
        #: threshold prices it and an honest serving's count is zero. ``cut`` prices the
        #: ending like any other position, which is what an early stop looks like.
        self.eot_price_of = eot_price_of or (lambda item, position: eot_price)
        self.cut = cut
        #: every item this reference was asked about, in call order.
        self.items: list[Envelope] = []
        #: how many separate calls it took to ask about them.
        self.calls = 0

    async def reference_logprobs(self, items):
        self.calls += 1
        out = []
        for item in items:
            tail = self.tail_of(item, len(self.items))
            self.items.append(item)
            rows = item.metadata["report_ids"]
            served = item.first_of("text").tokens
            prices = [math.log(self.eot_price_of(item, j)) for j in range(len(rows))]
            if served and served[-1] == _EOT_ID and not self.cut:
                prices[-1] = math.log(0.9)
            out.append({
                "p_on_q": [[NEG_INF if i == PAD_ID else self.head for i in row]
                           for row in rows],
                "p_tail_lp": [tail] * len(rows),
                "p_eot_lp": prices,
                "n_ctx": 12, "eot_ids": [_EOT_ID], "max_context": self.max_context})
        return out

    async def stack_report(self):
        return dict(_STACK)


def _run(task, prov, tmp_path, *, reference, member, engines=("vllm",), **kw):
    """One report calibration over the doubles, returning its artifact."""
    return asyncio.run(in_process_campaign(
        task, "tasks.demo.task_report", engines=list(engines),
        deploy=_make_deploy(reference, member=member), rows=prov,
        captures=tmp_path / "caps", **kw))


@pytest.fixture
def prov(tmp_path):
    """The provisioned rows a campaign's sessions are materialized from."""
    return _provisioned(tmp_path)


# ---- the capture round trip --------------------------------------------------

def _rows_of(task, prov) -> RowIndex:
    return RowIndex.from_sessions(load_sessions(task, prov, Split.TEST))


def _drive(task, prov, member=None):
    """One serving's Trace from the member double."""
    return asyncio.run((member or _ReportMember()).serve_traffic(
        load_sessions(task, prov, Split.TEST), seed=1, request_timeout_sec=REQUEST_TIMEOUT_SEC))


def _captured(task, prov, tmp_path, *, member=None, trace=None,
              draw: int = 0) -> tuple[CaptureStore, RowIndex]:
    """One serving written into a fresh capture store, and the row index to read it
    against. Scoring reads the store rather than a list of items, so a test that scores
    goes through the same file a campaign does."""
    trace = _drive(task, prov, member) if trace is None else trace
    items = capture_items(trace, engine="vllm", draw=draw, with_reports=True)
    store = CaptureStore(tmp_path / "caps", "fp")
    store.write_capture("vllm", draw, 7, _MEMBER_VERSION, items)
    return store, _rows_of(task, prov)


def test_a_sampled_capture_round_trips_its_report(prov, tmp_path):
    """The stored capture is what the scoring pass reads, so a report that did not
    survive it would leave the ceiling measured on nothing. Ids, log-probabilities and
    the reported finish reason all come back; the decoded token text does not, because
    nothing reads it."""
    task = _report_task()
    member = _ReportMember(replies=2, tokens=3)
    trace = asyncio.run(member.serve_traffic(
        load_sessions(task, prov, Split.TEST), seed=1, request_timeout_sec=REQUEST_TIMEOUT_SEC))
    items = capture_items(trace, engine="vllm", draw=0, with_reports=True)
    store = CaptureStore(tmp_path / "caps", "fp")
    store.write_capture("vllm", 0, 7, _MEMBER_VERSION, items)

    back, seed, version = store.read_capture("vllm", 0, rows=_rows_of(task, prov))
    assert (seed, version) == (7, _MEMBER_VERSION)
    served = items[0].first_of("text")
    read = back[0].first_of("text")
    assert read.tokens == served.tokens
    assert read.finish_reason == "stop"
    assert [e.token_id for e in read.logprobs] == [e.token_id for e in served.logprobs]
    assert [e.logprob for e in read.logprobs] == [e.logprob for e in served.logprobs]
    assert ([[a.token_id for a in e.top_logprobs] for e in read.logprobs]
            == [[a.token_id for a in e.top_logprobs] for e in served.logprobs])
    assert all(e.token == "" for e in read.logprobs)


def test_a_flip_capture_carries_no_report_fields(prov, tmp_path):
    """The flip measurement reads neither the report nor the reported finish reason, and
    a report is an entry per served token -- so a capture taken for it carries neither."""
    task = _report_task()
    trace = _drive(task, prov, _ReportMember(replies=1, tokens=3))
    items = capture_items(trace, engine="vllm", draw=0)
    assert items[0].first_of("text").logprobs is None
    store = CaptureStore(tmp_path / "caps", "fp")
    store.write_capture("vllm", 0, 7, _MEMBER_VERSION, items)
    with gzip.open(store.capture_path("vllm", 0), "rt", encoding="utf-8") as fh:
        record = json.loads(fh.read().splitlines()[1])
    assert "logprobs" not in record and "finish_reason" not in record


def test_a_scored_report_serving_round_trips_through_the_store(prov, tmp_path):
    """The aggregate reads a scored serving back from the replies its score role
    published, so its records have to survive the store unchanged."""
    task = _report_task()
    store = CaptureStore(tmp_path / "caps", "fp")
    scored = _score_one(task, prov, tmp_path, _ReportReference())
    store.write_scored(scored)
    back = store.read_report_scored("vllm", 0)
    assert back == scored


# ---- the scoring pass --------------------------------------------------------

def _score_one(task, prov, tmp_path, reference, *, member=None,
               trace=None) -> ReportDrawStats:
    """Score one serving of the member double through the pinned-reference double,
    reading it from a capture store as a campaign does."""
    store, rows = _captured(task, prov, tmp_path, trace=trace,
                            member=member or _ReportMember(replies=2, tokens=3))
    return asyncio.run(score_report_capture(
        report_verifier(task), store, "vllm", 0, rows, reference,
        max_context_len=task.max_context_len))


def test_scoring_measures_every_position_against_the_reference(prov, tmp_path):
    """The whole measurement: a flat report's partition is one tail atom holding all
    the mass, so each position's divergence is exactly minus the reference's log tail
    mass. Two replies of three tokens at a 0.1-nat tail is 0.1 per position."""
    task = _report_task()
    reference = _ReportReference()
    scored = _score_one(task, prov, tmp_path, reference)
    assert (scored.seed, scored.serving_version) == (7, _MEMBER_VERSION)
    pooled = scored.pooled()
    assert (pooled.n_replies, pooled.n_tokens) == (2, 6)
    assert pooled.mean_kl == pytest.approx(0.1, rel=1e-9)
    assert pooled.n_violations == 0
    # Every served token fell outside the one atom the flat partition keeps.
    assert pooled.n_outside == 6
    # One reference call for the whole serving: the batch bound is far above two items.
    assert reference.calls == 1
    # The items carry the partition the reference gathered at, one row per served token.
    rows = reference.items[0].metadata["report_ids"]
    assert len(rows) == 3 and all(len(r) == TOP_LOGPROBS_KEPT for r in rows)


def test_a_matching_peaked_report_measures_zero_divergence(prov, tmp_path):
    """The ordinary report shape, and the sanity check on the reduction: a report the
    reference agrees with exactly diverges by nothing."""
    task = _report_task()
    reference = _ReportReference(tail=math.log(0.1), head=math.log(0.9))
    scored = _score_one(task, prov, tmp_path, reference,
                        member=_ReportMember(replies=2, tokens=3, peaked=True))
    pooled = scored.pooled()
    assert pooled.mean_kl == pytest.approx(0.0, abs=1e-12)
    assert pooled.n_outside == 0


def test_a_reply_with_no_report_is_a_violation_outside_the_mean(prov, tmp_path):
    """A pool member is trusted, so a reply whose report cannot describe its own
    emission is a finding about the member rather than a verdict: it is counted, named,
    and left out of every sum, and the replies that were evidence still measure the
    ceiling."""
    task = _report_task()
    scored = _score_one(task, prov, tmp_path, _ReportReference(),
                        member=_ReportMember(replies=2, tokens=3,
                                             no_report_on=frozenset({0})))
    pooled = scored.pooled()
    assert scored.n_violations == 1
    assert [ev.violation for ev in scored.events if ev.violation] == [
        "the reply carries no reported logprobs"]
    assert (pooled.n_replies, pooled.n_tokens) == (2, 3)
    assert pooled.mean_kl == pytest.approx(0.1, rel=1e-9)


def test_an_early_stop_off_the_end_of_turn_ids_is_a_violation(prov, tmp_path):
    """The ending rule reaches the calibration too, and it is the deployed gate's own:
    a reply claiming a normal stop must have ended on one of the checkpoint's
    end-of-turn ids."""
    task = _report_task()
    trace = _drive(task, prov, _ReportMember(replies=1, tokens=3))
    # Re-serve the reply ending on an ordinary token instead of the end-of-turn id.
    tokens = _reply_payload(3).tokens[:-1] + [123]
    trace.events[0].response = ArtifactResponse(output=Envelope(
        request_id=trace.events[0].request_id,
        contents=[TextPayload(text="x", tokens=tokens, finish_reason="stop",
                              logprobs=[_flat_entry(t) for t in tokens])]))
    scored = _score_one(task, prov, tmp_path, _ReportReference(), trace=trace)
    assert scored.n_violations == 1
    assert "not one of the checkpoint's end-of-turn ids" in scored.events[0].violation


def test_a_reference_call_is_bounded_by_positions_as_well_as_replies():
    """A reply's payload is its length, so 256 replies at an 8,192-token cap would
    return 42 million floats in one call. Both sides of the gate ask the same
    predicate, so the bound cannot drift between them."""
    from hls.verify.verifiers import (_REPORT_BATCH, _REPORT_POSITIONS,
                                      report_call_is_full)

    assert not report_call_is_full(1, 1)
    assert report_call_is_full(_REPORT_BATCH, 1)
    assert report_call_is_full(1, _REPORT_POSITIONS)


def test_a_long_serving_is_scored_in_several_calls_and_measures_the_same(
        monkeypatch, prov, tmp_path):
    """Scoring reads and sends a serving in groups, because a serving's reports do not
    fit in memory. Splitting must change no measured number, and the checkpoint facts
    the first group learns have to serve the whole serving -- including a reply refused
    after the last group, whose record names whether it ended in band."""
    import hls.verify.verifiers as verifiers

    task = _report_task()
    member = _ReportMember(replies=3, tokens=3, no_report_on=frozenset({2}))
    whole = _score_one(task, prov, tmp_path / "whole", _ReportReference(), member=member)
    assert whole.pooled().n_replies == 3

    # Three positions per call against three-token replies: one call per scored reply.
    monkeypatch.setattr(verifiers, "_REPORT_POSITIONS", 3)
    reference = _ReportReference()
    split = _score_one(task, prov, tmp_path / "split", reference, member=member)
    assert reference.calls == 2
    assert [ev.to_json() for ev in split.events] == [ev.to_json() for ev in whole.events]
    # The refused reply ends on the checkpoint's end-of-turn id, which only the
    # reference knows and only the earlier groups reported.
    violated = [ev for ev in split.events if ev.violation is not None]
    assert len(violated) == 1 and violated[0].ends_with_eot is True


def test_the_report_is_released_once_its_partition_exists(prov, tmp_path):
    """The report is the expensive part of a captured reply by an order of magnitude,
    and nothing reads it after the partition is built -- so the scorer drops it there,
    which is what keeps a serving of millions of positions inside memory."""
    from hls.calibrate import _captured_event, _without_report

    task = _report_task()
    trace = _drive(task, prov, _ReportMember(replies=1, tokens=3))
    items = capture_items(trace, engine="vllm", draw=0, with_reports=True)
    ev = _captured_event(items[0])
    assert ev.response.output.first_of("text").logprobs is not None

    light = _without_report(ev)
    payload = light.response.output.first_of("text")
    assert payload.logprobs is None
    # Everything the ending rule and the record read survives.
    assert payload.tokens == ev.response.output.first_of("text").tokens
    assert payload.finish_reason == "stop"
    assert light.event_key == ev.event_key
    assert light.input.metadata["max_tokens"] == ev.input.metadata["max_tokens"]


# ---- the campaign and its artifact -------------------------------------------

def test_one_serving_per_engine_sets_the_ceiling(prov, tmp_path):
    """The default campaign: one serving of the task's own traffic per engine, and
    kappa is the highest upper limit over the pool. With every reply diverging by the
    same 0.1 nats the standard error is zero, so the ceiling is the mean itself."""
    task = _report_task()
    member = _ReportMember(replies=3, tokens=4)
    cal = _run(task, prov, tmp_path, reference=_ReportReference(),
               member=member)
    assert isinstance(cal, ReportCalibration)
    assert cal.kind == "report"
    assert set(cal.engines) == {"vllm"}
    engine = cal.engines["vllm"]
    assert (engine.n_servings, engine.n_replies, engine.n_tokens) == (1, 3, 12)
    assert engine.mean_kl == pytest.approx(0.1, rel=1e-9)
    assert engine.se_kl == pytest.approx(0.0, abs=1e-12)
    assert engine.precision_passed
    assert cal.kappa == pytest.approx(engine.mean_kl + UPPER_Z * engine.se_kl, rel=1e-9)
    # The served-outside rate, measured on the same records: a flat report keeps no
    # atom, so every served token was outside and every reply agrees.
    assert engine.outside_rate == pytest.approx(1.0)
    assert engine.se_outside == pytest.approx(0.0, abs=1e-12)
    assert engine.outside_upper == pytest.approx(1.0)
    assert cal.outside_ceiling == pytest.approx(1.0)
    assert len(member.drives) == DEFAULT_REPORT_SERVINGS
    # The artifact is loadable: its own validators re-derive the ceiling it records.
    path = write_calibration(cal, tmp_path)
    assert json.loads(path.read_text())["kind"] == "report"


def test_the_ceiling_covers_the_worst_engine_in_the_pool(prov, tmp_path):
    """Grading is against the most permissive trusted stack, so the ceiling is the
    highest of the pool's upper limits rather than an average of them."""
    task = _report_task()
    # One reference scores both members' captures, telling them apart by the ids their
    # replies start with, so the two measured means do not depend on drive order.
    reference = _ReportReference(
        tail_of=lambda item, index:
            -0.1 if item.first_of("text").tokens[0] == 10 else -0.4)
    members = {"vllm": _ReportMember(replies=3, tokens=4, first_id=10),
               "sglang": _ReportMember(replies=3, tokens=4, first_id=50)}
    deploy = _make_deploy(reference, members=members)
    cal = asyncio.run(in_process_campaign(
        task, "tasks.demo.task_report", engines=["vllm", "sglang"], deploy=deploy,
        rows=prov, captures=tmp_path / "caps"))
    means = {name: rec.mean_kl for name, rec in cal.engines.items()}
    assert means["vllm"] == pytest.approx(0.1, rel=1e-9)
    assert means["sglang"] == pytest.approx(0.4, rel=1e-9)
    assert cal.kappa == pytest.approx(max(r.upper for r in cal.engines.values()),
                                      rel=1e-12)
    assert cal.kappa == pytest.approx(0.4, rel=1e-9)


def test_a_report_campaign_runs_no_fp32_preflight(prov, tmp_path):
    """The fp32-agreement preflight teacher-forces committed probes, which a sampled
    decode refuses -- there is no argmax for a gap to be measured against. So the
    instrument phase brings the reference up for its stack record and nothing else, and
    the artifact carries no preflight to gate on."""
    deploy = _make_deploy(_ReportReference(), member=_ReportMember(replies=3, tokens=4))
    cal = asyncio.run(in_process_campaign(
        _report_task(), "tasks.demo.task_report", engines=["vllm"], deploy=deploy, rows=prov,
        captures=tmp_path / "caps"))
    assert not hasattr(cal, "preflight")
    assert "fp32_anchor" not in deploy.deployed
    assert cal.kappa == pytest.approx(0.1, rel=1e-9)


def test_the_precision_check_names_the_engine_and_the_servings_needed(prov, tmp_path):
    """The one measured number has to be pinned closely enough to grade against. Unlike
    the flip campaign's between-serving spread, this error does fall with more replies,
    so the refusal says how many servings would meet the target."""
    task = _report_task()
    # Two replies diverging by 0.1 and 0.5 nats per position: the mean is 0.3 and the
    # reply-clustered error is 0.2, far outside a 5% target.
    reference = _ReportReference(tail_of=lambda item, index: (-0.1, -0.5)[index % 2])
    with pytest.raises(CalibrationError) as exc:
        _run(task, prov, tmp_path, reference=reference,
             member=_ReportMember(replies=2, tokens=3))
    message = str(exc.value)
    assert "precision target (eps=0.05)" in message
    assert "vllm:" in message and "178 servings are needed" in message


def test_a_calibration_measuring_nothing_is_refused_with_the_reason(prov, tmp_path):
    """Every reply violating the contract leaves the ceiling measured on nothing, which
    must refuse rather than emit a ceiling of zero -- and must name why, because this
    refusal is the whole diagnosis a member serving the wrong report shape gets and the
    reasons are otherwise only inside the gzipped per-reply records."""
    task = _report_task()
    with pytest.raises(CalibrationError) as exc:
        _run(task, prov, tmp_path, reference=_ReportReference(),
             member=_ReportMember(replies=2, tokens=3,
                                  no_report_on=frozenset({0, 1})))
    message = str(exc.value)
    assert "no gradable position at all" in message
    assert "vllm: the reply carries no reported logprobs" in message


def test_a_pool_that_never_served_outside_its_atoms_is_refused(prov, tmp_path):
    """A peaked member whose served token is always its one atom measures a positive
    divergence against a reference that leaves a different tail, so the ceiling is
    fine; what it never measured is the served-outside rate, and a 0 ceiling there
    would fail a faithful trace at its first tail token."""
    task = _report_task()
    with pytest.raises(CalibrationError, match="served no token outside"):
        _run(task, prov, tmp_path,
             reference=_ReportReference(tail=math.log(0.05), head=math.log(0.9)),
             member=_ReportMember(replies=2, tokens=3, peaked=True))


def test_a_zero_divergence_pool_is_refused(prov, tmp_path):
    """A ceiling of zero fails any trace whose reports differ from the reference at
    all, which two correct implementations of one checkpoint always do."""
    task = _report_task()
    with pytest.raises(CalibrationError) as exc:
        _run(task, prov, tmp_path,
             reference=_ReportReference(tail=math.log(0.1), head=math.log(0.9)),
             member=_ReportMember(replies=2, tokens=3, peaked=True))
    assert "the measured ceiling is 0" in str(exc.value)


# ---- the end-of-turn count ---------------------------------------------------

def test_the_threshold_is_the_largest_candidate_that_holds_the_declaration(prov, tmp_path):
    """Which positions the count reads is the one thing it needs chosen, and it is
    chosen by declaring the expected count instead: the widest threshold prices the most
    positions, so a deviation has the fewest places to hide, and the declaration is what
    stops the expectation -- and with it the check's own noise -- from growing."""
    task = _report_task()
    # Per reply: one position priced at 0.02, two at 1e-4, and the ending at 0.9.
    reference = _ReportReference(
        eot_price_of=lambda item, position: 0.02 if position == 0 else 1e-4)
    member = _ReportMember(replies=3, tokens=4)
    wide = _run(task, prov, tmp_path / "wide", reference=reference, member=member)
    # Nothing in this traffic threatens the default declaration, so the grid's widest
    # threshold stands.
    assert wide.end_of_turn.eps == max(EOT_EPS_GRID)
    assert wide.end_of_turn.expected_max == DEFAULT_EOT_EXPECTED_MAX
    assert wide.end_of_turn.ids == [_EOT_ID]
    assert wide.end_of_turn.engines["vllm"].expected == pytest.approx(
        3 * (0.02 + 2e-4))

    tight = _run(task, prov, tmp_path / "tight", reference=reference,
                 member=_ReportMember(replies=3, tokens=4), eot_expected_max=0.01)
    # 0.03 and above would expect 0.0606 endings, over the declaration; 0.01 expects
    # the six cheap positions' 6e-4 and is the largest candidate that fits.
    assert tight.end_of_turn.eps == 0.01
    count = tight.end_of_turn.engines["vllm"]
    assert count.priced_positions == 6
    assert count.expected == pytest.approx(6e-4)
    assert count.observed == 0 and count.p_value == 1.0


def test_a_pool_member_that_ends_turns_the_reference_did_not_price_is_refused(
        prov, tmp_path):
    """The check's null says a faithful engine ends its turns at the rate the reference
    prices. A trusted stack that does not is evidence the null is wrong, and shipping
    the calibration anyway would deploy a check that fails honest servers."""
    task = _report_task()
    with pytest.raises(CalibrationError) as exc:
        _run(task, prov, tmp_path, reference=_ReportReference(cut=True),
             member=_ReportMember(replies=3, tokens=4))
    message = str(exc.value)
    assert "rejects a trusted pool member" in message
    assert "observed 3" in message


def test_a_declaration_no_threshold_can_hold_is_refused(prov, tmp_path):
    """The refusal names the smallest threshold's expectation, which is what says
    whether the traffic or the declaration is the thing to change."""
    task = _report_task()
    with pytest.raises(CalibrationError) as exc:
        _run(task, prov, tmp_path, reference=_ReportReference(),
             member=_ReportMember(replies=3, tokens=4), eot_expected_max=1e-9)
    message = str(exc.value)
    assert "no published end-of-turn threshold holds the expected count" in message
    assert "--eot-expected-max" in message


def test_the_checkpoints_end_of_turn_ids_survive_the_scored_store(prov, tmp_path):
    """The aggregate reads scored servings back from the replies their score roles
    published, and the ids are the reference's to report -- so the scored file carries
    them or the aggregate could not name what the count was taken over."""
    task = _report_task()
    store = CaptureStore(tmp_path / "caps", "fp")
    scored = _score_one(task, prov, tmp_path, _ReportReference())
    assert scored.eot_ids == (_EOT_ID,)
    store.write_scored(scored)
    assert store.read_report_scored("vllm", 0).eot_ids == (_EOT_ID,)


def test_a_serving_scored_without_the_end_of_turn_ids_is_refused(prov, tmp_path):
    """A store with no end-of-turn ids carries no price per position either, so
    pooling it with servings that do would measure the count over part of the
    campaign. The refusal says which file and what to delete."""
    task = _report_task()
    store = CaptureStore(tmp_path / "caps", "fp")
    store.write_scored(_score_one(task, prov, tmp_path, _ReportReference()))
    path = store.replies_path("vllm", 0)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = json.loads(lines[0])
    del header["eot_ids"]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    with pytest.raises(CalibrationError, match="has no eot_ids in its header"):
        store.read_report_scored("vllm", 0)


def test_a_serving_scored_position_by_position_is_refused(prov, tmp_path):
    """A store can carry replies measured at fewer positions than they served. Pooling one
    would divide a divergence summed over part of a reply by the whole of it, so the
    reader refuses it where it can still name the file and what to delete -- rather than
    inside the pooling, which knows neither."""
    task = _report_task()
    store = CaptureStore(tmp_path / "caps", "fp")
    store.write_scored(_score_one(task, prov, tmp_path, _ReportReference()))
    path = store.replies_path("vllm", 0)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    record = json.loads(lines[1])
    assert record["n_tokens"] == len(record["u"]) == 3
    record["n_refused"] = 1
    record["u"] = record["u"][:-1]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join([lines[0], json.dumps(record), *lines[2:]]) + "\n")
    with pytest.raises(CalibrationError) as exc:
        store.read_report_scored("vllm", 0)
    assert "measured 2 of the 3 positions it served" in str(exc.value)
    assert str(path) in str(exc.value)
    assert "Delete the replies file" in str(exc.value)


def test_a_violated_reply_in_a_stored_serving_is_not_read_as_partly_measured(
        prov, tmp_path):
    """A violated reply carries no transforms at all against the positions it served,
    which its reason already says -- so the refusal above is not a statement about one,
    whichever reduction wrote it."""
    task = _report_task()
    store = CaptureStore(tmp_path / "caps", "fp")
    scored = _score_one(task, prov, tmp_path, _ReportReference())
    store.write_scored(scored)
    path = store.replies_path("vllm", 0)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    record = {**json.loads(lines[1]), "u": [], "p_eot_lp": [], "n_refused": 0,
              "violation": "the reply carries no reported logprobs"}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join([lines[0], json.dumps(record), *lines[2:]]) + "\n")
    back = store.read_report_scored("vllm", 0)
    assert back.pooled().n_violations == 1


def test_more_servings_shrink_the_measured_error(prov, tmp_path):
    """``--servings-per-engine`` is the knob the precision refusal points at, so it has
    to actually take that many servings."""
    task = _report_task()
    member = _ReportMember(replies=2, tokens=3)
    cal = _run(task, prov, tmp_path, reference=_ReportReference(),
               member=member, servings_per_engine=3)
    assert len(member.drives) == 3
    assert cal.engines["vllm"].n_servings == 3
    assert cal.engines["vllm"].n_replies == 6


def test_servings_per_engine_is_refused_for_a_token_level_calibration(prov, tmp_path):
    """The flip campaign's serving count is an output of its exposure counter, so that
    the precision check is a single look at a pre-committed size. Accepting a count
    there would silently make it a second look."""
    from tests.unit.test_calibrate import _FakePinned

    with pytest.raises(CalibrationError) as exc:
        asyncio.run(in_process_campaign(
            _token_task(), "tasks.demo.task", engines=["vllm"],
            deploy=_make_deploy(_FakePinned(0.1)), rows=prov,
            captures=tmp_path / "caps", servings_per_engine=2))
    assert "does not apply to a token-level calibration" in str(exc.value)


def test_the_measurement_seam_says_what_each_kind_needs(prov, tmp_path):
    """The report measurement keeps reports in its captures and reads them back as
    report records; the flip one does neither. One campaign, one kind throughout."""
    measure = report_measure(report_verifier(_report_task()), None)
    assert measure.with_reports is True
    assert measure.read is CaptureStore.read_report_scored


# ---- a member whose divergence is not a number ------------------------------

def _inf_stats(engine: str = "vllm", *, n_replies: int = 4, n_inf: int = 2):
    """One serving whose first ``n_inf`` replies carry an infinite divergence.

    Built directly rather than driven, because what is under test is the assembly's
    refusal and not the arithmetic that produced the infinity: a reference price of
    minus infinity at an atom the report weights makes one position's term infinite and
    the reply's sum with it."""
    from hls.calibrate import ReportDrawStats
    from hls.verify.report import ReplyReport
    events = tuple(
        ReplyReport(event_key=f"t0:r{i}", n_tokens=10, n_outside=2,
                    kl_sum=math.inf if i < n_inf else 0.05,
                    s_sum=0.0, var_sum=1.0, n_moment=10,
                    u=tuple(0.5 for _ in range(10)),
                    p_eot_lp=tuple(-20.0 for _ in range(10)),
                    eot_positions=(9,), finish_reason="stop",
                    ends_with_eot=True, at_cap=False, violation=None)
        for i in range(n_replies))
    return ReportDrawStats(engine=engine, draw=0, seed=1, serving_version="v1",
                           eot_ids=(7,), events=events)


def _assemble_with(stats_list, task, *, eps: float = 0.3):
    from hls.calibrate import Campaign, assemble_report
    from hls.verify.calibration import StackInfo
    campaign = Campaign()
    for s in stats_list:
        campaign.add(s)
    stack = StackInfo(torch="2.4.0", runtime={"cuda": "12.4", "cudnn": "90100"},
                      device_name="H200")
    return assemble_report(campaign, task=task, module="tasks.demo.task_report",
                           engines=["vllm"], stack=stack, eps=eps,
                           eot_expected_max=0.5, alpha=0.01, n_servings=1,
                           stopped="count")


def test_a_member_whose_divergence_is_infinite_is_refused_by_name():
    """An infinite mean has to refuse here, naming the replies that carry it.

    Nothing downstream catches it. The precision test is `se <= eps * mean`, which is
    vacuously true at infinity, and the artifact serializes a non-finite float as null,
    so without this the campaign writes a calibration.json whose ceiling is absent and
    every later arm is graded against nothing."""
    task = _report_task()
    with pytest.raises(CalibrationError) as exc:
        _assemble_with([_inf_stats(n_replies=4, n_inf=2)], task)
    message = str(exc.value)
    assert "not a finite number" in message
    assert "vllm" in message and "2 of 4 replies" in message
    assert "t0:r0" in message and "t0:r1" in message


def test_a_member_whose_divergences_are_all_finite_passes_the_check():
    """The refusal is about non-finite sums and nothing else, so an ordinary serving
    reaches the checks that follow it rather than being caught here."""
    task = _report_task()
    try:
        _assemble_with([_inf_stats(n_replies=4, n_inf=0)], task)
    except CalibrationError as exc:
        assert "not a finite number" not in str(exc)
