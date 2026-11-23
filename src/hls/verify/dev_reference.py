"""A task's dev reference: its reference's answers to the dev rows, each scored with the gate's
own metric, published beside the dev rows for the contestant.

A gate that is relative to a reference asks the contestant to keep a share of what the reference
scores, and a contestant cannot check that without knowing what the reference scored: the
decode, the metric's normalizer and its edge cases each move the number. So the reference answers
the dev rows once, each answer is scored as the gate scores it, and the result is handed over row
by row, which lets a contestant check its own implementation of the metric example by example.
The dev rows only: the test split it is measured on stays withheld.

A data source asks for one by naming the file it is published under
(``DataSource.measured_dev_reference``). It is a measurement, kept as the other measurements are
(``hls.measured``): the store's ``measured/<task id>/dev_reference.json``, written by the oracle
fill (``hls oracle fill``, ``hls.verify.fill``), or the task's default
``defaults/dev_reference.json``. The file is a ``DevReference`` record: the text each declaring
tenant's contestant gets, byte for byte, and the fingerprint of what the text was measured on
(``dev_reference_fingerprint``), which a reader holds to the task as it is now
(``read_dev_reference``). Each stage whose roles see the dev rows publishes the text beside them
(``hls.provision.mounts.dev_view``); it is not part of the rows entry, so producing it after
provisioning changes no provisioning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from hls.core.capabilities import Generate, cap_method
from hls.core.payloads import Envelope
from hls.core.poly import dump_spec
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.provision.rows import tenant_subpath
from hls.spec.datasets import Split
from hls.spec.drivers import stamp_request, user_message

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec
    from hls.verify.inprocess import EngineHost


class DevReferenceError(Exception):
    """A dev reference that does not fit the task, or a task whose dev reference cannot be
    produced."""


class DevReferenceFile(BaseModel):
    """One tenant's published dev reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The name it is published under beside the tenant's dev rows.
    name: str
    #: Its text, as the contestant gets it.
    text: str


class DevReference(BaseModel):
    """``dev_reference.json``: the dev reference of each tenant that declares one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["dev_reference"] = "dev_reference"
    #: The task module it was measured for.
    task: str
    #: ``dev_reference_fingerprint`` of the task it was measured on.
    dev_fingerprint: str
    #: Each declaring tenant's file, by tenant id.
    tenants: dict[str, DevReferenceFile]


def _graders(tenant: Any) -> list[Any]:
    """The verifiers of *tenant* that score a reference's answers with a metric: each one's
    ``reference`` answers the rows and its ``metric`` scores the answers."""
    return [v for v in tenant.verifiers if getattr(v, "reference", None) is not None
            and getattr(v, "metric", None) is not None]


def declaring_tenants(task: TaskSpec) -> list[Any]:
    """The tenants of *task* whose data source declares a measured dev reference."""
    return [t for t in task.effective_tenants()
            if t.data_source is not None and t.data_source.measured_dev_reference is not None]


def dev_reference_payload(task: TaskSpec) -> dict:
    """What a dev reference is a measurement of, as plain JSON: the task model, and for each
    declaring tenant the name it is published under, its data source (the dev rows), the output
    cap its prompts are stamped with (``dev_prompts``), and the references that answer them and the
    metrics that score the answers. The data source's published files (``dev_reference``) are left
    out: they are disclosure, not rows."""
    tenants = []
    for tenant in declaring_tenants(task):
        source = dump_spec(tenant.data_source)
        source.pop("dev_reference", None)
        graders = _graders(tenant)
        tenants.append({"id": tenant.id, "name": tenant.data_source.measured_dev_reference, "data_source": source,
                        "max_tokens": tenant.workload.output_length_cap.max_tokens(),
                        "references": [dump_spec(v.reference) for v in graders],
                        "metrics": [dump_spec(v.metric) for v in graders]})
    return {"model": dump_spec(task.model), "tenants": tenants}


def dev_reference_fingerprint(task: TaskSpec) -> str:
    """The sha256 of ``dev_reference_payload`` as canonical JSON."""
    return hashlib.sha256(json.dumps(dev_reference_payload(task), sort_keys=True).encode()).hexdigest()


def read_dev_reference(path: Path, task: TaskSpec) -> tuple[DevReference, bytes]:
    """The dev reference record at *path*, held to *task*, and the bytes it was parsed from. Raise
    DevReferenceError for a missing file, one that is not a record, one measured on another dev
    split, reference or metric (``dev_reference_fingerprint``), and one that does not name the
    file each declaring tenant publishes."""
    path = Path(path)
    fix = "produce it again with `hls oracle fill <module>`, which writes it into the store"
    if not path.is_file():
        raise DevReferenceError(f"no dev reference at {path}; {fix}")
    data = path.read_bytes()
    try:
        record = DevReference.model_validate_json(data)
    except ValidationError as e:
        raise DevReferenceError(f"{path} is not a dev reference ({e.error_count()} errors); {fix}") from None
    fingerprint = dev_reference_fingerprint(task)
    if record.dev_fingerprint != fingerprint:
        raise DevReferenceError(f"stale dev reference {path}: it was measured for dev_fingerprint="
                                f"{record.dev_fingerprint!r} but this task's is {fingerprint!r}; {fix}")
    declared = {t.id: t.data_source.measured_dev_reference for t in declaring_tenants(task)}
    if {tenant: f.name for tenant, f in record.tenants.items()} != declared:
        raise DevReferenceError(f"dev reference {path} publishes {sorted(record.tenants)}, not each declaring "
                                f"tenant's file {declared}; {fix}")
    return record, data


def dev_prompts(task: TaskSpec, tenant: Any, rows_root: Path) -> list[tuple[Envelope, Any]]:
    """Each dev row of *tenant*'s data source, read from ``<rows_root>/tenants/<id>/dev``, as the
    prompt a reference is asked (shaped through ``stamp_request``, as the drive shapes a request)
    with the row's ground truth, less the rows the removal list names. Raise DevReferenceError for
    a row of more than one turn: its later prompts carry a server's replies."""
    removed = set(tenant.data_source.removed_rows)
    cap = tenant.workload.output_length_cap.max_tokens()
    out = []
    for item in tenant.data_source.dataset.items_at(Path(rows_root) / tenant_subpath(tenant.id), Split.DEV):
        if item.request_id in removed:
            continue
        if len(item.turns) != 1:
            raise DevReferenceError(f"{tenant.id}/{item.request_id} has {len(item.turns)} turns; a dev reference "
                                    "is produced for single-turn corpora only")
        turn = item.turns[0]
        prompt = stamp_request(turn, [user_message(turn)], tenant_id=tenant.id, session_id=item.request_id,
                               turn_idx=0, max_tokens=cap)
        out.append((prompt, item.ground_truth_turns[0] if item.ground_truth_turns else None))
    return out


def _text(env: Envelope | None) -> str:
    """The text of an answer: its first text payload, or nothing."""
    part = env.first_of("text") if env is not None else None
    return getattr(part, "text", "") or ""


#: The normalizer the transcript and answer metrics score over, as a dev reference names it.
_NORMALIZER = (
    "both sides normalised by Whisper's EnglishTextNormalizer "
    "(transformers.models.whisper.english_normalizer) constructed with an empty spelling "
    "map. That is the normalizer open-asr-leaderboard scores this corpus with: it "
    "lowercases, strips punctuation, and folds numbers and currency to a canonical form, "
    "so a spelled-out number and a formatted one compare equal.")
#: How a transcript's accuracy is scored (``WERMetric``), as a dev reference states it.
_WER_METRIC = ("accuracy = 1 - WER, where WER is word edit distance / reference word count, over "
               + _NORMALIZER + " An empty reference scores 1.0 against an empty prediction and 0.0 otherwise.")
#: How the two arms of a mixed-modality task are scored (``MixedModalityMatchMetric``).
_MIXED_METRIC = (
    "Two arms, gated independently, both must hold. image: SQuAD-style word F1 "
    "between the answer and the reference answer. audio: smoothed sentence BLEU "
    "(exact unigram precision, add-one smoothing for n>=2, standard brevity penalty) "
    "against the reference translation. Both are computed over " + _NORMALIZER
    + " <think>...</think> blocks are stripped before scoring. The arms are not "
    "comparable to each other, which is why neither may subsidise the other.")
#: What every dev reference says of the bar.
_BAR = "The graded bar is the same measurement over the withheld test split."


def _layout(task_id: str, tenant_id: str, graders: list[Any]) -> str:
    """Which document a tenant graded by *graders* publishes: ``transcripts`` for one
    ``WERMetric``, ``arms`` for ``MixedModalityMatchMetric`` arms. Raise DevReferenceError for any
    other metrics, which have no published document."""
    from hls.verify.metrics.asr import WERMetric
    from hls.verify.metrics.omni import MixedModalityMatchMetric

    kinds = {type(g.metric) for g in graders}
    if kinds == {WERMetric} and len(graders) == 1:
        return "transcripts"
    if kinds == {MixedModalityMatchMetric}:
        return "arms"
    raise DevReferenceError(f"{task_id}/{tenant_id} declares a dev reference graded by "
                            f"{sorted(k.__name__ for k in kinds)}, which has no published document: a dev reference "
                            "is published for one WERMetric or for MixedModalityMatchMetric arms")


def _document(layout: str, scored: list[tuple[str, str, str, float]], values: dict[str, float],
              directory: str) -> dict:
    """The published document of a tenant's dev reference laid out as *layout* (``_layout``), from
    *scored*, each row's request id, the label of the grader that scored it, the answer's text and
    its score, in row order, and *values*, each grader's reduced score by label. ``transcripts``
    gets each row's transcript and accuracy; ``arms`` gets each row's arm, answer (its ``<think>``
    blocks stripped, as the metric strips them) and score, with each arm's mean and row count; both
    get the metric in words and a worked example of the normaliser."""
    from hls.verify.metrics.asr import normalize_transcript
    from hls.verify.metrics.omni import strip_think

    if layout == "transcripts":
        rows = {rid: {"transcript": text, "accuracy": round(score, 6)} for rid, _, text, score in scored}
        doc = {"what": f"Oracle transcripts and per-row accuracy on the dev split. {_BAR}", "metric": _WER_METRIC,
               "oracle_mean_accuracy": round(next(iter(values.values())), 6), "n": len(rows)}
        raw = next(iter(rows.values()))["transcript"] if rows else ""
    else:
        rows = {rid: {"arm": arm, "answer": strip_think(text), "score": round(score, 6)}
                for rid, arm, text, score in scored}
        doc = {"what": f"Oracle answers and per-row scores on the dev split for {directory}. {_BAR}",
               "metric": _MIXED_METRIC, **{f"oracle_mean_{arm}": round(v, 6) for arm, v in values.items()},
               "n_by_arm": {arm: sum(1 for _, a, _, _ in scored if a == arm) for arm in values}, "n": len(rows)}
        raw = next(iter(rows.values()))["answer"] if rows else ""
    doc["normalised_example"] = {"raw": raw, "normalised": normalize_transcript(raw)}
    doc["rows"] = rows
    return doc


async def produce(task: TaskSpec, module: str, rows_root: Path, *, host: EngineHost) -> DevReference:
    """The dev reference of *task*, the ``TASK`` of *module*: for each declaring tenant, its
    references' answers to its dev rows (``dev_prompts``), one reference engine at a time on
    *host*, each answer scored by each grading verifier's metric (``_graders``), published as
    ``_document`` lays it out. Raise
    DevReferenceError for a task that declares none, and for a declaring tenant with no verifier
    that scores a reference's answers or whose metrics have no published document (``_layout``),
    each before any engine runs."""
    tenants = declaring_tenants(task)
    if not tenants:
        raise DevReferenceError(f"{task.id} declares no measured dev reference (DataSource.measured_dev_reference)")
    layouts = {}
    for tenant in tenants:
        graders = _graders(tenant)
        if not graders:
            raise DevReferenceError(f"{task.id}/{tenant.id} declares a dev reference but no verifier that scores a "
                                    "reference's answers with a metric")
        layouts[tenant.id] = _layout(task.id, tenant.id, graders)
    files = {}
    for tenant in tenants:
        graders = _graders(tenant)
        asked = dev_prompts(task, tenant, rows_root)
        answers: dict[str, dict[str, Envelope]] = {}
        for grader in graders:
            fingerprint = grader.reference.fingerprint(task)
            if fingerprint in answers:
                continue
            engine = grader.reference.build_engine(task)
            proxy, release = await host.acquire(engine)
            try:
                replies = await getattr(proxy, cap_method(Generate))([p for p, _ in asked])
            finally:
                await release()
            answers[fingerprint] = {p.request_id: r for (p, _), r in zip(asked, replies, strict=True)}
        verdicts: dict[str, tuple[str, str, float]] = {}
        values: dict[str, float] = {}
        for index, grader in enumerate(graders):
            replies = answers[grader.reference.fingerprint(task)]
            events = [TraceEvent(request_id=p.request_id, input=p, response=ArtifactResponse(output=replies[p.request_id]),
                                 ground_truth=truth, t_send=0.0) for p, truth in asked]
            result = grader.metric.score(events)
            label = str(getattr(grader.metric, "arm", None) or index)
            values[label] = result.value
            for p, _ in asked:
                verdict = result.per_event.get(p.request_id)
                if verdict is not None and verdict.score is not None:
                    verdicts[p.request_id] = (label, _text(replies[p.request_id]), verdict.score)
        scored = [(p.request_id, *verdicts[p.request_id]) for p, _ in asked if p.request_id in verdicts]
        doc = _document(layouts[tenant.id], scored, values, module.rsplit(".", 2)[-2])
        files[tenant.id] = DevReferenceFile(name=tenant.data_source.measured_dev_reference,
                                            text=json.dumps(doc, indent=1) + "\n")
    return DevReference(task=module, dev_fingerprint=dev_reference_fingerprint(task), tenants=files)
