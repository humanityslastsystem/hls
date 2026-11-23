"""RolloutScanner: reads rollout records from a directory of ``<task>/<rollout_id>/`` trees, in
the published layout or in the store's.

- Published: a rollout directory holds ``rollout.json`` and ``evals/<id>/{verdict,score,
  grading}.json`` with the trace in one of the published forms (``PUBLISHED_TRACES``), and
  its task directory may hold an ``index.jsonl``. A directory is listed and opened under its own
  name, since the published layout names a cell ``<task>/<repeat>/<arm>/`` while its record keeps
  the run's ids. Nothing is folded from a directory's events: one with no ``rollout.json`` but a
  ``config.json`` is a run that was never sealed, and is shown as ``UNSEALED`` from its
  ``config.json`` and the gradings a later ``hls grade`` gave it, with no build record.
- Store: an instance directory holds ``plan/`` and ``stages/<stage>/MANIFEST.json`` with its
  attempts, and its task directory one index row per sealed instance under ``index/``. An
  instance's records are its finalize stage's outputs, or ``failed/`` for one that ended failed;
  an instance with neither is shown from its latest plan. Its gradings are the instances under
  its ``gradings/``. An eval's trace is the chunked trace its drive stage published, read chunk
  by chunk against ``END.json`` (``hls.trace.store``). An output is read only at the key its
  stage's own attempt holds it under, and only when its path resolves inside the instance being
  shown. A file a finalize scan quarantined is neither served nor read (``_Quarantine``). Pointed
  at a store's ``warmers`` area, the scanner lists warmer instances and their evaluations the
  same way.

A store record's build reaches the page only as the stage pipeline records it, a
``hls.obs.record.RolloutBuild``; one that does not validate as one is served as none, with the
field locations at which it fails and never a value it holds (``_served_build``). A published
record's build, and the producer
in its provenance that names the arm that built it, are in a shape the
viewer does not read: the record is served with no build, and its row with no cost and, unless its
index row names one, no implementer. The page says why each is absent.

A trace is read an event at a time, never whole: a store trace's chunks a line at a time, a
published one decompressed and parsed incrementally (``hls.viewer.published``). Of each event the
viewer keeps only the row its pages show, each text in it cut at ``ROW_TEXT_CHARS`` and each other
value at ``ROW_VALUE_CHARS``, marked where it was cut (``CUT_MARK``), so an eval's index takes at
most its events count times ``row_chars``; every view of the eval is served from that one index.

The store's answers area is never served: a viewing root that is or holds it is refused, and so
is every path through it below one (``check_viewing_root``).

The viewer only reads. Its hls.* imports are leaf modules: the plan, manifest and build record
models, the store's layout and lock records, the trace reader, ``hls.pipeline.follow`` for a lease's
freshness, and ``hls.spec.workload.p99_ms_of`` so that the viewer's p99 uses the exact index rule
of the SLO gate."""

from __future__ import annotations

import functools
import json
import logging
import mimetypes
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from hls.agent.transcript import INDEX_FILE
from hls.drive.record import TRACE_END
from hls.obs.record import RolloutBuild
from hls.pipeline.follow import lease_held
from hls.pipeline.plan import Outcome, Plan, revision_numbers
from hls.spec.workload import p99_ms_of
from hls.store.base import StoreError, check_segment
from hls.store.layout import GRADINGS_DIR, INDEX_DIR, MANIFEST_FILE
from hls.store.lock import GenerationLock, LockRecord
from hls.store.manifest import Manifest
from hls.store.posix import PosixStore
from hls.trace.store import END_FILE, TraceIncomplete, iter_events
from hls.viewer.models import (
    GenericVerifierSummary,
    TaskAbsoluteSummary,
    TaskRelativeSummary,
    TimelinePayload,
    TimelineSLO,
    TimelineTenant,
    TokenLevelSummary,
    TracePage,
    TraceSummary,
    VerifierSummary,
)
from hls.viewer.published import TraceUnreadable, published_events

logger = logging.getLogger(__name__)

#: Where the orchestrator writes a failed instance's records (finalize's ``FAILED_DIR``).
FAILED_DIR = "failed"
#: The records finalize writes.
ROLLOUT_FILE, GRADING_FILE = "rollout.json", "grading.json"
#: The verify and score stages' outputs the trace views read.
VERDICT_FILE, SCORE_FILE = "verdict.json", "score.json"
#: Where agent_build copies the agent's logs among its outputs, and the trajectory harbor writes
#: beside that copy.
TRANSCRIPT_DIR, TRAJECTORY_FILE = "transcript", "trajectory.json"
#: The eval id of an instance's own grading; any other names a grading under it.
TERMINAL_EVAL = "terminal"
#: The backends a published rollout's checkpoint names before its first colon (``docker:<ref>``,
#: ``apptainer:<sif>``), which the summary's backend column shows.
CHECKPOINT_BACKENDS = ("docker", "modal", "apptainer")
#: The status of a published rollout that was never sealed (no ``rollout.json``).
UNSEALED = "unsealed"
#: How a store instance can end, as its plan's outcome says. An instance that ended without a
#: record (void, cancelled, or a plan without a finalize stage) shows its outcome as its status.
INSTANCE_ENDINGS: frozenset[str] = frozenset(get_args(Outcome.model_fields["status"].annotation))
#: The status of a store instance that has not ended and that no orchestrator holds; one that an
#: orchestrator holds is "running", a record's own status.
STALLED = "stalled"

#: A published eval's trace, in the forms the published layout keeps it, in the order the viewer
#: takes the first one present: the plain file and that file gzipped (``PUBLISHED_TRACES``);
#: gzipped and split into numbered parts, ``PART_PREFIX`` with ``00`` on, read concatenated in
#: their numbers' order; and gzipped with its media replaced by digests (``ELIDED_GLOB``:
#: ``trace.elided.json.gz``, ``trace.audio-elided.json.gz``, ``trace.media-elided.json.gz``).
PUBLISHED_TRACES = ("trace.json", "trace.json.gz")
PART_PREFIX = "trace.json.gz.part"
ELIDED_GLOB = "trace.*elided.json.gz"
#: The store area that holds the reference's answers to the withheld test split
#: (``hls.verify.answers.ANSWERS_AREA``). The viewer never serves it.
ANSWERS_AREA = "oracle"
#: The list a finalize scan writes of the files it quarantined (finalize's ``QUARANTINE_FILE``).
QUARANTINE_FILE = "quarantine.json"
#: Where the quarantine lists lie under an instance's directory: its own stages', and those of
#: the gradings under it.
QUARANTINE_GLOBS = (f"stages/*/attempts/*/outputs/{QUARANTINE_FILE}",
                    f"{GRADINGS_DIR}/*/stages/*/attempts/*/outputs/{QUARANTINE_FILE}")
#: The most bytes of a quarantine list the viewer reads; a longer one counts as unreadable.
QUARANTINE_MAX_BYTES = 64 * 2**20
#: How the scrubber marks a secret's value in what it scrubbed (``hls.substrate.secrets.Scrubber``).
_REDACTED = re.compile(r"<redacted:[^>]*>")
#: Name endings of the types a browser runs, or renders with script (HTML, SVG, XML,
#: JavaScript), and media types that name one; such a file is served as plain text.
ACTIVE_SUFFIXES = frozenset({".html", ".htm", ".svg", ".xml", ".xhtml", ".js", ".mjs"})
_ACTIVE_TYPE = re.compile(r"html|xml|javascript|ecmascript")


class ViewRefused(ValueError):
    """What the viewer never serves: a file a finalize scan quarantined, or the store's answers
    area. Its message names neither the file nor its contents."""


def check_viewing_root(root: Path) -> None:
    """Raise ViewRefused for a viewing root that is the store's answers area, lies in it or holds
    it: a root named ``ANSWERS_AREA``, under a directory of that name, or holding one. Further
    below a root the area is refused path by path (``RolloutScanner._rollout_dir``,
    ``resolve_file``, ``list_files``)."""
    resolved = root.resolve()
    if any(p.name == ANSWERS_AREA for p in (resolved, *resolved.parents)) or (resolved / ANSWERS_AREA).is_dir():
        raise ViewRefused(f"{root} is, lies in or holds a store's answers area ({ANSWERS_AREA}/), which the "
                          "viewer never serves; view the store's runs/ or warmers/ instead")


def _shown(key: str) -> str:
    """*key* as finalize writes a store key into a record: a byte of a name that is not UTF-8 as
    its escape (``hls.stages.finalize._shown``)."""
    return key.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


def _quarantine_list(path: Path) -> list[str] | None:
    """The store keys a quarantine list names; None when it cannot be read or is not one."""
    try:
        if path.stat().st_size > QUARANTINE_MAX_BYTES:
            return None
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    files = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(files, dict):
        return None
    return list(files)


@dataclass(frozen=True)
class _Quarantine:
    """The files of a store instance, and of the gradings under it, that a finalize scan
    quarantined. A list names each by its store key, scrubbed: a value in the key stands as the
    scrubber's marker, which here matches any text."""

    #: The instance's directory.
    root: Path
    #: Its store key, by which its files are named; None when no list names any.
    key: str | None
    #: The keys the lists name, as one pattern; None for none.
    named: re.Pattern[str] | None
    #: Whether every file is refused: a list could not be read, or the instance's key is unknown.
    everything: bool

    def refuses(self, path: Path) -> bool:
        """Whether the viewer may not serve or read the file at *path*, a path under ``root``."""
        if self.everything:
            return True
        if self.named is None:
            return False
        try:
            rel = path.relative_to(self.root).as_posix()
        except ValueError:
            return True
        return self.named.fullmatch(_shown(f"{self.key}/{rel}")) is not None


def _quarantine(inst: Path) -> _Quarantine:
    """What the quarantine lists under the instance directory *inst* name (``QUARANTINE_GLOBS``).
    A list that cannot be read, or an instance whose plan cannot, refuses every file."""
    lists = sorted(p for pattern in QUARANTINE_GLOBS for p in inst.glob(pattern))
    if not lists:
        return _Quarantine(root=inst, key=None, named=None, everything=False)
    plan = _latest_plan(inst)
    patterns: list[str] = []
    for path in lists:
        keys = _quarantine_list(path)
        if keys is None or plan is None:
            return _Quarantine(root=inst, key=None, named=None, everything=True)
        patterns += ["(?:" + ".+".join(re.escape(part) for part in _REDACTED.split(key)) + ")" for key in keys]
    named = re.compile("|".join(patterns), re.DOTALL) if patterns else None
    return _Quarantine(root=inst, key=plan.instance, named=named, everything=False)


# Index rows written with seal-time enrichment carry these keys; when all four
# are present the scanner skips re-reading rollout.json for that row.
_ENRICHMENT_KEYS: frozenset[str] = frozenset(
    {"backend", "model_fingerprint", "accelerator", "agent_implementer_model"}
)


#: Final-verdict line of a reasoned reply ("... explanation ...\nAnswer: B").
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")


def _answer_correct(artifact_text: str, ground_truth: Any) -> bool | None:
    """MMLU-style answer accuracy vs ground truth.

    Accepts short-answer replies that begin with the chosen option ("A. 0" vs
    gt "A") and free-form reasoning replies that end with an "Answer: <option>"
    line, in which case the last such line is the verdict. Returns None when
    there is no ground truth to grade against.
    """
    gt = str(ground_truth).strip() if ground_truth is not None else ""
    if not gt:
        return None

    def _leads_with_gt(text: str) -> bool:
        # gt must match the leading token, not just a leading substring: "A"
        # matches "A. 0" and "Answer: A" but not "Applying ..." -- a bare
        # startswith would count that as answer "A".
        return re.match(rf"{re.escape(gt)}($|\W)", text) is not None

    text = artifact_text.strip()
    if _leads_with_gt(text):
        return True
    answers = _ANSWER_LINE.findall(text)
    return bool(answers) and _leads_with_gt(answers[-1])


def gradings_of(rollout_dir: Path) -> list[dict[str, Any]]:
    """Every grading of the rollout at *rollout_dir*, read from
    ``evals/*/grading.json``.

    Sorted by eval id with ``terminal`` first, so the run's own grading leads and later
    ones follow chronologically (grading ids carry a UTC stamp). A directory with no
    grading.json yields nothing: a rollout whose episode is inlined in its rollout.json."""
    out: list[dict[str, Any]] = []
    for p in (rollout_dir / "evals").glob("*/grading.json"):
        try:
            doc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            logger.debug("bad grading.json at %s", p)
            continue
        if not isinstance(doc, dict):
            logger.debug("grading.json at %s is not an object", p)
            continue
        out.append(doc)
    # ``or {}`` rather than a get default: a document carrying an explicit null
    # episode has the key, so the default never applies and the read would raise --
    # taking down every listing under runs/, not just this rollout's.
    out.sort(
        key=lambda g: (
            (g.get("episode") or {}).get("eval_id") != "terminal",
            (g.get("episode") or {}).get("eval_id") or "",
        )
    )
    return out


def episodes_of(rollout_dir: Path) -> list[dict[str, Any]]:
    """The graded episodes of the rollout at *rollout_dir*, terminal first.

    The episode projection of ``gradings_of``, for readers that want the score, the
    checkpoint or the eval count and not the grading's own provenance. Empty for a
    rollout with no ``grading.json``."""
    return [ep for g in gradings_of(rollout_dir) if (ep := g.get("episode"))]


def _spec_summary(spec: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Extract (model_fingerprint, accelerator_kind) from a serialized TaskSpec dict."""
    if not spec:
        return None, None
    try:
        model_fp = (spec.get("model") or {}).get("fingerprint")
        accel = (spec.get("hardware") or {}).get("accelerator") or {}
        return model_fp, accel.get("kind")
    except Exception:  # noqa: BLE001
        return None, None


_VERIFIER_CLASS_MAP: dict[str, str] = {
    "TokenFaithfulnessVerifier": "token_level",
    # Keyed on the class name persisted in each rollout's on-disk verdict.json.
    # The two entries below are class names recorded verdicts carry, not code
    # references -- dropping them silently demotes those rollouts to "generic"
    # (no token-level turn table, no rollup card).
    "GreedyMatchVerifier": "token_level",
    "GreedyMatch": "token_level",
    "ReferenceRelativeVerifier": "task_relative",
    "MetricVerifier": "task_absolute",
}


def _verifier_entries(verdict: dict[str, Any] | None) -> list[tuple[str, Any]]:
    """The ``"<idx>:<ClassName>"`` entries of a verdict's diagnostics, in order.

    The aggregate verdict also carries run-level keys (``failed_gates``); they are not
    verifiers and their position would misalign the per-verifier matrix rows.
    """
    if verdict is None:
        return []
    diags: dict[str, Any] = verdict.get("diagnostics") or {}
    return [(k, v) for k, v in diags.items()
            if ":" in k and k.split(":", 1)[0].isdigit()]


def _detect_verifier_kind(verdict: dict[str, Any] | None) -> str | None:
    """Detect the verifier kind from the first verifier entry, or None when there is
    none."""
    entries = _verifier_entries(verdict)
    if not entries:
        return None
    class_name = entries[0][0].split(":")[-1]
    return _VERIFIER_CLASS_MAP.get(class_name, "generic")


def _build_verifier_summaries(verdict: dict[str, Any] | None) -> list[VerifierSummary]:
    """Build a VerifierSummary for each verifier in verdict.diagnostics (in order).

    Per-verifier gate_passed is derived from that verifier's own diagnostics,
    not sliced from the top-level aggregate_passed (the AND of all verifiers).
    """
    entries = _verifier_entries(verdict)
    if not entries:
        return []
    pvr: list[list[dict[str, Any]]] = (verdict or {}).get("per_verifier_results") or []
    summaries: list[VerifierSummary] = []
    for idx, (key, diag_val) in enumerate(entries):
        class_name = key.split(":")[-1]
        kind = _VERIFIER_CLASS_MAP.get(class_name, "generic")

        if kind == "token_level":
            # The gate is trace-level, so its verdict is in the matrix rather than in
            # its diagnostics, which publish measurements and ceilings and leave every
            # comparison to the reader.
            summaries.append(TokenLevelSummary(
                name=class_name,
                gate_passed=_matrix_verdict(pvr, idx),
                flip_rate=diag_val.get("flip_rate"),
                rate_ratio=diag_val.get("rate_ratio"),
                curve_fired_levels=_curve_fired_levels(diag_val.get("flip_curve")),
                n_tokens=diag_val.get("n_tokens"),
            ))
        elif kind == "task_relative":
            artifact_value = diag_val.get("artifact_value")
            reference_value = diag_val.get("reference_value")
            tol = diag_val.get("tol") or 0.0
            gate_passed: bool | None = None
            if artifact_value is not None and reference_value is not None:
                gate_passed = (artifact_value + tol) >= reference_value
            summaries.append(TaskRelativeSummary(
                name=class_name,
                gate_passed=gate_passed,
                artifact_value=artifact_value,
                reference_value=reference_value,
                tol=tol,
            ))
        elif kind == "task_absolute":
            artifact_value = diag_val.get("value")
            min_value = diag_val.get("min_value")
            gate_passed = None
            if artifact_value is not None and min_value is not None:
                gate_passed = artifact_value >= min_value
            summaries.append(TaskAbsoluteSummary(
                name=class_name,
                gate_passed=gate_passed,
                artifact_value=artifact_value,
                min_value=min_value,
            ))
        else:
            summaries.append(GenericVerifierSummary(
                name=class_name, gate_passed=_matrix_verdict(pvr, idx)))
    return summaries


def _matrix_verdict(pvr: list[list[dict[str, Any]]], idx: int) -> bool | None:
    """Verifier ``idx``'s own verdict, read off the per-verifier matrix.

    ``passed=None`` means this verifier did not grade the event (a foreign tenant in a
    multi-tenant rollout, see ``verify_trace``'s ``per_verifier_results``), not a failure, so
    those cells are excluded. None when the verifier graded nothing at all."""
    if idx >= len(pvr) or not pvr[idx]:
        return None
    graded = [entry.get("passed") for entry in pvr[idx] if entry.get("passed") is not None]
    return all(graded) if graded else None


def _curve_fired_levels(flip_curve: list[dict[str, Any]] | None) -> list[float] | None:
    """The gap levels (nats) whose flip-count ceiling a trace exceeded.

    Each row carries the enforced ceiling per engine and the trace's count at that
    level; grading is against the most permissive engine, so the operative ceiling is
    the largest of them. None when the record carries no curve rows."""
    if flip_curve is None:
        return None
    fired = []
    for row in flip_curve:
        ceilings = (row.get("ceiling_per_engine") or {}).values()
        if ceilings and row.get("count", 0) > max(ceilings):
            fired.append(row["c"])
    return fired


def _p99_from_latencies(latencies_ms: list[float]) -> float | None:
    """Return p99 latency in ms, using the same index rule as the SLO gate
    (hls.spec.workload.p99_ms_of) so the viewer's p99 matches the scored one.
    p99_ms_of expects seconds, so convert at the boundary."""
    if not latencies_ms:
        return None
    return p99_ms_of([v / 1000.0 for v in latencies_ms])


#: The fields of an event's row that hold a text a reader reads in the expanded row: the prompt,
#: the reply, the expected answer, the failure, and the reference's reply and the answers the
#: verifier parsed.
ROW_TEXTS = ("prompt", "artifact_text", "ground_truth", "error", "reference_text", "predicted", "oracle_predicted")
#: The other fields of an event's row: its ids, times, counts and verdicts. A row holds these, the
#: texts, and ``verifier_cells``, one cell per verifier of a name, whether it passed and a detail.
ROW_VALUES = ("turn_index", "request_id", "session_id", "artifact_tokens", "correct", "token_match", "worst_gap",
              "n_tokens", "first_above_cap_index", "latency_ms", "t_send", "t_recv", "t_planned_send",
              "t_first_packet", "t_last_packet", "tenant_id", "oracle_correct")
#: The most characters of one text a row holds, a verifier cell's detail among them. Of a longer
#: one the row keeps the head and the tail, where a chat prompt's latest turn, a reply's answer and
#: a code prompt's cursor lie.
ROW_TEXT_CHARS = 16_384
#: The most characters of any other value a row holds, a list or an object counted as its JSON, a
#: verifier cell's name and verdict among them: an id, a time, a count or a flag takes far fewer.
ROW_VALUE_CHARS = 256
#: What a row holds in place of the characters it cut from the middle of a longer value, between
#: the head and the tail it keeps, so a reader sees it was cut and how much.
CUT_MARK = "\n[... cut {cut:,} of {total:,} characters ...]\n"
#: The most characters a cut's mark takes: no text is longer than ``sys.maxsize``.
_MARK_CHARS = len(CUT_MARK.format(cut=sys.maxsize, total=sys.maxsize))
#: The most trace indexes the viewer keeps (``_build_trace_index``): it holds at most this many
#: times the most events of a trace it keeps times ``row_chars`` (``_TraceIndex``).
TRACE_INDEXES_CACHED = 16


def _held(value: Any, chars: int) -> Any:
    """*value* as a row holds it under a bound of *chars* characters: None, a flag or a float (whose
    JSON is at most 24 characters, under every bound) as it is, and so a text, a number, a list or an
    object of at most *chars* characters (its JSON's, for all but a text); a longer one as its text
    (or JSON) cut to its first and last halves of *chars* with ``CUT_MARK`` between them."""
    if value is None or isinstance(value, (bool, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value)
    if len(text) <= chars:
        return value
    tail = chars // 2
    return text[:chars - tail] + CUT_MARK.format(cut=len(text) - chars, total=len(text)) + text[len(text) - tail:]


def row_chars(verifiers: int) -> int:
    """The most characters an event's row holds when its verdict names *verifiers* verifiers: each
    text (``ROW_TEXTS``) at ``ROW_TEXT_CHARS`` and each other value (``ROW_VALUES``) at
    ``ROW_VALUE_CHARS``, each with its cut's mark, and per verifier a cell of a text (its detail)
    and two values (its name, and whether it passed); a list or an object counts its JSON."""
    text, value = ROW_TEXT_CHARS + _MARK_CHARS, ROW_VALUE_CHARS + _MARK_CHARS
    return len(ROW_TEXTS) * text + len(ROW_VALUES) * value + verifiers * (text + 2 * value)


@dataclass
class _TraceIndex:
    """What the trace views read of one (eval, file-version) pair: each event's row and the filter
    indices. It holds no event, only rows, so it takes at most its events count times
    ``row_chars`` of its verdict's verifiers: in memory up to 4 bytes a character of a text, and
    more of a list or an object (an object nested in an object, the most, takes about 37 bytes a
    character of its JSON)."""

    #: Each event's row, in the trace's order (``ROW_TEXTS``, ``ROW_VALUES``, ``verifier_cells``).
    rows: list[dict[str, Any]]
    #: The positions in ``rows`` of the rows judged incorrect.
    incorrect: list[int]
    #: The positions of the rows whose verdict says they did not match.
    diverged: list[int]
    #: The positions of the rows a verifier failed.
    failing: list[int]
    #: The positions of each session's rows, by the session id the rows hold.
    sessions: dict[str, list[int]] = field(default_factory=dict)
    #: Whether the trace was there and read whole; when not, the index holds no rows.
    read: bool = False


def _file_key(path: Path) -> tuple[int, int]:
    """Return (st_mtime_ns, st_size) or (0, 0) if the file doesn't exist."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


@functools.lru_cache(maxsize=TRACE_INDEXES_CACHED)
def _build_trace_index(
    trace_path_str: str,
    verdict_path_str: str,
    trace_key: tuple[int, int],
    verdict_key: tuple[int, int],
) -> _TraceIndex:
    """Read the trace (a store trace's chunks, or a published ``trace.json`` in one of its published
    forms) an event at a time, with verdict.json, and build each event's row and the filter
    indices. Each row is built from its whole event, then held with each value cut at its bound
    (``_held``), and no event is kept past its row (``_TraceIndex``).

    Keyed by file identity (path + mtime_ns + size; a store trace's by its ``END.json``, which
    its writer writes last) so the result is safely reused across paginated requests and
    invalidated automatically when either changes on disk. A trace that is not there, or does not
    read whole (``_trace_events``), gives an index of no rows.
    """
    verdict: dict[str, Any] | None = None
    if verdict_key != (0, 0):
        try:
            verdict = json.loads(Path(verdict_path_str).read_text())
        except Exception:  # noqa: BLE001
            verdict = None
    records = _VerdictRecords.of(verdict if isinstance(verdict, dict) else None)

    index = _TraceIndex(rows=[], incorrect=[], diverged=[], failing=[])
    if trace_key == (0, 0):
        return index
    try:
        for i, ev in enumerate(_trace_events(Path(trace_path_str))):
            _index_event(index, i, ev, records)
    except TraceUnreadable as e:
        logger.debug("unreadable trace at %s: %s", trace_path_str, e)
        return _TraceIndex(rows=[], incorrect=[], diverged=[], failing=[])
    index.read = True
    return index


@dataclass(frozen=True)
class _VerdictRecords:
    """What a trace's rows read of its verdict."""

    #: The page's verifier kind (``_detect_verifier_kind``).
    kind: str | None
    #: Whether each event passed, by its event key or its request id.
    per_event: dict[str, bool]
    #: Each verifier's per-event records, in the order of its events.
    pvr: list[list[dict[str, Any]]]
    #: The first verifier's per-event diagnostics, in the order of its events.
    per_row_diag: list[dict[str, Any]]
    #: A verdict's per-event diagnostics, by request id, when it has no per-verifier records.
    fallback_diag: dict[str, dict[str, Any]]
    #: Each verifier's position, class name as a row holds it, and kind.
    verifier_meta: list[tuple[int, str, str]]

    @classmethod
    def of(cls, verdict: dict[str, Any] | None) -> _VerdictRecords:
        """The records of *verdict*; none for no verdict."""
        per_event: dict[str, bool] = {}
        per_row_diag: list[dict[str, Any]] = []
        fallback_diag: dict[str, dict[str, Any]] = {}
        pvr: list[list[dict[str, Any]]] = []
        if verdict is not None:
            per_event = verdict.get("per_event") or {}
            pvr = verdict.get("per_verifier_results") or []
            if pvr and pvr[0]:
                per_row_diag = [entry.get("diagnostics") or {} for entry in pvr[0]]
            else:
                raw_diag: dict[str, Any] = verdict.get("diagnostics") or {}
                if raw_diag:
                    first_val = next(iter(raw_diag.values()), {})
                    if isinstance(first_val, dict):
                        inner_sample = next(iter(first_val.values()), None)
                        if isinstance(inner_sample, dict):
                            fallback_diag = first_val  # type: ignore[assignment]

        verifier_meta: list[tuple[int, str, str]] = []
        if verdict is not None:
            diags_map: dict[str, Any] = verdict.get("diagnostics") or {}
            for vi, vkey in enumerate(diags_map.keys()):
                vclass = vkey.split(":")[-1]
                vkind = _VERIFIER_CLASS_MAP.get(vclass, "generic")
                verifier_meta.append((vi, _held(vclass, ROW_VALUE_CHARS), vkind))
        return cls(kind=_detect_verifier_kind(verdict), per_event=per_event, pvr=pvr, per_row_diag=per_row_diag,
                   fallback_diag=fallback_diag, verifier_meta=verifier_meta)


def _index_event(index: _TraceIndex, i: int, ev: dict[str, Any], records: _VerdictRecords) -> None:
    """Add event *i* of a trace, *ev*, to *index*: its row, built from the whole event and the
    verdict's *records* of it, each value then held at its bound (``_held``), and its place in the
    filters."""
    verifier_kind_page, per_event, pvr = records.kind, records.per_event, records.pvr
    per_row_diag, fallback_diag, verifier_meta = records.per_row_diag, records.fallback_diag, records.verifier_meta
    request_id: str = ev.get("request_id") or ""
    # verdict.per_event may be keyed on event_key ("tenant_id:request_id")
    # or on request_id alone; resolve whichever is present.
    tenant_id_ev = ((ev.get("input") or {}).get("metadata") or {}).get("tenant_id")
    _ek = f"{tenant_id_ev}:{request_id}"
    pe_key = _ek if tenant_id_ev is not None and _ek in per_event else request_id
    gt = ev.get("ground_truth")
    output = (ev.get("response") or {}).get("output") or {}
    contents = output.get("contents") or [{}]
    first_content = contents[0] if contents else {}
    artifact_text: str = first_content.get("text") or ""
    artifact_tokens = first_content.get("tokens")
    if artifact_tokens is not None:
        artifact_tokens = (
            len(artifact_tokens) if isinstance(artifact_tokens, list) else artifact_tokens
        )
    # Last streamed packet's timestamp, matching StreamingLatencySLO._meets's
    # ts[-1]. Differs from t_recv, which is stamped after the SSE stream's
    # trailing `data: [DONE]` and so lands at or after the last packet;
    # consumers computing TPOT (e.g. the timeline view) should use this
    # instead of t_recv to match the scorer.
    _packet_timestamps = (ev.get("response") or {}).get("packet_timestamps")
    t_last_packet = _packet_timestamps[-1] if _packet_timestamps else None

    t_send = ev.get("t_send")
    t_recv = ev.get("t_recv")
    latency_ms: float | None = (
        (t_recv - t_send) * 1000 if t_send is not None and t_recv is not None else None
    )

    if verifier_kind_page in ("task_relative", "task_absolute") and pe_key in per_event:
        correct: bool | None = per_event[pe_key]
    else:
        correct = _answer_correct(artifact_text, gt)

    token_match: bool | None = per_event.get(pe_key) if pe_key in per_event else None

    # In a multi-tenant run, every verifier's array covers every event but
    # only fills in diagnostics for its own tenant's events; find whichever
    # verifier actually scored this row (pvr[0] alone only covers the first
    # tenant).
    diag: dict[str, Any] = {}
    for vpr in pvr:
        if i < len(vpr):
            d = vpr[i].get("diagnostics") or {}
            if d:
                diag = d
                break
    if not diag:
        diag = per_row_diag[i] if i < len(per_row_diag) else (fallback_diag.get(request_id) or {})

    reference_text: str | None = (diag.get("reference") or {}).get("text")
    predicted: str | None = diag.get("predicted")
    oracle_predicted_val: str | None = (diag.get("reference") or {}).get("predicted")
    oracle_correct_val: bool | None = None
    expected_val = diag.get("expected")
    if oracle_predicted_val is not None and expected_val is not None:
        oracle_correct_val = str(oracle_predicted_val) == str(expected_val)

    verifier_cells: list[dict[str, Any]] = []
    for vi, vclass, vkind in verifier_meta:
        cell_passed: bool | None = None
        cell_detail: str = ""
        if vi < len(pvr) and pvr[vi] and i < len(pvr[vi]):
            ventry = pvr[vi][i]
            cell_passed = ventry.get("passed")
            vdiag = ventry.get("diagnostics") or {}
            if vkind == "token_level":
                if cell_passed:
                    cell_detail = "match"
                else:
                    # Only some records carry first_above_cap_index; one with no
                    # per-event offending index gets the detail "diverges".
                    fdi = vdiag.get("first_above_cap_index")
                    cell_detail = f"diverges@{fdi}" if fdi is not None else "diverges"
            elif vkind in ("task_relative", "task_absolute"):
                predicted_cell = vdiag.get("predicted")
                expected_cell = vdiag.get("expected")
                if predicted_cell is not None and expected_cell is not None:
                    if cell_passed:
                        cell_detail = f"{predicted_cell}={expected_cell}"
                    else:
                        cell_detail = f"{predicted_cell}\u2260{expected_cell}"
            # generic: detail stays ""
        verifier_cells.append({"name": vclass, "passed": _held(cell_passed, ROW_VALUE_CHARS),
                               "detail": _held(cell_detail, ROW_TEXT_CHARS)})

    row: dict[str, Any] = {
        "turn_index": ev.get("turn_index"),
        "request_id": request_id,
        "session_id": ev.get("session_id"),
        "prompt": ((ev.get("input") or {}).get("contents") or [{}])[0].get("text"),
        "artifact_text": artifact_text,
        "artifact_tokens": artifact_tokens,
        "ground_truth": gt,
        "correct": correct,
        "token_match": token_match,
        # The row keeps the SPA's field name; the verifier's per-event record calls
        # it max_gap (hls.verify.flips.ReplyFlips).
        "worst_gap": diag.get("max_gap"),
        "n_tokens": diag.get("n_tokens"),
        "first_above_cap_index": diag.get("first_above_cap_index"),
        "latency_ms": latency_ms,
        "t_send": t_send,
        "t_recv": t_recv,
        "t_planned_send": ev.get("t_planned_send"),
        "t_first_packet": (ev.get("response") or {}).get("t_first_packet"),
        "t_last_packet": t_last_packet,
        "tenant_id": ((ev.get("input") or {}).get("metadata") or {}).get("tenant_id"),
        "error": ev.get("error"),
        "reference_text": reference_text,
        "predicted": predicted,
        "oracle_predicted": oracle_predicted_val,
        "oracle_correct": oracle_correct_val,
    }
    row = {name: _held(value, ROW_TEXT_CHARS if name in ROW_TEXTS else ROW_VALUE_CHARS)
           for name, value in row.items()}
    row["verifier_cells"] = verifier_cells
    index.rows.append(row)

    if correct is False:
        index.incorrect.append(i)
    if token_match is False:
        index.diverged.append(i)
    if any(cell.get("passed") is False for cell in verifier_cells):
        index.failing.append(i)
    sid = row["session_id"]
    if sid is not None:
        index.sessions.setdefault(str(sid), []).append(i)


@dataclass(frozen=True)
class _EvalFiles:
    """Where one eval keeps what the trace views read; None for what it lacks."""

    #: ``trace.json`` (published) or the chunked trace's directory (store).
    trace: Path | None
    #: ``verdict.json``.
    verdict: Path | None
    #: ``score.json``.
    score: Path | None


def _published_trace(eval_dir: Path) -> Path | None:
    """The trace of the published eval at *eval_dir*, in the first form present (see
    ``PUBLISHED_TRACES``): its file, or a split trace's first part; None for none."""
    for name in PUBLISHED_TRACES:
        if (eval_dir / name).is_file():
            return eval_dir / name
    if (eval_dir / f"{PART_PREFIX}00").is_file():
        return eval_dir / f"{PART_PREFIX}00"
    elided = sorted(p for p in eval_dir.glob(ELIDED_GLOB) if p.is_file())
    return elided[0] if elided else None


def _trace_parts(first: Path) -> list[Path] | None:
    """Every part of the split trace whose first part is *first*, in their numbers' order; None
    when a number is missing."""
    numbered: dict[int, Path] = {}
    for p in first.parent.glob(f"{PART_PREFIX}*"):
        number = p.name[len(PART_PREFIX):]
        if number.isascii() and number.isdigit() and p.is_file():
            numbered[int(number)] = p
    if sorted(numbered) != list(range(len(numbered))):
        return None
    return [numbered[n] for n in range(len(numbered))]


def _trace_events(trace: Path) -> Iterator[dict[str, Any]]:
    """The events of a trace one at a time, each as ``trace.json`` holds it: from that file or
    a published form of it (gzipped, split, elided), read and parsed incrementally
    (``hls.viewer.published``), or from a store trace's chunks, each event as ``model_dump_json``
    writes it (non-finite numbers as null, as ``trace.json`` holds them). Raise TraceUnreadable where
    the trace turns out not to read whole: a store trace incomplete or altered (every chunk is
    checked against ``END.json`` before any of its events is yielded), a split one missing a part,
    a published one the reader refuses. A reader drops what it took of such a trace, so one bad
    chunk leaves the whole trace unread."""
    if trace.is_dir():
        try:
            for ev in iter_events(trace):
                yield json.loads(ev.model_dump_json())
        except (TraceIncomplete, OSError, ValueError) as e:
            raise TraceUnreadable(f"{trace}: {e}") from None
        return
    if trace.name.startswith(PART_PREFIX):
        parts = _trace_parts(trace)
        if parts is None:
            raise TraceUnreadable(f"{trace}: a part of the split trace is missing")
        yield from published_events(parts, gzipped=True)
        return
    yield from published_events([trace], gzipped=trace.name.endswith(".gz"))


def _first_event_time(events: Path) -> float | None:
    """The time (``ts``) of the first event in an ``events.jsonl``; None when there is none."""
    try:
        with events.open() as f:
            first = next((line for line in f if line.strip()), "")
        ts = json.loads(first).get("ts") if first else None
    except (OSError, ValueError, AttributeError):
        return None
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None


def _served_build(build: Any) -> tuple[Any, list[str] | None]:
    """A store record's *build* as the page is served it, and where it fails to validate as a
    ``RolloutBuild``: the build as written and None when it validates or is None; else None and the
    location of each field that fails, below the record's ``build`` (``build.usage``, or ``build``
    for a build that is not an object), sorted. A location names a field, never a value."""
    if build is None:
        return None, None
    try:
        RolloutBuild.model_validate(build)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        return None, sorted({".".join(["build", *map(str, e["loc"])]) for e in errors})
    return build, None


def _trace_key(trace: Path | None) -> tuple[int, int]:
    """The cache key of a trace: its file's, a split trace's parts' (the latest change and the
    total size), or a store trace's ``END.json``'s; (0, 0) for none."""
    if trace is None:
        return (0, 0)
    if trace.name.startswith(PART_PREFIX):
        keys = [_file_key(p) for p in _trace_parts(trace) or []]
        return (max(k[0] for k in keys), sum(k[1] for k in keys)) if keys else (0, 0)
    return _file_key(trace / END_FILE if trace.is_dir() else trace)


def _latest_plan(inst: Path) -> Plan | None:
    """The newest plan revision in the store instance directory *inst*; None when it holds none
    or the newest does not parse."""
    try:
        revisions = revision_numbers(p.name for p in (inst / "plan").iterdir())
    except OSError:
        return None
    if not revisions:
        return None
    try:
        return Plan.model_validate_json((inst / "plan" / f"rev-{revisions[-1]:08d}.json").read_bytes())
    except (OSError, ValidationError):
        logger.debug("unreadable plan in %s", inst)
        return None


def _lease_record(inst: Path) -> LockRecord | None:
    """The current generation of the lease of the store instance at *inst*; None when it was
    never taken or cannot be read."""
    try:
        return GenerationLock(PosixStore(inst), "lease", ttl_sec=1).current()
    except (StoreError, OSError, ValidationError):
        return None


def _contained(inst: Path, path: Path) -> Path | None:
    """*path* resolved, when it resolves inside the instance directory *inst*; else None: a
    record or an output never leads the viewer out of the instance it was asked about."""
    resolved = path.resolve()
    return resolved if inst.resolve() in resolved.parents else None


def _addressed(rec: dict[str, Any], task: str, rollout_id: str) -> dict[str, Any]:
    """*rec* addressed by its directories: its ``task_id`` and ``rollout_id`` are its task's and its
    own directory's names, from which the page builds every view of it. A record that carries
    other ids, as a published cell named by its arm does, keeps them as ``record_task_id`` and
    ``record_rollout_id``."""
    for name, value in (("task_id", task), ("rollout_id", rollout_id)):
        own = rec.get(name)
        if own is not None and own != value:
            rec[f"record_{name}"] = own
        rec[name] = value
    return rec


class RolloutScanner:
    """Reads rollout records from a directory of ``<task>/<rollout_id>/`` trees, published or in
    the store's layout (see the module docstring)."""

    def __init__(self, runs_dir: Path) -> None:
        #: The directory of task directories: a published runs tree, or a store's ``runs`` area
        #: (or its ``warmers`` area).
        self.runs_dir = Path(runs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_rollouts(self) -> list[dict[str, Any]]:
        """Summary rows for every rollout under runs_dir, newest first: from each task's index
        (``index.jsonl``, or the store's ``index/``), then from each instance directory no index
        row names (a store instance running or ended without a record, a sealed published rollout
        missing from its index, an unsealed published one). A grading's row is left out: its
        rollout's page shows it."""
        rows: list[dict[str, Any]] = []
        if not self.runs_dir.exists():
            return rows
        for task_dir in self.runs_dir.iterdir():
            if not task_dir.is_dir():
                continue
            indexed = self._published_index_rows(task_dir) + self._store_index_rows(task_dir)
            rows += indexed
            indexed_ids = {r.get("rollout_id") or "" for r in indexed}
            for rdir in task_dir.iterdir():
                if rdir.is_dir() and rdir.name not in indexed_ids:
                    row = self._unindexed_row(task_dir.name, rdir)
                    if row is not None:
                        rows.append(row)
        rows.sort(key=lambda r: r.get("created_at") or 0.0, reverse=True)
        return rows

    def get_rollout(self, task: str, rollout_id: str) -> dict[str, Any] | None:
        """The full record of task/rollout_id, with its episodes under ``evals``; None for none.

        A published rollout's ``rollout.json`` stores no episodes: they come from its gradings;
        one without a ``rollout.json`` is shown unsealed (``_unsealed_record``). A store
        instance's record is its ``rollout.json`` (its finalize stage's, or ``failed/``); one
        without, running or ended without a record, gets a record made from its latest plan. Its
        episodes are its gradings', its own first. Either way this read surface composes the
        records the UI reads as one, addressed by its directories (``_addressed``). A store
        record's build is served only when it is a ``RolloutBuild``, and ``build_invalid`` says
        where one that is not fails (``_served_build``); a published record's, in a shape the
        viewer does not read, never."""
        rdir = self._rollout_dir(task, rollout_id)
        if (rdir / "plan").is_dir():
            rec = self._store_rollout(task, rdir)
            if rec is not None:
                rec["build"], rec["build_invalid"] = _served_build(rec.get("build"))
        elif (rec := self._read_json(rdir / "rollout.json")) is None:
            rec = self._unsealed_record(task, rdir)
        else:
            rec["evals"] = episodes_of(rdir)
            rec["build"] = None
        return None if rec is None else _addressed(rec, task, rollout_id)

    def get_config(self, task: str, rollout_id: str) -> dict[str, Any] | None:
        """What the rollout ran with: a published rollout's ``config.json``; a store instance's
        recorded provenance (its ``rollout.json``'s ``config``, else its own grading's
        ``provenance``). None when there is none. Raise ValueError for a ``config.json`` that is
        not valid JSON."""
        rdir = self._rollout_dir(task, rollout_id)
        if (rdir / "plan").is_dir():
            plan = _latest_plan(rdir)
            if plan is None:
                return None
            config = (self._store_record(rdir, plan, ROLLOUT_FILE) or {}).get("config")
            if not isinstance(config, dict):
                config = (self._store_record(rdir, plan, GRADING_FILE) or {}).get("provenance")
            return config if isinstance(config, dict) else None
        path = rdir / "config.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError("config.json is not valid JSON") from exc

    def quarantined(self, task: str, rollout_id: str) -> Callable[[Path], bool]:
        """Whether a file under task/rollout_id's directory is one a finalize scan of it or of a
        grading under it quarantined, as the lists stand now (``_Quarantine.refuses``)."""
        return _quarantine(self._rollout_dir(task, rollout_id)).refuses

    def is_store(self, task: str, rollout_id: str) -> bool:
        """Whether task/rollout_id is a store instance: its directory holds ``plan/``."""
        return (self._rollout_dir(task, rollout_id) / "plan").is_dir()

    def instance_ended(self, task: str, rollout_id: str) -> bool:
        """Whether the store instance task/rollout_id has ended: its plan has an outcome and no
        orchestrator holds its lease. True for one whose plan cannot be read: nothing of it is
        left to follow."""
        rdir = self._rollout_dir(task, rollout_id)
        plan = _latest_plan(rdir)
        return plan is None or (plan.outcome is not None
                                and not lease_held(_lease_record(rdir), now=time.time()))

    def get_trajectory(self, task: str, rollout_id: str) -> dict[str, Any] | None:
        """The build's trajectory: a published rollout's ``build/trajectory.atif.json``, or the
        ``trajectory.json`` harbor writes beside a store build's transcript copy."""
        rdir = self._rollout_dir(task, rollout_id)
        if not (rdir / "plan").is_dir():
            return self._read_json(rdir / "build" / "trajectory.atif.json")
        plan = _latest_plan(rdir)
        index = None if plan is None else self._store_output(rdir, plan, "agent_build",
                                                              f"{TRANSCRIPT_DIR}/{INDEX_FILE}")
        trajectory = None if index is None else _contained(rdir, index.parent / TRAJECTORY_FILE)
        return None if trajectory is None or _quarantine(rdir).refuses(trajectory) else self._read_json(trajectory)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def _published_index_rows(self, task_dir: Path) -> list[dict[str, Any]]:
        """The rollout rows of a published task's ``index.jsonl``. A grading's row is left out:
        its rollout_id is the parent's, so it would show one rollout twice under two scores, and
        it is left out of the ids the directory walk skips too, so a rollout whose only row is a
        grading's still reaches the walk. A row without seal-time enrichment is filled
        from its ``rollout.json``."""
        index = task_dir / "index.jsonl"
        if not index.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in index.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("bad index row in %s", index)
                continue
            if row.get("grading_id"):
                continue
            row["task_id"] = task_dir.name
            rid = row.get("rollout_id") or ""
            if rid and not _ENRICHMENT_KEYS.issubset(row.keys()):
                rec = self._read_json(task_dir / rid / "rollout.json")
                if rec is not None:
                    if row.get("finished_at") is None:
                        row["finished_at"] = rec.get("finished_at")
                    self._enrich_summary(row, rec, episodes_of(task_dir / rid))
            rows.append(row)
        return rows

    def _store_index_rows(self, task_dir: Path) -> list[dict[str, Any]]:
        """The rollout rows of a store task's index: one file per sealed instance, each one JSON
        line with finalize's columns. A grading's file (``<rollout>+<grading>.json``) is left out,
        as in the published index."""
        index = task_dir / INDEX_DIR
        if not index.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(index.glob("*.json")):
            try:
                row = json.loads(path.read_text())
            except (OSError, ValueError):
                logger.debug("bad index row %s", path)
                continue
            if not isinstance(row, dict) or row.get("grading_id"):
                continue
            row["task_id"] = task_dir.name
            row["layout"] = "store"
            rows.append(row)
        return rows

    def _unindexed_row(self, task: str, rdir: Path) -> dict[str, Any] | None:
        """A row for an instance directory no index row names: a store instance from its latest
        plan, a published rollout from its ``rollout.json`` or, unsealed, from its
        ``config.json``; None for anything else. A published row has no cost: the record's build,
        which holds it, is in a shape the viewer does not read. The row is addressed by the directories
        (``_addressed``)."""
        if (rdir / "plan").is_dir():
            plan = _latest_plan(rdir)
            if plan is None:
                return None
            status, phase = self._store_status(rdir, plan)
            return {"task_id": task, "rollout_id": rdir.name, "created_at": plan.created_at,
                    "finished_at": plan.outcome.at if plan.outcome is not None else None, "status": status,
                    "score": None, "gate_passed": None, "image_digests": {}, "phase": phase,
                    "layout": "store"}
        rec = self._read_json(rdir / "rollout.json")
        if rec is None:
            return self._unsealed_row(task, rdir)
        episodes = episodes_of(rdir)
        terminal = episodes[0] if episodes else None
        score = (terminal or {}).get("score") or {}
        row = {
            "task_id": rec.get("task_id", task),
            "rollout_id": rec.get("rollout_id", rdir.name),
            "created_at": rec.get("created_at"),
            "finished_at": rec.get("finished_at"),
            "status": rec.get("status"),
            "score": score.get("value"),
            "gate_passed": score.get("gate_passed"),
            "cost_usd": None,
            "image_digests": rec.get("config", {}).get("image_digests", {}),
        }
        self._enrich_summary(row, rec, episodes)
        return _addressed(row, task, rdir.name)

    def _unsealed_record(self, task: str, rdir: Path) -> dict[str, Any] | None:
        """The record of a published rollout with no ``rollout.json``, which was never sealed
        though a later ``hls grade`` may have graded it: its ``config.json``, its first
        event's time and its gradings' episodes, with the status ``UNSEALED`` and no build record.
        A build trajectory it kept (``build/trajectory.atif.json``) the trajectory endpoint serves.
        Nothing is folded from its events. None without a ``config.json``."""
        config = self._read_json(rdir / "config.json")
        if config is None:
            return None
        return _addressed({"rollout_id": rdir.name, "task_id": config.get("task_id", task),
                           "created_at": _first_event_time(rdir / "events.jsonl"), "finished_at": None,
                           "status": UNSEALED, "config": config, "build": None,
                           "evals": episodes_of(rdir)}, task, rdir.name)

    def _unsealed_row(self, task: str, rdir: Path) -> dict[str, Any] | None:
        """The row of a published rollout with no ``rollout.json`` (``_unsealed_record``), scored
        by its last grading."""
        rec = self._unsealed_record(task, rdir)
        if rec is None:
            return None
        last = (rec["evals"][-1] if rec["evals"] else {}).get("score") or {}
        row = {"task_id": task, "rollout_id": rdir.name, "created_at": rec["created_at"], "finished_at": None,
               "status": UNSEALED, "score": last.get("value"), "gate_passed": last.get("gate_passed"),
               "cost_usd": None, "image_digests": rec["config"].get("image_digests") or {}}
        self._enrich_summary(row, rec, rec["evals"])
        return row

    # ------------------------------------------------------------------
    # Store instances
    # ------------------------------------------------------------------

    def _store_rollout(self, task: str, rdir: Path) -> dict[str, Any] | None:
        """The record of the store instance at *rdir*: its ``rollout.json``, or one made from its
        latest plan when it has none; with its gradings' episodes under ``evals``."""
        plan = _latest_plan(rdir)
        if plan is None:
            return None
        rec = self._store_record(rdir, plan, ROLLOUT_FILE)
        if rec is None:
            status, phase = self._store_status(rdir, plan)
            rec = {"rollout_id": rdir.name, "task_id": task, "created_at": plan.created_at,
                   "finished_at": plan.outcome.at if plan.outcome is not None else None, "status": status,
                   "phase": phase, "reason": plan.outcome.reason if plan.outcome is not None else None,
                   "config": None, "build": None}
        rec["evals"] = [doc["episode"] for doc in self._store_gradings(rdir, plan)
                        if isinstance(doc.get("episode"), dict)]
        rec["layout"] = "store"
        rec["instance"] = plan.instance
        return rec

    def _store_output(self, inst: Path, plan: Plan, kind: str, file: str) -> Path | None:
        """The path of output *file* (its path under its attempt's ``outputs/``) that the first
        stage of *kind* in *plan* published in the store instance at *inst*; None when that
        stage has not published it. Only the key where the publishing attempt keeps it counts
        (``<instance>/stages/<stage>/attempts/<n>/outputs/<file>``), so a manifest naming a key of
        another instance, a grading under this one among them, leads nowhere; the path must
        resolve inside *inst*; and no finalize scan may have quarantined the file."""
        stage = next((s.name for s in plan.stages if s.kind == kind), None)
        if stage is None:
            return None
        try:
            manifest = Manifest.model_validate_json((inst / "stages" / stage / MANIFEST_FILE).read_bytes())
        except (OSError, ValidationError):
            return None
        rel = f"stages/{stage}/attempts/{manifest.attempt}/outputs/{file}"
        if f"{plan.instance}/{rel}" not in {ref.key for ref in manifest.outputs.values()}:
            return None
        path = _contained(inst, inst / rel)
        return None if path is None or _quarantine(inst).refuses(path) else path

    def _store_record(self, inst: Path, plan: Plan, file: str) -> dict[str, Any] | None:
        """Record *file* (``rollout.json`` or ``grading.json``) of the store instance at *inst*:
        its finalize stage's output, else, for an instance that ended failed, the orchestrator's
        copy under ``failed/``; never a file a finalize scan quarantined."""
        path = self._store_output(inst, plan, "finalize", file) or _contained(inst, inst / FAILED_DIR / file)
        return None if path is None or _quarantine(inst).refuses(path) else self._read_json(path)

    def _store_gradings(self, rdir: Path, plan: Plan) -> list[dict[str, Any]]:
        """The gradings of the store instance at *rdir*: its own (a rollout's terminal eval) and
        each of the grading instances under its ``gradings/``; terminal first, then by eval id,
        as ``gradings_of`` orders a published rollout's."""
        found = [doc for doc in [self._store_record(rdir, plan, GRADING_FILE)] if doc is not None]
        graded = rdir / GRADINGS_DIR
        for entry in sorted(graded.iterdir()) if graded.is_dir() else []:
            inst = _contained(rdir, entry)
            inst_plan = None if inst is None else _latest_plan(inst)
            doc = None if inst is None or inst_plan is None else self._store_record(inst, inst_plan, GRADING_FILE)
            if doc is not None:
                found.append(doc)
        found.sort(key=lambda g: ((g.get("episode") or {}).get("eval_id") != TERMINAL_EVAL,
                                  (g.get("episode") or {}).get("eval_id") or ""))
        return found

    def _store_status(self, rdir: Path, plan: Plan) -> tuple[str, str | None]:
        """A store instance's status and phase from its plan: how it ended; else "running" while
        an orchestrator holds its lease and "stalled" while none does, with the first stage not
        yet published as its phase."""
        if plan.outcome is not None:
            return plan.outcome.status, None
        status = "running" if lease_held(_lease_record(rdir), now=time.time()) else STALLED
        waiting = next((s.name for s in plan.stages
                        if not (rdir / "stages" / s.name / MANIFEST_FILE).exists()), None)
        return status, None if waiting is None else f"stage {waiting}"

    def _graded_instance(self, rdir: Path, eval_id: str) -> Path | None:
        """The instance that holds eval *eval_id* of the store instance at *rdir*: the instance
        itself for its own terminal eval, else the grading of that id under its ``gradings/``;
        None for an eval id that names neither."""
        if eval_id == TERMINAL_EVAL:
            return rdir
        try:
            check_segment(eval_id)
        except ValueError:
            return None
        grading = _contained(rdir, rdir / GRADINGS_DIR / eval_id)
        return grading if grading is not None and (grading / "plan").is_dir() else None

    def list_files(self, task: str, rollout_id: str) -> list[dict[str, Any]]:
        """Return a sorted flat listing of all files/dirs under the rollout dir, but those
        ``resolve_file`` refuses: a store's answers area, and a file a finalize scan
        quarantined, or a link to one."""
        base = self._rollout_dir(task, rollout_id)
        if not base.is_dir():
            return []
        quarantine = _quarantine(base)
        out: list[dict[str, Any]] = []
        for p in sorted(base.rglob("*"), key=lambda q: str(q).lower()):
            if "__pycache__" in p.parts or p.name.endswith(".pyc"):
                continue
            if ANSWERS_AREA in p.relative_to(base).parts or quarantine.refuses(p):
                continue
            if p.is_symlink() and quarantine.refuses(p.resolve()):
                continue
            try:
                is_dir = p.is_dir()
                size = None if is_dir else p.stat().st_size
            except OSError:
                # Dangling symlink, or a file removed mid-walk by the still-running
                # orchestrator -- skip this entry rather than 500 the whole listing.
                continue
            rel = p.relative_to(base).as_posix()
            out.append(
                {
                    "path": rel,
                    "name": p.name,
                    "is_dir": is_dir,
                    "size": size,
                }
            )
        return out

    def resolve_file(self, task: str, rollout_id: str, relpath: str) -> tuple[Path, str]:
        """Validate relpath is confined to the rollout dir and resolve it to a safe
        absolute path, without reading its contents -- callers that only need to
        stream the file (e.g. FileResponse) should use this instead of read_file
        to avoid buffering large files in memory.

        Raises ValueError on path traversal or missing file, and ViewRefused for a path
        through a store's answers area or a file a finalize scan quarantined (named by its path
        or by the path a link leads to).
        Returns (absolute_path, media_type), the media type ``text/plain`` for a type a browser
        would run (``ACTIVE_SUFFIXES``): a contestant names the files, and the viewer shows them.
        """
        base = self._rollout_dir(task, rollout_id).resolve()
        full = (base / relpath).resolve()
        if base != full and base not in full.parents:
            raise ValueError("path traversal blocked")
        if ANSWERS_AREA in full.relative_to(base).parts:
            raise ViewRefused(f"the store's answers area ({ANSWERS_AREA}/) is never served")
        if not full.is_file():
            raise ValueError("file not found")
        quarantine = _quarantine(base)
        if quarantine.refuses(base / relpath) or quarantine.refuses(full):
            raise ViewRefused("a finalize scan quarantined this file: it holds a secret's value")
        media = mimetypes.guess_type(full.name)[0] or "application/octet-stream"
        if full.suffix.lower() in ACTIVE_SUFFIXES or _ACTIVE_TYPE.search(media):
            media = "text/plain"
        return full, media

    def read_file(self, task: str, rollout_id: str, relpath: str) -> tuple[bytes, str]:
        """Read a file confined to the rollout dir, fully into memory.

        Raises ValueError on path traversal or missing file.
        Returns (bytes, media_type). Prefer resolve_file() + a streaming response
        for files that may be large.
        """
        full, media = self.resolve_file(task, rollout_id, relpath)
        return full.read_bytes(), media

    def trace_summary(self, task: str, rollout_id: str, eval_id: str) -> TraceSummary | None:
        """Compute aggregate statistics for one eval's trace, from its cached rows
        (``_build_trace_index``), which hold each event's judgement and latency.

        Returns None when the eval has no trace, or its trace does not read whole. Verdict /
        score reads are best-effort: failures yield None fields rather than raising.
        """
        files = self._eval_files(task, rollout_id, eval_id)
        index = self._trace_index(files)
        if not index.read:
            return None
        rows = index.rows
        total = len(rows)

        token_match: int | None = None
        token_match_total: int | None = None
        per_event: dict[str, bool] = {}
        verdict = self._read_json(files.verdict)
        if verdict is not None:
            per_event = verdict.get("per_event") or {}
            token_match = sum(1 for v in per_event.values() if v)
            token_match_total = len(per_event)

        verifier_kind = _detect_verifier_kind(verdict)
        gate_passed: bool | None = verdict.get("aggregate_passed") if verdict is not None else None
        artifact_value: float | None = None
        reference_value: float | None = None
        min_value: float | None = None
        tol: float | None = None

        if verifier_kind in ("task_relative", "task_absolute", "generic"):
            # Suppress bogus oracle-parity chip for non-token-level verifiers
            token_match = None
            token_match_total = None

        if verifier_kind is not None and verdict is not None:
            diags: dict[str, Any] = verdict.get("diagnostics") or {}
            if diags:
                diag_val: dict[str, Any] = next(iter(diags.values()))
                if verifier_kind == "task_relative":
                    artifact_value = diag_val.get("artifact_value")
                    reference_value = diag_val.get("reference_value")
                    tol = diag_val.get("tol")
                elif verifier_kind == "task_absolute":
                    artifact_value = diag_val.get("value")
                    min_value = diag_val.get("min_value")

        # _answer_correct is MMLU-style (expects short answers like "A. 0") and
        # misreports near-zero accuracy for other verifier kinds, whose ground
        # truth is a full string. Use the verdict's per-event verdicts instead
        # when available, falling back to the heuristic only when there's no
        # verdict yet. A row holds both, each judged on its whole event: under these
        # kinds its "correct" is _answer_correct's, and its "token_match" is the
        # verdict's for the event, None for one the verdict does not judge.
        judged = "correct" if verifier_kind in (None, "token_level") else "token_match"
        graded = sum(1 for r in rows if r[judged] is not None)
        correct = sum(1 for r in rows if r[judged])

        # p99_ms -- prefer score.json, fall back to recompute
        p99_ms: float | None = None
        score = self._read_json(files.score)
        if score is not None:
            try:
                p99_ms = (score.get("breakdown") or {}).get("slo", {}).get("p99_ms")
            except Exception:  # noqa: BLE001
                p99_ms = None
        if p99_ms is None and rows:
            p99_ms = _p99_from_latencies([r["latency_ms"] for r in rows if r["latency_ms"] is not None])

        return TraceSummary(
            total=total,
            token_match=token_match,
            token_match_total=token_match_total,
            correct=correct,
            graded=graded if graded > 0 else None,
            p99_ms=p99_ms,
            verifier_kind=verifier_kind,
            gate_passed=gate_passed,
            artifact_value=artifact_value,
            reference_value=reference_value,
            min_value=min_value,
            tol=tol,
            verifiers=_build_verifier_summaries(verdict),
        )

    @staticmethod
    def _timeline_tenant_meta(spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, TimelineSLO]]:
        """Best-effort (name, SLO) per tenant id from a recorded task spec.

        Reads ``spec.tenants[]`` (multi-tenant) or ``spec.workload`` (single-tenant,
        keyed under ``"t0"``). Returns ({tenant_id: name}, {tenant_id: TimelineSLO}).
        Any parse failure yields empty maps so the timeline still renders.
        """
        names: dict[str, str] = {}
        slos: dict[str, TimelineSLO] = {}

        def _slo_of(workload: dict[str, Any]) -> TimelineSLO | None:
            slo = (workload or {}).get("slo") or {}
            if "ttft_ms" in slo:
                return TimelineSLO(kind="streaming", ttft_ms=slo.get("ttft_ms"), tpot_ms=slo.get("tpot_ms"))
            if "target_ms" in slo:
                return TimelineSLO(kind="latency", target_ms=slo.get("target_ms"))
            return None

        try:
            tenants = spec.get("tenants")
            if isinstance(tenants, list) and tenants:
                for t in tenants:
                    tid = t.get("id")
                    if not tid:
                        continue
                    ds = t.get("data_source") or {}
                    names[tid] = str(ds.get("name") or ds.get("path") or tid)
                    s = _slo_of(t.get("workload") or {})
                    if s is not None:
                        slos[tid] = s
            else:
                s = _slo_of(spec.get("workload") or {})
                if s is not None:
                    slos["t0"] = s
        except (AttributeError, TypeError, ValueError):
            return {}, {}
        return names, slos

    def trace_page(
        self,
        task: str,
        rollout_id: str,
        eval_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
        only: str | None = None,
    ) -> TracePage:
        """Return a windowed page of per-turn trace rows.

        ``only`` filters before slicing:
        - ``None`` or ``""`` -> no filter (all rows)
        - ``"incorrect"`` -> rows where correct is False
        - ``"diverged"`` -> rows where token_match is False
        - ``"failing"`` -> rows where at least one verifier_cell failed
        - any other non-empty value -> session_id filter

        Paging and filtering are O(limit) -- row dicts and filter indices are
        built once and cached in ``_build_trace_index`` keyed by file identity
        (path + mtime_ns + size), so repeated calls with different offsets /
        ``only`` values do not re-parse the files.
        """
        # Normalise: empty/whitespace only string -> no filter
        only = (only or "").strip() or None

        index = self._trace_index(self._eval_files(task, rollout_id, eval_id))

        all_count = len(index.rows)
        if only is None:
            selected: list[int] | range = range(all_count)
        elif only == "incorrect":
            selected = index.incorrect
        elif only == "diverged":
            selected = index.diverged
        elif only == "failing":
            selected = index.failing
        else:
            # session_id filter
            selected = index.sessions.get(only, [])

        total = len(selected)
        page_indices = list(selected)[offset : offset + limit]

        # Build lightweight copies for the dense list; the prompt and the
        # artifact/reference text the row holds are only shown in the expanded row
        # and lazy-loaded via trace_row. Must not mutate index.rows, since the
        # cache needs the rows it holds.
        def _light(row: dict[str, Any]) -> dict[str, Any]:
            r = dict(row)
            r.pop("prompt", None)
            for f in ("artifact_text", "reference_text"):
                v = r.get(f)
                if isinstance(v, str) and len(v) > 160:
                    r[f] = v[:160] + "..."
            return r

        page_rows = [_light(index.rows[i]) for i in page_indices]

        return TracePage(offset=offset, limit=limit, total=total, rows=page_rows)

    def trace_row(
        self,
        task: str,
        rollout_id: str,
        eval_id: str,
        request_id: str,
    ) -> dict[str, Any] | None:
        """Return the cached row for request_id, as the index holds it: its prompt, and
        its texts as long as ``ROW_TEXT_CHARS``, a longer one cut and marked (``_held``).

        Builds (or reuses) the same cached _TraceIndex as trace_page, then linearly
        scans for the first row whose request_id, as the row holds it, matches. Returns
        None if not found.
        """
        index = self._trace_index(self._eval_files(task, rollout_id, eval_id))

        for row in index.rows:
            if row.get("request_id") == request_id:
                return row
        return None

    def timeline(
        self, task: str, rollout_id: str, eval_id: str
    ) -> TimelinePayload | None:
        """Compact per-request timeline for one eval; None if it has no trace.

        Reuses the cached trace index, anchors all times to t0 = min(t_send),
        discovers the tenant list from event tenant_id (single-tenant -> "t0"),
        and enriches name + SLO best-effort from the spec the eval recorded.
        """
        files = self._eval_files(task, rollout_id, eval_id)
        if _trace_key(files.trace) == (0, 0):
            return None
        index = self._trace_index(files)
        rows = index.rows
        sends = [r["t_send"] for r in rows if r.get("t_send") is not None]
        if not sends:
            return TimelinePayload(t0=0.0, duration=0.0, tenants=[], requests=[])
        t0 = min(sends)

        order: list[str] = []
        counts: dict[str, int] = {}
        for r in rows:
            tid = r.get("tenant_id") or "t0"
            if tid not in counts:
                counts[tid] = 0
                order.append(tid)
            counts[tid] += 1

        names, slos = self._timeline_tenant_meta(self._spec_for(task, rollout_id, eval_id))
        ti_of = {tid: i for i, tid in enumerate(order)}
        tenants = [
            TimelineTenant(id=tid, name=names.get(tid), req_n=counts[tid], slo=slos.get(tid))
            for tid in order
        ]

        requests: list[dict[str, Any]] = []
        tmax = t0
        for r in rows:
            ts = r.get("t_send")
            if ts is None:
                continue
            tr = r.get("t_recv")
            tp = r.get("t_planned_send")
            tf = r.get("t_first_packet")
            tl = r.get("t_last_packet")
            tid = r.get("tenant_id") or "t0"
            requests.append(
                {
                    "ti": ti_of[tid],
                    "sid": r.get("session_id"),
                    "turn": r.get("turn_index") or 0,
                    "send": round(ts - t0, 4),
                    "recv": round(tr - t0, 4) if tr is not None else None,
                    "plan": round(tp - t0, 4) if tp is not None else None,
                    "ttft": round(tf - t0, 4) if tf is not None else None,
                    # Last streamed packet's timestamp, the same instant
                    # StreamingLatencySLO._meets uses for the end of its TPOT
                    # window. Differs from "recv": t_recv is stamped after the
                    # trailing `data: [DONE]`, so a TPOT built from recv
                    # over-counts and can flag a request the scorer credited.
                    # None when no per-packet timing was captured (e.g.
                    # return_token_ids=False); the frontend falls back to recv.
                    "tlast": round(tl - t0, 4) if tl is not None else None,
                    "ntok": r.get("artifact_tokens"),
                    "err": bool(r.get("error")),
                }
            )
            for t in (ts, tr, tp, tf, tl):
                if t is not None and t > tmax:
                    tmax = t
        return TimelinePayload(
            t0=t0, duration=round(tmax - t0, 4), tenants=tenants, requests=requests
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout_dir(self, task: str, rollout_id: str) -> Path:
        """Resolve task/rollout_id to a directory confined to runs_dir.

        task and rollout_id come straight from URL path segments and cannot be
        trusted as-is: a segment of ".." (or one containing a separator) would
        make the naive join land outside runs_dir, which every path built from
        the result -- including read_file's own relpath check -- would then
        treat as a legitimate root. Reject those segments, then re-verify the
        resolved path is still inside runs_dir against symlink tricks. Raises
        ValueError on rejection, same as read_file's own guard, and ViewRefused for a
        directory in a store's answers area.
        """
        for segment in (task, rollout_id):
            if not segment or segment in (".", "..") or "/" in segment or "\\" in segment:
                raise ValueError("path traversal blocked")
        base = (self.runs_dir / task / rollout_id).resolve()
        root = self.runs_dir.resolve()
        if base != root and root not in base.parents:
            raise ValueError("path traversal blocked")
        if ANSWERS_AREA in base.relative_to(root).parts:
            raise ViewRefused(f"the store's answers area ({ANSWERS_AREA}/) is never served")
        return base

    def rollout_dir(self, task: str, rollout_id: str) -> Path:
        """Public, validated accessor for a rollout's directory.

        Raises ValueError if task/rollout_id would escape runs_dir. Used by
        server.py endpoints that need the directory directly rather than through
        one of the read/list helpers below.
        """
        return self._rollout_dir(task, rollout_id)

    def _eval_files(self, task: str, rollout_id: str, eval_id: str) -> _EvalFiles:
        """Where eval *eval_id* of task/rollout_id keeps its trace, verdict and score: a
        published rollout's ``evals/<eval_id>/``, its trace in the first form present
        (``_published_trace``); for a store instance, the outputs that the drive, verify and score
        stages of the instance holding the eval (``_graded_instance``) published, and no trace
        when a finalize scan quarantined any of its files. What it lacks is None."""
        rdir = self._rollout_dir(task, rollout_id)
        if not (rdir / "plan").is_dir():
            d = rdir / "evals" / eval_id
            return _EvalFiles(trace=_published_trace(d), verdict=d / "verdict.json", score=d / "score.json")
        inst = self._graded_instance(rdir, eval_id)
        plan = None if inst is None else _latest_plan(inst)
        if inst is None or plan is None:
            return _EvalFiles(trace=None, verdict=None, score=None)
        end = self._store_output(inst, plan, "deploy_drive", TRACE_END)
        quarantine = _quarantine(inst)
        if end is not None and any(quarantine.refuses(p) for p in end.parent.iterdir()):
            end = None
        return _EvalFiles(trace=None if end is None else end.parent,
                          verdict=self._store_output(inst, plan, "verify", VERDICT_FILE),
                          score=self._store_output(inst, plan, "score", SCORE_FILE))

    @staticmethod
    def _trace_index(files: _EvalFiles) -> _TraceIndex:
        """The cached rows and filters of an eval's trace (``_build_trace_index``)."""
        return _build_trace_index(
            str(files.trace) if files.trace is not None else "",
            str(files.verdict) if files.verdict is not None else "",
            _trace_key(files.trace),
            _file_key(files.verdict) if files.verdict is not None else (0, 0),
        )

    def _spec_for(self, task: str, rollout_id: str, eval_id: str) -> dict[str, Any]:
        """The task spec eval *eval_id* recorded: a published rollout's ``config.json``'s, a
        store eval's grading provenance's; {} when there is none."""
        rdir = self._rollout_dir(task, rollout_id)
        if (rdir / "plan").is_dir():
            inst = self._graded_instance(rdir, eval_id)
            plan = None if inst is None else _latest_plan(inst)
            grading = None if inst is None or plan is None else self._store_record(inst, plan, GRADING_FILE)
            spec = ((grading or {}).get("provenance") or {}).get("spec")
        else:
            spec = (self._read_json(rdir / "config.json") or {}).get("spec")
        return spec if isinstance(spec, dict) else {}

    def _read_json(self, path: Path | None) -> dict[str, Any] | None:
        """The JSON object in the file at *path*; None when there is none, or it is not one."""
        if path is None or not path.is_file():
            return None
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            logger.debug("failed to parse %s", path)
            return None
        return doc if isinstance(doc, dict) else None

    @staticmethod
    def _enrich_summary(
        row: dict[str, Any], rec: dict[str, Any], episodes: list[dict[str, Any]]
    ) -> None:
        """Fill a published row's enrichment columns from its record dict, in place, best-effort:
        the model fingerprint and the accelerator from its spec, the backend from the terminal
        episode, which *episodes* leads with. The implementer is not among them: the record names
        it in a producer the viewer does not read."""
        cfg = rec.get("config") or {}
        model_fp, accel = _spec_summary(cfg.get("spec"))
        row.setdefault("model_fingerprint", model_fp)
        row.setdefault("accelerator", accel)
        terminal = episodes[0] if episodes else None
        checkpoint = (terminal or {}).get("checkpoint") or ""
        if ":" in checkpoint:
            prefix = checkpoint.split(":", 1)[0].strip()
            if prefix in CHECKPOINT_BACKENDS:
                row.setdefault("backend", prefix)

