"""Reference answers on the store, shared by every verify role of a task.

A reference's answer to a corpus row depends on the row, the reference's own config, the
silicon it runs on and the image it runs in, and never on the contestant, so it is computed
once and served to every later verify of every arm. The verify stage mounts the task's area of
the store, ``oracle/<task>/``, into its role, and the role keeps the answers there in one
directory per answer key: the sha256 of the reference's fingerprint, the GPU kind and the
reference image's digest (``answer_key``). An answer computed on other silicon or in another
image misses.

Writers add files and never change one. After each call that answered a miss, the role writes
that call's answers as ``<key>/<writer>.<n>.json`` (an ``AnswerFile``), create-only, so the
verify roles of concurrent gradings never drop each other's answers, and a role that dies keeps
every call it finished. Reading merges every file of the key in the order the files were written,
and the first answer to a question, a request id with its prompt's digest, is served from then
on, so every later verify serves the same answer to it. A request id whose prompt changed (a task
edit the key does not cover, such as the output cap) is a new question: answered once, it is
served too, whatever an earlier file answered the old prompt.

These are the reference's answers to the withheld test split. The area lies outside every
instance, so no instance stores or publishes it, and only harness containers mount it: the
verify role, and the oracle fill (``hls.verify.fill``), which fills it ahead of the gradings. No
contestant container may mount anything in the store (``StageContext.start_container``). No
contestant's bytes reach it either. The verify and oracle fill stages refuse at plan time a
caching reference on a task that declares an agent or a workload of more than one turn per
session (``multi_turn_caching``), whose prompts carry the server's replies. Whatever a task
declares, a prompt stamped past its session's first turn is answered afresh, neither served
from the area nor kept in it (``later_turn``).

The area is an optimisation and never a gate. A key that
cannot be listed reads as empty; an entry whose name is no store segment, that is not a regular
file (a link, a directory, a FIFO), that holds more than ``ANSWER_FILE_MAX_BYTES``, that does not
parse or that names another key is skipped; each with a warning naming it. A write that fails is
logged while the run keeps its answers in memory. Only the oracle fill's writer is strict
(``StoreAnswers.strict``): it exists to bank its answers, so a write of its that fails raises
``AnswersNotWritten``, which ends the fill as infra that is retried.

Beside the answers files, a key's directory holds the question sets of the fills that completed
there (``QuestionSet``, ``questions_name``): every question one fill asked the reference about
the test rows of one rows entry, its prompts shaped as one task declaration shapes them,
written once the fill had banked an answer to every one. A plan-time check reads it in place of
the rows (``hls.pipeline.preflight``), so its presence is the evidence that the fill completed,
and the answers files are the evidence that the answers are still there. Reading the answers
passes over it."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from hls.core.capabilities import Capability, Generate
from hls.core.payloads import Envelope
from hls.provision.model import ROWS_KEY_PATTERN, check_rows_key
from hls.reference.engines import tools_of
from hls.store.base import AlreadyExists, NotFound, Store, StoreError, check_segment
from hls.store.digest import Sha256Hex, check_sha256_hex, sha256_bytes
from hls.util import validation_errors

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec

log = logging.getLogger(__name__)

#: The store area that holds every task's answers, one directory per task.
ANSWERS_AREA = "oracle"
#: The most bytes an answers file may hold to be read; a larger one is skipped unread. One
#: call answers at most a whole test split, and the largest file one writes is about 16 MB:
#: the longest reference answer is 550 B on
#: multimodal_asr and 389 B on multimodal_omni, an entry of at most 3,430 B (ASR's, with its 440
#: token ids at full width), and the largest split is omni's 4,600 rows. This is eight times that.
ANSWER_FILE_MAX_BYTES = 128 * 2**20

#: The capabilities whose answer a corpus row alone determines, and the only ones the area
#: holds: the verify role wraps no other, ``check_wiring`` refuses a caching reference that a
#: verifier requires for any other, and ``hls oracle fill`` (``hls.verify.fill``) fills in these
#: ahead of the gradings. Each takes a list of prompt envelopes and returns one answer envelope
#: per prompt, the one shape ``StoreAnswers.wrap`` stores.
#:
#: Every other capability is either not a question about a row (``StackReport``
#: describes the container) or a question about a contestant's reply
#: (``TeacherForce`` and ``ReferenceLogprobs`` price served tokens or codes,
#: ``DecodeCodes`` decodes served codes). Their inputs do not exist until an arm has
#: run and no two arms ask the same one, so a stored answer could never be served.
CORPUS_CAPABILITIES: tuple[type[Capability], ...] = (Generate,)

#: Prompt metadata a reference reads besides its tools. The rest is harness bookkeeping --
#: tenant, session, turn index, send time -- which varies between runs and between tenants
#: serving the same row, so digesting it would stop an answer ever being served.
_PROMPT_METADATA = ("messages", "max_tokens")


def prompt_digest(prompt: Envelope) -> str:
    """Fingerprint of everything a reference reads from *prompt*: its contents, the
    ``messages`` and ``max_tokens`` the driver stamps on, and the tools its turn offers, which
    the engines render into the prompt. The tools are read as the engines read them
    (``tools_of``), so an absent, empty or non-list ``tools`` is no tools alike.

    Stored with each answer, and an answer is served only to a prompt with the same request id
    and digest: request ids are the corpus row's identity and stable across runs, but unique
    only within a tenant and a turn, and a task edit the answer key does not cover can change
    the prompt under one."""
    metadata = {k: prompt.metadata.get(k) for k in _PROMPT_METADATA}
    metadata["tools"] = tools_of(prompt)
    payload = {"contents": [c.model_dump(mode="json") for c in prompt.contents], "metadata": metadata}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def later_turn(prompt: Envelope) -> bool:
    """Whether *prompt* is stamped with a turn index other than 0, past its session's first
    turn, where its messages carry the server's earlier replies."""
    return prompt.metadata.get("turn_index", 0) != 0


def answers_area(task: str) -> str:
    """The store key of task *task*'s answers."""
    return f"{ANSWERS_AREA}/{check_segment(task)}"


def multi_turn_caching(task: TaskSpec) -> list[str]:
    """Why *task* may not keep reference answers in the store: one reason for each verifier
    with a caching reference (``Reference.cache``) on a tenant whose prompts carry the server's
    earlier replies. Those are every tenant of a task that declares an agent (``TaskSpec.agent``),
    whose driver runs one trajectory per row with as many calls as the agent takes, whatever the
    workload declares, and a tenant whose workload runs more than one turn per session
    (``Workload.mean_turns_per_session``), from turn two. The answers would depend on a
    contestant's text."""
    reasons = []
    for tenant in task.effective_tenants():
        turns = tenant.workload.mean_turns_per_session
        if task.agent is not None:
            why = (f"the task declares an agent, so tenant {tenant.id}'s trajectories send calls that carry the "
                   "server's earlier replies whatever its workload declares")
        elif turns > 1:
            why = (f"tenant {tenant.id}'s workload runs {turns:g} turns per session, so its prompts carry the "
                   "server's replies from turn two")
        else:
            continue
        for verifier in tenant.verifiers:
            for cfg in {id(c): c for c in verifier.engines()}.values():
                if getattr(cfg, "cache", False):
                    reasons.append(f"{why}, and {type(verifier).__name__} caches the answers of its reference "
                                   f"{type(cfg).__name__} (cache=True)")
    return reasons


def answer_key(fingerprint: str, gpu_kind: str, image: str) -> str:
    """The directory, inside a task's area, of the answers a reference with *fingerprint*
    gives on *gpu_kind* GPUs in the image whose file has digest *image*."""
    body = {"fingerprint": fingerprint, "gpu_kind": gpu_kind, "image": image}
    return sha256_bytes(json.dumps(body, sort_keys=True).encode())


def _envelope(answer: dict[str, Any]) -> dict[str, Any]:
    """*answer*, once it validates as the envelope ``StoreAnswers.wrap`` serves it as."""
    Envelope.model_validate(answer)
    return answer


class AnswerEntry(BaseModel):
    """One answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The asking prompt's request id.
    request_id: str
    #: The asking prompt's digest (``prompt_digest``), which a later prompt must match to be
    #: served this answer.
    digest: str
    #: The answer envelope, as ``Envelope.model_dump(mode="json")``. Checked to be one, so a
    #: file holding anything else is skipped when read rather than failing the verify it serves.
    answer: Annotated[dict[str, Any], AfterValidator(_envelope)]


class AnswerFile(BaseModel):
    """The answers one call of one writer added."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The reference's fingerprint.
    fingerprint: str
    #: The kind of GPU the reference ran on, or ``cpu`` for a role without one.
    gpu_kind: str
    #: The sha256 of the reference image's file.
    image: str
    #: The writer, unique to one verify attempt.
    writer: str
    #: When the writer wrote the file (POSIX time); the earlier file's answer wins.
    written_at: float
    #: The call's answers, in its order.
    answers: list[AnswerEntry]


class AnswersNotWritten(RuntimeError):
    """A strict writer's answers could not be written to the area (``StoreAnswers.strict``)."""


class QuestionSetConflict(ValueError):
    """A question set already in the area under a fill's name holds other questions than the
    fill asked (``StoreAnswers.record_questions``)."""

    def __init__(self, path: str) -> None:
        #: Its key in the area.
        self.path = path
        super().__init__(self.describe(path))

    @staticmethod
    def describe(where: str) -> str:
        """The conflict, naming the set as *where*, and what clears it."""
        return (f"{where} holds another question set than this fill asked: the prompts of these rows digest otherwise "
                "now; remove that file and fill again")


#: How many hex digits of a question set's shape its name holds.
_SHAPE_DIGITS = 16
# The name of a question set, as ``questions_name`` makes it.
_QUESTIONS = re.compile(rf"questions-{ROWS_KEY_PATTERN}-[0-9a-f]{{{_SHAPE_DIGITS}}}\.json")


def questions_name(rows: str, shape: str) -> str:
    """The name, in a key's directory, of the question set of the test rows of the rows entry
    *rows* whose prompts a task declaration shapes as the digest *shape* says
    (``hls.verify.fill.question_shape``). Raise ValueError for a *rows* that is no rows key or
    a *shape* that is no sha256."""
    return f"questions-{check_rows_key(rows)}-{check_sha256_hex(shape)[:_SHAPE_DIGITS]}.json"


class Question(BaseModel):
    """One question a reference is asked: a prompt of one tenant, as ``StoreAnswers`` keys it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The tenant whose row it is.
    tenant: str
    #: The prompt's request id.
    request_id: str
    #: The prompt's digest (``prompt_digest``).
    digest: Sha256Hex


class QuestionSet(BaseModel):
    """``questions-<rows>-<shape>.json`` in a key's directory: every question a completed oracle
    fill asked the key's reference about the test rows of one rows entry, its prompts shaped as
    one task declaration shapes them. It holds no time and no writer, so every completed fill of
    the same rows and shape writes the same bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The reference's fingerprint.
    fingerprint: str
    #: The kind of GPU its answers were computed on.
    gpu_kind: str
    #: The sha256 of the reference image's file.
    image: str
    #: The key of the rows entry whose test rows were asked.
    rows: str
    #: The digest of what shaped their prompts (``hls.verify.fill.question_shape``).
    shape: Sha256Hex
    #: The questions, sorted by tenant, request id and digest.
    questions: list[Question]


class StoreAnswers:
    """The answers of one task's area of the store, for one GPU kind and one reference image,
    written by one writer, and the wrapper that serves them (``wrap``). An answer is held under
    its question: a request id with its prompt's digest (``prompt_digest``)."""

    def __init__(self, store: Store, *, gpu_kind: str, image: str, writer: str,
                 clock: Callable[[], float] = time.time, strict: bool = False) -> None:
        #: The task's area of the store, as a store of its own.
        self.store = store
        #: The kind of GPU the role's reference engines run on.
        self.gpu_kind = gpu_kind
        #: The sha256 of the reference image's file.
        self.image = image
        #: This writer's name, the prefix of every file it adds.
        self.writer = check_segment(writer)
        #: Time source for ``AnswerFile.written_at``.
        self.clock = clock
        #: Whether a write that fails raises ``AnswersNotWritten`` rather than being logged: the
        #: oracle fill's writer, which exists to bank its answers. A verify's is not, since the
        #: area is never a gate there.
        self.strict = strict
        #: Every file merged into the answers, by key in the area, to its sha256.
        self.read: dict[str, str] = {}
        #: Every file merged into the answers, by key in the area, to the writer it names.
        self.read_writers: dict[str, str] = {}
        #: Every file this writer added, by key in the area, to its sha256.
        self.written: dict[str, str] = {}
        # Fingerprint to question (request id, prompt digest) to the answer envelope, dumped.
        self._answers: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        # The fingerprints whose files were merged, each once.
        self._loaded: set[str] = set()
        # How many files this writer has tried to add; the next one is numbered one more.
        self._files = 0

    def _key(self, fingerprint: str) -> str:
        return answer_key(fingerprint, self.gpu_kind, self.image)

    def _names(self, key: str) -> list[str]:
        """The entries of *key*'s directory; none, with a warning, when it cannot be listed."""
        try:
            return self.store.list_children(key)
        except (StoreError, OSError) as exc:
            log.warning("skipping reference answers %s, whose files cannot be listed (%s); the oracle answers "
                        "afresh", key, exc)
            return []

    def _read(self, path: str) -> bytes:
        """The bytes of answers file *path*. Raise StoreError, before reading anything, for an
        entry that is not a regular file (``Store.size`` reads only its metadata: a link, a
        directory, a FIFO), and for one that holds more than ``ANSWER_FILE_MAX_BYTES``."""
        size = self.store.size(path)
        if size <= ANSWER_FILE_MAX_BYTES:
            # One byte past the bound is read, so a file that grew since is refused too.
            with self.store.open_read(path) as f:
                data = f.read(ANSWER_FILE_MAX_BYTES + 1)
            size = len(data)
        if size > ANSWER_FILE_MAX_BYTES:
            raise StoreError(f"it holds {size} bytes, more than the {ANSWER_FILE_MAX_BYTES} an answers file may")
        return data

    def _load(self, fingerprint: str) -> None:
        """Merge every readable file of *fingerprint*'s key into the answers, earliest first,
        each question keeping the first answer read."""
        key = self._key(fingerprint)
        found = []
        for name in self._names(key):
            try:
                check_segment(name)
            except ValueError:
                log.warning("skipping %r in reference answers %s: its name is no store segment", name, key)
                continue
            if _QUESTIONS.fullmatch(name):
                # A fill's question set, which holds no answer.
                continue
            path = f"{key}/{name}"
            try:
                data = self._read(path)
            except (StoreError, OSError, ValueError) as exc:
                log.warning("skipping reference answers %s (%s)", path, exc)
                continue
            try:
                file = AnswerFile.model_validate_json(data)
            except Exception as exc:
                # A payload's own validator may raise any type, which pydantic passes through
                # unwrapped (a codec position's KeyError, TypeError or OverflowError). The bytes
                # are bounded, so the parse is too.
                # A validation's errors are named by type and place: its text would hold the
                # file's values, and the warning reaches a plan's refusal.
                detail = validation_errors(exc) if isinstance(exc, ValidationError) else exc
                log.warning("skipping reference answers %s, which does not parse (%s: %s)", path,
                            type(exc).__name__, detail)
                continue
            if (file.fingerprint, file.gpu_kind, file.image) != (fingerprint, self.gpu_kind, self.image):
                log.warning("skipping reference answers %s, which belong to another reference, GPU kind or "
                            "image", path)
                continue
            found.append((file.written_at, name, file, sha256_bytes(data)))
        held = self._answers.setdefault(fingerprint, {})
        for _, name, file, digest in sorted(found, key=lambda f: (f[0], f[1])):
            for entry in file.answers:
                held.setdefault((entry.request_id, entry.digest), entry.answer)
            self.read[f"{key}/{name}"] = digest
            self.read_writers[f"{key}/{name}"] = file.writer
        log.info("reference answers %s: %d files, %d answers", key, len(found), len(held))

    def _held(self, fingerprint: str) -> dict[tuple[str, str], dict[str, Any]]:
        """The answers held for *fingerprint*, by question, merging its key's files the first
        time it is asked."""
        if fingerprint not in self._loaded:
            self._load(fingerprint)
            self._loaded.add(fingerprint)
        return self._answers.setdefault(fingerprint, {})

    def answered(self, fingerprint: str, request_id: str, digest: str) -> bool:
        """Whether the area holds an answer of the reference with *fingerprint* to the question
        of a prompt with *request_id* whose digest (``prompt_digest``) is *digest*: what a call
        ``wrap`` wraps serves it, for a prompt of its session's first turn."""
        return (request_id, digest) in self._held(fingerprint)

    def unanswered(self, fingerprint: str, prompt: Envelope) -> bool:
        """Whether a call ``wrap`` wraps for *fingerprint* would hand *prompt* to the engine: a
        prompt past its session's first turn (``later_turn``), which is never served, or one
        whose question the area does not hold (``answered``). The area only grows (writers
        create files and never change or remove one), so a prompt this finds answered stays
        answered for every later reader of the same key."""
        return later_turn(prompt) or not self.answered(fingerprint, prompt.request_id, prompt_digest(prompt))

    def record_questions(self, fingerprint: str, *, rows: str, shape: str,
                         questions: Iterable[Question]) -> tuple[str, str]:
        """Write the question set of the reference with *fingerprint* (``QuestionSet``): every
        one of *questions*, asked about the test rows of the rows entry *rows* with prompts shaped
        as *shape* says, sorted. Return its key in the area and its sha256. A fill calls it once
        it has banked an answer to every question, and only then.

        Create-only: a set already there under the name (``questions_name``) is left as it is
        when it holds the same bytes, which every completed fill of the same rows and shape
        writes. Raise QuestionSetConflict when it holds others: a stale set would stand for
        questions no fill answered, and another fill would find it again. Raise
        AnswersNotWritten when the set cannot be written or the one there read."""
        path = f"{self._key(fingerprint)}/{questions_name(rows, shape)}"
        record = QuestionSet(fingerprint=fingerprint, gpu_kind=self.gpu_kind, image=self.image, rows=rows,
                             shape=shape, questions=sorted(set(questions),
                                                           key=lambda q: (q.tenant, q.request_id, q.digest)))
        data = record.model_dump_json().encode()
        try:
            self.store.create(path, data)
        except AlreadyExists:
            try:
                there = self._read(path)
            except (StoreError, OSError) as exc:
                raise AnswersNotWritten(f"could not read question set {path}: {exc}") from exc
            if there != data:
                raise QuestionSetConflict(path) from None
        except (StoreError, OSError) as exc:
            raise AnswersNotWritten(f"could not write question set {path}: {exc}") from exc
        return path, sha256_bytes(data)

    def question_set(self, fingerprint: str, *, rows: str, shape: str) -> QuestionSet | None:
        """The question set a completed fill recorded for the reference with *fingerprint*, the
        test rows of the rows entry *rows* and prompts shaped as *shape* says; None when none is
        there. It is read as an answers file is, so a link, a directory or a file over
        ``ANSWER_FILE_MAX_BYTES`` raises StoreError; one that does not parse raises
        ValidationError, and one that names another reference, GPU kind, image, rows entry or
        shape ValueError."""
        path = f"{self._key(fingerprint)}/{questions_name(rows, shape)}"
        try:
            data = self._read(path)
        except NotFound:
            return None
        record = QuestionSet.model_validate_json(data)
        want = {"fingerprint": fingerprint, "gpu_kind": self.gpu_kind, "image": self.image, "rows": rows,
                "shape": shape}
        other = [field for field, value in want.items() if getattr(record, field) != value]
        if other:
            raise ValueError(f"{path} is the question set of another {', '.join(other)}")
        return record

    def wrap(self, fingerprint: str, call: Callable) -> Callable:
        """Wrap a corpus capability's call (``CORPUS_CAPABILITIES``) of the reference with
        *fingerprint* so the area's answers are served, merging the area's files of its key the
        first time it is wrapped. Each answer the call computes for a prompt the area may hold is
        kept: held for the rest of the run and added to the area (``bank``).

        Answers come back stamped with the asking prompt's request id, so a caller that keys on
        the returned ids is not handed the id of whichever grading first asked.

        ``call`` must take a list of prompt envelopes and return one answer envelope per prompt:
        that is what gets digested, stored and served. It is also why an answer depends on the
        row and the reference's config and never on the contestant, for what is wrapped. A
        teacher-force call takes a contestant's reply and returns gap vectors, so its answer has
        no envelope to store and depends on that one reply, and the verify role never wraps it.

        A prompt past its session's first turn (``later_turn``) carries the server's replies, so
        its answer depends on a contestant's text: it is asked among the call's misses and its
        answer returned, never served from the area nor kept in it."""
        held = self._held(fingerprint)
        # Whether this wrapper has served a prompt, and how many answers it added, for the
        # warning about an area none of whose answers match.
        served = False
        added = 0

        async def cached_call(prompts: list[Envelope]) -> list[Envelope]:
            nonlocal served, added
            questions = [(p.request_id, prompt_digest(p)) for p in prompts]
            # The prompts the area may serve and keep answers to; the rest miss.
            keep = [not later_turn(p) for p in prompts]
            answers = [held.get(q) if k else None for q, k in zip(questions, keep, strict=True)]
            misses = [i for i, a in enumerate(answers) if a is None]
            # Of the prompts the area may serve, how many there are and how many missed.
            servable = sum(keep)
            missed = sum(keep[i] for i in misses)
            # Logged before the engine runs, because that is the only moment the number is
            # actionable: a run that recomputes a split it should have been served otherwise
            # reports it hours later, or not at all if it dies first.
            log.info("reference answers %s: %d hits, %d to answer", fingerprint, len(prompts) - len(misses),
                     len(misses))
            if servable < len(prompts):
                log.warning("reference answers %s: %d of %d prompts come after their session's first turn, so "
                            "they carry the server's earlier replies; the oracle answers them afresh and none "
                            "is kept", fingerprint, len(prompts) - servable, len(prompts))
            # Answers are held and none matches a prompt the area may serve: an area keyed to
            # prompts nobody asks for, which looks like a filled one. A cold area holds nothing,
            # and the answers this wrapper added itself do not count, nor does any call after it
            # served one: a fill asks its split chunk by chunk through one wrapper, so each chunk
            # after the first misses whole while the area holds the chunks before it, and a
            # resumed fill misses whole chunks once it is past the rows a dead one banked.
            stored = len(held) - added
            if missed and missed == servable and stored and not served:
                log.warning("reference answers %s hold %d answers and none match the %d prompts this call asks, "
                            "so the oracle answers all of them. The stored digests were taken over different "
                            "prompts; fill the area with hls oracle fill", fingerprint, stored, servable)
            served = served or missed < servable
            if misses:
                computed = await call([prompts[i] for i in misses])
                kept: list[AnswerEntry] = []
                before = len(held)
                for i, env in zip(misses, computed, strict=True):
                    answers[i] = env.model_dump(mode="json")
                    if keep[i]:
                        held[questions[i]] = answers[i]
                        kept.append(AnswerEntry(request_id=questions[i][0], digest=questions[i][1],
                                                answer=answers[i]))
                added += len(held) - before
                if kept:
                    self.bank(fingerprint, kept)
            return [Envelope.model_validate(a).model_copy(update={"request_id": p.request_id})
                    for p, a in zip(prompts, answers, strict=True)]

        return cached_call

    def bank(self, fingerprint: str, entries: list[AnswerEntry]) -> None:
        """Add the answers one call just computed, *entries* in the call's order, as this writer's
        next file, create-only. A write that fails is logged, or raised as ``AnswersNotWritten``
        by a strict writer."""
        self._files += 1
        path = f"{self._key(fingerprint)}/{self.writer}.{self._files:06d}.json"
        file = AnswerFile(fingerprint=fingerprint, gpu_kind=self.gpu_kind, image=self.image, writer=self.writer,
                          written_at=self.clock(), answers=entries)
        data = file.model_dump_json().encode()
        try:
            self.store.create(path, data)
        except (StoreError, OSError) as exc:
            if self.strict:
                raise AnswersNotWritten(f"could not write reference answers {path}: {exc}") from exc
            log.warning("could not write reference answers %s (%s); this run keeps them in memory and the "
                        "next one recomputes them", path, exc)
            return
        self.written[path] = sha256_bytes(data)
