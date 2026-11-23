"""A task's reference answers to its test split, computed ahead of its gradings.

    python -m hls.verify.fill --task-ref FILE --rows DIR --rows-key KEY --answers DIR --writer NAME
        --gpu-kind KIND --reference-image SHA256 --out FILE --outputs-out REPORT [--chunk 96] [--limit N]
        [--dev-out FILE]

A caching reference's answer to a corpus row depends on the row and the reference and never on
the contestant, so every grading of the task can read it from the store instead of computing it
inside its own wall, a pass every graded run would otherwise repeat. This role
computes them once into the task's answers area (``hls.verify.answers``), for the GPU kind and
the reference image the options name, which are the ones the task's gradings verify on.

It builds the prompts the drive sends for each test row of the tenants a caching reference is
asked about, shaped through the driver's own ``stamp_request`` so they digest as a verifier's
lookups do, and asks each caching reference (``caching_references``) through ``StoreAnswers``
*chunk* rows at a time, for the rows of the tenants whose verifiers hold it, as a verify asks
it. Each chunk that answered a miss is banked as it finishes, so a role that dies keeps every
chunk it finished and a rerun asks only for what is missing. Banking is what the fill is for,
so its writer is strict (``StoreAnswers.strict``): an answer it cannot write ends the role. A
reference is loaded in this process, on the engine host the fill creates, on its first miss and
released after its last chunk, before the next one loads, so a rerun that finds every answer
loads nothing. Once filled, every grading serves the same answers.

Once every reference has been asked every prompt and every answer it computed is banked, the
role records, for each reference fingerprint, the question set it asked
(``StoreAnswers.record_questions``): each prompt's tenant, request id and digest, filed in the
fingerprint's key directory under the name of the rows entry ``KEY`` and the shape of the prompts
(``question_shape``). It is written last, so its presence says the fill completed for those rows;
a fill that dies writes none, and so does one limited to its first prompts (``--limit``). A
plan-time check reads it in place of the rows (``hls.pipeline.preflight``).

Single-turn corpora only, for the tenants read: from turn two, the prompt carries the
contestant's own replies.
``DIR`` holds each tenant's split directories under ``tenants/<id>``, as ``/data`` does in
the role's container, and ``KEY`` is the rows entry they were copied from.

With ``--dev-out``, the role then produces the task's dev reference (``hls.verify.dev_reference``):
its references' answers to the dev rows under ``DIR``, each scored with the gate's metric, and
writes the record there, beside ``FILE``. No answer of it enters the answers area.

The role writes ``FILE`` (``FillRecord``), then reports it, keyed by its name in its directory,
the attempt's outputs directory as the role sees it, in ``REPORT`` for the stage's manifest
(``hls.store.manifest.report_outputs``). The exit status says whether a retry can help, as the
verify role's does:
- 0: ``FILE`` written and reported;
- 3 (``REFUSED_EXIT``): a deterministic refusal (a multi-turn corpus, no caching reference, an
  engine that is not the class its reference declares, a question set already in the area under
  its name that holds other questions, ``QuestionSetConflict``, named by its key in the store),
  one stderr line starting with ``REFUSED_PREFIX``;
- 1: anything else, with the traceback on stderr: an answer that could not be written to the
  area among it (``AnswersNotWritten``, naming the write's error), which a retry asks again; and
  an engine the engine host gave up on, ending the role at once as the verify role's does
  (``hls.verify.role.GIVE_UP``): one that died, or whose release did not return in its bound. The
  answers it banked before stand: each is whole, and the question set that says the fill
  completed is written only after every reference was released."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hls.core.capabilities import Capability, cap_method
from hls.core.payloads import Envelope, TextPayload
from hls.core.poly import dump_spec
from hls.obs.logsetup import configure_logging
from hls.provision.model import check_rows_key
from hls.provision.rows import tenant_subpath
from hls.reference.references import Reference
from hls.spec.datasets import Split
from hls.spec.drivers import stamp_request, user_message
from hls.spec.taskspec import TaskSpec
from hls.store.digest import sha256_bytes
from hls.store.manifest import report_outputs
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, load_task
from hls.verify.answers import (CORPUS_CAPABILITIES, Question, QuestionSetConflict, StoreAnswers, answers_area,
                                prompt_digest)
from hls.verify.base import WiringError, capability_providers
from hls.verify.dev_reference import DevReference, DevReferenceError, produce
from hls.verify.inprocess import EngineHost, Release
from hls.verify.role import GIVE_UP, REFUSED_EXIT, REFUSED_PREFIX

# Named, not ``__name__``: run as a program the module is ``__main__``, which no hls logging
# handler covers.
log = logging.getLogger("hls.verify.fill")

#: Rows asked per chunk.
DEFAULT_CHUNK = 96
#: The file the role writes.
FILL_FILE = "fill.json"


class FillRefused(ValueError):
    """The task cannot be filled ahead of its gradings."""


class ReferenceFill(BaseModel):
    """What one caching reference was asked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The reference's fingerprint, which its answers are filed under.
    fingerprint: str
    #: The engine class it builds.
    engine: str
    #: Prompts it was asked.
    prompts: int
    #: Of those, the ones its engine answered (misses).
    answered: int
    #: Of those, the ones served from the answers area.
    served: int


class FillRecord(BaseModel):
    """``fill.json``: what the fill asked and what it read and added."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each caching reference, in the order it was asked.
    references: list[ReferenceFill]
    #: Each answers file read, by its key in the task's answers area, to its sha256.
    answers_read: dict[str, str]
    #: Each answers file added, by its key in the task's answers area, to its sha256.
    answers_written: dict[str, str]
    #: Each question set the fill recorded once it had banked every answer, by its key in the
    #: task's answers area, to its sha256: added by it, or found there holding the same bytes.
    #: Empty when ``fill`` was given no rows key, as for a role limited to its first prompts
    #: (``--limit``).
    questions: dict[str, str]


def corpus_prompts(task: TaskSpec, rows_root: Path, *, tenants: Collection[str] | None = None) -> list[Envelope]:
    """The envelopes a verifier would hand its reference for *task*'s test split, in corpus
    order: the rows of each tenant, or of each one *tenants* names, of its own dataset, read
    from ``<rows_root>/tenants/<id>/test``, less the rows its removal list names
    (``DataSource.removed_rows``), which the drive drops before it serves. A tenant with no data
    source is served nothing and asked nothing.

    Taken from the dataset rather than from a trace, then shaped through ``stamp_request``,
    the seam the drive loop sends through, with the tenant's output cap. The shaping is what
    makes the answer reachable: the answers are digested over a prompt's contents and its
    ``messages`` and ``max_tokens`` metadata, which a dataset turn carries only once the driver
    stamps them on.

    Raise FillRefused for a row with more than one turn of a tenant read: from turn two the
    stitched history carries the contestant's own replies, so no answer could be shared across
    arms."""
    out: list[Envelope] = []
    for tenant in task.effective_tenants():
        if tenant.data_source is None or (tenants is not None and tenant.id not in tenants):
            continue
        removed = set(tenant.data_source.removed_rows)
        cap = tenant.workload.output_length_cap.max_tokens()
        for item in tenant.data_source.dataset.items_at(Path(rows_root) / tenant_subpath(tenant.id), Split.TEST):
            if item.request_id in removed:
                continue
            if len(item.turns) != 1:
                raise FillRefused(f"{tenant.id}/{item.request_id} has {len(item.turns)} turns; a reference answer "
                                  "is filled ahead of a grading only for single-turn corpora, since turn two "
                                  "onward carries the contestant's replies")
            turn = item.turns[0]
            out.append(stamp_request(turn, [user_message(turn)], tenant_id=tenant.id, session_id=item.request_id,
                                     turn_idx=0, max_tokens=cap))
    return out


@dataclass(frozen=True)
class CachingReference:
    """A reference the fill asks, and for whose rows."""

    #: The corpus capability it is asked through (``CORPUS_CAPABILITIES``).
    capability: type[Capability]
    #: The reference, which caches its answers.
    reference: Reference
    #: The ids of the tenants whose rows it is asked, in the task's order: those whose verifiers
    #: hold it, or every tenant for a pooled verifier, which a verify hands every tenant's events.
    tenants: tuple[str, ...]


def caching_references(task: TaskSpec) -> list[CachingReference]:
    """Each distinct (capability, reference) pair of *task*'s verifiers whose reference caches
    its answers to a corpus capability (``CORPUS_CAPABILITIES``), in declaration order, with the
    tenants whose rows a verify asks it about: the calls the verification core serves from the
    answers area."""
    every = [tenant.id for tenant in task.effective_tenants()]
    found: dict[tuple[type[Capability], int], tuple[type[Capability], Reference, set[str]]] = {}
    for tenant in task.effective_tenants():
        for verifier in tenant.verifiers:
            asked = every if getattr(verifier, "pooled", False) else [tenant.id]
            for cap, cfg in capability_providers(verifier).items():
                if cap in CORPUS_CAPABILITIES and cfg.cache:
                    found.setdefault((cap, id(cfg)), (cap, cfg, set()))[2].update(asked)
    return [CachingReference(capability=cap, reference=cfg, tenants=tuple(t for t in every if t in tenants))
            for cap, cfg, tenants in found.values()]


def question_scopes(task: TaskSpec) -> dict[str, tuple[str, ...]]:
    """Each fingerprint of *task*'s caching references (``caching_references``), the name of one
    answer key, to the tenants whose rows any reference under it is asked, in the task's order:
    the scope of that key's question set."""
    every = [tenant.id for tenant in task.effective_tenants()]
    asked: dict[str, set[str]] = {}
    for ref in caching_references(task):
        asked.setdefault(ref.reference.fingerprint(task), set()).update(ref.tenants)
    return {fingerprint: tuple(t for t in every if t in tenants) for fingerprint, tenants in asked.items()}


#: A turn no corpus holds. Stamped and digested as every prompt is (``_canary``), it puts the
#: code every task shares for that into each question set's shape.
_CANARY = Envelope(request_id="canary", contents=[TextPayload(text="the canary of the question shape")])


def _canary() -> str:
    """The digest of ``_CANARY`` stamped as the drive stamps a first turn and digested as the
    answers area digests a prompt (``stamp_request``, ``prompt_digest``): it changes when that
    shared code changes how any prompt is shaped or digested."""
    stamped = stamp_request(_CANARY, [user_message(_CANARY)], tenant_id="canary", session_id="canary", turn_idx=0,
                            max_tokens=1)
    return prompt_digest(stamped)


def question_shape(task: TaskSpec, tenants: Collection[str]) -> str:
    """The digest of what shapes the prompts ``corpus_prompts`` builds for *task*'s *tenants*
    besides the bytes of their rows, in the task's order: each such tenant's id, its data source
    as it renders a row into a prompt (its dataset and templates, its split, its removal list),
    and the output cap stamped on every prompt; and the digest of a fixed turn put through the
    code every task stamps and digests its prompts with (``_canary``). The files a data source
    discloses beside the dev rows change no prompt, so they are left out. A change to a
    dataset's own rendering code is not covered: it moves its prompts' digests and leaves the
    shape as it was, so a fill of the same rows after it finds its question set conflicting
    (``QuestionSetConflict``) and refuses."""
    shaped = []
    for tenant in task.effective_tenants():
        if tenant.data_source is None or tenant.id not in tenants:
            continue
        source = dump_spec(tenant.data_source)
        source.pop("dev_reference", None)
        source.pop("measured_dev_reference", None)
        shaped.append({"tenant": tenant.id, "data_source": source,
                       "max_tokens": tenant.workload.output_length_cap.max_tokens()})
    return sha256_bytes(json.dumps({"canary": _canary(), "tenants": shaped}, sort_keys=True).encode())


async def _fill_one(task: TaskSpec, cap: type[Capability], cfg: Reference, prompts: list[Envelope],
                    answers: StoreAnswers, *, chunk: int, host: EngineHost) -> ReferenceFill:
    """Ask *cfg*'s engine for its *cap* answers to *prompts*, *chunk* at a time, through
    *answers*. The engine is loaded on *host* on the first miss and released after the last
    chunk. Raise WiringError for an engine that is not the class *cfg* declares, and as
    *answers* does for an answer it cannot write."""
    engine = cfg.build_engine(task)
    name = type(engine).__name__
    if not isinstance(engine, cfg.engine_cls):
        raise WiringError(f"{type(cfg).__name__}.build_engine returned {name}, not the declared engine_cls "
                          f"{cfg.engine_cls.__name__}")
    proxy = None
    release: Release | None = None
    answered = 0

    async def call(misses: list[Envelope]) -> list[Envelope]:
        nonlocal proxy, release, answered
        if proxy is None:
            log.info("loading %s for %d prompts", name, len(misses))
            proxy, release = await host.acquire(engine)
        replies = await getattr(proxy, cap_method(cap))(misses)
        answered += len(misses)
        return replies

    fingerprint = cfg.fingerprint(task)
    wrapped = answers.wrap(fingerprint, call)
    try:
        for start in range(0, len(prompts), chunk):
            await wrapped(prompts[start:start + chunk])
            log.info("%s: %d/%d prompts", name, min(start + chunk, len(prompts)), len(prompts))
    finally:
        if release is not None:
            await release()
    return ReferenceFill(fingerprint=fingerprint, engine=name, prompts=len(prompts), answered=answered,
                         served=len(prompts) - answered)


async def fill(task: TaskSpec, prompts: list[Envelope], answers: StoreAnswers, *,
               chunk: int = DEFAULT_CHUNK, rows_key: str | None = None) -> FillRecord:
    """Fill *answers* with every caching reference of *task*'s answers to the *prompts* of its
    tenants (``CachingReference.tenants``, by each prompt's stamped ``tenant_id``), one reference
    at a time on an engine host of this process. With *rows_key*, the rows entry *prompts* are
    every prompt of (``corpus_prompts`` over its test rows), record each fingerprint's question
    set (``question_scopes``, ``StoreAnswers.record_questions``) once every reference has been
    asked and every answer banked; with None, as for some prompts only, record none. Raise
    FillRefused when *task* has no caching reference."""
    refs = caching_references(task)
    if not refs:
        raise FillRefused(f"{task.id} has no reference that caches its answers to a corpus question "
                          "(cache=True on a reference asked through Generate)")
    done = []
    async with EngineHost(give_up=GIVE_UP) as host:
        for ref in refs:
            asked = [p for p in prompts if p.metadata["tenant_id"] in ref.tenants]
            done.append(await _fill_one(task, ref.capability, ref.reference, asked, answers, chunk=chunk,
                                        host=host))
    # Last: a set on the store says every answer to it was banked.
    questions = {}
    if rows_key is not None:
        for fingerprint, tenants in question_scopes(task).items():
            asked = [Question(tenant=p.metadata["tenant_id"], request_id=p.request_id, digest=prompt_digest(p))
                     for p in prompts if p.metadata["tenant_id"] in tenants]
            path, digest = answers.record_questions(fingerprint, rows=rows_key, shape=question_shape(task, tenants),
                                                    questions=asked)
            questions[path] = digest
    return FillRecord(references=done, answers_read=dict(answers.read), answers_written=dict(answers.written),
                      questions=questions)


async def _dev_reference(task: TaskSpec, module: str, rows: Path) -> DevReference:
    """The dev reference of *task*, the ``TASK`` of *module*, produced on an engine host of this
    process (``hls.verify.dev_reference.produce``)."""
    async with EngineHost(give_up=GIVE_UP) as host:
        return await produce(task, module, rows, host=host)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.verify.fill")
    p.add_argument("--task-ref", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--rows-key", type=check_rows_key, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--writer", required=True)
    p.add_argument("--gpu-kind", required=True)
    p.add_argument("--reference-image", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (a smoke run)")
    p.add_argument("--dev-out", type=Path, default=None, help="also write the task's dev reference here")
    args = p.parse_args(argv)
    if args.chunk < 1 or (args.limit is not None and args.limit < 1):
        p.error("--chunk and --limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    configure_logging()
    try:
        task = load_task(TaskRef.model_validate_json(args.task_ref.read_bytes()))
        # Only the rows a caching reference is asked about: another tenant's may be of a
        # multi-turn corpus, which no reference caches for.
        asked = {tenant for ref in caching_references(task) for tenant in ref.tenants}
        prompts = corpus_prompts(task, args.rows, tenants=asked)
        if args.limit is not None:
            prompts = prompts[:args.limit]
        answers = StoreAnswers(PosixStore(args.answers), gpu_kind=args.gpu_kind, image=args.reference_image,
                               writer=args.writer, strict=True)
        # A fill of the first prompts only completes nothing a grading could rely on.
        rows_key = args.rows_key if args.limit is None else None
        record = asyncio.run(fill(task, prompts, answers, chunk=args.chunk, rows_key=rows_key))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(record.model_dump_json(indent=2))
        written = [args.out]
        if args.dev_out is not None:
            module = TaskRef.model_validate_json(args.task_ref.read_bytes()).module
            args.dev_out.write_text(asyncio.run(_dev_reference(task, module, args.rows)).model_dump_json(indent=1)
                                    + "\n")
            written.append(args.dev_out)
        report_outputs(args.outputs_out, args.out.parent, *written)
    except (FillRefused, WiringError, QuestionSetConflict, DevReferenceError) as e:
        # A conflicting set is named by its key in the store, the task's area being the task's id.
        why = QuestionSetConflict.describe(f"{answers_area(task.id)}/{e.path}") if isinstance(
            e, QuestionSetConflict) else str(e)
        reason = " ".join(why.split())
        print(f"{REFUSED_PREFIX}{type(e).__name__}: {reason}", file=sys.stderr)
        return REFUSED_EXIT
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(f"fill role failed; no {FILL_FILE} was reported", file=sys.stderr)
        return 1
    log.info("filled %d prompts for %d references", len(prompts), len(record.references))
    return 0


if __name__ == "__main__":
    sys.exit(main())
