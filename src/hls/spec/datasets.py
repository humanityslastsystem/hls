"""L1 -- Dataset + Split + HFDataset.

  - Split enum (DEV, TEST)
  - Dataset (poly family base)
  - HFDataset (generic HuggingFace dataset configured by column-templates)

A new task is *config*, not a subclass. `HFDataset` supports two rendering
modes: single-turn (``prompt``/``target`` column templates) and conversational
multi-turn (``context_col``/``questions_col``/``answers_col``). The mode is
inferred from which fields are set; both groups are mutually exclusive.
The ground truth is the rendered target value.

The dataset config holds corpus identity (`path`/`subset`/`revision`) and
rendering. Split policy lives on `SplitSpec` (hls.spec.datasource), not here.
`items_at(mount_dir, split)` is pure rendering: it reads the mounted raw rows
under ``mount_dir/tenants/<id>/<split>/rows.json`` (provisioning is the single
data ingress; nothing downloads at runtime) and renders each row through the
configured mode. Every row carries its own identity (``Dataset.row_id``): the
partition hashes it, and it is the rendered item's ``request_id``.
"""

from __future__ import annotations
import enum
import hashlib
import json
import random
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

from pydantic import Field, model_validator

from hls.core.dataitem import DatasetItem
from hls.core.faults import DatasetError
from hls.core.payloads import AudioPayload, Envelope, ImagePayload, TextPayload
from hls.core.poly import Poly
from hls.prompt import JinjaPromptTemplate, PromptTemplate
from hls.spec.hub import declared_hub_revisions

if TYPE_CHECKING:
    from datasets import Dataset as ArrowDataset


#: The container mount root. Provisioning attaches each tenant's dev/test partitions
#: under ``MOUNT_ROOT/tenants/<id>/<split>/rows.json``, and the drive loop reads them
#: from there. The one definition of the path both sides agree on; patched in tests.
MOUNT_ROOT = Path("/data")

#: Row field holding a row's position in its source split, written by ``corpus()`` for
#: a dataset whose source carries no identifier of its own; that position is then the
#: row's identity (``Dataset.row_id``).
SOURCE_INDEX = "source_index"


# ---- Split ---------------------------------------------------------------

class Split(str, enum.Enum):
    DEV = "dev"
    TEST = "test"


# ---- Dataset family base -------------------------------------------------

class Dataset(Poly):
    """A source of evaluation items plus the metadata to describe them (L1).

    `items_at(mount_dir, split)` yields DatasetItems (prompt envelopes +
    ground-truth turns) from rows mounted under a given directory.
    A concrete task is usually config of the generic HFDataset, not a subclass.
    Split policy is not a dataset concern -- it lives on SplitSpec; `items_at`
    reads the mounted partition for the split, in file order (provisioning writes
    the rows in the partition's hash order). In-memory test doubles ignore
    ``mount_dir`` and yield their items directly.
    """

    name: ClassVar[str] = ""
    version: ClassVar[str] = "1.0"

    @abstractmethod
    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]: ...

    # Provisioning interface ------------------------------------------------
    # Provisioning depends only on these (never on HF-only fields), so a Dataset
    # is provisionable iff it can yield its corpus. The corpus is a memory-mapped
    # ``datasets.Dataset`` (Arrow), so the full set is never loaded into RAM --
    # SplitSpec partitions it lazily via ``.select``. Non-provisionable datasets
    # (stashed onto the engine, e.g. inline test doubles) inherit the clear error.
    def corpus(self) -> "ArrowDataset":
        """The full corpus as a memory-mapped ``datasets.Dataset`` (SplitSpec
        partitions it). Provisionable datasets override this."""
        raise NotImplementedError(
            f"{type(self).__name__} is not provisionable (no corpus()); stash it "
            "onto the engine instead of running mount mode.")

    def hub_revisions(self) -> list[tuple[str, str]]:
        """Every hub repo ``corpus()`` reads, with the revision it reads it at, as
        ``(repo, revision)``. The rows key stands for fixed bytes only when each revision is
        a commit, which the provision role checks (``hls.spec.hub``). Empty only for a
        dataset whose ``corpus()`` reads nothing from the hub. A dataset declares this beside
        its ``corpus()``, so a subclass that reads more says so; this one raises, so a dataset
        that declares nothing is refused."""
        raise NotImplementedError(f"{type(self).__name__} declares no hub sources")

    def corpus_id(self) -> str:
        """Stable corpus-origin string for content-addressing + the provisioned
        dir name. Provisionable datasets override this."""
        raise NotImplementedError(
            f"{type(self).__name__} has no corpus_id(); it is not provisionable.")

    def row_id(self, row: dict) -> str:
        """The row's identity in the dataset's own terms, read off a corpus row and so
        off a mounted row: BFCL's test id, WildChat's conversation hash, a Multi-IF key,
        or -- for a source with no identifier -- the row's position in its source split
        (``SOURCE_INDEX``). It is the rendered item's ``request_id`` (turn ``k`` of a
        session is ``<row_id>.<k>``), what the dev/test partition hashes
        (``hls.spec.datasource.partition_by_hash``) and what a task's removal list
        names. Provisionable datasets override this."""
        raise NotImplementedError(
            f"{type(self).__name__} has no row_id(); it is not provisionable.")

    def forced_split(self, row: dict) -> str | None:
        """The split this row belongs in by declaration, or None to let the hash decide.
        Only ``"test"`` is honoured (``partition_by_hash``'s ``pinned_test``).

        For a corpus holding a handful of rows that every graded run must contain -- a few
        long sessions among many short ones, say -- where a hash gives no such guarantee
        and its answer moves whenever the corpus does. A dataset that declares none
        partitions exactly as it always has."""
        return None

    def draws(self, row: dict) -> int:
        """How many sessions this row supplies. One unless a dataset says otherwise.

        A row is normally one session. Where a few rows carry traffic nothing else can,
        drawing one several times is the only way past that ceiling without inventing
        corpus content. Each draw is a session of its own with its own id; what they share
        is the prompt they start from, so they are repeat samples of one trajectory."""
        return 1


# ---- Provisioned-rows base -----------------------------------------------

class ProvisionedRowsDataset(Dataset):
    """A provisionable Dataset whose dev/test partition is materialized to
    ``<root>/<split>/rows.json`` (by the provision rows role) and read back row-by-row.

    Owns the mount/rows plumbing ONCE, so a concrete dataset supplies only the
    write side (``corpus()``/``corpus_id()``/``row_id()``) and a per-row renderer
    (``_render_row``). ``items_at`` honors the ``mount_dir`` the driver passes --
    the per-tenant mount ``MOUNT_ROOT/tenants/<id>`` -- so a multi-tenant run reads
    each tenant's own rows."""

    def _rows_at(self, mount_dir, split: Split) -> list:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        return json.loads(path.read_text())

    @abstractmethod
    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem | None:
        """Render one raw row into a DatasetItem, or return None to skip it (e.g.
        an empty multi-turn conversation).

        The request id must be ``row_id(row)`` (turn ``k`` of a session is
        ``<row_id>.<k>``), never ``idx``. ``idx`` is the row's position in the
        mounted partition, which moves when the split sizes or the corpus size
        move; ``row_id`` names the same row whatever order the partition landed in,
        which is what the stored reference answers are keyed on
        (``hls.verify.answers``). Pass ``idx``
        through as metadata if the item wants it."""
        ...

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        for idx, row in enumerate(self._rows_at(mount_dir, split)):
            item = self._render_row(row, idx, split)
            if item is not None:
                yield item

    def _indexed(self, corpus: "ArrowDataset") -> "ArrowDataset":
        """Stamp each row with its position in the emitted corpus, the identity
        ``row_id`` reads. Call it on whatever ``corpus()`` returns: these datasets
        build their rows rather than passing a provider's through, so position is the
        only identifier they have, and it must be assigned once at encode time so the
        dev/test partition is stable across runs."""
        return corpus.add_column(SOURCE_INDEX, list(range(len(corpus))))

    def row_id(self, row: dict) -> str:
        if SOURCE_INDEX not in row:
            # Reachable from a mounted row, not just at provision time: the renderers
            # key their request ids on this. A mount whose rows lack the column would
            # otherwise surface as a bare KeyError inside the drive.
            raise DatasetError(
                f"{type(self).__name__} row carries no {SOURCE_INDEX!r}, which corpus() "
                f"stamps on every row it writes; re-provision this corpus")
        return str(row[SOURCE_INDEX])


# ---- Generic HFDataset ---------------------------------------------------

class HFDataset(ProvisionedRowsDataset):
    """A HuggingFace dataset with two rendering modes. A new task is
    config, not a subclass.

    **Single-turn mode** (default): set ``prompt`` + ``target``. Each row is
    rendered into one Envelope via ``prompt``/``target`` PromptTemplates.

    **Conversational multi-turn mode**: set ``context_col`` + ``questions_col``
    + ``answers_col`` instead. Each row is a shared-context conversation (e.g.
    CoQA); turn 0 prepends the passage via ``first_turn_template``; later turns
    send only the question. Conversations are capped to ``max_turns``.

    The two modes are mutually exclusive (validated at construction). Identity
    is ``path``/``subset``/``revision``; ``items_at()`` renders the partition
    mounted under ``MOUNT_ROOT/tenants/<id>/<split>/rows.json`` (no runtime
    download)."""

    path: str
    subset: str | None = None
    revision: str = "main"
    #: The corpus column that identifies a row (e.g. Multi-IF's ``key``). ``None``
    #: when the source has none: the row's position in ``source_split`` is its identity
    #: then, and ``corpus()`` records it on the row as ``SOURCE_INDEX``.
    id_column: str | None = None
    #: The HuggingFace provider split that provisioning loads as the corpus, which
    #: SplitSpec then partitions into the dev/test units. This is the corpus
    #: origin, not the dev/test split. Required (no default): a "test" default would
    #: pick the wrong corpus for a dataset without a test split. Each task declares it
    #: explicitly (e.g. MMLU -> "test", CoQA -> "validation").
    source_split: str
    # Single-turn rendering (the default): one envelope per row from prompt/target column templates.
    prompt: PromptTemplate | None = None
    target: PromptTemplate | None = None
    # Multi-turn (conversational) rendering: set these instead to render each row as a capped
    # multi-turn conversation. Mutually exclusive with prompt/target.
    context_col: str | None = None
    questions_col: str | None = None
    answers_col: str | None = None
    answers_text_key: str | None = "input_text"
    first_turn_template: str = (
        "Answer each question with a short answer based on the passage.\n\n"
        "Passage:\n{context}\n\nQuestion: {question}")
    max_turns: int = 5

    def _is_multi_turn(self) -> bool:
        """Single source of the single-vs-multi-turn mode decision: multi-turn iff
        any conversational column (``context_col``/``questions_col``/``answers_col``)
        is set. Used by both ``_validate_mode`` (which then enforces that all three
        are present) and ``items()`` so the invariant lives in one place."""
        return (self.context_col is not None
                or self.questions_col is not None
                or self.answers_col is not None)

    @model_validator(mode="after")
    def _validate_mode(self) -> "HFDataset":
        single = self.prompt is not None or self.target is not None
        multi = self._is_multi_turn()
        if single and multi:
            raise ValueError("HFDataset: set EITHER prompt+target (single-turn) OR "
                             "context_col+questions_col+answers_col (multi-turn), not both")
        if not single and not multi:
            raise ValueError("HFDataset: must set prompt+target (single-turn) or "
                             "context_col+questions_col+answers_col (multi-turn)")
        if single and (self.prompt is None or self.target is None):
            raise ValueError("single-turn HFDataset requires both prompt and target")
        if multi and not (self.context_col and self.questions_col and self.answers_col):
            raise ValueError("multi-turn HFDataset requires context_col, questions_col, and answers_col")
        return self

    # -- single-turn helpers -----------------------------------------------

    def _single_turn_item(self, row: dict, req_id: str, idx: int, split_value: str) -> DatasetItem:
        env = Envelope(request_id=req_id,
                       contents=[TextPayload(text=self.prompt.render(**row))])
        return DatasetItem(
            request_id=req_id,
            turns=(env,),
            ground_truth_turns=(self.target.render(**row),),
            metadata={"split": split_value, "index": idx})

    # -- multi-turn helpers ------------------------------------------------

    def _answer_texts(self, answers) -> list:
        if self.answers_text_key is not None and isinstance(answers, dict):
            return list(answers.get(self.answers_text_key, []))
        return list(answers)

    def _multi_turn_item(self, row: dict, req_id: str, idx: int, split_value: str) -> DatasetItem | None:
        context = str(row[self.context_col])
        questions = list(row[self.questions_col])
        answer_texts = self._answer_texts(row[self.answers_col])
        n = min(self.max_turns, len(questions), len(answer_texts))
        if n == 0:
            return None
        envs = []
        for i in range(n):
            if i == 0:
                user_text = self.first_turn_template.format(
                    context=context, question=str(questions[0]))
            else:
                user_text = str(questions[i])
            envs.append(Envelope(request_id=f"{req_id}.{i}",
                                 contents=[TextPayload(text=user_text)]))
        gts = tuple(str(answer_texts[i]) for i in range(n))
        return DatasetItem(
            request_id=req_id, turns=tuple(envs), ground_truth_turns=gts,
            metadata={"split": split_value, "index": idx, "n_turns": n})

    # -- per-row renderer --------------------------------------------------

    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem | None:
        """One row -> one DatasetItem, keyed by the row's own identity (``row_id``)
        rather than its position, so a request id names the same row whatever order
        the partition landed in."""
        req_id = self.row_id(row)
        if self._is_multi_turn():
            return self._multi_turn_item(row, req_id, idx, split.value)
        return self._single_turn_item(row, req_id, idx, split.value)

    def corpus(self) -> "ArrowDataset":
        from datasets import load_dataset
        raw = load_dataset(self.path, self.subset, revision=self.revision,
                           split=self.source_split)
        raw = self._project(raw)
        # After any column projection, so a projection that does not list it cannot
        # drop the fallback identity the partition is hashed on.
        if self.id_column is None:
            raw = raw.add_column(SOURCE_INDEX, list(range(len(raw))))
        return raw

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.path, self.revision)]

    def _project(self, raw: "ArrowDataset") -> "ArrowDataset":
        """Narrow the loaded corpus before it is materialized. The identity here;
        ``ColumnProjectedHFDataset`` keeps only an allow-list of columns."""
        return raw

    def corpus_id(self) -> str:
        """Identity of the provisioned corpus, and the name of its cache directory.

        ``id_column`` belongs here because it changes what is written: leaving it unset
        appends a synthetic ``SOURCE_INDEX`` column in ``corpus()``, and it is what
        ``row_id`` keys each row by, so it decides the dev/test partition and what a
        removal list names. Two configs differing only in it would otherwise share a
        directory, and whichever provisioned first would serve the other."""
        return (f"{self.path}:{self.subset}@{self.revision}|src={self.source_split}"
                f"|id={self.id_column if self.id_column is not None else SOURCE_INDEX}")

    def row_id(self, row: dict) -> str:
        return str(row[self.id_column if self.id_column is not None else SOURCE_INDEX])


class ColumnProjectedHFDataset(HFDataset):
    """An HFDataset that keeps only an allow-list of its source columns.

    Provisioning materializes the dev/test partition as JSON, so a corpus carrying a
    column that does not serialize -- a TTS prompt set shipping the reference audio
    beside the text, say -- cannot be written at all. Projecting to the columns the
    task actually renders drops the offender before it reaches the dump.

    A separate class rather than a field on ``HFDataset`` because the projection is a
    step only some corpora need. Its ``kind`` and ``keep_columns`` are in the dump, so
    the workload identity names the projection, and its corpus id does too."""

    #: Columns to keep, applied before materialization. Must cover every column the
    #: prompt/target templates read, and ``id_column`` when one is set.
    keep_columns: list[str]

    def _project(self, raw: "ArrowDataset") -> "ArrowDataset":
        return raw.select_columns(self.keep_columns)

    def corpus_id(self) -> str:
        # The projection changes which rows.json columns exist, so it belongs in the
        # corpus identity: two otherwise-equal datasets keeping different columns are
        # different corpora and must not share a provisioned directory.
        return super().corpus_id() + f"|keep={','.join(self.keep_columns)}"


# ---- Multi-IF (multi-turn instruction following) -------------------------

class MultiIFDataset(HFDataset):
    """facebook/Multi-IF: a multi-turn instruction-following benchmark.

    A third rendering mode, distinct from HFDataset's single-turn (prompt/target) and
    CoQA-style conversational (context/questions/answers) modes. Each row is a
    conversation of up to ``max_turns`` turns held in fixed ``turn_{n}_prompt`` /
    ``turn_{n}_instruction_id_list`` / ``turn_{n}_kwargs`` columns. Every turn's ground
    truth is an IFEval spec ``{"instruction_id_list", "kwargs"}`` (graded by
    IFEvalConstraint), not a text answer -- so it rides through ``ground_truth_turns``
    (typed Any) as a dict. Turn i's instruction list is cumulative (it already includes
    the constraints introduced in turns 1..i), matching Multi-IF's per-turn scoring.

    Only the ``language`` subset (default English) is kept: the checker registry and
    langdetect target English prompts, and the corpus mixes many languages. Provisioning
    is inherited from HFDataset; ``corpus()`` is narrowed to the chosen language so
    SplitSpec samples only those rows.
    """

    #: Multi-IF has exactly three turn columns; cap conversations here.
    max_turns: int = 3
    #: Corpus language subset to keep (Multi-IF is multilingual; English by default).
    language: str = "English"
    #: Multi-IF's own per-conversation key.
    id_column: str | None = "key"

    @model_validator(mode="after")
    def _validate_mode(self) -> "MultiIFDataset":
        # Override HFDataset's prompt/target-vs-conversational requirement: Multi-IF
        # renders from fixed turn_{n}_* columns and sets none of those fields.
        return self

    @staticmethod
    def _load_list(cell):
        """A Multi-IF list cell is stored as a JSON string (list); tolerate a real list."""
        if cell is None:
            return []
        return json.loads(cell) if isinstance(cell, str) else list(cell)

    def _turn_spec(self, ids_cell, kwargs_cell) -> dict:
        """Render one turn's IFEval ground-truth spec. Each kwargs entry is itself a
        JSON string (the on-disk shape is a list-of-JSON-strings), decoded to a dict."""
        ids = [str(i) for i in self._load_list(ids_cell)]
        kwargs = [json.loads(k) if isinstance(k, str) else dict(k)
                  for k in self._load_list(kwargs_cell)]
        return {"instruction_id_list": ids, "kwargs": kwargs}

    @staticmethod
    def _prompt_text(cell) -> str | None:
        """Extract a turn's user text from its ``{"role","content"}`` prompt cell
        (a JSON string on disk). Returns None when the turn is absent."""
        if cell is None:
            return None
        obj = json.loads(cell) if isinstance(cell, str) else cell
        return str(obj.get("content", "")) if isinstance(obj, dict) else str(obj)

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        rows = json.loads(path.read_text())
        for idx, row in enumerate(rows):
            req_id = self.row_id(row)
            envs: list[Envelope] = []
            gts: list[dict] = []
            for t in range(self.max_turns):
                n = t + 1
                text = self._prompt_text(row.get(f"turn_{n}_prompt"))
                if text is None:
                    break  # conversations are contiguous; stop at the first absent turn
                envs.append(Envelope(request_id=f"{req_id}.{t}",
                                     contents=[TextPayload(text=text)]))
                gts.append(self._turn_spec(row.get(f"turn_{n}_instruction_id_list"),
                                           row.get(f"turn_{n}_kwargs")))
            if envs:
                yield DatasetItem(
                    request_id=req_id, turns=tuple(envs), ground_truth_turns=tuple(gts),
                    metadata={"split": split.value, "index": idx, "n_turns": len(envs)})

    def corpus(self) -> "ArrowDataset":
        return super().corpus().filter(lambda r: r["language"] == self.language)

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() reshapes the parent's corpus and reads nothing more.
        return super().hub_revisions()

    def corpus_id(self) -> str:
        return super().corpus_id() + f"|lang={self.language}"


# ---- ChatRAG-Bench (TopiOCQA) ---------------------------------------------

# Verbatim from the official ChatQA rendering (nvidia/Llama3-ChatQA-1.5-8B model
# card, get_formatted_input()), fetched from:
# https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B/raw/main/README.md
CHATQA_INSTRUCTION = (
    "System: This is a chat between a user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions "
    "based on the context. The assistant should also indicate when the answer cannot be "
    "found in the context.\n\n")
CHATQA_QUESTION_INSTRUCTION = "Please give a full and complete answer for the question."


class ChatRAGDataset(HFDataset):
    """nvidia/ChatRAG-Bench conversational QA (TopiOCQA config).

    A fourth rendering mode. Raw ChatRAG rows are per-turn (full gold history in
    ``messages`` + per-turn retrieved ``ctxs``); ``corpus()`` groups them into one
    row per conversation so SplitSpec partitions conversation-granularly, and
    ``items_at`` renders each turn's fresh content only ([ctxs_k + q_k]; turn 0
    also carries the ChatQA instruction). The Driver stitches history with the
    model's own replies -- same contract as CoQA. Ground truth = gold answer
    strings, graded per-turn by CoqaF1Metric via ReferenceRelativeVerifier.

    The source carries no conversation id, so a conversation's position in the
    source split's turn-ordered sequence identifies it (``SOURCE_INDEX``)."""

    n_ctx: int = 5
    instruction: str = CHATQA_INSTRUCTION

    @model_validator(mode="after")
    def _validate_mode(self) -> "ChatRAGDataset":
        return self  # fixed rendering; no prompt/target or context_col fields

    @staticmethod
    def _group_rows(raw_rows: list, n_ctx: int) -> list[dict]:
        """Group ChatRAG's per-turn raw rows into one dict per conversation:
        ``{"questions", "ctxs", "answers"}``, each a per-turn list in turn order.

        Boundaries come from the gold history every row carries: a row's user-message
        count is that row's 1-based turn number, so a row with one user message opens a
        conversation and any later row continues the one before it. Rows must therefore
        arrive grouped by conversation and in turn order.

        That is also checkable from the same count, and is checked: turn k may only
        extend a conversation already holding k-1 turns. A reordered or partially
        filtered corpus otherwise stitches a turn onto the wrong conversation, and the
        run then grades those answers against another conversation's gold -- a wrong
        score with nothing anywhere saying so."""
        convs: list[dict] = []
        for i, row in enumerate(raw_rows):
            msgs = row.get("messages") or []
            n_user = sum(1 for m in msgs if m.get("role") == "user")
            question = next((m["content"] for m in reversed(msgs)
                             if m.get("role") == "user"), "")
            ctx = "\n".join((c.get("text") or "") for c in (row.get("ctxs") or [])[:n_ctx])
            answer = (row.get("answers") or [""])[0]
            if n_user <= 1 or not convs:
                convs.append({SOURCE_INDEX: len(convs), "questions": [], "ctxs": [],
                              "answers": []})
            elif n_user != len(convs[-1]["questions"]) + 1:
                raise DatasetError(
                    f"ChatRAG row {i} carries {n_user} user message(s) -- turn {n_user} "
                    f"of its conversation -- but follows a conversation of "
                    f"{len(convs[-1]['questions'])} turn(s). Raw rows must arrive "
                    f"grouped by conversation in turn order; grouping a reordered or "
                    f"filtered corpus would attach this turn to the wrong conversation.")
            convs[-1]["questions"].append(str(question))
            convs[-1]["ctxs"].append(ctx)
            convs[-1]["answers"].append(str(answer))
        return convs

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        for idx, conv in enumerate(json.loads(path.read_text())):
            req_id = self.row_id(conv)
            envs = []
            for k, (q, ctx) in enumerate(zip(conv["questions"], conv["ctxs"])):
                if k == 0:
                    text = (f"{self.instruction}{ctx}\n\n"
                            f"User: {q} {CHATQA_QUESTION_INSTRUCTION}")
                else:
                    text = f"{ctx}\n\n{q} {CHATQA_QUESTION_INSTRUCTION}"
                envs.append(Envelope(request_id=f"{req_id}.{k}",
                                     contents=[TextPayload(text=text)]))
            if envs:
                yield DatasetItem(
                    request_id=req_id, turns=tuple(envs),
                    ground_truth_turns=tuple(str(a) for a in conv["answers"]),
                    metadata={"split": split.value, "index": idx, "n_turns": len(envs)})

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS, load_dataset
        raw = load_dataset(self.path, self.subset, revision=self.revision,
                           split=self.source_split)
        return ArrowDS.from_list(self._group_rows(raw.to_list(), self.n_ctx))

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.path, self.revision)]

    def corpus_id(self) -> str:
        return super().corpus_id() + f"|convs|nctx={self.n_ctx}"


# ---- MRCR (long-context multi-round coreference) --------------------------

#: Tokenizer memoization for the corpus()-time token filters (MRCR, WildChat,
#: CodeCompletion) -- AutoTokenizer.from_pretrained is expensive (first call
#: downloads/loads the full tokenizer); keyed by tokenizer id and revision so configs that
#: share one (the common case) load it once per process.
_TOKENIZER_CACHE: dict[tuple[str, str], object] = {}


def _tokenizer(tokenizer_id: str, revision: str):
    """The memoized HuggingFace tokenizer ``tokenizer_id`` at ``revision`` (corpus-time only)."""
    tok = _TOKENIZER_CACHE.get((tokenizer_id, revision))
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
        _TOKENIZER_CACHE[(tokenizer_id, revision)] = tok
    return tok


class MRCRDataset(HFDataset):
    """openai/mrcr: long-context multi-round-coreference resolution.

    A fifth rendering mode. Each raw row already carries the full scripted
    OpenAI conversation (``prompt``, a JSON string of ``{role, content}``
    messages) ending in the final user question to answer -- there is no
    driver-side stitching to do. Each item is single-turn and its Envelope
    pre-renders the entire history into ``metadata["messages"]`` for the
    Driver's passthrough (``Driver._session``: a non-empty pre-rendered
    ``messages`` list is used verbatim instead of the driver-stitched
    history). Grading (``MRCRRatio``) needs the per-item random ``prepend``
    string plus the gold ``answer``, both carried verbatim in
    ``ground_truth_turns``.

    ``replay`` turns each conversation into a session instead: one request per
    scripted user turn, each carrying the dataset's own message list up to and
    including that turn as ``metadata["messages"]``. The history stays scripted
    (never the artifact's live replies), so every turn's context is exactly the
    corpus's and the needle structure is preserved, while each turn is a distinct
    request with a growing prefill.

    ``corpus()`` applies MRCR's official two-stage length filter so
    provisioning only keeps rows that fit ``max_input_tokens``: first a cheap
    char-count pre-filter (``n_chars <= 8 * max_input_tokens``, a loose
    chars-per-token lower bound that cheaply rejects the long tail before
    tokenizing), then an exact token count of the concatenated message
    contents via the configured tokenizer. The source carries no row id, so a
    row's position in the unfiltered source split identifies it
    (``SOURCE_INDEX``); the filter changes which rows are kept, never their ids.
    """

    max_input_tokens: int = 8176
    tokenizer_id: str = "Qwen/Qwen2.5-32B"
    #: The commit the corpus-time tokenizer is loaded at. A filter's cut depends on the
    #: tokenizer, so the commit decides which rows are kept.
    tokenizer_revision: str = "main"
    #: MRCR ships a single HuggingFace split ("train"); dev/test are carved
    #: from it by SplitSpec, not by the provider.
    source_split: str = "train"
    #: Cumulative scripted replay (see the class docstring). Rendering-only, so it
    #: stays out of corpus_id(): both modes read the same provisioned rows.
    replay: bool = False
    #: Cap the number of rendered user turns per replay session (first N turns).
    #: Rendering-only, like ``replay`` -- it truncates the session, not the
    #: provisioned pool, so it stays out of ``corpus_id()`` (both capped and
    #: uncapped renders read the same rows). ``None`` = unbounded.
    #: Overrides the parent HFDataset.max_turns (int = 5, conversational-mode
    #: cap) with a nullable sentinel that is meaningless outside replay mode.
    max_turns: int | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "MRCRDataset":
        return self  # fixed rendering; no prompt/target or context_col fields

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        return next((str(m.get("content", "")) for m in reversed(messages)
                    if m.get("role") == "user"), "")

    def _replay_item(self, row: dict, req_id: str, idx: int, split_value: str) -> DatasetItem:
        """One session per conversation: turn k = the scripted history through user
        turn k. Each turn's ground truth is the reply the corpus scripts for it; the
        last user turn has no scripted reply, so it carries MRCR's own gold
        ``{"answer", "prepend"}`` pair."""
        messages = row["messages"]
        envs: list[Envelope] = []
        gts: list[dict] = []
        for j, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            if self.max_turns is not None and len(envs) >= self.max_turns:
                break
            nxt = messages[j + 1] if j + 1 < len(messages) else None
            envs.append(Envelope(
                request_id=f"{req_id}.{len(envs)}",
                contents=[TextPayload(text=str(msg.get("content", "")))],
                metadata={"messages": messages[:j + 1]}))
            gts.append({"answer": str(nxt["content"]), "prepend": ""}
                       if nxt is not None and nxt.get("role") == "assistant"
                       else {"answer": row["answer"], "prepend": row["prepend"]})
        return DatasetItem(request_id=req_id, turns=tuple(envs),
                           ground_truth_turns=tuple(gts),
                           metadata={"split": split_value, "index": idx,
                                     "n_turns": len(envs)})

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        for idx, row in enumerate(json.loads(path.read_text())):
            req_id = self.row_id(row)
            if self.replay:
                item = self._replay_item(row, req_id, idx, split.value)
                if item.turns:
                    yield item
                continue
            messages = row["messages"]
            env = Envelope(request_id=req_id,
                           contents=[TextPayload(text=self._last_user_text(messages))],
                           metadata={"messages": messages})
            yield DatasetItem(
                request_id=req_id, turns=(env,),
                ground_truth_turns=({"answer": row["answer"], "prepend": row["prepend"]},),
                metadata={"split": split.value, "index": idx})

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS, load_dataset
        tok = _tokenizer(self.tokenizer_id, self.tokenizer_revision)
        raw = load_dataset(self.path, revision=self.revision, split=self.source_split,
                           streaming=True)
        char_cap = 8 * self.max_input_tokens
        rows = []
        for position, ex in enumerate(raw):
            if ex["n_chars"] > char_cap:
                continue
            messages = json.loads(ex["prompt"])
            text = "\n".join(m["content"] for m in messages)
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_tok > self.max_input_tokens:
                continue
            rows.append({SOURCE_INDEX: position, "messages": messages, "answer": ex["answer"],
                        "prepend": ex["random_string_to_prepend"]})
        return ArrowDS.from_list(rows)

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() counts tokens with the corpus-time tokenizer, which it loads from the hub.
        return [(self.path, self.revision), (self.tokenizer_id, self.tokenizer_revision)]

    def corpus_id(self) -> str:
        return (super().corpus_id()
                + f"|cap={self.max_input_tokens}|tok={self.tokenizer_id}@{self.tokenizer_revision}")


# ---- BFCL v3 (Berkeley Function-Calling Leaderboard, AST categories) ------

# This is not gorilla's `_DEFAULT_SYSTEM_PROMPT` constant -- that constant is
# explicitly labelled "NOT USED, just for reference" in gorilla's own source.
# For every plain AST-category test id (no ":" in the id -- true for all 8
# categories in `categories` below), gorilla's live composition path is:
#   extract_prompt_format_from_id(id) -> DEFAULT_SYSTEM_PROMPT_FORMAT
#     ("ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic")
#   -> formulate_system_prompt() renders
#        PROMPT_STYLE_TEMPLATES["classic"] + _PLAINTEXT_SYSTEM_PROMPT_TEMPLATE
#        + format_function_doc(functions, "json")   # == json.dumps(functions, indent=4)
# This constant is that exact composed text, byte for byte, verified by
# replicating gorilla's own functions (not paraphrasing them) against a
# fixture and diffing the output (see tests/unit/test_bfcl.py's golden-string
# assertion). The `{functions}` placeholder is filled with
# `json.dumps(functions, indent=4)` in `items_at()` below -- pretty-printed,
# embedded mid-template, matching `format_function_doc`'s "json" branch and
# PROMPT_STYLE_TEMPLATES["classic"]["available_tools"] exactly (not appended
# compact after the template, and not persona+task separated by a space --
# gorilla's plaintext template concatenates them directly). Pinned sha:
# https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/
#   berkeley-function-call-leaderboard/bfcl_eval/constants/default_prompts.py
#   berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py
#   berkeley-function-call-leaderboard/bfcl_eval/utils.py
BFCL_SYSTEM_TEMPLATE = (
    "You are an expert in composing functions."
    "You are given a question and a set of possible functions. Based on the question, you will "
    "need to make one or more function/tool calls to achieve the purpose. If none of the "
    "functions can be used, point it out. If the given question lacks the parameters required "
    "by the function, also point it out.\n\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]."
    "  You SHOULD NOT include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user "
    "within the current turn. Continue to output functions to call until you have fulfilled "
    "the user's request to the best of your ability. Once you have no more functions to call, "
    "the system will consider the current turn complete and proceed to the next turn "
    "or task.\n\n"
    "Here is a list of functions in json format that you can invoke.\n"
    "{functions}\n"
)


class BFCLDataset(Dataset):
    """gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3): single-turn
    function-calling over 8 AST categories, graded against gold
    ``possible_answer`` allowed-value lists (``BFCLAstMatch``, pure-Python AST
    comparison -- no execution). Prompting mode (no native tool-calling API):
    the official gorilla system template + the row's JSON function schemas
    render as one system message; the item text is the user question.

    A sixth Dataset rendering mode. Unlike `HFDataset`'s single
    ``load_dataset()``-backed corpus, BFCL_v3's raw files are plain JSONL
    fetched per category via ``hf_hub_download`` -- a question file plus a
    separate ``possible_answer`` file, joined on ``id`` -- so this subclasses
    `Dataset` directly rather than `HFDataset`.

    Each item is single-turn and pre-renders the full two-message OpenAI
    conversation into ``metadata["messages"]`` for the Driver's passthrough
    (same contract as `MRCRDataset`). Ground truth carries both the
    allowed-value lists and the function schemas verbatim:
    ``{"ground_truth": [...], "functions": [...]}``.
    """

    repo: str = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    #: The dataset revision to fetch. No default: the pin is part of a task's
    #: corpus identity, so a task states it rather than inheriting one that could
    #: move under an already-calibrated workload.
    revision: str
    categories: tuple[str, ...] = (
        "simple", "multiple", "parallel", "parallel_multiple",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple")

    @staticmethod
    def _first_user_text(question: list) -> str:
        """BFCL's raw `question` field is a list of TURNS, each a list of
        {role, content} messages -- a multi-turn container even for these
        single-turn AST categories (exactly one turn, one user message).
        Take the first turn's first user message's content."""
        for turn in question:
            for m in turn:
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content", ""))
        return ""

    @staticmethod
    def _decode(cell):
        """A corpus row's `functions`/`ground_truth` cell may be a JSON string
        (as `corpus()` stores it -- pyarrow can't infer one shared struct
        schema across BFCL's per-function, per-parameter-shaped dicts, whose
        value types vary call to call) or an already-decoded list (in-memory
        test fixtures). Tolerate both, matching `MultiIFDataset._load_list`."""
        return json.loads(cell) if isinstance(cell, str) else cell

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        for idx, row in enumerate(json.loads(path.read_text())):
            req_id = self.row_id(row)
            functions = self._decode(row["functions"])
            ground_truth = self._decode(row["ground_truth"])
            system_text = BFCL_SYSTEM_TEMPLATE.format(
                functions=json.dumps(functions, indent=4))
            messages = [{"role": "system", "content": system_text},
                       {"role": "user", "content": row["question"]}]
            env = Envelope(request_id=req_id,
                           contents=[TextPayload(text=row["question"])],
                           metadata={"messages": messages})
            yield DatasetItem(
                request_id=req_id, turns=(env,),
                ground_truth_turns=({"ground_truth": ground_truth, "functions": functions},),
                metadata={"split": split.value, "index": idx, "category": row.get("category")})

    def row_id(self, row: dict) -> str:
        """BFCL's own test id (``simple_12``, ``live_multiple_1010-239-0``, ...)."""
        return str(row["id"])

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS
        from huggingface_hub import hf_hub_download
        rows = []
        for cat in self.categories:
            qf = hf_hub_download(self.repo, f"BFCL_v3_{cat}.json", repo_type="dataset",
                                 revision=self.revision)
            af = hf_hub_download(self.repo, f"possible_answer/BFCL_v3_{cat}.json",
                                 repo_type="dataset", revision=self.revision)
            answers: dict = {}
            with open(af) as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line)
                        answers[j["id"]] = j["ground_truth"]
            with open(qf) as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    gt = answers.get(j["id"])
                    if not gt:
                        continue
                    rows.append({
                        "id": j["id"], "category": cat,
                        "question": self._first_user_text(j["question"]),
                        # JSON-encoded: pyarrow's schema inference fails on BFCL's
                        # heterogeneous per-function/per-parameter dict shapes
                        # (mixed str/int/bool/list values sharing a column); same
                        # workaround as MultiIFDataset's turn_n_kwargs cells.
                        "functions": json.dumps(j["function"]),
                        "ground_truth": json.dumps(gt),
                    })
        return ArrowDS.from_list(rows)

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.repo, self.revision)]

    def corpus_id(self) -> str:
        return f"{self.repo}@{self.revision}|cats={','.join(self.categories)}"


# ---- MMLU-Pro (10-way multiple-choice knowledge QA) -----------------------

MMLU_PRO_INSTRUCTION = (
    "Answer the following multiple choice question. Think step by step, then "
    "finish your response with \"The answer is (X)\" where X is the correct letter."
)


class MMLUProDataset(HFDataset):
    """TIGER-Lab/MMLU-Pro: 10-way multiple-choice knowledge QA, single-turn,
    0-shot chain-of-thought, independent questions.

    Rides HFDataset's single-turn (``prompt``/``target``) rendering mode as-is
    -- not a new rendering mode. ``corpus()`` pre-renders each kept row into a
    ``prompt`` column (the full instruction-first question+options text, via
    ``render_prompt``) and a ``target`` column (the gold letter); the
    ``prompt``/``target`` PromptTemplates are trivial pass-throughs
    (``{{ prompt }}`` / ``{{ target }}``) so the inherited single-turn
    machinery (``_single_turn_item`` / ``items_at``) renders each column
    verbatim -- unchanged from HFDataset.

    Subsampled to exactly ``subsample_size`` rows via a seeded uniform draw
    over the full source-split indices (not stratified by category -- uniform
    sampling over a large pool such as the 12,032-row test split preserves
    category proportions in expectation while giving an exact row count with
    no per-category dependency)."""

    path: str = "TIGER-Lab/MMLU-Pro"
    source_split: str = "test"
    revision: str = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    #: MMLU-Pro's own question id, carried onto the pre-rendered row.
    id_column: str | None = "question_id"
    #: Exact post-subsample row count (a seeded uniform draw; see sample_indices).
    subsample_size: int = 2500
    #: Recorded default seed for the subsample draw -- change it only on purpose (it
    #: is folded into corpus_id() so provisioning re-caches when it changes).
    subsample_seed: int = 20240915
    # Pass-through single-turn templates: corpus() pre-renders the full
    # instruction-first question+options text and the gold letter into the
    # `prompt`/`target` row columns itself, so rendering here is just echoing
    # each column verbatim. Defaulted here (rather than via a _validate_mode
    # override) satisfies HFDataset's own single-turn-mode validator, which is
    # inherited unchanged. Field(default_factory=...) is used, not a bare
    # instance default, so the model is frozen (Poly/Spec) but no shared
    # mutable default is an issue either way -- consistent with this file's
    # other frozen-model construction.
    prompt: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ prompt }}"))
    target: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ target }}"))

    @staticmethod
    def sample_indices(n: int, k: int, seed: int) -> list[int]:
        """Deterministic seeded uniform draw of k indices from range(n), sorted.
        Factored out of corpus() so it is unit-testable without HF access."""
        return sorted(random.Random(seed).sample(range(n), k))

    @staticmethod
    def render_prompt(question: str, options: list[str]) -> str:
        """Instruction-first single-turn text so the fixed instruction is a
        shared prefix (prefix caching): "<INSTRUCTION>\\n\\nQuestion: <q>\\n
        Options:\\nA. <o0>\\nB. <o1>\\n...\\nAnswer:". Letters via
        "ABCDEFGHIJ"[i] for i in range(len(options)) (up to 10 options)."""
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        return f"{MMLU_PRO_INSTRUCTION}\n\nQuestion: {question}\nOptions:\n{opts}\nAnswer:"

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS, load_dataset
        raw = load_dataset(self.path, self.subset, revision=self.revision, split=self.source_split)
        idx = self.sample_indices(len(raw), self.subsample_size, self.subsample_seed)
        rows = [{"question_id": row["question_id"],
                 "prompt": self.render_prompt(row["question"], list(row["options"])),
                 "target": str(row["answer"])}
                for row in raw.select(idx)]
        return ArrowDS.from_list(rows)

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.path, self.revision)]

    def corpus_id(self) -> str:
        return super().corpus_id() + f"|n={self.subsample_size}|seed={self.subsample_seed}"


# ---- WildChat (real multi-turn user conversations) ------------------------

class WildChatDataset(HFDataset):
    """allenai/WildChat-1M (ODC-BY): real user conversations with a public
    ChatGPT-style service.

    A seventh rendering mode. Each item is one conversation rendered as its user
    turns only (capped at ``max_turns``): the artifact generates every assistant
    turn live, so the Driver stitches its own replies into the running history
    between turns -- same contract as `MultiIFDataset`, and the reason no
    ``metadata['messages']`` is pre-rendered here. Ground truth carries the
    corpus's own assistant replies; they are descriptive only (this dataset's
    tenant gates at the token level, which never reads them).

    ``corpus()`` streams the pinned revision (the release is 3.4 GB; a task needs
    ~1k conversations) and keeps the first ``subsample_size`` conversations that
    are tagged ``language``, are non-toxic when ``drop_toxic``, and whose user
    text fits ``max_input_tokens``. That last filter is a serving-envelope
    budget, not a length preference: the final turn's context is the whole
    conversation's user text plus up to ``max_turns - 1`` generated replies, so
    the tenant's output cap C bounds the worst-case prompt at
    ``max_input_tokens + (max_turns - 1) * C``.

    Conversations are also de-duplicated on their first user message: WildChat
    carries the same opening prompt many times over (popular copy-paste prompts,
    and the same prompt logged against several models), and each repeat is a
    byte-identical turn-0 request, i.e. a free response-cache hit for a served
    system."""

    path: str = "allenai/WildChat-1M"
    source_split: str = "train"
    #: WildChat's own per-conversation hash, carried onto the row.
    id_column: str | None = "conversation_hash"
    #: Conversation-level language tag to keep (the corpus is multilingual).
    language: str = "English"
    #: Turn cap per conversation (a turn = one user message + its live reply).
    max_turns: int = 6
    #: Exact number of conversations kept, in the pinned revision's stream order.
    subsample_size: int = 1000
    #: Per-conversation budget on the sum of its user-turn tokens (see docstring).
    max_input_tokens: int = 3072
    tokenizer_id: str = "Qwen/Qwen2.5-32B"
    #: The commit the corpus-time tokenizer is loaded at. A filter's cut depends on the
    #: tokenizer, so the commit decides which rows are kept.
    tokenizer_revision: str = "main"
    #: Skip conversations WildChat's moderation pass flagged toxic.
    drop_toxic: bool = True

    @model_validator(mode="after")
    def _validate_mode(self) -> "WildChatDataset":
        return self  # fixed rendering; no prompt/target or context_col fields

    def _turns_of(self, conversation) -> tuple[list[str], list[str]]:
        """Split one raw conversation into (user texts, scripted replies), capped at
        ``max_turns``. A turn opens on a non-empty user message and closes on the
        next assistant message (missing/empty replies render as ""), so a
        conversation that ends on a user message still yields that turn."""
        users: list[str] = []
        replies: list[str] = []
        for msg in conversation or []:
            role, content = msg.get("role"), str(msg.get("content") or "")
            if role == "user" and content.strip():
                if len(users) == self.max_turns:
                    break
                users.append(content)
                replies.append("")
            elif role == "assistant" and users and not replies[-1]:
                replies[-1] = content
        return users, replies

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        for idx, row in enumerate(json.loads(path.read_text())):
            req_id = self.row_id(row)
            users = list(row["user_turns"])
            replies = list(row["assistant_turns"])
            if not users:
                continue
            envs = tuple(Envelope(request_id=f"{req_id}.{k}",
                                  contents=[TextPayload(text=text)])
                         for k, text in enumerate(users))
            yield DatasetItem(
                request_id=req_id, turns=envs,
                ground_truth_turns=tuple(str(r) for r in replies[:len(users)]),
                metadata={"split": split.value, "index": idx, "n_turns": len(envs)})

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS, load_dataset
        tok = _tokenizer(self.tokenizer_id, self.tokenizer_revision)
        raw = load_dataset(self.path, revision=self.revision, split=self.source_split,
                           streaming=True)
        rows: list[dict] = []
        seen: set[str] = set()
        for ex in raw:
            if ex.get("language") != self.language:
                continue
            if self.drop_toxic and ex.get("toxic"):
                continue
            users, replies = self._turns_of(ex.get("conversation"))
            if not users:
                continue
            opening = hashlib.sha1(users[0].encode()).hexdigest()
            if opening in seen:
                continue
            n_tok = len(tok("\n".join(users), add_special_tokens=False)["input_ids"])
            if n_tok > self.max_input_tokens:
                continue
            seen.add(opening)
            rows.append({"conversation_hash": ex["conversation_hash"],
                         "user_turns": users, "assistant_turns": replies})
            if len(rows) == self.subsample_size:
                break
        return ArrowDS.from_list(rows)

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() counts tokens with the corpus-time tokenizer, which it loads from the hub.
        return [(self.path, self.revision), (self.tokenizer_id, self.tokenizer_revision)]

    def corpus_id(self) -> str:
        return (super().corpus_id()
                + f"|lang={self.language}|turns={self.max_turns}|n={self.subsample_size}"
                + f"|cap={self.max_input_tokens}|tok={self.tokenizer_id}@{self.tokenizer_revision}"
                + f"|toxic={'drop' if self.drop_toxic else 'keep'}")


# ---- Code completion (file-prefix continuation) ---------------------------

CODE_COMPLETION_INSTRUCTION = (
    "Continue this Python file. Output code only, no explanation.")


class CodeCompletionDataset(HFDataset):
    """Real source files rendered as code-completion requests: a truncated file
    prefix plus a fixed instruction, answered by a short continuation.

    Rides HFDataset's single-turn (``prompt``/``target``) rendering mode as-is --
    not a new rendering mode, same arrangement as `MMLUProDataset`: ``corpus()``
    pre-renders the ``prompt`` column (instruction-first, so the instruction is a
    shared prefix) and the ``target`` column (the file's own next
    ``target_tokens`` tokens), and the pass-through templates echo them verbatim.

    Default corpus: ``codeparrot/codeparrot-clean-valid`` -- Python files from
    GitHub carrying a per-file ``license`` column, filtered here to the
    ``licenses`` permissive allowlist. Kept files are cut at a seeded per-item
    prefix length drawn uniformly from ``[min_prefix_tokens, max_prefix_tokens]``
    (``truncation_seed`` + the row's stream position, so the same revision always
    yields the same windows), which is what puts real, differently-sized prefill
    on the wire instead of one fixed length."""

    path: str = "codeparrot/codeparrot-clean-valid"
    source_split: str = "train"
    #: The corpus's own per-file content hash, carried onto the pre-rendered row.
    id_column: str | None = "hash"
    #: Per-file license allowlist (the corpus's own `license` column).
    licenses: tuple[str, ...] = ("mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
                                 "isc")
    #: Exact number of files kept, in the pinned revision's stream order.
    subsample_size: int = 1000
    #: Seeded per-item prefix window (tokens).
    min_prefix_tokens: int = 1024
    max_prefix_tokens: int = 3072
    #: Tokens of the file's true continuation kept as ground truth.
    target_tokens: int = 64
    #: Recorded seed for the truncation draw -- change it only on purpose (it is folded
    #: into corpus_id() so provisioning re-caches when it changes).
    truncation_seed: int = 20260727
    tokenizer_id: str = "Qwen/Qwen2.5-32B"
    #: The commit the corpus-time tokenizer is loaded at. A filter's cut depends on the
    #: tokenizer, so the commit decides which rows are kept.
    tokenizer_revision: str = "main"
    instruction: str = CODE_COMPLETION_INSTRUCTION
    # Pass-through single-turn templates (see MMLUProDataset for the same pattern).
    prompt: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ prompt }}"))
    target: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ target }}"))

    @staticmethod
    def prefix_tokens(lo: int, hi: int, seed: int, idx: int) -> int:
        """The seeded prefix length for item ``idx``, uniform in ``[lo, hi]``.
        Factored out of corpus() so it is unit-testable without HF access."""
        return random.Random(f"{seed}:{idx}").randint(lo, hi)

    def render_prompt(self, prefix: str) -> str:
        """Instruction-first request text so the fixed instruction is a shared
        prefix (prefix caching): "<INSTRUCTION>\\n\\n<code prefix>"."""
        return f"{self.instruction}\n\n{prefix}"

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as ArrowDS, load_dataset
        tok = _tokenizer(self.tokenizer_id, self.tokenizer_revision)
        raw = load_dataset(self.path, revision=self.revision, split=self.source_split,
                           streaming=True)
        allowed = set(self.licenses)
        need = self.max_prefix_tokens + self.target_tokens
        # Only the first `need` tokens are ever used, so tokenize a bounded head of
        # the file (8 chars/token is a loose upper bound for source text) instead of
        # whole multi-hundred-KB files.
        head_chars = 8 * need
        rows: list[dict] = []
        for ex in raw:
            if ex.get("license") not in allowed or ex.get("autogenerated"):
                continue
            content = ex.get("content") or ""
            if len(content) < need:      # cheap floor: >= 1 char/token
                continue
            ids = tok(content[:head_chars], add_special_tokens=False)["input_ids"]
            if len(ids) < need:
                continue
            n_prefix = self.prefix_tokens(self.min_prefix_tokens, self.max_prefix_tokens,
                                          self.truncation_seed, len(rows))
            rows.append({
                "hash": ex["hash"],
                "prompt": self.render_prompt(tok.decode(ids[:n_prefix])),
                "target": tok.decode(ids[n_prefix:n_prefix + self.target_tokens]),
            })
            if len(rows) == self.subsample_size:
                break
        return ArrowDS.from_list(rows)

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() counts tokens with the corpus-time tokenizer, which it loads from the hub.
        return [(self.path, self.revision), (self.tokenizer_id, self.tokenizer_revision)]

    def corpus_id(self) -> str:
        return (super().corpus_id()
                + f"|lic={','.join(sorted(self.licenses))}|n={self.subsample_size}"
                + f"|win={self.min_prefix_tokens}-{self.max_prefix_tokens}"
                + f"|tgt={self.target_tokens}|seed={self.truncation_seed}"
                + f"|tok={self.tokenizer_id}@{self.tokenizer_revision}")


# ---- Audio (ASR) dataset -------------------------------------------------

class HFAudioDataset(ProvisionedRowsDataset):
    """A HuggingFace speech dataset for ASR: one audio clip per row plus its
    reference transcript. A new ASR task is config of this dataset, not a
    subclass.

    The crux of fitting the provisioning model: provisioning materializes the
    dev/test partition by ``corpus().select(idx).to_list()`` -> ``json.dumps``, so the
    corpus rows must be JSON-serializable. Raw HF audio columns are numpy arrays /
    encoded bytes and would not survive that. So ``corpus()`` PRE-ENCODES each clip
    to a base64 PCM16 WAV string column (decoded torch-free on the host via
    soundfile/soxr), alongside ``sample_rate`` and the transcript. ``items()`` then
    reads the mounted ``rows.json`` and wraps each base64 clip in an ``AudioPayload``
    envelope (modality "audio"); the transcript is the ground truth. Single-turn.

    Identity is ``path``/``subset``/``revision``/``source_split``; the dev/test
    split lives on ``SplitSpec`` (hls.spec.datasource), as for every Dataset."""

    path: str
    subset: str | None = None
    revision: str = "main"
    #: The HF provider split provisioning loads as the corpus to partition (the
    #: corpus ORIGIN, e.g. "test.clean"), NOT the dev/test split. Required.
    source_split: str
    audio_col: str = "audio"
    text_col: str = "text"
    target_sample_rate: int = 16_000
    #: Optional per-clip cap in seconds (long-form corpora). None = full clip.
    max_audio_sec: float | None = None
    #: Optional cap on how many source rows to encode in ``corpus()`` (the encode is
    #: per-clip and runs at provision time, so an unbounded corpus on a huge dataset is
    #: slow). None = encode the whole provider split. SplitSpec partitions whatever
    #: pool corpus() returns into the dev/test units.
    corpus_size: int | None = None

    def corpus(self) -> "ArrowDataset":
        from datasets import Audio, load_dataset
        from hls.core.audio import decode_audio_bytes, encode_wav_b64

        ds = load_dataset(self.path, self.subset, revision=self.revision,
                          split=self.source_split)
        if self.corpus_size is not None:
            ds = ds.select(range(min(self.corpus_size, len(ds))))
        # decode=False keeps audio decoding off torchcodec/torch: we get the raw
        # encoded bytes and decode with soundfile (decode_audio_bytes), so the host
        # stays torch-free (the model only runs inside containers).
        ds = ds.cast_column(self.audio_col, Audio(decode=False))
        sr, cap = self.target_sample_rate, self.max_audio_sec
        audio_col, text_col = self.audio_col, self.text_col

        def _encode(row: dict) -> dict:
            clip = row[audio_col]                    # {"bytes": ..., "path": ...}
            # ``is not None`` (not truthiness): b"" is a real 0-length blob, not "missing";
            # falling through to open(None) would crash the whole partition map. Use a
            # context manager on the path branch so the descriptor doesn't leak per clip.
            blob = clip.get("bytes")
            if blob is None:
                with open(clip["path"], "rb") as fh:
                    blob = fh.read()
            arr, got_sr = decode_audio_bytes(blob, sr)
            text = str(row[text_col])
            if cap is not None:
                # Capping the AUDIO without capping the TRANSCRIPT breaks WER by
                # construction (a perfect transcription of the first `cap` seconds
                # scores ~0 against a full-episode reference). No word timestamps
                # exist here, so truncate the reference proportionally (words x
                # cap/duration), fixed at encode time, identical for every
                # contestant, and not gameable.
                full_dur = len(arr) / float(got_sr or 1)
                if full_dur > cap:
                    arr = arr[: int(cap * got_sr)]
                    words = text.split()
                    text = " ".join(words[: max(1, round(len(words) * cap / full_dur))])
            return {"audio_b64": encode_wav_b64(arr, got_sr),
                    "sample_rate": got_sr, "text": text}

        return self._indexed(ds.map(_encode, remove_columns=ds.column_names))

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.path, self.revision)]

    def corpus_id(self) -> str:
        # Every knob that changes the ENCODED rows must be in the identity, or a
        # knob change silently reuses a stale provisioned unit (cache hit on rows
        # encoded under the old knobs -- e.g. a max_audio_sec change serving
        # old-length clips).
        # "audio:2" = encode-schema version (the cap also truncates the transcript
        # proportionally); bump when _encode's row derivation changes shape/meaning.
        return (f"{self.path}:{self.subset}@{self.revision}|src={self.source_split}"
                f"|audio:2|sr={self.target_sample_rate}|cap={self.max_audio_sec}"
                f"|n={self.corpus_size}|cols={self.audio_col},{self.text_col}")

    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem:
        req_id = self.row_id(row)
        # duration_sec is deliberately left None: nothing reads it off an ASR *input*
        # clip, and filling it here would base64-decode every (multi-MB) clip on the
        # eagerly-materialized items() hot path.
        env = Envelope(request_id=req_id, contents=[AudioPayload(
            audio_b64=row["audio_b64"],
            sample_rate=int(row.get("sample_rate", self.target_sample_rate)))])
        return DatasetItem(request_id=req_id, turns=(env,),
                           ground_truth_turns=(str(row["text"]),),
                           metadata={"split": split.value, "index": idx})


#: The hub repo ``FleursS2TTDataset`` reads.
FLEURS_REPO = "google/fleurs"


class FleursS2TTDataset(ProvisionedRowsDataset):
    """FLEURS speech-to-text translation pairs: speech in ``src_lang``, reference
    translation in ``tgt_lang`` -- the S2TT setup the Qwen3-Omni report evaluates
    (FLEURS, BLEU). FLEURS carries no translations per row; it is n-way parallel
    (the same FLoRes sentence ``id`` exists across language configs), so ``corpus()``
    INNER-JOINS the source config's audio with the target config's
    ``raw_transcription`` (natural-cased -- the right BLEU reference) on ``id``.

    Row schema matches ``HFAudioDataset``'s (``audio_b64``/``sample_rate``/``text``
    with text = the TARGET-language translation), so this dataset drops into
    ``MixedModalityDataset``'s audio arm unchanged -- pair it with an
    ``audio_instruction`` that asks for translation, and score the arm with
    ``sentence_bleu`` (``MixedModalityMatchMetric(audio_measure="bleu")``)."""

    src_lang: str = "fr_fr"                  # FLEURS config supplying the AUDIO
    tgt_lang: str = "en_us"                  # FLEURS config supplying the reference text
    #: The hub revision both language configs are read at.
    revision: str = "main"
    source_split: str = "test"
    target_sample_rate: int = 16_000         # FLEURS is natively 16 kHz
    corpus_size: int | None = None           # cap on JOINED pairs encoded at provision

    def corpus(self) -> "ArrowDataset":
        from datasets import Audio, Dataset as HFDS, load_dataset
        from hls.core.audio import decode_audio_bytes, encode_wav_b64

        src = load_dataset(FLEURS_REPO, self.src_lang, split=self.source_split, revision=self.revision)
        tgt = load_dataset(FLEURS_REPO, self.tgt_lang, split=self.source_split, revision=self.revision)
        src = src.cast_column("audio", Audio(decode=False))
        # The target side contributes TEXT only -- drop its audio column before any
        # row access, or datasets tries to decode it (torchcodec/torch, absent on the
        # torch-free host).
        tgt = tgt.select_columns(["id", "raw_transcription"])
        # A FLoRes sentence id can appear more than once in a config (multiple
        # recordings); keep the first target text per id, and join source rows on it.
        tgt_text: dict[int, str] = {}
        for row in tgt:
            tgt_text.setdefault(int(row["id"]), str(row["raw_transcription"]))
        out_rows: list[dict] = []
        for row in src:
            translation = tgt_text.get(int(row["id"]))
            if translation is None:
                continue                     # inner join: no target-side sentence
            clip = row["audio"]
            blob = clip.get("bytes")
            if blob is None:
                with open(clip["path"], "rb") as fh:
                    blob = fh.read()
            arr, got_sr = decode_audio_bytes(blob, self.target_sample_rate)
            out_rows.append({"audio_b64": encode_wav_b64(arr, got_sr),
                             "sample_rate": got_sr, "text": translation})
            if self.corpus_size is not None and len(out_rows) >= self.corpus_size:
                break
        return self._indexed(HFDS.from_list(out_rows))

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(FLEURS_REPO, self.revision)]

    def corpus_id(self) -> str:
        # "s2tt:1" = encode-schema version (see HFAudioDataset.corpus_id).
        return (f"{FLEURS_REPO}:{self.src_lang}->{self.tgt_lang}@{self.source_split}"
                f"|rev={self.revision}"
                f"|s2tt:1|sr={self.target_sample_rate}|n={self.corpus_size}")

    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem:
        req_id = self.row_id(row)
        env = Envelope(request_id=req_id, contents=[
            AudioPayload(audio_b64=row["audio_b64"],
                         sample_rate=int(row.get("sample_rate", self.target_sample_rate))),
            TextPayload(text="Translate the speech into English; "
                             "output only the translation.")])
        return DatasetItem(request_id=req_id, turns=(env,),
                           ground_truth_turns=(str(row["text"]),),
                           metadata={"split": split.value, "index": idx})


# ---- Visual-QA (image) dataset -------------------------------------------

class HFVQADataset(ProvisionedRowsDataset):
    """A HuggingFace visual-QA dataset for omni input: one image per row plus a
    question and a reference answer. Like HFAudioDataset, ``corpus()`` pre-encodes
    each image to a base64 string column (raw PNG/JPEG bytes, no PIL on the host) so the
    rows survive JSON materialization; ``items()`` wraps each into an ImagePayload +
    question TextPayload envelope, and the answer column is the ground truth. Single-turn,
    image-input."""

    path: str
    subset: str | None = None
    revision: str = "main"
    source_split: str
    question_col: str = "question"
    answer_col: str = "answer"
    image_col: str = "image"
    question_template: str = "{question}"
    #: When the answer column is a dict (e.g. {"text": [...]}), pull this key first.
    answers_text_key: str | None = None
    corpus_size: int | None = None

    def _answer(self, raw) -> str:
        if self.answers_text_key is not None and isinstance(raw, dict):
            raw = raw.get(self.answers_text_key, "")
        if isinstance(raw, (list, tuple)):
            return str(raw[0]) if raw else ""
        return str(raw)

    def corpus(self) -> "ArrowDataset":
        import base64
        from datasets import Image as HFImage, load_dataset
        from hls.core.image import sniff_mime

        ds = load_dataset(self.path, self.subset, revision=self.revision,
                          split=self.source_split)
        if self.corpus_size is not None:
            ds = ds.select(range(min(self.corpus_size, len(ds))))
        ds = ds.cast_column(self.image_col, HFImage(decode=False))   # raw bytes, no PIL
        image_col, question_col, answer_col = self.image_col, self.question_col, self.answer_col
        answer = self._answer

        def _encode(row: dict) -> dict:
            img = row[image_col]                       # {"bytes": ..., "path": ...}
            # ``is not None`` (not truthiness) + context manager: b"" is a real
            # 0-length blob, and the path branch must not leak a descriptor
            # (mirrors HFAudioDataset._encode).
            raw = img.get("bytes")
            if raw is None:
                with open(img["path"], "rb") as fh:
                    raw = fh.read()
            return {"image_b64": base64.b64encode(raw).decode("ascii"),
                    "mime_type": sniff_mime(raw),
                    "question": str(row[question_col]),
                    "answer": answer(row[answer_col])}

        return self._indexed(ds.map(_encode, remove_columns=ds.column_names))

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.path, self.revision)]

    def corpus_id(self) -> str:
        # Encode-affecting knobs belong in the identity (see HFAudioDataset.corpus_id).
        return (f"{self.path}:{self.subset}@{self.revision}|src={self.source_split}"
                f"|vqa|n={self.corpus_size}"
                f"|cols={self.image_col},{self.question_col},{self.answer_col}")

    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem:
        req_id = self.row_id(row)
        question = self.question_template.format(question=row["question"])
        env = Envelope(request_id=req_id, contents=[
            ImagePayload(image_b64=row["image_b64"],
                         mime_type=row.get("mime_type", "image/png")),
            TextPayload(text=question)])
        return DatasetItem(
            request_id=req_id, turns=(env,), ground_truth_turns=(str(row["answer"]),),
            metadata={"split": split.value, "index": idx, "input_modality": "image"})


# ---- Mixed-modality (omni) dataset ---------------------------------------

class MixedModalityDataset(ProvisionedRowsDataset):
    """A composite that interleaves a ``vision`` source (HFVQADataset: image+question
    -> answer) and an ``audio`` source (HFAudioDataset: speech -> transcript) into ONE
    corpus -- the omni workload (multimodal-omni). It fits single-corpus provisioning by
    building a single merged Arrow corpus with a unified schema (one row carries either
    an image+prompt or an audio clip, tagged by ``modality``); SplitSpec then partitions
    that merged corpus into the dev/test units.

    Interleaving is deterministic (a Bresenham-style accumulator on ``mix_ratio`` = the
    fraction of image rows), capped to ``corpus_size`` rows. ``items()`` reads the mount
    and rebuilds the modality-appropriate payload per row; each carries
    ``metadata["input_modality"]`` so the omni metric scores it by the right measure."""

    vision: HFVQADataset
    #: Any audio-arm dataset whose corpus rows carry {audio_b64, sample_rate, text}:
    #: HFAudioDataset (ASR -- text = transcript) or FleursS2TTDataset (S2TT -- text =
    #: the reference translation).
    audio: HFAudioDataset | FleursS2TTDataset
    mix_ratio: float = 0.5
    corpus_size: int = 40
    #: Human name for the merged corpus in the dumped spec (and so in the recorded
    #: workload a calibration is keyed on). Nothing served reads it.
    label: str = "mixed-omni"
    #: Text part sent beside each audio clip (read-time; see items()).
    audio_instruction: str = "Transcribe the audio exactly; output only the transcript."

    def model_post_init(self, __context) -> None:
        if not 0.0 <= self.mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {self.mix_ratio}")

    def _interleave(self, v_rows: Iterable[dict], a_rows: Iterable[dict]) -> list:
        """Mix the two arms into merged rows, at most ``corpus_size`` of them.

        Both arms are read strictly in order, one row at a time, and only as far as
        the mix consumes them, so a caller can pass the memory-mapped Arrow corpora
        instead of lists. The merged rows reference the arm rows' strings rather than
        copying them, so what the streaming saves is the tail each arm never
        contributes -- everything the mix does use is resident either way."""
        v_it, a_it = iter(v_rows), iter(a_rows)
        v, a = next(v_it, None), next(a_it, None)
        merged: list = []
        acc = 0.0
        while len(merged) < self.corpus_size and (v is not None or a is not None):
            acc += self.mix_ratio
            want_vision = acc >= 1.0
            if want_vision:
                acc -= 1.0
            take_vision = (want_vision and v is not None) or a is None
            if take_vision and v is not None:
                merged.append({"modality": "image", "image_b64": v["image_b64"],
                               "mime_type": v["mime_type"], "audio_b64": "", "sample_rate": 0,
                               "prompt": v["question"], "gt": v["answer"]})
                v = next(v_it, None)
            elif a is not None:
                merged.append({"modality": "audio", "image_b64": "", "mime_type": "",
                               "audio_b64": a["audio_b64"], "sample_rate": int(a["sample_rate"]),
                               "prompt": "", "gt": a["text"]})
                a = next(a_it, None)
        return merged

    def corpus(self) -> "ArrowDataset":
        from datasets import Dataset as HFDS
        # Iterated, not .to_list()-ed: each arm's corpus() is memory-mapped Arrow, and
        # pulling a whole arm into Python costs the rows the mix never reaches. An arm
        # sized close to its share of corpus_size (multimodal_omni's are) barely
        # notices; one sized well above it pays for the difference.
        merged = self._interleave(self.vision.corpus(), self.audio.corpus())
        # The arms are stamped in their own corpora too, but those indices repeat
        # across arms; the merged corpus restamps so the id is unique here.
        return self._indexed(HFDS.from_list(merged))

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() is the two arms' corpora, so their sources are its sources.
        return declared_hub_revisions(self.vision, reads="corpus") + declared_hub_revisions(self.audio, reads="corpus")

    def corpus_id(self) -> str:
        # corpus_size caps the merged row count, so it shapes the encoded unit too.
        return (f"mixed[{self.vision.corpus_id()}|{self.audio.corpus_id()}]"
                f"@mix={self.mix_ratio}|n={self.corpus_size}")

    def _render_row(self, row: dict, idx: int, split: Split) -> DatasetItem:
        req_id = self.row_id(row)
        modality = row["modality"]
        if modality == "image":
            # Render the vision arm's question_template HERE (read time, like the
            # audio_instruction): the merged corpus stores the RAW question, so
            # without this the template's answer-shape instruction ("answer with
            # one word...") never reaches the model -- a correct-but-verbose
            # answer then scores ~0 word-F1 against a 1-2 word VQA reference.
            contents = [ImagePayload(image_b64=row["image_b64"],
                                     mime_type=row.get("mime_type", "image/png")),
                        TextPayload(text=self.vision.question_template.format(
                            question=str(row["prompt"])))]
        else:
            # The instruction rides beside the audio: a bare clip invites an omni
            # chat model to comment on the audio rather than transcribe it, which
            # scores a WER near 1. Rendered at read time -- presentation,
            # not encoded data -- so changing it does not re-provision the corpus.
            contents = [AudioPayload(audio_b64=row["audio_b64"],
                                     sample_rate=int(row["sample_rate"])),
                        TextPayload(text=self.audio_instruction)]
        env = Envelope(request_id=req_id, contents=contents)
        return DatasetItem(
            request_id=req_id, turns=(env,), ground_truth_turns=(str(row["gt"]),),
            metadata={"split": split.value, "index": idx, "input_modality": modality})


# ---- Fill-in-the-middle (FIM) ----------------------------------------------

#: A FIM prompt in prefix-suffix-middle order: the code on both sides of the cursor
#: between sentinels, with the model generating the middle after the last one. This is
#: the Qwen2.5-Coder spelling of the three sentinels, offered for tasks on that family.
#: Other families spell the same markers differently, so each FIM dataset takes its
#: template from the task rather than defaulting to this one.
FIM_PSM_PROMPT = "<|fim_prefix|>{{ prefix }}<|fim_suffix|>{{ suffix }}<|fim_middle|>"


def _fim_cuts(n_lines: int, max_steps: int) -> list[int]:
    """Line indices at which a middle of ``n_lines`` lines is cut into typing steps:
    up to ``max_steps`` evenly spaced boundaries, always including 0 (the original
    row) and never the end (a step with nothing left to complete). Empty for an
    empty middle: there are no lines to cut, so there is no step to yield."""
    if n_lines == 0:
        return []
    cuts = sorted({round(j * n_lines / max_steps) for j in range(max_steps)})
    return [c for c in cuts if c < n_lines]


class FimSessionDataset(HFDataset):
    """A FIM-shaped corpus (prefix, suffix, middle columns) served as typing sessions.

    Real tab-completion traffic re-sends almost the same prompt as the developer
    accepts a suggestion and types on. No released dataset carries such sessions, so
    each row is cut into up to ``max_steps`` steps at evenly spaced line boundaries of
    its middle: step ``j`` sends the prefix plus the lines typed so far, the suffix, and
    is graded on the rest of the middle. Step 0 is the original row. Each step
    pre-renders its own single user turn in ``metadata["messages"]`` so the driver
    sends only that step's prompt (``hls.spec.drivers.request_messages``), never a
    stitched history.

    ``prompt`` has no default: the sentinels belong to a checkpoint's vocabulary, and a
    template carrying another family's spelling renders markers the served model reads
    as ordinary text. The task states the one its checkpoint uses.

    ``max_steps`` and the template are rendering knobs: they change every prompt the
    reference sees (so they sit in the workload fingerprint through ``dump_spec``) but
    not which rows are provisioned, so they stay out of ``corpus_id``. Default corpus:
    Real-FIM-Eval (``gonglinyuan/real_fim_eval``), whose ``prompt`` column is the
    prefix; the source carries no row id, so a row's position identifies it.
    ``corpus()`` keeps the three code columns and that position only: the source's
    provenance columns (repository, ref, path, language, commit timestamp) feed no
    prompt, and the timestamp is a datetime the provisioner cannot write to
    ``rows.json``. Which three those are decides what lands on disk, so the column
    names are in ``corpus_id``."""

    path: str = "gonglinyuan/real_fim_eval"
    source_split: str = "test"
    #: Corpus columns holding the code before, after and at the cursor.
    prefix_col: str = "prompt"
    suffix_col: str = "suffix"
    middle_col: str = "canonical_solution"
    #: Most steps a row is cut into. A row with fewer middle lines yields fewer.
    max_steps: int = 8
    #: Rendered per step with ``prefix`` = prefix + typed lines and ``suffix``.
    prompt: PromptTemplate
    #: Rendered per step with ``middle`` = the lines not yet typed.
    target: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ middle }}"))

    @model_validator(mode="after")
    def _validate_max_steps(self) -> "FimSessionDataset":
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        return self

    def corpus(self) -> "ArrowDataset":
        raw = super().corpus()
        keep = {self.prefix_col, self.suffix_col, self.middle_col,
                self.id_column if self.id_column is not None else SOURCE_INDEX}
        return raw.remove_columns([c for c in raw.column_names if c not in keep])

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() reshapes the parent's corpus and reads nothing more.
        return super().hub_revisions()

    def corpus_id(self) -> str:
        # corpus() prunes to these three columns, so two configs naming different ones
        # produce different rows.json. corpus_id is the only dataset term in the
        # provisioning digest, so without them the second config mounts the first's rows.
        return (super().corpus_id()
                + f"|fim-cols={self.prefix_col},{self.suffix_col},{self.middle_col}")

    def _session_item(self, row: dict, req_id: str, idx: int, split_value: str) -> DatasetItem | None:
        """One corpus row as a typing session: a turn per line-boundary cut of its
        middle, each carrying its own pre-rendered prompt and the lines still to be
        typed as its ground truth. ``idx`` is the row's position in the mounted split
        and ``split_value`` names the split, both recorded on the item's metadata.
        A row whose middle has no lines has no cut to yield a step at, so this
        returns ``None`` and the caller skips it, mirroring
        ``HFDataset._multi_turn_item``."""
        prefix = str(row[self.prefix_col])
        suffix = str(row[self.suffix_col])
        lines = str(row[self.middle_col]).splitlines(keepends=True)
        cuts = _fim_cuts(len(lines), self.max_steps)
        if not cuts:
            return None
        envs: list[Envelope] = []
        gts: list[str] = []
        for j, cut in enumerate(cuts):
            text = self.prompt.render(prefix=prefix + "".join(lines[:cut]), suffix=suffix)
            envs.append(Envelope(request_id=f"{req_id}.{j}",
                                 contents=[TextPayload(text=text)],
                                 metadata={"messages": [{"role": "user", "content": text}]}))
            gts.append(self.target.render(middle="".join(lines[cut:])))
        return DatasetItem(request_id=req_id, turns=tuple(envs), ground_truth_turns=tuple(gts),
                           metadata={"split": split_value, "index": idx, "n_turns": len(envs)})

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        path = Path(mount_dir) / split.value / "rows.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{split.value} split not mounted at {path}; provisioning did not attach it "
                f"for this role (TEST is withheld from the agent)")
        for idx, row in enumerate(json.loads(path.read_text())):
            item = self._session_item(row, self.row_id(row), idx, split.value)
            if item is not None:
                yield item


class SafimDataset(HFDataset):
    """SAFIM (``gonglinyuan/safim``) rendered as single FIM requests.

    SAFIM is not FIM-shaped: its ``prompt`` column is a problem statement and the code
    lives in ``eval_prompt`` with a placeholder where the middle goes. ``corpus()``
    splits ``eval_prompt`` at that placeholder into ``prefix`` and ``suffix`` columns
    and drops ``eval_prompt`` and ``unit_tests`` (execution material this task never
    reads); the FIM template then renders the split columns. A row where the
    placeholder does not occur exactly once is refused, which is how a corpus revision
    that changed the format announces itself. ``corpus_id`` records the split so the
    provisioned rows are never confused with the unsplit corpus.

    ``prompt`` has no default, for the reason ``FimSessionDataset`` gives: the sentinel
    spelling is a property of the checkpoint, so the task states it."""

    path: str = "gonglinyuan/safim"
    source_split: str = "test"
    id_column: str | None = "task_id"
    #: Column holding the code with the placeholder, and the placeholder itself.
    eval_prompt_col: str = "eval_prompt"
    placeholder: str = "{{completion}}"
    #: Rendered per row with ``prefix`` and ``suffix``, the split ``eval_prompt``.
    prompt: PromptTemplate
    target: PromptTemplate | None = Field(
        default_factory=lambda: JinjaPromptTemplate(template="{{ ground_truth }}"))

    def split_eval_prompt(self, eval_prompt: str) -> tuple[str, str]:
        """``(prefix, suffix)`` around the placeholder; raises unless it occurs once."""
        parts = eval_prompt.split(self.placeholder)
        if len(parts) != 2:
            raise DatasetError(
                f"{self.path}:{self.subset}: expected the placeholder {self.placeholder!r} "
                f"exactly once in {self.eval_prompt_col}, found {len(parts) - 1} times")
        return parts[0], parts[1]

    def corpus(self) -> "ArrowDataset":
        raw = super().corpus()
        dropped = [c for c in (self.eval_prompt_col, "unit_tests") if c in raw.column_names]

        def _split(row: dict) -> dict:
            prefix, suffix = self.split_eval_prompt(str(row[self.eval_prompt_col]))
            return {"prefix": prefix, "suffix": suffix}

        return raw.map(_split, remove_columns=dropped)

    def hub_revisions(self) -> list[tuple[str, str]]:
        # corpus() reshapes the parent's corpus and reads nothing more.
        return super().hub_revisions()

    def corpus_id(self) -> str:
        # Both knobs decide what corpus() writes: eval_prompt_col is the column split
        # into prefix/suffix and then dropped, placeholder is where it splits.
        return (super().corpus_id()
                + f"|fim-split={self.eval_prompt_col}@{self.placeholder}")
