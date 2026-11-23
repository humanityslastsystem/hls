"""Test-only in-memory multi-turn dataset. Production tasks use HFDataset
(multi-turn mode); this lives in test infra so src/ carries no debug-only
Dataset implementations."""
from __future__ import annotations

from typing import Any, ClassVar, Iterator

from pydantic import model_validator

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.spec.datasets import Dataset, Split


class InlineMultiTurnDataset(Dataset):
    """A self-contained, in-memory multi-turn conversational dataset (no HF
    download). Each conversation is ``(conv_id, [(user_text, ground_truth), ...])``
    and becomes one DatasetItem with N turns. Each turn's Envelope carries that
    turn's user message; per-turn ground truth is the expected answer substring.
    Designed so later turns depend on earlier context, which only resolves
    correctly when the driver stitches history.

    ``Dataset`` is a frozen pydantic ``Poly``; ``conversations`` is the one
    field. ``name`` is a per-class ClassVar (the Dataset identity), so a
    ``name=`` kwarg is accepted and dropped by a before-validator."""

    version: ClassVar[str] = "1.0"

    conversations: list[tuple[str, list[tuple[str, str]]]]

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "name" in data:
            data = {k: v for k, v in data.items() if k != "name"}
        return data

    @model_validator(mode="after")
    def _require_nonempty(self) -> "InlineMultiTurnDataset":
        if not self.conversations:
            raise ValueError("InlineMultiTurnDataset needs at least one conversation")
        return self

    @property
    def _n(self) -> int:
        # Describe the dataset by its longest conversation (turn handling is
        # per-item, derived from each item's actual turn count by the driver).
        return max(len(turns) for _, turns in self.conversations)

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        # In-memory dataset -- mount_dir is unused, conversations are already resident.
        for conv_id, turns in self.conversations:
            envs = tuple(
                Envelope(request_id=f"{conv_id}.{i}",
                         contents=[TextPayload(text=user_text)])
                for i, (user_text, _gt) in enumerate(turns))
            gts = tuple(gt for _user, gt in turns)
            yield DatasetItem(
                request_id=conv_id, turns=envs, ground_truth_turns=gts,
                metadata={"split": split.value, "n_turns": len(turns)})

    def corpus(self):
        # Inline corpus via datasets.from_list -- no HF download; provisionable.
        from datasets import Dataset as ArrowDataset
        return ArrowDataset.from_list(
            [{"conv_id": cid, "n_turns": len(turns)} for cid, turns in self.conversations])

    def corpus_id(self) -> str:
        return f"inline-mt:{self.version}@{len(self.conversations)}"
