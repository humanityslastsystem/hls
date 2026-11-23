"""L0 -- DatasetItem.

`turns` is a 1-tuple for single-shot, N-tuple for multi-turn; `ground_truth_turns`
mirrors that shape. The ground-truth value is a generic, dataset-rendered target
-- no per-dataset GT class. The single-turn `.request`/`.ground_truth`
accessors are what verifiers read; they fail loud on multi-turn,
whose turns the driver stitches into one conversation (``hls.spec.drivers``).
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from hls.core.payloads import Envelope


class DatasetItem(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request_id: str
    turns: tuple[Envelope, ...]
    ground_truth_turns: tuple[Any, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def request(self) -> Envelope:
        if len(self.turns) != 1:
            raise ValueError(
                f"item {self.request_id!r} has {len(self.turns)} turns; single-turn "
                f".request accessor is undefined for multi-turn")
        return self.turns[0]

    @property
    def ground_truth(self) -> Any:
        if len(self.ground_truth_turns) != 1:
            raise ValueError(f"item {self.request_id!r} is multi-turn; .ground_truth undefined")
        return self.ground_truth_turns[0]
