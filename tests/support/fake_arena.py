"""A minimal, self-contained arena module for the driver-entry test.

The remote driver entry (``hls._modal_driver_entry``) re-imports an arena module
by id and runs its REAL driver against the artifact. This module stands in for a
real arena (``tasks.sample_task.task``) without HF/network or a GPU stack: it exposes a
valid ``TASK`` (TaskSpec) whose dataset is in-memory and whose verifier
(``MetricVerifier+LetterMatchMetric``) reads the dataset's own ground truth -- no
reference oracle / engine needed -- so the entry's re-import + drive path can be
exercised against a fake HTTP artifact.
"""

from __future__ import annotations

from collections.abc import Iterator

from hls.core.dataitem import DatasetItem
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.verify import LetterMatchMetric, MetricVerifier
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Dataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, Uncapped, Workload)

N_ITEMS = 2


class _InMemoryDataset(Dataset):
    """Yields a fixed handful of single-turn items with letter ground truths --
    no HuggingFace download, so the drive loop runs offline. A Dataset family
    member so it can ride inside a DataSource (the dataset the Runner stashes onto
    the engine); items_at() is self-contained and ignores the mount."""

    def items_at(self, mount_dir, split: Split) -> Iterator[DatasetItem]:
        for i in range(N_ITEMS):
            env = Envelope(request_id=str(i), contents=[TextPayload(text=f"q{i}")])
            yield DatasetItem(
                request_id=str(i), turns=(env,), ground_truth_turns=("A",), metadata={"index": i}
            )

    def corpus(self):
        # Inline corpus via datasets.from_list -- no HF download; provisionable.
        from datasets import Dataset as ArrowDataset
        return ArrowDataset.from_list([{"q": f"q{i}", "a": "A"} for i in range(N_ITEMS)])

    def corpus_id(self) -> str:
        return f"inline:in_memory@{N_ITEMS}"

    def row_id(self, row: dict) -> str:
        return row["q"]


MODEL = ModelSpec(base_checkpoint="fake/model", arch="TX", chat_format="llama3-instruct")
HARDWARE = Hardware(
    accelerator=AcceleratorSpec(kind="CPU", count=1, memory_gb_per_device=1),
    cpu_count=1,
    memory_gb=1,
    storage_gb=1,
)
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http",
)
DATASET = _InMemoryDataset()
DATA_SOURCE = DataSource(
    dataset=DATASET,
    split=SplitSpec(dev=Count(value=N_ITEMS), test=Count(value=N_ITEMS)))
VERIFIERS = [MetricVerifier(metric=LetterMatchMetric(), min_value=0.5)]
WORKLOAD = Workload(
    arrival=AllAtOnce(),
    inter_turn_delay=NoDelay(),
    output_length_cap=Uncapped(),
    slo=NoSLO(),
)
SCORER = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))

TASK = TaskSpec(
    id="fake-arena",
    model=MODEL,
    hardware=HARDWARE,
    contract=CONTRACT,
    workload=WORKLOAD,
    verifiers=VERIFIERS,
    scorer=SCORER,
    data_source=DATA_SOURCE,
)

