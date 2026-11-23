"""Every provisionable dataset must be able to identify its own rows.

Provisioning assigns rows to dev/test by hashing each row's identity
(``partition_by_hash`` over ``Dataset.row_id``), so a dataset that emits a corpus but
no row id is not provisionable at all -- and nothing catches it until a real run
reaches the partition step, because rendering from already-written rows never calls
``row_id``. These tests close that gap for the datasets that build their own rows.
"""
import pytest

from hls.spec.datasets import (
    SOURCE_INDEX,
    FleursS2TTDataset,
    HFAudioDataset,
    HFVQADataset,
    MixedModalityDataset,
    ProvisionedRowsDataset,
)


def _every_provisioned_rows_dataset():
    """Concrete ProvisionedRowsDataset subclasses, so a dataset added later is covered
    without being listed here."""
    seen, out = set(), []
    stack = [ProvisionedRowsDataset]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            out.append(sub)
            stack.append(sub)
    return out


def test_every_built_rows_dataset_defines_row_id():
    # HFDataset and its subclasses read an id off the provider's row; the datasets that
    # build rows from scratch inherit the positional fallback. Either way the method
    # must resolve to something other than the base class's refusal.
    from hls.spec.datasets import Dataset

    for cls in _every_provisioned_rows_dataset():
        assert cls.row_id is not Dataset.row_id, (
            f"{cls.__name__} inherits Dataset.row_id, which raises: provisioning "
            f"cannot partition it")


@pytest.mark.parametrize("ds", [
    HFAudioDataset(path="x/y", source_split="test"),
    FleursS2TTDataset(),
    HFVQADataset(path="x/y", source_split="test"),
    MixedModalityDataset(vision=HFVQADataset(path="x/y", source_split="test"),
                         audio=FleursS2TTDataset()),
])
def test_row_id_reads_the_stamped_index(ds):
    assert ds.row_id({SOURCE_INDEX: 7, "text": "ignored"}) == "7"


def test_row_id_is_unique_across_a_stamped_corpus():
    # What the partition actually needs: distinct ids, so two rows cannot collide into
    # one side of the split.
    rows = [{SOURCE_INDEX: i} for i in range(50)]
    ds = HFAudioDataset(path="x/y", source_split="test")
    assert len({ds.row_id(r) for r in rows}) == 50
