"""A request id must name the row, not the row's position in the partition.

The stored reference answers (``hls.verify.answers``) are filed under the request id,
which is justified by the id being the corpus row's identity. A renderer that builds the
id out of ``idx`` -- the row's offset in the mounted partition -- breaks that: change the
split sizes, the corpus size or the removal list and id ``N`` is a different row, so the
banked answers stop matching anything asked. The digest keeps that from serving a wrong
answer, but every banked answer then misses and the oracle recomputes them, which is the
cost the stored answers exist to avoid.

Each test below mounts the same rows in two different orders and asserts the ids
follow the rows.
"""
import json

import pytest

from hls.spec.datasets import (
    SOURCE_INDEX, FleursS2TTDataset, HFAudioDataset, HFVQADataset,
    MixedModalityDataset, Split,
)


def _audio_row(i, gt):
    return {SOURCE_INDEX: i, "audio_b64": "AA", "sample_rate": 16000, "text": gt}


def _vqa_row(i, gt):
    return {SOURCE_INDEX: i, "question": "q", "image_b64": "II",
            "mime_type": "image/png", "answer": gt}


def _mixed_row(i, gt):
    return {SOURCE_INDEX: i, "modality": "image", "image_b64": "II",
            "mime_type": "image/png", "audio_b64": "", "sample_rate": 0,
            "prompt": "q", "gt": gt}


#: (dataset, row builder). The four ProvisionedRowsDataset subclasses that build their
#: own rows; HFDataset reads an id off the provider and is covered by its own tests.
_CASES = [
    (HFAudioDataset(path="x/y", source_split="test"), _audio_row),
    (FleursS2TTDataset(), _audio_row),
    (HFVQADataset(path="x/y", source_split="test"), _vqa_row),
    (MixedModalityDataset(vision=HFVQADataset(path="x/y", source_split="test"),
                          audio=FleursS2TTDataset()), _mixed_row),
]


def _ids_by_ground_truth(ds, mount_dir, rows):
    """Render ``rows`` from a fresh mount and return {ground truth: request id}."""
    d = mount_dir / Split.TEST.value
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.json").write_text(json.dumps(rows))
    return {it.ground_truth_turns[0]: it.request_id
            for it in ds.items_at(mount_dir, Split.TEST)}


@pytest.mark.parametrize("ds,row", _CASES, ids=lambda x: getattr(x, "kind", ""))
def test_request_id_follows_the_row_not_its_position(ds, row, tmp_path):
    # The same two corpus rows, landing in opposite order in two partitions -- what a
    # changed split size or a removal does to a real mount.
    forward = _ids_by_ground_truth(ds, tmp_path / "a", [row(7, "seven"), row(9, "nine")])
    reverse = _ids_by_ground_truth(ds, tmp_path / "b", [row(9, "nine"), row(7, "seven")])
    assert forward == reverse


@pytest.mark.parametrize("ds,row", _CASES, ids=lambda x: getattr(x, "kind", ""))
def test_request_id_is_the_rows_own_id(ds, row, tmp_path):
    ids = _ids_by_ground_truth(ds, tmp_path, [row(7, "seven"), row(9, "nine")])
    assert ids == {"seven": "7", "nine": "9"}


@pytest.mark.parametrize("ds,row", _CASES, ids=lambda x: getattr(x, "kind", ""))
def test_a_mount_without_the_stamp_is_named_rather_than_a_keyerror(ds, row, tmp_path):
    from hls.core.faults import DatasetError

    stripped = row(0, "x")
    del stripped[SOURCE_INDEX]
    with pytest.raises(DatasetError, match=SOURCE_INDEX):
        _ids_by_ground_truth(ds, tmp_path, [stripped])
