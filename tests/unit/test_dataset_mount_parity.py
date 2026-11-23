"""Every provisionable dataset must read the mount directory it is handed.

The driver mounts each tenant's rows at ``/data/tenants/<id>`` and passes that path
to ``items_at(mount_dir, split)`` (see drivers.py). A dataset that reads any other
location serves one tenant's rows to another. Each test below writes a decoy row at
one path and the expected row at another, then asserts the items come from the path
actually passed in."""
import json

from hls.spec.datasets import (
    SOURCE_INDEX, FleursS2TTDataset, HFAudioDataset, HFVQADataset,
    MixedModalityDataset, Split,
)


def _write(mount_dir, split, rows):
    d = mount_dir / split.value
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.json").write_text(json.dumps(rows))


# Row builders stamp the ground truth with a marker naming which mount the rows came
# from: "TENANT" is the directory passed to items_at, "DECOY" is a sibling that must
# never be read. Each row carries SOURCE_INDEX, as a provisioned row does.
def _audio_row(gt):
    return {SOURCE_INDEX: 0, "audio_b64": "AA", "sample_rate": 16000, "text": gt}

def _fleurs_row(gt):
    return {SOURCE_INDEX: 0, "audio_b64": "AA", "sample_rate": 16000, "text": gt}

def _vqa_row(gt):
    return {SOURCE_INDEX: 0, "question": "q", "image_b64": "II",
            "mime_type": "image/png", "answer": gt}

def _mixed_row(gt):
    return {SOURCE_INDEX: 0, "modality": "image", "image_b64": "II",
            "mime_type": "image/png", "audio_b64": "", "sample_rate": 0,
            "prompt": "q", "gt": gt}


def _assert_items_at_honors_mount(ds, row_builder, tmp_path):
    """``items_at(tenant_dir)`` must read tenant_dir and nowhere else."""
    decoy_dir = tmp_path / "decoy"
    tenant_dir = tmp_path / "tenant"
    _write(decoy_dir, Split.TEST, [row_builder("DECOY")])
    _write(tenant_dir, Split.TEST, [row_builder("TENANT")])

    items = list(ds.items_at(tenant_dir, Split.TEST))
    assert len(items) == 1
    assert items[0].ground_truth_turns[0] == "TENANT"


def test_audio_items_at_honors_mount_dir(tmp_path):   # control: already correct
    ds = HFAudioDataset(path="x", source_split="test")
    _assert_items_at_honors_mount(ds, _audio_row, tmp_path)


def test_fleurs_items_at_honors_mount_dir(tmp_path):
    ds = FleursS2TTDataset()
    _assert_items_at_honors_mount(ds, _fleurs_row, tmp_path)


def test_vqa_items_at_honors_mount_dir(tmp_path):
    ds = HFVQADataset(path="x", source_split="test")
    _assert_items_at_honors_mount(ds, _vqa_row, tmp_path)


def test_mixed_items_at_honors_mount_dir(tmp_path):
    ds = MixedModalityDataset(vision=HFVQADataset(path="x", source_split="test"),
                              audio=FleursS2TTDataset())
    _assert_items_at_honors_mount(ds, _mixed_row, tmp_path)
