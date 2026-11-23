"""ASR, TTS and omni read every hub source at a pinned commit.

Each source declares that commit in its own ``revision`` field, so the dumped spec, and with it
the task's workload fingerprint, names it. Rows rebuilt at it equal the published rows byte for byte."""

from __future__ import annotations

import importlib
import json

import pytest

from hls.measured import BASELINE_CREDIT_FILE, defaults_path
from hls.taskref import TaskRef
from hls.core.poly import dump_spec
from hls.provision import role
from hls.provision.model import is_revision
from hls.provision.rows import unit_descriptions
from hls.spec.datasets import MixedModalityDataset
from hls.spec.hub import declared_hub_revisions
from hls.verify.calibration import workload_fingerprint, workload_payload

#: The commit of each source whose rows the published task uses.
ASR_LIBRISPEECH = ("hf-audio/open-asr-leaderboard", "b6bdcd0beb34f8975dc659796176d88f43aff502")
TTS_LIBRITTS = ("mythicinfinity/libritts", "35f26bdafa9883b5524f17931216af78632db8eb")
OMNI_VQAV2 = ("lmms-lab/VQAv2", "32665d35052eb4a6d4414851c3c829a72754915a")
OMNI_FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")

PINS = {
    "tasks.multimodal_asr.task": [ASR_LIBRISPEECH],
    "tasks.multimodal_tts.task": [TTS_LIBRITTS],
    "tasks.multimodal_omni.task": [OMNI_VQAV2, OMNI_FLEURS],
}


def _task(module: str):
    return importlib.import_module(module).TASK


def _datasets(module: str) -> list:
    """Every dataset the task's provisioning reads: each tenant's dataset and dev dataset."""
    out = []
    for unit in unit_descriptions(_task(module)):
        out.append(unit["data_source"].dataset)
        if unit["dev_dataset"] is not None:
            out.append(unit["dev_dataset"])
    return out


def _sources(ds) -> list:
    """The datasets that read the hub themselves: a mixed dataset's two arms, else *ds*."""
    return [ds.vision, ds.audio] if isinstance(ds, MixedModalityDataset) else [ds]


@pytest.mark.parametrize("module", sorted(PINS))
def test_every_hub_source_of_the_task_is_read_at_the_published_rows_commit(module):
    declared = [source for ds in _datasets(module) for source in declared_hub_revisions(ds, reads="corpus")]
    assert declared == PINS[module]
    assert all(is_revision(revision) for _, revision in declared)


@pytest.mark.parametrize("module", sorted(PINS))
def test_each_source_declares_its_commit_in_its_own_revision_field(module):
    """The commit is in the spec, so the workload payload the fingerprint hashes names it."""
    sources = [source for ds in _datasets(module) for source in _sources(ds)]
    assert [dump_spec(s)["revision"] for s in sources] == [revision for _, revision in PINS[module]]
    payload = repr(workload_payload(_task(module)))
    assert all(f"'revision': '{revision}'" in payload for _, revision in PINS[module])


@pytest.mark.parametrize("module", sorted(PINS))
def test_the_provision_role_accepts_the_tasks_rows_as_pinned(module):
    role._check_rows_are_pinned(unit_descriptions(_task(module)))


@pytest.mark.parametrize("module", sorted(PINS))
def test_the_rows_key_names_each_commit(module):
    """A rows key stands for the bytes under it only when the corpus identity it digests names
    the commit every source is read at."""
    corpus_ids = [unit["corpus_id"] for unit in unit_descriptions(_task(module))]
    for _, revision in PINS[module]:
        assert any(revision in cid for cid in corpus_ids), (revision, corpus_ids)


#: Each task's workload fingerprint, which its committed baseline credit is stamped with.
FINGERPRINTS = {
    "tasks.multimodal_asr.task": "b8d4f8d47b4857527ad42d880366a6a03d1ef8496b18d1e4711fd778591e1a5a",
    "tasks.multimodal_tts.task": "c521957c6543c87e6e5813c5131eca5c54e39c1899d23934528da99d3bd8d98e",
    "tasks.multimodal_omni.task": "182af13dbe50d0e4191b4db965f0e23e91e7764378222db6c1d6003dae647adb",
}


@pytest.mark.parametrize("module", sorted(FINGERPRINTS))
def test_the_default_baseline_credit_carries_the_tasks_fingerprint(module):
    assert workload_fingerprint(_task(module)) == FINGERPRINTS[module]
    credit = defaults_path(TaskRef(module=module), BASELINE_CREDIT_FILE)
    assert json.loads(credit.read_text())["campaign"]["workload_fingerprint"] == FINGERPRINTS[module]
