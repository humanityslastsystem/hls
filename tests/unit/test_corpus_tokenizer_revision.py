"""A corpus filtered by a tokenizer names the tokenizer's commit in its own spec
(``tokenizer_revision``): two commits can cut the same source rows differently, so the commit is
part of the workload identity, of the corpus identity (and the rows key built from it), and of
the hub sources the provision stage checks are pinned."""
from __future__ import annotations

import importlib

import pytest

from hls.core.poly import dump_spec
from hls.provision.rows import unit_descriptions
from hls.spec import datasets
from hls.spec.datasets import CodeCompletionDataset, MRCRDataset, WildChatDataset
from hls.spec.hub import declared_hub_revisions
from hls.verify.calibration import workload_payload

SHA = "1" * 40
OTHER = "f" * 40

MAKERS = [
    lambda **kw: MRCRDataset(path="openai/mrcr", revision="a" * 40, **kw),
    lambda **kw: WildChatDataset(revision="a" * 40, **kw),
    lambda **kw: CodeCompletionDataset(revision="a" * 40, **kw),
]
IDS = ["mrcr", "wildchat", "code_completion"]


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_the_tokenizer_commit_is_part_of_the_corpus_identity(make):
    ds = make(tokenizer_revision=SHA)
    assert f"|tok={ds.tokenizer_id}@{SHA}" in ds.corpus_id()
    assert f"|tok={ds.tokenizer_id}@{OTHER}" in make(tokenizer_revision=OTHER).corpus_id()


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_the_tokenizer_commit_is_in_the_dumped_spec(make):
    """The workload payload holds the dataset's dump, so the fingerprint covers the commit."""
    assert dump_spec(make(tokenizer_revision=SHA))["tokenizer_revision"] == SHA


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_the_tokenizer_is_a_hub_source_at_its_declared_revision(make):
    ds = make(tokenizer_revision=SHA)
    assert declared_hub_revisions(ds, reads="corpus")[-1] == (ds.tokenizer_id, SHA)
    # Undeclared, it is read at main, which the provision stage refuses.
    assert declared_hub_revisions(make(), reads="corpus")[-1] == (ds.tokenizer_id, "main")


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_the_corpus_cuts_its_rows_with_the_tokenizer_at_its_declared_revision(make, monkeypatch):
    """The commit the identity names is the one the filter loads, at the corpus call site."""
    import datasets as hf_datasets

    asked = []

    def tokenizer(tokenizer_id, revision):
        asked.append((tokenizer_id, revision))
        return object()

    def load_dataset(*args, **kwargs):
        raise RuntimeError("loaded")

    monkeypatch.setattr(datasets, "_tokenizer", tokenizer)
    monkeypatch.setattr(hf_datasets, "load_dataset", load_dataset)
    ds = make(tokenizer_id="org/tok", tokenizer_revision=SHA)
    with pytest.raises(RuntimeError, match="loaded"):
        ds.corpus()
    assert asked == [("org/tok", SHA)]


def test_the_tokenizer_is_loaded_at_the_revision_asked_once_per_commit(monkeypatch):
    loads = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id, **kwargs):
            loads.append((tokenizer_id, kwargs))
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr(datasets, "_TOKENIZER_CACHE", {})
    first = datasets._tokenizer("org/tok", SHA)
    assert datasets._tokenizer("org/tok", SHA) is first
    assert datasets._tokenizer("org/tok", OTHER) is not first
    assert loads == [("org/tok", {"revision": SHA}), ("org/tok", {"revision": OTHER})]


def test_multitenant_slo_declares_the_tokenizer_commit_its_published_rows_were_cut_with():
    """``Qwen/Qwen2.5-32B`` at 1818d358, the commit that last touched the repo's tokenizer files,
    cut the published rows of the chat and long_session tenants."""
    task = importlib.import_module("tasks.multitenant_slo.task").TASK
    filtered = [unit["data_source"].dataset for unit in unit_descriptions(task)
                if isinstance(unit["data_source"].dataset, (MRCRDataset, WildChatDataset))]
    assert len(filtered) == 2
    for ds in filtered:
        assert (ds.tokenizer_id, ds.tokenizer_revision) == (
            "Qwen/Qwen2.5-32B", "1818d35814b8319459f4bd55ed1ac8709630f003")
    assert repr(workload_payload(task)).count("'tokenizer_revision': '1818d35814b8319459f4bd55ed1ac8709630f003'") == 2
