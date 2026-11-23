"""Every dataset and tool bundle declares the hub sources it reads (``hub_revisions``)."""

from __future__ import annotations

import inspect

import pytest

from hls.prompt import JinjaPromptTemplate
from hls.spec import datasets as datasets_mod
from hls.spec.agent import ToolAssets
from hls.spec.datasets import (BFCLDataset, Dataset, FleursS2TTDataset, HFAudioDataset, HFDataset, HFVQADataset,
                               MixedModalityDataset, MRCRDataset)
from hls.spec.hub import UndeclaredHubSources, declared_hub_revisions, undeclared_by

SHA = "0123456789abcdef0123456789abcdef01234567"


def _reading_classes() -> list[type]:
    return [c for c in vars(datasets_mod).values()
            if inspect.isclass(c) and issubclass(c, Dataset) and "corpus" in vars(c) and c is not Dataset]


@pytest.mark.parametrize("cls", _reading_classes(), ids=lambda c: c.__name__)
def test_every_dataset_class_that_reads_a_corpus_declares_what_it_reads_beside_it(cls):
    assert undeclared_by(cls, reads="corpus") is None


def _hf(**kw) -> HFDataset:
    return HFDataset(path="org/d", revision=SHA, source_split="test", id_column="q",
                     prompt=JinjaPromptTemplate(template="{{q}}"), target=JinjaPromptTemplate(template="{{q}}"), **kw)


def test_a_dataset_declares_the_repo_and_revision_it_loads():
    assert declared_hub_revisions(_hf(), reads="corpus") == [("org/d", SHA)]
    assert declared_hub_revisions(HFAudioDataset(path="org/a", source_split="test"), reads="corpus") == [
        ("org/a", "main")]


def test_a_corpus_time_tokenizer_is_a_hub_source_too():
    mrcr = MRCRDataset(path="openai/mrcr", revision=SHA, tokenizer_id="org/tok", tokenizer_revision=OTHER)
    assert declared_hub_revisions(mrcr, reads="corpus") == [("openai/mrcr", SHA), ("org/tok", OTHER)]


def test_bfcl_declares_its_repo():
    bfcl = BFCLDataset(repo="gorilla-llm/Berkeley-Function-Calling-Leaderboard", revision=SHA, categories=["simple"])
    assert declared_hub_revisions(bfcl, reads="corpus") == [("gorilla-llm/Berkeley-Function-Calling-Leaderboard", SHA)]


OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def test_a_source_declared_at_main_is_read_at_main():
    """A dataset's revision field is the one place its commit is named: a source left at
    ``main`` reads ``main``, which the provision stage refuses, whatever repo it names."""
    asr = HFAudioDataset(path="hf-audio/open-asr-leaderboard", source_split="test")
    assert declared_hub_revisions(asr, reads="corpus") == [("hf-audio/open-asr-leaderboard", "main")]
    assert "@main|" in asr.corpus_id()
    fleurs = FleursS2TTDataset()
    assert declared_hub_revisions(fleurs, reads="corpus") == [("google/fleurs", "main")]
    assert "rev=main" in fleurs.corpus_id()


def test_fleurs_reads_and_names_its_revision_field(monkeypatch):
    import datasets

    loaded = []

    def load_dataset(path, name, *, split, revision=None):
        loaded.append((path, name, split, revision))
        if len(loaded) == 2:
            raise RuntimeError("both loaded")

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    fleurs = FleursS2TTDataset(revision=SHA)
    with pytest.raises(RuntimeError, match="both loaded"):
        fleurs.corpus()
    assert loaded == [("google/fleurs", "fr_fr", "test", SHA), ("google/fleurs", "en_us", "test", SHA)]
    assert declared_hub_revisions(fleurs, reads="corpus") == [("google/fleurs", SHA)]
    assert f"rev={SHA}" in fleurs.corpus_id()
    assert fleurs.model_dump()["revision"] == SHA


def _template() -> JinjaPromptTemplate:
    return JinjaPromptTemplate(template="{{q}}")


def _deepresearch(name: str, **kw):
    from tasks.deepresearch import datasets as dr

    return getattr(dr, name)(path="org/a", source_split="test", **kw)


#: One dataset of every class with a ``corpus()`` of its own, or one inherited through
#: ``super()``, each declared on repo ``org/a`` at the revision it is given.
LOADERS = {
    "hf": lambda **kw: HFDataset(path="org/a", source_split="test", id_column="q", prompt=_template(),
                                 target=_template(), **kw),
    "column-projected": lambda **kw: datasets_mod.ColumnProjectedHFDataset(
        path="org/a", source_split="test", keep_columns=["q"], prompt=_template(), target=_template(), **kw),
    "multi-if": lambda **kw: datasets_mod.MultiIFDataset(path="org/a", source_split="test", **kw),
    "chat-rag": lambda **kw: datasets_mod.ChatRAGDataset(path="org/a", source_split="test", **kw),
    "mrcr": lambda **kw: MRCRDataset(path="org/a", **kw),
    "mmlu-pro": lambda **kw: datasets_mod.MMLUProDataset(path="org/a", **kw),
    "wildchat": lambda **kw: datasets_mod.WildChatDataset(path="org/a", **kw),
    "code-completion": lambda **kw: datasets_mod.CodeCompletionDataset(path="org/a", **kw),
    "fim-session": lambda **kw: datasets_mod.FimSessionDataset(path="org/a", prompt=_template(), **kw),
    "safim": lambda **kw: datasets_mod.SafimDataset(path="org/a", prompt=_template(), **kw),
    "audio": lambda **kw: HFAudioDataset(path="org/a", source_split="test", **kw),
    "vqa": lambda **kw: HFVQADataset(path="org/a", source_split="test", **kw),
    "deepresearch-qa": lambda **kw: _deepresearch("DeepResearchQADataset", **kw),
    "browsecomp-qa": lambda **kw: _deepresearch("BrowseCompQADataset", **kw),
}


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_a_loader_reads_its_revision_field(monkeypatch, name):
    import datasets

    loaded = []

    def load_dataset(path, *args, revision=None, **kw):
        loaded.append((path, revision))
        raise RuntimeError("loaded")

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    # The token-filtering corpora load their tokenizer first; none is fetched here.
    monkeypatch.setattr(datasets_mod, "_tokenizer", lambda *args: object())
    for revision in (SHA, OTHER):
        with pytest.raises(RuntimeError, match="loaded"):
            LOADERS[name](revision=revision).corpus()
    assert loaded == [("org/a", SHA), ("org/a", OTHER)]


def test_bfcl_downloads_its_files_at_its_revision_field(monkeypatch):
    import huggingface_hub

    fetched = []

    def hf_hub_download(repo, filename, *, repo_type=None, revision=None, **kw):
        fetched.append((repo, filename, revision))
        if len(fetched) == 2:
            raise RuntimeError("both fetched")
        return "/nonexistent"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    with pytest.raises(RuntimeError, match="both fetched"):
        BFCLDataset(repo="org/a", revision=SHA, categories=["simple"]).corpus()
    assert fetched == [("org/a", "BFCL_v3_simple.json", SHA), ("org/a", "possible_answer/BFCL_v3_simple.json", SHA)]


def test_a_composite_declares_what_each_arm_declares():
    mixed = MixedModalityDataset(vision=HFVQADataset(path="org/v", revision=SHA, source_split="val"),
                                 audio=FleursS2TTDataset(revision=OTHER))
    assert declared_hub_revisions(mixed, reads="corpus") == [("org/v", SHA), ("google/fleurs", OTHER)]


def test_a_class_that_reads_without_declaring_is_undeclared():
    class Reads(HFDataset):
        def corpus(self):
            return None

    class Declares(Reads):
        def hub_revisions(self):
            return [("org/d", SHA), ("org/extra", SHA)]

    assert "Reads defines corpus() without declaring" in undeclared_by(Reads, reads="corpus")
    assert undeclared_by(Declares, reads="corpus") is None
    with pytest.raises(UndeclaredHubSources, match="Reads defines corpus"):
        declared_hub_revisions(Reads(path="org/d", revision=SHA, source_split="t",
                                     prompt=JinjaPromptTemplate(template="x"),
                                     target=JinjaPromptTemplate(template="x")), reads="corpus")


def test_a_dataset_that_is_not_provisionable_declares_nothing():
    class InMemory(Dataset):
        def items_at(self, mount_dir, split):
            return iter(())

    with pytest.raises(UndeclaredHubSources, match="InMemory declares no hub sources"):
        declared_hub_revisions(InMemory(), reads="corpus")


def test_a_composite_arm_that_reads_without_declaring_is_undeclared():
    class Vision(HFVQADataset):
        def corpus(self):
            return None

    mixed = MixedModalityDataset(vision=Vision(path="org/v", revision=SHA, source_split="val"),
                                 audio=HFAudioDataset(path="org/a", revision=SHA, source_split="t"))
    with pytest.raises(UndeclaredHubSources, match="Vision defines corpus"):
        declared_hub_revisions(mixed, reads="corpus")


def test_a_tool_bundle_must_declare_its_hub_sources_to_exist():
    class NoDeclaration(ToolAssets):
        name = "x"

        def identity(self):
            return "x"

        def build(self, variant, **kw):
            pass

    with pytest.raises(TypeError, match="hub_revisions"):
        NoDeclaration()
