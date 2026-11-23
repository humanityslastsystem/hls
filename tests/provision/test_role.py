from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.provision import role
from hls.provision.hf import SourceFile, git_blob_sha1
from hls.provision.model import Resolved
from hls.provision.rows import dev_digest
from hls.prompt import JinjaPromptTemplate
from hls.spec.datasets import (FleursS2TTDataset, HFAudioDataset, HFDataset, HFVQADataset, MixedModalityDataset,
                               MRCRDataset)
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.store.cache import read_digests
from hls.store.manifest import ReportedOutput, read_outputs_report
from tests.provision import fake_hub
from tests.provision.tasks import (DATA_SHA, SHA, corpus, dataset, no_data_task, task,
                                   two_tenant_task)
from tests.store.entries import entry_reads, recording_reads

OTHER_SHA = "f" * 40


class _Api:
    """Stands in for huggingface_hub.HfApi: every branch resolves to ``SHA``."""

    calls: list = []

    def model_info(self, repo, *, revision=None, token=None, files_metadata=False):
        _Api.calls.append((repo, revision, token, files_metadata))
        return SimpleNamespace(sha=SHA, siblings=[])


@pytest.fixture
def hub(monkeypatch):
    _Api.calls = []
    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    return _Api


@pytest.fixture
def fake_corpus(monkeypatch):
    monkeypatch.setattr(role, "_corpus_for", lambda ds: corpus())


def test_the_rows_key_does_not_name_the_model(hub):
    """The rows are read off the datasets alone: no tokenizer, chat template or other file of
    the model goes into them, so the same rows under another model keep their key."""
    assert role.resolve(task(revision=SHA)).rows_key == role.resolve(task(revision=OTHER_SHA)).rows_key
    other = task()
    other = other.model_copy(update={"model": other.model.model_copy(update={"base_checkpoint": "org/other"})})
    assert role.resolve(other).rows_key == role.resolve(task()).rows_key


def test_a_dev_reference_change_moves_the_rows_key(hub):
    """A dataset's dev reference files are written into the rows entry, so changing one, or
    declaring none, gives the rows another key."""
    keys = [role.resolve(task(dev_reference=files)).rows_key
            for files in ({"prompt_prefix.txt": "A"}, {"prompt_prefix.txt": "B"}, {})]
    assert all(keys) and len(set(keys)) == 3


def test_resolve_pins_the_model_and_keys_the_rows(hub):
    resolved = role.resolve(task())
    assert resolved.weights.model_dump() == {"repo": "org/model", "revision": SHA}
    assert hub.calls == []
    assert resolved.tenants == ["t0"] and resolved.tools is None and resolved.rows_key.startswith("rows-")


@pytest.mark.parametrize("revision", ["main", "r1", "v1.0", SHA.upper(), SHA[:12]])
def test_weights_at_a_branch_are_refused_without_asking_the_hub(hub, revision):
    """As a dataset at a branch is: the weights key stands for fixed bytes only when the model
    is declared at a commit, and the task's identity names that commit."""
    with pytest.raises(role.Refused, match=r"model: org/model: revision '" + re.escape(revision)
                                           + "' is not a commit id, so the weights could change"):
        role.resolve(task(revision=revision))
    assert hub.calls == []


def test_a_bad_model_repo_is_refused(hub):
    t = task()
    t = t.model_copy(update={"model": t.model.model_copy(update={"base_checkpoint": "../etc"})})
    with pytest.raises(role.Refused, match="model: .*not a hub repo id"):
        role.resolve(t)
    assert hub.calls == []


def test_the_model_is_resolved_at_the_commit_it_declares(hub):
    resolved = role.resolve(task(revision=OTHER_SHA))
    assert resolved.weights.revision == OTHER_SHA and hub.calls == []


def test_a_task_without_data_resolves_its_model_alone(hub):
    resolved = role.resolve(no_data_task())
    assert resolved.weights.model_dump() == {"repo": "org/model", "revision": SHA}
    assert resolved.rows_key is None and resolved.tenants == [] and resolved.tools is None
    assert hub.calls == []


def test_tool_assets_without_a_data_source_are_refused(hub):
    from tests.unit.test_tool_assets import FakeAssets
    from tests.unit.test_trajectory_driver import _agent

    t = no_data_task()
    t.agent = replace(_agent(), assets=FakeAssets())
    with pytest.raises(ValueError, match="ToolAssets but no tenant carries a data source"):
        role.resolve(t)


@pytest.mark.parametrize("revision", ["main", "r1", "v1.0", SHA.upper(), SHA[:12]])
def test_rows_are_keyed_only_for_a_dataset_pinned_to_a_commit(hub, revision):
    with pytest.raises(ValueError, match=r"tenant t0: dataset .*: revision '" + re.escape(revision)
                                         + "' is not a commit id"):
        role.resolve(task(data_revision=revision))


def test_the_dev_dataset_is_pinned_too(hub):
    with pytest.raises(ValueError, match=r"tenant t0: dev_dataset .*: revision 'main' is not a commit id"):
        role.resolve(task(dev_dataset=dataset(revision="main")))
    role.resolve(task(dev_dataset=dataset(revision=OTHER_SHA)))


def test_a_corpus_time_tokenizer_must_be_pinned_to_a_commit(hub):
    def source(**tokenizer):
        mrcr = MRCRDataset(path="openai/mrcr", revision=DATA_SHA, **tokenizer)
        return DataSource(dataset=mrcr, split=SplitSpec(dev=Count(value=5), test=Count(value=4)))

    with pytest.raises(role.Refused, match="Qwen/Qwen2.5-32B: revision 'main' is not a commit id"):
        role.resolve(task(source=source()))
    with pytest.raises(role.Refused, match="org/tok: revision 'main' is not a commit id"):
        role.resolve(task(source=source(tokenizer_id="org/tok", tokenizer_revision="main")))
    role.resolve(task(source=source(tokenizer_revision=OTHER_SHA)))


def test_a_dataset_that_declares_no_hub_sources_is_refused(hub):
    # The rows key stands for fixed bytes only when every hub read behind it is pinned, and a
    # dataset that does not say what it reads cannot be shown to be.
    source = SimpleNamespace(dataset=SimpleNamespace(corpus_id=lambda: "ds"),
                             split=SimpleNamespace(dev=Count(value=1), test=Count(value=1)))
    t = no_data_task()
    t.effective_tenants = lambda: [SimpleNamespace(id="t0", data_source=source)]
    with pytest.raises(role.Refused, match=r"tenant t0: dataset ds: SimpleNamespace .*\(hub_revisions\)"):
        role.resolve(t)


class _ReadsMore(HFDataset):
    """An HFDataset whose corpus() reads a second repo it does not declare."""

    def corpus(self):
        raise AssertionError("resolve reads no corpus")


def test_a_dataset_class_that_reads_the_hub_without_declaring_it_is_refused(hub):
    undeclared = _ReadsMore(path="corp", revision=DATA_SHA, source_split="test", id_column="q",
                            prompt=JinjaPromptTemplate(template="{{q}}"), target=JinjaPromptTemplate(template="{{q}}"))
    source = DataSource(dataset=undeclared, split=SplitSpec(dev=Count(value=5), test=Count(value=4)))
    with pytest.raises(role.Refused, match="_ReadsMore defines corpus.. without declaring the hub sources it reads"):
        role.resolve(task(source=source))


def _vqa(revision: str = DATA_SHA) -> HFVQADataset:
    return HFVQADataset(path="org/vqa", revision=revision, source_split="validation")


def _asr(revision: str = DATA_SHA) -> HFAudioDataset:
    return HFAudioDataset(path="org/asr", revision=revision, source_split="test")


@pytest.mark.parametrize("vision, audio, refused", [
    (_vqa(), _asr(), None),
    (_vqa("main"), _asr(), "org/vqa: revision 'main' is not a commit id"),
    (_vqa(), _asr("main"), "org/asr: revision 'main' is not a commit id"),
    (_vqa(), FleursS2TTDataset(), "google/fleurs: revision 'main' is not a commit id"),
    (_vqa(), FleursS2TTDataset(revision=OTHER_SHA), None),
], ids=["pinned", "vision", "audio", "fleurs", "fleurs-pinned"])
def test_a_composite_dataset_is_pinned_only_when_every_arm_is(hub, vision, audio, refused):
    mixed = MixedModalityDataset(vision=vision, audio=audio, corpus_size=8)
    source = DataSource(dataset=mixed, split=SplitSpec(dev=Count(value=2), test=Count(value=2)))
    if refused is None:
        assert role.resolve(task(source=source)).tenants == ["t0"]
        return
    with pytest.raises(role.Refused, match=f"tenant t0: dataset mixed.*: {re.escape(refused)}"):
        role.resolve(task(source=source))


@pytest.mark.parametrize("sources, refused", [
    ([], None),
    ([("org/corpus", DATA_SHA), ("org/encoder", SHA)], None),
    ([("org/corpus", DATA_SHA), ("org/encoder", "main")], "org/encoder: revision 'main' is not a commit id"),
    ([("../corpus", DATA_SHA)], "not a hub repo id: '../corpus'"),
], ids=["none", "pinned", "branch", "bad-repo"])
def test_tool_assets_must_pin_every_hub_source_they_build_from(hub, monkeypatch, sources, refused):
    from tests.unit.test_tool_assets import FakeAssets

    monkeypatch.setattr(FakeAssets, "hub_revisions", lambda self: sources)
    t = _tools_task()
    if refused is None:
        assert role.resolve(t).tools.name == "fake"
        return
    with pytest.raises(role.Refused, match=f"tool assets fake: {re.escape(refused)}"):
        role.resolve(t)


def test_a_tool_bundle_class_that_builds_from_the_hub_without_declaring_it_is_refused(hub):
    from tests.unit.test_tool_assets import FakeAssets

    class BuildsMore(FakeAssets):
        def build(self, variant, **kw):
            raise AssertionError("resolve builds nothing")

    t = task(agent=replace(_tools_task().agent, assets=BuildsMore()))
    with pytest.raises(role.Refused, match="tool assets fake: BuildsMore defines build.. without declaring"):
        role.resolve(t)


@pytest.mark.parametrize("module", ["tasks.sample_task.task", "tests.support.token_level_task"])
def test_the_sample_tasks_pin_their_datasets(hub, module):
    resolved = role.resolve(role.load_task(module))
    assert resolved.rows_key is not None and resolved.tenants == ["t0"]


#: The sha256 of every file the rows role writes for ``task()`` over the 20-row test corpus. A
#: real task's rows are held against its published entry's digests when they are rebuilt.
ROWS_DIGESTS = {
    "ids.json": "d883c67d47d0acb81066b19d9de771358bb6d6a895899bdcf7f8260a23531784",
    "tenants/t0/dev/prompt_prefix.txt": "5c62e091b8c0565f1bafad0dad5934276143ae2ccef7a5381e8ada5b1a8d26d2",
    "tenants/t0/dev/rows.json": "4d4b880e2249d2eae3afb147322a81a44449c0c8b1b3381267cc667d4420d168",
    "tenants/t0/test/rows.json": "a6a9b42b5a64eb68f2ec9e77045459360db96ae1c759ad8db19b11dac84bdd15",
}


def test_the_rows_entry_holds_the_recorded_bytes(tmp_path, fake_corpus):
    out = tmp_path / "entry"
    out.mkdir()
    role.write_rows_entry(task(), out)
    assert _hashes(out) == ROWS_DIGESTS
    ids = json.loads((out / role.IDS_FILE).read_text())
    dev = [str(r["q"]) for r in json.loads((out / "tenants" / "t0" / "dev" / "rows.json").read_text())]
    test = [str(r["q"]) for r in json.loads((out / "tenants" / "t0" / "test" / "rows.json").read_text())]
    assert ids == {"dev": dev, "test": test}
    # Row 8 falls in the dev partition and is on the removal list, so it is screened out.
    assert "8" not in dev and len(dev) == 4
    assert (out / "tenants" / "t0" / "dev" / "prompt_prefix.txt").read_text() == "P"


def test_two_tenants_over_one_corpus_name_each_row_once(tmp_path, fake_corpus):
    single = tmp_path / "single"
    single.mkdir()
    role.write_rows_entry(task(), single)
    both = tmp_path / "both"
    both.mkdir()
    role.write_rows_entry(two_tenant_task(), both)
    ids = json.loads((both / role.IDS_FILE).read_text())
    assert ids == json.loads((single / role.IDS_FILE).read_text())
    assert len(ids["dev"]) == len(set(ids["dev"])) and len(ids["test"]) == len(set(ids["test"]))
    for tenant in ("t0", "t1"):                 # each tenant still has its own rows
        assert (both / "tenants" / tenant / "dev" / "rows.json").is_file()


def test_rows_need_an_empty_directory_and_a_data_source(tmp_path, fake_corpus):
    out = tmp_path / "entry"
    out.mkdir()
    (out / "leftover").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        role.write_rows_entry(task(), out)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no tenant"):
        role.write_rows_entry(no_data_task(), empty)


def test_weights_land_without_the_hub_clients_metadata_and_with_the_hubs_listing(tmp_path, monkeypatch):
    seen = {}

    def download(*, repo_id, revision, local_dir, token):
        seen.update(repo_id=repo_id, revision=revision, token=token)
        root = tmp_path / "entry"
        assert local_dir == str(root)
        (root / "config.json").write_text("{}")
        (root / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (root / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("m")

    listing = {"config.json": SourceFile(size=2, sha256=None, git_sha1="c" * 40)}
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(role, "source_files", lambda repo, revision, *, token: listing)
    out, side = tmp_path / "entry", tmp_path / "source.json"
    out.mkdir()
    role.fetch_weights("org/model", SHA, out, side, token="tok")
    assert seen == {"repo_id": "org/model", "revision": SHA, "token": "tok"}
    assert sorted(p.name for p in out.iterdir()) == ["config.json"]
    assert json.loads(side.read_text()) == {"config.json": {"size": 2, "sha256": None, "git_sha1": "c" * 40}}


def _outputs(tmp_path) -> list[str]:
    """The option that names where a role reports the digests of the stage outputs it writes."""
    return ["--outputs-out", str(tmp_path / "outputs.json")]


def _resolve(module: str, out, tmp_path) -> list[str]:
    """The resolve command of the role program for *module*, writing *out*."""
    return ["resolve", "--task-module", module, "--out", str(out), *_outputs(tmp_path)]


def _report(tmp_path) -> list[str]:
    """The option that names a fill role's digest report, outside its entry."""
    return ["--digests-out", str(tmp_path / "digests.json")]


def _hashes(root) -> dict[str, str]:
    """Each file under *root* by path with the sha256 of its bytes, as the test reads them."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_weights_role_reports_every_file_it_wrote_and_the_blob_id_of_each_git_file(tmp_path, monkeypatch):
    """A fill role hashes what it wrote, synced, and reports it, so the stage's own process reads
    none of it; for a file the hub keeps in git it also reports the git blob id of those bytes."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    shard = b"\x01" * 100

    def download(*, repo_id, revision, local_dir, token):
        (Path(local_dir) / "config.json").write_text("{}")
        (Path(local_dir) / "model.safetensors").write_bytes(shard)

    listing = {"config.json": SourceFile(size=2, sha256=None, git_sha1="c" * 40),
               "model.safetensors": SourceFile(size=100, sha256=hashlib.sha256(shard).hexdigest(),
                                               git_sha1=None)}
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr(role, "source_files", lambda repo, revision, *, token: listing)
    out, side, blobs = tmp_path / "entry", tmp_path / "source.json", tmp_path / "blobs.json"
    out.mkdir()
    with recording_reads() as reads:
        assert role.main(["weights", "--repo", "org/model", "--revision", SHA, "--out", str(out),
                          "--source-out", str(side), *_outputs(tmp_path), *_report(tmp_path), "--blobs-out",
                          str(blobs)]) == 0
    # The shard is read once, to hash it; the small git file once more, for its blob id.
    shard, config = str(out / "model.safetensors"), str(out / "config.json")
    assert sorted(entry_reads(reads, [out])) == sorted([shard, config, config])
    assert read_digests(tmp_path / "digests.json") == _hashes(out)
    assert json.loads(blobs.read_text()) == {"config.json": git_blob_sha1(out / "config.json")}
    # The hub's list is a stage output, reported by its name.
    listed = hashlib.sha256(side.read_bytes()).hexdigest()
    assert read_outputs_report(tmp_path / "outputs.json") == {
        "source.json": ReportedOutput(sha256=listed, size=side.stat().st_size)}


@pytest.mark.parametrize("command", ["rows", "tools"])
def test_the_rows_and_tools_roles_report_every_file_they_wrote(tmp_path, monkeypatch, fake_corpus, command):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    t = _tools_task()
    rows = tmp_path / "rows"
    rows.mkdir()
    role.write_rows_entry(t, rows)
    out = tmp_path / "entry"
    out.mkdir()
    args = ["--task-module", "x", "--out", str(out), *_report(tmp_path)]
    if command == "tools":
        args += ["--variant", "full", "--ids", str(rows / role.IDS_FILE)]
    monkeypatch.setattr(role, "load_task", lambda module: t)
    with recording_reads() as reads:
        assert role.main([command, *args]) == 0
    assert read_digests(tmp_path / "digests.json") == _hashes(out) != {}
    # Each file it wrote is read back once, to hash it.
    assert sorted(entry_reads(reads, [out])) == sorted(str(out / p) for p in _hashes(out))


@pytest.mark.parametrize("command", ["rows", "tools"])
def test_a_fill_roles_report_never_lies_inside_its_entry(tmp_path, monkeypatch, fake_corpus, command):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    out = tmp_path / "entry"
    out.mkdir()
    args = ["--task-module", "tests.provision.tasks", "--out", str(out), "--digests-out", str(out / "d.json")]
    if command == "tools":
        args += ["--variant", "full", "--ids", str(tmp_path / "ids.json")]
    with pytest.raises(ValueError, match="inside the entry"):
        role.main([command, *args])
    assert list(out.iterdir()) == []


def test_weights_refuse_a_branch_name_and_a_listing_inside_the_entry(tmp_path):
    out = tmp_path / "entry"
    out.mkdir()
    with pytest.raises(ValueError, match="inside the entry"):
        role.fetch_weights("org/model", SHA, out, out / "source.json", token=False)
    with pytest.raises(ValueError, match="not a hub commit id"):
        role.fetch_weights("org/model", "main", out, tmp_path / "source.json", token=False)


def test_weights_fill_only_an_empty_directory(tmp_path, monkeypatch):
    downloads = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: downloads.append(kw))
    out = tmp_path / "entry"
    out.mkdir()
    (out / "leftover").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        role.fetch_weights("org/model", SHA, out, tmp_path / "source.json", token=False)
    assert downloads == [] and [p.name for p in out.iterdir()] == ["leftover"]


def _tools_task():
    from tests.unit.test_tool_assets import FakeAssets
    from tests.unit.test_trajectory_driver import _agent

    return task(agent=replace(_agent(), assets=FakeAssets()))


def test_tools_build_from_the_rows_entrys_ids(tmp_path, fake_corpus):
    t = _tools_task()
    rows = tmp_path / "rows"
    rows.mkdir()
    role.write_rows_entry(t, rows)
    ids = json.loads((rows / role.IDS_FILE).read_text())
    built = {}
    for variant in ("full", "dev"):
        out = tmp_path / variant
        out.mkdir()
        role.build_tools(t, variant, ids, out, token=False)
        built[variant] = json.loads((out / "manifest.json").read_text())
        assert [p.name for p in out.iterdir()] == ["manifest.json"]     # no work directory is left behind
    assert built["full"]["variant"] == "full" and built["dev"]["dev"] == len(ids["dev"])
    # The digest the stage keys the dev bundle with, derived from ids.json.
    assert dev_digest(ids["dev"], ids["test"]) == "5cf65582e5d2"


def test_a_builder_works_inside_the_staging_directory_and_only_there(tmp_path, monkeypatch):
    """The real builder writes ``<out_dir>.partial`` beside its out_dir and renames it into place."""
    seen = []

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        seen.append(out_dir)
        partial = out_dir.with_name(out_dir.name + ".partial")
        partial.mkdir(parents=True)
        (partial / "bundle.bin").write_text(variant)
        os.replace(partial, out_dir)

    from tests.unit.test_tool_assets import FakeAssets

    monkeypatch.setattr(FakeAssets, "build", build)
    parent = tmp_path / "staging-area"
    out = parent / "entry"
    out.mkdir(parents=True)
    role.build_tools(_tools_task(), "full", {"dev": ["1"], "test": ["2"]}, out, token=False)
    assert [p.name for p in parent.iterdir()] == ["entry"]              # nothing beside the staging directory
    assert [p.name for p in out.iterdir()] == ["bundle.bin"]
    assert (out / "bundle.bin").read_text() == "full"
    assert seen[0].is_relative_to(out) and seen[0] != out               # the builder never got out itself


def test_a_granted_token_reaches_the_builder_and_an_ungranted_one_does_not(tmp_path, monkeypatch):
    from tests.unit.test_tool_assets import FakeAssets

    got = []

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        got.append(hf_token)
        out_dir.mkdir(parents=True)

    monkeypatch.setattr(FakeAssets, "build", build)
    for token in ("hf_granted", False):
        out = tmp_path / f"entry-{token}"
        out.mkdir()
        role.build_tools(_tools_task(), "full", {"dev": [], "test": []}, out, token=token)
    assert got == ["hf_granted", None]


def test_a_failed_build_leaves_the_staging_directory_empty(tmp_path, monkeypatch):
    from tests.unit.test_tool_assets import FakeAssets

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        out_dir.mkdir(parents=True)
        (out_dir / "half.bin").write_text("x")
        raise RuntimeError("builder died")

    monkeypatch.setattr(FakeAssets, "build", build)
    out = tmp_path / "entry"
    out.mkdir()
    with pytest.raises(RuntimeError, match="builder died"):
        role.build_tools(_tools_task(), "dev", {"dev": [], "test": []}, out, token=False)
    assert list(out.iterdir()) == []


def test_a_builder_that_leaves_a_deep_tree_beside_its_bundle_still_leaves_only_the_bundle(tmp_path, monkeypatch):
    from tests.unit.test_tool_assets import FakeAssets

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        p = out_dir.parent / "scratch"
        for _ in range(sys.getrecursionlimit() + 100):
            p = p / "a"
            p.mkdir(parents=True)
        out_dir.mkdir()
        (out_dir / "bundle.bin").write_text(variant)

    monkeypatch.setattr(FakeAssets, "build", build)
    out = tmp_path / "entry"
    out.mkdir()
    role.build_tools(_tools_task(), "full", {"dev": [], "test": []}, out, token=False)
    assert [p.name for p in out.iterdir()] == ["bundle.bin"]


def test_a_builder_that_writes_no_bundle_is_an_error(tmp_path, monkeypatch):
    from tests.unit.test_tool_assets import FakeAssets

    monkeypatch.setattr(FakeAssets, "build", lambda self, variant, **kw: None)
    out = tmp_path / "entry"
    out.mkdir()
    with pytest.raises(ValueError, match="wrote no bundle"):
        role.build_tools(_tools_task(), "full", {"dev": [], "test": []}, out, token=False)
    assert list(out.iterdir()) == []


def test_tools_need_assets_an_empty_directory_and_a_known_variant(tmp_path, fake_corpus):
    ids = {"dev": ["1"], "test": ["2"]}
    out = tmp_path / "entry"
    out.mkdir()
    with pytest.raises(ValueError, match="no tool assets"):
        role.build_tools(task(), "full", ids, out, token=False)
    with pytest.raises(ValueError, match="unknown bundle variant"):
        role.build_tools(_tools_task(), "partial", ids, out, token=False)
    (out / "leftover").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        role.build_tools(_tools_task(), "full", ids, out, token=False)
    assert [p.name for p in out.iterdir()] == ["leftover"]


def _weights(tmp_path, n: int) -> list[str]:
    """The weights command of the role program for ``org/model`` at ``SHA``, into a fresh entry."""
    out = tmp_path / f"entry{n}"
    out.mkdir()
    return ["weights", "--repo", "org/model", "--revision", SHA, "--out", str(out), "--source-out",
            str(tmp_path / f"source{n}.json"), *_outputs(tmp_path), "--blobs-out", str(tmp_path / f"blobs{n}.json"),
            *_report(tmp_path)]


def test_the_token_comes_from_the_environment_never_the_command_line(tmp_path, monkeypatch, hub):
    downloads = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: downloads.append(kw["token"]))
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    assert role.main(_weights(tmp_path, 1)) == 0
    assert downloads == ["hf_from_the_env_file"]
    assert hub.calls == [("org/model", SHA, "hf_from_the_env_file", True)]
    monkeypatch.delenv("HF_TOKEN")
    hub.calls.clear()
    # The library is already imported here, so stand in for it having read the variable at import.
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    assert role.main(_weights(tmp_path, 2)) == 0
    # With nothing granted the hub client is told to use no token at all.
    assert downloads == ["hf_from_the_env_file", False]
    assert hub.calls == [("org/model", SHA, False, True)]
    with pytest.raises(SystemExit):
        role.main([*_weights(tmp_path, 3), "--token", "x"])


def test_resolve_asks_the_hub_nothing(tmp_path, monkeypatch, hub):
    out = tmp_path / "resolved.json"
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    assert role.main(_resolve("tests.provision.tasks", out, tmp_path)) == 0
    assert Resolved.model_validate_json(out.read_text()).weights.revision == SHA
    assert hub.calls == []


@pytest.mark.parametrize("module, reason", [("tests.provision", "defines no TASK"),
                                            ("tests.provision.unpinned_task", "is not a commit id"),
                                            ("tests.provision.invalid_task", "does not validate")])
def test_a_task_resolve_refuses_exits_with_its_own_status_and_says_why(tmp_path, monkeypatch, capsys, hub, module,
                                                                      reason):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    out = tmp_path / "resolved.json"
    assert role.main(_resolve(module, out, tmp_path)) == role.REFUSED_EXIT
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith("refused: ") and reason in last and not out.exists()


def _unpinned_tokenizer_task():
    mrcr = MRCRDataset(path="openai/mrcr", revision=DATA_SHA, tokenizer_id="org/unpinned")
    return task(source=DataSource(dataset=mrcr, split=SplitSpec(dev=Count(value=5), test=Count(value=4))))


def _tools_without_data():
    from tests.unit.test_tool_assets import FakeAssets
    from tests.unit.test_trajectory_driver import _agent

    t = no_data_task()
    t.agent = replace(_agent(), assets=FakeAssets())
    return t


@pytest.mark.parametrize("make, reason", [
    (lambda: task(data_revision="main"), "is not a commit id"),
    (lambda: task(dev_dataset=dataset(revision="main")), "dev_dataset .* is not a commit id"),
    (_unpinned_tokenizer_task, "org/unpinned: revision 'main' is not a commit id"),
    (_tools_without_data, "ToolAssets but no tenant carries a data source"),
])
def test_what_resolve_refuses_in_a_task_is_a_refusal(hub, make, reason):
    with pytest.raises(role.Refused, match=reason):
        role.resolve(make())


def test_a_task_whose_provisioning_does_not_validate_is_a_refusal(hub):
    from tests.unit.test_tool_assets import FakeAssets
    from tests.unit.test_trajectory_driver import _agent

    class Unnamed(FakeAssets):
        name = 5

    with pytest.raises(role.Refused, match="(?s)the task's provisioning does not validate: .*ToolSource.*name"):
        role.resolve(task(agent=replace(_agent(), assets=Unnamed())))


def _hub_raising(monkeypatch, error: Exception) -> None:
    class Api:
        def model_info(self, repo, **kwargs):
            raise error

    monkeypatch.setattr("huggingface_hub.HfApi", Api)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_hub_that_refuses_a_weights_download_ends_the_role_with_the_hub_refusal_status(tmp_path, monkeypatch,
                                                                                         capsys, status):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    def refused(**kw):
        raise fake_hub.hub_error(status, "https://huggingface.co/api/models/org/model/revision/" + SHA)

    monkeypatch.setattr("huggingface_hub.snapshot_download", refused)
    out = tmp_path / "entry"
    out.mkdir()
    args = ["weights", "--repo", "org/model", "--revision", SHA, "--out", str(out), "--source-out",
            str(tmp_path / "source.json"), *_outputs(tmp_path), "--blobs-out", str(tmp_path / "blobs.json"),
            *_report(tmp_path)]
    assert role.main(args) == role.HUB_REFUSED_EXIT
    assert capsys.readouterr().err.strip().splitlines()[-1].startswith(f"refused: the hub answered {status}: ")


def _dataset_not_found():
    """What ``datasets.load_dataset`` raises for a repo or revision the hub does not have: its own
    error, re-raised without the hub client's error as its cause."""
    from datasets.exceptions import DatasetNotFoundError

    try:
        try:
            raise fake_hub.hub_error(404, "https://huggingface.co/api/datasets/corp")
        except Exception as e:
            raise DatasetNotFoundError("Dataset 'corp' doesn't exist on the Hub or cannot be accessed.") from e
    except DatasetNotFoundError as e1:
        try:
            raise e1 from None
        except DatasetNotFoundError as e2:
            return e2


def _wrapped_hub_error():
    """A loader's own error, raised from the hub client's 404."""
    try:
        raise fake_hub.hub_error(404, "https://huggingface.co/api/datasets/corp")
    except Exception as e:
        try:
            raise RuntimeError("could not load corp") from e
        except RuntimeError as wrapped:
            return wrapped


@pytest.mark.parametrize("make_error, says", [
    (lambda: fake_hub.hub_error(404, "https://huggingface.co/api/datasets/corp"), "the hub answered 404"),
    (_wrapped_hub_error, "the hub answered 404"),
    (_dataset_not_found, "the hub has no such dataset or revision, or it is gated"),
], ids=["hub", "wrapped", "datasets"])
def test_a_hub_that_refuses_a_corpus_ends_the_rows_role_with_the_hub_refusal_status(tmp_path, monkeypatch, capsys,
                                                                                    make_error, says):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    def refused(ds):
        raise make_error()

    monkeypatch.setattr(role, "_corpus_for", refused)
    out = tmp_path / "entry"
    out.mkdir()
    assert role.main(["rows", "--task-module", "tests.provision.tasks", "--out", str(out),
                      *_report(tmp_path)]) == role.HUB_REFUSED_EXIT
    assert capsys.readouterr().err.strip().splitlines()[-1].startswith(f"refused: {says}")


def _timeout_while_handling_a_404(*, suppress: bool):
    """A timeout of a fallback request a library makes while it handles a 404 for an optional
    file: the 404 is the timeout's context, not its cause, and no answer to what was asked."""
    try:
        try:
            raise fake_hub.hub_error(404, "https://huggingface.co/api/datasets/corp/tree/main/optional.json")
        except Exception:
            if suppress:
                raise TimeoutError("read timed out while fetching the fallback listing") from None
            raise TimeoutError("read timed out while fetching the fallback listing")
    except TimeoutError as e:
        return e


@pytest.mark.parametrize("make_error", [
    lambda: fake_hub.hub_error(500), lambda: fake_hub.hub_error(503), lambda: fake_hub.hub_error(429),
    # What the client raises offline, when the file is in no local cache either: no answer at all.
    lambda: __import__("huggingface_hub.errors", fromlist=["x"]).LocalEntryNotFoundError("no connection"),
    lambda: ConnectionError("Couldn't reach 'corp' on the Hub (ConnectError)"),
    lambda: _timeout_while_handling_a_404(suppress=False),
    lambda: _timeout_while_handling_a_404(suppress=True),
], ids=["500", "503", "429", "offline", "unreachable", "timeout-in-404-handler", "timeout-in-404-handler-suppressed"])
def test_a_hub_that_fails_for_now_is_an_ordinary_failure_of_any_role(tmp_path, monkeypatch, make_error):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    error = make_error()

    def failing(ds):
        raise error

    monkeypatch.setattr(role, "_corpus_for", failing)
    out = tmp_path / "entry"
    out.mkdir()
    with pytest.raises(type(error)):
        role.main(["rows", "--task-module", "tests.provision.tasks", "--out", str(out), *_report(tmp_path)])


def test_rows_the_corpus_cannot_partition_as_declared_are_a_refusal(tmp_path, monkeypatch, capsys, fake_corpus):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    out = tmp_path / "entry"
    out.mkdir()
    assert role.main(["rows", "--task-module", "tests.provision.oversized_task", "--out", str(out),
                      *_report(tmp_path)]) == role.REFUSED_EXIT
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last == "refused: tenant t0: split dev(15) + test(10) exceeds corpus size 20"


def test_a_task_with_no_data_is_a_refusal_of_the_rows_role(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    out = tmp_path / "entry"
    out.mkdir()
    monkeypatch.setattr(role, "load_task", lambda module: no_data_task())
    args = ["rows", "--task-module", "x", "--out", str(out), *_report(tmp_path)]
    assert role.main(args) == role.REFUSED_EXIT
    assert capsys.readouterr().err.strip().splitlines()[-1] == "refused: no tenant of the task carries a data source"


def test_a_bundle_its_builder_refuses_is_a_refusal_and_any_other_build_error_is_not(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    from hls.spec.agent import AssetsRefused
    from tests.unit.test_tool_assets import FakeAssets

    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"dev": ["1"], "test": ["2"]}))
    monkeypatch.setattr(role, "load_task", lambda module: _tools_task())

    def refusing(self, variant, **kw):
        raise AssetsRefused("the dev bundle would hold test evidence")

    monkeypatch.setattr(FakeAssets, "build", refusing)
    (tmp_path / "a").mkdir()
    args = ["tools", "--task-module", "x", "--variant", "dev", "--ids", str(ids), *_report(tmp_path)]
    assert role.main([*args, "--out", str(tmp_path / "a")]) == role.REFUSED_EXIT
    assert capsys.readouterr().err.strip().splitlines()[-1] == "refused: the dev bundle would hold test evidence"

    def dying(self, variant, **kw):
        raise RuntimeError("builder died")

    monkeypatch.setattr(FakeAssets, "build", dying)
    (tmp_path / "b").mkdir()
    with pytest.raises(RuntimeError, match="builder died"):
        role.main([*args, "--out", str(tmp_path / "b")])


def test_a_garbled_hub_reply_is_no_refusal(tmp_path, monkeypatch):
    """Not every ValueError is the task's: a reply the client cannot parse is an ordinary
    failure, which the stage retries."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: None)
    _hub_raising(monkeypatch, json.JSONDecodeError("Expecting value", "<html>bad gateway</html>", 0))
    with pytest.raises(json.JSONDecodeError):
        role.main(_weights(tmp_path, 1))


def test_a_task_module_that_does_not_import_is_an_ordinary_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    with pytest.raises(ModuleNotFoundError):
        role.main(_resolve("tests.provision.no_such_module", tmp_path / "r.json", tmp_path))


def test_the_roles_apply_the_task_overrides_they_are_given(tmp_path, monkeypatch, fake_corpus, hub):
    """A grading's override of its test split is what its provision stage provisions: resolve keys
    the rows by the overridden split, and the rows role writes that many test rows."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    overrides = ["--overrides", json.dumps({"task.data_source.split.test.value": 3})]
    keys = []
    for name, extra in (("bare", []), ("changed", overrides)):
        where = tmp_path / name
        where.mkdir()
        assert role.main([*_resolve("tests.provision.tasks", where / "resolved.json", where), *extra]) == 0
        keys.append(Resolved.model_validate_json((where / "resolved.json").read_text()).rows_key)
    assert keys[0] != keys[1]
    out = tmp_path / "entry"
    out.mkdir()
    assert role.main(["rows", "--task-module", "tests.provision.tasks", "--out", str(out), *_report(tmp_path),
                      *overrides]) == 0
    assert len(json.loads((out / role.IDS_FILE).read_text())["test"]) == 3


@pytest.mark.parametrize("overrides, why", [
    ({"task.no_such_field": 1}, "name no field of the task"),
    ({"seed": 1}, "TaskRef: value_error overrides"),
    ({"task.seed": "seven"}, "int_parsing seed"),
], ids=["no-such-field", "not-a-task-path", "refused-value"])
def test_an_override_the_task_does_not_take_is_a_refusal(tmp_path, monkeypatch, capsys, hub, overrides, why):
    """The role reads its overrides as the plan's task reference holds them: a key that is no
    ``task.<field>`` path, a field the task does not have and a value its field refuses are each
    the task as declared refused."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_the_env_file")
    out = tmp_path / "resolved.json"
    assert role.main([*_resolve("tests.provision.tasks", out, tmp_path), "--overrides",
                      json.dumps(overrides)]) == role.REFUSED_EXIT
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith("refused: task module tests.provision.tasks: its overrides do not apply")
    assert why in last and not out.exists()
