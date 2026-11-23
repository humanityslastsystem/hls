"""deepresearch's task module: its id, the catalog's name for it, every hub source it reads at a
commit, and no corpus or answer kind left that no task uses."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hls.provision import role
from hls.provision.model import is_revision
from hls.provision.rows import unit_descriptions
from hls.recipes import catalog
from hls.spec.hub import declared_hub_revisions
from tasks.deepresearch import datasets as qa
from tasks.deepresearch import task


def test_the_module_has_the_tasks_id_and_the_catalog_names_it():
    assert task.TASK.id == "deepresearch"
    assert catalog.task_module("deepresearch") == "tasks.deepresearch.task"
    assert catalog.catalog_task("tasks.deepresearch.task") == "deepresearch"


def test_every_hub_source_the_task_reads_is_at_a_commit():
    units = unit_descriptions(task.TASK)
    datasets = [d for u in units for d in (u["data_source"].dataset, u["dev_dataset"]) if d is not None]
    rows = [source for d in datasets for source in declared_hub_revisions(d, reads="corpus")]
    tools = declared_hub_revisions(task.TASK.agent.assets, reads="build")
    assert rows == [("Tevatron/browsecomp-plus", "144cff8e35b5eaef7e526346aa60774a9deb941f")]
    assert tools and all(is_revision(revision) for _, revision in rows + tools)
    role._check_rows_are_pinned(units)
    role._check_tools_are_pinned(task.TASK.agent.assets)


def test_no_corpus_or_answer_kind_is_left_that_no_task_uses():
    assert is_revision(qa.BROWSECOMP.revision)
    assert "answer_kind" not in qa.DeepResearchQADataset.model_fields


@pytest.mark.parametrize("cls", [qa.DeepResearchQADataset, qa.BrowseCompQADataset])
def test_a_qa_dataset_without_a_revision_is_refused(cls):
    # Provisioning refuses a branch, so a dataset is declared at a commit; there is no default.
    with pytest.raises(ValidationError, match="revision"):
        cls(path="Tevatron/browsecomp-plus", source_split="test")
