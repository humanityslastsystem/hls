from __future__ import annotations

import json

from datasets import Dataset as ArrowDataset

from hls.prompt import JinjaPromptTemplate
from hls.provision.rows import provisioning_digest, unit_descriptions, write_rows
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from tests.provision.tasks import DATA_SHA, corpus, dataset, task, two_tenant_task


def test_a_unit_with_its_own_dev_corpus_screens_and_names_its_dev_rows_by_that_corpus(tmp_path):
    """The dev corpus names its rows by another column than the test corpus does, so a removal
    list or an id read through the wrong dataset fails outright instead of passing by luck."""
    main = ArrowDataset.from_list([{"q": f"M{i}"} for i in range(10)])
    dev = ArrowDataset.from_list([{"d": f"D{i}"} for i in range(10)])
    dev_dataset = HFDataset(path="devcorp", revision=DATA_SHA, source_split="train", id_column="d",
                            prompt=JinjaPromptTemplate(template="{{d}}"),
                            target=JinjaPromptTemplate(template="{{d}}"))
    source = DataSource(dataset=dataset(), dev_dataset=dev_dataset,
                        split=SplitSpec(dev=Count(value=10), test=Count(value=10)), removed_rows=("D3",))
    dev_ids, test_ids = write_rows(unit_descriptions(task(source=source)), tmp_path,
                                   corpus_for=lambda ds: main, corpus_of=lambda d: dev)
    written = [r["d"] for r in json.loads((tmp_path / "tenants" / "t0" / "dev" / "rows.json").read_text())]
    assert sorted(written) == [f"D{i}" for i in range(10) if i != 3]
    assert dev_ids == written
    tested = [r["q"] for r in json.loads((tmp_path / "tenants" / "t0" / "test" / "rows.json").read_text())]
    assert sorted(tested) == sorted(f"M{i}" for i in range(10)) and test_ids == tested


def test_the_rows_key_moves_with_the_corpus_the_dev_corpus_and_the_removal_list():
    """Each decides the rows written: another revision of the corpus or of the dev corpus is
    other rows, and the removal list decides which dev rows are written."""
    def key(*, data_revision: str = DATA_SHA, dev_revision: str = "a" * 40,
            removed: tuple[str, ...] = ("8",)) -> str:
        return provisioning_digest(unit_descriptions(task(data_revision=data_revision, removed=removed,
                                                          dev_dataset=dataset(revision=dev_revision))))

    assert key() == key()
    assert key(data_revision="c" * 40) != key()
    assert key(dev_revision="b" * 40) != key()
    assert key(removed=("8", "9")) != key()


def test_the_rows_key_moves_with_each_tenants_own_corpus():
    """Each tenant's rows are read off its own corpus: another revision of either tenant's corpus
    is other rows, and so are the same two revisions held by the other tenants."""
    a, b, c = "a" * 40, "b" * 40, "c" * 40

    def key(t0: str, t1: str) -> str:
        return provisioning_digest(unit_descriptions(two_tenant_task(revisions=(t0, t1))))

    assert key(a, b) == key(a, b)
    assert key(c, b) != key(a, b)
    assert key(a, c) != key(a, b)
    assert key(b, a) != key(a, b)


def test_the_removal_list_screens_the_dev_rows_and_leaves_the_test_rows_whole(tmp_path):
    """The driver drops listed rows from the test split as it draws a schedule, so test rows
    are written whole; only the dev copy is screened. The list here names one dev row and every
    test row."""
    def rows(base, split: str) -> list[dict]:
        return json.loads((base / "tenants" / "t0" / split / "rows.json").read_text())

    def write(base, removed: tuple[str, ...]) -> tuple[list[str], list[str]]:
        return write_rows(unit_descriptions(task(removed=removed)), base, corpus_for=lambda ds: corpus(),
                          corpus_of=lambda d: corpus())

    write(tmp_path / "whole", ())
    dev, test = rows(tmp_path / "whole", "dev"), rows(tmp_path / "whole", "test")
    listed = tuple(sorted([str(dev[0]["q"]), *(str(r["q"]) for r in test)]))
    dev_ids, test_ids = write(tmp_path / "screened", listed)
    assert rows(tmp_path / "screened", "dev") == dev[1:] and dev_ids == [str(r["q"]) for r in dev[1:]]
    assert rows(tmp_path / "screened", "test") == test and test_ids == [str(r["q"]) for r in test]


def test_units_over_one_corpus_fetch_it_once_and_the_fetchers_take_only_the_data_source(tmp_path):
    fetched = []

    def corpus_for(ds):
        fetched.append(ds)
        return corpus()

    units = unit_descriptions(two_tenant_task())
    write_rows(units, tmp_path, corpus_for=corpus_for, corpus_of=lambda d: corpus())
    assert len(fetched) == 1
    for tenant in ("t0", "t1"):
        assert (tmp_path / "tenants" / tenant / "test" / "rows.json").is_file()

