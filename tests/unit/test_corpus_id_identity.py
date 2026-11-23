"""`corpus_id` must name everything that changes the provisioned corpus.

It is the provisioning cache key and the directory name, so two configs that write
different rows must not share one. `id_column` does change what is written: unset, it
makes `corpus()` append a synthetic `SOURCE_INDEX` column, and it is what `row_id` keys
each row by, which decides the dev/test partition and what a removal list names.
"""

import pytest

from hls.spec.datasets import SOURCE_INDEX, HFDataset
from hls.prompt import JinjaPromptTemplate


def _ds(**kw):
    base = dict(path="cais/mmlu", subset="abstract_algebra", revision="abc123",
                source_split="test",
                prompt=JinjaPromptTemplate(template="{{ question }}"),
                target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))
    return HFDataset(**{**base, **kw})


@pytest.mark.unit
def test_id_column_is_in_the_corpus_id():
    """These two write different columns and different row ids."""
    assert _ds(id_column=None).corpus_id() != _ds(id_column="question").corpus_id()


@pytest.mark.unit
def test_unset_id_column_names_the_synthetic_column():
    """Unset is spelled as the column `corpus()` actually adds, not as `None`, so the
    key says what is on disk."""
    assert _ds(id_column=None).corpus_id().endswith(f"|id={SOURCE_INDEX}")
    assert _ds(id_column="question").corpus_id().endswith("|id=question")


@pytest.mark.unit
def test_an_empty_id_column_is_a_real_column_name_not_the_synthetic_one():
    """`corpus()` and `row_id` both branch on `is None`, so `""` takes the real-column
    path; the key has to say the same rather than claim `SOURCE_INDEX`."""
    assert _ds(id_column="").corpus_id().endswith("|id=")
    assert _ds(id_column="").corpus_id() != _ds(id_column=None).corpus_id()


@pytest.mark.unit
def test_the_rest_of_the_identity_is_unchanged():
    cid = _ds(id_column=None).corpus_id()
    assert cid.startswith("cais/mmlu:abstract_algebra@abc123|src=test|")


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [("path", "other/ds"), ("subset", "other"), ("revision", "def456"),
     ("source_split", "train")],
)
def test_every_other_identity_field_still_separates(field, value):
    assert _ds(id_column=None).corpus_id() != _ds(id_column=None, **{field: value}).corpus_id()


@pytest.mark.unit
def test_subclasses_inherit_the_field():
    """Overrides extend the base string rather than replacing it, so the new component
    reaches every provisionable HFDataset subclass."""
    from hls.spec.datasets import WildChatDataset

    a = WildChatDataset(path="allenai/WildChat-1M", revision="r1", id_column=None)
    b = WildChatDataset(path="allenai/WildChat-1M", revision="r1",
                        id_column="conversation_hash")
    assert a.corpus_id() != b.corpus_id()
    assert "|id=" in a.corpus_id()
