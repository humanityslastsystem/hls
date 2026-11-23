"""``partition_by_hash``: dev/test membership decided by a hash of each row's own id,
so a row's side depends on nothing but the row and the split sizes."""
import hashlib

import pytest

from hls.spec.datasource import Count, DataSource, Fraction, PartitionError, SplitSpec, partition_by_hash


def _ids(n: int) -> list[str]:
    return [f"row-{i}" for i in range(n)]


def _sides(ids, dev, test):
    """The two splits as sets of ids rather than positions."""
    dev_idx, test_idx = partition_by_hash(ids, dev=dev, test=test)
    return {ids[i] for i in dev_idx}, {ids[i] for i in test_idx}


def test_partition_is_disjoint_and_sized():
    dev, test = partition_by_hash(_ids(100), dev=Count(value=10), test=Count(value=20))
    assert len(dev) == 10 and len(test) == 20
    assert set(dev).isdisjoint(test) and all(0 <= i < 100 for i in dev + test)


def test_membership_follows_the_id_not_the_position():
    """Reordering the corpus moves no row: each row's side is a function of its id."""
    ids = _ids(100)
    a = _sides(ids, Count(value=10), Count(value=20))
    b = _sides(ids[::-1], Count(value=10), Count(value=20))
    assert a == b


def test_removing_a_row_moves_no_other_row():
    """A row that leaves the corpus is replaced from the unassigned middle of the
    ranking; every other row keeps its side, whichever side the leaver was on."""
    ids = _ids(100)
    dev, test = _sides(ids, Count(value=10), Count(value=20))
    middle = set(ids) - dev - test
    for victim in (min(dev), min(test), min(middle)):
        rest = [i for i in ids if i != victim]
        dev2, test2 = _sides(rest, Count(value=10), Count(value=20))
        assert dev - {victim} <= dev2 and test - {victim} <= test2
        # The one newcomer (if any) came from the middle, never from the other side.
        assert (dev2 - dev) <= middle and (test2 - test) <= middle
        assert len(dev2) == 10 and len(test2) == 20


def test_one_side_alone_may_cover_the_whole_corpus():
    """A source with its own dev corpus ranks each corpus by itself, so a single side
    is not bound by the dev+test coupling and may take every row."""
    ids = _ids(30)
    dev, test = partition_by_hash(ids, dev=Count(value=30))
    assert sorted(dev) == list(range(30)) and test == []
    dev, test = partition_by_hash(ids, test=Count(value=30))
    assert dev == [] and sorted(test) == list(range(30))


def test_splits_come_back_in_hash_order():
    ids = _ids(50)
    dev, test = partition_by_hash(ids, dev=Count(value=5), test=Count(value=5))
    key = lambda i: hashlib.sha256(ids[i].encode()).hexdigest()  # noqa: E731
    assert dev == sorted(dev, key=key) and test == sorted(test, key=key)
    assert key(dev[-1]) < key(test[0])


def test_partition_supports_fractions():
    dev, test = partition_by_hash(_ids(100), dev=Fraction(value=0.1), test=Fraction(value=0.2))
    assert len(dev) == 10 and len(test) == 20


def test_partition_fails_loud_when_too_large():
    with pytest.raises(ValueError, match="(?i)exceed"):
        partition_by_hash(_ids(10), dev=Count(value=8), test=Count(value=5))


def test_a_split_that_yields_no_row_is_refused():
    """A split that is asked for must hold a row, whatever its size's kind: a Fraction of a small
    corpus can round to none, and a task graded, or tuned, on no row is a task-definition error.
    One row passes, and a Count larger than its corpus stays refused."""
    with pytest.raises(PartitionError, match=r"^test split resolves to 0 rows of a corpus of 4 rows"):
        partition_by_hash(_ids(4), dev=Count(value=1), test=Fraction(value=0.1))
    with pytest.raises(PartitionError, match=r"^dev split resolves to 0 rows of a corpus of 4 rows"):
        partition_by_hash(_ids(4), dev=Fraction(value=0.1), test=Count(value=1))
    with pytest.raises(PartitionError, match=r"^test split resolves to 0 rows of a corpus of 0 rows"):
        partition_by_hash([], test=Fraction(value=0.5))
    dev, test = partition_by_hash(_ids(10), dev=Fraction(value=0.1), test=Fraction(value=0.1))
    assert (len(dev), len(test)) == (1, 1)
    with pytest.raises(PartitionError, match=r"exceeds corpus size 10"):
        partition_by_hash(_ids(10), dev=Count(value=8), test=Count(value=5))


def test_ids_must_be_unique_and_must_not_look_like_turn_ids():
    with pytest.raises(ValueError, match="unique"):
        partition_by_hash(["a", "b", "a"], dev=Count(value=1))
    with pytest.raises(ValueError, match="turn suffix"):
        partition_by_hash(["a", "b.0"], dev=Count(value=1))
    with pytest.raises(ValueError, match="non-empty"):
        partition_by_hash(["a", ""], dev=Count(value=1))


def test_count_rejects_non_positive():
    with pytest.raises(Exception):
        Count(value=0)


def test_fraction_rejects_out_of_open_interval():
    with pytest.raises(Exception):
        Fraction(value=1.0)
    with pytest.raises(Exception):
        Fraction(value=0.0)


def test_datasource_serializes_with_real_dataset():
    from hls.core.poly import dump_spec
    from hls.prompt import JinjaPromptTemplate
    from hls.spec.datasets import HFDataset

    ds = DataSource(
        dataset=HFDataset(path="cais/mmlu", subset="anatomy", revision="main", source_split="test",
                          prompt=JinjaPromptTemplate(template="{{question}}"),
                          target=JinjaPromptTemplate(template="{{answer}}")),
        split=SplitSpec(dev=Count(value=5), test=Count(value=10)))
    rec = dump_spec(ds)
    assert rec["split"]["dev"]["value"] == 5 and rec["split"]["dev"]["kind"] == "Count"
    assert rec["split"]["test"]["value"] == 10 and rec["split"]["test"]["kind"] == "Count"
    assert rec["dataset"]["kind"] == "HFDataset" and rec["dataset"]["path"] == "cais/mmlu"
    assert rec["dataset"]["prompt"]["template"] == "{{question}}"
