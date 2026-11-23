import pytest
from pydantic import ValidationError

from hls.core.poly import dump_spec


def test_full_taskspec_dumps():
    from tests.support.fake_arena import TASK
    rec = dump_spec(TASK)
    assert rec["id"] == TASK.id
    assert dump_spec(TASK) == rec                  # stable, deterministic serialization


def test_taskspec_is_frozen():
    from tests.support.fake_arena import TASK
    with pytest.raises(ValidationError):
        TASK.id = "mutated"
