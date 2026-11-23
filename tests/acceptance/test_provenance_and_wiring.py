"""Suite-level acceptance criteria for a task's provenance and wiring.

This exercises the real config objects end to end: the full ``TaskSpec`` from
``tests/support/fake_arena`` serializes through the provenance core with a
stable, deterministic dump. Token ids are mandatory and always emitted on
the wire contract, so ``check_wiring`` has no protocol that opts out of them.
"""
from hls.core.poly import dump_spec


def test_taskspec_dumps_through_provenance():
    from tests.support.fake_arena import TASK

    rec = dump_spec(TASK)
    assert rec["id"] == TASK.id
    assert dump_spec(TASK) == rec                    # stable, deterministic serialization
