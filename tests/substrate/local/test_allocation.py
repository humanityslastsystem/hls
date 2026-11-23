"""A local allocation's placement, the host check and the job log's name."""

from __future__ import annotations

import pytest

from hls.substrate.local.allocation import LocalPlacement, check_node_name, check_placement, is_job_log, placement
from hls.substrate.local.supervisor import output_name
from tests.substrate.local.helpers import FakeInventory, shape


def test_a_local_placement_names_nothing_and_serializes_as_its_kind():
    p = placement(partition=None, qos=None, socket_tasks=None, exclude=())
    assert p.model_dump_json() == '{"kind":"local"}'
    assert (p.partition, p.qos, p.socket_tasks, p.exclude) == (None, None, None, [])
    assert p.request_problem(shape()) is None
    assert LocalPlacement.model_validate_json('{"kind": "local"}') == p


@pytest.mark.parametrize("kw", [dict(partition="gpu"), dict(qos="normal"), dict(socket_tasks=4),
                                dict(exclude=["host1"])])
def test_a_placement_that_asks_for_what_the_host_lacks_is_refused(kw):
    with pytest.raises(ValueError):
        placement(**{**dict(partition=None, qos=None, socket_tasks=None, exclude=()), **kw})


def test_the_host_check_refuses_what_the_empty_host_cannot_hold():
    host = FakeInventory(gpus=2).read(kept_memory_gib=0)
    check_placement(shape(gpus=2), LocalPlacement(), host)
    with pytest.raises(ValueError, match="3 H200 GPUs"):
        check_placement(shape(gpus=3), LocalPlacement(), host)


def test_the_job_log_is_named_by_the_job_id():
    assert is_job_log(output_name("1795000000"))
    assert not is_job_log("slurm-12.out") and not is_job_log("job-1.out.tmp") and not is_job_log("job-x.out")


def test_no_node_name_is_one_a_retry_can_exclude():
    with pytest.raises(ValueError):
        check_node_name("host1")
