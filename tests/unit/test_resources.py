"""Tests for hls.spec.resources: shape validation, combination and grant checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hls.spec.resources import (GrantedResources, Mismatch, ResourceShape, check_grant, concurrent,
                                gpu_kind_of, sequential)

BUILD = ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=180, disk_gib=200, time_sec=12 * 3600)
DRIVE = ResourceShape(gpus=2, gpu_kind="H200", cpus=8, memory_gib=160, disk_gib=50, time_sec=3600)
SCORE = ResourceShape(cpus=2, memory_gib=16, time_sec=600)


def test_gpus_need_a_kind():
    with pytest.raises(ValidationError):
        ResourceShape(gpus=1, cpus=8, memory_gib=10, time_sec=60)


def test_no_gpus_means_no_kind():
    with pytest.raises(ValidationError):
        ResourceShape(gpus=0, gpu_kind="H200", cpus=8, memory_gib=10, time_sec=60)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ResourceShape(cpus=1, memory_gib=1, time_sec=1, gpu=1)


def test_sequential_takes_max_and_sums_time():
    s = sequential([BUILD, DRIVE, SCORE])
    assert (s.gpus, s.gpu_kind, s.cpus, s.memory_gib, s.disk_gib) == (2, "H200", 8, 180, 200)
    assert s.time_sec == 12 * 3600 + 3600 + 600


def test_concurrent_sums_and_takes_max_time():
    artifact = ResourceShape(gpus=1, gpu_kind="H200", cpus=6, memory_gib=120, time_sec=3600)
    driver = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=40, time_sec=3000)
    c = concurrent([artifact, driver])
    assert (c.gpus, c.cpus, c.memory_gib, c.time_sec) == (2, 8, 160, 3600)


def test_combining_different_gpu_kinds_fails():
    other = ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=10, time_sec=60)
    with pytest.raises(ValueError, match="gpu kinds"):
        sequential([BUILD, other])


def test_combining_nothing_fails():
    with pytest.raises(ValueError):
        sequential([])


def test_check_grant_exact_match_passes():
    g = GrantedResources(gpus=1, gpu_kind="H200", cpus=8, memory_gib=180.0, disk_free_gib=500.0)
    assert check_grant(BUILD, g) == []


def test_check_grant_reports_every_mismatch():
    g = GrantedResources(gpus=8, gpu_kind="H100", cpus=16, memory_gib=170.0, disk_free_gib=10.0)
    fields = {m.field for m in check_grant(BUILD, g)}
    assert fields == {"gpus", "gpu_kind", "cpus", "memory_gib", "disk_gib"}


def test_more_cpus_than_declared_is_a_mismatch():
    # the brief states the declared shape, so a larger grant makes it false too
    g = GrantedResources(gpus=1, gpu_kind="H200", cpus=16, memory_gib=180.0, disk_free_gib=500.0)
    assert check_grant(BUILD, g) == [Mismatch("cpus", 8, 16)]


def test_memory_within_tolerance_passes():
    g = GrantedResources(gpus=1, gpu_kind="H200", cpus=8, memory_gib=179.5, disk_free_gib=500.0)
    assert check_grant(BUILD, g) == []


def test_unknown_gpu_kind_rejected():
    with pytest.raises(ValidationError, match="unknown gpu kind"):
        ResourceShape(gpus=1, gpu_kind="H20", cpus=1, memory_gib=1, time_sec=1)
    with pytest.raises(ValidationError, match="unknown gpu kind"):
        GrantedResources(gpus=1, gpu_kind="h200", cpus=1, memory_gib=1.0, disk_free_gib=1.0)


@pytest.mark.parametrize("name, kind", [
    ("NVIDIA H200", "H200"),
    ("NVIDIA H100 80GB HBM3", "H100"),
    ("NVIDIA L40", "L40"),
    ("NVIDIA L40S", "L40S"),
    ("NVIDIA A100-SXM4-80GB", "A100"),
    ("NVIDIA GeForce RTX 4090", None),
    ("", None),
])
def test_gpu_kind_of_product_names(name, kind):
    assert gpu_kind_of(name) == kind
