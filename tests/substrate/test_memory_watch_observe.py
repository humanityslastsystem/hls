"""The memory watch's evidence for the build rule beside the in-build reference service: a cgroup
it only observes (the job's batch step) is sampled and never killed, every watched step keeps its
highest sample, and ``memory.peak`` is read where the kernel keeps it."""

from __future__ import annotations

from pathlib import Path

from hls.substrate.memory_watch import MemoryWatch, read_peak

GIB = 2**30


def _files(directory: Path, gib: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.current").write_text(f"{gib * GIB}\n")
    (directory / "memory.stat").write_text(f"anon {gib * GIB}\nfile 0\nshmem 0\n")
    (directory / "memory.events").write_text("oom 0\noom_kill 0\n")


def test_an_observed_cgroup_is_sampled_and_never_killed_and_keeps_its_highest_sample(tmp_path):
    watch = MemoryWatch(enforce=True)
    batch = tmp_path / "step_batch"
    _files(batch, 8)
    watch.observe("batch", batch, role="runner")
    watch.sample_once()
    _files(batch, 3)
    watch.sample_once()
    assert watch.last("batch").footprint == 3 * GIB and watch.highest("batch").footprint == 8 * GIB
    assert watch.breach("batch") is None


def test_a_watched_step_keeps_its_highest_sample_and_is_killed_past_its_limit_as_before(tmp_path):
    killed = []
    watch = MemoryWatch(enforce=True)
    step = tmp_path / "step_3" / "user"
    _files(step, 2)
    watch.watch("1.3", step, 4 * GIB, lambda: killed.append(True), role="r")
    watch.sample_once()
    _files(step, 1)
    watch.sample_once()
    assert watch.highest("1.3").footprint == 2 * GIB and not killed
    _files(step, 5)
    watch.sample_once()
    assert killed == [True] and watch.highest("1.3").footprint == 5 * GIB


def test_memory_peak_is_read_where_the_file_is_and_is_none_elsewhere(tmp_path):
    assert read_peak(tmp_path) is None
    (tmp_path / "memory.peak").write_text(f"{7 * GIB}\n")
    assert read_peak(tmp_path) == 7 * GIB
    (tmp_path / "memory.peak").write_text("garbage\n")
    assert read_peak(tmp_path) is None
