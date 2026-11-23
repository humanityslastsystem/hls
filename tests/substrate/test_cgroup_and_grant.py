from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.substrate.cgroup import job_dir, kill_all, own_cgroup, pids, read_keyed, read_limit, step_dir
from hls.spec.resources import HardwareRecord
from hls.substrate.grant import cpu_model, gpus_computing, list_gpus, measure, openable_gpu_devices
from tests.substrate.slurm_fakes import FakeRun, done

STEP = "/system.slice/slurmstepd.scope/job_1234110/step_1/user/task_0"
BATCH = "/system.slice/slurmstepd.scope/job_1234110/step_batch/user/task_0"
SMI = ("GPU-d7fdcb95-ed58-0b76-ad0f-88ff81569074, NVIDIA H200\n"
       "GPU-6fe08ff0-96d2-f6db-285c-0abad16b8d5a, NVIDIA H200\n")


def test_own_cgroup_and_the_job_and_step_directories(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{STEP}\n")
    assert own_cgroup(proc) == STEP
    assert job_dir(STEP, tmp_path) == tmp_path / "system.slice/slurmstepd.scope/job_1234110"
    assert step_dir(STEP, tmp_path) == tmp_path / "system.slice/slurmstepd.scope/job_1234110/step_1/user"
    assert step_dir(BATCH, tmp_path).name == "user"
    with pytest.raises(RuntimeError):
        step_dir("/user.slice/user-1.slice/session-3.scope")
    proc.write_text("12:memory:/x\n")
    with pytest.raises(RuntimeError):
        own_cgroup(proc)


def test_limit_and_keyed_files(tmp_path):
    (tmp_path / "memory.max").write_text("17179869184\n")
    (tmp_path / "memory.swap.max").write_text("max\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 0\nmax 127\noom 1\noom_kill 1\noom_group_kill 0\n")
    assert read_limit(tmp_path / "memory.max") == 16 * 2**30
    assert read_limit(tmp_path / "memory.swap.max") is None
    assert read_keyed(tmp_path / "memory.events")["oom_kill"] == 1


def test_kill_all_signals_every_process_in_the_step(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "task_0").mkdir()
        (tmp_path / "task_0" / "cgroup.procs").write_text(f"{child.pid}\n")
        (tmp_path / "cgroup.procs").write_text("")
        assert pids(tmp_path) == [child.pid]
        assert kill_all(tmp_path) == 1
        assert child.wait(10) == -9
    finally:
        if child.poll() is None:
            child.kill()


def test_gone_processes_are_skipped(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(10)
    (tmp_path / "cgroup.procs").write_text(f"{child.pid}\n")
    time.sleep(0.05)
    assert kill_all(tmp_path) == 0


def _limits(d: Path, mem: str, swap: str | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.max").write_text(mem + "\n")
    if swap is not None:
        (d / "memory.swap.max").write_text(swap + "\n")
    return d


def _dev(d: Path, names: list[str], denied: list[str] = ()) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in [*names, *denied, "nvidiactl", "nvidia-uvm"]:
        (d / n).write_text("")
    for n in denied:
        (d / n).chmod(0)
    return d


# A mode-0 file stands in for a device node the device cgroup denies; root opens it anyway.
needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root opens mode-0 files")


@needs_non_root
def test_openable_devices_are_the_gpu_count(tmp_path):
    dev = _dev(tmp_path / "dev", ["nvidia3", "nvidia6"], denied=["nvidia0"])
    assert openable_gpu_devices(dev) == ["nvidia3", "nvidia6"]


def test_measure_an_allocation(tmp_path):
    run = FakeRun().on("nvidia-smi", done(SMI))
    m = measure(limit_dir=_limits(tmp_path / "job", "34359738368", "max"), scratch=tmp_path,
                dev=_dev(tmp_path / "dev", ["nvidia3", "nvidia6"]), run=run, cpus=lambda: 8)
    assert (m.granted.gpus, m.granted.gpu_kind, m.granted.cpus, m.granted.memory_gib) == (2, "H200", 8, 32.0)
    assert m.gpu_uuids[0].startswith("GPU-d7fdcb95") and m.swap_max_bytes is None and m.problems == []


def test_measure_records_the_gpu_and_cpu_models_as_the_node_reports_them(tmp_path):
    """The grant's hardware record is the GPUs' product name, their count and
    the CPU model, each distinct name sorted and joined when the node reports several."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Example  CPU 9000\n\nprocessor\t: 1\n"
                       "model name\t: Example CPU 9000\n\nprocessor\t: 2\nmodel name\t: Another CPU 1\n")
    run = FakeRun().on("nvidia-smi", done(SMI))
    m = measure(limit_dir=_limits(tmp_path / "job", "34359738368", "max"), scratch=tmp_path,
                dev=_dev(tmp_path / "dev", ["nvidia3", "nvidia6"]), cpuinfo=cpuinfo, run=run, cpus=lambda: 8)
    assert m.granted.hardware() == HardwareRecord(gpu_model="NVIDIA H200", gpus=2,
                                            cpu_model="Another CPU 1; Example CPU 9000", cpus=8, memory_gib=32.0)
    cpuinfo.write_text("processor\t: 0\nBogoMIPS\t: 2000.00\n")
    assert cpu_model(cpuinfo) is None


def test_measure_reports_a_gpu_that_does_not_open_and_no_memory_limit(tmp_path):
    run = FakeRun().on("nvidia-smi", done(SMI))
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       2097152 kB\n")
    m = measure(limit_dir=_limits(tmp_path / "job", "max", "0"), scratch=tmp_path,
                dev=_dev(tmp_path / "dev", ["nvidia3"]), meminfo=meminfo, run=run, cpus=lambda: 1)
    assert m.swap_max_bytes == 0 and m.granted.memory_gib == 2.0
    assert any("2 GPUs but 1 device nodes" in p for p in m.problems)
    assert any("no memory limit" in p for p in m.problems)


@needs_non_root
def test_measure_without_gpus_does_not_ask_nvidia_smi(tmp_path):
    m = measure(limit_dir=_limits(tmp_path / "s", "1073741824", "0"), scratch=tmp_path,
                dev=_dev(tmp_path / "dev", [], denied=["nvidia0"]), run=FakeRun(), cpus=lambda: 1)
    assert (m.granted.gpus, m.granted.gpu_kind, m.problems) == (0, None, [])


def test_a_card_computes_when_the_hosts_nvidia_smi_lists_any_compute_process_on_it():
    smi = Path("/opt/nvidia/bin/nvidia-smi")
    run = FakeRun().on("nvidia-smi", done(stdout="GPU-a, 100\nGPU-b, 200\nGPU-a, 300\n"))
    assert gpus_computing(smi, run) == {"GPU-a", "GPU-b"}
    assert run.calls[-1] == [str(smi), "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    assert gpus_computing(smi, FakeRun().on("nvidia-smi", done(stdout=""))) == set()
    assert gpus_computing(smi, FakeRun().on("nvidia-smi", done(stderr="Failed to initialize NVML", rc=9))) is None
    # A program that cannot be run lists nothing, and no card counts as free.
    assert gpus_computing(smi, FakeRun().on("nvidia-smi", FileNotFoundError())) is None


def test_list_gpus_when_nvidia_smi_fails():
    assert list_gpus(FakeRun().on("nvidia-smi", done(stdout="No devices were found", rc=6))) == []
    assert list_gpus(FakeRun().on("nvidia-smi", done(stderr="Failed to initialize NVML", rc=9))) is None
    assert list_gpus(FakeRun().on("nvidia-smi", FileNotFoundError())) == []


def test_kill_all_skips_a_process_it_may_not_signal(tmp_path, monkeypatch):
    (tmp_path / "cgroup.procs").write_text("111\n222\n333\n")
    signalled = []

    def fake_kill(pid, sig):
        if pid == 111:
            raise PermissionError(pid)
        if pid == 222:
            raise ProcessLookupError(pid)
        signalled.append(pid)

    monkeypatch.setattr("hls.substrate.cgroup.os.kill", fake_kill)
    assert kill_all(tmp_path) == 1
    assert signalled == [333]


def test_measure_reports_nvidia_smi_failing_with_a_device_open(tmp_path):
    run = FakeRun().on("nvidia-smi", done(stderr="Failed to initialize NVML", rc=9))
    m = measure(limit_dir=_limits(tmp_path / "job", "1073741824", "0"), scratch=tmp_path,
                dev=_dev(tmp_path / "dev", ["nvidia0"]), run=run, cpus=lambda: 1)
    assert any("nvidia-smi failed" in p for p in m.problems)
    assert m.gpu_uuids == []
