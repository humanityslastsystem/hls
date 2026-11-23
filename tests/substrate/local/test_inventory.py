"""The host's inventory, read from a stand-in ``/sys`` and ``/proc`` and a stand-in nvidia-smi."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hls.substrate.local.inventory import HostFacts, InventoryError, SysInventory, read_cores, read_memory_gib


def _root(tmp_path: Path, *, cores: list[list[int]], nodes: dict[int, str] | None, gpus: int = 0,
          siblings_file: str = "core_cpus_list") -> Path:
    root = tmp_path / "root"
    cpu = root / "sys/devices/system/cpu"
    cpu.mkdir(parents=True)
    online = sorted(c for core in cores for c in core)
    (cpu / "online").write_text(",".join(map(str, online)) + "\n")
    for core in cores:
        for c in core:
            topology = cpu / f"cpu{c}" / "topology"
            topology.mkdir(parents=True)
            (topology / siblings_file).write_text(",".join(map(str, core)) + "\n")
    if nodes is not None:
        for n, cpulist in nodes.items():
            (root / f"sys/devices/system/node/node{n}").mkdir(parents=True)
            (root / f"sys/devices/system/node/node{n}/cpulist").write_text(cpulist + "\n")
        (root / "sys/devices/system/node/possible").write_text("0-1\n")
    (root / "proc").mkdir(parents=True)
    (root / "proc/meminfo").write_text(f"MemTotal:       {64 * 1024 * 1024} kB\nMemFree:        1 kB\n")
    if gpus:
        for i in range(gpus):
            (root / "proc/driver/nvidia/gpus" / f"0000:{i:02x}:00.0").mkdir(parents=True)
        (root / "proc/driver/nvidia/version").write_text(
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.211.01  Tue Nov 25 16:36:36 UTC 2025\n")
        lib = tmp_path / "lib64"
        lib.mkdir()
        for name in ("libcuda.so", "libnvidia-ml.so"):
            (lib / f"{name}.570.211.01").touch()
            (lib / f"{name}.1").symlink_to(f"{name}.570.211.01")
    return root


def _smi(*lines: str, returncode: int = 0, lib: Path | None = None):
    def run(argv, **kw):
        if argv[0] == "/sbin/ldconfig":
            assert lib is not None
            return subprocess.CompletedProcess(argv, 0, stderr="", stdout="".join(
                f"\t{name}.1 (libc6,x86-64) => {lib}/{name}.1\n" for name in ("libcuda.so", "libnvidia-ml.so")))
        assert argv[0] == "nvidia-smi"
        return subprocess.CompletedProcess(argv, returncode, stdout="".join(f"{line}\n" for line in lines),
                                           stderr="")
    return run


def test_cores_group_their_threads_on_their_numa_node(tmp_path):
    root = _root(tmp_path, cores=[[0, 4], [1, 5], [2, 6], [3, 7]], nodes={0: "0-1,4-5", 1: "2-3,6-7"})
    cores = read_cores(root)
    assert [(c.numa, c.cpus) for c in cores] == [(0, (0, 4)), (0, (1, 5)), (1, (2, 6)), (1, (3, 7))]


def test_an_older_kernels_sibling_list_and_no_numa_nodes_read_as_node_0(tmp_path):
    root = _root(tmp_path, cores=[[0], [1]], nodes=None, siblings_file="thread_siblings_list")
    assert [(c.numa, c.cpus) for c in read_cores(root)] == [(0, (0,)), (0, (1,))]


def test_a_core_split_over_two_nodes_is_refused(tmp_path):
    root = _root(tmp_path, cores=[[0, 1]], nodes={0: "0", 1: "1"})
    with pytest.raises(InventoryError):
        read_cores(root)


def test_the_host_reads_gpus_by_uuid_with_their_kind_and_names_its_nvidia_smi(tmp_path):
    """The scheduler's passes run with no PATH of the user's, so the inventory names the nvidia-smi
    on this process's PATH by its absolute path, which lists the GPUs' compute processes for each
    pass."""
    root = _root(tmp_path, cores=[[0], [1]], nodes={0: "0-1"}, gpus=2)
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text("")
    lib = tmp_path / "lib64"
    facts = SysInventory(root=root, run=_smi("GPU-aaa, NVIDIA H200", "GPU-bbb, NVIDIA H200", lib=lib),
                         which=lambda name: str(smi) if name == "nvidia-smi" else None).read(kept_memory_gib=8)
    assert [(g.uuid, g.kind) for g in facts.gpus] == [("GPU-aaa", "H200"), ("GPU-bbb", "H200")]
    assert facts.nvidia_smi == smi and facts.memory_gib == 64 and facts.allocatable_memory_gib == 56
    # Beside it, the driver's files every GPU role is given.
    assert facts.driver is not None and facts.driver.version == "570.211.01"
    assert {f.path.name: f.names for f in facts.driver.libraries} == {
        "libnvidia-ml.so.570.211.01": ("libnvidia-ml.so.1", "libnvidia-ml.so.570.211.01"),
        "libcuda.so.570.211.01": ("libcuda.so", "libcuda.so.1", "libcuda.so.570.211.01")}
    assert [f.path for f in facts.driver.programs] == [smi]
    assert HostFacts.model_validate_json(facts.model_dump_json()) == facts


def test_a_gpu_host_whose_driver_a_gpu_role_cannot_be_given_is_refused_in_one_line(tmp_path):
    root = _root(tmp_path, cores=[[0], [1]], nodes={0: "0-1"}, gpus=2)
    (tmp_path / "lib64" / "libcuda.so.1").unlink()
    (tmp_path / "lib64" / "libcuda.so.1").symlink_to("libcuda.so.575.57.08")
    (tmp_path / "lib64" / "libcuda.so.575.57.08").touch()
    with pytest.raises(InventoryError, match=r"^the host's NVIDIA driver cannot be given to a GPU role: libcuda.so.1 "
                                             r"on this host is .* not the running NVIDIA driver's libcuda.so.570.211.01$"):
        SysInventory(root=root, run=_smi("GPU-aaa, NVIDIA H200", "GPU-bbb, NVIDIA H200", lib=tmp_path / "lib64"),
                     which=lambda name: "/bin/true").read(kept_memory_gib=8)
    without = SysInventory(root=root, run=_smi("GPU-aaa, NVIDIA H200", "GPU-bbb, NVIDIA H200", lib=tmp_path / "lib64"),
                           which=lambda name: None)
    with pytest.raises(InventoryError, match="no nvidia-smi on PATH"):
        (tmp_path / "lib64" / "libcuda.so.1").unlink()
        (tmp_path / "lib64" / "libcuda.so.1").symlink_to("libcuda.so.570.211.01")
        without.read(kept_memory_gib=8)


def test_a_host_without_gpus_records_no_driver(tmp_path):
    root = _root(tmp_path, cores=[[0]], nodes={0: "0"})
    assert SysInventory(root=root, run=_smi(), which=lambda name: None).read(kept_memory_gib=0).driver is None


def test_gpus_nvidia_smi_and_the_driver_disagree_on_are_refused(tmp_path):
    root = _root(tmp_path, cores=[[0]], nodes={0: "0"}, gpus=2)
    with pytest.raises(InventoryError, match="lists 1 GPUs but the driver lists 2"):
        SysInventory(root=root, run=_smi("GPU-aaa, NVIDIA H200")).read(kept_memory_gib=0)
    with pytest.raises(InventoryError, match="nvidia-smi failed"):
        SysInventory(root=root, run=_smi(returncode=9)).read(kept_memory_gib=0)


def test_a_host_without_the_driver_has_no_gpus(tmp_path):
    root = _root(tmp_path, cores=[[0]], nodes={0: "0"})

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])

    assert SysInventory(root=root, run=missing).read(kept_memory_gib=0).gpus == ()


def test_a_host_with_nvidia_smi_and_no_driver_loaded_has_no_gpus_and_nvidia_smi_is_not_asked(tmp_path):
    root = _root(tmp_path, cores=[[0]], nodes={0: "0"})
    asked = []

    def no_driver(argv, **kw):
        asked.append(argv)
        return subprocess.CompletedProcess(argv, 9, stdout="", stderr="couldn't communicate with the NVIDIA driver")

    assert SysInventory(root=root, run=no_driver).read(kept_memory_gib=0).gpus == ()
    assert asked == []


def test_this_hosts_cpus_and_memory_read():
    cores = read_cores(Path("/"))
    assert cores and all(c.cpus for c in cores)
    assert read_memory_gib(Path("/")) > 0
