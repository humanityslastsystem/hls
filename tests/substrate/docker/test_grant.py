"""The host's GPUs by device node minor number, from nvidia-smi and the driver's ``information``
files, on the output of a driver 570 host and on the ``information`` file of an older driver."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hls.substrate.docker.grant import GpusUnknown, device_minor, driver_gpu_dir, gpus_by_minor
from hls.substrate.grant import Gpu

#: ``nvidia-smi --query-gpu=uuid,name,pci.bus_id --format=csv,noheader`` on a driver 570 host of
#: eight H200s: an eight-digit domain in upper-case hex.
SMI_570 = """\
GPU-967cdcf6-43c4-c8ce-6c7a-d9cdbbd37d77, NVIDIA H200, 00000000:03:00.0
GPU-7302451d-9342-ea40-a113-3a4e0872aeca, NVIDIA H200, 00000000:23:00.0
GPU-a4c23cef-f2f2-4528-7092-11b569e23985, NVIDIA H200, 00000000:43:00.0
GPU-9c505405-1d92-eeca-909e-7ef684eb7196, NVIDIA H200, 00000000:63:00.0
GPU-051dc7a0-39ec-d7e5-55ce-fbd99802d713, NVIDIA H200, 00000000:83:00.0
GPU-6b9761c5-beeb-b620-45cf-7fef8ebc6aeb, NVIDIA H200, 00000000:A3:00.0
GPU-b929d321-5037-cb9b-728a-d58621149726, NVIDIA H200, 00000000:C3:00.0
GPU-9032fd52-bd9f-22c0-eea2-37c118ce21aa, NVIDIA H200, 00000000:E3:00.0
"""
#: The same host's device minors, by bus, as each ``information`` file gives them: not in index order.
MINORS_570 = {"03": 4, "23": 1, "43": 0, "63": 3, "83": 6, "a3": 7, "c3": 5, "e3": 2}
#: That host's ``information`` file of the GPU on bus 03, as the driver writes it: a tab and a
#: space after each key, two tabs after the first two.
INFORMATION_570 = ("Model: \t\t NVIDIA H200\nIRQ:   \t\t 1276\nGPU UUID: \t GPU-967cdcf6-43c4-c8ce-6c7a-d9cdbbd37d77\n"
                   "Video BIOS: \t 96.00.a5.00.03\nBus Type: \t PCIe\nDMA Size: \t 52 bits\n"
                   "DMA Mask: \t 0xfffffffffffff\nBus Location: \t 0000:03:00.0\nDevice Minor: \t 4\n"
                   "GPU Firmware: \t 570.211.01\nGPU Excluded:\t No\n")
#: An older driver's ``information`` file: no firmware line, and ``Blacklisted`` for ``GPU Excluded``.
INFORMATION_OLD = ("Model: \t\t Tesla V100-SXM2-16GB\nIRQ:   \t\t 95\nGPU UUID: \t GPU-5fd3a8f2-0000-1111-2222-333344445555\n"
                   "Video BIOS: \t 88.00.4f.00.09\nBus Type: \t PCIe\nDMA Size: \t 47 bits\n"
                   "DMA Mask: \t 0x7fffffffffff\nBus Location: \t 0000:00:1e.0\nDevice Minor: \t 0\n"
                   "Blacklisted:\t No\n")


def _driver(proc: Path, minors: dict[str, int], uuids: dict[str, str]) -> None:
    for bus, minor in minors.items():
        directory = proc / "driver" / "nvidia" / "gpus" / f"0000:{bus}:00.0"
        directory.mkdir(parents=True)
        text = INFORMATION_570.replace("Device Minor: \t 4", f"Device Minor: \t {minor}")
        (directory / "information").write_text(text.replace("GPU-967cdcf6-43c4-c8ce-6c7a-d9cdbbd37d77", uuids[bus]))


def _smi(stdout: str = "", *, rc: int = 0, stderr: str = ""):
    def run(argv, **kwargs):
        assert argv == ["nvidia-smi", "--query-gpu=uuid,name,pci.bus_id", "--format=csv,noheader"]
        return subprocess.CompletedProcess(argv, rc, stdout, stderr)
    return run


def test_the_driver_570_host_maps_each_minor_to_its_gpu_out_of_index_order(tmp_path):
    uuids = {line.split(", ")[2][9:11].lower(): line.split(", ")[0] for line in SMI_570.splitlines()}
    _driver(tmp_path, MINORS_570, uuids)
    table = gpus_by_minor(_smi(SMI_570), proc=tmp_path)
    assert table[4] == Gpu("GPU-967cdcf6-43c4-c8ce-6c7a-d9cdbbd37d77", "NVIDIA H200")
    assert table[0] == Gpu("GPU-a4c23cef-f2f2-4528-7092-11b569e23985", "NVIDIA H200")
    assert table[7] == Gpu("GPU-6b9761c5-beeb-b620-45cf-7fef8ebc6aeb", "NVIDIA H200")
    assert sorted(table) == list(range(8))


@pytest.mark.parametrize("text, minor", [(INFORMATION_570, 4), (INFORMATION_OLD, 0)], ids=["driver-570", "older"])
def test_the_minor_is_read_off_a_new_and_an_old_drivers_information_file(text, minor):
    assert device_minor(text) == minor
    with pytest.raises(ValueError, match="no Device Minor"):
        device_minor(text.replace("Device Minor", "Device Major"))


@pytest.mark.parametrize("bus_id, directory", [
    ("00000000:A3:00.0", "0000:a3:00.0"),
    ("0000:03:00.0", "0000:03:00.0"),
    ("00000001:E3:00.7", "0001:e3:00.7"),
])
def test_nvidia_smis_bus_id_names_the_drivers_directory(bus_id, directory):
    assert driver_gpu_dir(bus_id) == directory


@pytest.mark.parametrize("bad", ["N/A", "00000000:A3", "0000:zz:00.0", "00000000:A3:00.8"])
def test_anything_but_a_pci_address_is_refused(bad):
    with pytest.raises(ValueError):
        driver_gpu_dir(bad)


def test_gpus_that_cannot_be_told_by_their_minors_fail_closed_in_one_line(tmp_path):
    """A driver that refuses the query, a line that is no GPU, and a GPU whose driver file is
    missing or gives no minor each raise, saying why; no nvidia-smi, or none listed, is no GPU."""
    with pytest.raises(GpusUnknown, match='nvidia-smi exited 2: Field "minor_number" is not a valid field'):
        gpus_by_minor(_smi(rc=2, stdout='Field "minor_number" is not a valid field to query.\n'), proc=tmp_path)
    with pytest.raises(GpusUnknown, match="not a UUID, a name and a PCI bus id"):
        gpus_by_minor(_smi("GPU-a, NVIDIA H200\n"), proc=tmp_path)
    with pytest.raises(GpusUnknown, match="information of GPU GPU-a cannot be read"):
        gpus_by_minor(_smi("GPU-a, NVIDIA H200, 00000000:03:00.0\n"), proc=tmp_path)
    (tmp_path / "driver" / "nvidia" / "gpus" / "0000:03:00.0").mkdir(parents=True)
    (tmp_path / "driver" / "nvidia" / "gpus" / "0000:03:00.0" / "information").write_text("Model: \t\t NVIDIA H200\n")
    with pytest.raises(GpusUnknown, match="gives no Device Minor"):
        gpus_by_minor(_smi("GPU-a, NVIDIA H200, 00000000:03:00.0\n"), proc=tmp_path)
    assert gpus_by_minor(_smi(rc=6, stdout="No devices were found\n"), proc=tmp_path) == {}

    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    assert gpus_by_minor(missing, proc=tmp_path) == {}
