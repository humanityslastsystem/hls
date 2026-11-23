"""The NVIDIA driver as a Docker role is given it: its files found on a host as ``ldconfig -p`` lists
them, the device nodes of a role's GPUs, and the gate's search paths, against a recorded host."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from hls.substrate.docker import nvidia
from hls.substrate.docker.launcher import GATE
from hls.substrate.docker.nvidia import (
    DriverFile,
    DriverUnusable,
    NvidiaDriver,
    driver_devices,
    driver_libraries,
    driver_programs,
    driver_version,
    read_driver,
)

VERSION = "570.211.01"
#: ``/proc/driver/nvidia/version`` of an 8-GPU host on driver 570.
VERSION_570 = """\
NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.211.01  Tue Nov 25 16:36:36 UTC 2025
GCC version:  gcc version 11.5.0 20240719 (Red Hat 11.5.0-11) (GCC)
"""
#: The same file of a driver with NVIDIA's open kernel modules.
VERSION_OPEN = """\
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05  Release Build  (dvs-builder@U16-I3-B05-4-1)  Thu Sep 25 2025
GCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
"""
#: ``ldconfig -p`` of that host: its driver's libraries in /lib64, a stray library of driver 575
#: beside the 570 one, and its CUDA toolkits' libraries, which are none of the driver's.
LDCONFIG_570 = """\
	libnvrtc.so.12 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12
	libnvrtc.so.12 (libc6,x86-64) => /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12
	libnvoptix.so.1 (libc6,x86-64) => /lib64/libnvoptix.so.1
	libnvidia-wayland-client.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-wayland-client.so.570.211.01
	libnvidia-vksc-core.so.1 (libc6,x86-64) => /lib64/libnvidia-vksc-core.so.1
	libnvidia-tls.so.570.211.01 (libc6,x86-64, OS ABI: Linux 2.3.99) => /lib64/libnvidia-tls.so.570.211.01
	libnvidia-sandboxutils.so.1 (libc6,x86-64) => /lib64/libnvidia-sandboxutils.so.1
	libnvidia-rtcore.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-rtcore.so.570.211.01
	libnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => /lib64/libnvidia-ptxjitcompiler.so.1
	libnvidia-pkcs11-openssl3.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-pkcs11-openssl3.so.570.211.01
	libnvidia-opticalflow.so.1 (libc6,x86-64) => /lib64/libnvidia-opticalflow.so.1
	libnvidia-opencl.so.1 (libc6,x86-64) => /lib64/libnvidia-opencl.so.1
	libnvidia-nvvm.so.4 (libc6,x86-64) => /lib64/libnvidia-nvvm.so.4
	libnvidia-ngx.so.1 (libc6,x86-64) => /lib64/libnvidia-ngx.so.1
	libnvidia-ml.so.1 (libc6,x86-64) => /lib64/libnvidia-ml.so.1
	libnvidia-gtk3.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-gtk3.so.570.211.01
	libnvidia-gpucomp.so.575.57.08 (libc6,x86-64) => /lib64/libnvidia-gpucomp.so.575.57.08
	libnvidia-gpucomp.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-gpucomp.so.570.211.01
	libnvidia-glvkspirv.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-glvkspirv.so.570.211.01
	libnvidia-glsi.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-glsi.so.570.211.01
	libnvidia-glcore.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-glcore.so.570.211.01
	libnvidia-fbc.so.1 (libc6,x86-64) => /lib64/libnvidia-fbc.so.1
	libnvidia-encode.so.1 (libc6,x86-64) => /lib64/libnvidia-encode.so.1
	libnvidia-encode.so (libc6,x86-64) => /lib64/libnvidia-encode.so
	libnvidia-eglcore.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-eglcore.so.570.211.01
	libnvidia-egl-xlib.so.1 (libc6,x86-64) => /lib64/libnvidia-egl-xlib.so.1
	libnvidia-egl-xcb.so.1 (libc6,x86-64) => /lib64/libnvidia-egl-xcb.so.1
	libnvidia-egl-wayland.so.1 (libc6,x86-64) => /lib64/libnvidia-egl-wayland.so.1
	libnvidia-egl-gbm.so.1 (libc6,x86-64) => /lib64/libnvidia-egl-gbm.so.1
	libnvidia-container.so.1 (libc6,x86-64) => /lib64/libnvidia-container.so.1
	libnvidia-container-go.so.1 (libc6,x86-64) => /lib64/libnvidia-container-go.so.1
	libnvidia-cfg.so.1 (libc6,x86-64) => /lib64/libnvidia-cfg.so.1
	libnvidia-api.so.1 (libc6,x86-64) => /lib64/libnvidia-api.so.1
	libnvidia-allocator.so.1 (libc6,x86-64) => /lib64/libnvidia-allocator.so.1
	libnvcuvid.so.1 (libc6,x86-64) => /lib64/libnvcuvid.so.1
	libnvcuvid.so (libc6,x86-64) => /lib64/libnvcuvid.so
	libicudata.so.67 (libc6,x86-64) => /lib64/libicudata.so.67
	libicudata.so (libc6,x86-64) => /lib64/libicudata.so
	libgstcuda-1.0.so.0 (libc6,x86-64) => /lib64/libgstcuda-1.0.so.0
	libcudart.so.12 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
	libcudart.so.12 (libc6,x86-64) => /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12
	libcudadebugger.so.1 (libc6,x86-64) => /lib64/libcudadebugger.so.1
	libcuda.so.1 (libc6,x86-64) => /lib64/libcuda.so.1
	libcuda.so (libc6,x86-64) => /lib64/libcuda.so
	libGLX_nvidia.so.0 (libc6,x86-64) => /lib64/libGLX_nvidia.so.0
	libGLESv2_nvidia.so.2 (libc6,x86-64) => /lib64/libGLESv2_nvidia.so.2
	libGLESv2.so.2 (libc6,x86-64) => /lib64/libGLESv2.so.2
	libGLESv2.so (libc6,x86-64) => /lib64/libGLESv2.so
	libGLESv1_CM_nvidia.so.1 (libc6,x86-64) => /lib64/libGLESv1_CM_nvidia.so.1
	libGLESv1_CM.so.1 (libc6,x86-64) => /lib64/libGLESv1_CM.so.1
	libGLESv1_CM.so (libc6,x86-64) => /lib64/libGLESv1_CM.so
	libEGL_nvidia.so.0 (libc6,x86-64) => /lib64/libEGL_nvidia.so.0
"""
#: What the toolkit would give a container of that host for compute and utility: each library
#: with the names a container sees it under.
EXPECTED_570 = {
    "libnvidia-ml.so.570.211.01": ("libnvidia-ml.so.1", "libnvidia-ml.so.570.211.01"),
    "libnvidia-cfg.so.570.211.01": ("libnvidia-cfg.so.1", "libnvidia-cfg.so.570.211.01"),
    "libcuda.so.570.211.01": ("libcuda.so", "libcuda.so.1", "libcuda.so.570.211.01"),
    "libcudadebugger.so.570.211.01": ("libcudadebugger.so.1", "libcudadebugger.so.570.211.01"),
    "libnvidia-opencl.so.570.211.01": ("libnvidia-opencl.so.1", "libnvidia-opencl.so.570.211.01"),
    "libnvidia-gpucomp.so.570.211.01": ("libnvidia-gpucomp.so.570.211.01",),
    "libnvidia-ptxjitcompiler.so.570.211.01": ("libnvidia-ptxjitcompiler.so.1",
                                               "libnvidia-ptxjitcompiler.so.570.211.01"),
    "libnvidia-allocator.so.570.211.01": ("libnvidia-allocator.so.1", "libnvidia-allocator.so.570.211.01"),
    "libnvidia-pkcs11-openssl3.so.570.211.01": ("libnvidia-pkcs11-openssl3.so.570.211.01",),
    "libnvidia-nvvm.so.570.211.01": ("libnvidia-nvvm.so.4", "libnvidia-nvvm.so.570.211.01"),
}
_FULL = re.compile(r"\.\d+\.\d+\.\d+$")


def host_libs(tmp_path: Path, listing: str = LDCONFIG_570, *, links: dict[str, str] | None = None) -> str:
    """The libraries *listing* names, made under *tmp_path* as a driver installs them: a name with
    a full version is the file, any other a link to the name's ``.570.211.01`` file, or to the
    file *links* gives it. Return the listing with its paths under *tmp_path*."""
    lib = tmp_path / "lib64"
    lib.mkdir(exist_ok=True)
    for line in listing.splitlines():
        name, path = line.split()[0], line.rsplit("=> ", 1)[1]
        if not path.startswith("/lib64/"):
            continue
        if _FULL.search(name):
            (lib / name).touch()
            continue
        target = (links or {}).get(name) or re.sub(r"(\.\d+)*$", "", name) + f".{VERSION}"
        (lib / target).touch()
        (lib / name).unlink(missing_ok=True)
        (lib / name).symlink_to(target)
    return listing.replace("=> /lib64/", f"=> {lib}/")


def libraries(tmp_path: Path, listing: str = LDCONFIG_570, **kw) -> dict[str, tuple[str, ...]]:
    found = driver_libraries(host_libs(tmp_path, listing, **kw), VERSION, machine="x86_64")
    assert all(f.path.parent == tmp_path / "lib64" for f in found)
    return {f.path.name: f.names for f in found}


def test_the_recorded_hosts_driver_libraries_are_the_toolkits_for_compute_and_utility(tmp_path):
    """Each library is its 570 file, under the soname the loader asks for, its own name and, for
    libcuda, the name a compiler links against; the stray 575 gpucomp, the graphics and video
    libraries and the CUDA toolkits' libraries are not given."""
    assert libraries(tmp_path) == EXPECTED_570


def test_the_driver_version_is_the_kernel_modules():
    assert nvidia._VERSION.search(VERSION_570)[1] == VERSION
    assert nvidia._VERSION.search(VERSION_OPEN)[1] == "580.95.05"


def test_the_driver_version_is_read_from_proc_and_its_absence_refuses(tmp_path):
    (tmp_path / "driver/nvidia").mkdir(parents=True)
    with pytest.raises(DriverUnusable, match="cannot be read from"):
        driver_version(tmp_path / "nowhere")
    (tmp_path / "driver/nvidia/version").write_text("NVRM version: none\n")
    with pytest.raises(DriverUnusable, match="names no NVIDIA kernel module version"):
        driver_version(tmp_path)
    (tmp_path / "driver/nvidia/version").write_text(VERSION_570)
    assert driver_version(tmp_path) == VERSION


@pytest.mark.parametrize("drop, why", [
    ("libcuda.so", "lists no libcuda.so.570.211.01, the running NVIDIA driver's libcuda.so"),
    ("libnvidia-ml.so", "lists no libnvidia-ml.so.570.211.01, the running NVIDIA driver's libnvidia-ml.so"),
])
def test_a_host_without_cuda_or_nvml_of_its_driver_is_refused(tmp_path, drop, why):
    listing = "".join(line + "\n" for line in LDCONFIG_570.splitlines() if f"\t{drop}" not in line)
    with pytest.raises(DriverUnusable, match=re.escape(why)):
        libraries(tmp_path, listing)


def test_a_soname_listed_only_under_its_full_version_is_refused(tmp_path):
    """NVML's file is the driver's, but the soname its users load is not on the host."""
    listing = LDCONFIG_570.replace("\tlibnvidia-ml.so.1 (libc6,x86-64) => /lib64/libnvidia-ml.so.1",
                                   "\tlibnvidia-ml.so.570.211.01 (libc6,x86-64) => /lib64/libnvidia-ml.so.570.211.01")
    with pytest.raises(DriverUnusable, match="lists no libnvidia-ml.so.1, the soname"):
        libraries(tmp_path, listing)


@pytest.mark.parametrize("name", ["libcuda.so.1", "libcuda.so", "libnvidia-opencl.so.1", "libnvidia-nvvm.so.4"])
def test_a_name_the_loader_asks_for_that_resolves_to_another_drivers_file_is_refused(tmp_path, name):
    base = re.sub(r"(\.\d+)*$", "", name)
    with pytest.raises(DriverUnusable, match=re.escape(f"resolves to {base}.575.57.08, not the running NVIDIA "
                                                       f"driver's {base}.{VERSION}")):
        libraries(tmp_path, links={name: f"{base}.575.57.08"})


def test_a_listed_name_whose_file_is_gone_is_refused(tmp_path):
    listing = host_libs(tmp_path)
    (tmp_path / "lib64" / "libcuda.so.1").unlink()
    with pytest.raises(DriverUnusable, match="libcuda.so.1 on this host is .*, which resolves to a file that does not"):
        driver_libraries(listing, VERSION, machine="x86_64")


def test_the_drivers_file_in_two_places_is_refused(tmp_path):
    listing = host_libs(tmp_path)
    other = tmp_path / "usr-lib64"
    other.mkdir()
    (other / f"libcuda.so.{VERSION}").touch()
    listing += f"\tlibcuda.so.{VERSION} (libc6,x86-64) => {other}/libcuda.so.{VERSION}\n"
    with pytest.raises(DriverUnusable, match="lies in two places"):
        driver_libraries(listing, VERSION, machine="x86_64")


def test_libraries_of_another_abi_are_not_the_hosts(tmp_path):
    """The 32-bit libcuda a host may have beside its own is passed over, even one of another
    driver; a host of no ABI hls knows is refused."""
    listing = host_libs(tmp_path) + "\tlibcuda.so.1 (libc6) => /usr/lib/libcuda.so.1\n"
    assert {f.path.name for f in driver_libraries(listing, VERSION, machine="x86_64")} == set(EXPECTED_570)
    with pytest.raises(DriverUnusable, match="no ldconfig ABI of a ppc64le host"):
        driver_libraries(listing, VERSION, machine="ppc64le")


def test_the_drivers_programs_are_found_on_path_and_nvidia_smi_is_required(tmp_path):
    for name in ("nvidia-smi", "nvidia-debugdump"):
        (tmp_path / name).touch()
    found = driver_programs(lambda name: str(tmp_path / name) if (tmp_path / name).exists() else None)
    assert [(f.path, f.names) for f in found] == [(tmp_path / "nvidia-smi", ("nvidia-smi",)),
                                                  (tmp_path / "nvidia-debugdump", ("nvidia-debugdump",))]
    with pytest.raises(DriverUnusable, match="no nvidia-smi on PATH"):
        driver_programs(lambda name: None)


def test_read_driver_records_the_running_drivers_files_and_ldconfig_failing_refuses(tmp_path):
    proc = tmp_path / "proc"
    (proc / "driver/nvidia").mkdir(parents=True)
    (proc / "driver/nvidia/version").write_text(VERSION_570)
    listing = host_libs(tmp_path)
    (tmp_path / "nvidia-smi").touch()

    def run(argv, **kw):
        assert argv == ["/sbin/ldconfig", "-p"]
        return subprocess.CompletedProcess(argv, 0, f"{len(listing.splitlines())} libs found in cache\n" + listing, "")

    driver = read_driver(proc=proc, run=run, which=lambda name: str(tmp_path / name) if name == "nvidia-smi" else None,
                         machine="x86_64")
    assert driver.version == VERSION and {f.path.name: f.names for f in driver.libraries} == EXPECTED_570
    assert driver.programs == (DriverFile(path=tmp_path / "nvidia-smi", names=("nvidia-smi",)),)
    binds = dict((target, source) for source, target in driver.binds())
    assert binds["/usr/local/nvidia/lib64/libcuda.so.1"] == tmp_path / "lib64" / f"libcuda.so.{VERSION}"
    assert binds["/usr/local/nvidia/bin/nvidia-smi"] == tmp_path / "nvidia-smi"
    assert NvidiaDriver.model_validate_json(driver.model_dump_json()) == driver

    def failing(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "", "")

    with pytest.raises(DriverUnusable, match="ldconfig -p exited 1"):
        read_driver(proc=proc, run=failing, which=lambda name: None, machine="x86_64")


def test_a_roles_devices_are_its_cards_and_the_drivers_nodes(tmp_path):
    for name in ("nvidia0", "nvidia3", "nvidia5", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-nvswitch0"):
        (tmp_path / name).touch()
    assert driver_devices([5, 3], dev=tmp_path) == [(tmp_path / n, f"/dev/{n}") for n in
                                                    ("nvidia5", "nvidia3", "nvidiactl", "nvidia-uvm",
                                                     "nvidia-uvm-tools")]
    (tmp_path / "nvidia-modeset").touch()
    assert driver_devices([0], dev=tmp_path)[-1] == (tmp_path / "nvidia-modeset", "/dev/nvidia-modeset")
    with pytest.raises(DriverUnusable, match="nvidia7 does not exist"):
        driver_devices([7], dev=tmp_path)
    (tmp_path / "nvidia-uvm").unlink()
    with pytest.raises(DriverUnusable, match=r"nvidia-uvm does not exist \(the host makes it when a CUDA program"):
        driver_devices([0], dev=tmp_path)


# -- the gate's search paths -----------------------------------------------------------------


def gate_env(tmp_path: Path, *, image_env: dict[str, str], compat: list[str] = (), role_env: str = "",
             driver: str = VERSION) -> dict[str, str]:
    """The environment the gate gives its command with ``--nvidia driver``, in a container whose
    image sets *image_env* and holds the CUDA forward-compatibility libraries *compat*: the gate
    run by the host's sh with its paths moved under *tmp_path*."""
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    compat_dir = tmp_path / "compat"
    compat_dir.mkdir()
    for name in compat:
        (compat_dir / name).touch()
    (gate_dir / "env").write_text(role_env)
    script = (GATE.read_text().replace("/.hls-gate/", f"{gate_dir}/")
              .replace("/usr/local/cuda/compat/libcuda", f"{compat_dir}/libcuda"))
    (gate_dir / "gate").write_text(script)
    out = tmp_path / "env.out"
    subprocess.run(["/bin/sh", str(gate_dir / "gate"), "--nvidia", driver, "--", "/bin/sh", "-c",
                    f"env > {out}"], env=image_env, check=True, timeout=30)
    return dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_the_gate_puts_the_drivers_libraries_after_the_images_own(tmp_path):
    env = gate_env(tmp_path, image_env={"PATH": "/usr/bin:/bin",
                                        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/x"})
    assert env["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib:/opt/x:/usr/local/nvidia/lib64"
    assert env["PATH"] == "/usr/bin:/bin:/usr/local/nvidia/bin"


def test_the_gate_gives_an_image_without_a_library_path_the_drivers(tmp_path):
    env = gate_env(tmp_path, image_env={"PATH": "/bin"})
    assert env["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64" and env["PATH"] == "/bin:/usr/local/nvidia/bin"


@pytest.mark.parametrize("compat, ahead", [
    (["libcuda.so.580.82.07", "libcuda.so.1"], True),
    (["libcuda.so.570.86.10"], False),
    (["libcuda.so.565.57.01"], False),
    (["libcuda.so.580.82.07", "libcuda.so.590.10.01"], False),
    ([], False),
])
def test_the_images_forward_compatibility_libcuda_goes_ahead_of_an_older_major_driver(tmp_path, compat, ahead):
    """As the toolkit's enable-cuda-compat hook: exactly one libcuda.so.X.Y.Z, of a major version
    above the driver's."""
    env = gate_env(tmp_path, image_env={"PATH": "/usr/local/nvidia/bin:/bin"}, compat=compat)
    lib = "/usr/local/cuda/compat:/usr/local/nvidia/lib64" if ahead else "/usr/local/nvidia/lib64"
    assert env["LD_LIBRARY_PATH"] == lib and env["PATH"] == "/usr/local/nvidia/bin:/bin"


def test_a_roles_own_library_path_replaces_the_gates(tmp_path):
    env = gate_env(tmp_path, image_env={"PATH": "/bin"}, role_env="LD_LIBRARY_PATH='/mine'\n")
    assert env["LD_LIBRARY_PATH"] == "/mine"


def test_the_gate_refuses_a_driver_version_that_is_not_one(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as e:
        gate_env(tmp_path, image_env={}, driver="5;rm")
    assert e.value.returncode == 2
