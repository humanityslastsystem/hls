from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from hls.substrate.apptainer.argv import HOST_FILES, START_MARKER, ApptainerRuntime, passed_by_cleanenv
from hls.substrate.container import ContainerSpec, Mount
from tests.substrate import fake_apptainer
from tests.substrate.sandboxes import stand_in

ROOT = Path("/node/hls-1234110/sandboxes/" + "ab" * 32)
#: Where the step wrapper hands a container its env file.
ENV = "/dev/fd/3"
WORK = Path("/node/hls-1234110/roles/artifact/container")


def _spec(**changes) -> ContainerSpec:
    """A spec; an image given as a path is the extracted directory of a stand-in pinned image."""
    fields = {"image": ROOT, "command": ("python", "/workspace/server.py", "--port", "8000"), "workdir": WORK}
    fields.update(changes)
    if isinstance(fields["image"], Path):
        fields["image"] = stand_in(fields["image"])
    return ContainerSpec(**fields)


def _root(tmp_path: Path) -> Path:
    """An extracted directory the fake apptainer can run from."""
    root = tmp_path / "image"
    root.mkdir(exist_ok=True)
    return root


def test_the_argv_spells_out_every_isolation_flag_and_binds_only_what_is_declared():
    spec = _spec(mounts=(Mount(Path("/node/hls-1234110/weights/q"), "/data/weights"),
                         Mount(Path("/node/hls-1234110/roles/artifact/out"), "/out", read_only=False)),
                 gpus=True, cwd="/workspace")
    argv = ApptainerRuntime(apptainer="/opt/apptainer/bin/apptainer").argv(spec, env_file="/dev/fd/3")
    assert argv == [
        "/opt/apptainer/bin/apptainer", "exec", "--containall", "--cleanenv", "--pid",
        "--no-mount", "hostfs,bind-paths,cwd", "--workdir", str(WORK), "--nv", "--cwd", "/workspace",
        "--bind", "/etc/resolv.conf:/etc/resolv.conf:ro", "--bind", "/etc/hosts:/etc/hosts:ro",
        "--bind", "/node/hls-1234110/weights/q:/data/weights:ro",
        "--bind", "/node/hls-1234110/roles/artifact/out:/out",
        "--env-file", "/dev/fd/3", str(ROOT),
        "/bin/sh", "-c", 'exec 3<&-; printf "%s\\n" "$0" >&2 && exec "$@"', START_MARKER,
        "python", "/workspace/server.py", "--port", "8000",
    ]


def test_apptainer_runs_the_spawner_confinement_as_any_container():
    """Apptainer confines no container against the namespaces and mounts the build sandbox's
    spawner makes, so a spec that asks for the ``spawner`` confinement runs on the same command
    line as one that does not."""
    runtime = ApptainerRuntime()
    plain, spawner = _spec(cwd="/workspace"), _spec(cwd="/workspace", confinement="spawner")
    assert runtime.argv(spawner, env_file=ENV) == runtime.argv(plain, env_file=ENV)
    with pytest.raises(ValueError, match="default or spawner"):
        _spec(confinement="unconfined")


def test_without_gpus_or_a_cwd_those_flags_are_absent():
    argv = ApptainerRuntime().argv(_spec(), env_file=ENV)
    assert argv[0] == "apptainer"
    for flag in ("--nv", "--cwd", "--pwd"):
        assert flag not in argv
    assert argv[argv.index(str(ROOT)) + 1:argv.index(str(ROOT)) + 3] == ["/bin/sh", "-c"]


def test_every_container_takes_an_env_file():
    # start_container always hands one over, which may hold no variable; there is no other case.
    with pytest.raises(ValueError, match="env_file"):
        ApptainerRuntime().argv(_spec(), env_file=None)


def test_the_host_files_are_bound_read_only_before_any_declared_mount():
    argv = ApptainerRuntime().argv(_spec(mounts=(Mount(Path("/node/w"), "/data/weights"),)), env_file=ENV)
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert binds == [f"{f}:{f}:ro" for f in HOST_FILES] + ["/node/w:/data/weights:ro"]


def test_the_runtime_names_its_start_marker():
    assert ApptainerRuntime().start_marker == START_MARKER == "hls-container-started"


# What apptainer's bind parser and option handling would read differently.
_UNEXACT = "cannot be passed to apptainer exactly"
_SOURCE_NORMAL = "source must be an absolute path in normal form"
_TARGET_NORMAL = "target must be an absolute container path in normal form"


@pytest.mark.parametrize("changes, match", [
    ({"image": Path("images/x.sif")}, "image must be an absolute path"),
    ({"workdir": Path("roles/artifact")}, "workdir must be an absolute path"),
    ({"command": ()}, "no command"),
    ({"cwd": "workspace"}, "cwd must be an absolute container path"),
    ({"mounts": (Mount(Path("weights"), "/data/weights"),)}, "source must be an absolute path"),
    ({"mounts": (Mount(Path("/node/scratch/../w"), "/data"),)}, _SOURCE_NORMAL),
    ({"mounts": (Mount(Path("//node/w"), "/data"),)}, _SOURCE_NORMAL),
    ({"mounts": (Mount(Path("/node/w"), "data/weights"),)}, _TARGET_NORMAL),
    ({"mounts": (Mount(Path("/node/w"), "/data/../etc"),)}, _TARGET_NORMAL),
    ({"mounts": (Mount(Path("/node/w"), "/data/weights/"),)}, _TARGET_NORMAL),
    ({"mounts": (Mount(Path("/node/w"), "//data"),)}, _TARGET_NORMAL),
    ({"mounts": (Mount(Path("/node/a"), "/data"), Mount(Path("/node/b"), "//data"))}, _TARGET_NORMAL),
    ({"mounts": (Mount(Path("/node/w"), "/"),)}, "cannot mount over /"),
    ({"mounts": (Mount(Path("/node/a"), "/data"), Mount(Path("/node/b"), "/data"))}, "two mounts on /data"),
    ({"mounts": (Mount(Path("/node/a:b"), "/data"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a,b"), "/data"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a"), "/data:rw"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a"), "/data,rw"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a^b"), "/data"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a"), "/data^b"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a "), "/data"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a"), "/data "),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a\nb"), "/data"),)}, _UNEXACT),
    ({"mounts": (Mount(Path("/node/a"), "/data\\x"),)}, _UNEXACT),
    ({"mounts": (Mount(Path('/node/a"b'), "/data"),)}, _UNEXACT),
])
def test_a_spec_the_runtime_cannot_express_exactly_is_refused(changes, match):
    with pytest.raises(ValueError, match=match):
        ApptainerRuntime().argv(_spec(**changes), env_file=ENV)


@pytest.mark.parametrize("field, normal", [("image", "image must be an absolute path in normal form"),
                                           ("workdir", "workdir must be an absolute path in normal form"),
                                           ("cwd", "cwd must be an absolute container path in normal form")])
@pytest.mark.parametrize("path", ["/node/../etc/x", "//node/x"])
def test_an_image_workdir_or_cwd_not_in_normal_form_is_refused(field, normal, path):
    value = path if field == "cwd" else Path(path)
    with pytest.raises(ValueError, match=normal):
        ApptainerRuntime().argv(_spec(**{field: value}), env_file=ENV)


@pytest.mark.parametrize("cwd", ["/workspace/", "/workspace/./src"])
def test_a_cwd_with_a_trailing_slash_or_a_dot_component_is_refused(cwd):
    with pytest.raises(ValueError, match="cwd must be an absolute container path in normal form"):
        ApptainerRuntime().argv(_spec(cwd=cwd), env_file=ENV)


@pytest.mark.parametrize("field", ["image", "workdir", "cwd"])
@pytest.mark.parametrize("path", ["/node/a b", "/node/a,b", "/node/a:b", "/node/a^b", "/node/a\nb", "/node/a "])
def test_an_image_workdir_or_cwd_apptainer_would_read_differently_is_refused(field, path):
    value = path if field == "cwd" else Path(path)
    with pytest.raises(ValueError, match=re.escape(f"{field} {path!r} cannot be passed to apptainer exactly")):
        ApptainerRuntime().argv(_spec(**{field: value}), env_file=ENV)


def test_a_cwd_of_the_container_root_is_accepted():
    argv = ApptainerRuntime().argv(_spec(cwd="/"), env_file=ENV)
    assert argv[argv.index("--cwd") + 1] == "/"


def test_the_host_root_cannot_be_a_mount_source():
    with pytest.raises(ValueError, match="cannot mount the host root"):
        ApptainerRuntime().argv(_spec(mounts=(Mount(Path("/"), "/host"),)), env_file=ENV)


@pytest.mark.parametrize("host_file", HOST_FILES)
def test_a_mount_over_a_host_file_the_runtime_binds_is_refused(host_file):
    with pytest.raises(ValueError, match=f"{host_file} is bound by the runtime"):
        ApptainerRuntime().argv(_spec(mounts=(Mount(Path("/node/h"), host_file),)), env_file=ENV)
    with pytest.raises(ValueError, match=_TARGET_NORMAL):
        ApptainerRuntime().argv(_spec(mounts=(Mount(Path("/node/h"), "/" + host_file),)), env_file=ENV)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("path", ["/proc", "/proc/1/root", "/sys", "/sys/fs/cgroup", "/dev", "/dev/shm"])
def test_a_mount_source_or_target_under_proc_sys_or_dev_is_refused(side, path):
    mount = Mount(Path(path), "/data") if side == "source" else Mount(Path("/node/w"), path)
    with pytest.raises(ValueError, match=f"{path} is under /(proc|sys|dev)"):
        ApptainerRuntime().argv(_spec(mounts=(mount,)), env_file=ENV)


def test_a_path_that_only_starts_with_proc_sys_or_dev_is_not_refused():
    mounts = (Mount(Path("/node/devices"), "/devices"), Mount(Path("/processing"), "/system"))
    argv = ApptainerRuntime().argv(_spec(mounts=mounts), env_file=ENV)
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert binds[len(HOST_FILES):] == ["/node/devices:/devices:ro", "/processing:/system:ro"]


@pytest.mark.parametrize("env_file", ["/dev/fd/3", "/node/hls-1234110/roles/artifact/env.sh"])
def test_a_plain_absolute_env_file_path_is_accepted(env_file):
    argv = ApptainerRuntime().argv(_spec(), env_file=env_file)
    assert argv[argv.index("--env-file") + 1] == env_file


@pytest.mark.parametrize("env_file", ["/dev/fd/3,4", "/run/env file", "/run/env:x", "//dev/fd/3", "/run/../env",
                                      "run/env", "/run/env/", "/run/env^"])
def test_an_env_file_path_apptainer_would_read_differently_is_refused(env_file):
    with pytest.raises(ValueError, match="env_file"):
        ApptainerRuntime().argv(_spec(), env_file=env_file)


# The fake apptainer: what later tests use in place of the real binary.

def _run(tmp_path: Path, spec: ContainerSpec, *, env_file: Path | None = None, env: dict | None = None,
         **kwargs) -> subprocess.CompletedProcess:
    """Run *spec* under the fake apptainer with *env_file*, or an empty one."""
    fake = fake_apptainer.install(tmp_path / "bin")
    env_file = env_file if env_file is not None else _env_file(tmp_path / "empty.env")
    argv = ApptainerRuntime(apptainer=str(fake)).argv(spec, env_file=str(env_file))
    return subprocess.run(argv, env={"PATH": os.environ["PATH"], **(env or {})}, capture_output=True, text=True,
                          timeout=30, **kwargs)


def _env_file(path: Path, **env: str) -> Path:
    path.write_text("".join(f"{k}='{v}'\n" for k, v in env.items()))
    return path


def test_the_fake_prints_the_marker_then_runs_the_command_with_the_env_file_only(tmp_path):
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd",
                 command=("sh", "-c", 'echo "token=${TOKEN:-unset} leak=${LEAK:-unset}"; exit 3'))
    done = _run(tmp_path, spec, env_file=_env_file(tmp_path / "env", TOKEN="s3cr3t-value"), env={"LEAK": "host"})
    assert done.returncode == 3
    assert done.stderr.splitlines()[0] == START_MARKER
    assert done.stdout.strip() == "token=s3cr3t-value leak=unset"


# Shell texts that open descriptor 3 the way the step wrapper hands the env file over, then run the fake.
# "$0" is the env file (a file) or its text (a pipe).
FILE_ON_FD3 = 'exec 3<"$0"; exec "$@"'
PIPE_ON_FD3 = 'printf "%s\\n" "$0" | { exec 3<&0 </dev/null; exec "$@"; }'


def _run_with_fd3(tmp_path: Path, spec: ContainerSpec, script: str, source: str) -> subprocess.CompletedProcess:
    """Run the fake under the shell text *script*, which opens descriptor 3 in the child only and then runs "$@".
    "$0" in the script is *source*."""
    fake = fake_apptainer.install(tmp_path / "bin")
    argv = ApptainerRuntime(apptainer=str(fake)).argv(spec, env_file="/dev/fd/3")
    return subprocess.run(["/bin/sh", "-c", script, source, *argv], env={"PATH": os.environ["PATH"]},
                          capture_output=True, text=True, timeout=30)


def test_the_fake_reads_its_env_file_from_an_inherited_descriptor(tmp_path):
    env_file = _env_file(tmp_path / "env", TOKEN="from-the-fd-1")
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd", command=("sh", "-c", "echo $TOKEN"))
    done = _run_with_fd3(tmp_path, spec, FILE_ON_FD3, str(env_file))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "from-the-fd-1"


def test_the_fake_drains_an_env_file_that_arrives_as_a_pipe_on_descriptor_3(tmp_path):
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd", command=("sh", "-c", "echo $TOKEN"))
    done = _run_with_fd3(tmp_path, spec, PIPE_ON_FD3, "TOKEN='from-the-pipe'")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "from-the-pipe"


def test_the_command_does_not_inherit_the_env_file_descriptor(tmp_path):
    env_file = _env_file(tmp_path / "env", TOKEN="s3cr3t-value")
    probe = 'if [ -e /proc/self/fd/3 ]; then echo "fd3 open: $(readlink /proc/self/fd/3)"; else echo fd3 closed; fi'
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd",
                 command=("sh", "-c", f"echo $TOKEN; {probe}"))
    done = _run_with_fd3(tmp_path, spec, FILE_ON_FD3, str(env_file))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["s3cr3t-value", "fd3 closed"]


def test_the_command_does_not_run_when_the_marker_cannot_be_written(tmp_path):
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd", command=("sh", "-c", "touch ran"))
    fake = fake_apptainer.install(tmp_path / "bin")
    argv = ApptainerRuntime(apptainer=str(fake)).argv(spec, env_file=str(_env_file(tmp_path / "empty.env")))
    with open("/dev/full", "w") as full:
        done = subprocess.run(argv, env={"PATH": os.environ["PATH"]}, stderr=full, timeout=30, cwd=tmp_path)
    assert done.returncode != 0
    assert not (tmp_path / "ran").exists()


def test_the_fake_forwards_apptainerenv_variables_and_backs_home_with_the_workdir(tmp_path):
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd",
                 command=("sh", "-c", 'echo "$CUDA_VISIBLE_DEVICES"; touch "$HOME/written"; echo "$HOME"'))
    done = _run(tmp_path, spec, env={"APPTAINERENV_CUDA_VISIBLE_DEVICES": "GPU-1,GPU-2"})
    assert done.returncode == 0, done.stderr
    cuda, home = done.stdout.split()
    assert cuda == "GPU-1,GPU-2" and Path(home).is_relative_to(tmp_path / "wd") and (Path(home) / "written").exists()


def test_the_fake_passes_term_and_the_proxy_variables_through_cleanenv_like_apptainer(tmp_path):
    passed = {"TERM": "xterm", "http_proxy": "http://p:1", "HTTP_PROXY": "http://p:2", "https_proxy": "http://p:3",
              "HTTPS_PROXY": "http://p:4", "no_proxy": "n1", "NO_PROXY": "n2", "all_proxy": "a1", "ALL_PROXY": "a2",
              "ftp_proxy": "f1", "FTP_PROXY": "f2"}
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd",
                 command=("sh", "-c", 'for v in $(env | cut -d= -f1 | sort); do echo "$v"; done'))
    done = _run(tmp_path, spec, env={**passed, "Http_Proxy": "mixed-case", "PROXY": "x", "LEAK": "host"})
    assert done.returncode == 0, done.stderr
    names = set(done.stdout.split()) - {"PWD", "SHLVL", "_", "OLDPWD"}
    assert names == {*passed, "PATH", "HOME"}


def test_every_variable_cleanenv_passes_is_one_the_wrapper_removes_from_a_container_step():
    assert all(passed_by_cleanenv(name) for name in fake_apptainer._PASSED_BY_CLEANENV)
    assert all(passed_by_cleanenv(name) for name in ("socks_proxy", "Http_Proxy", "RSYNC_PROXY"))
    assert not any(passed_by_cleanenv(name) for name in ("PROXY", "TERMINFO", "PATH", "HF_TOKEN", "proxy_url"))


def test_a_runtime_failure_prints_fatal_and_no_marker(tmp_path):
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd", command=("sh", "-c", "touch ran"))
    done = _run(tmp_path, spec, env={"FAKE_APPTAINER_FAIL": "1"}, cwd=tmp_path)
    assert done.returncode == 255
    assert done.stderr.startswith("FATAL:") and START_MARKER not in done.stderr
    assert not (tmp_path / "ran").exists()


def test_the_fake_refuses_an_argv_without_the_isolation_flags(tmp_path):
    fake = fake_apptainer.install(tmp_path / "bin")
    argv = ApptainerRuntime(apptainer=str(fake)).argv(_spec(image=_root(tmp_path), workdir=tmp_path / "wd"),
                                                      env_file=str(_env_file(tmp_path / "empty.env")))
    for flag in ("--containall", "--cleanenv", "--pid"):
        broken = [a for a in argv if a != flag]
        done = subprocess.run(broken, capture_output=True, text=True, timeout=30)
        assert done.returncode == 2 and flag in done.stderr
    no_mount = argv.index("--no-mount")
    broken = argv[:no_mount] + argv[no_mount + 2:]
    done = subprocess.run(broken, capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "--no-mount" in done.stderr


def test_the_fake_records_what_it_was_asked_to_bind(tmp_path):
    record = tmp_path / "record.json"
    spec = _spec(image=_root(tmp_path), workdir=tmp_path / "wd", command=("true",), gpus=True,
                 mounts=(Mount(tmp_path / "w", "/data/weights"),))
    done = _run(tmp_path, spec, env_file=_env_file(tmp_path / "env", TOKEN="s3cr3t-value"),
                env={"FAKE_APPTAINER_RECORD": str(record)})
    assert done.returncode == 0, done.stderr
    seen = fake_apptainer.read_record(record)
    assert seen["image"] == str(tmp_path / "image") and seen["nv"] is True
    assert f"{tmp_path / 'w'}:/data/weights:ro" in seen["binds"]
    assert seen["env_names"] == ["TOKEN"] and "s3cr3t-value" not in record.read_text()
    assert seen["command"] == ["true"]
