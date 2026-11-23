from __future__ import annotations

import ast
import asyncio
import dataclasses
import errno
import inspect
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from hls.egress import proxy as proxy_mod
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.egress.proxy import EgressProxy
from hls.settings import SiteSettings
from hls.substrate.apptainer import egress
from hls.substrate import seal as seal_mod
from hls.substrate.apptainer.argv import START_MARKER, ApptainerRuntime
from hls.substrate.apptainer.egress import CLOSED_DNS_VAR, SEAL_SCRIPT, seal_env, seal_prefix, seal_script
from hls.substrate.seal import (EGRESS_FWMARK, EGRESS_INDEX_PORT, EGRESS_TRANSPARENT_PORT, INTERCEPTOR_PORTS,
                                UPSTREAM_PROXY_VAR, IndexStatus, InterceptorPortTaken, OversizedPage, PayloadEnding,
                                client_env, index_status, ingress_port, namespace_ports, payload_ended_at, payload_ending,
                                policy_refusals, refuse_interceptor_port, seal_files)
from hls.substrate.container import ContainerSpec, SealPlan
from hls.substrate.exits import SEAL_REFUSED_EXIT, SEAL_UNBOUND_EXIT
from hls.substrate.role_exec import ENV_FD
from tests.substrate import fake_pasta
from tests.substrate.sandboxes import stand_in

#: What a build seal's interceptor enforces: the air gap, its index refusing the harness's deny list.
SEAL = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": list(DENIED_PACKAGES)}}))
#: Its JSON, as the seal hands it to the interceptor.
POLICY = SEAL.model_dump_json()
#: The deny list the producer of POLICY declares, which a seal with an index carries.
DENY = tuple(DENIED_PACKAGES)


def _free_ports() -> tuple[int, int]:
    """Two different free loopback ports for the interceptor's listeners. The first stays bound
    while the second is picked: two picks one after the other can return the same port."""
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def _seal(tmp_path: Path, plan: list[str], *, ingress: str = "", index: bool = True, payload_rc: int = 0,
          payload_sec: float = 0.0, proxy: str = "serve", started_file: bool = True, stale_start: bool = False,
          env_fd: str = "", env_text: str | None = None, step_input: str = "",
          step_env: dict[str, str] | None = None, payload_first: str = "") -> dict:
    """Run the real seal script around a payload with fake pasta, iptables and interpreter, and
    return what happened. The payload runs the shell text *payload_first*, reads its standard
    input and, when it has one, its env file on descriptor 3, sleeps *payload_sec*, notes the
    run and exits *payload_rc*. With *env_text*, the step hands the seal a pipe holding it on
    descriptor 3, as the step wrapper does, and the fakes note whether they hold that pipe.
    *step_env* adds to the step's variables."""
    bin_dir = fake_pasta.install(tmp_path / "bin")
    (tmp_path / "plan").write_text("".join(f"{line}\n" for line in plan))
    runs = tmp_path / "runs"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HLS_PASTA": str(bin_dir / "pasta"),
           "HLS_SEAL_TOOL_PATH": str(bin_dir),  # where the runner tells the seal its iptables lives
           "FAKE_PASTA_PLAN": str(tmp_path / "plan"), "FAKE_PASTA_LOG": str(tmp_path / "pasta.log"),
           "FAKE_PYTHON_COUNTER": str(tmp_path / "counter"), "FAKE_INTERCEPTOR": proxy,
           "HLS_PROXY_STATUS_FILE": str(tmp_path / "status.json"), "HLS_SEAL_ENDED_FILE": str(tmp_path / "ended")}
    if started_file:
        env["HLS_SEAL_STARTED_FILE"] = str(tmp_path / "started")
        if stale_start:
            (tmp_path / "started").touch()
    else:
        (tmp_path / "tmpdir").mkdir()
        env["TMPDIR"] = str(tmp_path / "tmpdir")
    if ingress:
        env["HLS_INGRESS_PORT_FILE"] = str(tmp_path / "port")
    env.update(step_env or {})
    if env_fd or env_text is not None:
        env["HLS_SEAL_ENV_FD"] = env_fd or "3"
    transparent, index_port = _free_ports()
    payload = (f"{payload_first}cat > {tmp_path / 'stdin'}; readlink /proc/self/fd/0 > {tmp_path / 'stdin_is'}; "
               f"if [ -e /dev/fd/3 ]; then cat /dev/fd/3 > {tmp_path / 'env'}; fi; "
               f"sleep {payload_sec}; echo ran >> {runs}; exit {payload_rc}")
    argv = ["bash", "-c", seal_script(), "hls-seal", str(transparent),
            str(index_port) if index else "", str(EGRESS_FWMARK), str(bin_dir / "python"), POLICY, ingress,
            "sh", "-c", payload]
    pass_fds: tuple[int, ...] = ()
    if env_text is not None:
        read_end, write_end = os.pipe()
        os.write(write_end, env_text.encode())
        os.close(write_end)
        env.update(FAKE_PIPE_INODE=str(os.fstat(read_end).st_ino), FAKE_PIPE_HOLDERS=str(tmp_path / "holders"))
        # The pipe goes on descriptor 3, as the step wrapper puts it.
        argv = ["bash", "-c", f'exec 3<&{read_end} {read_end}<&-; exec "$@"', "step", *argv]
        pass_fds = (read_end,)
    try:
        proc = subprocess.run(argv, env=env, pass_fds=pass_fds, input=step_input, capture_output=True, text=True,
                              timeout=120)
    finally:
        for fd in pass_fds:
            os.close(fd)
    counter = tmp_path / "counter"

    def text(name: str) -> str | None:
        return (tmp_path / name).read_text() if (tmp_path / name).exists() else None

    return {"rc": proc.returncode, "stderr": proc.stderr, "calls": fake_pasta.calls(tmp_path / "pasta.log"),
            "runs": len(runs.read_text().splitlines()) if runs.exists() else 0,
            "scans": int(counter.read_text()) if counter.exists() else 0,
            "started": (tmp_path / "started").exists(),
            "port_file": (tmp_path / "port").read_text().strip() if (tmp_path / "port").exists() else None,
            "status": index_status(tmp_path / "status.json"), "ended": payload_ending(seal_files(tmp_path)),
            "stdin": text("stdin"), "stdin_is": (text("stdin_is") or "").strip(), "env_seen": text("env"),
            "holders": (text("holders") or "").split()}


def _wait_for(path: Path) -> None:
    for _ in range(300):
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def _forward(flags: list[str]) -> str:
    """The host port of pasta's ``-t 0.0.0.0/<host>:<guest>`` forward."""
    m = re.fullmatch(r"0\.0\.0\.0/(\d+):8000", flags[flags.index("-t") + 1])
    assert m is not None, flags
    return m.group(1)


def test_the_seal_script_parses():
    subprocess.run(["bash", "-n", str(SEAL_SCRIPT)], check=True, capture_output=True)


@pytest.mark.parametrize("ingress", ["", "8000"], ids=["build-seal", "served-seal"])
def test_a_payload_that_fills_the_disk_leaves_its_ending_and_the_status_whole(tmp_path, ingress):
    """A payload can fill the disk its seal's files are on. Both are made at full size before
    the payload starts and only written over afterwards, so its ending and the interceptor's
    heartbeat still land, whole. Here the payload leaves the seal's directory unable to take a
    new file, a rename or a removal once it starts, as a full disk would refuse the new blocks
    those need; every size stays as it was made."""
    seal = tmp_path / "seal"
    seal.mkdir()
    files = seal_files(seal)
    env = {"HLS_PROXY_STATUS_FILE": str(files.status), "HLS_SEAL_ENDED_FILE": str(files.ended),
           "HLS_SEAL_STARTED_FILE": str(files.started), **({"HLS_INGRESS_PORT_FILE": str(files.port)} if ingress else {})}
    try:
        out = _seal(tmp_path, ["run"], ingress=ingress, payload_rc=3, step_env=env,
                    payload_first=f"chmod 500 {seal}; sleep 1; ")
    finally:
        seal.chmod(0o700)
    assert out["rc"] == 3, out["stderr"]
    assert payload_ending(files) == PayloadEnding(exit_code=3, interceptor_alive=True)
    status = index_status(files.status)
    assert status is not None and status.heartbeat >= files.started.stat().st_mtime
    sizes = {p.name: p.stat().st_size for p in seal.iterdir() if p.name != "port"}
    assert sizes == {"status.json": proxy_mod.STATUS_FILE_BYTES, "started": 0, "ended": 512, "ended.pending": 512}


@pytest.mark.parametrize("plan", [["run"], ["fail:1"]], ids=["ran", "never-started"])
def test_the_payloads_end_is_when_its_seal_wrote_its_ending(tmp_path, plan):
    """The seal writes the payload's ending as soon as the payload ends, into a file no container
    sees, so the file's time is when the payload ended: after its start and its second of work,
    before the seal returned. A payload that never started has no ending, and no time."""
    seal = tmp_path / "seal"
    seal.mkdir()
    files = seal_files(seal)
    env = {"HLS_PROXY_STATUS_FILE": str(files.status), "HLS_SEAL_ENDED_FILE": str(files.ended),
           "HLS_SEAL_STARTED_FILE": str(files.started)}
    out = _seal(tmp_path, plan, payload_rc=3, payload_sec=1, step_env=env)
    returned = time.time()
    at = payload_ended_at(files)
    if plan == ["fail:1"]:
        assert out["rc"] == 1 and payload_ending(files) is None and at is None
        return
    assert out["rc"] == 3, out["stderr"]
    assert at is not None and at - files.started.stat().st_mtime >= 0.99 and at <= returned


def test_a_working_seal_runs_the_payload_once_and_says_it_started(tmp_path):
    out = _seal(tmp_path, ["run"])
    assert (out["rc"], out["runs"], out["started"]) == (0, 1, True), out["stderr"]
    assert len(out["calls"]) == 1
    status = out["status"]
    assert (status.total, status.mid_stream, status.oversized, status.policy_refused, status.fetches_refused) == (
        0, 0, (), (), 0)
    assert abs(status.heartbeat - time.time()) < 60


def test_the_build_seal_forwards_no_port_either_way(tmp_path):
    """pasta forwards every port the namespace binds, on every host address, in a direction its
    command line leaves unnamed, so a build seal that named no inbound forward would let any host
    that reaches the node connect to a build's listener, over TCP and UDP. The build seal names all
    four directions, and forwards nothing. Nor does its namespace reach the host's loopback
    through its gateway's address (``--no-map-gw``), nor does pasta relay its ICMP echo
    (``--no-icmp``)."""
    out = _seal(tmp_path, ["run"])
    assert out["rc"] == 0, out["stderr"]
    assert out["calls"] == [["--config-net", "-f", "-t", "none", "-u", "none", "-T", "none", "-U", "none",
                             "--no-map-gw", "--no-icmp"]]
    assert fake_pasta.forwards(out["calls"][0]) == fake_pasta.NO_FORWARDS


def test_the_runtime_seal_forwards_its_served_tcp_port_alone(tmp_path):
    """The runtime seal forwards one TCP port in, a free host port to the served port, and no UDP
    port in and nothing out to the host's loopback."""
    out = _seal(tmp_path, ["run"], ingress="8000", index=False)
    assert out["rc"] == 0, out["stderr"]
    assert out["calls"] == [["--config-net", "-f", "-t", f"0.0.0.0/{out['port_file']}:8000", "-u", "none",
                             "-T", "none", "-U", "none", "--no-map-gw", "--no-icmp"]]


def test_an_attempt_after_a_host_port_race_names_every_forward_too(tmp_path):
    out = _seal(tmp_path, ["fail:1", "run"], ingress="8000", index=False)
    assert out["rc"] == 0, out["stderr"]
    first, second = out["calls"]
    assert fake_pasta.forwards(first) == {**fake_pasta.NO_FORWARDS, "-t": [f"0.0.0.0/{_forward(first)}:8000"]}
    assert fake_pasta.forwards(second) == {**fake_pasta.NO_FORWARDS, "-t": [f"0.0.0.0/{out['port_file']}:8000"]}


def _stage_build_seal(python: str) -> list[str]:
    return seal_prefix(SealPlan(policy=SEAL, index=True, deny_list=DENY), python=python)


def _stage_served_seal(python: str) -> list[str]:
    return seal_prefix(SealPlan(policy=SEAL, index=False, ingress_port=8000), python=python)


#: Every launch of the seal, by the prefix its launcher builds, and the guest port it serves
#: (None for a build seal): the stages' build seal (the agent build's sandbox, the package
#: stage's contestant steps) and served seal (the drive's seal probe and artifact).
LAUNCHES = {"stage-build": (_stage_build_seal, None), "stage-served": (_stage_served_seal, 8000)}


@pytest.mark.parametrize("launch", list(LAUNCHES))
def test_every_launch_of_the_seal_names_each_forward_of_its_role(tmp_path, launch):
    """Each launcher's own prefix, run with a fake pasta: a build seal forwards nothing, a served
    seal its served TCP port alone, neither leaves a direction to pasta's default, and neither maps
    its gateway's address to the host's loopback."""
    prefix_of, served = LAUNCHES[launch]
    bin_dir = fake_pasta.install(tmp_path / "bin")
    (tmp_path / "plan").write_text("pass\n")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HLS_PASTA": str(bin_dir / "pasta"), "HLS_SEAL_TOOL_PATH": str(bin_dir),
           "FAKE_PASTA_PLAN": str(tmp_path / "plan"), "FAKE_PASTA_LOG": str(tmp_path / "pasta.log"),
           "FAKE_PYTHON_COUNTER": str(tmp_path / "counter"), "HLS_SEAL_STARTED_FILE": str(tmp_path / "started"),
           "HLS_INGRESS_PORT_FILE": str(tmp_path / "port")}
    proc = subprocess.run([*prefix_of(str(bin_dir / "python")), "true"], env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    [flags] = fake_pasta.calls(tmp_path / "pasta.log")
    assert "--no-map-gw" in flags and "--no-icmp" in flags
    if served is None:
        assert fake_pasta.forwards(flags) == fake_pasta.NO_FORWARDS
    else:
        host_port = (tmp_path / "port").read_text().strip()
        assert fake_pasta.forwards(flags) == {**fake_pasta.NO_FORWARDS, "-t": [f"0.0.0.0/{host_port}:{served}"]}


@pytest.mark.parametrize("ingress, dns", [("", "closed"), ("", ""), ("8000", "")],
                         ids=["closed-build-seal", "build-seal", "served-seal"])
def test_the_seal_drops_the_payloads_icmp(tmp_path, ingress, dns):
    """pasta opens ping sockets to the payload's group, so with no CAP_NET_RAW it could still send
    ICMP echo, data and all, out of any seal and read the replies. Before the payload starts, the
    seal drops every ICMP packet but loopback's, after the interceptor's own and loopback's
    allowances; IPv6, ICMPv6 with it, is dropped whole."""
    logs = {"FAKE_IPTABLES_LOG": str(tmp_path / "v4.log"), "FAKE_IP6TABLES_LOG": str(tmp_path / "v6.log")}
    if dns:
        logs[CLOSED_DNS_VAR] = dns
    out = _seal(tmp_path, ["run"], ingress=ingress, index=not ingress, step_env=logs)
    assert out["rc"] == 0, out["stderr"]
    v4 = [r for r in fake_pasta.calls(tmp_path / "v4.log") if r[:2] != ["-t", "nat"]]
    icmp = v4.index(["-A", "OUTPUT", "-p", "icmp", "-j", "DROP"])
    assert v4.index(["-A", "OUTPUT", "-m", "mark", "--mark", str(EGRESS_FWMARK), "-j", "ACCEPT"]) < icmp
    assert v4.index(["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]) < icmp
    assert fake_pasta.calls(tmp_path / "v6.log") == [
        ["-A", "OUTPUT", "-m", "mark", "--mark", str(EGRESS_FWMARK), "-j", "RETURN"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "RETURN"], ["-A", "OUTPUT", "-j", "DROP"]]


def test_a_seal_that_cannot_start_never_runs_the_payload(tmp_path):
    out = _seal(tmp_path, ["fail:1"])
    assert (out["rc"], out["runs"], out["started"], len(out["calls"])) == (1, 0, False, 1)


def test_an_interceptor_that_never_binds_is_retried_and_the_payload_never_runs(tmp_path):
    out = _seal(tmp_path, ["run"] * 5, proxy="die")
    assert out["rc"] == SEAL_UNBOUND_EXIT and out["runs"] == 0 and not out["started"]
    assert len(out["calls"]) == 5
    assert "FATAL: egress interceptor did not become ready" in out["stderr"]
    assert out["stderr"].count("retrying") == 4


def test_a_readiness_failure_of_the_build_seal_is_retried_and_the_payload_runs_once(tmp_path):
    out = _seal(tmp_path, [f"fail:{SEAL_UNBOUND_EXIT}", "run"])
    assert (out["rc"], out["runs"], len(out["calls"])) == (0, 1, 2), out["stderr"]


def test_the_readiness_retry_keeps_the_port_it_wrote(tmp_path):
    out = _seal(tmp_path, [f"fail:{SEAL_UNBOUND_EXIT}", "run"], ingress="8000", index=False)
    assert (out["rc"], out["runs"], out["scans"]) == (0, 1, 1), out["stderr"]
    first, second = (_forward(c) for c in out["calls"])
    assert first == second == out["port_file"]


def test_a_host_port_race_is_retried_on_a_new_port(tmp_path):
    out = _seal(tmp_path, ["fail:1", "run"], ingress="8000", index=False)
    assert (out["rc"], out["runs"], out["scans"], out["started"]) == (0, 1, 2, True), out["stderr"]
    first, second = (_forward(c) for c in out["calls"])
    assert first != second and out["port_file"] == second


@pytest.mark.parametrize("rc", [SEAL_UNBOUND_EXIT, SEAL_REFUSED_EXIT])
def test_a_payload_that_exits_with_one_of_the_seals_own_statuses_runs_once(tmp_path, rc):
    # Once the payload started, its status is its own, whatever number it is.
    out = _seal(tmp_path, ["run", "run"], payload_rc=rc)
    assert (out["rc"], out["runs"], len(out["calls"])) == (rc, 1, 1)


def test_a_served_payload_that_fails_within_seconds_runs_once_on_its_port(tmp_path):
    # Retrying a fast failure of the runtime seal on a new port would re-run a payload that had
    # started and leave a reader on the port it abandoned.
    out = _seal(tmp_path, ["run", "run"], ingress="8000", index=False, payload_rc=1)
    assert (out["rc"], out["runs"], out["scans"], len(out["calls"])) == (1, 1, 1, 1)
    assert out["port_file"] == _forward(out["calls"][0])


def test_a_started_file_left_by_an_earlier_launch_counts_for_nothing(tmp_path):
    out = _seal(tmp_path, [f"fail:{SEAL_UNBOUND_EXIT}", "run"], stale_start=True)
    assert (out["rc"], out["runs"], len(out["calls"])) == (0, 1, 2), out["stderr"]


def test_without_a_named_started_file_the_seal_keeps_a_private_one(tmp_path):
    out = _seal(tmp_path, [f"fail:{SEAL_UNBOUND_EXIT}", "run", "run"], payload_rc=SEAL_UNBOUND_EXIT,
                started_file=False)
    assert (out["rc"], out["runs"], len(out["calls"])) == (SEAL_UNBOUND_EXIT, 1, 2)
    assert list((tmp_path / "tmpdir").iterdir()) == []


@pytest.mark.parametrize("plan, ingress", [(["run"], ""), ([f"fail:{SEAL_UNBOUND_EXIT}", "run"], ""), (["pass"], ""),
                                          (["fail:1", "run"], "8000")],
                         ids=["run", "retried", "pass", "served-after-a-port-race"])
def test_the_env_file_on_descriptor_3_reaches_the_payload_alone_through_pasta(tmp_path, plan, ingress):
    """pasta closes every descriptor above stderr when it starts, the env file's pipe that the
    step wrapper hands over on descriptor 3 among them. The seal passes the pipe through pasta
    as standard input, and the in-namespace block hands it to the payload alone, on 3, with an
    empty standard input: neither the interceptor, the firewall commands nor the port scan
    holds it. An attempt retried before the payload started leaves the pipe unread."""
    out = _seal(tmp_path, plan, ingress=ingress, index=not ingress, env_text="PLAIN='1'\n",
                step_input="the step's own input")
    assert out["rc"] == 0, out["stderr"]
    assert out["env_seen"] == "PLAIN='1'\n"
    assert (out["stdin"], out["stdin_is"]) == ("", "/dev/null")
    assert out["holders"] == []


def test_without_an_env_file_the_payload_keeps_the_steps_input(tmp_path):
    out = _seal(tmp_path, ["run"], step_input="the step's own input")
    assert out["rc"] == 0, out["stderr"]
    assert out["stdin"] == "the step's own input" and out["env_seen"] is None


def test_the_port_file_is_replaced_whole_never_rewritten_in_place(tmp_path):
    stale = tmp_path / "port"
    stale.write_text("1\n")
    inode = stale.stat().st_ino
    out = _seal(tmp_path, ["run"], ingress="8000", index=False)
    assert out["rc"] == 0, out["stderr"]
    assert out["port_file"] == _forward(out["calls"][0]) and stale.stat().st_ino != inode


def test_the_seal_records_the_payloads_own_ending_once_pasta_returns(tmp_path):
    out = _seal(tmp_path, ["run"], payload_rc=3)
    assert out["rc"] == 3, out["stderr"]
    assert out["ended"] == PayloadEnding(exit_code=3, interceptor_alive=True)
    # The block record, copied whole over the one kept under the name itself, in place.
    assert (tmp_path / "ended.pending").read_bytes() == (tmp_path / "ended").read_bytes()


def test_an_interceptor_that_ends_before_the_payload_is_recorded_as_gone(tmp_path):
    out = _seal(tmp_path, ["run"], proxy="quit_after:0.2", payload_sec=1.5, payload_rc=1)
    assert out["rc"] == 1, out["stderr"]
    assert out["ended"] == PayloadEnding(exit_code=1, interceptor_alive=False)


def test_a_pasta_that_ends_before_the_payload_leaves_no_ending(tmp_path):
    # An ending a previous run left behind is cleared, not taken for this one.
    (tmp_path / "ended").write_text('{"exit": 1, "interceptor": true}')
    out = _seal(tmp_path, ["die_after:0.2"], payload_sec=1.5, payload_rc=1)
    assert (out["rc"], out["started"], out["ended"]) == (137, True, None), out["stderr"]
    # The block outlives pasta, and what it records later is never taken as the ending.
    _wait_for(tmp_path / "ended.pending")
    assert payload_ending(seal_files(tmp_path)) is None


def test_each_attempt_starts_with_its_own_line(tmp_path):
    out = _seal(tmp_path, [f"fail:{SEAL_UNBOUND_EXIT}", "run"])
    lines = [line for line in out["stderr"].splitlines() if line.startswith(egress.ATTEMPT_LINE)]
    assert lines == [f"{egress.ATTEMPT_LINE}1", f"{egress.ATTEMPT_LINE}2"]


def test_a_descriptor_the_seal_cannot_hand_over_never_starts_pasta(tmp_path):
    out = _seal(tmp_path, ["run"], env_fd="4")
    assert out["rc"] != 0 and out["runs"] == 0 and out["calls"] == []
    assert "HLS_SEAL_ENV_FD" in out["stderr"]


@pytest.mark.parametrize("port, listener", [(8888, "transparent listener"), (8899, "package index")])
def test_a_contract_port_the_interceptor_listens_on_is_refused_in_one_line(port, listener):
    # The interceptor listens in the network namespace the sealed role's server shares, so that
    # server could never bind the port.
    with pytest.raises(ValueError) as refused:
        refuse_interceptor_port("multitenant_slo", port)
    assert type(refused.value) is InterceptorPortTaken
    assert str(refused.value) == (f"multitenant_slo's contract port {port} is refused: the egress interceptor's "
                                  f"{listener} listens on {port} in the network namespace a sealed role's server "
                                  "shares; declare another contract.port")


def test_the_refused_ports_are_the_ones_the_seal_hands_the_interceptor_and_no_other():
    build = seal_prefix(SealPlan(policy=SEAL, index=True, deny_list=DENY), python="/venv/bin/python")
    assert sorted(INTERCEPTOR_PORTS) == sorted(int(port) for port in build[4:6])
    for port in (8000, 8887, 8889, 8898, 8900):
        refuse_interceptor_port("multitenant_slo", port)


def test_the_prefix_is_the_seal_script_and_its_parameters_in_order():
    build = seal_prefix(SealPlan(policy=SEAL, index=True, deny_list=DENY), python="/venv/bin/python")
    assert build == ["bash", "-c", seal_script(), "hls-seal", str(EGRESS_TRANSPARENT_PORT),
                     str(EGRESS_INDEX_PORT), str(EGRESS_FWMARK), "/venv/bin/python", POLICY, ""]
    runtime = seal_prefix(SealPlan(policy=SEAL, index=False, ingress_port=8000), python="/p")
    assert runtime[5] == "" and runtime[9] == "8000"


def test_the_namespace_ports_are_those_the_prefix_gives_the_seal():
    """The ports a probe of the seal must not take for itself come from the seal's own config, the
    same that its prefix hands the script: the interceptor's listeners, and a served port."""
    build = SealPlan(policy=SEAL, index=True, deny_list=DENY)
    served = SealPlan(policy=SEAL, index=False, ingress_port=8000)
    for seal in (build, served):
        prefix = seal_prefix(seal, python="/p")
        named = {int(prefix[i]) for i in (4, 5, 9) if prefix[i]}
        assert namespace_ports(seal) == named
    assert namespace_ports(build) == {EGRESS_TRANSPARENT_PORT, EGRESS_INDEX_PORT}
    assert namespace_ports(served) == {EGRESS_TRANSPARENT_PORT, 8000}


OPEN_PACKAGES = SealPolicy(network=NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": []}}))
ALLOW_PACKAGES = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "allow", "entries": ["httpx"]}}))
#: A producer's own deny list of 12 serving frameworks and kernels, with no vllm or flashinfer.
TWELVE = ("sglang", "lmdeploy", "tensorrt-llm", "text-generation-inference", "ctranslate2", "flash-attn", "xformers",
          "exllamav2", "auto-gptq", "autoawq", "llama-cpp-python", "deepspeed")


def _deny(names: tuple[str, ...]) -> SealPolicy:
    return SealPolicy(network=NetworkPolicy.model_validate(
        {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": list(names)}}))


@pytest.mark.parametrize("kwargs", [
    {"policy": SEAL, "index": False, "ingress_port": 0},
    {"policy": SEAL, "index": False, "ingress_port": 65536},
    {"policy": OPEN_PACKAGES, "index": True, "deny_list": DENY},
    {"policy": ALLOW_PACKAGES, "index": True, "deny_list": ("vllm",)},
    {"policy": SEAL, "index": True},
    {"policy": SEAL, "index": False, "deny_list": DENY},
    {"policy": _deny(TWELVE[1:]), "index": True, "deny_list": TWELVE},
], ids=["port-0", "port-65536", "index-without-package-rule", "index-with-an-allow-rule",
        "index-without-a-deny-list", "deny-list-without-an-index", "index-missing-a-name-of-its-list"])
def test_a_seal_plan_refuses_what_the_seal_cannot_use(kwargs):
    with pytest.raises(ValueError):
        SealPlan(**kwargs)


@pytest.mark.parametrize("policy", [POLICY, SEAL.network, json.loads(POLICY)], ids=["json", "network", "dict"])
def test_a_seal_plan_takes_only_a_validated_seal_policy(policy):
    """The interceptor reads back exactly what a ``SealPolicy`` dumps, so a seal is planned with
    one: its JSON, a bare network policy or a dict of it is refused, never handed on unchecked."""
    with pytest.raises(TypeError, match="SealPolicy"):
        SealPlan(policy=policy, index=True, deny_list=DENY)


def test_a_build_seal_holds_its_producers_own_deny_list():
    """Most producers declare the harness list; one may keep its own 12 names, and its seal
    holds exactly those. A policy missing one of them is refused."""
    SealPlan(policy=_deny(TWELVE), index=True, deny_list=TWELVE)
    SealPlan(policy=SEAL, index=True, deny_list=DENY)
    SealPlan(policy=OPEN_PACKAGES, index=False)
    with pytest.raises(ValueError, match="permits \\['sglang'\\] and 0 more"):
        SealPlan(policy=_deny(TWELVE[1:]), index=True, deny_list=TWELVE)


def test_the_client_env_points_pip_at_the_index_and_empties_no_proxy_variable():
    env = client_env(SealPlan(policy=SEAL, index=True, deny_list=DENY))
    assert env["PIP_INDEX_URL"] == env["UV_INDEX_URL"] == f"http://127.0.0.1:{EGRESS_INDEX_PORT}/simple/"
    assert env["NO_PROXY"] == "*" and "" not in env.values()
    assert client_env(SealPlan(policy=SEAL, index=False)) == {"NO_PROXY": "*", "no_proxy": "*"}


def test_seal_env_names_the_host_files_pasta_the_upstream_proxy_and_the_env_file(tmp_path):
    files = seal_files(tmp_path)
    served = SealPlan(policy=SEAL, index=False, ingress_port=8000)
    env = seal_env(files, served, pasta=Path("/opt/passt/pasta"), upstream_proxy="http://proxy.site:3128")
    assert env == {"HLS_SEAL_STARTED_FILE": str(files.started), "HLS_PROXY_STATUS_FILE": str(files.status),
                   "HLS_SEAL_ENDED_FILE": str(files.ended), "HLS_PASTA": "/opt/passt/pasta",
                   "HLS_INGRESS_PORT_FILE": str(files.port), "HLS_UPSTREAM_PROXY_URL": "http://proxy.site:3128",
                   "HLS_SEAL_ENV_FD": str(ENV_FD)}
    assert set(seal_env(files, SealPlan(policy=SEAL, index=False), pasta=None, upstream_proxy=None)) == {
        "HLS_SEAL_STARTED_FILE", "HLS_PROXY_STATUS_FILE", "HLS_SEAL_ENDED_FILE", "HLS_SEAL_ENV_FD"}
    build = seal_env(files, SealPlan(policy=SEAL, index=True, deny_list=DENY), pasta=None,
                     upstream_proxy=None)
    assert build[CLOSED_DNS_VAR] == "closed"


def test_seal_env_names_the_firewall_tools_dir_only_when_given_one(tmp_path):
    """The seal resolves iptables and ip6tables from HLS_SEAL_TOOL_PATH when the step has it, and
    from the system sbin dirs otherwise; production leaves it unset, so only a runner whose own
    environment names it (a test's fake tools) passes it to the step."""
    files = seal_files(tmp_path)
    plan = SealPlan(policy=SEAL, index=False)
    assert egress.SEAL_TOOL_PATH_VAR not in seal_env(files, plan, pasta=None, upstream_proxy=None)
    env = seal_env(files, plan, pasta=None, upstream_proxy=None, tool_path="/opt/fake/bin")
    assert env[egress.SEAL_TOOL_PATH_VAR] == "/opt/fake/bin"
    assert egress.SEAL_TOOL_PATH_VAR == "HLS_SEAL_TOOL_PATH"


@pytest.mark.parametrize("text, ending", [
    (None, None), ("{", None), ('{"exit": 1}', None), ('{"exit": true, "interceptor": true}', None),
    ('{"exit": 1, "interceptor": 1}', None), ('{"exit": -1, "interceptor": true}', None),
    ('{"exit": 256, "interceptor": true}', None), ('{"exit": 1, "interceptor": true, "more": 1}', None),
    ("[1, true]", None), ('{"exit": 1, "interceptor": false}', PayloadEnding(1, False)),
    ('{"exit": 0, "interceptor": true}', PayloadEnding(0, True)),
])
def test_payload_ending_reads_only_a_valid_record(tmp_path, text, ending):
    files = seal_files(tmp_path)
    if text is not None:
        files.ended.write_text(text)
    assert payload_ending(files) == ending


def test_the_upstream_proxy_travels_under_a_name_the_step_wrapper_keeps():
    from hls.substrate.role_exec import dropped

    assert UPSTREAM_PROXY_VAR == "HLS_UPSTREAM_PROXY_URL"
    assert not dropped(UPSTREAM_PROXY_VAR, container=True)
    # The interceptor reads it by name, without importing hls.
    assert f'os.environ.get("{UPSTREAM_PROXY_VAR}")' in Path(proxy_mod.__file__).read_text()


#: A page the index refused for its size, as its status file records it.
_PAGE = {"page": "https://pypi.org/simple/huge/", "read": 70, "cap": 64}


#: A field the status leaves out.
_MISSING = object()


def _status_text(errors: object = 3, mid_stream: object = 1, oversized: object = (), hosts: object = (),
                 fetches_refused: object = 0, heartbeat: object = 1000.5, **more: object) -> str:
    """A status file's text: *hosts* the hosts the policy refused, *fetches_refused* the fetches
    refused past the cap; a field given as ``_MISSING`` is left out."""
    status = {"upstream_errors": errors, "mid_stream_errors": mid_stream, "oversized_pages": list(oversized)
              if isinstance(oversized, tuple) else oversized,
              "policy_refused": list(hosts) if isinstance(hosts, tuple) else hosts,
              "fetches_refused": fetches_refused, "heartbeat": heartbeat, **more}
    return json.dumps({k: v for k, v in status.items() if v is not _MISSING})


@pytest.mark.parametrize("text, status", [
    (None, None), ("{", None), ("[3]", None), (_status_text(errors=-1), None), (_status_text(errors=True), None),
    (_status_text(errors=3, mid_stream=4), None), ('{"upstream_errors": 3, "mid_stream_errors": 1}', None),
    # Every key is required, a status written before any refusal among them.
    ('{"upstream_errors": 3, "mid_stream_errors": 1, "oversized_pages": []}', None),
    ('{"upstream_errors": 3, "mid_stream_errors": 1, "policy_refused": []}', None),
    (_status_text(hosts="a.example"), None), (_status_text(hosts=(3,)), None), (_status_text(hosts={}), None),
    (_status_text(hosts=_MISSING), None),
    (_status_text(more=0), None), (_status_text(oversized={}), None), (_status_text(oversized=(3,)), None),
    (_status_text(oversized=({**_PAGE, "more": 0},)), None), (_status_text(oversized=({**_PAGE, "read": 64},)), None),
    (_status_text(oversized=({**_PAGE, "read": True},)), None), (_status_text(oversized=({**_PAGE, "cap": 0},)), None),
    (_status_text(oversized=({**_PAGE, "cap": True},)), None), (_status_text(oversized=({**_PAGE, "read": 70.5},)), None),
    (_status_text(oversized=({**_PAGE, "read": "70"},)), None),
    (_status_text(oversized=({**_PAGE, "page": 3},)), None),
    (_status_text(fetches_refused=-1), None), (_status_text(fetches_refused=True), None), (_status_text(fetches_refused=_MISSING), None),
    (_status_text(heartbeat=_MISSING), None), (_status_text(heartbeat="1000"), None),
    (_status_text(heartbeat=True), None), (_status_text(heartbeat=-1.0), None),
    (_status_text(heartbeat=float("inf")), None), (_status_text(heartbeat=float("nan")), None),
    (_status_text(), IndexStatus(total=3, mid_stream=1, oversized=(), policy_refused=(), fetches_refused=0, heartbeat=1000.5)),
    (_status_text(oversized=(_PAGE,), fetches_refused=2, heartbeat=7),
     IndexStatus(total=3, mid_stream=1, oversized=(OversizedPage(**_PAGE),), policy_refused=(), fetches_refused=2,
                 heartbeat=7.0)),
    (_status_text(hosts=("a.example", "b.example")),
     IndexStatus(total=3, mid_stream=1, oversized=(), policy_refused=("a.example", "b.example"), fetches_refused=0,
                 heartbeat=1000.5)),
])
def test_index_status_reads_only_a_valid_status(tmp_path, text, status):
    path = tmp_path / "status.json"
    if text is not None:
        path.write_text(text)
    assert index_status(path) == status


def test_the_ingress_port_is_known_only_once_the_payload_started(tmp_path):
    files = seal_files(tmp_path)
    files.port.write_text("41234\n")
    assert ingress_port(files) is None
    files.started.touch()
    assert ingress_port(files) == 41234
    for junk in ("junk", "0"):
        files.port.write_text(junk)
        assert ingress_port(files) is None


def test_a_sealed_container_runs_behind_the_seal_and_loses_the_network_caps(tmp_path):
    seal = SealPlan(policy=SEAL, index=True, deny_list=DENY)
    spec = ContainerSpec(image=stand_in(tmp_path / "image"), command=("pip", "wheel", "."),
                         workdir=tmp_path / "wd", seal=seal)
    argv = ApptainerRuntime().argv(spec, env_file="/dev/fd/3", seal_python="/venv/bin/python")
    prefix = seal_prefix(seal, python="/venv/bin/python")
    assert argv[:len(prefix)] == prefix
    rest = argv[len(prefix):]
    assert rest[:2] == ["apptainer", "exec"]
    assert rest[rest.index("--drop-caps") + 1] == "CAP_NET_ADMIN,CAP_NET_RAW,CAP_SYS_RESOURCE"
    assert "/etc/resolv.conf:/etc/resolv.conf:ro" in rest
    assert rest[-4:] == [START_MARKER, "pip", "wheel", "."]
    unsealed = ApptainerRuntime().argv(ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",),
                                                     workdir=tmp_path / "wd"), env_file="/dev/fd/3")
    assert unsealed[0] == "apptainer" and "--drop-caps" not in unsealed


def test_a_sealed_contestant_payload_loses_every_capability(tmp_path):
    # A container with nothing of the harness inside keeps no capability; unsealed, it runs as the
    # node's own user, which holds none.
    seal = SealPlan(policy=SEAL, index=False, ingress_port=8000)
    spec = ContainerSpec(image=stand_in(tmp_path / "image"), command=("serve",), workdir=tmp_path / "wd", seal=seal,
                         contestant_payload=True)
    argv = ApptainerRuntime().argv(spec, env_file="/dev/fd/3", seal_python="/venv/bin/python")
    rest = argv[len(seal_prefix(seal, python="/venv/bin/python")):]
    assert rest[:2] == ["apptainer", "exec"] and rest.count("--drop-caps") == 1
    assert rest[rest.index("--drop-caps") + 1] == "ALL"
    unsealed = ApptainerRuntime().argv(dataclasses.replace(spec, seal=None), env_file="/dev/fd/3")
    assert unsealed[0] == "apptainer" and "--drop-caps" not in unsealed


def test_a_sealed_container_needs_the_interceptors_interpreter(tmp_path):
    spec = ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",), workdir=tmp_path / "wd",
                         seal=SealPlan(policy=SEAL, index=False))
    with pytest.raises(ValueError, match="interpreter"):
        ApptainerRuntime().argv(spec, env_file="/dev/fd/3")


def test_pasta_path_must_be_absolute(tmp_path):
    base = {"partition": "p", "store_root": str(tmp_path), "node_scratch": str(tmp_path),
            "secrets_path": str(tmp_path / "s")}
    assert SiteSettings.model_validate(base).pasta_path is None
    assert SiteSettings.model_validate({**base, "pasta_path": "/opt/pasta"}).pasta_path == Path("/opt/pasta")
    with pytest.raises(ValueError, match="absolute"):
        SiteSettings.model_validate({**base, "pasta_path": "bin/pasta"})


@pytest.mark.parametrize("url", ["http://proxy.site:3128", "http://10.0.0.1:8080/"])
def test_a_site_names_its_upstream_proxy_as_host_and_port(tmp_path, url):
    base = {"partition": "p", "store_root": str(tmp_path), "node_scratch": str(tmp_path),
            "secrets_path": str(tmp_path / "s")}
    assert SiteSettings.model_validate(base).egress_upstream_proxy is None
    assert SiteSettings.model_validate({**base, "egress_upstream_proxy": url}).egress_upstream_proxy == url


@pytest.mark.parametrize("url", ["https://proxy.site:3128", "http://proxy.site", "http://user:pw@proxy.site:3128",
                                 "http://proxy.site:3128/path", "http://proxy.site:3128/?q=1", "proxy.site:3128",
                                 "http://:3128", "http://proxy.site:99999", "http://proxy.site:0"])
def test_an_upstream_proxy_with_anything_but_host_and_port_is_refused(tmp_path, url):
    base = {"partition": "p", "store_root": str(tmp_path), "node_scratch": str(tmp_path),
            "secrets_path": str(tmp_path / "s")}
    with pytest.raises(ValueError, match="http://<host>:<port>"):
        SiteSettings.model_validate({**base, "egress_upstream_proxy": url})


def _without_heartbeat(status: Path) -> dict:
    """The status file at *status*, its heartbeat, a time within the last few seconds, left out."""
    data = json.loads(status.read_text())
    assert abs(data.pop("heartbeat") - time.time()) < 5
    return data


def _proxy(status: Path) -> EgressProxy:
    return EgressProxy({"domains": {"mode": "allow", "entries": []}, "packages": None}, host="127.0.0.1", port=0,
                       status_file=str(status))


async def test_the_proxy_writes_its_status_at_start_and_at_each_upstream_failure(tmp_path):
    status = tmp_path / "status.json"
    proxy = _proxy(status)
    task = asyncio.create_task(proxy.serve_forever())
    try:
        for _ in range(100):
            if status.exists():
                break
            await asyncio.sleep(0.02)
        assert _without_heartbeat(status) == {"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [],
                                              "policy_refused": [], "fetches_refused": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/down":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(503)

        await proxy._fetch("https://pypi.org/down", hold=proxy._pages.hold(), transport=httpx.MockTransport(handler))
        assert _without_heartbeat(status) == {"upstream_errors": 1, "mid_stream_errors": 0, "oversized_pages": [],
                                              "policy_refused": [], "fetches_refused": 0}
        await proxy._fetch("https://pypi.org/busy", hold=proxy._pages.hold(), transport=httpx.MockTransport(handler))
        assert _without_heartbeat(status) == {"upstream_errors": 2, "mid_stream_errors": 0, "oversized_pages": [],
                                              "policy_refused": [], "fetches_refused": 0}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sorted(p.name for p in tmp_path.iterdir()) == ["status.json"]


async def test_a_served_containers_interceptor_makes_its_status_at_full_size_when_it_starts(tmp_path):
    """A served container's seal runs the interceptor with no index, only its transparent
    listener. It too writes its status at full size before it listens, so before the container
    starts, and later only over it."""
    status = tmp_path / "status.json"
    proxy = EgressProxy({"domains": {"mode": "allow", "entries": []}, "packages": None}, host="127.0.0.1",
                        port=None, transparent_port=0, status_file=str(status))
    task = asyncio.create_task(proxy.serve_forever())
    try:
        for _ in range(100):
            if status.exists():
                break
            await asyncio.sleep(0.02)
        assert status.stat().st_size == proxy_mod.STATUS_FILE_BYTES
        assert _without_heartbeat(status) == {"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [],
                                              "policy_refused": [], "fetches_refused": 0}
        proxy._note_policy_refusal("a.example")
        assert policy_refusals(status) == ("a.example",) and status.stat().st_size == proxy_mod.STATUS_FILE_BYTES
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_each_status_is_written_over_the_last_one_whole_in_one_write(tmp_path, monkeypatch):
    """One write at the file's start, of the status and the spaces after it, covering at least
    what the last status took: nothing of an earlier status is left, and nothing is renamed."""
    status = tmp_path / "status.json"
    writes, real = [], os.pwrite

    def pwrite(fd, data, offset):
        writes.append((offset, len(data), data.rstrip(b" ")))
        return real(fd, data, offset)

    monkeypatch.setattr(proxy_mod.os, "pwrite", pwrite)
    monkeypatch.setattr(proxy_mod.os, "replace", lambda *a: pytest.fail("the status was renamed"))
    proxy = _proxy(status)
    proxy._note_upstream_error()
    proxy._note_oversized("https://pypi.org/simple/" + "x" * 300 + "/", 99)
    first, longer = len(writes[0][2]), len(writes[1][2])
    proxy._oversized.clear()
    proxy._note_upstream_error()
    assert [(offset, span) for offset, span, _ in writes] == [(0, proxy_mod.STATUS_FILE_BYTES), (0, longer),
                                                             (0, longer)]
    assert first < longer and len(writes[2][2]) < longer
    written = json.loads(status.read_text())
    assert abs(written.pop("heartbeat") - time.time()) < 5
    assert written == {"upstream_errors": 2, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": [],
                       "fetches_refused": 0}
    assert status.stat().st_size == proxy_mod.STATUS_FILE_BYTES


def test_one_status_holds_the_refused_pages_and_the_refused_hosts(tmp_path):
    status = tmp_path / "status.json"
    proxy = _proxy(status)
    proxy._note_oversized("https://pypi.org/simple/huge/", proxy_mod.PAGE_MAX_BYTES + 1)
    proxy._note_policy_refusal("a.example")
    page = OversizedPage(page="https://pypi.org/simple/huge/", read=proxy_mod.PAGE_MAX_BYTES + 1,
                         cap=proxy_mod.PAGE_MAX_BYTES)
    read = index_status(status)
    assert read is not None and (read.total, read.mid_stream, read.oversized, read.policy_refused, read.fetches_refused) == (
        0, 0, (page,), ("a.example",), 0)
    assert policy_refusals(status) == ("a.example",)


def refusing_new_space(monkeypatch, root: Path) -> list[str]:
    """From here on, every way this process can take new space on the disk under *root* fails
    with ENOSPC, as on a disk a step filled: making a file, truncating one, renaming into or out
    of it, or writing past a file's end. Writing over bytes a file holds still works. Returns
    the attempts refused."""
    refused: list[str] = []
    real_open, real_pwrite, real_write = os.open, os.pwrite, os.write
    real_replace, real_rename = os.replace, os.rename

    def under(path) -> bool:
        return Path(os.fsdecode(path)).resolve().is_relative_to(root)

    def full(what: str):
        refused.append(what)
        return OSError(errno.ENOSPC, "No space left on device", what)

    def open_(path, flags, mode=0o777, *, dir_fd=None):
        if under(path) and (flags & os.O_TRUNC or flags & os.O_CREAT and not os.path.exists(path)):
            raise full(f"open {path}")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def past_end(fd: int, offset: int, size: int) -> bool:
        return under(os.readlink(f"/proc/self/fd/{fd}")) and offset + size > os.fstat(fd).st_size

    def pwrite(fd, data, offset):
        if past_end(fd, offset, len(data)):
            raise full(f"pwrite {fd}")
        return real_pwrite(fd, data, offset)

    def write(fd, data):
        if past_end(fd, os.lseek(fd, 0, os.SEEK_CUR), len(data)):
            raise full(f"write {fd}")
        return real_write(fd, data)

    def moving(real):
        def move(src, dst, **kw):
            if under(src) or under(dst):
                raise full(f"rename {src} {dst}")
            return real(src, dst, **kw)
        return move

    for name, fake in (("open", open_), ("pwrite", pwrite), ("write", write), ("replace", moving(real_replace)),
                       ("rename", moving(real_rename))):
        monkeypatch.setattr(os, name, fake)
    return refused


def test_the_status_is_rewritten_in_place_on_a_disk_the_step_filled(tmp_path, monkeypatch):
    """The interceptor writes its status at full size when it starts, before its step's payload
    runs, and then only over the bytes it holds: a step that fills the disk fails none of its
    writes, however the status grows, and each one leaves it valid."""
    seal = tmp_path / "seal"
    seal.mkdir()
    status = seal / "status.json"
    proxy = _proxy(status)
    proxy._write_status()
    before = status.stat()
    assert before.st_size == proxy_mod.STATUS_FILE_BYTES
    refused = refusing_new_space(monkeypatch, seal)
    for n in range(proxy_mod._OVERSIZED_KEPT + 2):
        proxy._note_upstream_error(mid_stream=n % 2 == 1)
        proxy._note_oversized(f"https://pypi.org/simple/{chr(0xe9) * 600}{n}/", 70 * 2**20)
        proxy._fetches_refused += 10**9
        proxy._write_status()
        data = json.loads(status.read_text())
        assert data["upstream_errors"] == n + 1 and data["fetches_refused"] == (n + 1) * 10**9
    after = status.stat()
    assert refused == [] and (after.st_ino, after.st_size) == (before.st_ino, before.st_size)
    assert index_status(status) is not None


def test_the_largest_status_fills_well_under_half_the_file():
    """Eight refused pages whose URLs, cut at 512 characters, each escape to six bytes a
    character, with every count at its largest: about 25 KiB, so the 64 KiB file holds it with
    room for what a later status adds."""
    proxy = EgressProxy({"domains": {"mode": "allow", "entries": []}, "packages": None}, host="127.0.0.1", port=0)
    for n in range(proxy_mod._OVERSIZED_KEPT):
        proxy._note_oversized(f"{n}{chr(1) * 600}", 2**63)
    proxy._upstream_errors = proxy._mid_stream_errors = proxy._fetches_refused = 2**63
    size = len(json.dumps(proxy._status()).encode())
    assert 24 * 1024 < size < proxy_mod.STATUS_FILE_BYTES // 2 and proxy_mod.STATUS_FILE_BYTES == 64 * 1024


def test_the_largest_status_with_every_refused_host_still_fits_its_file(tmp_path):
    """The largest status of all: the eight largest refused pages and every count at its
    largest, as above, and the 64 hosts the policy refused, each a name JSON writes in the 253
    bytes a kept name may take. It takes about 41 KiB, so the interceptor writes it, and a third
    of the file stays free. A client's name can hold characters JSON writes in six bytes each:
    one written in more than 253 bytes is not kept, however few characters it has. Kept, 26
    names of 253 such characters would pass the file's size, and the interceptor could write no
    status at all."""
    status = tmp_path / "status.json"
    proxy = _proxy(status)
    for n in range(proxy_mod._OVERSIZED_KEPT):
        proxy._note_oversized(f"{n}{chr(1) * 600}", 2**63)
    proxy._upstream_errors = proxy._mid_stream_errors = proxy._fetches_refused = 2**63
    too_long = [f"{n:02d}" + chr(1) * 251 for n in range(proxy_mod._POLICY_REFUSALS_KEPT)] + ["0" + chr(1) * 42 + "x"]
    longest = [f"{n:03d}" + chr(1) * 41 + "x" * 4 for n in range(proxy_mod._POLICY_REFUSALS_KEPT + 1)]
    assert [len(json.dumps(h)) - 2 for h in too_long] == [1508] * proxy_mod._POLICY_REFUSALS_KEPT + [254]
    assert {len(h) for h in too_long[:-1]} == {proxy_mod._HOST_NAME_MAX}
    assert {len(json.dumps(h)) - 2 for h in longest} == {proxy_mod._HOST_NAME_MAX}
    for host in [*too_long, *longest]:
        proxy._note_policy_refusal(host)
    read = index_status(status)
    assert read is not None and read.policy_refused == tuple(longest[:proxy_mod._POLICY_REFUSALS_KEPT])
    assert len(read.oversized) == proxy_mod._OVERSIZED_KEPT and read.fetches_refused == 2**63
    size = len(json.dumps(proxy._status()).encode())
    assert 40 * 1024 < size < proxy_mod.STATUS_FILE_BYTES * 2 // 3
    assert status.stat().st_size == proxy_mod.STATUS_FILE_BYTES


def test_a_status_that_cannot_be_written_is_removed_rather_than_left_stale(tmp_path, monkeypatch):
    status = tmp_path / "status.json"
    proxy = _proxy(status)
    proxy._write_status()
    assert status.exists()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(proxy_mod.os, "pwrite", fail)
    with pytest.raises(OSError):
        proxy._note_upstream_error()
    assert index_status(status) is None


def test_from_env_reads_the_status_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HLS_EGRESS_POLICY", OPEN_PACKAGES.model_dump_json())
    monkeypatch.setenv("HLS_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("HLS_PROXY_PORT", "0")
    monkeypatch.setenv("HLS_PROXY_STATUS_FILE", str(tmp_path / "s.json"))
    monkeypatch.setenv("HLS_UPSTREAM_PROXY_URL", "http://proxy.site:3128")
    proxy = EgressProxy.from_env()
    assert proxy._status_file == str(tmp_path / "s.json") and proxy._upstream_proxy == "http://proxy.site:3128"
    monkeypatch.delenv("HLS_PROXY_STATUS_FILE")
    monkeypatch.delenv("HLS_UPSTREAM_PROXY_URL")
    proxy = EgressProxy.from_env()
    assert proxy._status_file is None and proxy._upstream_proxy is None


#: The variables the interceptor reads that reach it through the sealed step's own environment
#: (``seal_env``, which the runner gives the step, and which the seal's bash passes on to pasta and
#: the interceptor) rather than from the seal script.
_FROM_THE_STEP = {
    # Where the interceptor keeps its status: a host path in the role's seal directory, which only
    # the runner knows (seal_files), outside every container.
    "HLS_PROXY_STATUS_FILE",
    # The site's upstream proxy (site.egress_upstream_proxy), under a name the step wrapper keeps.
    "HLS_UPSTREAM_PROXY_URL",
}


#: The names through which Python code reaches the process environment.
_ENVIRONMENT = frozenset({"environ", "environb", "getenv", "getenvb"})


def _environ_reads(source: str) -> set[str]:
    """The names of the variables the Python *source* reads from the environment, docstrings and
    comments aside. It knows three reads, each by a literal name: ``os.environ[NAME]``,
    ``os.environ.get(NAME, ...)`` (or ``.pop``) and ``os.getenv(NAME, ...)``. Any other use of one
    of ``_ENVIRONMENT`` (``"X" in os.environ``, ``dict(os.environ)``, ``from os import environ``,
    ``getattr(os, "environ")``, a write, a name that is not a literal) fails, since it could read
    a variable the caller never sees."""
    def environ(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "environ"

    def names_the_environment(node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr in _ENVIRONMENT
                or isinstance(node, ast.Name) and node.id in _ENVIRONMENT
                or isinstance(node, ast.alias) and node.name in _ENVIRONMENT
                or isinstance(node, ast.Constant) and node.value in _ENVIRONMENT)

    def literal(read: ast.AST, key: ast.AST) -> str:
        assert isinstance(key, ast.Constant) and isinstance(key.value, str), ast.unparse(read)
        return key.value

    tree = ast.parse(source)
    names: set[str] = set()
    # The id of each node naming the environment inside one of the known reads.
    known: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and environ(node.value) and isinstance(node.ctx, ast.Load):
            names.add(literal(node, node.slice))
            known.add(id(node.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            if node.func.attr in ("get", "pop") and environ(node.func.value):
                names.add(literal(node, node.args[0]))
                known.add(id(node.func.value))
            elif node.func.attr == "getenv":
                names.add(literal(node, node.args[0]))
                known.add(id(node.func))
    unknown = [f"line {node.lineno}: {ast.unparse(node)}" for node in ast.walk(tree)
               if names_the_environment(node) and id(node) not in known]
    assert not unknown, f"uses of the environment that are none of the known reads: {unknown}"
    return names


def test_the_environment_reads_are_named_by_their_literal_names():
    source = ('import os\nos.environ["A"]\nos.environ.get("B", "")\nos.environ.pop("C", None)\n'
              'os.getenv("D", default="")\n')
    assert _environ_reads(source) == {"A", "B", "C", "D"}


@pytest.mark.parametrize("use", [
    'dict(os.environ).get("X")', '"X" in os.environ', "os.environ.copy()", "{**os.environ}",
    "[k for k in os.environ.items()]", 'os.environ.setdefault("X", "")', 'os.environ["X"] = "1"',
    'del os.environ["X"]', "os.environ[name]", "os.environ.get(*names)", 'os.getenv(key="X")',
    "os.getenv(name)", "f(os.getenv)", 'os.environb[b"X"]', 'os.getenvb(b"X")', "from os import environ",
    "from os import getenv", 'getattr(os, "environ")',
])
def test_a_use_of_the_environment_the_reads_do_not_know_fails(use):
    # The agreement test below sees only the reads it knows, so any other use of the environment
    # could read a variable it never checks.
    with pytest.raises(AssertionError):
        _environ_reads(f"import os\n{use}\n")


def _interceptor_variables(tmp_path: Path, seal: SealPlan, monkeypatch) -> tuple[dict[str, str], set[str]]:
    """Run the real seal around a payload that exits 0, its command line from ``seal_prefix`` and
    its step variables from ``seal_env`` alone, with fake pasta, iptables and interpreter; return
    the ``HLS_*`` variables the interceptor was started with, and the names of the step's own."""
    bin_dir = fake_pasta.install(tmp_path / "bin")
    transparent, index = _free_ports()
    monkeypatch.setattr(seal_mod, "EGRESS_TRANSPARENT_PORT", transparent)
    monkeypatch.setattr(seal_mod, "EGRESS_INDEX_PORT", index)
    (tmp_path / "seal").mkdir()
    step = seal_env(seal_files(tmp_path / "seal"), seal, pasta=bin_dir / "pasta",
                    upstream_proxy="http://proxy.site:3128", tool_path=str(bin_dir))
    seen = tmp_path / "interceptor.json"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_PYTHON_COUNTER": str(tmp_path / "counter"),
           "FAKE_INTERCEPTOR_VARIABLES": str(seen), **step}
    argv = [*seal_prefix(seal, python=str(bin_dir / "python")), "sh", "-c", "exit 0"]
    read_end, write_end = os.pipe()
    os.close(write_end)
    try:
        # The env file's pipe goes on descriptor 3, as the step wrapper puts it.
        proc = subprocess.run(["bash", "-c", f'exec 3<&{read_end} {read_end}<&-; exec "$@"', "step", *argv],
                              env=env, pass_fds=(read_end,), capture_output=True, text=True, timeout=120)
    finally:
        os.close(read_end)
    assert proc.returncode == 0, proc.stderr
    return json.loads(seen.read_text()), set(step)


def test_the_seal_script_and_the_proxy_agree_on_their_variables(tmp_path, monkeypatch):
    """Every variable the interceptor reads is one the seal gives it, and it reads every variable
    of its own the seal gives it (``HLS_PROXY_*``, ``HLS_EGRESS_*``, ``HLS_UPSTREAM_*``; the
    ``HLS_SEAL_*``, ``HLS_PASTA`` and ``HLS_INGRESS_PORT_FILE`` it inherits are the script's). The
    seal runs as a stage builds it, for a build seal that refuses its container's DNS and for a
    served container's seal, so every variable either one sets reaches the interceptor."""
    read = _environ_reads(inspect.getsource(proxy_mod))
    given: dict[str, dict[str, str]] = {}
    from_step: set[str] = set()
    for kind, seal in (("build", SealPlan(policy=SEAL, index=True, deny_list=DENY)),
                       ("served", SealPlan(policy=SEAL, index=False, ingress_port=8000))):
        (tmp_path / kind).mkdir()
        given[kind], step = _interceptor_variables(tmp_path / kind, seal, monkeypatch)
        from_step |= step
    received = set(given["build"]) | set(given["served"])
    assert read <= received, f"the proxy reads {sorted(read - received)}, which the seal never sets"
    own = {name for name in received if name.startswith(("HLS_PROXY_", "HLS_EGRESS_", "HLS_UPSTREAM_"))}
    assert own <= read, f"the seal sets {sorted(own - read)}, which the proxy never reads"
    assert read & from_step == _FROM_THE_STEP
    # The interceptor builds itself from exactly what each seal gives it.
    for kind, variables in given.items():
        for name in [name for name in os.environ if name.startswith("HLS_")]:
            monkeypatch.delenv(name)
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        proxy = EgressProxy.from_env()
        assert proxy.host == "127.0.0.1" and proxy._fwmark == EGRESS_FWMARK
        assert proxy.transparent_port is not None and (proxy.port is not None) is (kind == "build")
        assert proxy._own_dns is (kind == "build") and proxy._status_file is not None
        assert proxy._upstream_proxy == "http://proxy.site:3128"


def _split_tools(bin_dir: Path, sbin: Path, *, tools: tuple[str, ...] = ("iptables", "ip6tables", "sysctl")) -> None:
    """Move the fake *tools*, iptables, ip6tables and sysctl, out of *bin_dir* into *sbin*, so a step
    PATH that names only *bin_dir* no longer finds them, as a real step PATH omits /usr/sbin."""
    sbin.mkdir(parents=True, exist_ok=True)
    for tool in tools:
        (bin_dir / tool).rename(sbin / tool)


#: The step's PATH these tests give the seal: the fakes (pasta, python) and /usr/bin:/bin for bash
#: and sh, as a real step PATH has; iptables, ip6tables and sysctl are moved off it into a sbin dir,
#: as a real step PATH omits /usr/sbin.
def _step_path(bin_dir: Path) -> str:
    return f"{bin_dir}:/usr/bin:/bin"


def _seal_with_tools(tmp_path: Path, *, tool_path: str,
                     tools: tuple[str, ...] = ("iptables", "ip6tables", "sysctl")) -> subprocess.CompletedProcess:
    bin_dir = fake_pasta.install(tmp_path / "bin")
    _split_tools(bin_dir, tmp_path / "sbin", tools=tools)
    transparent, index_port = _free_ports()
    (tmp_path / "plan").write_text("up\n")
    env = {"PATH": _step_path(bin_dir), "HLS_PASTA": str(bin_dir / "pasta"), "HLS_SEAL_TOOL_PATH": tool_path,
           "FAKE_PASTA_PLAN": str(tmp_path / "plan"), "FAKE_PASTA_LOG": str(tmp_path / "pasta.log"),
           "FAKE_PYTHON_COUNTER": str(tmp_path / "counter"), "FAKE_INTERCEPTOR": "serve",
           "HLS_PROXY_STATUS_FILE": str(tmp_path / "status.json"), "HLS_SEAL_ENDED_FILE": str(tmp_path / "ended"),
           "HLS_SEAL_STARTED_FILE": str(tmp_path / "started")}
    # The payload records its own PATH and that it ran; the seal must not have changed the PATH.
    payload = f"printf '%s' \"$PATH\" > {tmp_path / 'payload_path'}; echo ran >> {tmp_path / 'runs'}; exit 0"
    argv = ["bash", "-c", seal_script(), "hls-seal", str(transparent), str(index_port),
            str(EGRESS_FWMARK), str(bin_dir / "python"), POLICY, "", "sh", "-c", payload]
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)


def test_the_seal_installs_the_rules_with_iptables_off_the_step_path(tmp_path):
    """A role step's PATH carries the toolchain, not /usr/sbin where iptables and ip6tables live
    (a seal that looks only on the step PATH fails with `iptables: command not found`). With
    iptables moved off the step PATH and HLS_SEAL_TOOL_PATH naming the sbin dir, the seal resolves
    the tools there, installs the rules and runs the payload once."""
    sbin = tmp_path / "sbin"
    proc = _seal_with_tools(tmp_path, tool_path=str(sbin))
    assert proc.returncode == 0, proc.stderr
    assert "command not found" not in proc.stderr
    runs = tmp_path / "runs"
    assert runs.exists() and len(runs.read_text().splitlines()) == 1
    # The fake iptables logs each rule it installed; the nat REDIRECT is among them.
    rules = fake_pasta.calls(tmp_path / "iptables.log") if (tmp_path / "iptables.log").exists() else []
    del rules  # the fake logs rules only with FAKE_IPTABLES_LOG; the payload running is the proof


def test_the_seal_does_not_change_the_payloads_path(tmp_path):
    """The seal resolves iptables to an absolute path instead of adding the sbin dirs to PATH, so a
    build sees exactly the PATH the role was given (a different PATH could change which tools it
    finds)."""
    bin_dir = tmp_path / "bin"
    proc = _seal_with_tools(tmp_path, tool_path=str(tmp_path / "sbin"))
    assert proc.returncode == 0, proc.stderr
    payload_path = (tmp_path / "payload_path").read_text()
    assert payload_path == _step_path(bin_dir)
    assert str(tmp_path / "sbin") not in payload_path.split(":")


def test_a_missing_firewall_tool_fails_the_seal_at_setup_as_infra(tmp_path):
    """With the override pointed at an empty dir, iptables is nowhere the seal looks, so the seal
    fails at setup, before the payload runs, with an infra error naming the tool and no `refused:`
    line (which would read as the payload's)."""
    (tmp_path / "empty").mkdir()
    proc = _seal_with_tools(tmp_path, tool_path=str(tmp_path / "empty"))
    assert proc.returncode != 0
    assert "iptables not found" in proc.stderr and "refused:" not in proc.stderr
    assert not (tmp_path / "runs").exists() and not (tmp_path / "started").exists()


#: A role's limits as its stage gives them (``hls.substrate.ucounts.ucounts_env``).
UCOUNTS = "max_user_namespaces=484 max_inotify_instances=64 max_inotify_watches=8192"


@pytest.mark.parametrize("ingress", ["", "8000"], ids=["build-seal", "served-seal"])
def test_the_seal_writes_its_roles_limits_in_its_namespace_before_the_payload(tmp_path, ingress):
    """Each NAME=LIMIT of HLS_SEAL_UCOUNTS goes to /proc/sys/user/NAME in pasta's user namespace,
    which holds everything of the role, and is read back, before the payload starts."""
    out = _seal(tmp_path, ["run"], ingress=ingress, index=not ingress, step_env={"HLS_SEAL_UCOUNTS": UCOUNTS},
                payload_first=f'cat "$FAKE_NETNS_SYSCTLS" > {tmp_path / "seen"}; ')
    assert out["rc"] == 0, out["stderr"]
    assert json.loads((tmp_path / "seen").read_text()) == {
        "user.max_user_namespaces": "484", "user.max_inotify_instances": "64", "user.max_inotify_watches": "8192"}


def test_a_seal_given_no_limits_writes_none(tmp_path):
    out = _seal(tmp_path, ["run"], step_env={"FAKE_SYSCTL_LOG": str(tmp_path / "sysctl.log")})
    assert out["rc"] == 0, out["stderr"]
    assert not (tmp_path / "sysctl.log").exists()


@pytest.mark.parametrize("ingress", ["", "8000"], ids=["build-seal", "served-seal"])
@pytest.mark.parametrize("knob", ["FAKE_SYSCTL_STUCK", "FAKE_SYSCTL_FAIL"], ids=["does-not-take", "refused"])
def test_a_limit_the_seal_cannot_write_never_runs_the_payload(tmp_path, knob, ingress):
    # A write that fails, or one that reads back otherwise, leaves the role uncapped: infra, before
    # the payload. The seal's own exit for it is never taken for a host-port race, so a served seal
    # does not try again on a new port: the next attempt would fail the same way.
    out = _seal(tmp_path, ["run"] * 5, ingress=ingress, index=not ingress,
                step_env={"HLS_SEAL_UCOUNTS": UCOUNTS, knob: "1"})
    assert (out["rc"], out["runs"], out["started"], len(out["calls"])) == (SEAL_REFUSED_EXIT, 0, False, 1)
    assert "FATAL: " in out["stderr"] and "user.max_user_namespaces" in out["stderr"]
    assert "refused:" not in out["stderr"] and "retrying" not in out["stderr"]


@pytest.mark.parametrize("pairs", ["max_user_namespaces", "max_user_namespaces=", "max_user_namespaces=12a",
                                   "../max_x=1", "user.max_x=1", "max_X=1", "nr_hugepages=1", "max_*=1",
                                   "max_user_namespaces=-1"])
def test_a_malformed_limit_never_runs_the_payload(tmp_path, pairs):
    # Only NAME=LIMIT with a NAME of /proc/sys/user's own form and a LIMIT of digits: anything else
    # could write elsewhere, or nothing. A served seal does not try again on a new port.
    out = _seal(tmp_path, ["run"] * 5, ingress="8000", index=False,
                step_env={"HLS_SEAL_UCOUNTS": f"max_inotify_instances=64 {pairs}",
                          "FAKE_SYSCTL_LOG": str(tmp_path / "sysctl.log")})
    assert (out["rc"], out["runs"], out["started"], len(out["calls"])) == (SEAL_REFUSED_EXIT, 0, False, 1)
    assert "FATAL: HLS_SEAL_UCOUNTS" in out["stderr"] and "retrying" not in out["stderr"]
    assert not (tmp_path / "sysctl.log").exists()  # every pair is checked before any is written


def test_a_missing_sysctl_fails_the_seal_at_setup_as_infra(tmp_path):
    """sysctl, which writes the role's limits, is looked for where iptables is: left off the tool
    dir, the seal fails at setup, before the payload runs, as infra."""
    proc = _seal_with_tools(tmp_path, tool_path=str(tmp_path / "sbin"), tools=("iptables", "ip6tables"))
    assert proc.returncode != 0
    assert "sysctl not found" in proc.stderr and "refused:" not in proc.stderr
    assert not (tmp_path / "runs").exists() and not (tmp_path / "started").exists()
