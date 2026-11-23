"""The firewall a seal installs, invocation by invocation, pinned by a recording.

Every ``iptables`` and ``ip6tables`` call the real seal script makes is recorded, in order, for
each kind of seal: a build seal whose DNS is open, one whose DNS is closed, a served container's
seal, and a seal on a node whose ``ip6tables`` fails. The calls must equal
``seal_rules_plan.json`` exactly. The recording names the mark and the transparent port by
placeholders, and the DNS allowances of an open seal by one entry, which stands for one rule per
IPv4 nameserver of the node's ``/etc/resolv.conf``, in its order."""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from hls.substrate.apptainer.egress import CLOSED_DNS_VAR, SEAL_SCRIPT
from hls.substrate.exits import SEAL_REFUSED_EXIT, SEAL_UNBOUND_EXIT
from hls.substrate.seal import EGRESS_FWMARK, SEAL_RULES
from tests.substrate import fake_pasta
from tests.substrate import test_egress_seal
from tests.substrate.test_egress_seal import _seal

RECORDING = Path(__file__).with_name("seal_rules_plan.json")
#: The entry of the recording that stands for the DNS allowances of an open seal.
RESOLVERS = "<resolv.conf nameservers>"
#: A firewall tool that records its arguments, under its own name, in ``$SEAL_RULES_LOG``, and
#: fails when ``$SEAL_RULES_FAIL`` names it.
_RECORDER = """#!/bin/sh
{python} -c 'import json, sys; open(sys.argv[1], "a").write(json.dumps(sys.argv[2:]) + "\\n")' \\
  "$SEAL_RULES_LOG" {tool} "$@"
[ "${{SEAL_RULES_FAIL:-}}" != {tool} ]
"""

#: Each kind of seal recorded: (DNS mode, index, served port, the tool that fails).
CASES = {
    "build seal": ("", True, "", ""),
    "build seal, DNS closed": ("closed", True, "", ""),
    "served seal": ("", False, "8000", ""),
    "ip6tables fails": ("", True, "", "ip6tables"),
}


def _recorders(directory: Path) -> Path:
    directory.mkdir(parents=True)
    for tool in ("iptables", "ip6tables"):
        path = directory / tool
        path.write_text(_RECORDER.format(python=shlex.quote(sys.executable), tool=tool))
        path.chmod(0o755)
    return directory


def resolver_rules(resolv_conf: Path = Path("/etc/resolv.conf")) -> list[list[str]]:
    """The DNS allowance an open seal installs for each IPv4 nameserver of *resolv_conf*."""
    rules = []
    for line in resolv_conf.read_text().splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == "nameserver" and ":" not in words[1]:
            rules.append(["iptables", "-A", "OUTPUT", "-p", "udp", "-d", words[1], "--dport", "53", "-j", "ACCEPT"])
    return rules


def expected_calls(recorded: list[list[str]], transparent: int,
                   resolv_conf: Path = Path("/etc/resolv.conf")) -> list[list[str]]:
    """The calls *recorded* stands for, with the seal's mark, *transparent* as its port and the
    resolvers of *resolv_conf*, the namespace's."""
    out: list[list[str]] = []
    for call in recorded:
        if call == [RESOLVERS]:
            out.extend(resolver_rules(resolv_conf))
            continue
        out.append([arg.replace("{mark}", str(EGRESS_FWMARK)).replace("{tport}", str(transparent))
                    for arg in call])
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_the_seal_installs_the_recorded_firewall(tmp_path, monkeypatch, case):
    dns, index, ingress, failing = CASES[case]
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        ports = first.getsockname()[1], second.getsockname()[1]
    monkeypatch.setattr(test_egress_seal, "_free_ports", lambda: ports)
    log = tmp_path / "rules.log"
    step_env = {"HLS_SEAL_TOOL_PATH": f"{_recorders(tmp_path / 'rec')}:{tmp_path / 'bin'}",
                "SEAL_RULES_LOG": str(log), "SEAL_RULES_FAIL": failing}
    if dns:
        step_env[CLOSED_DNS_VAR] = dns
    out = _seal(tmp_path, ["run"], ingress=ingress, index=index, step_env=step_env)
    if not failing:
        assert out["rc"] == 0, out["stderr"]
    assert len(out["calls"]) == 1, out["stderr"]
    recorded = json.loads(RECORDING.read_text())[case]
    assert fake_pasta.calls(log) == expected_calls(recorded, ports[0])


def test_the_seal_script_refuses_to_run_without_its_rules(tmp_path):
    """Without the rule text its builder sets, the seal stops at its first lines, naming the
    variable, before it does anything."""
    env = {"PATH": "/usr/bin:/bin", "HLS_SEAL_REFUSED_EXIT": str(SEAL_REFUSED_EXIT),
           "HLS_SEAL_UNBOUND_EXIT": str(SEAL_UNBOUND_EXIT)}
    p = subprocess.run(["bash", str(SEAL_SCRIPT)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout == ""
    [line] = p.stderr.splitlines()
    assert "HLS_SEAL_RULES: is set by the seal builder" in line


def test_the_rule_text_reaches_no_process_of_the_seal(tmp_path):
    """The rule text heads the in-namespace block's program and is not exported, so neither the
    interceptor nor the payload is started with it."""
    seen = tmp_path / "interceptor.json"
    out = _seal(tmp_path, ["run"], step_env={"FAKE_INTERCEPTOR_VARIABLES": str(seen)},
                payload_first=f"env > {tmp_path / 'payload.env'}; ")
    assert out["rc"] == 0, out["stderr"]
    assert "HLS_SEAL_RULES" not in json.loads(seen.read_text())
    assert "HLS_SEAL_RULES" not in (tmp_path / "payload.env").read_text()
    assert "seal_rules" not in (tmp_path / "payload.env").read_text()


def test_the_rule_file_defines_seal_rules_and_nothing_else(tmp_path):
    """Sourced, the rule file installs nothing: it defines the one function and sets no variable."""
    script = (f". {shlex.quote(str(SEAL_RULES))}; declare -F; "
              "declare -p | grep -E '^declare -[a-zA-Z-]* (m|_kw|_ns)=' || true")
    p = subprocess.run(["bash", "--norc", "--noprofile", "-c", script], env={"PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, check=True)
    assert p.stdout.splitlines() == ["declare -f seal_rules"]
