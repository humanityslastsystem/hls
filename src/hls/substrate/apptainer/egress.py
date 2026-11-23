"""The network seal of a container role, on Apptainer.

A sealed container runs inside a user and network namespace that ``pasta`` creates on the
host side of the role's step. ``transparent_seal.sh`` redirects every TCP connection the
container makes to ``hls.egress.proxy``, the interceptor, which reads the destination host
without decrypting (TLS SNI or HTTP Host) and relays only what the policy allows. For a step
that installs packages it also serves a pull-through index on the container's loopback, whose
package rule refuses denied names. The interceptor runs with the plan's interpreter in the
namespaces pasta makes (network, PID, mount, IPC and UTS) but outside the container, which
apptainer starts in a PID namespace of its own (``--pid``, ``--containall``) and without
CAP_NET_ADMIN, CAP_NET_RAW and CAP_SYS_RESOURCE (a contestant's payload without any capability),
so the container can neither see nor signal it, nor rewrite the firewall, nor raise the limits
the seal wrote. pasta forwards no port into the namespace but a served container's one TCP port,
from a free host port; nothing of the namespace reaches the host's loopback (no forward out of
it, no mapping of its gateway's address); and no ICMP leaves it (pasta relays none, and its
firewall drops it). The seal script names each of pasta's
four forward directions, since in a direction left unnamed pasta forwards every port bound on
the far side that lies outside the host's ephemeral range and that it may bind unprivileged.

The seal comes before ``apptainer exec`` on the role's command line, so the step runs the
wrapper, then the seal, pasta, the in-namespace block, and apptainer. Until pasta has made
the namespace the step is on the host's network, and only harness code runs: the seal's bash
picks a host port and starts pasta. Apptainer, and so everything of the container's, starts
only inside the namespace, after the redirect is installed and the interceptor has bound;
a seal that cannot get that far never starts it. The container prints its start marker inside
the container, after all of them, so a seal that fails before the container starts leaves no
marker and the role ends ``runtime_failed`` (infra), whatever status the seal exits with.

The step wrapper removes BASH_ENV and every variable apptainer reads, its options and every
``*_proxy`` variable among them, from a container step before the seal's bash starts
(``hls.substrate.role_exec.dropped``), so the seal and apptainer see neither a startup file
nor an option a site's prolog added. A site whose nodes reach the internet only through a
proxy names it in ``site.egress_upstream_proxy``, which the seal hands to the interceptor
under a name of its own (``UPSTREAM_PROXY_VAR``); the container never sees it. The step
wrapper hands the container's env file over on descriptor ``SEAL_ENV_FD``, which pasta would
close; the seal carries it through pasta on standard input and gives it to the payload alone.

Each attempt of the seal starts with an ``ATTEMPT_LINE`` on stderr, so a reader of the role's
stderr can tell the attempt that failed from the earlier ones it retried.

The evidence a stage reads (``hls.substrate.seal``) lives in the seal's state directory. Here
the port file holds the host port pasta forwards; the ended file is kept only when pasta
returned after the payload; and the seal's block writes the payload's ending into
``<ended>.pending`` right after the payload ended and the block stopped the interceptor.
A seal that refuses the container's DNS (``hls.substrate.seal.closes_dns``) is told so by
``CLOSED_DNS_VAR``."""

from __future__ import annotations

import shlex
from pathlib import Path

from hls.substrate.container import SealPlan
from hls.substrate.exits import SEAL_REFUSED_EXIT, SEAL_UNBOUND_EXIT
from hls.substrate.seal import (EGRESS_FWMARK, SEAL_RULES, UPSTREAM_PROXY_VAR, SealFiles, closes_dns,
                               interceptor_ports)

#: The seal script, embedded in every sealed command line.
SEAL_SCRIPT = Path(__file__).with_name("transparent_seal.sh")
#: The descriptor the step wrapper hands a container's env file over on
#: (``hls.substrate.role_exec.ENV_FD``), the only one the seal script carries through pasta.
SEAL_ENV_FD = 3
#: How the line the seal prints on stderr before each attempt begins; the attempt's number
#: follows.
ATTEMPT_LINE = "hls-seal: attempt "
#: The step variable that tells the seal to refuse the container's DNS (``closes_dns``).
CLOSED_DNS_VAR = "HLS_SEAL_DNS"
#: The step variable naming the directories the seal resolves iptables and ip6tables from,
#: ":"-separated. Production leaves it unset, so the seal looks in /usr/sbin and /sbin; a runner
#: whose own environment sets it (a test's fake tools) passes it to the sealed step.
SEAL_TOOL_PATH_VAR = "HLS_SEAL_TOOL_PATH"


def seal_script() -> str:
    """The seal's program as ``bash -c`` runs it: ``SEAL_SCRIPT`` after the statuses it exits with
    when it refuses its payload or its interceptor never binds, set from their one owner
    (``hls.substrate.exits``), and the text of its firewall (``SEAL_RULES``), which the script
    refuses to run without."""
    return (f"HLS_SEAL_REFUSED_EXIT={SEAL_REFUSED_EXIT}\nHLS_SEAL_UNBOUND_EXIT={SEAL_UNBOUND_EXIT}\n"
            f"HLS_SEAL_RULES={shlex.quote(SEAL_RULES.read_text())}\n" + SEAL_SCRIPT.read_text())


def seal_prefix(seal: SealPlan, *, python: str) -> list[str]:
    """The command line before ``apptainer exec`` that seals it (``seal_script``). *python* runs
    the interceptor and must import ``hls``: the plan's interpreter."""
    transparent, index = interceptor_ports(seal)
    return ["bash", "-c", seal_script(), "hls-seal", str(transparent), "" if index is None else str(index),
            str(EGRESS_FWMARK), python, seal.policy.model_dump_json(),
            "" if seal.ingress_port is None else str(seal.ingress_port)]


def seal_env(files: SealFiles, seal: SealPlan, *, pasta: Path | None, upstream_proxy: str | None,
             tool_path: str | None = None) -> dict[str, str]:
    """The role step's own variables the seal *seal* reads on the host side: where its files
    are, the descriptor the container's env file arrives on (``SEAL_ENV_FD``), the port file of
    a served container, ``CLOSED_DNS_VAR`` when it refuses the container's DNS
    (``closes_dns``), the pasta program when the site names one, the site's upstream proxy
    when it has one, and *tool_path* (``SEAL_TOOL_PATH_VAR``), the directories the seal resolves
    its firewall tools from, when the runner names them. None of them reaches the container."""
    env = {"HLS_SEAL_STARTED_FILE": str(files.started), "HLS_PROXY_STATUS_FILE": str(files.status),
           "HLS_SEAL_ENDED_FILE": str(files.ended), "HLS_SEAL_ENV_FD": str(SEAL_ENV_FD)}
    if pasta is not None:
        env["HLS_PASTA"] = str(pasta)
    if seal.ingress_port is not None:
        env["HLS_INGRESS_PORT_FILE"] = str(files.port)
    if closes_dns(seal):
        env[CLOSED_DNS_VAR] = "closed"
    if upstream_proxy is not None:
        env[UPSTREAM_PROXY_VAR] = upstream_proxy
    if tool_path is not None:
        env[SEAL_TOOL_PATH_VAR] = tool_path
    return env
