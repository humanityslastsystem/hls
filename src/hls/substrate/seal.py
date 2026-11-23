"""The network seal of a container role, as every substrate's seal presents it to a stage.

A sealed container's every TCP connection goes to ``hls.egress.proxy``, the interceptor, which
runs outside the container in the network namespace the container shares, reads the destination
host without decrypting (TLS SNI or HTTP Host) and relays only what the policy allows. For a step
that installs packages it also serves a pull-through index on the container's loopback, whose
package rule refuses denied names. How the namespace is made and who starts the interceptor is
the substrate's (``hls.substrate.apptainer.egress`` for Apptainer); this module holds the ports,
the interceptor's mark and variables, and the files a stage reads.

The evidence a stage reads lives in host files that no container mounts (``seal_files``):
- the started file, which the seal creates right before the payload starts. An attempt that
  leaves it ran the payload, and the seal never retries that attempt;
- a served container's port file: the host port that reaches the container's one served port,
  which ``ingress_port`` reads once the started file exists;
- the interceptor's status file (``index_status``), which holds its count of failed upstream
  fetches by the index, how many of them failed while a file was already streaming to the
  container, the simple pages the index refused for their size, the hosts its policy refused
  (``policy_refusals``), the requests it refused past its fetch cap, and its heartbeat: the
  time it last wrote the file, which it does every few seconds while it runs. A rise in the
  count tells an index outage apart from a contestant's broken dependencies, without trusting
  anything the container wrote. Only a failure mid-stream can follow from how a client read:
  one that stalled its download could leave the upstream waiting until it gave up. A refused
  page names the cause of a failure that stays the submission's. A refused host is the seal
  probe's evidence that the seal refuses a host its policy does not name. An old heartbeat
  tells an interceptor that stopped answering from one that counted nothing;
- the ended file (``payload_ending``): the payload's own exit status and whether the
  interceptor was still running when it ended, which the seal keeps only when it saw the
  payload end. A stage scores a payload's status only with it: a seal or interceptor that
  died while the payload ran is never the contestant's failure. The time the seal wrote it is
  when the payload ended (``payload_ended_at``), which a stage judges the heartbeat against.

A build seal whose policy reaches no host but through its index (``closes_dns``: an index and
an air-gapped domain rule) also refuses the container's DNS queries, so a build hook cannot
carry data out in the names it looks up. The interceptor, which must still resolve the
index's upstream, then resolves names itself over sockets carrying its mark, the only DNS the
firewall lets out."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hls.egress.policy import EgressPosture

if TYPE_CHECKING:
    from hls.substrate.container import SealPlan

#: Loopback port of the interceptor's transparent listener, the target of the redirect.
EGRESS_TRANSPARENT_PORT = 8888
#: Loopback port of the pull-through package index.
EGRESS_INDEX_PORT = 8899
#: Each port the interceptor listens on in a sealed role's network namespace, which the role's
#: container shares, to the listener there: the transparent listener in every seal, the index in a
#: seal that has one. No task's contract port may be one of them (``refuse_interceptor_port``).
INTERCEPTOR_PORTS = {EGRESS_TRANSPARENT_PORT: "transparent listener", EGRESS_INDEX_PORT: "package index"}
#: Firewall mark on the interceptor's own upstream sockets, which the redirect exempts.
EGRESS_FWMARK = 0x4853
#: Seconds a reader should wait for a served container's seal to start before giving up.
#: The interceptor alone can take a minute to bind on a loaded node, and the image pages in
#: after that.
INGRESS_WAIT_SEC = 300.0
#: The firewall every seal installs in its namespace, a bash function, ``seal_rules``, that the
#: program which runs there defines by starting with this text.
SEAL_RULES = Path(__file__).with_name("seal_rules.sh")
#: The variable the interceptor reads its upstream proxy from (``hls.egress.proxy`` reads it by
#: this name without importing hls). It does not end in ``_proxy``, so the step wrapper keeps it.
UPSTREAM_PROXY_VAR = "HLS_UPSTREAM_PROXY_URL"


@dataclass(frozen=True)
class SealFiles:
    """The host files of one sealed role, all in a directory no container mounts."""

    #: The interceptor's status: ``{"upstream_errors": N, "mid_stream_errors": M,
    #: "oversized_pages": [{"page": URL, "read": R, "cap": C}, ...], "policy_refused": [host, ...],
    #: "fetches_refused": F, "heartbeat": T}``.
    status: Path
    #: Created right before the payload starts.
    started: Path
    #: The host port of a served container's forward.
    port: Path
    #: How the payload ended: ``{"exit": N, "interceptor": true|false}``, kept only when the seal
    #: saw the payload end.
    ended: Path


@dataclass(frozen=True)
class OversizedPage:
    """A simple page the index refused because it was larger than the index reads."""

    #: The page's upstream URL.
    page: str
    #: Bytes of it the index read before it stopped, more than *cap*.
    read: int
    #: The most bytes of a page the index reads.
    cap: int


@dataclass(frozen=True)
class IndexStatus:
    """What the interceptor recorded in its status file: its index's failed upstream fetches,
    the pages the index refused for their size and the requests it refused past its fetch cap,
    the hosts its transparent listener refused by its policy, and when it last wrote the file."""

    #: Every failed fetch.
    total: int
    #: Those that failed while a file was already streaming to the client, counted only when the
    #: client was reading it promptly.
    mid_stream: int
    #: The pages refused for their size: the first few the interceptor keeps, each once.
    oversized: tuple[OversizedPage, ...]
    #: The hosts refused by the policy, first refused first: the first few the interceptor
    #: keeps, each once. A host the policy let through that did not resolve, or could not be
    #: reached, is not among them.
    policy_refused: tuple[str, ...]
    #: The requests it answered itself, with no fetch from its upstream, once the step had made
    #: all the upstream fetches one step may make.
    fetches_refused: int
    #: When the interceptor last wrote the status, in seconds since the epoch: it writes it every
    #: few seconds while it runs, so an old one says it stopped answering.
    heartbeat: float


@dataclass(frozen=True)
class PayloadEnding:
    """How a sealed container's payload ended, as its seal recorded it."""

    #: The payload's own exit status.
    exit_code: int
    #: Whether the interceptor was still running when the payload ended.
    interceptor_alive: bool


def seal_files(state_dir: Path) -> SealFiles:
    """The seal files of a role whose seal keeps its state in *state_dir*."""
    return SealFiles(status=state_dir / "status.json", started=state_dir / "started", port=state_dir / "port",
                     ended=state_dir / "ended")


def interceptor_ports(seal: SealPlan) -> tuple[int, int | None]:
    """The ports the interceptor of *seal* listens on in its namespace: its transparent listener,
    and its index, None for a seal without one."""
    return EGRESS_TRANSPARENT_PORT, EGRESS_INDEX_PORT if seal.index else None


def namespace_ports(seal: SealPlan) -> frozenset[int]:
    """The ports *seal*'s namespace holds for the seal's own listeners (``interceptor_ports``) and
    for a served container's port. A probe of the seal binds none of them for itself."""
    transparent, index = interceptor_ports(seal)
    return frozenset(p for p in (transparent, index, seal.ingress_port) if p is not None)


class InterceptorPortTaken(ValueError):
    """A task whose contract port is one the interceptor listens on (``INTERCEPTOR_PORTS``)."""


def refuse_interceptor_port(task_id: str, port: int) -> None:
    """Raise InterceptorPortTaken, in one line, when *port*, the contract port of task *task_id*,
    is one the interceptor listens on (``INTERCEPTOR_PORTS``): the task's server shares the
    interceptor's network namespace in a sealed role, so it could never bind that port. Every plan
    that serves a task's artifact or reads its contract calls this when it loads the task
    (``hls.pipeline.commands.task_to_plan``)."""
    listener = INTERCEPTOR_PORTS.get(port)
    if listener is not None:
        raise InterceptorPortTaken(f"{task_id}'s contract port {port} is refused: the egress interceptor's {listener} "
                                   f"listens on {port} in the network namespace a sealed role's server shares; "
                                   "declare another contract.port")


def closes_dns(seal: SealPlan) -> bool:
    """Whether *seal* refuses its container's DNS queries: a seal with an index whose policy
    lets the container reach no host (its domain posture is the air gap, ``EgressPosture.of``),
    so its container never needs a name resolved."""
    return seal.index and EgressPosture.of(seal.policy.network.domains) is EgressPosture.AIR_GAP


def client_env(seal: SealPlan) -> dict[str, str]:
    """Variables the sealed container gets: pip and uv pointed at the index when there is
    one, and ``NO_PROXY=*`` so a proxy-aware client dials directly, where the redirect sees
    it. Nothing here empties a proxy variable: none reaches the container from the step,
    whose wrapper removes them, and one the image declares is the image's own."""
    env = {"NO_PROXY": "*", "no_proxy": "*"}
    if seal.index:
        index = f"http://127.0.0.1:{EGRESS_INDEX_PORT}/simple/"
        env.update(PIP_INDEX_URL=index, PIP_TRUSTED_HOST="127.0.0.1", UV_INDEX_URL=index,
                   UV_INSECURE_HOST=f"127.0.0.1:{EGRESS_INDEX_PORT}")
    return env


def index_status(status_file: Path) -> IndexStatus | None:
    """The interceptor's status; None when it wrote no valid one (the seal never got as far as
    starting it)."""
    try:
        data = json.loads(status_file.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != {"upstream_errors", "mid_stream_errors", "oversized_pages",
                                                   "policy_refused", "fetches_refused", "heartbeat"}:
        return None
    total, mid_stream, pages = data["upstream_errors"], data["mid_stream_errors"], data["oversized_pages"]
    hosts, fetches_refused, heartbeat = data["policy_refused"], data["fetches_refused"], data["heartbeat"]
    if not _counts(total, mid_stream, fetches_refused) or mid_stream > total or not isinstance(pages, list):
        return None
    if isinstance(heartbeat, bool) or not isinstance(heartbeat, (int, float)) or not 0 <= heartbeat < math.inf:
        return None
    if not isinstance(hosts, list) or not all(isinstance(host, str) for host in hosts):
        return None
    oversized = []
    for page in pages:
        if (not isinstance(page, dict) or set(page) != {"page", "read", "cap"} or not isinstance(page["page"], str)
                or not _counts(page["read"], page["cap"]) or not 0 < page["cap"] < page["read"]):
            return None
        oversized.append(OversizedPage(page=page["page"], read=page["read"], cap=page["cap"]))
    return IndexStatus(total=total, mid_stream=mid_stream, oversized=tuple(oversized), policy_refused=tuple(hosts),
                       fetches_refused=fetches_refused, heartbeat=float(heartbeat))


def _counts(*values: object) -> bool:
    """Whether each of *values* is an int of at least 0 (a bool is not)."""
    return all(not isinstance(n, bool) and isinstance(n, int) and n >= 0 for n in values)


def policy_refusals(status_file: Path) -> tuple[str, ...] | None:
    """The hosts the interceptor refused by its policy, as it recorded them in its status file
    (``index_status``), first refused first; empty when it refused none. None when there is no
    valid status: the interceptor never started, or never wrote it."""
    status = index_status(status_file)
    return None if status is None else status.policy_refused


def payload_ending(files: SealFiles) -> PayloadEnding | None:
    """How the payload ended, as the seal recorded it once it saw the payload end; None when
    there is no valid record: the seal ended before the payload, or the payload never started."""
    try:
        data = json.loads(files.ended.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != {"exit", "interceptor"}:
        return None
    code, alive = data["exit"], data["interceptor"]
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255 or not isinstance(alive, bool):
        return None
    return PayloadEnding(exit_code=code, interceptor_alive=alive)


def payload_ended_at(files: SealFiles) -> float | None:
    """When the payload ended, in seconds since the epoch, on the node's own clock: the time the
    seal wrote the payload's ending into ``<ended>.pending``, right after the payload ended and
    the seal stopped the interceptor. The seal alone writes that file, in a directory no
    container mounts, and the node's kernel stamps the time, so no client can set it. None
    without a valid ending (``payload_ending``): the payload never started, or the seal ended
    first."""
    if payload_ending(files) is None:
        return None
    try:
        return files.ended.with_name(f"{files.ended.name}.pending").stat().st_mtime
    except OSError:
        return None


def ingress_port(files: SealFiles) -> int | None:
    """The host port that reaches a served container, once its seal has started the payload;
    None before then. The seal writes the port file whole each time it picks a port, and picks
    one again only before the payload starts, so after the started file appears the port
    file holds the port of the attempt that runs."""
    if not files.started.exists():
        return None
    try:
        text = files.port.read_text().strip()
    except OSError:
        return None
    if not text.isdecimal() or not 0 < int(text) < 65536:
        return None
    return int(text)
