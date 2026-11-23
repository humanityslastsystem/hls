"""The harness side of the in-build reference service: the protocol facts the brief states and the
build stage relies on, the engine the service builds, its command line, and a reader for its
``/status`` reply.

``service.py`` beside this module is the service. It runs in its own harness container on the
task's measured baseline image, cannot import hls, holds a task's pinned reference asleep on the build node's GPUs
and scores token sequences the agent submits over a unix socket. This module holds what both sides
agree on: the paths the build sandbox finds it at, how long it may take to start, and the states
its ``/status`` reply names.
"""

from __future__ import annotations

import base64
import binascii
import json
import shlex
import socket
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hls.reference.references import VLLMReference

#: Where the service's script is mounted, read-only, in its own container.
SCRIPT_DEST = "/opt/hls-reference-service/service.py"
#: The script's path in the checkout the plan pins, which the build stage mounts at
#: ``SCRIPT_DEST``.
SCRIPT_REL = "src/hls/reference/service.py"
#: Where the socket's host directory is bound, in the build sandbox and in the service's
#: own container alike, so the one path is right on both sides.
SOCKET_DIR = "/run/hls-reference"
SOCKET_PATH = f"{SOCKET_DIR}/ref.sock"
#: The build sandbox env var naming ``SOCKET_PATH``.
SOCKET_ENV_VAR = "HLS_REFERENCE_SOCKET"
#: Idle seconds after which the service puts its engine back to sleep, returning the
#: GPU memory to the agent.
IDLE_SLEEP_SEC = 120
#: How long a service may stay starting at a stretch, building its engine and first going to
#: sleep, at its launch or at a load a wake asked for. The engine build takes minutes on a large
#: checkpoint, more with a cold compile cache. A service still starting after this is ended,
#: since the engine it is building holds most of every card the agent works on.
READY_TIMEOUT_SEC = 1800.0
#: The ``/status`` state of a service building its engine. It binds its socket first and
#: reports this until the engine is built and asleep; the agent runs meanwhile.
STARTING_STATE = "starting"
#: The ``/status`` state a service enters once its engine is built: asleep, its GPU memory
#: free for the agent. The first state after ``STARTING_STATE`` on every start.
READY_STATE = "asleep"
#: The ``/status`` states of a built service. The agent runs while the service starts, so
#: by the time the harness looks it may already have woken it: any of these is ready.
READY_STATES = frozenset({READY_STATE, "waking", "awake", "sleeping"})
#: The ``/status`` state that means it will never be ready; its reply carries "error".
FAILED_STATE = "failed"
#: The ``/status`` state of a service whose engine is not built because memory was short when
#: it loaded; its reply's "memory_error" says which memory, and a ``POST /wake`` loads it again.
UNLOADED_STATE = "unloaded"
#: Bound on one ``/status`` exchange.
STATUS_TIMEOUT_SEC = 10.0
#: Bytes of a ``/status`` reply read at most. A reply is under a kilobyte per device.
STATUS_MAX_BYTES = 2**20


@dataclass(frozen=True)
class ServiceEngine:
    """The vLLM engine the service builds. Each field is the grader's reference engine's
    (``VLLMReference``), so a score the agent reads comes from the engine that grades it."""

    tensor_parallel_size: int      # cards the weights are split across: every GPU the service shares
    max_model_len: int | None      # context length; None lets vLLM infer it from the model
    gpu_memory_utilization: float  # fraction of each card vLLM claims while awake
    dtype: str                     # "auto" or an explicit dtype name
    enforce_eager: bool            # no CUDA-graph capture
    batch_invariant: bool          # batch-invariant kernels

    @classmethod
    def from_reference(cls, ref: "VLLMReference") -> "ServiceEngine":
        """The engine ``ref`` builds when it grades."""
        return cls(tensor_parallel_size=ref.tensor_parallel_size,
                   max_model_len=ref.max_model_len,
                   gpu_memory_utilization=ref.gpu_memory_utilization,
                   dtype=ref.dtype, enforce_eager=ref.enforce_eager,
                   batch_invariant=ref.batch_invariant)

    def argv(self, *, weights: str) -> list[str]:
        """The service's full command line inside its container, loading ``weights``."""
        from hls.verify.report import TOP_LOGPROBS_REQUESTED

        argv = ["python3", SCRIPT_DEST, "--socket", SOCKET_PATH, "--weights", weights,
                "--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.max_model_len is not None:
            argv += ["--max-model-len", str(self.max_model_len)]
        argv += ["--gpu-memory-utilization", str(self.gpu_memory_utilization),
                 "--dtype", self.dtype]
        if self.enforce_eager:
            argv.append("--enforce-eager")
        if self.batch_invariant:
            argv.append("--batch-invariant")
        # The grader reads this many alternatives per position, so the service reports
        # the same number.
        argv += ["--top-logprobs", str(TOP_LOGPROBS_REQUESTED),
                 "--idle-sleep-sec", str(IDLE_SLEEP_SEC)]
        return argv


def service_engine_for(task) -> ServiceEngine:
    """The engine of ``task``'s pinned vLLM reference: the ``VLLMReference`` a verifier of
    the task grades against.

    Raises ``WiringError`` when the task has none, or has references that would build
    different engines: the service serves one engine, and choosing between two would
    hand the agent scores from an engine that does not grade it."""
    from hls.reference.references import VLLMReference
    from hls.verify.base import WiringError

    engines = {
        ServiceEngine.from_reference(v.reference)
        for tenant in task.effective_tenants()
        for v in (tenant.verifiers or [])
        if isinstance(getattr(v, "reference", None), VLLMReference)
    }
    if len(engines) != 1:
        raise WiringError(
            f"producer.reference_service is set, but task {getattr(task, 'id', None)!r} "
            f"has {len(engines)} distinct VLLMReference engines among its verifiers' "
            "references; the service serves the one pinned vLLM reference the task grades "
            "against, so exactly one is required.")
    return engines.pop()


def http_get_request(path: str) -> bytes:
    """A minimal HTTP/1.0 GET of ``path``. HTTP/1.0 with ``Connection: close`` makes the
    server close after one reply, so a reader finds the end of the reply at EOF."""
    return f"GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()


def parse_http_response(raw: bytes) -> tuple[int, bytes]:
    """``(status code, body)`` of a complete raw HTTP response. Raises ``ValueError`` on
    bytes that are not one."""
    head, sep, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].split()
    if (not sep or len(status_line) < 2 or not status_line[0].startswith(b"HTTP/")
            or not status_line[1].isdigit()):
        raise ValueError(f"not an HTTP response: {raw[:200]!r}")
    return int(status_line[1]), body


def read_status(code: int, body: bytes) -> dict[str, Any]:
    """The service's ``/status`` reply as a dict. Raises ``ValueError`` on a reply that is
    not a JSON object carrying a "state"."""
    try:
        status = json.loads(body)
    except ValueError as exc:
        raise ValueError(f"/status answered HTTP {code} with a body that is not JSON: "
                         f"{body[:200]!r}") from exc
    if not isinstance(status, dict) or "state" not in status:
        raise ValueError(f"/status answered HTTP {code} without a state: {body[:200]!r}")
    return status


def get_status(socket_path: str, *, timeout: float = STATUS_TIMEOUT_SEC) -> dict[str, Any]:
    """One blocking ``GET /status`` against the service at ``socket_path`` on this host, the
    whole exchange within ``timeout`` and its reply within ``STATUS_MAX_BYTES``. Raises
    ``OSError`` when the socket cannot be reached or the exchange runs out of either, and
    ``ValueError`` on a reply ``read_status`` rejects."""
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    size = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        sock.sendall(http_get_request("/status"))
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"/status at {socket_path} did not answer within {timeout:.0f} s")
            sock.settimeout(left)
            chunk = sock.recv(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > STATUS_MAX_BYTES:
                raise OSError(f"/status at {socket_path} answered more than {STATUS_MAX_BYTES} bytes")
            chunks.append(chunk)
    return read_status(*parse_http_response(b"".join(chunks)))


def probe_command(*, python: str, timeout: float = STATUS_TIMEOUT_SEC) -> str:
    """A shell command that runs one ``GET /status`` against ``SOCKET_PATH`` from inside
    the build sandbox with the interpreter *python* and prints the raw reply base64-encoded,
    for ``read_probe`` on the host. Base64 because an exec transport that decodes text may
    rewrite the reply's CRLF line ends. Standard library only."""
    script = (
        "import base64,socket,sys;"
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);"
        f"s.settimeout({timeout!r});"
        f"s.connect({SOCKET_PATH!r});"
        f"s.sendall({http_get_request('/status')!r});"
        "print(base64.b64encode(b''.join(iter(lambda:s.recv(65536),b''))).decode())"
    )
    return f"{shlex.quote(python)} -I -c {shlex.quote(script)}"


def read_probe(stdout: str) -> dict[str, Any]:
    """Parse ``probe_command``'s output into the ``/status`` dict; raises ``ValueError``
    on anything else."""
    try:
        raw = base64.b64decode(stdout.strip(), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"the probe printed something other than base64: "
                         f"{stdout[:200]!r}") from exc
    return read_status(*parse_http_response(raw))
