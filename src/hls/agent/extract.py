"""Harbor's reading of the agent's run, its trajectory and its spend, from the transcript's copy,
in a harness role of its own.

The agent_build stage runs this program as a host harness role once the transcript role has
copied the agent's logs, with that copy as ``--logs`` and the attempt's ``outputs/transcript``
as ``--out``::

    python -m hls.agent.extract --harness H --model M --logs DIR --out DIR --record FILE \\
                                --outputs-out FILE

Harbor's adapter for harness *H* (``hls.arms``), given the model string *M* the arm ran, reads
the agent's session files from the copy (``populate_context_post_run``) into an ATIF trajectory,
which it writes as ``--out``'s ``trajectory.json``, beside the transcript's index, where no name
the agent gives a file of its logs can meet it, and finds the agent's spend: its tokens and
their cost as harbor prices it. The program writes the spend, and whether harbor wrote a
trajectory, as ``--record`` (``ExtractionRecord``), and reports the trajectory's sha256 and size
in ``--outputs-out``, keyed by its path under the outputs directory (``--out`` is a directory
of it), as ``hls.store.manifest.report_outputs`` records them, so the stage publishes it by its
writer's digest.

The copy holds what the agent wrote, up to ``hls.agent.transcript.MAX_BYTES``, and harbor's
readers hold every event of a session in memory at once, so a crafted transcript can take many
times its size. The program therefore runs in a role of its
own, with its own memory limit (``hls.pipeline.pipelines.EXTRACT_SHAPE``) and time
(``hls.stages.agent_build.EXTRACT_SEC``), after the commit point: a role that runs out of its
memory, fails or runs past its time leaves the trajectory and spend unavailable, with why, and
the build goes on to be packaged and graded. The runner parses nothing the agent wrote.

Exit 0 once the record is written and the trajectory, when harbor wrote one, reported; 1 on
anything else, with the reason on stderr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hls.arms import ADAPTERS, adapter_named

#: The trajectory harbor writes, in ``--out``.
TRAJECTORY_FILE = "trajectory.json"
#: The most bytes of harbor's trajectory a build keeps: 256 MiB. A larger one is withheld: removed,
#: reported as no output, and the record says why. The stage holds what is reported to it too.
TRAJECTORY_MAX_BYTES = 256 * 2**20


class Usage(BaseModel):
    """What harbor read of the agent's spend from its transcript, the fields the
    ``build.finished`` event carries: tokens as the provider reported them, and their cost as
    harbor prices it from its tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Input tokens, the ones read from the provider's cache included; None when harbor read none.
    n_input_tokens: int | None
    #: Of those, the tokens read from the provider's cache; None when harbor read none.
    n_cache_tokens: int | None
    #: Output tokens; None when harbor read none.
    n_output_tokens: int | None
    #: The cost harbor priced, in USD; None when it priced none.
    cost_usd: float | None


class ExtractionRecord(BaseModel):
    """What the program read: whether harbor wrote a trajectory, and the spend it found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Whether harbor wrote ``trajectory.json`` and the program reports it: harbor writes none
    #: when the copy holds no session it can read, and the program withholds one past
    #: ``TRAJECTORY_MAX_BYTES``.
    trajectory: bool
    #: Why the trajectory harbor wrote is withheld: it took more than ``TRAJECTORY_MAX_BYTES``;
    #: None otherwise.
    trajectory_withheld: str | None
    #: The spend harbor found; None when it found none.
    usage: Usage | None


def _agent(harness: str, model: str, logs: Path, out: Path):
    """Harbor's adapter for *harness*, reading the copy *logs* and writing its trajectory into
    *out*: the one the harness's adapter names (``HarnessAdapter.log_reader``), with nothing but
    its model string, since reading the logs needs none of the arm's route or settings. Raise
    ArmError for a harness no adapter runs."""
    agent = adapter_named(harness).log_reader(model, logs)
    agent.trajectory_dir = out
    return agent


def extract(harness: str, model: str, logs: Path, out: Path) -> ExtractionRecord:
    """Have harbor read the agent's run from *logs*, writing its trajectory into *out*, and
    return what it read. A trajectory past ``TRAJECTORY_MAX_BYTES`` is removed and withheld."""
    from harbor.models.agent.context import AgentContext

    context = AgentContext()
    _agent(harness, model, logs, out).populate_context_post_run(context)
    usage = Usage(n_input_tokens=context.n_input_tokens, n_cache_tokens=context.n_cache_tokens,
                  n_output_tokens=context.n_output_tokens, cost_usd=context.cost_usd)
    found = any(value is not None for value in usage.model_dump().values())
    trajectory, withheld = out / TRAJECTORY_FILE, None
    if trajectory.is_file() and (size := trajectory.stat().st_size) > TRAJECTORY_MAX_BYTES:
        trajectory.unlink()
        withheld = f"harbor's trajectory took {size} bytes, past the {TRAJECTORY_MAX_BYTES} the build keeps"
    return ExtractionRecord(trajectory=trajectory.is_file(), trajectory_withheld=withheld,
                            usage=usage if found else None)


def main(argv: list[str] | None = None) -> int:
    from hls.store.manifest import report_outputs

    ap = argparse.ArgumentParser(prog="hls.agent.extract", description="Read the agent's trajectory and spend.")
    ap.add_argument("--harness", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--outputs-out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        record = extract(args.harness, args.model, args.logs, args.out)
        trajectory = [args.out / TRAJECTORY_FILE] if record.trajectory else []
        report_outputs(args.outputs_out, args.out.parent, *trajectory)
        args.record.write_text(record.model_dump_json())
    except Exception as e:  # noqa: BLE001 -- any failure is the role's, recorded by the stage
        print(f"hls-extract: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    written = "written" if record.trajectory else record.trajectory_withheld or "not written"
    print(f"hls-extract: trajectory {written}, spend {'read' if record.usage is not None else 'not read'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
