"""Stands in for ``hls.agent.extract`` in the agent build's tests: fails, as the role does when
harbor raises reading the copy, leaving the part of a trajectory it had written in ``--out``."""
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
(out / "trajectory.json").write_text('{"schema_version": "ATIF-v1.2", "steps": [')
print("hls-extract: RuntimeError: boom", file=sys.stderr)
sys.exit(1)
