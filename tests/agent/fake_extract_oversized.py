"""Stands in for ``hls.agent.extract`` in the agent build's tests: ends cleanly with a record
longer than the stage reads (``EXTRACT_RECORD_MAX_BYTES``), whole JSON padded with spaces, and
a trajectory beside it."""
import json
import sys
from pathlib import Path

from hls.stages.agent_build import EXTRACT_RECORD_MAX_BYTES

out = Path(sys.argv[sys.argv.index("--out") + 1])
record = Path(sys.argv[sys.argv.index("--record") + 1])
(out / "trajectory.json").write_text('{"schema_version": "ATIF-v1.2", "steps": []}')
record.write_text(json.dumps({"trajectory": False, "trajectory_withheld": None, "usage": None})
                  + " " * EXTRACT_RECORD_MAX_BYTES)
