"""Stands in for ``hls.agent.extract`` in the agent build's tests: sleeps until killed, as the
extraction role does while it reads a huge transcript, with the part of a trajectory it had
written in ``--out``."""
import sys
import time
from pathlib import Path

out = Path(sys.argv[sys.argv.index("--out") + 1])
(out / "trajectory.json").write_text('{"schema_version": "ATIF-v1.2", "steps": [')
while True:
    time.sleep(60)
