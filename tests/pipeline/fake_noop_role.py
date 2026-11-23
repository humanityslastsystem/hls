"""What a no-op role reports, run by the local launcher in place of hls.substrate.noop_role: it
reports the fake GPU UUIDs the launcher assigned, as CUDA would show them (or no view of
them with ``--no-view``), then sleeps."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

from hls.substrate.noop_role import NoopRoleReport
from hls.store.manifest import report_outputs

p = argparse.ArgumentParser()
p.add_argument("--uuids", default="")
p.add_argument("--no-view", action="store_true")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--outputs-out", type=Path, required=True)
p.add_argument("--sleep-sec", type=float, default=0.0)
args = p.parse_args()
uuids = None if args.no_view else [u for u in args.uuids.split(",") if u]
report = NoopRoleReport(host=socket.gethostname(), cuda_uuids=uuids, cuda_visible_devices=args.uuids or None)
args.out.write_text(report.model_dump_json())
report_outputs(args.outputs_out, args.out.parent, args.out)
time.sleep(args.sleep_sec)
