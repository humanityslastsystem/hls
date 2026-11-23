"""What the step wrapper does with a role's env file, for the local launcher, which has no
srun and no step wrapper: take the file (open it, then unlink it, before anything else),
hand it over on descriptor 3 or in the environment, remove the variables the step wrapper
removes (``hls.substrate.role_exec.dropped``, a container's with ``--env-fd``), and become the
command. It measures nothing; ``hls.substrate.role_exec`` does the rest on a cluster.

``python fake_step.py --env-file PATH [--env-fd] -- <argv...>``"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from hls.substrate.role_exec import _hand_over, drop_site_variables, take_env_file

p = argparse.ArgumentParser()
p.add_argument("--env-file", type=Path, required=True)
p.add_argument("--env-fd", action="store_true")
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
fd, problem = take_env_file(args.env_file)
if fd is None:
    sys.exit(f"fake step: {problem}")
drop_site_variables(os.environ, container=args.env_fd)
_hand_over(fd, on_fd=args.env_fd)
program = shutil.which(command[0]) or command[0]
os.execv(program, command)
