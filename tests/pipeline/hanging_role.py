"""What a test runs in place of a role program: it sleeps until it is killed, as a copy of a large
entry would run on, or, when ``HANGING_ROLE_EXIT`` is set, exits at once with that status having
written nothing."""

from __future__ import annotations

import os
import sys
import time

if "HANGING_ROLE_EXIT" in os.environ:
    sys.exit(int(os.environ["HANGING_ROLE_EXIT"]))
time.sleep(120)
