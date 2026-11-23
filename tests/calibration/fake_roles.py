"""``python -m tests.calibration.fake_roles <role> ...``: the calibration role programs
(``hls.verify.calibration_roles``) with the engines of ``tests.calibration.engines`` installed.
The stage and end-to-end tests point the stages at it (``hls.stages.calibration.ROLE_MODULE``),
so every role runs its real program, in a fake container or on the host, with fake engines.
With ``RELEASE_SEC_ENV`` set, its engine host gives each step of a release that many seconds, so a
test whose release hangs waits seconds, not the host's own bound."""

from __future__ import annotations

import functools
import os
import sys

from hls.verify import calibration_roles
from tests.calibration import engines

#: The variable that names the seconds the engine host gives each step of a release.
RELEASE_SEC_ENV = "FAKE_ROLES_RELEASE_SEC"

if __name__ == "__main__":
    engines.install()
    if (seconds := os.environ.get(RELEASE_SEC_ENV)) is not None:
        calibration_roles.EngineHost = functools.partial(calibration_roles.EngineHost, release_sec=float(seconds))
    sys.exit(calibration_roles.main())
