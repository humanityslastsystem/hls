"""SPA build helpers for the HLS viewer.

harbor-free: zero harbor.* imports allowed in this module.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# The package lives at src/hls/viewer/; the repo root is three parents up.
_VIEWER_PKG_DIR = Path(__file__).parent
_REPO_ROOT = _VIEWER_PKG_DIR.parent.parent.parent  # src/hls/viewer -> src/hls -> src -> repo root
_APPS_VIEWER_DIR = _REPO_ROOT / "apps" / "viewer"
_STATIC_DIR = _VIEWER_PKG_DIR / "static"


def static_dir() -> Path | None:
    """Return the path to the built SPA static assets, or None if not yet built."""
    if _STATIC_DIR.exists():
        return _STATIC_DIR
    return None


def build_spa() -> bool:
    """Build the SPA in apps/viewer/ and copy the output to src/hls/viewer/static.

    Runs ``bun install && bun run build`` in apps/viewer/, then copies
    ``apps/viewer/build/client`` -> ``src/hls/viewer/static``.

    Returns True on success, False when apps/viewer is missing or bun is unavailable
    (logs a warning; never raises).
    """
    if not _APPS_VIEWER_DIR.exists():
        logger.warning("build_spa: apps/viewer not found at %s; skipping SPA build", _APPS_VIEWER_DIR)
        return False

    bun = shutil.which("bun")
    if bun is None:
        logger.warning("build_spa: bun not found on PATH; skipping SPA build")
        return False

    try:
        result = subprocess.run(
            [bun, "install"],
            cwd=str(_APPS_VIEWER_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("build_spa: bun install failed:\n%s", result.stderr)
            return False

        result = subprocess.run(
            [bun, "run", "build"],
            cwd=str(_APPS_VIEWER_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("build_spa: bun run build failed:\n%s", result.stderr)
            return False
    except OSError as exc:
        logger.warning("build_spa: failed to run bun: %s", exc)
        return False

    client_dir = _APPS_VIEWER_DIR / "build" / "client"
    if not client_dir.exists():
        logger.warning("build_spa: expected output %s not found after build", client_dir)
        return False

    if _STATIC_DIR.exists():
        shutil.rmtree(_STATIC_DIR)
    shutil.copytree(str(client_dir), str(_STATIC_DIR))
    logger.debug("build_spa: static assets written to %s", _STATIC_DIR)
    return True


def find_available_port(host: str, start: int, end: int) -> int | None:
    """Return the first TCP port in [start, end] that can be bound on host, else None."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                logger.debug("find_available_port: port %d on %s is taken", port, host)
    return None
