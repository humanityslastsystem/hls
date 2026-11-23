from __future__ import annotations

import subprocess

from tests.substrate import fake_pasta


def test_the_fake_interpreter_runs_the_cache_role_and_refuses_anything_else(tmp_path):
    # It stands in for a plan's interpreter, so a seal that ran an unexpected program fails loudly.
    fake_pasta.install(tmp_path)
    python = str(tmp_path / "python")
    refused = subprocess.run([python, "-m", "json.tool", "--help"], capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2 and "unexpected" in refused.stderr
    role = subprocess.run([python, "-m", "hls.store.cache_role", "--help"], capture_output=True, text=True,
                          timeout=60)
    assert role.returncode == 0, role.stderr
