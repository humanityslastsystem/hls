"""Plans for tests, built by the real planner from the no-op pipelines."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hls.pipeline.checkout import CodeState
from hls.pipeline.commands import known_task_ids
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK, noop_two
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.settings import Settings
from hls.store.layout import InstanceLayout
from hls.substrate.secrets import SecretsFile
from hls.substrate.versions import ToolVersions

CODE = CodeState(checkout=Path("/c/hls"), hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None)
#: The tool versions the tests' runners record with their grants, as an el8 node prints them.
VERSIONS = ToolVersions(slurm="slurm 23.02.7", apptainer="apptainer version 1.3.4-1.el8",
                        pasta="pasta 2024_09_06.6b38f07", kernel="4.18.0-513.24.1.el8_9.x86_64")


def pin_harbor(monkeypatch, code: CodeState) -> None:
    """Have HEAD's tree in *code*'s checkout pin harbor at *code*'s harbor commit, as a clean
    checkout's does, for a login node whose checkout state a test stands in (``code_state``): the
    committed-tree check (``hls.pipeline.preflight.check_committed_tree``) then judges *code* as it
    stands. Every other path of every checkout is read as it is."""
    from hls.pipeline import preflight

    read = preflight._head_entry

    def head_entry(root: Path, path: str) -> tuple[str, str] | None:
        if path == "harbor" and Path(root) == code.checkout:
            return "commit", code.harbor_commit
        return read(root, path)

    monkeypatch.setattr(preflight, "_head_entry", head_entry)


def settings(tmp_path: Path, *, site: dict[str, Any] | None = None, resources: dict[str, Any] | None = None,
             sandbox_stores: dict[str, str] | None = None, **run: Any) -> Settings:
    """Fast settings for tests: one-second heartbeats and polls, and no launch headroom in an
    allocation's time unless *site* gives an allowance (``step_launch_allowance_sec``)."""
    return Settings.model_validate({
        "site": {"partition": "gpu-a", "account": "acct", "store_root": str(tmp_path / "store"),
                 "node_scratch": str(tmp_path / "node"), "secrets_path": "/x", "step_launch_allowance_sec": 0,
                 **(site or {})},
        "run": {"heartbeat_sec": 1, "lease_ttl_sec": 30, "poll_sec": 1, "allocation_overhead_sec": 360,
                "allocation_dead_after_sec": 10, **run},
        "resources": resources or {},
        "sandbox_stores": sandbox_stores or {},
    })


def noop_plan(tmp_path: Path, *, decls: Sequence[StageDecl] | None = None, instance: str = "r1",
              site: dict[str, Any] | None = None, resources: dict[str, Any] | None = None,
              secrets: SecretsFile | None = None, known_patterns: Sequence[str] = (),
              sandbox_stores: dict[str, str] | None = None, **run: Any) -> Plan:
    """A first plan of the two-stage no-op pipeline, or of *decls*, granted what *secrets*
    grants; *known_patterns* are numbered stage names settings tables may name. Its resources
    tables may be keyed by any task of this checkout (``known_task_ids``), and its site names
    *sandbox_stores*."""
    decls = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600) if decls is None else decls
    return build_plan(decls=decls, task=NOOP_TASK, pipeline="noop2", layout=InstanceLayout.rollout(NOOP_TASK, instance),
                      settings=settings(tmp_path, site=site, resources=resources, sandbox_stores=sandbox_stores,
                                        **run), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-noop", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                      now=1000.0, secrets=secrets, known_patterns=known_patterns)
