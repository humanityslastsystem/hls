"""The provision tests' task with a tool bundle: ``tests.provision.tasks``'s task, whose agent
declares a tiny ``ToolAssets`` that writes one manifest file per variant."""

from __future__ import annotations

import json
from dataclasses import replace

from hls.spec.agent import ToolAssets
from tests.provision.tasks import task
from tests.unit.test_trajectory_driver import _agent


class TinyAssets(ToolAssets):
    """A bundle whose one file says which variant it is: the dev one names the dev rows it was
    built from, and the full one only its pinned sources, as the ``ToolAssets`` contract asks
    of a bundle cached by ``identity()`` alone."""

    name = "tiny"

    def identity(self) -> str:
        return "tiny-v1"

    def hub_revisions(self) -> list[tuple[str, str]]:
        return []

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        content = {"variant": "dev", "rows": dev_row_ids} if variant == "dev" else {"variant": "full",
                                                                                   "identity": self.identity()}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(content))


TASK = task(agent=replace(_agent(), assets=TinyAssets()))
