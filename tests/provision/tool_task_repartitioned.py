"""``tool_task``'s task with one test row fewer: the same model and tool bundle over another
partition, so its rows and its dev bundle are new while its full bundle is not."""

from __future__ import annotations

from dataclasses import replace

from hls.spec.datasource import Count, DataSource, SplitSpec
from tests.provision.tasks import dataset, task
from tests.provision.tool_task import TinyAssets
from tests.unit.test_trajectory_driver import _agent

SOURCE = DataSource(dataset=dataset(), split=SplitSpec(dev=Count(value=5), test=Count(value=3)),
                    removed_rows=("8",), dev_reference={"prompt_prefix.txt": "P"})
TASK = task(agent=replace(_agent(), assets=TinyAssets()), source=SOURCE)
