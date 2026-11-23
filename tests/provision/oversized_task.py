"""A task whose split asks for more rows than its 20-row corpus has: ``resolve`` cannot see
it, and the rows role refuses it, as a rerun would meet the same corpus."""

from __future__ import annotations

from hls.spec.datasource import Count, SplitSpec
from tests.provision.tasks import data_source, task

TASK = task(source=data_source().model_copy(update={"split": SplitSpec(dev=Count(value=15), test=Count(value=10))}))
