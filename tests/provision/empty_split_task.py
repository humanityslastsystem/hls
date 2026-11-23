"""A task whose test split is a fraction of its 20-row corpus that rounds to no row: ``resolve``
cannot see it, and the rows role refuses it, as a rerun would meet the same corpus."""

from __future__ import annotations

from hls.spec.datasource import Count, Fraction, SplitSpec
from tests.provision.tasks import data_source, task

TASK = task(source=data_source().model_copy(update={"split": SplitSpec(dev=Count(value=5),
                                                                       test=Fraction(value=0.01))}))
