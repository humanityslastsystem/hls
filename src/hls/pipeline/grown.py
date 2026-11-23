"""What each grower may add to a plan: the names of the stages it adds as they are, and the
patterns of the numbered names it adds.

The module that makes a grower's stage names declares them here (``declare_grower``), apart from
the function that decides what the grower adds (``hls.pipeline.growth``): a calibration's are
declared by ``hls.stages.calibration``, which the stage registry imports. So the planner, which
imports the stage registry, checks a plan's grower and every plan's ``[resources]`` and
``[run.retry]`` tables against what any grower may add (``grown_stages``) without importing a
grower, and ``hls.pipeline.growth.grow`` refuses a stage its grower did not declare."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from hls.store.layout import check_stage


@dataclass(frozen=True)
class GrownStages:
    """The stages a grower may add: settings tables may name them before any plan holds them."""

    #: Names it adds as they are, such as a calibration's ``aggregate``.
    names: frozenset[str] = frozenset()
    #: Patterns, matched whole, of the numbered names it adds, such as ``score_[1-9][0-9]*``.
    patterns: tuple[str, ...] = ()

    def holds(self, name: str) -> bool:
        """Whether a grower declared so may add a stage called *name*."""
        return name in self.names or any(re.fullmatch(p, name) for p in self.patterns)


#: What each grower may add, by grower name (``declare_grower``).
GROWN_STAGES: dict[str, GrownStages] = {}


def declare_grower(name: str, *, names: Sequence[str] = (), patterns: Sequence[str] = ()) -> None:
    """Declare grower *name* with the stage *names* it adds and the *patterns* of the numbered
    names it adds. Raise ValueError for a name or pattern that cannot be a stage's, or for a
    second declaration of *name* that says something else."""
    for n in names:
        check_stage(n)
    for p in patterns:
        re.compile(p)
    declared = GrownStages(names=frozenset(names), patterns=tuple(patterns))
    if GROWN_STAGES.get(name, declared) != declared:
        raise ValueError(f"grower {name} is declared twice, with other stages")
    GROWN_STAGES[name] = declared


def grown_stages(grower: str | None = None) -> GrownStages:
    """What *grower* may add, or with None what any declared grower may add. Raise ValueError
    for a grower that is not declared."""
    if grower is not None:
        if grower not in GROWN_STAGES:
            raise ValueError(f"grower {grower} is not declared")
        return GROWN_STAGES[grower]
    every = GROWN_STAGES.values()
    return GrownStages(names=frozenset().union(*(g.names for g in every)),
                       patterns=tuple(dict.fromkeys(p for g in every for p in g.patterns)))
