"""hls -- curated public API for task authoring.

This module is the STABLE surface task authors should import from:

    from hls import TaskSpec, Workload, Baseline, ReferenceRelativeVerifier, ...

instead of reaching into the ~15 internal modules those types actually live
in (``hls.spec.taskspec``, ``hls.spec.workload``, ``hls.spec.baseline``,
``hls.verify.verifiers``, ...). Every name below is re-exported from exactly one
internal module; the docstring on the underlying class/function remains the source
of truth for behavior. This file only curates WHICH names are public and WHERE they come
from -- it adds no new behavior.

Every internal module remains directly importable:
``from hls.spec.taskspec import TaskSpec`` works as well.
This module is an additional, curated entry point layered on top, not a
replacement for the deep-import paths.

Deliberately NOT re-exported here (import these directly if you need them):
the substrate (``hls.substrate``), the stages and their pipelines
(``hls.stages``, ``hls.pipeline``), the records (``hls.obs``), the egress
proxy/index internals (``hls.egress.proxy``, ``hls.egress.prescan``) and the
viewer (``hls.viewer``) -- these are harness machinery, not task-authoring
vocabulary. ``Budget`` (the declarative wallclock/turns spec used by
``AgentProducerSpec``) IS exported, since tasks construct it directly.

Implementation note -- lazy re-exports (PEP 562): every name below is
resolved on first access via module-level ``__getattr__`` rather than an
eager ``from .x import Y`` at the top of this file. Nothing under
``src/hls`` imports the ``hls`` package root back (verified: no
``from hls import ...`` anywhere in the tree, only ``from hls.<submodule>
import ...``), so an eager re-export here would not actually cycle. But
laziness costs nothing to keep, and it makes three properties provable by
inspection rather than by reasoning about a ~30-module import graph:

  1. ``import hls`` alone never imports any submodule, so it can never
     cycle at package-import time, no matter what a submodule's transitive
     imports look like today or grow to look like tomorrow.
  2. ``import hls`` pays zero extra import-time cost beyond defining this
     module; the cost of resolving (e.g.) ``TaskSpec`` is deferred to the
     first ``hls.TaskSpec`` / ``from hls import TaskSpec`` access, same as
     importing ``hls.spec.taskspec`` directly today.
  3. A future accidental heavy top-level import in some deep module (e.g.
     someone adding a module-level ``import torch``) cannot make a bare
     ``import hls`` slow or fail; only actually touching the affected name
     would.

``__all__`` still lists every public name (for ``from hls import *``,
``dir(hls)``, and static analysis / IDE autocomplete); ``TYPE_CHECKING``
imports below give type checkers the real symbols without executing them
at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Real imports, executed only by type checkers (mypy/pyright) -- never at
    # runtime. Keep this block's names in lockstep with _EXPORTS below.
    from hls.artifact.artifact import (
        AgentProducedArtifact,
        AgentProducerSpec,
        Artifact,
        Budget,
    )
    from hls.core.decode import DecodeConfig, Greedy, Sample
    from hls.core.model import (
        AcceleratorSpec,
        Hardware,
        InterconnectSpec,
        ModelSpec,
    )
    from hls.egress.policy import AccessList, NetworkPolicy
    from hls.prompt import ImplementerBrief, JinjaPromptTemplate, PromptTemplate
    from hls.reference.references import HFReference, Reference, VLLMReference
    from hls.scoring import (
        ConstantSoL,
        GoodputScorer,
        RequestUnit,
        Scorer,
        SoLCeiling,
        TokenUnit,
        Unit,
    )
    from hls.spec.baseline import Baseline
    from hls.spec.contracts import ArtifactContract, RequestResponseContract
    from hls.spec.datasets import Dataset, FimSessionDataset, HFDataset, SafimDataset
    from hls.spec.datasource import Count, DataSource, Fraction, SplitSize, SplitSpec
    from hls.spec.drivers import Driver
    from hls.spec.taskspec import TaskSpec
    from hls.spec.tenant import Tenant
    from hls.spec.wire import (OpenAIChatProtocol, OpenAICompletionProtocol,
                               WireProtocol)
    from hls.spec.workload import (
        AllAtOnce,
        ArrivalProcess,
        ConstantInterTurn,
        DeterministicArrival,
        Gamma,
        InterTurnDelay,
        NoDelay,
        Normal,
        NoSLO,
        OutputLengthCap,
        Poisson,
        ShortOutputCap,
        SLO,
        StreamingLatencySLO,
        Uncapped,
        Uniform,
        Workload,
    )
    from hls.verify import (
        ContainsMatchMetric,
        CoqaF1Metric,
        LetterMatchMetric,
        MetricVerifier,
        ReferenceRelativeVerifier,
        RolloutMetric,
        TokenFaithfulnessVerifier,
        Verifier,
    )
    from hls.verify.metrics import Gsm8kExactMatch, HotpotEmF1, IFEvalConstraint


__all__ = [
    # -- artifact / contestant config (hls.artifact.artifact) --
    "Artifact",
    "AgentProducedArtifact",
    "AgentProducerSpec",
    "Budget",
    # -- egress policy (hls.egress.policy) --
    "NetworkPolicy",
    "AccessList",
    # -- baseline (hls.spec.baseline) --
    "Baseline",
    # -- contract + wire protocol (hls.spec.contracts, hls.spec.wire) --
    "ArtifactContract",
    "RequestResponseContract",
    "WireProtocol",
    "OpenAIChatProtocol",
    "OpenAICompletionProtocol",
    # -- dataset (hls.spec.datasets) --
    "Dataset",
    "HFDataset",
    "FimSessionDataset",
    "SafimDataset",
    # -- data source / split sizing (hls.spec.datasource) --
    "DataSource",
    "SplitSpec",
    "SplitSize",
    "Count",
    "Fraction",
    # -- task spec (hls.spec.taskspec) --
    "TaskSpec",
    # -- tenant (hls.spec.tenant) --
    "Tenant",
    # -- driver (hls.spec.drivers) --
    "Driver",
    # -- workload: arrival / inter-turn-delay / output-cap / SLO families
    #    (hls.spec.workload) --
    "Workload",
    "ArrivalProcess",
    "AllAtOnce",
    "DeterministicArrival",
    "Poisson",
    "Gamma",
    "Uniform",
    "Normal",
    "InterTurnDelay",
    "NoDelay",
    "ConstantInterTurn",
    "OutputLengthCap",
    "Uncapped",
    "ShortOutputCap",
    "SLO",
    "NoSLO",
    "StreamingLatencySLO",
    # -- prompting / implementer brief (hls.prompt) --
    "PromptTemplate",
    "ImplementerBrief",
    "JinjaPromptTemplate",
    # -- decode config (hls.core.decode) --
    "DecodeConfig",
    "Greedy",
    "Sample",
    # -- model + hardware (hls.core.model) --
    "ModelSpec",
    "AcceleratorSpec",
    "Hardware",
    "InterconnectSpec",
    # -- reference oracles (hls.reference.references) --
    "Reference",
    "HFReference",
    "VLLMReference",
    # -- performance scoring (hls.scoring) --
    "Scorer",
    "GoodputScorer",
    "Unit",
    "RequestUnit",
    "TokenUnit",
    "SoLCeiling",
    "ConstantSoL",
    # -- correctness verifiers + metrics (hls.verify, hls.verify.metrics) --
    "Verifier",
    "RolloutMetric",
    "MetricVerifier",
    "ReferenceRelativeVerifier",
    "TokenFaithfulnessVerifier",
    "LetterMatchMetric",
    "ContainsMatchMetric",
    "CoqaF1Metric",
    "Gsm8kExactMatch",
    "HotpotEmF1",
    "IFEvalConstraint",
]


# name -> the internal module that defines it. Single source of truth for the
# lazy __getattr__ below; keep in lockstep with __all__ and the TYPE_CHECKING
# block above (all three must agree on exactly this set of names).
_EXPORTS = {
    # artifact / contestant config
    "Artifact": "hls.artifact.artifact",
    "AgentProducedArtifact": "hls.artifact.artifact",
    "AgentProducerSpec": "hls.artifact.artifact",
    "Budget": "hls.artifact.artifact",
    # egress policy
    "NetworkPolicy": "hls.egress.policy",
    "AccessList": "hls.egress.policy",
    # baseline
    "Baseline": "hls.spec.baseline",
    # contract + wire protocol
    "ArtifactContract": "hls.spec.contracts",
    "RequestResponseContract": "hls.spec.contracts",
    "WireProtocol": "hls.spec.wire",
    "OpenAIChatProtocol": "hls.spec.wire",
    "OpenAICompletionProtocol": "hls.spec.wire",
    # dataset
    "Dataset": "hls.spec.datasets",
    "HFDataset": "hls.spec.datasets",
    "FimSessionDataset": "hls.spec.datasets",
    "SafimDataset": "hls.spec.datasets",
    # data source / split sizing
    "DataSource": "hls.spec.datasource",
    "SplitSpec": "hls.spec.datasource",
    "SplitSize": "hls.spec.datasource",
    "Count": "hls.spec.datasource",
    "Fraction": "hls.spec.datasource",
    # task spec
    "TaskSpec": "hls.spec.taskspec",
    # tenant
    "Tenant": "hls.spec.tenant",
    # driver
    "Driver": "hls.spec.drivers",
    # workload families
    "Workload": "hls.spec.workload",
    "ArrivalProcess": "hls.spec.workload",
    "AllAtOnce": "hls.spec.workload",
    "DeterministicArrival": "hls.spec.workload",
    "Poisson": "hls.spec.workload",
    "Gamma": "hls.spec.workload",
    "Uniform": "hls.spec.workload",
    "Normal": "hls.spec.workload",
    "InterTurnDelay": "hls.spec.workload",
    "NoDelay": "hls.spec.workload",
    "ConstantInterTurn": "hls.spec.workload",
    "OutputLengthCap": "hls.spec.workload",
    "Uncapped": "hls.spec.workload",
    "ShortOutputCap": "hls.spec.workload",
    "SLO": "hls.spec.workload",
    "NoSLO": "hls.spec.workload",
    "StreamingLatencySLO": "hls.spec.workload",
    # prompting / implementer brief
    "PromptTemplate": "hls.prompt",
    "ImplementerBrief": "hls.prompt",
    "JinjaPromptTemplate": "hls.prompt",
    # decode config
    "DecodeConfig": "hls.core.decode",
    "Greedy": "hls.core.decode",
    "Sample": "hls.core.decode",
    # model + hardware
    "ModelSpec": "hls.core.model",
    "AcceleratorSpec": "hls.core.model",
    "Hardware": "hls.core.model",
    "InterconnectSpec": "hls.core.model",
    # reference oracles
    "Reference": "hls.reference.references",
    "HFReference": "hls.reference.references",
    "VLLMReference": "hls.reference.references",
    # performance scoring
    "Scorer": "hls.scoring",
    "GoodputScorer": "hls.scoring",
    "Unit": "hls.scoring",
    "RequestUnit": "hls.scoring",
    "TokenUnit": "hls.scoring",
    "SoLCeiling": "hls.scoring",
    "ConstantSoL": "hls.scoring",
    # correctness verifiers + metrics
    "Verifier": "hls.verify",
    "RolloutMetric": "hls.verify",
    "MetricVerifier": "hls.verify",
    "ReferenceRelativeVerifier": "hls.verify",
    "TokenFaithfulnessVerifier": "hls.verify",
    "LetterMatchMetric": "hls.verify",
    "ContainsMatchMetric": "hls.verify",
    "CoqaF1Metric": "hls.verify",
    "Gsm8kExactMatch": "hls.verify.metrics",
    "HotpotEmF1": "hls.verify.metrics",
    "IFEvalConstraint": "hls.verify.metrics",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute resolution: import the owning submodule (and only
    that submodule) on first access, cache the result on this module so repeat
    access is a plain attribute lookup, and return it."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
