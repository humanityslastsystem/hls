"""Small tasks for the provision tests. ``TASK`` is what ``--task-module tests.provision.tasks``
loads: one tenant over a 20-row corpus, with a removal list and a dev reference file, so
the rows exercise every path of ``hls.provision.rows``."""

from __future__ import annotations

from types import SimpleNamespace

from datasets import Dataset as ArrowDataset

from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.prompt import JinjaPromptTemplate
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier

#: A model commit id.
SHA = "0123456789abcdef0123456789abcdef01234567"
#: A dataset commit id.
DATA_SHA = "89abcdef0123456789abcdef0123456789abcdef"


def corpus(n: int = 20) -> ArrowDataset:
    return ArrowDataset.from_list([{"q": i} for i in range(n)])


def dataset(*, revision: str = DATA_SHA, id_column: str = "q") -> HFDataset:
    return HFDataset(path="corp", revision=revision, source_split="test", id_column=id_column,
                     prompt=JinjaPromptTemplate(template="{{" + id_column + "}}"),
                     target=JinjaPromptTemplate(template="{{" + id_column + "}}"))


def data_source(*, revision: str = DATA_SHA, removed: tuple[str, ...] = ("8",),
                dev_reference: dict[str, str] | None = None, dev_dataset=None) -> DataSource:
    return DataSource(dataset=dataset(revision=revision), dev_dataset=dev_dataset,
                      split=SplitSpec(dev=Count(value=5), test=Count(value=4)), removed_rows=removed,
                      dev_reference={"prompt_prefix.txt": "P"} if dev_reference is None else dev_reference)


def _spec_parts(revision: str) -> dict:
    return dict(
        id="T-provision", model=ModelSpec(base_checkpoint="org/model", revision=revision, arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="CPU", count=1, memory_gb_per_device=1),
                          cpu_count=1, memory_gb=1, storage_gb=1),
        contract=RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                         output_modalities=frozenset({"text"}), transport="http"),
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)), seed=1)


def _workload() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=NoSLO())


def _verifiers() -> list:
    return [MetricVerifier(metric=LetterMatchMetric(), min_value=0.5)]


def task(*, revision: str = SHA, agent=None, removed: tuple[str, ...] = ("8",),
         dev_reference: dict[str, str] | None = None, data_revision: str = DATA_SHA,
         dev_dataset=None, source: DataSource | None = None) -> TaskSpec:
    """One tenant (``t0``) over a 5/4 split of a 20-row corpus, with row ``8`` on the removal
    list and one dev reference file, or over ``source`` when one is given."""
    if source is None:
        source = data_source(revision=data_revision, removed=removed, dev_reference=dev_reference,
                             dev_dataset=dev_dataset)
    return TaskSpec(**_spec_parts(revision), workload=_workload(), verifiers=_verifiers(),
                    data_source=source, agent=agent)


def two_tenant_task(*, agent=None, revisions: tuple[str, str] = (DATA_SHA, DATA_SHA)) -> TaskSpec:
    """Two tenants, ``t0`` and ``t1``, over the same data source but for the corpus revision each
    takes from *revisions*; by default one and the same, so each row is written twice."""
    tenants = [Tenant(id=tid, data_source=data_source(revision=revision), workload=_workload(),
                      verifiers=_verifiers())
               for tid, revision in zip(("t0", "t1"), revisions, strict=True)]
    return TaskSpec(**_spec_parts(SHA), workload=None, verifiers=None, tenants=tenants, agent=agent)


def no_data_task():
    """A stand-in task whose only tenant has no data source."""
    tenant = SimpleNamespace(id="t0", data_source=None)
    return SimpleNamespace(model=SimpleNamespace(base_checkpoint="org/model", revision=SHA),
                           effective_tenants=lambda: [tenant], agent=None)


TASK = task()
