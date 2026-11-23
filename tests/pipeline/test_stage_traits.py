"""Every trait a stage kind declares is declared on ``hls.pipeline.stage.Stage`` with its type and
default, and ``register`` refuses a class whose traits are not as declared: a misspelled trait
would otherwise leave the real one at its default, which for the contestant-file traits flips
how finalize attributes a secret it finds."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import hls.stages.registry  # registers every stage implementation
from hls.pipeline.stage import STAGES, Stage, StageFault, most_roles, register, retry_policy, stage_class
from hls.settings import RetryPolicy, RunSettings

#: The traits a stage kind must set itself, with no default on ``Stage``.
REQUIRED = ("compressed_files", "unpublished_containers", "published_dirs")
#: The hls package's sources.
SRC = Path(hls.stages.registry.__file__).resolve().parents[1]


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _stage(kind: str, *, base: type = Stage, omit: tuple[str, ...] = (), **attrs: Any) -> type:
    """A stage class of *kind* that declares every required trait and method, less those *omit*
    names, with *attrs* set beside them."""
    members: dict[str, Any] = {
        "kind": kind, "params_model": _Params, "compressed_files": (), "unpublished_containers": (),
        "published_dirs": (), "check": classmethod(lambda cls, shape, params, *, runner_memory_gib: None),
        "max_roles": classmethod(lambda cls, params: 1), "run": lambda self, ctx: {}}
    for name in omit:
        del members[name]
    return type(f"Stage_{kind}", (base,), members | attrs)


@pytest.mark.parametrize(("name", "value"), [
    ("runs_contestant_codes", True),
    ("raw_output", ("transcript/",)),
    ("grants_reaches_contestant", True),
    ("contestant_output", ("verdict.json",)),
    ("needs_secrets_registry", True),
    ("max_role", classmethod(lambda cls, params: 1)),
])
def test_a_stage_class_with_a_misspelled_trait_is_refused(name, value):
    """A class that sets a public name ``Stage`` does not declare is refused when it registers,
    naming the attribute, so the trait it meant is never read at its default."""
    cls = _stage(f"misspelled_{name}", **{name: value})
    with pytest.raises(TypeError, match=name):
        register(cls)
    assert cls.kind not in STAGES


def test_a_misspelled_trait_on_a_base_stage_class_is_refused_for_its_subclass():
    """A trait set on a stage class another stage class inherits counts as the subclass's own."""
    base = _stage("misspelled_base", runs_contestant_codes=True)
    child = type("Child", (base,), {"kind": "misspelled_child"})
    with pytest.raises(TypeError, match="runs_contestant_codes"):
        register(child)
    assert "misspelled_child" not in STAGES


@pytest.mark.parametrize(("name", "value"), [
    ("runs_contestant_code", "yes"),
    ("grants_reach_contestant", 1),
    ("needs_secret_registry", None),
    ("compressed_files", "outputs/source.bundle"),
    ("compressed_files", (b"outputs/source.bundle",)),
    ("raw_outputs", "transcript/"),
    ("contestant_outputs", (b"verdict.json",)),
    ("contestant_log_roles", ["verify"]),
    ("decides_contestant_faults", "true"),
    ("contestant_host_roles", 0),
    ("default_retries", -1),
    ("default_retries", True),
    ("kind", ""),
    ("params_model", dict),
])
def test_a_trait_of_the_wrong_type_is_refused(name, value):
    """A trait whose value is not of its declared type is refused, a bare string of paths among
    them, which would be read as its characters."""
    cls = _stage("wrong_type")
    setattr(cls, name, value)
    with pytest.raises(TypeError, match=name):
        register(cls)
    assert "wrong_type" not in STAGES and "" not in STAGES


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_stage_kind_that_does_not_declare_its_containers_is_refused(missing):
    """Finalize keeps a stage's compressed files out of any export by what its kind declares
    (``Stage.compressed_files``, ``Stage.unpublished_containers``, ``Stage.published_dirs``): a
    kind that leaves one out is refused when it registers, so none is read as declaring none."""
    cls = _stage(f"undeclared_{missing}", omit=(missing,))
    with pytest.raises(TypeError, match=missing):
        register(cls)
    assert cls.kind not in STAGES


@pytest.mark.parametrize("missing", ["check", "max_roles", "run"])
def test_a_stage_kind_without_a_required_method_is_refused(missing):
    """``max_roles`` sizes finalize's scan and an allocation's step headroom, so every kind gives
    it, as it gives ``check`` and ``run``."""
    cls = _stage(f"no_{missing}", omit=(missing,))
    with pytest.raises(TypeError, match=missing):
        register(cls)
    assert cls.kind not in STAGES


@pytest.mark.parametrize("name", ["check", "max_roles", "scored_server_cpus", "sandbox_stores", "declared_grants"])
def test_a_trait_given_as_a_plain_function_is_refused(name):
    """The planner calls these on the class, before any stage exists."""
    cls = _stage(f"plain_{name}", **{name: lambda *a, **k: None})
    with pytest.raises(TypeError, match=name):
        register(cls)
    assert cls.kind not in STAGES


def test_a_class_that_does_not_inherit_stage_is_refused():
    """Its traits would have no declared defaults to fall back on."""
    cls = _stage("not_a_stage", base=object)
    with pytest.raises(TypeError, match="Stage"):
        register(cls)
    assert "not_a_stage" not in STAGES


def test_a_well_declared_stage_class_registers_with_the_declared_defaults():
    """A class that sets only the required traits reads every optional one at its default."""
    cls = register(_stage("declared_defaults"))
    try:
        assert stage_class("declared_defaults") is cls
        assert (cls.needs_secret_registry, cls.grants_reach_contestant, cls.runs_contestant_code,
                cls.contestant_host_roles, cls.decides_contestant_faults) == (False,) * 5
        assert (cls.raw_outputs, cls.contestant_outputs, cls.contestant_log_roles) == ((), (), ())
        assert cls.declared_grants is None and cls.scored_server_cpus(_Params()) is None
        assert cls.sandbox_stores(_Params()) == ()
        assert cls.default_retries == 2
    finally:
        del STAGES["declared_defaults"]


@pytest.mark.parametrize("unread", [("snapshots/",), "outputs/source.bundle"], ids=["not-compressed", "not-a-tuple"])
def test_a_stage_kind_whose_unread_files_are_not_its_compressed_files_is_refused(unread):
    """Finalize does not read a kind's unread files (``Stage.unread_files``), so each must be a
    compressed file the kind declares, which a package scan reads inside or finalize quarantines
    whole: an unread file that is not one would go out unread."""
    cls = _stage("unread_elsewhere", compressed_files=("outputs/source.bundle",), unread_files=unread)
    with pytest.raises(TypeError, match="unread_files"):
        register(cls)
    assert "unread_elsewhere" not in STAGES


def test_a_stage_kinds_unread_files_among_its_compressed_files_register():
    """Unread files are optional, () by default, and any of the kind's compressed files may be one."""
    assert Stage.unread_files == ()
    register(_stage("unread_compressed", compressed_files=("outputs/source.bundle", "snapshots/"),
                    unread_files=("snapshots/",)))
    try:
        assert stage_class("unread_compressed").unread_files == ("snapshots/",)
    finally:
        del STAGES["unread_compressed"]


def test_every_registered_kind_declares_its_traits_as_stage_does():
    """Each production kind passed ``register``: it inherits ``Stage`` and gives ``max_roles``."""
    for kind, cls in STAGES.items():
        assert issubclass(cls, Stage), kind
        assert not cls.__abstractmethods__, kind


def test_most_roles_of_a_kind_no_module_registers_is_refused():
    """No default stands in for a kind's roles: finalize and the planner size by the kind's own."""
    with pytest.raises(ValueError, match="no_such_kind"):
        most_roles("no_such_kind", {})


def test_a_stages_retry_default_is_its_kinds():
    """The agent build retries once by default, whatever the stage is named; any other kind
    twice; a ``[run.retry.<stage>]`` table's count wins over either."""
    run = RunSettings()
    assert stage_class("agent_build").default_retries == 1
    assert retry_policy(run, "agent_build", "agent_build") == RetryPolicy(max=1)
    assert retry_policy(run, "build_2", "agent_build") == RetryPolicy(max=1)
    assert retry_policy(run, "verify", "verify") == RetryPolicy(max=2)
    assert retry_policy(run, "agent_build", "verify") == RetryPolicy(max=2)
    tabled = RunSettings(retry={"build_2": RetryPolicy(max=4, backoff_sec=5)})
    assert retry_policy(tabled, "build_2", "agent_build") == RetryPolicy(max=4, backoff_sec=5)
    off = RunSettings(retry={"build_2": RetryPolicy(enabled=False)})
    assert retry_policy(off, "build_2", "agent_build") == RetryPolicy(max=1, enabled=False)


def test_an_infra_stage_fault_carries_its_retry_and_evidence():
    """``StageFault.infra`` is the one way a stage builds an infra fault; it has no default for
    whether a retry could help."""
    fault = StageFault.infra("the node's disk is full", retryable=False, role="driver").fault
    assert (fault.fault_class, fault.reason, fault.retryable, fault.evidence) == (
        "infra", "the node's disk is full", False, {"role": "driver"})
    assert StageFault.infra("x", retryable=True).fault.retryable is True
    with pytest.raises(TypeError, match="retryable"):
        StageFault.infra("x")  # type: ignore[call-arg]


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _named(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def test_every_infra_stage_fault_is_built_by_stage_fault_infra_and_says_whether_it_is_retried():
    """No module builds an infra StageFault by hand or through a helper of its own, whose default
    for ``retryable`` would differ from another's: every one is ``StageFault.infra``, and every
    call names ``retryable``, which a call that is never run in the tests could otherwise leave
    out unnoticed."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for call in _calls(ast.parse(path.read_text())):
            where = f"{path.relative_to(SRC)}:{call.lineno}"
            func = call.func
            if _named(func, "StageFault") and call.args and isinstance(call.args[0], ast.Call):
                inner = call.args[0].func
                if _named(inner, "infra_fault") or _named(inner, "FaultRecord"):
                    found.append(f"{where} builds an infra StageFault by hand")
            if (isinstance(func, ast.Attribute) and func.attr == "infra" and _named(func.value, "StageFault")
                    and not any(k.arg == "retryable" for k in call.keywords)):
                found.append(f"{where} does not say whether it is retried")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name in ("_infra", "infra") and "stages" in path.parts:
                found.append(f"{path.relative_to(SRC)}:{node.lineno} defines its own infra fault helper")
    assert found == []
