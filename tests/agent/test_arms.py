"""Arm records: what an arm declares, the routes it names, what each harness can express, and how
the plan's record is resolved and compared."""

from __future__ import annotations

import ast
import textwrap
import traceback
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from hls.arms import (ADAPTERS, Arm, ArmError, ArmHarness, ArmRecord, ClaudeCodeAdapter, CodexAdapter, HarnessAdapter,
                      Route, adapter_for, adapter_named, arm_differences, resolve_arm, resolve_effort, route_of,
                      route_values)
from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable, ProviderLevel
from hls.settings import SettingsError, load_settings, resolved, settings_sha256

SITE = """
[site]
partition = "gpu"
store_root = "/shared/u/hls-store"
node_scratch = "/tmp"
secrets_path = "/shared/u/secrets.toml"
"""
ROUTES = """
[routes.kimi]
credential = "kimi_token"
base_url = "kimi_url"
auth = "bearer"

[routes.openai]
credential = "openai_key"
"""
K3 = """
[arms.k3]
harness = "claude-code"
model = "kimi-k3"
context_window = 1000000
output_limit = 64000
route = "kimi"
"""
GPT = """
[arms.arm-b]
harness = "codex"
model = "gpt-5.5"
context_window = 258400
output_limit = 128000
route = "openai"
"""
#: A relay's base URL, as the secrets file holds it; it never appears in hls.toml or a record.
RELAY_URL = "https://relay.example/anthropic"


def _settings(tmp_path: Path, text: str = K3 + GPT, routes: str = ROUTES):
    path = tmp_path / "hls.toml"
    path.write_text(textwrap.dedent(SITE + routes + text))
    return load_settings(path)


def _arm(**kw) -> Arm:
    base = {"harness": "claude-code", "model": "claude-opus-5", "context_window": 1_000_000, "output_limit": 64_000,
            "route": "anthropic"}
    return Arm.model_validate({**base, **kw})


def test_an_arm_table_loads_into_the_settings_and_their_record(tmp_path):
    s = _settings(tmp_path)
    assert sorted(s.arms) == ["arm-b", "k3"]
    k3 = s.arms["k3"]
    assert (k3.harness, k3.model, k3.route, k3.context_window, k3.output_limit, k3.effort) == (
        "claude-code", "kimi-k3", "kimi", 1_000_000, 64_000, None)
    assert s.routes["kimi"] == Route(credential="kimi_token", base_url="kimi_url", auth="bearer")
    assert s.routes["openai"] == Route(credential="openai_key")
    assert resolved(s)["arms"]["k3"]["route"] == "kimi"
    assert resolved(s)["routes"]["kimi"] == {"credential": "kimi_token", "base_url": "kimi_url", "auth": "bearer"}
    assert settings_sha256(s) != settings_sha256(_settings(tmp_path, K3.replace("64000", "32000") + GPT))


@pytest.mark.parametrize("text, routes, where", [
    (K3 + "windw = 5\n", ROUTES, r"unknown setting arms\.k3\.windw"),
    (K3, ROUTES.replace('auth = "bearer"', 'auth = "bearer"\ntoken = "x"'), r"unknown setting routes\.kimi\.token"),
    (K3.replace("1000000", "400000"), ROUTES, r"arms\.k3: .*claude-code can run with a context window of 200,000 or "
                                              r"1,000,000 tokens, not 400,000"),
    (K3.replace("[arms.k3]", "[arms.\"k 3\"]"), ROUTES, r"arms\.\"k 3\""),
    (K3, ROUTES.replace("[routes.kimi]", "[routes.\"ki mi\"]"), r"routes\.\"ki mi\""),
], ids=["unknown-key", "unknown-route-key", "window", "name", "route-name"])
def test_a_wrong_arm_table_fails_the_settings_naming_where(tmp_path, text, routes, where):
    with pytest.raises(SettingsError, match=where):
        _settings(tmp_path, text, routes)


def test_an_arm_names_a_declared_route(tmp_path):
    with pytest.raises(SettingsError, match=r"arms\.k3: route names no \[routes\.<name>\] table; \[routes\] "
                                            r"declares \['kimi', 'openai'\]") as e:
        _settings(tmp_path, K3.replace('route = "kimi"', 'route = "kimmi"') + GPT)
    assert "kimmi" not in str(e.value)
    with pytest.raises(SettingsError, match=r"arms\.k3: route names no \[routes\.<name>\] table; \[routes\] "
                                            r"declares none"):
        _settings(tmp_path, K3, routes="")
    with pytest.raises(SettingsError, match=r"arms\.k3\.route: .*a route's name is one store key segment") as e:
        _settings(tmp_path, K3.replace('route = "kimi"', 'route = "ki/mi"'))
    assert "ki/mi" not in str(e.value)


@pytest.mark.parametrize("value, where", [
    (RELAY_URL, r"arms\.k3\.route: .*a route's name is one store key segment.*a base URL belongs in the secrets file"),
    ("sk-ant-api03-PASTEDKEYVALUE", r"arms\.k3: route names no \[routes\.<name>\] table"),
], ids=["url", "key-shaped"])
def test_an_arms_route_is_refused_without_its_value(tmp_path, value, where):
    # [arms.<name>].route is the field a URL or a credential is likeliest to be pasted into, so
    # the refusal names the arm and the field instead of the value.
    with pytest.raises(SettingsError, match=where) as e:
        _settings(tmp_path, K3.replace('route = "kimi"', f'route = "{value}"'))
    shown = "".join(traceback.format_exception(e.value))
    assert value not in shown and "relay.example" not in shown


def test_a_route_refusal_outside_the_settings_does_not_quote_the_value(tmp_path):
    with pytest.raises(ValidationError, match="a route's name is one store key segment") as e:
        _arm(route=RELAY_URL)
    assert RELAY_URL not in "; ".join(err["msg"] for err in e.value.errors())
    key = "sk-ant-api03-PASTEDKEYVALUE"
    with pytest.raises(ArmError, match=r"route names no \[routes\.<name>\] table") as e:
        route_of(_settings(tmp_path), _arm(route=key))
    assert key not in "".join(traceback.format_exception(e.value))


PLANTED = "https://relay.example/PLANTEDVALUE"


def _k3_fields(**kw) -> dict:
    return {"harness": "claude-code", "model": "kimi-k3", "context_window": 1_000_000, "output_limit": 64_000,
            "route": "kimi", **kw}


@pytest.mark.parametrize("build", [
    lambda: Route(credential=PLANTED),
    lambda: Route(credential="kimi_token", base_url=PLANTED),
    lambda: Route(credential="kimi_token", auth=PLANTED),
    lambda: Route(credential="kimi_token", extra=PLANTED),
    lambda: Arm(**_k3_fields(route=PLANTED)),
    lambda: Arm(**_k3_fields(extra=PLANTED)),
    lambda: ArmRecord(name="k3", route_table={"credential": PLANTED}, **_k3_fields()),
    lambda: ArmRecord(name="k3", route_table={"credential": "kimi_token", "base_url": PLANTED}, **_k3_fields()),
    lambda: ArmRecord(name="k3", route_table={"credential": "kimi_token"}, **_k3_fields(route=PLANTED)),
    lambda: ArmRecord(name="k3", route_table={"credential": "kimi_token"}, **_k3_fields(extra=PLANTED)),
], ids=["route-credential", "route-base-url", "route-auth", "route-extra", "arm-route", "arm-extra",
        "record-credential", "record-base-url", "record-route", "record-extra"])
def test_a_directly_built_model_does_not_show_its_input_in_a_validation_error(build):
    # Settings hides the input of its own errors; a model built directly, as the planner does for a
    # record, must hide it too, since the input may be a pasted credential or URL.
    with pytest.raises(ValidationError) as e:
        build()
    assert PLANTED not in str(e.value)
    assert PLANTED not in "".join(traceback.format_exception(e.value))


@pytest.mark.parametrize("value", [RELAY_URL, "relay.example", "KIMI_URL", "kimi-url"])
def test_a_base_url_names_a_secret_and_a_url_is_refused(tmp_path, value):
    with pytest.raises(ValidationError, match="base_url must name a secret.*the base URL is a secret") as e:
        Route(credential="kimi_token", base_url=value)
    assert value not in "; ".join(err["msg"] for err in e.value.errors())
    with pytest.raises(SettingsError, match=r"routes\.kimi\.base_url: .*the base URL is a secret") as e:
        _settings(tmp_path, K3, ROUTES.replace('base_url = "kimi_url"', f'base_url = "{value}"'))
    assert value not in str(e.value)


@pytest.mark.parametrize("name", ["KIMI", "kimi-token", "1kimi", "", "sk-ant-api03-abcdefgh"])
def test_a_credential_names_a_secret(name):
    with pytest.raises(ValidationError, match="credential must name a secret") as e:
        Route(credential=name)
    if name:
        assert name not in "; ".join(err["msg"] for err in e.value.errors())


@pytest.mark.parametrize("model", ["claude-opus-5[1m]", "kimi-k3[1m]", "claude opus", "claude-opus-5\n", ""])
def test_a_model_id_carries_no_window_and_no_whitespace(model):
    with pytest.raises(ValidationError, match="bare model id"):
        _arm(model=model)


@pytest.mark.parametrize("field, value", [("context_window", 0), ("output_limit", 0), ("output_limit", -1)])
def test_window_and_output_limit_are_positive(field, value):
    with pytest.raises(ValidationError, match="greater than 0"):
        _arm(**{field: value})


def test_only_claude_code_and_codex_arms_exist():
    with pytest.raises(ValidationError, match="harness"):
        _arm(harness="terminus-2")
    assert sorted(ADAPTERS) == ["claude-code", "codex"]
    assert sorted(get_args(ArmHarness)) == sorted(ADAPTERS)
    assert all(adapter.name == name for name, adapter in ADAPTERS.items())


def test_every_adapter_has_every_member_of_the_protocol():
    """No type checker runs on the tree, so a member an adapter lacks would surface only when
    an agent of its harness reached it."""
    members = {*HarnessAdapter.__annotations__,
               *(name for name, value in vars(HarnessAdapter).items() if callable(value) and not name.startswith("_"))}
    assert {"check", "build_agent", "preflight", "transcript_turns", "log_reader", "launch_env"} <= members
    for adapter in ADAPTERS.values():
        assert [m for m in sorted(members) if not hasattr(adapter, m)] == [], type(adapter).__name__


def test_a_harness_no_adapter_runs_is_refused_naming_it():
    with pytest.raises(ArmError, match=r"no adapter runs harness 'terminus-2'; hls runs claude-code, codex"):
        adapter_named("terminus-2")
    # A record copied past its validation reaches no adapter either, and never another harness's.
    stray = _arm().model_copy(update={"harness": "terminus-2"})
    with pytest.raises(ArmError, match="'terminus-2'"):
        adapter_for(stray)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _harness_branches(path: Path) -> list[str]:
    """Where the module at *path* decides on a harness by its name or its harbor class: a
    comparison with an ``harness`` attribute or name, a harness's name as a string, or an
    ``isinstance`` against a harbor agent class."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        where = f"{path.relative_to(_REPO_ROOT)}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(isinstance(s, ast.Attribute) and s.attr == "harness" or isinstance(s, ast.Name) and s.id == "harness"
                   for s in sides):
                found.append(f"{where} compares a harness")
        elif isinstance(node, ast.Constant) and node.value in ADAPTERS:
            found.append(f"{where} names {node.value!r}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance"
              and any(isinstance(n, ast.Name) and n.id in ("ClaudeCode", "Codex", "HlsCodex")
                      for n in ast.walk(node.args[1]))):
            found.append(f"{where} tells a harness by its agent's class")
    return found


def test_only_the_adapters_know_a_harness_by_its_name():
    """Each harness's behaviour is its adapter's (``hls.arms``); no other module branches on a
    harness's name or its harbor class, so none can treat an unknown harness as a known one."""
    src = _REPO_ROOT / "src" / "hls"
    found = [hit for path in sorted(src.rglob("*.py")) if path != src / "arms.py" for hit in _harness_branches(path)]
    assert found == []


def test_claude_code_takes_its_window_as_the_model_ids_suffix():
    one_m, default = _arm(context_window=1_000_000), _arm(context_window=200_000)
    cc = adapter_for(one_m)
    assert cc.model_string(one_m) == "claude-opus-5[1m]"
    assert cc.model_string(default) == "claude-opus-5"
    assert cc.cli_config(one_m) == {}
    with pytest.raises(ValidationError, match="200,000 or 1,000,000 tokens, not 2,000,000"):
        _arm(context_window=2_000_000)


@pytest.mark.parametrize("model", ["k3", "claude-opus-5"])
@pytest.mark.parametrize("window", [200_000, 1_000_000])
def test_claude_code_compacts_against_the_arms_window(model, window):
    # Claude Code 2.1.251 compacts a [1m] model id by its own threshold only when a window is set
    # (CLAUDE_CODE_AUTO_COMPACT_WINDOW or a setting); otherwise it waits for a "prompt is too long"
    # refusal, which Moonshot's k3 endpoint never sends, so a session there grows past its limit.
    arm = _arm(model=model, context_window=window)
    env = adapter_for(arm).plain_env(arm)
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == str(window)
    # The CLI accepts a window from 100,000 to 1,000,000 tokens, which holds every window it can run.
    assert all(100_000 <= w <= 1_000_000 for w in ClaudeCodeAdapter.windows)


def test_codex_takes_any_window_and_the_output_limit_as_configuration():
    arm = _arm(harness="codex", model="gpt-5.5", context_window=258_400, output_limit=128_000)
    codex = adapter_for(arm)
    assert codex.model_string(arm) == "gpt-5.5"
    assert codex.cli_config(arm) == {"model_context_window": 258_400, "model_max_output_tokens": 128_000}


@pytest.mark.parametrize("auth, var", [("api_key", "ANTHROPIC_API_KEY"), ("bearer", "ANTHROPIC_AUTH_TOKEN")])
def test_claude_code_is_granted_exactly_its_routes_credential(auth, var):
    route = Route(credential="kimi_token", auth=auth)
    cc = ADAPTERS["claude-code"]
    assert cc.grants(route) == {var: "kimi_token"}
    assert var in cc.credential_vars


def test_a_route_grants_its_credential_and_its_base_url():
    cc, codex = ADAPTERS["claude-code"], ADAPTERS["codex"]
    assert (cc.base_url_var, codex.base_url_var) == ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")
    assert cc.credential_vars == ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert codex.credential_vars == ("OPENAI_API_KEY",)
    relay = Route(credential="kimi_token", base_url="kimi_url", auth="bearer")
    assert cc.grants(relay) == {"ANTHROPIC_AUTH_TOKEN": "kimi_token", "ANTHROPIC_BASE_URL": "kimi_url"}
    gateway = Route(credential="k_key", base_url="gw_url")
    assert codex.grants(gateway) == {"OPENAI_API_KEY": "k_key", "OPENAI_BASE_URL": "gw_url"}
    assert codex.grants(Route(credential="openai_key")) == {"OPENAI_API_KEY": "openai_key"}
    assert cc.grants(Route(credential="anthropic_key")) == {"ANTHROPIC_API_KEY": "anthropic_key"}


def test_codex_refuses_a_bearer_route_when_the_settings_load(tmp_path):
    bearer = ROUTES.replace("[routes.openai]\n", "[routes.openai]\nauth = \"bearer\"\n")
    with pytest.raises(SettingsError, match=r"arms\.arm-b: .*codex presents its credential as OPENAI_API_KEY; "
                                            r"route\.auth must be api_key"):
        _settings(tmp_path, K3 + GPT, bearer)
    # Claude Code takes the same route.
    assert _settings(tmp_path, K3.replace('route = "kimi"', 'route = "openai"'), bearer).arms["k3"].route == "openai"
    with pytest.raises(ValueError, match="route.auth must be api_key"):
        CodexAdapter().grants(Route(credential="openai_key", auth="bearer"))


def test_the_routes_plain_variables_name_no_secret():
    cc = _arm()
    assert adapter_for(cc).plain_env(cc) == {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
                                             "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000"}
    codex = _arm(harness="codex", model="gpt-5.5", context_window=258_400)
    assert adapter_for(codex).plain_env(codex) == {}
    for arm in (cc, codex):
        adapter = adapter_for(arm)
        assert not {*adapter.credential_vars, adapter.base_url_var} & set(adapter.plain_env(arm))


@pytest.mark.parametrize("adapter, base_url, host", [
    (ClaudeCodeAdapter(), None, "api.anthropic.com"),
    (ClaudeCodeAdapter(), "https://relay.example:8443/anthropic", "relay.example"),
    (CodexAdapter(), None, "api.openai.com"),
    (CodexAdapter(), "https://gw.example/v1", "gw.example"),
], ids=["anthropic", "relay", "openai", "gateway"])
def test_the_seal_opens_exactly_the_routes_host(adapter, base_url, host):
    assert adapter.route_host(base_url) == host


@pytest.mark.parametrize("value", ["http://relay.example/anthropic", "https://", "relay.example/anthropic",
                                   "ftp://relay.example", "https://[relay.example", "https://[relay.example]/x"])
def test_a_granted_base_url_is_https_with_a_host_and_never_shown(value):
    with pytest.raises(ValueError, match="base URL must be an https URL with a host") as e:
        ClaudeCodeAdapter().route_host(value)
    assert value not in str(e.value) and e.value.__cause__ is None
    # The parser's own error for "https://[relay.example]/x" quotes the host.
    assert "relay.example" not in "".join(traceback.format_exception(e.value))


def test_the_route_values_an_environment_holds_are_named_longest_first():
    env = {"OPENAI_API_KEY": "sk-test-0123456789", "OPENAI_BASE_URL": "https://gw.example:8443/v1", "PATH": "/x",
           "ANTHROPIC_API_KEY": "not codex's"}
    assert route_values(CodexAdapter(), env) == [("OPENAI_BASE_URL", "https://gw.example:8443/v1"),
                                                 ("OPENAI_API_KEY", "sk-test-0123456789"),
                                                 ("OPENAI_BASE_URL.host", "gw.example")]
    assert route_values(ClaudeCodeAdapter(), {"ANTHROPIC_AUTH_TOKEN": "tok-0123456789", "ANTHROPIC_BASE_URL": None,
                                              "ANTHROPIC_API_KEY": ""}) == [("ANTHROPIC_AUTH_TOKEN", "tok-0123456789")]
    # A base URL with no host to read is still named whole.
    assert route_values(CodexAdapter(), {"OPENAI_BASE_URL": "https://[gw.example"}) == [
        ("OPENAI_BASE_URL", "https://[gw.example")]
    # Its host is a route value only when it identifies a machine (``secret_forms.identifies``), and
    # without the trailing dot a resolver ignores.
    assert route_values(CodexAdapter(), {"OPENAI_BASE_URL": "https://relay:8443/v1"}) == [
        ("OPENAI_BASE_URL", "https://relay:8443/v1")]
    assert route_values(CodexAdapter(), {"OPENAI_BASE_URL": "https://127.1:8443/v1"}) == [
        ("OPENAI_BASE_URL", "https://127.1:8443/v1")]
    assert route_values(CodexAdapter(), {"OPENAI_BASE_URL": "https://gw.example.:8443/v1"})[-1] == (
        "OPENAI_BASE_URL.host", "gw.example")


@pytest.mark.parametrize("harness, model", [("claude-code", "claude-opus-5"), ("codex", "gpt-5.5")])
def test_an_effort_the_harness_does_not_accept_is_refused(harness, model):
    assert _arm(harness=harness, model=model, effort="xhigh").effort == "xhigh"
    with pytest.raises(ValidationError, match="accepts effort low, medium, high, xhigh, max, not 'ultra'"):
        _arm(harness=harness, model=model, effort="ultra")


def test_codex_runs_no_record_without_an_effort(tmp_path):
    """Codex always runs a reasoning effort, harbor's "high" when it is given none, so a record
    without one is refused. A declared arm without one takes its task's when it is resolved."""
    s = _settings(tmp_path)
    declared = s.arms["arm-b"]
    assert declared.effort is None
    CodexAdapter().check(declared)
    unset = resolve_arm(s, "arm-b", task_effort=HighestAvailable()).model_copy(update={"effort": None})
    with pytest.raises(ValueError, match=r"^codex always runs a reasoning effort \(harbor's default is high\), so "
                                         r"the arm must declare one or take its task's$"):
        CodexAdapter().check(unset)
    with pytest.raises(ValidationError, match="codex always runs a reasoning effort"):
        ArmRecord.model_validate(unset.model_dump())
    with pytest.raises(ArmError, match="arm arm-b: codex always runs a reasoning effort"):
        resolve_arm(s, "arm-b", task_effort=None)
    # Claude Code launches with no --effort and takes its CLI's level.
    assert resolve_arm(s, "k3", task_effort=None).effort is None


@pytest.mark.parametrize("name, task_effort, effort", [
    ("k3", HighestAvailable(), "max"),
    ("arm-b", HighestAvailable(), "xhigh"),
    ("k3", ProviderLevel(level="high"), "high"),
    ("k3", None, None),
], ids=["claude-highest", "codex-highest-per-model", "provider-level", "unset"])
def test_an_unset_effort_takes_the_tasks_producer_setting(tmp_path, name, task_effort, effort):
    record = resolve_arm(_settings(tmp_path), name, task_effort=task_effort)
    assert isinstance(record, ArmRecord) and record.name == name and record.effort == effort


def test_an_arms_own_effort_wins_over_the_tasks(tmp_path):
    s = _settings(tmp_path, K3 + 'effort = "medium"\n')
    assert resolve_arm(s, "k3", task_effort=HighestAvailable()).effort == "medium"


def test_an_effort_the_task_resolves_to_must_be_one_the_harness_accepts(tmp_path):
    with pytest.raises(ArmError, match="arm k3: .*accepts effort .* not 'ultra'"):
        resolve_arm(_settings(tmp_path), "k3", task_effort=ProviderLevel(level="ultra"))


def test_a_model_without_a_known_top_level_refuses_highest_available(tmp_path):
    s = _settings(tmp_path, GPT.replace("gpt-5.5", "gpt-9"))
    with pytest.raises(ArmError, match="arm arm-b: .*codex lists no top effort for model 'gpt-9'"):
        resolve_arm(s, "arm-b", task_effort=HighestAvailable())


def test_each_adapter_holds_its_harness_top_effort():
    assert dict(ADAPTERS["claude-code"].top_effort) == {None: "max"}
    assert dict(ADAPTERS["codex"].top_effort) == {"gpt-5.6-sol": "max", "gpt-5.5": "xhigh"}


@pytest.mark.parametrize("harness, model, effort, level", [
    pytest.param("claude-code", "claude-opus-5", HighestAvailable(), "max", id="claude-any-model"),
    pytest.param("codex", "gpt-5.6-sol", HighestAvailable(), "max", id="codex-sol"),
    pytest.param("codex", "gpt-5.5", HighestAvailable(), "xhigh", id="codex-55"),
    pytest.param("codex", "gpt-5.5", ProviderLevel(level="high"), "high", id="level-as-written"),
    pytest.param("codex", "gpt-5.5", None, None, id="unset"),
])
def test_an_effort_resolves_through_the_adapters_table(harness, model, effort, level):
    assert resolve_effort(effort, adapter=ADAPTERS[harness], model=model) == level


def test_a_models_own_entry_wins_over_the_entry_for_every_model(monkeypatch):
    """A hit is a key that is present: the model's entry is read first, and an entry mapped to
    None, a provider that publishes no levels, is an answer and not a miss."""
    codex = ADAPTERS["codex"]
    monkeypatch.setattr(codex, "top_effort", {"gpt-5.5": "xhigh", "gpt-x": None, None: "low"})
    assert resolve_effort(HighestAvailable(), adapter=codex, model="gpt-5.5") == "xhigh"
    assert resolve_effort(HighestAvailable(), adapter=codex, model="gpt-x") is None
    assert resolve_effort(HighestAvailable(), adapter=codex, model="gpt-9") == "low"


def test_a_model_the_table_does_not_list_is_refused_naming_it():
    with pytest.raises(ArmError, match=r"codex lists no top effort for model 'gpt-9'"):
        resolve_effort(HighestAvailable(), adapter=ADAPTERS["codex"], model="gpt-9")
    with pytest.raises(ArmError, match=r"add it to CodexAdapter\.top_effort, or set the arm's effort"):
        resolve_effort(HighestAvailable(), adapter=ADAPTERS["codex"], model="gpt-9")


@pytest.mark.parametrize("given", [{}, {"reasoning_effort": None}], ids=["unset", "none"])
def test_a_producer_authors_its_effort(given):
    """An arm that leaves its effort unset takes its task's producer setting (``resolve_arm``),
    and harness defaults are neither uniform nor visible, so every producer authors one."""
    with pytest.raises(ValidationError, match=r"reasoning_effort\n"):
        AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600), **given)


def test_the_effort_is_typed_not_a_bare_string():
    """A bare string could collide with a provider's own level name; the spec takes only the
    two tagged forms."""
    with pytest.raises(ValidationError):
        AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600), reasoning_effort="max")
    spec = AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600),
                             reasoning_effort={"kind": "provider", "level": "a-new-level"})
    assert spec.reasoning_effort == ProviderLevel(level="a-new-level")
    spec = AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600),
                             reasoning_effort={"kind": "highest_available"})
    assert spec.reasoning_effort == HighestAvailable()


def test_an_unknown_arm_is_refused_naming_the_declared_ones(tmp_path):
    with pytest.raises(ArmError, match=r"no arm named 'arm-a'.*\['arm-b', 'k3'\]"):
        resolve_arm(_settings(tmp_path), "arm-a", task_effort=None)


def test_a_resolved_record_round_trips_through_the_plan(tmp_path):
    record = resolve_arm(_settings(tmp_path), "k3", task_effort=HighestAvailable())
    assert ArmRecord.model_validate_json(record.model_dump_json()) == record


def test_a_resolved_record_names_its_route_and_holds_no_url(tmp_path):
    s = _settings(tmp_path)
    text = resolve_arm(s, "k3", task_effort=HighestAvailable()).model_dump_json()
    assert '"route":"kimi"' in text
    for leak in (RELAY_URL, "relay.example", "https://"):
        assert leak not in text
    assert "https://" not in str(resolved(s))


def test_a_resolved_record_keeps_its_routes_secret_names_and_auth(tmp_path):
    s = _settings(tmp_path)
    k3 = resolve_arm(s, "k3", task_effort=None)
    assert (k3.route, k3.route_table) == ("kimi", Route(credential="kimi_token", base_url="kimi_url", auth="bearer"))
    gpt = resolve_arm(s, "arm-b", task_effort=HighestAvailable())
    assert (gpt.route, gpt.route_table) == ("openai", Route(credential="openai_key"))
    assert k3.model_dump(mode="json")["route_table"] == {"credential": "kimi_token", "base_url": "kimi_url",
                                                          "auth": "bearer"}


def test_a_record_holds_a_route_its_harness_can_use(tmp_path):
    gpt = resolve_arm(_settings(tmp_path), "arm-b", task_effort=HighestAvailable())
    with pytest.raises(ValidationError, match="codex presents its credential as OPENAI_API_KEY"):
        ArmRecord.model_validate({**gpt.model_dump(), "route_table": {"credential": "openai_key", "auth": "bearer"}})


def test_an_arms_route_resolves_from_the_settings(tmp_path):
    s = _settings(tmp_path)
    record = resolve_arm(s, "k3", task_effort=None)
    assert route_of(s, record) == Route(credential="kimi_token", base_url="kimi_url", auth="bearer")
    assert route_of(s, s.arms["arm-b"]) == Route(credential="openai_key")
    with pytest.raises(ArmError, match=r"arm k3: route names no \[routes\.<name>\] table; \[routes\] declares "
                                       r"\['openai'\]"):
        route_of(_settings(tmp_path, GPT, '[routes.openai]\ncredential = "openai_key"\n'), record)
    bearer = ROUTES.replace("[routes.openai]\n", "[routes.openai]\nauth = \"bearer\"\n")
    codex = resolve_arm(s, "arm-b", task_effort=HighestAvailable())
    with pytest.raises(ArmError, match="arm arm-b: codex presents its credential as OPENAI_API_KEY"):
        route_of(_settings(tmp_path, K3, bearer), codex)


def test_the_differences_name_each_changed_field(tmp_path):
    s = _settings(tmp_path)
    old = resolve_arm(s, "k3", task_effort=HighestAvailable())
    assert arm_differences(old, old) == []
    new = old.model_copy(update={"context_window": 200_000, "route": "kimi2"})
    assert arm_differences(old, new) == ["route: 'kimi' -> 'kimi2'", "context_window: 1000000 -> 200000"]
    renamed = old.model_copy(update={"model": "k3"})
    assert arm_differences(old, renamed) == ["model: 'kimi-k3' -> 'k3'"]


@pytest.mark.parametrize("change, difference", [
    (('base_url = "kimi_url"', 'base_url = "kimi_url_b"'), "route_table.base_url: 'kimi_url' -> 'kimi_url_b'"),
    (('credential = "kimi_token"', 'credential = "kimi_token_b"'),
     "route_table.credential: 'kimi_token' -> 'kimi_token_b'"),
    (('auth = "bearer"', 'auth = "api_key"'), "route_table.auth: 'bearer' -> 'api_key'"),
    (('base_url = "kimi_url"\n', ""), "route_table.base_url: 'kimi_url' -> None"),
], ids=["base-url-secret", "credential-secret", "auth", "no-base-url"])
def test_a_route_repointed_under_the_same_name_is_an_arm_change(tmp_path, change, difference):
    # The operator edits [routes.kimi] and leaves [arms.k3] alone: the rerun's arm differs.
    old = resolve_arm(_settings(tmp_path), "k3", task_effort=HighestAvailable())
    new = resolve_arm(_settings(tmp_path, routes=ROUTES.replace(*change)), "k3", task_effort=HighestAvailable())
    assert arm_differences(old, new) == [difference]


def test_a_record_resolves_only_against_the_route_it_names(tmp_path):
    # Grants come from the settings' route, so a record made from another declaration of the
    # route would name secrets other than those the stage is granted.
    record = resolve_arm(_settings(tmp_path), "k3", task_effort=None)
    repointed = _settings(tmp_path, routes=ROUTES.replace('base_url = "kimi_url"', 'base_url = "kimi_url_b"'))
    with pytest.raises(ArmError, match=r"arm k3: its record holds route kimi as another \[routes\.kimi\] declared it"):
        route_of(repointed, record)


@pytest.mark.parametrize("url, host", [("https://relay-0123456789abcdef.example.net/v1",
                                        "relay-0123456789abcdef.example.net"),
                                       ("http://relay:8000/v1", None), ("http://127.0.0.1:8000/v1", None),
                                       ("https://Relay.Example.com./v1", "relay.example.com")],
                         ids=["dns-name", "single-label", "loopback", "trailing-dot"])
def test_a_routes_host_is_held_back_only_when_it_is_one_of_the_secrets_forms(url, host):
    adapter = ADAPTERS["claude-code"]
    values = dict(route_values(adapter, {adapter.base_url_var: url}))
    assert values.get(f"{adapter.base_url_var}.host") == host and values[adapter.base_url_var] == url


def test_each_adapter_declares_the_text_its_clis_launch_holds_and_one_without_is_refused():
    """The build tells a failure before the agent's CLI first ran from one after by the exec
    server's start lines, which say whether a command holds the arm's launch marker: an adapter
    with none would leave every failure of its agent's part infra, so registering one fails."""
    from hls import arms

    assert {name: adapter.cli_launch for name, adapter in ADAPTERS.items()} == {
        "claude-code": "exec claude ", "codex": "codex exec "}

    class _Unmarked(ClaudeCodeAdapter):
        name = "unmarked"
        cli_launch = " "

    with pytest.raises(ArmError, match="harness 'unmarked' declares no text its CLI's launch holds"):
        arms.registry(ClaudeCodeAdapter(), _Unmarked())
    assert set(arms.registry(ClaudeCodeAdapter(), CodexAdapter())) == {"claude-code", "codex"}
