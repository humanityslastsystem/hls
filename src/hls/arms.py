"""Arms: which agent builds a submission, and through which route.

An arm is declared once, in hls.toml, and names a route declared beside it::

    [routes.relay]
    credential = "relay_token"
    base_url = "relay_url"
    auth = "bearer"

    [arms.example]
    harness = "claude-code"
    model = "example-model"
    context_window = 1000000
    output_limit = 64000
    effort = "max"
    route = "relay"

The arm says everything that decides what the agent is: its harness, the exact model id, the
context window in tokens, the output-token limit, the reasoning effort and the name of its route.
A route says how the agent reaches its model, by the names of secrets in the site's secrets file
only: the secret that holds its credential and how that credential is presented, and the secret
that holds its base URL (none for the harness's own provider). A base URL is a secret like the
credential, so no plan, record or event holds it or its host; they name the route.

The arm is harness-neutral. Only the harness's adapter turns it into CLI settings, so nobody
writes a window into a model id: Claude Code gets the id with ``[1m]`` for a 1M window and the
window as the one it compacts against, Codex a context-window config value. The adapter holds
everything else that differs between harnesses too: what the CLI reports back, where its
session's transcript lies and how it reads, the harbor agent that runs the arm and the one that
reads its logs. Every other module asks the arm's adapter (``adapter_for``), never the harness's
name, and a harness no adapter runs is refused.

The planner resolves an arm into an ``ArmRecord`` and puts it into the plan. The record holds
the route's name and its table, the names of its secrets and its ``auth``, so a rerun on a
repointed route is a different arm (``arm_differences``). The agent is granted exactly its
route's secrets, as the variables its harness reads, and nothing else
(``HarnessAdapter.grants``). The planner checks that a granted base URL is one the harness
takes (``check_granted_base_urls``), and the sandbox's seal opens exactly its host
(``HarnessAdapter.route_host``)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, ValidationError, ValidationInfo,
                      field_validator, model_validator)

from hls.agent.health import TranscriptLayout, Turn, claude_turns, codex_turns
from hls.artifact.artifact import HighestAvailable, ProviderLevel
from hls.store.base import check_segment
from hls.substrate.secret_forms import SECRET_NAME, forms_of

if TYPE_CHECKING:
    from hls.agent.harnesses import AgentRun
    from hls.agent.preflight import PreflightResult
    from hls.settings import Settings


def _route_name(name: str) -> str:
    # The value is never shown: an arm's route is the field a URL or a credential is likeliest
    # to be pasted into.
    try:
        check_segment(name)
    except ValueError:
        raise ValueError("a route's name is one store key segment (a letter, a digit or @, then letters, digits "
                         "and ._@=+-); a base URL belongs in the secrets file, named by the route's base_url") from None
    return name


#: An arm's name: one store key segment, as in ``[arms.<name>]``.
ArmName = Annotated[str, AfterValidator(check_segment)]
#: A route's name: one store key segment, as in ``[routes.<name>]``. Its refusal does not show
#: the value.
RouteName = Annotated[str, AfterValidator(_route_name)]
#: The harnesses an arm may run on.
ArmHarness = Literal["claude-code", "codex"]
# Characters a model id may not hold: those of a window suffix.
_MODEL_FORBIDDEN = ("[", "]")
# Why a route field is refused. The value is never shown: it may be a pasted credential or URL.
_NOT_A_NAME = {
    "credential": f"credential must name a secret of the secrets file ({SECRET_NAME.pattern})",
    "base_url": (f"base_url must name a secret of the secrets file ({SECRET_NAME.pattern}); the base URL is a "
                 "secret itself, so the URL goes in the secrets file and its name here"),
}


class ArmError(ValueError):
    """An arm cannot be resolved, or its harness cannot express it."""


class Route(BaseModel):
    """How an arm's agent reaches its model, by the names of secrets only."""

    # A refused value stays out of the error's text, as in Settings: a route or arm built
    # directly may hold a pasted credential or URL.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    #: The name of the secret in the site's secrets file that holds the credential.
    credential: str
    #: The name of the secret that holds the API's base URL; None for the harness's own provider.
    base_url: str | None = None
    #: How the credential is presented: an API key, or a bearer token.
    auth: Literal["api_key", "bearer"] = "api_key"

    @field_validator("credential", "base_url")
    @classmethod
    def _secret_name(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None and not SECRET_NAME.fullmatch(v):
            raise ValueError(_NOT_A_NAME[str(info.field_name)])
        return v


class Arm(BaseModel):
    """One ``[arms.<name>]`` table: an agent and the name of its route."""

    # A refused value stays out of the error's text, as in Settings: a route or arm built
    # directly may hold a pasted credential or URL.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    #: The harness the agent runs on.
    harness: ArmHarness
    #: The model id exactly as the provider names it, with no window suffix.
    model: str
    #: The name of the route to the model, a ``[routes.<name>]`` table of hls.toml.
    route: RouteName
    #: The context window, in tokens.
    context_window: int = Field(gt=0)
    #: The output-token limit of one model response.
    output_limit: int = Field(gt=0)
    #: The provider's reasoning-effort level; None to take the task's producer setting.
    effort: str | None = None

    @field_validator("model")
    @classmethod
    def _bare_model(cls, v: str) -> str:
        if not v or any(c in v for c in _MODEL_FORBIDDEN) or any(c.isspace() for c in v):
            raise ValueError(f"model must be a bare model id, without brackets or whitespace (a window is "
                             f"context_window), got {v!r}")
        return v

    @model_validator(mode="after")
    def _expressible(self) -> Arm:
        adapter_for(self).check(self)
        return self


class ArmRecord(Arm):
    """An arm as the plan records it: its table, its name and its route's table.

    The route's table makes a route repointed under the same name a different arm: another
    credential secret, another base-URL secret or another ``auth`` shows in
    ``arm_differences``. A value changed in the secrets file under the same secret name, a new
    URL for the same base-URL secret for instance, does not. The record holds names only, and
    a digest of a value would let a reader of a published record confirm a guessed host."""

    #: The arm's name, its ``[arms.<name>]`` key.
    name: ArmName
    #: The ``[routes.<name>]`` table that ``route`` named when the record was made: the names
    #: of its secrets and its ``auth``, never a value.
    route_table: Route

    @model_validator(mode="after")
    def _usable_route(self) -> ArmRecord:
        adapter_for(self).check_route(self.route_table)
        return self


class HarnessAdapter(Protocol):
    """How one harness runs an arm. Everything that differs between harnesses is here: every
    translation of an arm or its route into CLI settings, what the CLI reports back, where its
    session's transcript lies and how it reads, and the harbor agents that run the arm and read
    its logs. The members that need harbor or the agent's modules import them when called: harbor
    is loaded only in a process that runs or reads an agent, and those modules import this one."""

    #: The harness's name.
    name: str
    #: The API host the harness reaches when the route names no base URL.
    default_host: str
    #: Each context window the harness can be given, to the suffix its model id takes for it;
    #: empty when it takes any positive window through its configuration.
    windows: Mapping[int, str]
    #: The reasoning-effort levels the harness accepts.
    efforts: tuple[str, ...]
    #: The level ``HighestAvailable`` resolves to, by model id; the None key holds the level for
    #: every model the table does not name. An entry mapped to None says the provider publishes
    #: no levels for that model (``resolve_effort``).
    top_effort: Mapping[str | None, str | None]
    #: The variables a credential may arrive in, one for each way of presenting it the harness
    #: accepts.
    credential_vars: tuple[str, ...]
    #: The variable the harness reads its base URL from.
    base_url_var: str
    #: The variable harbor sets in the agent's exec requests to the CLI's configuration
    #: directory.
    config_dir_var: str
    #: Where the CLI writes a session's transcript under the agent's logs.
    transcript_layout: TranscriptLayout
    #: Text every command harbor sends to launch the harness's CLI holds, and no other command it
    #: sends does: the exec server's start line says whether a command holds it
    #: (``hls.agent.exec_protocol.is_cli_launch``), which the build reads as the evidence that a
    #: contestant process ran.
    cli_launch: str

    def check(self, arm: Arm) -> None:
        """Raise ValueError when the harness cannot run *arm* exactly as declared, or, for an
        ``ArmRecord``, exactly as recorded."""

    def check_route(self, route: Route) -> None:
        """Raise ValueError when the harness cannot use *route* as declared."""

    def grants(self, route: Route) -> dict[str, str]:
        """The variables the agent is granted for *route*: each to the name of the secret it
        holds."""

    def plain_env(self, arm: Arm) -> dict[str, str]:
        """The agent's variables that are settings, never secrets."""

    def model_string(self, arm: Arm) -> str:
        """The model as the harness is given it."""

    def cli_config(self, arm: Arm) -> dict[str, int]:
        """Configuration values the harness takes on its command line, by key."""

    def route_host(self, base_url: str | None) -> str:
        """The host the sandbox's seal lets the agent reach: that of *base_url*, the value of
        the route's granted base URL, or the harness's own host when the route names none."""

    def reported_model(self, arm: Arm) -> str:
        """The model the CLI says it ran when it runs *arm* (``hls.agent.preflight``)."""

    def reported_window(self, arm: Arm) -> int:
        """The context window the CLI says it runs *arm* with (``hls.agent.preflight``)."""

    def transcript_turns(self, objs: list[dict[str, Any]]) -> tuple[list[Turn], list[bool]]:
        """The assistant turns of a session transcript's JSON objects, and whether each tool
        result is a call the model could not form (``hls.agent.health``)."""

    def build_agent(self, arm: ArmRecord, run: AgentRun) -> Any:
        """The harbor agent that runs *arm* as *run* says (``hls.agent.harnesses.build_agent``,
        which checks the arm and the stage's grants first)."""

    def launch_env(self, agent: Any) -> frozenset[str]:
        """The names of the variables harbor itself sets in *agent*'s exec requests, beside the
        arm's route and plain settings (its ``extra_env``). Names only; ``agent_env.json``
        records them."""

    async def preflight(self, env: Any, agent: Any, arm: ArmRecord, *, config_dir: str, cwd: str,
                        timeout_sec: int) -> PreflightResult:
        """Run *arm*'s CLI once as *agent* through the sandbox environment *env* and judge what
        it read back (``hls.agent.preflight.preflight``)."""

    def log_reader(self, model: str, logs: Path) -> Any:
        """Harbor's agent for this harness reading a finished run's session files from *logs*,
        given only *model*, the model string the arm ran (``hls.agent.extract``)."""


def _check_effort(adapter: HarnessAdapter, arm: Arm) -> None:
    if arm.effort is not None and arm.effort not in adapter.efforts:
        raise ValueError(f"{adapter.name} accepts effort {', '.join(adapter.efforts)}, not {arm.effort!r}")


def _route_grants(credential_var: str, base_url_var: str, route: Route) -> dict[str, str]:
    grants = {credential_var: route.credential}
    if route.base_url is not None:
        grants[base_url_var] = route.base_url
    return grants


def _host(adapter: HarnessAdapter, base_url: str | None) -> str:
    if base_url is None:
        return adapter.default_host
    # The value is a secret, so no message shows it, and the parser's own error is not chained.
    problem = "the route's base URL must be an https URL with a host"
    try:
        parts = urlsplit(base_url)
    except ValueError:
        raise ValueError(problem) from None
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(problem)
    return parts.hostname


class ClaudeCodeAdapter:
    """Claude Code: the window rides in the model id and in ``auto_compact_window_var``, the
    route in ``ANTHROPIC_*`` variables.

    The model id's suffix sets the window the CLI sends against; the variable makes it compact
    by its own threshold before the window is full. Without the variable, Claude Code 2.1.251
    compacts a ``[1m]`` session only after the provider refuses a request in Anthropic's words
    ("prompt is too long"); Moonshot's refusal ("exceeded model token limit") is not recognised,
    so such a ``[1m]`` session is never compacted and fails once it outgrows the limit. With the
    variable the CLI compacts once a request would reach the window less min(output limit,
    20,000) less 13,000 tokens: 967,000 for a 1M arm with 64,000 output tokens."""

    name = "claude-code"
    default_host = "api.anthropic.com"
    #: Claude Code takes a 1M window only as the ``[1m]`` suffix, and otherwise uses its
    #: default, 200,000 tokens; no other window can be set.
    windows: Mapping[int, str] = {200_000: "", 1_000_000: "[1m]"}
    #: The variable Claude Code reads the window it compacts against from (100,000 to 1,000,000
    #: tokens).
    auto_compact_window_var = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"
    efforts = ("low", "medium", "high", "xhigh", "max")
    #: Claude Code's top level is "max" whatever the model.
    top_effort: Mapping[str | None, str | None] = {None: "max"}
    credential_vars = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    base_url_var = "ANTHROPIC_BASE_URL"
    config_dir_var = "CLAUDE_CONFIG_DIR"
    #: ``sessions/projects/<project>/<session>.jsonl``; a subagent's transcript, a level deeper,
    #: is not the session's.
    transcript_layout = TranscriptLayout(dirs=("projects", None), prefix="", suffix=".jsonl")
    #: Harbor's Claude Code run launches the CLI as ``exec claude`` in a subshell that records its
    #: pid.
    cli_launch = "exec claude "

    def check(self, arm: Arm) -> None:
        if arm.context_window not in self.windows:
            raise ValueError(f"claude-code can run with a context window of "
                             f"{' or '.join(f'{w:,}' for w in self.windows)} tokens, not {arm.context_window:,}")
        _check_effort(self, arm)

    def check_route(self, route: Route) -> None:
        """Claude Code reads an API key and a bearer token alike."""

    def grants(self, route: Route) -> dict[str, str]:
        var = "ANTHROPIC_AUTH_TOKEN" if route.auth == "bearer" else "ANTHROPIC_API_KEY"
        return _route_grants(var, self.base_url_var, route)

    def plain_env(self, arm: Arm) -> dict[str, str]:
        return {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(arm.output_limit),
                self.auto_compact_window_var: str(arm.context_window)}

    def model_string(self, arm: Arm) -> str:
        return arm.model + self.windows[arm.context_window]

    def cli_config(self, arm: Arm) -> dict[str, int]:
        return {}

    def route_host(self, base_url: str | None) -> str:
        return _host(self, base_url)

    def reported_model(self, arm: Arm) -> str:
        """The stream's init event names the model string the CLI was given, suffix and all."""
        return self.model_string(arm)

    def reported_window(self, arm: Arm) -> int:
        return arm.context_window

    def transcript_turns(self, objs: list[dict[str, Any]]) -> tuple[list[Turn], list[bool]]:
        return claude_turns(objs)

    def build_agent(self, arm: ArmRecord, run: AgentRun) -> Any:
        from hls.agent.harnesses import claude_code_agent

        return claude_code_agent(self, arm, run)

    def launch_env(self, agent: Any) -> frozenset[str]:
        """Harbor's run environment for the CLI (``ClaudeCode._build_run_env``: its retry
        watchdog, the model, the background and traffic switches) and the configuration
        directory."""
        return frozenset({*agent._build_run_env(), self.config_dir_var})

    async def preflight(self, env: Any, agent: Any, arm: ArmRecord, *, config_dir: str, cwd: str,
                        timeout_sec: int) -> PreflightResult:
        from hls.agent.preflight import claude_code_preflight

        return await claude_code_preflight(self, env, agent, arm, config_dir=config_dir, cwd=cwd,
                                           timeout_sec=timeout_sec)

    def log_reader(self, model: str, logs: Path) -> Any:
        from harbor.agents.installed.claude_code import ClaudeCode

        return ClaudeCode(logs_dir=logs, model_name=model)


class CodexAdapter:
    """Codex: the window and the output limit are configuration values, the route
    ``OPENAI_*`` variables."""

    name = "codex"
    default_host = "api.openai.com"
    windows: Mapping[int, str] = {}
    efforts = ("low", "medium", "high", "xhigh", "max")
    #: What each model accepts as its highest ``model_reasoning_effort``; a model not listed has
    #: no known top level.
    top_effort: Mapping[str | None, str | None] = {"gpt-5.6-sol": "max", "gpt-5.5": "xhigh"}
    credential_vars = ("OPENAI_API_KEY",)
    base_url_var = "OPENAI_BASE_URL"
    config_dir_var = "CODEX_HOME"
    #: ``sessions/<year>/<month>/<day>/rollout-<time>-<thread id>.jsonl``.
    transcript_layout = TranscriptLayout(dirs=(None, None, None), prefix="rollout-", suffix=".jsonl")
    #: Harbor's codex run launches the CLI, and resumes it, as ``codex exec``.
    cli_launch = "codex exec "
    #: Percent of the configured window codex reports as the window it uses, as its readback
    #: shows it (``hls.agent.preflight``).
    reported_window_percent = 95

    def check(self, arm: Arm) -> None:
        """Codex always runs a reasoning effort, harbor's "high" when it is given none, so a
        record without one is refused rather than run on a level it does not show. A declared
        arm without one takes its task's when it is resolved (``resolve_arm``)."""
        _check_effort(self, arm)
        if isinstance(arm, ArmRecord) and arm.effort is None:
            raise ValueError("codex always runs a reasoning effort (harbor's default is high), so the arm must "
                             "declare one or take its task's")

    def check_route(self, route: Route) -> None:
        if route.auth != "api_key":
            raise ValueError("codex presents its credential as OPENAI_API_KEY; route.auth must be api_key")

    def grants(self, route: Route) -> dict[str, str]:
        self.check_route(route)
        return _route_grants("OPENAI_API_KEY", self.base_url_var, route)

    def plain_env(self, arm: Arm) -> dict[str, str]:
        return {}

    def model_string(self, arm: Arm) -> str:
        return arm.model

    def cli_config(self, arm: Arm) -> dict[str, int]:
        return {"model_context_window": arm.context_window, "model_max_output_tokens": arm.output_limit}

    def route_host(self, base_url: str | None) -> str:
        return _host(self, base_url)

    def reported_model(self, arm: Arm) -> str:
        """The rollout's turn context names the bare model id."""
        return arm.model

    def reported_window(self, arm: Arm) -> int:
        return arm.context_window * self.reported_window_percent // 100

    def transcript_turns(self, objs: list[dict[str, Any]]) -> tuple[list[Turn], list[bool]]:
        return codex_turns(objs)

    def build_agent(self, arm: ArmRecord, run: AgentRun) -> Any:
        from hls.agent.harnesses import codex_agent

        return codex_agent(self, arm, run)

    def launch_env(self, agent: Any) -> frozenset[str]:
        """The CLI's home, the one variable harbor's codex run sets beside the route's."""
        return frozenset({self.config_dir_var})

    async def preflight(self, env: Any, agent: Any, arm: ArmRecord, *, config_dir: str, cwd: str,
                        timeout_sec: int) -> PreflightResult:
        from hls.agent.preflight import codex_preflight

        return await codex_preflight(self, env, agent, arm, config_dir=config_dir, cwd=cwd, timeout_sec=timeout_sec)

    def log_reader(self, model: str, logs: Path) -> Any:
        from harbor.agents.installed.codex import Codex

        return Codex(logs_dir=logs, model_name=model)


def registry(*adapters: HarnessAdapter) -> dict[str, HarnessAdapter]:
    """*adapters* by their harness's name. Raise ArmError for one that declares no text its CLI's
    launch holds (``cli_launch``): no start line could say its CLI ran, so every failure of its
    agent's part would be taken for the harness's."""
    for adapter in adapters:
        if not str(getattr(adapter, "cli_launch", "")).strip():
            raise ArmError(f"harness {adapter.name!r} declares no text its CLI's launch holds (cli_launch)")
    return {adapter.name: adapter for adapter in adapters}


#: Every harness an arm may run on, by name.
ADAPTERS: dict[str, HarnessAdapter] = registry(ClaudeCodeAdapter(), CodexAdapter())


def adapter_named(harness: str) -> HarnessAdapter:
    """The adapter of the harness called *harness*. Raise ArmError for a name no adapter runs,
    so no harness is ever run as another."""
    try:
        return ADAPTERS[harness]
    except KeyError:
        raise ArmError(f"no adapter runs harness {harness!r}; hls runs {', '.join(sorted(ADAPTERS))}") from None


def adapter_for(arm: Arm) -> HarnessAdapter:
    """The adapter of *arm*'s harness (``adapter_named``)."""
    return adapter_named(arm.harness)


def route_values(adapter: HarnessAdapter, env: Mapping[str, str | None]) -> list[tuple[str, str]]:
    """Every form (``hls.substrate.secret_forms.forms_of``) of the values of *adapter*'s route
    variables that *env* holds, each with its label: the credential's and the base URL's under
    their variables, and the base URL's host, when it identifies a machine, as
    ``<variable>.host``. Longest first, so a value that holds another is replaced before it. No
    record, log or command string may hold any of them."""
    found = [(form.label, form.text) for var in (*adapter.credential_vars, adapter.base_url_var)
             for form in forms_of(var, env.get(var) or "")]
    return sorted(found, key=lambda item: -len(item[1]))


def check_granted_base_urls(grants: Mapping[str, str], values: Mapping[str, str]) -> None:
    """Raise ValueError when *grants* (variable to secret name) give a harness's base-URL
    variable a secret whose value in *values* (secret name to value) is not a base URL that
    harness takes (its ``route_host``). The message names the secret and the variable, never
    the value, and chains nothing."""
    for adapter in ADAPTERS.values():
        name = grants.get(adapter.base_url_var)
        if name is None:
            continue
        try:
            adapter.route_host(values[name])
        except ValueError:
            raise ValueError(f"the base URL secret {name}, granted as {adapter.base_url_var}, must hold an https "
                             "URL with a host") from None


def check_arm_route(arm: Arm, routes: Mapping[str, Route]) -> None:
    """Raise ValueError when *arm*'s route is not one of *routes* (route name to route) or its
    harness cannot use it."""
    if arm.route not in routes:
        # The value is not shown, as in RouteName's refusal: a pasted credential can be a valid name.
        raise ValueError(f"route names no [routes.<name>] table; [routes] declares {sorted(routes) or 'none'}")
    adapter_for(arm).check_route(routes[arm.route])


def route_of(settings: Settings, arm: Arm) -> Route:
    """The route *arm* names, from *settings*. Raise ArmError, naming a resolved arm, when the
    route is not declared there or the arm's harness cannot use it, or when *arm* is a record
    whose route table is not the route *settings* declare, so that what a stage is granted is
    what its record names."""
    try:
        check_arm_route(arm, settings.routes)
    except ValueError as e:
        raise ArmError(f"arm {arm.name}: {e}" if isinstance(arm, ArmRecord) else str(e)) from e
    route = settings.routes[arm.route]
    if isinstance(arm, ArmRecord) and arm.route_table != route:
        raise ArmError(f"arm {arm.name}: its record holds route {arm.route} as another [routes.{arm.route}] declared "
                       "it; resolve the arm again from these settings")
    return route


def resolve_effort(effort: HighestAvailable | ProviderLevel | None, *, adapter: HarnessAdapter,
                   model: str) -> str | None:
    """The provider level an arm with no effort of its own launches at, from the task's producer
    setting *effort*: a ``ProviderLevel`` as written, ``HighestAvailable`` through *adapter*'s
    ``top_effort`` (*model*'s entry, else the entry for every model), None when unset. A hit is a
    key that is present, not a value that is set: an entry mapped to None says the provider
    publishes no levels. Raise ArmError for a model the table does not list, so no arm launches
    at a guessed level."""
    if effort is None:
        return None
    if isinstance(effort, ProviderLevel):
        return effort.level
    for key in (model, None):
        if key in adapter.top_effort:
            return adapter.top_effort[key]
    raise ArmError(f"the task asks for the highest effort, but {adapter.name} lists no top effort for model "
                   f"{model!r}; add it to {type(adapter).__name__}.top_effort, or set the arm's effort")


def resolve_arm(settings: Settings, name: str, *,
                task_effort: HighestAvailable | ProviderLevel | None) -> ArmRecord:
    """Arm *name* of *settings* as the plan records it. An arm that leaves ``effort`` unset
    takes the task's producer setting *task_effort*, resolved for the arm's harness and model
    through the adapter's ``top_effort`` (``resolve_effort``); the level must be one the harness
    accepts. Raise ArmError naming the arm."""
    if name not in settings.arms:
        raise ArmError(f"no arm named {name!r} in hls.toml; [arms] declares {sorted(settings.arms) or 'none'}")
    arm = settings.arms[name]
    try:
        route = route_of(settings, arm)
    except ArmError as e:
        raise ArmError(f"arm {name}: {e}") from e
    effort = arm.effort
    if effort is None:
        try:
            effort = resolve_effort(task_effort, adapter=adapter_for(arm), model=arm.model)
        except ArmError as e:
            raise ArmError(f"arm {name}: {e}") from e
    try:
        return ArmRecord(name=name, route_table=route, **{**arm.model_dump(), "effort": effort})
    except ValidationError as e:
        reasons = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors(include_input=False))
        raise ArmError(f"arm {name}: {reasons}") from e


def arm_differences(old: ArmRecord, new: ArmRecord) -> list[str]:
    """Each field in which *new* differs from *old*, as "<field>: <old> -> <new>", the route
    table's fields dotted ("route_table.base_url: ..."); empty when the two are the same arm.
    The route is compared by its name, its secrets' names and its ``auth``, which is all a
    record holds."""
    a, b = _fields(old), _fields(new)
    return [f"{k}: {a[k]!r} -> {b[k]!r}" for k in a if a[k] != b[k]]


def _fields(record: ArmRecord) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in record.model_dump().items():
        if isinstance(value, dict):
            out.update({f"{key}.{sub}": v for sub, v in value.items()})
        else:
            out[key] = value
    return out
