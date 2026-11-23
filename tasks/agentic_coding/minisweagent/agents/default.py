"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.

Forked from mini-SWE-agent ``src/minisweagent/agents/default.py`` at
``04d809ceab9df28f9adaed044884180159172930``. MIT licensed, Copyright (c) 2025
Kilian A. Lieret and Carlos E. Jimenez; see ``LICENSE`` beside this package.

Divergences from upstream, each also commented where it appears:

1. ``run``/``step``/``query``/``execute_actions`` are ``async def``, and the two calls
   that leave this process -- ``self.model.query`` and ``self.env.execute`` -- are
   awaited. Nothing else about the control flow moved.
2. A context guard in ``query``, beside upstream's own limit checks. Upstream has
   none; this task needs one and ``ContextExhausted`` says why.
3. ``cost_limit`` defaults to 0 (upstream: 3.0) and ``wall_time_limit_seconds`` stays
   at upstream's 0. Both are disabled rather than deleted so the diff stays small;
   ``AgentConfig`` says why each is wrong here.
4. Templates render through ``str.format``, not jinja2. Nothing here has a placeholder
   ``str.format`` cannot fill, and jinja2 is a real dependency to carry into a build
   sandbox for no gain.
5. ``run`` takes a ``prefix``: a trajectory can start from a frozen conversation
   rather than from the rendered opening pair, because one from turn 0 does not reach
   deep prompts in a drive window. The opening is rendered either way, so a prefix
   cut under templates that have since changed is refused. See ``prefix.Prefix``.
6. ``output_path``, ``save`` and ``serialize`` are gone. Upstream writes a trajectory
   file; the graded record here is the ``TraceEvent`` stream, which the adapter emits
   per LLM call, so a second serialization of the same conversation would be a
   redundant format to keep in step. It was also quadratic: upstream calls ``save``
   after every step and ``serialize`` deep-merges the whole message list whether or
   not there is a path, over 250 turns of a transcript reaching hundreds of thousands
   of tokens.
"""

import logging
import time
import traceback

from pydantic import BaseModel

from ..exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from ..utils.serialize import recursive_merge


class ContextExhausted(LimitsExceeded):
    """Raised when the next request would not fit in the model's context window.

    Not upstream's: upstream lets the provider refuse an over-length request. Here the
    trajectory has to end on its own terms instead, because the task scores makespan
    and a server's rejection is an error rather than an outcome -- and because a
    coding agent has no final-answer turn to fall back on the way a research agent
    does. Its work is the state it left on disk.

    Every number this branches on has to be one the harness owns, for the reason
    ``AgentConfig.max_context_len`` gives."""


class EmptyReply(InterruptAgentFlow):
    """Raised when a completion carried no tokens at all.

    Not upstream's. It is the one ending that names the server rather than the model:
    every other way a turn fails to produce a call is the model answering badly, which
    is asked again. Calling those empty would report a starved turn as a broken
    artifact. Carries an ``exit`` message, so upstream's own ``InterruptAgentFlow``
    handling in ``run`` ends the trajectory with no extra branch."""


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 0.0
    """Stop agent after exceeding (!) this cost.

    Upstream defaults this to 3.0 US dollars. Disabled here, because nothing bills:
    the artifact is a server the contestant deployed on hardware the run already paid
    for, and ``query`` never adds anything to ``self.cost``. Left in place rather than
    removed so the check in ``query`` stays upstream's."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit.

    Left at upstream's 0, and it must stay there. Wall clock is the quantity this task
    scores, and the artifact is what spends it, so a trajectory that ended on a wall
    limit would hand a slow server the thing it is being scored on: shorter
    trajectories. The harness bounds wall time over the whole drive rather than per
    trajectory, where it cannot be gamed that way: the scheduled stop
    (``AgentSpec.wall_clock_sec``), which every server gets alike, and behind it
    ``AgentSpec.drive_timeout_sec``, whose remainder is recorded as faults."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    max_context_len: int | None = None
    """Context window, in tokens, the trajectory is budgeted against. None: no budget.

    Not upstream's. Declared by the task, never read off the server's own
    ``hls_context``: this is a stopping rule, the task scores how long a trajectory
    takes, and a server that reported a tiny window would end every trajectory on turn
    one and collect the speedup for it."""
    max_output_tokens: int | None = None
    """The per-request output cap the harness stamps on every turn. None reserves
    nothing.

    Not upstream's. Reserved beside the prompt by the context guard: a prompt that
    fits the window with no room left to answer is as fatal as one that does not fit."""


class DefaultAgent:
    def __init__(self, model, env, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return template.format(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    async def run(self, task: str = "", *, prefix=None, **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        opening = [
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        ]
        # Not upstream's. A trajectory can start from a frozen conversation instead of
        # from the opening pair; see `prefix.Prefix`. The opening is still rendered, so
        # a prefix cut under different templates is refused rather than served.
        if prefix is None:
            self.add_messages(*opening)
        else:
            prefix.check_opening(opening)
            self.add_messages(*(dict(m) for m in prefix.messages))
            # A stopping rule, so it resumes where the pre-roll left it. `n_calls` does
            # not: the step limit is what this session serves, not where it began.
            self.n_consecutive_format_errors = prefix.consecutive_format_errors
        while True:
            try:
                await self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    async def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return await self.execute_actions(await self.query())

    async def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        # Not upstream's, and checked before the send so an over-length prompt is never
        # a request. The numbers are config; the counting is asked of the model,
        # because that is what holds the checkpoint's tokenizer.
        if await self.over_context_budget():
            raise ContextExhausted(
                {
                    "role": "exit",
                    "content": "ContextExhausted",
                    "extra": {"exit_status": "ContextExhausted", "submission": ""},
                }
            )
        self.n_calls += 1
        message = await self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    async def over_context_budget(self) -> bool:
        """Whether sending the conversation so far would overrun the context window.

        Not upstream's. Unarmed when the task declared no window, or when nothing in
        this process can count the prompt -- the model says so by answering None, and
        an unarmed guard is the safe direction: it lets a trajectory run long rather
        than ending one that had room."""
        if self.config.max_context_len is None:
            return False
        prompt = await self.model.count_prompt_tokens(self.messages)
        if prompt is None:
            return False
        return prompt + (self.config.max_output_tokens or 0) > self.config.max_context_len

    async def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [await self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
