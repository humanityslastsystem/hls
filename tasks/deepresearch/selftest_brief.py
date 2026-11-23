"""The implementer brief for deepresearch: the objective and the workload block, in words a
builder can act on."""
from __future__ import annotations

from hls.prompt import ImplementerBrief

_BRIEF_BLOCKS = """{% extends "brief" %}
{% block goal %}{% set target = "{:,.0f}".format(ts.workload.slo.target_sec) -%}
Build a hyperoptimized serving stack for `{{ model.base_checkpoint }}` under the
workload below. Optimize until you reach the physical limits of the hardware or
exhaust your budget. The grader drives research trajectories against one instance of
your server.

**Your score is your speedup over a production serving framework in trajectory
makespan.** A trajectory is one research question end to end, every LLM call until
the final answer; its makespan runs from its first request sent to its last
response received. For every trajectory:

    r = makespan / {{ target }} s

    score = exp( mean over trajectories of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling; the {{ target }} s target only normalizes and is not a
pass/fail bar. Trajectories arrive open-loop on a schedule you do not control. Tool
execution and the think time between turns belong to the harness and count the same
for both stacks.

**Your implementation must not fail any request.** Every request must receive a
complete streaming response. Correctness is checked at
the token level against a bf16 reference implementation of this model, one item per
LLM call of every trajectory, carrying the conversation that request sent and that
turn's tool definitions: the distribution you report at every served position is
compared with the reference's on the same prefix, and the tokens you serve are tested
for being draws from what you report. The run fails if the mean KL divergence from your
reported distribution to the reference's exceeds {% if verification and verification.kind == "report" %}{{ verification.kl_ceiling_str }} nats per token{% else %}a ceiling calibrated on trusted serving stacks{% endif %},
or if the served tokens are not faithful draws from the report.

The thresholds are calibrated on fast production-grade serving stacks. What fails is
genuine deviation: quantized weights, altered algorithms, modified sampling. The gate
is pass/fail with no credit for margin -- a run that passes at 90% of a ceiling scores
identically to one at 10% -- so latency spent buying faithfulness headroom below the
ceiling is pure loss.{% endblock %}
{% block workload %}- Arrival: {{ workload.sessions }} (one arrival = one research trajectory)
- Per-request output cap: {{ workload.output_cap }}
- SLO: {{ workload.slo }}

**Shape of a trajectory.** The harness runs a deep-research loop against your server:
one chat completion per turn, carrying the system prompt, three tool definitions
(`web_search`, `web_open`, `web_find`) and the whole conversation so far. Your server
returns one assistant turn; if it contains a tool call, the harness executes it and
sends the result back as a `tool` message whose `tool_call_id` matches the
`tool_calls` it echoes on your assistant message, together with your
`reasoning_content`, and the tokenizer's chat template renders all of it. Then it
sends the next completion.
Your server never executes a tool. The `content` you return must be the reply decoded with
the model's control tokens kept (`skip_special_tokens=False`): the loop reads the
tool-call and reasoning delimiters from that text, and a decoder that drops special
tokens drops them too. A trajectory is about 18 completions at the median
(search turns p50 14, p95 29, max 33), prompts grow from a few thousand tokens to a
median of 25k and a p95 of 56k by the final turn, and completions are short (p50 200
tokens, p95 690) except for occasional runaway turns that hit the cap.
{%- if ts.agent and ts.agent.max_concurrent_llm %} At most {{ ts.agent.max_concurrent_llm }}
trajectories are in flight at once; turn-over-turn prompts share a long common prefix.
{% else %} Trajectories
are admitted as they arrive, so how many are in flight at once is whatever the
schedule produces rather than a fixed ceiling; turn-over-turn prompts share a long
common prefix.
{% endif %}

**Test with the real loop.** The loop and its retrieval tool are installed at
`/app/tasks/deepresearch/`; drive dev questions against your server with

    python /app/tasks/deepresearch/selftest.py --server http://localhost:8000 --n 8 --concurrency 8 --chat-format {{ model.chat_format }}

It reads the dev split from `/data/tenants/t0/dev/rows.json`, sends the grader's
request bodies byte for byte, and prints calls, prompt and completion sizes, latency,
makespan and dev exact match under the grader's own answer-matching rule. The
retrieval bundle at `/data/tools/browsecomp` is development-only data for that tool;
your server must not read it. During the self-test the retriever loads in your
sandbox and, when your GPU has 40 GB free, encodes on it (`--device cpu` keeps the
device for your server).
During evaluation the retriever runs in a separate harness container on its own GPU;
your server never shares a device with it, so reserve nothing for it.{% endblock %}
"""


class DeepResearchBrief(ImplementerBrief):
    """ImplementerBrief plus this task's two overridden blocks.

    The goal block, because the base template's says the score is goodput, and this
    task is graded on speedup over the baseline stack in trajectory makespan
    (``hls.scoring.SloLatencyScorer``), whose makespan target is a normalizer rather
    than a bar and whose failed-request rule is a hard run gate. Both facts have to be
    unmissable in the brief the agent actually reads.

    The workload block, for the tool-augmented trajectory shape, the self-test command,
    and where retrieval runs at build and at grade."""

    template_str: str | None = _BRIEF_BLOCKS
