# GLM-5.3-Flash reference log-probabilities

This directory holds the log-probabilities the reference stack produced on a fixed set
of prompts, and a script that scores your own server's log-probabilities against them
the way the sampling gate does.

## Files

| file | what it is |
| --- | --- |
| `prompts.jsonl` | The prompts. One JSON object per line. |
| `reference.jsonl` | What the reference says on each of them, on both of its two paths. One line per prompt. |
| `selfcheck.py` | Scores a file of your own against `reference.jsonl`. Standard library only. |

You cannot run the reference yourself. It is an eight-card deployment of the full
checkpoint under a serving engine your sandbox does not have, and the sandbox has
neither the hardware nor the engine. That is why this directory exists: the numbers
were produced once, on the machine that grades, and shipped.

The programs that produced them (`gen.py`, `forced.py`, `merge.py`, `serve.sh`,
`stop.sh`) are kept with the run that made this directory rather than shipped here.
They name paths on that machine and would not run anywhere else.

## The reference stack

| | |
| --- | --- |
| image | `hls-fs-3ad44931d896.sif` |
| engine | vLLM 0.29.1 nightly, commit `a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b` |
| checkpoint | `zai-org/GLM-5.3-Flash`, FP8 as published, revision `main` |
| hardware | one node of eight H200, NVLink |

### Two paths, because the gate uses one of them

The numbers were produced twice, by two runs on the same node.

**The decode path.** The baseline server sampling a continuation and streaming its
report, which is the thing your own server is built to replace. Engine arguments, from
`tasks/agentic_coding/serving_config.py` `engine_args()`:

```
dtype=bfloat16  tensor_parallel_size=8  enable_expert_parallel=True
max_model_len=1048576  enforce_eager=False  trust_remote_code=True
max_num_seqs=32  max_logprobs=21
```

`gpu_memory_utilization` is left at the engine's own default, which is what the shipped
baseline does. Decoding: temperature 1, top-p 1, no top-k, no min-p, no penalties, no
logit bias. The request body carries no sampling fields at all, which is also what a
graded request carries; the server pins those values itself. Each prompt was sent once,
non-streaming, with `logprobs: true`, `top_logprobs: 21` and `max_tokens: 512`; every
reply ended on a stop string well inside that cap.

The eight `continuation` prompts ask one more thing of the server. Their request carries
`continue_final_message: true` and `add_generation_prompt: false`, and the server has to
resume the final assistant message instead of opening a new turn after it. The baseline
server this pass first ran on ignored both fields: it appended `<|assistant|><think>`
after the cut message, so the model started a fresh thought, and the prompt it counted
came out two tokens longer than the conversation's own. Those eight were sampled again
through the same image and engine, with the server's prompt render changed to honour
both fields and nothing else changed. On every prompt in this directory the
`prompt_tokens` the server reported equals the prompt's `token_count` in
`prompts.jsonl`.

**The grader had the same fault, and it is fixed.** When it teacher-forced the second
request of a two-request turn, it rebuilt the context with a fresh generation prompt,
`...</think><|assistant|><think>`, instead of resuming the cut message. The grader now
renders the contract prompt: it resumes the cut message, as the rows in this directory do,
so on those turns it compares your tokens against the same context used here.

**The forced path, which is the one the gate reads.** The gate does not compare you
against what the reference streamed. It takes the tokens *your* server served, feeds
the conversation plus those tokens back through the reference as a single prompt, and
reads the reference's distribution at each of those positions off the prompt-logprob
path. That is a prefill, not a decode. So the reference was run a second time, this
time teacher-forced on the continuation the decode path had sampled, through the
reference engine the task itself pins:

```
dtype=auto (resolves to bfloat16)  quantization=fp8  tensor_parallel_size=8
gpu_memory_utilization=0.90  max_model_len=1048576  enforce_eager=False
enable_prefix_caching=False
```

`prompt_logprobs` was 21 rather than the gate's own 0: the gate reduces whole-vocabulary
rows to three numbers per position, and a file you can hold has to name ids instead. It
is the same forward over the same logits, reduced differently.

**Prefix caching is off on both.** A position served out of the cache is not recomputed,
and on this stack the numbers depend on how a prompt was prefilled, so a prompt whose
prefix a neighbouring prompt happened to warm would not get the numbers it gets cold. A
reference has to hold the same values every time it is generated.

### The two paths do not agree, and that is the floor

vLLM prefills and decodes with different kernels. The same weights, the same node, the
same tokens, and the two paths still differ. Measured over every generated position in
this directory, with `selfcheck.py`'s own arithmetic, the decode rows put where your
report would go and the forced rows as the reference:

| | mean_kl, nats per token | positions |
| --- | --- | --- |
| all prompts | 8.87e-03 | 4,201 |
| `dev_first_turn` | 6.74e-03 | 2,116 |
| `long_context` | 1.17e-02 | 1,485 |
| `continuation` | 9.25e-03 | 600 |
| the gate's divergence ceiling, for scale | 1.269864e-01 | |

Nothing can be asked to sit below that. If your own divergence is at this level you are
as close to the reference as the reference is to itself.

The three kinds sit within a factor of two of each other, and all of them sit more than
ten times under the ceiling. The `continuation` row was first measured at 2.75e-01,
above the ceiling. That number came from the render fault described under the decode
path, not from the kernels. The decode pass had sampled those continuations after a
fresh `<|assistant|><think>`, and the forced pass teacher-forced them where the
contract puts them, directly after the cut message. Rendered the same way on both
paths, a continuation turn is no harder than a first turn.

### The prompt rows move with the length of the forward

`prompt_forced` is exactly what the reference's prefill returned for the forward that
produced it: the prompt followed by the continuation shipped on the same line, at that
total length. It is deterministic. The same sequence run twice, in one engine session or
in two, gives the same rows to the last digit. It is also causal. Replacing the tokens
after a position with different tokens, at the same total length, leaves every row up to
that position bit-identical.

It does depend on the forward's total length, and by far more than the floor above.
`dev-06` and `cont-06` open with the same 886 tokens. Running that prefix at several
total lengths, forwards of 900, 953 and 1,001 tokens give bit-identical rows. Forwards of
886, 1,026, 1,040, 1,072, 1,100, 1,200 and 1,400 tokens fall into other classes. Two
classes differ at the same position like this:

| | |
| --- | --- |
| mean absolute difference in the occupying token's log-probability | 0.89 to 0.92 nats |
| positions whose most probable token changes | about 27% |
| median absolute difference at positions 1 to 63 | 0.17 to 0.24 nats |
| `selfcheck.py` divergence, `cont` rows against `dev` rows at shared positions, 8 pairs | 0.52 nats per token over 7,269 positions |

The same holds across the whole directory. Every prompt opens with the same system
prompt, and at those shared positions, forwards of close total length agree bit-exactly
while forwards of different length differ by this much.

This is not a matter of uncertain positions. Where the reference puts 99% or more on its
most probable token, two length classes still differ by 0.13 nats per token at prompt
positions. At generated positions, the reference's decode and forced rows differ by
2.5e-04 at the same confidence.

The generated positions are steady. Every forward's rows at its own continuation agree
with the decode path to the floor in the table above. The unsteady rows are the system
and user positions that make up most of every prompt. One explanation fits, though this
directory does not prove it: post-training teaches the model to produce assistant text,
not system or user text, so nothing holds its predictions at those positions in place and
they move with rounding. `long_context` prompts end in tool output, text of the same kind;
their rows there were not measured against a second forward.

For your first step, this means three things. Use `forced` as the prefill comparison.
Feed exactly the prompt followed by the reference's tokens, so your forward is the
reference's forward. Read `prompt_forced` with the table above: a divergence of half a
nat per token at prompt positions is what the reference shows against itself, and it
says nothing about your prefill.

## prompts.jsonl

Per line:

| field | meaning |
| --- | --- |
| `id` | Names the prompt. The same id keys `reference.jsonl`. |
| `kind` | `dev_first_turn`, `long_context` or `continuation`. |
| `token_count` | What the conversation renders to through the checkpoint's chat template with the bash tool definition, counted with `add_special_tokens=False`. |
| `messages` | The conversation, in OpenAI chat form, exactly as the grader puts it on the wire. |
| `grader_request_fields` | Every other field of the request body the grader sends for a turn of this shape. |
| `source` | Where the prompt came from. |
| `continue_final_message` | Whether this is the second request of a two-request turn. |

The request body is `messages` plus `grader_request_fields`. They are stored apart
because writing a half-million-token conversation twice per line would double the file
for no gain.

`grader_request_fields` was built by the grader's own wire protocol, so its field set is
the field set a graded request carries, including `tool_choice: "none"` and
`return_token_ids: true`. Its `max_tokens` is the cap the workload stamps: 32,768 on a
turn's first request and 16,384 on a continuation. The decode pass replaced that with 512 and
changed nothing else.

The three kinds:

**`dev_first_turn`** (28 prompts). The opening system and user messages of each
development-split row, rendered by the task's own agent code rather than by a
description of it. These are short, about 850 to 2,000 tokens.

**`long_context`** (12 prompts). A synthetic coding trajectory whose tool observations
are public CUDA, PTX and toolchain documentation, grown until the prompt reaches a
target length. Two prompts each at roughly 32k, 64k, 128k, 256k, 384k and 512k tokens.
The shape is the workload's: an assistant message carrying reasoning and one or more
bash calls, then one `tool` message per call holding the command's output rendered
through the loop's own observation renderer at its 32,000-character limit. The documents
are drawn in a per-prompt shuffled order, so the two prompts in one band hold different
text.

**`continuation`** (8 prompts). A first turn plus the assistant message a cut reply is
handed back as. Under the task's two-request rule, a reply that runs to its output cap
is returned to the model as an assistant message holding exactly the text the server
produced, with `</think>` appended when the cut landed inside the thought, and the second
request carries `continue_final_message: true` and `add_generation_prompt: false`. These
prompts are that second request.

## reference.jsonl

One line per prompt, keyed by the same `id` as `prompts.jsonl`:

```json
{"id": "dev-03-formal-crypto",
 "finish_reason": "stop",
 "n_prompt_tokens": 1101,
 "dense_tail": 1024, "stride": 64,
 "prompt_positions": [64, 128, ..., 1099, 1100],
 "prompt_forced": [{"token_id": 811, "logprob": -2.104, "top21": [[811, -2.104], ...]},
                   ...],
 "tokens": [{"decode": {"token_id": 12345, "logprob": -0.31,
                        "top21": [[12345, -0.31], [678, -1.94], ...]},
             "forced": {"token_id": 12345, "logprob": -0.309,
                        "top21": [[12345, -0.309], [678, -1.941], ...]}},
            ...]}
```

`tokens` is one entry per generated position, in order. Both sides name the same
`token_id`, the token the reference sampled there. `decode` is what the reference
reported while sampling it; `forced` is what the reference reports about it on the
prefill path. `top21` is the position's 21 most probable ids with their
log-probabilities, most probable first, ties broken on the id.

`prompt_forced` is the reference's prefill output at the prompt's own positions, before
anything was sampled: entry *k* is the distribution over the prompt's token at index
`prompt_positions[k]`, conditioned on everything before it, in the forward over the
prompt followed by the same line's continuation. The section on prompt rows above says
why that qualification matters. The positions sampled are
every index in the last 1024 tokens of the prompt plus every 64th index before that, so
a 512k-token prompt contributes about 9,200 rows and the deep-context end of the prompt
is taken whole. Index 0 is never included: nothing conditions it.

Log-probabilities are rounded to six decimal places on every path. That is 1e-6 nats,
four orders of magnitude under the divergence ceiling and well under the gap between
the two paths. Both paths are rounded the same way, so a difference between them comes
from the kernels and not from the file.

Decoded token text is not kept anywhere; the comparison keys on ids.

## selfcheck.py

```
python3 selfcheck.py reference.jsonl mine.jsonl
python3 selfcheck.py reference.jsonl mine.jsonl --against decode
python3 selfcheck.py --self-test
```

`mine.jsonl` is **teacher-forced on the reference's continuation**: for each prompt, one
entry per position of the reference's continuation, in order, where `token_id` is the
reference's token at that position, `logprob` is the log-probability **your** server
assigns it there, and `top21` is **your** server's 21 most probable tokens there. You
produce it by feeding your server the prompt followed by the reference's tokens and
reading off the per-position distributions. Your side has no `decode`/`forced` split:

```json
{"id": "dev-03-formal-crypto",
 "prompt_positions": [64, 128, ...],
 "prompt_forced": [{"token_id": 811, "logprob": -2.1, "top21": [...]}, ...],
 "tokens": [{"token_id": 12345, "logprob": -0.31, "top21": [...]}, ...]}
```

`prompt_forced` is optional. Include it and `selfcheck.py` reports your prefill
separately; leave it out and only the generated positions are scored.

By default your file is scored against the reference's `forced` side, because that is
the side the gate reads. `--against decode` scores the other, which is useful only for
telling the two apart.

### The two-step check

**First, your prefill against the reference's.** Feed your server each prompt followed
by the reference's tokens, rendered as `prompts.jsonl` renders it, and score the rows
against `forced`, which is the default. Both sides are prefill, so a divergence there is
a prefill fault and your decode kernel is not implicated. The reference's own decode
sits at the floor in the table above against these rows, so that floor is the scale for
reading your number. Score `prompt_forced` too, but read it against the section on
prompt rows above: the reference's own rows at prompt positions move by far more than
the floor when the forward's total length changes.

**Second, your decode against your own prefill.** Once your prefill matches, sample a
continuation with your own server, teacher-force it back through your own prefill, and
compare the two. Write a file shaped like `reference.jsonl` with your prefill rows as
`forced` and your decode rows as `decode`, and pass it to `selfcheck.py` as the
reference with your decode rows as `mine.jsonl`. The divergence it prints is your decode
against your prefill. The reference's own gap for the same kind of prompt, from the table
above, is the level below which a difference means nothing.

### What it checks

The gate runs five checks. Two are computable from a teacher-forced file and
`selfcheck.py` computes exactly those two.

**Divergence.** Per position, the divergence of your reported distribution from the
reference's, coarsened onto the partition your own report fixes:

$$\mathrm{KL} = \sum_i q_i\,(\log q_i - \log p_i) \;+\; q_{\text{tail}}\,(\log q_{\text{tail}} - \log p_{\text{tail}})$$

with $q$ your report and $p$ the reference's. The atoms are your report's entries
strictly above its 20th largest value, at most 19 of them, with a tie group at that
value tailed whole; duplicates and an entry for the served token that sits strictly
below every other entry are dropped first. The mean over every measured position is
compared against

| | |
| --- | --- |
| ceiling | `1.269864e-01` nats per token |

That ceiling is twice the divergence a second trusted implementation of this checkpoint
already shows against the same reference. Note what that comparison is: a trusted
**decode** report against the **forced** reference. Your teacher-forced file is a
prefill against a prefill, which is the much closer of the two comparisons, so a number
anywhere near the ceiling here is a bad sign rather than a near miss.

**The other four checks are not computable from a teacher-forced file.** Rank
uniformity, the log-score moment, the end-of-turn count and the served-outside rate all
read a null about *your* sampler drawing *your* served tokens. A teacher-forced file
holds no tokens your sampler drew; the tokens are the reference's. `selfcheck.py` does
not approximate them.

The served-outside rate is still printed, because it says how often your head disagrees
with the reference's about the token the reference happened to draw, and nothing fails
on it. For scale, the reference's own forced rows scored the same way give

| | |
| --- | --- |
| the reference against its own forced rows | `4.523e-03` |
| the gate's limit, on the pairing the gate actually measures | `4.719887e-03` |

The reference against itself already sits within 5% of the gate's limit on this
pairing. A server whose prefill reproduced the reference exactly would print the same
rate, so a rate near the limit here says nothing about check 5.

### Where it is exact and where it is not

The gate's reference is a full-vocabulary pass and can quote its log-probability at any
token id your report names. This file's reference is 21 entries per position. The two
agree exactly wherever every atom of your partition is one of the reference's reported
ids. Where one is not, that atom's reference probability is unknown; `selfcheck.py`
substitutes the largest value it can still be, which makes the position's divergence a
lower bound rather than a guess. It prints how many positions were scored that way and
how much of your own probability sat on those atoms. The second number is what the
substitution's error is weighted by, so it is the one that says whether the first
matters.

Separately, on every position, the gate sums the reference's tail mass in log space over
the vocabulary outside your atoms, while 21 entries admit only `log1p(-sum)` over the
atoms. The two differ once the atoms hold all but about 1e-12 of the mass.

`--self-test` checks the script against cases whose answers are known: a file equal to
the reference scores zero divergence and zero outside rate at generated positions and at
prompt positions, a perturbed file scores strictly above zero, a file that sampled the
prompt at different positions is refused rather than scored, the decode-against-forced
floor is zero when the two sides are the same record and above zero when they are not, a
row one alternative short is refused, a row claiming more than unit mass is refused, a
row of 21 equal values keeps no atom, the ceiling substituted for an unreported atom
comes from the reference's ranked entries rather than from its own served token, and the
clustered standard error matches a hand computation.

### Self-checking is not the gate

A pass here is evidence, not a verdict. It reads one of five checks, on prompts that are
not the graded ones, against a reference window 21 entries wide, on a pairing of
prefill against prefill where the gate pairs your decode against the reference's
prefill.
