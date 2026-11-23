# Scoring

A grading ends with one number, `value` in `score.json`. Every `hls run` ends with a grading of
the server the agent built (the submission), and `hls grade` runs one on its own. For every task
except `sample_task`, the value is a speedup over the task's baseline server. 1.0 means the submission answered as fast as the
baseline, 2.0 means a typical request took half as long, and there is no ceiling. A value of 0
means the run did not count, and `zero_reasons` says why.

To grade a submission, the harness starts its server and sends it the task's workload while
timing every request. It then checks the replies for correctness and turns the timings into the
score.

## The correctness check (the gate)

A run must pass a correctness check, which HLS calls the gate, to score above zero. The gate is
pass or fail, so a run that passes with room to spare scores the same as one that barely passes.

Every request must get a complete reply. If a single request errors, times out or returns a
malformed reply, the run fails before any of its content is checked. A server therefore cannot
drop requests to protect its latency.

Each task then declares one of three kinds of check. All three compare against a reference
implementation of the same model, which the harness runs itself:

| Check | Tasks | What it checks |
|---|---|---|
| Answer quality | `sample_task`, `multimodal_asr`, `multimodal_omni` | The reference answers the same requests. Both sets of answers are scored against the dataset's ground truth with the task's metric (letter accuracy, word error rate, word F1 or BLEU). The submission's mean must reach the task's fraction of the reference's. |
| Served tokens | `multitenant_slo`, `tab_completion` | The server decodes greedily and returns the token ids it emitted. The reference scores those exact tokens on the same prompts and counts each token that it ranks below its own first choice. The run fails when the count exceeds the task's calibrated limits. |
| Reported distributions | `deepresearch`, `agentic_coding`, `multimodal_tts` | The server samples, and at every position it reports the log-probability of the token it served and of the most probable alternatives. The reference computes its own distribution on the same prefix. The run fails when the reported distributions are farther from the reference's than the calibrated limit, or when the served tokens are not draws from the reported distributions. For `multimodal_tts`, the check also compares the audio with the reference codec's decoding of the same codes. |

Passing an answer-quality check means the answers are as good as the reference's, within the
task's margin. Passing a token or distribution check means the server computes the reference's
model, up to the numerical differences between faithful serving stacks. Quantized weights, a
changed sampling rule or a different model fail it.

## Calibrating the correctness check

The limits of the token and distribution checks come from a measurement, the task's
calibration. Trusted serving stacks (vLLM and SGLang) serve the task's workload, and the gate's
own reference scores each of them. `calibration.json` records how far a faithful stack sits from
the reference on this workload and this kind of GPU, and the gate reads its limits from that
file. Every task with a token or distribution check ships its calibration. `hls calibrate`
measures a new one for `multitenant_slo`, `tab_completion` and `deepresearch`. Answer-quality
checks need no calibration.

## Each request's latency, relative to its budget

Every request gets a normalized latency `r`: its latency divided by the budget the task sets for
it. An `r` of 1 is exactly on budget, and lower is faster. The budgets only set the scale: a
request over its budget still counts, and it earns less the further over it is.

| Replies | Tasks | `r` |
|---|---|---|
| Streamed tokens | `multitenant_slo`, `tab_completion`, `agentic_coding` | The larger of TTFT over its budget and mean TPOT over its budget |
| Whole reply | `multimodal_asr`, `multimodal_omni` | Time from request sent to reply complete, over the budget |
| Streamed audio | `multimodal_tts` | Time from request sent to the last audio frame, over the budget |
| Trajectory, one agent session of many requests | `deepresearch` | Time from the trajectory's first request sent to its last reply, over 1,800 s |

TTFT (time to first token) is the time from sending the request to the first chunk that carries
a token. Mean TPOT (time per output token) is the time from the first token to the last, divided
by the number of tokens after the first. A reply of one token is measured on TTFT alone. The
harness reads both times from its own clock as the stream arrives. In `agentic_coding` the TTFT
budget grows with the prompt: 1,400 ms plus 0.008 ms per prompt token.

Requests arrive on a schedule the server does not control, so a faster server gets no extra
requests to answer. Each task's arrival rates and budgets are in [tasks.md](tasks.md).

## The score formula

```
credit = ln(1 / r)                  for each request, or each trajectory
L      = mean credit over the run
score  = exp(L - B)
```

`B` is the baseline credit: the `L` that the task's baseline server earns on the same workload.
The baseline server is the production serving stack the task is scored against, vLLM or
vLLM-Omni, run with the flags the task declares. The score is therefore the geometric mean of
the per-request speedups over the baseline. Every request has the same weight, and halving every
latency doubles the score.

`sample_task` has no baseline. Its score is goodput: the number of correctly answered requests
per second of the timed workload.

## How the baseline credit B is measured

`hls baseline-credit` measures `B`. It serves the baseline, sends it the same workload at the
same arrival rates, and writes the result as `baseline.json`. A baseline credit belongs to one
workload, one set of arrival rates and budgets, one kind of GPU and one baseline server command.
A file that does not fit the task being graded is not used.

`hls run` and `hls grade` look for `baseline.json` and `calibration.json` in the store first
(`measured/<task id>/`), then among the files the task ships (`tasks/<directory>/defaults/`).
`hls run` measures whichever is missing before the agent starts building. Two scores of one task
are comparable when they used the same baseline credit, which `score-provenance.json` names.

## Why a run scores zero

A run that does not count scores 0, with `gate_passed` false. `zero_reasons` lists every reason
that applies:

| Reason | Meaning |
|---|---|
| `correctness` | The gate failed. `breakdown.verdict` holds each check's figures. |
| `failed_requests` | At least one request failed, so no check ran. `breakdown.verdict.failed_requests` counts the failures by their error. |
| `slo` | A request has no measurable latency: it never answered, or a streamed reply carried no token timing. |
| `invalid_artifact` | The build left nothing to grade: no commit on `main`, a repository over the limits the brief states, or a package that did not install. |
| `crashed` | The server exited before it was ready. |
| `not_ready` | The server ran but did not pass its readiness probes in time. |
| `contestant_fault` | A fault of the submission stopped a stage, and the run ended `failed`. `breakdown.contestant_fault` names the stage and the reason. |

A run zeroed for `correctness` still reports its latency, and `breakdown.upper_bound` is the
score it would have earned on latency alone.

## Reading the result files

[running.md](running.md#where-results-are-stored) says where a run's files are. `score.json` holds
the score:

| Field | Meaning |
|---|---|
| `value` | The score. |
| `gate_passed`, `zero_reasons` | Whether the run counts, and why not. |
| `achieved` | `exp(L)`: the run's speed against the budgets alone. For `sample_task`, requests per second. |
| `sol` | `exp(B)`: the baseline's speed against the same budgets. `value` is `achieved / sol`. For `sample_task`, 1. |
| `unit` | `slo_speedup` for a task scored against a baseline, `request` for goodput. |
| `breakdown.mean_log_credit`, `breakdown.baseline_log_credit` | `L` and `B`. |
| `breakdown.n_scored` | The number of requests or trajectories that `L` is a mean over. |
| `breakdown.slo` | Per tenant (a class of the workload's traffic with its own budgets): its budgets, tail latencies, mean credit, and `met_frac`, the share of requests within budget. |
| `breakdown.slo_attainment` | The share of requests that were within budget and correct, over all tenants. It is a diagnostic and does not enter the score. |
| `breakdown.verdict` | The gate's figures, one entry per check. |

`score-provenance.json` names the baseline credit the score used, under `baseline`: `log_credit`
(`B`), `blob_sha` (the git blob id of the `baseline.json` it came from), the `campaign_id` of
that measurement, and the `hardware` it ran on. For `sample_task`, `baseline` is null.

`verdict.json` is the gate's full output: `aggregate_passed`, `failure_mode`, each check's
diagnostics, and a pass or fail for each request.

`grading.json` is the record of one grading. Its `episode` holds the score and a summary of the
verdict. Its `provenance` holds the task as it was graded, the image digests and the harness
commit. Its `conditions` say how the grading ran, including which `baseline.json` and
`calibration.json` it used and whether they were the ones the task ships.
