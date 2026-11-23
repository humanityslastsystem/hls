# Running the benchmark

## Running a task

```sh
uv run hls run tasks.multitenant_slo.task --arm claude
```

`hls run` takes a task and an arm. A task is one benchmark problem: a model, the GPUs it runs on
and the workload its server must answer. You name a task by its Python module,
`tasks.<directory>.task`, and [tasks.md](tasks.md) lists the tasks that ship with HLS. An arm is
the agent that builds the server: an agent CLI, a model and the way the agent reaches that
model's API. You declare each arm as an `[arms.<name>]` table in `hls.toml`
([Choosing the agent and model](#choosing-the-agent-and-model-arms-and-routes) shows one).

HLS keeps its work in the store, the directory tree under `site.store_root` that holds plans,
stage outputs, records and the cache. An instance is one piece of submitted work, such as a run
or a grading, and its name is its path in the store, such as `runs/multitenant_slo/<id>`. An
allocation is one job on your cluster: a Slurm job, or on a Docker host a job of HLS's own
scheduler. A stage is one step of an instance, and it publishes its outputs once, when it
succeeds.

The command works in this order:

1. It checks the arm, the task and the checkout. If a check fails, it says why and submits
   nothing.
2. It prepares what the run is missing. For each step it prints one line saying what it runs and
   why, then follows the step to its end:
   - the task's model weights, dataset rows and tool bundles (`hls provision`);
   - the baseline credit and the calibration (`hls baseline-credit`, then `hls calibrate`), when
     neither the store nor the task's own files hold one that fits. Both are explained
     [below](#which-baseline-and-calibration-measurements-a-run-uses);
   - the reference's answers to the test prompts (`hls oracle fill`), for a task whose
     correctness check reads them from the store. The reference is a trusted implementation of
     the task's model, which the harness runs itself.
3. It prints the plan: the arm with its route and budget, the baseline credit and calibration it
   uses, the instance, and one line per allocation with its stages and resources.
4. It submits the run and follows it to its end.
5. It prints how the run ended and its score.

By default one allocation holds all of a run's stages, which run in this order:

| Stage | What it does |
|---|---|
| `provision` | Puts the task's weights, dataset rows and tool bundles in the cache, skipping what is already there |
| `images` | Builds the task's container images from their pinned recipes, skipping what is already there |
| `agent_build` | Starts the agent in its sealed sandbox ([tasks.md](tasks.md)), gives it the task's brief (the task description it works from) and its time budget, and collects the repository the agent committed |
| `source_scan` | Scans the agent's source for the secrets in your secrets file |
| `package` | Builds the agent's source into a submission image, the container image its server runs from. Its package installs go through the task's network policy |
| `deploy_drive` | Starts the submission's server on the task's GPUs and sends it the task's workload |
| `verify` | Checks the replies against the reference. This is the correctness check, which HLS calls the gate |
| `score` | Computes the score from the workload's timings and the baseline credit |
| `finalize` | Writes the run's records and scans every file of the instance for secrets |

The last four stages are the run's grading of the submission. `hls grade` runs a grading on its
own.

The command's output has these lines:

```text
instance runs/multitenant_slo/<id>
allocation 0: provision, images, agent_build, source_scan, package, deploy_drive, verify, score, finalize; ...
runs/multitenant_slo/<id>: succeeded: every stage published
runs/multitenant_slo/<id>: score <value> (<path of score.json in the store>)
```

[scoring.md](scoring.md) explains what the score means.

## Before a task's first run

Commit your local changes. `hls run` runs committed code only, and it refuses a checkout with
uncommitted changes. Git ignores `hls.toml`, so your settings file does not count as one.

The seven benchmark tasks need H200 GPUs. The sample task, `tasks.sample_task.task`, asks for one
H200. It also runs on an A100, H100, B200, L40 or L40S if you give the GPU kind after the task:

```sh
uv run hls run tasks.sample_task.task --arm claude --task.hardware.accelerator.kind=L40S
```

The run's plan records the GPU kind, so `hls resume` and `hls grade --rollout` need no flag.
Give the flag again on `hls grade` with `--package` or `--measured`.

Four tasks give the agent offline documentation of its toolchain: `multitenant_slo`,
`tab_completion`, `multimodal_tts` and `agentic_coding`. Assemble the documentation once per
checkout, into `tasks/assets/references`:

```sh
uv run python scripts/assemble_references.py
```

Without that folder, `hls run` refuses these tasks with a line that says
`copy in the reference documents the task pins`.

`agentic_coding` also needs a `[sandbox_stores]` table in `hls.toml`, which names the four
directories the task keeps beside the store. [tasks.md](tasks.md) lists them.

## Which baseline and calibration measurements a run uses

A task scored against a baseline needs a baseline credit: a measurement of how fast the task's
baseline server answers the task's workload. A run's score is its speedup over that baseline. A
task whose correctness check is calibrated needs a calibration: a measurement of how far trusted
serving stacks sit from the reference on the task's workload, which sets how far the submission
may sit from it. [scoring.md](scoring.md) defines both.

Each is kept as a file, `baseline.json` or `calibration.json`. For each, `hls run` uses the first
of these that fits the task:

1. the store's own file, `measured/<task id>/baseline.json` or `calibration.json`;
2. the file the task ships, `tasks/<directory>/defaults/baseline.json` or `calibration.json`;
3. a new measurement, which `hls run` makes and writes into the store.

A file fits when it was measured under the same conditions as this run: the same workload and
GPU kind, and for a baseline credit also the same load and the same baseline server. The command
prints which file it uses, and why it passed over any other. [adding-a-task.md](adding-a-task.md)
shows how to measure either one yourself, with more repetitions or other settings.

## Choosing the agent and model (arms and routes)

```toml
[routes.anthropic]
credential = "anthropic_key"
auth = "api_key"

[arms.claude]
harness = "claude-code"
model = "claude-opus-5"
context_window = 1000000
output_limit = 64000
effort = "max"
route = "anthropic"
```

An arm configures the agent that builds the server. `harness` is the agent CLI, `claude-code` or `codex`.
`model` is the model id as the provider names it. `context_window` and `output_limit` are in
tokens, and Claude Code takes a context window of 200000 or 1000000. `effort` is the reasoning
effort, one of `low`, `medium`, `high`, `xhigh` or `max`; without it, the arm takes the task's
setting. `route` names the route the agent uses.

A route says how the agent reaches its model's API. It holds the names of secrets in your
secrets file, not their values. `credential` names the secret that holds the API key or token.
If the agent goes through a relay or an API gateway, `base_url` names the secret that holds the
API's base URL. `auth` is `api_key` or `bearer`, and Codex takes `api_key` only. The agent gets
these secrets and no others.

Before it submits the run, `hls run` prints the arm as it resolved it. For the sample task:

```text
arm claude: claude-code, model claude-opus-5, context window 1,000,000 tokens, output limit 64,000 tokens, effort max
route anthropic: credential from secret anthropic_key, the harness's own provider, auth api_key
budget: told 1,800 s, held to 3,600 s
```

If an arm's table has changed since the task's last run with that arm, `hls run` refuses to run
it again until `--arm-change "<why>"` gives the reason. The run records the reason.

## The agent's time budget

Each task gives the agent two wall-clock limits: a budget the agent is told, and a later limit at
which the runner stops it. The seven benchmark tasks tell the agent 12 hours and stop it 30
minutes later. The sample task tells it 30 minutes and stops it after an hour. When its time
ends, HLS grades what the agent has committed to the branch `main` of its repository.

`--soft-sec` replaces the budget the agent is told, and `--hard-sec` the limit at which it is
stopped. They are given together, for one run:

```sh
uv run hls run tasks.multitenant_slo.task --arm claude --soft-sec 7200 --hard-sec 9000
```

## Changing settings for one command

A command that submits a job takes overrides after the task. `--site.<key>`, `--run.<key>` and
`--resources.<task id>.<stage>.<key>` replace a setting of `hls.toml` for that command:

```sh
uv run hls run tasks.multitenant_slo.task --arm claude --run.gpu_policy split
```

`--task.<path>` replaces a field of the task, as `--task.hardware.accelerator.kind` does for the
sample task's GPU kind. [commands.md](commands.md#overriding-settings-for-one-command) says which commands take which
overrides.

## Watching, resuming and cancelling a run

Once the command has printed `instance runs/...`, you can stop it and the run goes on. If you
stop it earlier, while one of its preparation steps runs, that step's job finishes but no run is
submitted; start `hls run` again.

These four commands take the instance:

```sh
uv run hls follow runs/multitenant_slo/<id>
uv run hls status runs/multitenant_slo/<id>
uv run hls resume runs/multitenant_slo/<id>
uv run hls cancel runs/multitenant_slo/<id>
```

| Command | What it does |
|---|---|
| `hls follow` | Prints the instance's events as they are written, until it ends |
| `hls status` | Shows the plan, the allocations and their jobs, and each stage's attempts |
| `hls resume` | Takes the instance over and follows it to its end, as the first command did |
| `hls cancel` | Cancels the instance's jobs |

The jobs already submitted keep running without the command. Run `hls resume` after the first
command has stopped. It retries a stage that an infrastructure fault ended, and it records and
prints how the run ended. Until a command records the end, `hls follow` keeps waiting for it.

## How a run ends

| Outcome | Meaning |
|---|---|
| `succeeded` | Every stage published its outputs. The run is scored, whatever the score. A submission that does not build is recorded as an invalid artifact and scores zero |
| `failed` | A fault of the submission itself ended a stage, such as a server that ran out of memory. The run scores zero and is not retried |
| `void` | An infrastructure fault that retries did not clear. The run says nothing about the submission; start it again |
| `cancelled` | `hls cancel` ended it |

HLS retries infrastructure faults itself, as far as `[run.retry.<stage>]` in `hls.toml` allows:
by default once for `agent_build` and twice for every other stage. A retried agent build starts
over with its whole time budget.

`hls run` and `hls grade` exit 0 when the instance was scored, a zero score included, and 1 when
it ended void or cancelled. [commands.md](commands.md#exit-statuses) lists every exit status.

## Building without grading, and grading a run again

```sh
uv run hls run tasks.multitenant_slo.task --arm claude --build-only
uv run hls grade tasks.multitenant_slo.task --rollout runs/multitenant_slo/<id>
```

`--build-only` ends the run once the submission image is built and recorded, and grades nothing.
`hls grade --rollout` grades the image that run packaged: it starts the server, sends it the
workload, and records a grading under the run, at
`runs/multitenant_slo/<id>/gradings/<grading id>`. It also grades a finished run again, the same
way. With `--artifact.rebuild`, it first packages the run's source again from the current
checkout.

## Grading your own server

You can grade a server that no agent built, such as one you wrote yourself. Commit the server's
source to the branch `main` of a git repository laid out as the task's brief describes. Each
task's brief is in `tests/instances/golden/<directory>.brief.md`. Then bundle the repository,
package the bundle into a submission image, and grade the package:

```sh
git bundle create /shared/bundles/server.bundle main
sha256sum /shared/bundles/server.bundle
uv run hls package tasks.multitenant_slo.task --bundle /shared/bundles/server.bundle --sha256 <hex>
uv run hls grade tasks.multitenant_slo.task --package warmers/multitenant_slo/<id>
```

`hls package` takes the bundle's absolute path and its sha256, as `sha256sum` prints it. It
prints the instance it built the image in, `warmers/<task id>/<id>`, which you then give to
`hls grade --package`. A bundle that does not build is recorded as an invalid artifact, and a
grading of it scores zero.

## Grading the baseline server

To grade the task's own baseline server, the serving stack its scores are compared with:

```sh
uv run hls grade tasks.multitenant_slo.task --measured baseline
```

## Where results are stored

Every instance has its own directory in the store:

| Instance | Path under `site.store_root` |
|---|---|
| A run | `runs/<task id>/<id>/` |
| A grading of a run | `runs/<task id>/<id>/gradings/<grading id>/` |
| A package from `hls package`, and a grading of it | `warmers/<task id>/<id>/`, `warmers/<task id>/<id>/gradings/<grading id>/` |
| A grading of the baseline server | `warmers/<task id>/measured-baseline/gradings/<grading id>/` |
| A baseline credit measurement | `credits/<task id>/<id>/` |
| A calibration | `calibrations/<task id>/<id>/` |

The task id can differ from the directory name: `tasks/multimodal_asr` has the id
`multimodal-asr`.

Inside an instance, each stage keeps its outputs under `stages/<stage>/attempts/<n>/outputs/`,
and `stages/<stage>/MANIFEST.json` names the attempt that counted. Start with these files:

| File | Stage | What it holds |
|---|---|---|
| `brief.md` | `agent_build` | The brief the agent was given |
| `transcript/` | `agent_build` | The agent's session |
| `source.bundle` | `agent_build` | The repository the agent committed, as a git bundle |
| `verdict.json` | `verify` | The gate's verdict |
| `score.json` | `score` | The score |
| `rollout.json`, `grading.json` | `finalize` | The records of the build and of its grading |

`runs/<task id>/index/` holds one row for each run and each grading of a run that ended
`succeeded` or `failed`. `warmers/<task id>/index/` holds the rows of gradings of a package or of
the baseline server. A run that ended `failed` has its records under `failed/` in its instance
directory. [scoring.md](scoring.md#reading-the-result-files) explains the fields of a record.

## Viewing results in a browser

```sh
uv run hls view
```

This serves a web viewer of the store's `runs/` on `127.0.0.1:8080` and prints its URL. The URL
carries a session token, which every request needs. `hls view <directory>` serves another
directory, such as the store's `warmers/`. On its first start, the command builds the viewer with
`bun`.

## Exporting a run

`hls export` copies a run out of the store, for example to share or archive it:

```sh
uv run hls export runs/multitenant_slo/<id> --to /shared/results/my-run
```

The export first scans every file of the run and its gradings for secrets. It then copies the
plan, each stage's counted attempt and the records into the new directory, with an `EXPORT.json`
that lists what it copied. The destination must not exist yet, and it must lie outside the store
and the checkout.

## When a command refuses to run

A command refuses what it cannot plan: it prints one line on stderr, exits with status 1 and
submits no instance. A problem with the arm is a usage error instead, with exit status 2. The
message names what to change:

| The message says | What to do |
|---|---|
| `has uncommitted changes` | Commit, or run from a clean checkout |
| `unknown setting <key>` | Fix the key in `hls.toml`; [configuration.md](configuration.md) lists every key |
| `copy in the reference documents` | Assemble the toolchain documentation, as in [Before a task's first run](#before-a-tasks-first-run) |
| `no arm named` | Give `--arm` the name of an `[arms.<name>]` table in `hls.toml` |
| `differs from its latest build` | The arm changed since the task's last run with it: add `--arm-change "<why>"` |
| `data is not all in the cache` | You ran `hls grade`, or `hls run --probe-only`, on a store where the task has not run. Run the `hls provision` command the message gives |
| `none fits the task` | `hls grade` found no baseline credit or calibration that fits the task. Run the command the message gives, or run the task once with `hls run` |
| `no such file; copy docs/examples/hls.toml` | Create `hls.toml`, or name your settings file with `--config` |
