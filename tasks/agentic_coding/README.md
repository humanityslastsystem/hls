# agentic_coding: serving a coding agent

In this task the agent builds an inference server for `zai-org/GLM-5.3-Flash` on one node of
eight H200s, at the checkpoint's full context window of 1,048,576 tokens. The server's traffic
is a coding agent that runs on the served model and works real tasks in a real shell. HLS is
the client. It runs a mini-SWE-agent loop over Terminal-Bench 4.0 tasks, sends each turn to the
server, runs the commands in the reply, and appends their output to the conversation. One
session is one trajectory: one task statement, then as many turns as the model takes. The prompt
starts long and grows with every command's output.

The correctness check, called the gate, tests whether the served tokens come from the model's
own distribution. The score is the speedup over the baseline serving stack, vLLM, in per-request
latency measured against a budget ([scoring.md](../../docs/scoring.md)). HLS does not measure
whether a task gets solved.

## The workload

A graded drive, the timed run of the workload against the server, offers 30 test sessions. They
arrive as a Poisson process, at a rate set so that all of them arrive within the first 15
minutes. At most 24 requests are in flight at once, and a session runs at most 250 turns. HLS
issues requests for 45 minutes. At 55 minutes it cuts every open stream and scores it on
what it streamed. A request whose stream delivers tokens and then goes 120 seconds without
another fails.

Three tasks start from a saved conversation. The three conversations were saved when their
prompts passed 200,000, 400,000 and 600,000 tokens. Each of the three tasks runs as two
sessions, and each of those sessions runs at most 10 further turns. Every other session starts
at turn 0.

## What a request looks like

Each turn is a streaming `POST /v1/chat/completions` carrying the whole conversation so far. It
sends the `bash` tool definition in `tools` with `tool_choice: "none"`, so the definition is
rendered into the prompt and the model's call stays in `content`. It asks for `logprobs` with
the top 21 alternatives. The server samples at temperature 1 from the full vocabulary. At every
position it reports the log-probability of the token it served and of its 21 most probable
alternatives, each with an integer token id.

A turn's first request sets `max_tokens` to 32,768. If the reply reaches that cap, HLS closes
the thinking block with `</think>` and sends a second request that continues the reply. The
second request sets `max_tokens` to 16,384, `continue_final_message: true` and
`add_generation_prompt: false`, and its final assistant message is the reply so far. Both
requests are scored and verified.

## How HLS runs the model's commands

HLS reads the call from the reply's text, in the syntax the checkpoint's chat template
documents:

```
<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>
```

It never reads a `tool_calls` field from the server. Before parsing, it decodes the reply's own
token ids and refuses a reply whose text differs from them
(`framework.check_text_matches_tokens`). A parser in the server therefore cannot change the
trajectory.

Each session gets its own container, started from its Terminal-Bench task's own image. Every
call in a reply runs in that container as one bash command in a fresh subshell, through the exec
server of harbor (the `harbor/` submodule). A command may run for 300 seconds. Its exit code and
output go back to the model as a JSON observation. When the output is 32,000 characters or
longer, the observation keeps its first and last 16,000.

A trajectory ends at the first of these:

- a command prints `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as its first line and exits 0;
- the trajectory reaches its turn limit;
- three replies in a row yield no command;
- a reply has no tokens;
- the next prompt would not fit the context window;
- the drive's wall clock runs out.

HLS replaces a container that stops answering, up to five times. After that it drops the
session. The turns the session served stay in the trace, the record of the drive's requests and
replies, and the trace's `END.json` records the drop. The drive goes on with the other sessions,
and the run is scored on what was served.

Every container gets 4 CPUs and 16 GiB of memory, whatever its task's `task.toml` declares. Its
writes go to a directory under the overlay store, and the disk behind that store needs 30 GiB
free per session in flight.

The loop is upstream mini-SWE-agent, vendored in `minisweagent/` with its license. That
package's `__init__.py` names the upstream commit and lists every change.

## The correctness check and the score

The gate is `ReportFaithfulnessVerifier`. Its reference is the checkpoint served by vLLM at the
commit `task.VLLM_COMMIT` names, at tensor parallel 8. The reference is teacher-forced on the
served tokens: it reads them as input and computes its own distribution at each position. Five
checks compare the server's tokens and reports with the reference:

- the mean divergence of the reports from the reference, against a calibrated ceiling;
- whether the served tokens' ranks under their reports are uniform, as true draws would be;
- the served tokens' log score;
- the count of replies that end where the reference rarely ends one;
- the fraction of served tokens outside the reported alternatives, against a calibrated ceiling.

The ceilings come from `defaults/calibration.json`, measured with SGLang as the trusted serving
stack. Any failed request, missing report or failed check scores the run zero.

Every request is scored on its own:

    budget = 1,400 ms + 0.0080 ms x prompt_tokens
    r      = max( TTFT / budget , mean TPOT / 20 ms )
    score  = exp( mean ln(1/r) over all requests - B )

TTFT is the time to the first token and TPOT the time per output token after it. B is the
baseline credit: the mean credit the baseline stack earned on this workload
(`defaults/baseline.json`). The time a command takes falls between requests, so it is not
charged to the server.

`metric.py` reports the shape of each trajectory and never a score: turns, commands, why each
trajectory stopped, and replies that held something shaped like a call that the parser could not
read.

## What a site needs for this task

### A Slurm site with Apptainer on its nodes

The trajectories' sandboxes are Apptainer containers that the driver, the HLS process that sends
the workload, starts on the node. The driver also needs `/dev/fuse` to mount the images.
On a Docker site, planning refuses the task in one line.

### The Terminal-Bench release

The corpus is `harbor-framework/terminal-bench` at tag `v4.0.0`, commit
`452bf305c6daa62fc59061d22133a7cbc7c1572e` (`datasets.RELEASE_URL`, `datasets.RELEASE_COMMIT`).
Provisioning fetches that commit itself, and refuses one whose task definitions do not hash to
the pinned digest. Of its 66 tasks, 8 have no image and 3 declare a GPU, so the corpus holds 55.

### Four sandbox stores

The task keeps four directories outside the store, called sandbox stores. The task names each
store and the path it is bound at inside the driver's container (`HostPath` in
`task.TOOL_RUNNER`). The site's `[sandbox_stores]` table in `hls.toml` gives the directory
behind each name. `hls run`, `hls grade` and `hls baseline-credit` refuse a plan whose stores
lack what the corpus needs.

`agentic-coding-images` holds the sandbox images: one Apptainer `.sif` per task, named
`<task_name>.sif`. Each is the environment image the release published for the task, pulled by
the digest of its manifest (`release_images.json`) and converted with `apptainer pull`.
`hls provision tasks.agentic_coding.task` pulls the ones the store lacks, in a stage named
`sandbox_images` that runs after `provision`. Beside each file it writes a record,
`<task_name>.json`, with its source and the file's sha256. The node that pulls them needs
Apptainer and a route to the registry. Nothing is built and no Docker daemon is used.

`agentic-coding-prefixes` holds the saved conversations. For each of the three tasks that start
from one, it holds `<task_name>.prefix.json` and the directory `<task_name>.state` beside it:
the conversation and the files the container had written when it was saved. `preroll.Preroll`
made them by driving the loop from turn 0 against the baseline. The repository keeps them in
`prefixes/`, each state as the archive `prefix.to_repository` writes (`<task_name>.state.tar`,
its overlay whiteouts as empty files), and the corpus pins each prefix's digest. Provisioning
extracts them into the store and makes the whiteouts and opaque marks again from the prefix's
manifest. It refuses a file that does not hash to its pin, and a state that is not what its
prefix records. A session also refuses a prefix that does not match. Placing a state that holds
whiteouts needs Linux 5.8 or later on the provisioning node, since from 5.8 any user can make
one.

`agentic-coding-overlays` is an empty writable directory, under which each session's container
keeps its writes. Provisioning makes it.

`agentic-coding-runtime` holds the Python runtime for harbor's exec server: a standalone
CPython 3.12.13 with fastapi 0.136.3 and uvicorn 0.48.0, for the images that cannot run the
server themselves. It is mounted read-only in every container at `/opt/harbor-server`.
Provisioning stages it in the store with uv (`hls.provision.sandbox_runtime`).

### The task's container images

The task's own images are declared in `task.py` (`IMAGES`). `hls images build agentic_coding`
builds them from the locks in `locks/`; otherwise the first run that needs them builds them.
The images are:

- the sandbox, on a layer of torch 2.11 and transformers 5.17.0;
- the baseline, the vendor's vLLM image by digest with `vllm_server.py` as `/app/server.py`;
- the reference, the same vendor image with Apptainer and harbor's packages for the driver.

## Running the task

Provision the task once, then run it. A run uses `defaults/baseline.json` for the baseline
credit when the store holds none that fits, so measuring a new one is optional.

```bash
# The corpus and the checkpoint into the site's cache, and the sandbox stores
uv run hls provision tasks.agentic_coding.task

# A new baseline credit B; a run uses defaults/baseline.json when the store holds none that fits
uv run hls baseline-credit tasks.agentic_coding.task

# A build by one arm's agent, then its grading
uv run hls run tasks.agentic_coding.task --arm <name>
```

Run the task's own tests with `uv run pytest tasks/agentic_coding/tests`.
