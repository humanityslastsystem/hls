# Commands

Run every command from the checkout as `uv run hls <command>`. `uv run hls --help` lists the
commands, and `uv run hls <command> --help` prints all of a command's options.

This page uses these HLS words. [running.md](running.md) walks through a run from start to end.

| Word | Meaning |
|---|---|
| arm | The agent that builds a server: a harness (Claude Code or Codex), a model, and a route that names the credential and endpoint for that model. An `[arms.<name>]` table of `hls.toml` declares it |
| submission | The server that is graded: the one an agent committed, or one you wrote |
| reference | HLS's own trusted implementation of the task's model, which a submission's outputs are checked against |
| drive | One timed run of the task's workload against a server |
| gate | The correctness check a submission must pass to score above zero |
| baseline credit | The measured speed of the task's baseline server on its workload, which the task's scores are relative to |
| calibration | How far trusted serving engines stray from the reference on the task's workload. The gate takes its tolerances from it |
| store | The directory tree under `site.store_root` where HLS keeps plans, outputs, records and the cache |
| instance | One execution of a command that submits work, such as a run or a grading. It is named by its path in the store, such as `runs/multitenant_slo/<id>` |
| job | One Slurm job, or one job of HLS's own scheduler on a Docker host. HLS's output calls it an allocation |

The commands:

| Command | What it does | When you need it |
|---|---|---|
| `hls run` | Has an arm's agent build a submission, then packages and grades it | Every benchmark run |
| `hls grade` | Serves one submission, runs the task's workload against it, then checks and scores it | To grade a run's build again, or to grade your own server |
| `hls package` | Builds a server's source into a submission image | Before `hls grade --package` |
| `hls status` | Shows an instance's plan, lease, jobs, stages and attempts | To see where a run is |
| `hls follow` | Prints an instance's events as they are written, until it ends | To watch a run |
| `hls resume` | Takes an instance over and follows it to its end | After the command that started it has stopped |
| `hls cancel` | Cancels an instance's jobs | To stop a run |
| `hls view` | Serves a web viewer of the instances in a directory | To read results |
| `hls export` | Copies finished instances out of the store, once a scan finds no secret in them | To share or archive a run |
| `hls provision` | Downloads a task's model weights, dataset rows and tool bundles into the cache | To fill the cache ahead of a run |
| `hls images build` | Builds a task's container images into the cache | To fill the cache ahead of a run |
| `hls oracle fill` | Computes the reference's answers to a task's test prompts and saves them in the store | To compute them ahead of a grading |
| `hls baseline-credit` | Measures a task's baseline credit | For a new task, or a more careful measurement |
| `hls calibrate` | Measures a task's calibration | For a new task or new hardware |
| `hls capped-requests` | Lists the sessions in which an engine's reply reached the output token cap during a calibration | When writing a task |
| `hls build-pasta` | Builds `pasta`, the program the network seal runs | On a Slurm cluster whose nodes lack the `passt` package |
| `hls images lock-apt`, `lock-pip`, `lock-npm`, `base-digest` | Write the version pins that image recipes install from | When changing an image recipe |

`hls run` does the work of `hls provision`, `hls images build`, `hls oracle fill`,
`hls baseline-credit` and `hls calibrate` itself when a run needs it, so none of them has to run
first. The exception is a run with `--probe-only` or `--preflight-only`. It prepares nothing, so
the task's data must already be in the cache.

## Options that many commands share

A command that takes a task takes its Python module, `tasks.<directory>.task`, as its first
argument. `hls images build` takes the directory name alone, such as `multitenant_slo`.

| Option | Commands | Meaning |
|---|---|---|
| `--config PATH` | Every command that reads `hls.toml` | The settings file, `./hls.toml` by default |
| `--store PATH` | `status`, `follow`, `resume`, `cancel`, `capped-requests` | The store's root directory, `site.store_root` of the settings file by default |
| `--job-name NAME` | Every command that submits a job | The job's name on the cluster. The default is `hls-` followed by the command's name, as in `hls-run` and `hls-oracle-fill`; `images build` uses `hls-images` and `baseline-credit` uses `hls-credit` |

While `site.warmers_one_at_a_time` is on, as it is by default, `hls provision`, `hls images build`
and `hls package` run under `site.warmer_job_name` and ignore `--job-name`, so that only one of
them runs at a time.

### Overriding settings for one command

A command that submits a job takes overrides after its first argument, as `--<path> <value>` or
`--<path>=<value>`. An override changes a value for this one command and leaves `hls.toml` as it
is.

| Override | What it replaces |
|---|---|
| `--site.<key>`, `--run.<key>` | A key of the `[site]` or `[run]` table of `hls.toml` |
| `--resources.<task id>.<stage>.<key>` | One resource of one stage: `gpus`, `gpu_kind`, `cpus`, `memory_gib`, `disk_gib` or `time_sec` |
| `--task.<path>` | A field of the task's definition, such as `--task.hardware.accelerator.kind`. Only `hls run`, `hls grade`, `hls provision`, `hls oracle fill`, `hls baseline-credit` and `hls calibrate` take it |

```sh
uv run hls run tasks.multitenant_slo.task --arm claude --run.gpu_policy split
uv run hls grade tasks.multitenant_slo.task --rollout runs/multitenant_slo/<id> --resources.multitenant_slo.verify.memory_gib 256
```

List and table settings, such as `site.sbatch_extra`, can be set only in `hls.toml`.
[configuration.md](configuration.md) lists every setting.

An override of `site.store_root` holds for that command only. `hls status`, `hls resume` and
`hls cancel` read the store's path from `hls.toml`, so give them the same path with `--store`.

## Running a task and grading a submission

### `hls run TASK --arm ARM`

Has the arm's agent build a submission in a sandbox, then packages and grades it. The sandbox is
a container whose network reaches only the hosts the task allows. [running.md](running.md) walks
through a run.

| Option | Meaning |
|---|---|
| `--arm NAME` | Required. The arm whose agent builds: the name of an `[arms.<name>]` table of `hls.toml` |
| `--arm-change TEXT` | Why the arm (its table, or its route's table) differs from the task's latest build with this arm. HLS refuses to rerun a changed arm without it, and refuses it for an arm that did not change or has no earlier build |
| `--build-only` | Stop once the package is built and recorded, without grading it. Grade it later with `hls grade --rollout` |
| `--soft-sec N`, `--hard-sec N` | Replace the task's two time budgets for this run: the seconds the agent is told it has, and the seconds after which HLS stops it. Give both or neither |
| `--probe-only` | Start the sandbox, run its checks and stop. No agent runs and nothing is packaged |
| `--preflight-only` | Like `--probe-only`, and also check that the arm's agent starts and reaches its model |
| `--probe-oom` | With `--probe-only`, also run a program that uses up the sandbox's memory, and record what ended it |

`--build-only`, `--probe-only` and `--preflight-only` exclude each other.

### `hls grade TASK`

Serves one submission on the task's GPUs and drives it. HLS then checks the replies against the
reference (the gate), scores the drive and records the result as a grading. Name exactly one
submission to grade:

| Option | What it grades |
|---|---|
| `--rollout runs/<task id>/<id>` | The image that `hls run` packaged in that run. With `--artifact.rebuild`, HLS first packages the run's source again at the current checkout and grades that |
| `--package warmers/<task id>/<id>` | The image that `hls package` built |
| `--measured baseline` | The task's own baseline server |
| `--artifact.image sha256:<digest> --from PATH` | A submission image file, named by the sha256 of the file. `--built-arm NAME` names the arm that built it, and the final secret scan then treats that arm's route secrets as ones its agent was given |

`hls grade` prepares nothing. It needs the task's data in the cache, a baseline credit and a
calibration that fit the task (in the store or among the task's shipped defaults) and, for a task
whose gate reads them from the store, the reference's answers. When one is missing, it refuses and
names the command that makes it. One `hls run` of the task makes them all.

### `hls package TASK --bundle PATH --sha256 HEX`

Builds a server's source into the task's submission image, which `hls grade --package` then
grades. The source is a git bundle of a repository's `main` branch. When the source does not
build, `hls package` records that instead, and a grading of it scores zero. The command prints
the instance it built the image in, `warmers/<task id>/<id>`.

| Option | Meaning |
|---|---|
| `--bundle PATH` | Required. The bundle's absolute path. HLS copies the bundle into the store before it plans the job |
| `--sha256 HEX` | Required. The sha256 of the bundle file, as 64 lowercase hex digits. HLS refuses a bundle whose bytes have another digest |

## Watching, resuming and cancelling an instance

`hls status`, `hls follow`, `hls resume` and `hls cancel` each take the instance, such as
`runs/multitenant_slo/<id>`, and the `--config` and `--store` options.

```sh
uv run hls status runs/multitenant_slo/<id>
```

The command that starts an instance follows it to its end. While it runs, it holds a lease on the
instance: a record in the store that it keeps renewing, so that no second command drives the same
instance. Stopping the command leaves the instance's jobs running.

`hls resume` takes the instance over at once when the earlier command is gone, and exits 2 while
that command still runs on the same host. When it cannot tell, it waits for the earlier command's
lease to lapse (`run.lease_ttl_sec`, 120 seconds by default) and exits 2 if the lease is still
held then.

`hls follow` prints the instance's events until it ends. When no command holds the instance, it
says so, names the `hls resume` command that takes it over, and keeps waiting.

`hls cancel` cancels only the jobs the instance recorded, by their ids, and waits until the
cancellation is confirmed. It gives up after `run.lease_ttl_sec` plus two `run.poll_sec` (180
seconds by default) and exits 3.

## Viewing and exporting results

### `hls view [DIRECTORY]`

Serves a web viewer of the instances in a directory. Without a directory it serves the `runs/`
directory of the store that `hls.toml` names. It prints the viewer's URL, which carries a session
token that every request needs. It refuses a directory that is, or holds, the store's area of
reference answers.

| Option | Meaning |
|---|---|
| `--host ADDRESS` | The address to listen on, `127.0.0.1` by default |
| `--port N` | The port to listen on, 8080 by default. When it is busy, the viewer takes the next free port up to nine above it |
| `--no-build` | Do not build the viewer's web page when it has not been built yet |

### `hls export INSTANCE... --to PATH`

Copies finished instances out of the store once a scan of every file has found no secret.
Exporting a run, or a package that `hls package` built, also exports the gradings under it. Every
instance named must belong to the same task. The scan looks for the values of the secrets file, so the export needs at
least one secret there.

| Option | Meaning |
|---|---|
| `--to PATH` | Required. An absolute path that does not exist yet, outside the store and the checkout |
| `--oracle` | Also export the task's reference answers (`oracle/<task id>/`), which are scanned like every other file |

## Preparing a task ahead of a run

`hls run` prepares a task itself. These three commands do the same work as jobs of their own. Use
them to fill a new store before several runs start together. Use them also before `hls grade`,
`hls baseline-credit` or `hls calibrate` on a store where the task has never run: those three
commands need the task's data in the cache, and when it is missing they refuse and name the
`hls provision` command to run.

The cache is the part of the store that holds downloaded weights, dataset rows, tool bundles and
built images. `hls provision` also pulls the sandbox images a task declares, when its sandbox
store lacks them. On a Slurm cluster with `site.cpu_partition` set, `hls provision` and
`hls images build` run on that partition and need no GPU. `hls oracle fill` runs as one job on the
GPU partition.

```sh
uv run hls provision tasks.multitenant_slo.task
uv run hls images build multitenant_slo
uv run hls oracle fill tasks.multimodal_asr.task
```

| Command | Option | Meaning |
|---|---|---|
| `hls provision TASK` | `--rows-only` | Leave out the model's weights |
| `hls oracle fill TASK` | `--chunk N` | How many rows the reference answers per chunk, 96 by default. Each chunk's answers are saved as soon as it finishes |

## Measuring the baseline credit and the calibration

[adding-a-task.md](adding-a-task.md) says when a task needs each of these, and
[scoring.md](scoring.md) says how the score uses them.

### `hls baseline-credit TASK`

Serves the task's baseline server, drives it, and writes the baseline credit, `baseline.json`, into
the store, where runs of the task find it. The baseline credit measures the baseline server's speed
on the task's workload, and each score of the task compares a submission with it.

| Option | Meaning |
|---|---|
| `--drives N` | How many fresh drives to pool, 1 by default. With one drive, the record holds no spread between drives |
| `--fanout K` | How many jobs drive side by side, from 1 to `--drives`. 1 by default |

### `hls calibrate TASK`

Measures the task's calibration and writes it, `calibration.json`, into the store, where runs of
the task find it. Each trusted serving engine serves the task's workload one or more times, and the
reference scores each of these servings. The calibration records how far a faithful engine strays
from the reference on this workload and GPU, and the gate takes its ceilings from it.

| Option | Meaning |
|---|---|
| `--engines LIST` | Comma-separated engines to serve with, `vllm,sglang` by default |
| `--fanout K` | How many jobs take servings side by side, 4 by default. For a gate that checks served tokens, the last round can add up to K minus one servings per engine beyond the number needed |
| `--servings-per-engine N` | How many servings each engine takes, for a task whose gate checks reported distributions, 1 by default. Refused for a gate that checks served tokens: there, HLS adds servings until enough tokens have been observed |
| `--max-servings-per-engine N` | The most servings an engine may take, 24 by default. HLS refuses the calibration when an engine would need more |
| `--sessions-per-draw N` | How many sessions one serving drives, the whole test split by default. A subsample keeps each tenant's arrival rate and shortens the drive |

`uv run hls calibrate --help` explains the gates' tolerances: `--eps` (0.05 by default),
`--mu-per-mtok` (0.3), `--delta` (0.001) and `--eot-expected-max` (5.0).

### `hls capped-requests TASK --from calibrations/<task id>/<id>`

Lists, for each tenant (each traffic class of the task), the sessions in which an engine's reply
reached the output token cap during a calibration. For each tenant it also prints how many of its
sessions the list removes, and how many turns reached the cap. A task author uses the list
to remove those sessions from the task's data (`DataSource.removed_rows`). The command reads the
calibration's saved captures and runs nothing.

`--engine NAME` picks the engine whose replies count, `vllm` by default. `--write` hands the rows to
the task module's `record_removed_rows`, which adds them to the task's list; for a task module
without that function, `--write` is a usage error.

## Setting up a Slurm cluster

### `hls build-pasta DIRECTORY`

Builds the `pasta` release that HLS pins into a directory every compute node mounts, outside the
checkout, and prints the `site.pasta_path` line to add to `hls.toml`. The network seal, which
limits a container's network to the hosts its task allows, runs `pasta`. A cluster whose compute
nodes lack the `passt` package needs this build. Run it on a compute node, inside a Slurm
allocation, as `srun ... uv run hls build-pasta DIRECTORY`; it refuses to run on a login node.
`--cc` names the C compiler, gcc 13 or later (`cc` by default, or `$CC` when it is set).
[setup-slurm.md](setup-slurm.md) shows the whole procedure.

## Pinning the packages that images install

These four commands make the version pins that the image recipes install from. They run where you
start them and submit no job. The shared lock files are in `src/hls/recipes/locks/`.
[adding-a-task.md](adding-a-task.md) shows `lock-pip` for a task's own images.

| Command | What it does |
|---|---|
| `hls images lock-apt {ubuntu\|nvidia} NAME` | Pins each apt package that `NAME.in` lists to the highest version the repository carries (the pinned Ubuntu snapshot, or NVIDIA's repository), and writes `NAME.txt`. An NVIDIA lock also records each `.deb` file's sha256. With `--task TASK` it reads and writes `tasks/TASK/locks/`, from the Ubuntu snapshot |
| `hls images lock-pip {base\|layer} NAME` | Compiles `NAME.in` into the hashed pip lock `NAME.txt`. A base is compiled alone. A layer is compiled against its parent's pins and keeps only what its parent lacks |
| `hls images lock-npm NAME` | Writes the npm lock of the agent CLIs, such as `npm-agent.json` |
| `hls images base-digest REPO TAG` | Prints `REPO:TAG@<digest>` for a Docker Hub tag: the digest a recipe's base image is pinned by. It asks Docker Hub anonymously and writes nothing |

`hls images lock-pip` refuses a lock that names a package on the deny list, or one that needs a
source build it was not told about. Its options:

| Option | Meaning |
|---|---|
| `--parent NAME` | The lock of the image a layer builds on: the task's own lock of that name, else the shared one. The CUDA base has none |
| `--layer-owned PACKAGE` | For a base: a package it leaves out, for each layer to install at its own version. Repeat for each package |
| `--source-build PACKAGE` | For a layer: a package it builds from its source distribution, as the layer's `source_builds` in the image catalog declare. Repeat for each package |
| `--task TASK` | A layer that a task declares for itself: read and write `tasks/TASK/locks/` |
| `--server` | With `--task`, the task's server layer, which may hold a serving framework the deny list names |
| `--no-deps` | For a task's server layer: lock exactly the pins its `.in` file names, resolving no dependencies |

## Exit statuses

A command refuses what it cannot plan with one line on stderr and exit status 1, and submits no
instance. A mistake in the options themselves, such as two options that cannot go together, is a
usage error, with exit status 2.

| Command | Exit status |
|---|---|
| `hls run` | 0 when the instance was scored, a score of zero included; 1 otherwise, as when it ended void or cancelled. 2 for an arm that cannot be resolved, and for a rerun of a changed arm without `--arm-change`. When a preparation step that `hls run` started first does not succeed, that step's status |
| `hls grade` | 0 when the instance was scored, a score of zero included; 1 otherwise, as when it ended void or cancelled. 2 for a `--built-arm` that cannot be resolved |
| `hls provision`, `hls images build`, `hls package`, `hls oracle fill`, `hls baseline-credit`, `hls calibrate`, `hls export` | 0 when the instance succeeded; 1 otherwise |
| `hls resume` | The status of the command that started the instance; 2 when the store holds no such instance or another command still holds it |
| `hls status` | 0; 2 when the store holds no such instance |
| `hls follow` | 0 once the instance has ended; 2 when the store holds no such instance |
| `hls cancel` | 0 when the instance ended cancelled; 1 when it ended otherwise; 2 when the store holds no such instance; 3 when the cancellation was requested and not confirmed |

`hls resume`, `hls status`, `hls follow` and `hls cancel` also exit 1 when `hls.toml` is not valid,
or when this version of HLS cannot read the instance's stored plan. Every other command exits 0, or
1 after a line on stderr that says what it refused.
