# Configuration reference

HLS reads two files. `hls.toml` holds the settings: where jobs run, where results are kept and
which agents build. A command reads it from the directory it starts in, or from the path that
`--config` gives. The secrets file holds the credentials, and `site.secrets_path` names it.
[`examples/hls.toml`](examples/hls.toml) and [`examples/secrets.toml`](examples/secrets.toml) are
complete examples to copy from.

HLS checks the whole of `hls.toml` before it runs anything. It refuses an unknown key, a key in
the wrong table and a value of the wrong type, and names the key by its dotted path, as in
`unknown setting site.partiton`. Every path is absolute or starts with `~`. To change a key of
`[site]`, `[run]` or `[resources]` for one command, give it as an option, such as
`--site.partition gpu-debug` ([commands.md](commands.md#overriding-settings-for-one-command)).

This page uses these HLS words:

| Word | Meaning |
|---|---|
| site | The cluster or host that HLS runs jobs on |
| substrate | How the site runs jobs: Slurm with Apptainer, or one host's Docker |
| allocation | One job on the site |
| store | The directory tree under `site.store_root` where HLS keeps plans, stage outputs, records and the cache |
| instance | One thing HLS plans and runs, such as a run or a grading |
| stage | One step of an instance, such as `agent_build`, `package` or `verify` |
| submission | The server that is graded, usually the one an agent built |
| network seal | The part of HLS that limits a container's network to the hosts its task allows |

## Settings in hls.toml

`hls.toml` has these tables and no others.

| Table | What it holds |
|---|---|
| [`[site]`](#site-the-cluster-or-host-and-the-store) | The cluster or host, and where the store is. Required. |
| [`[site.docker]`](#sitedocker-docker-settings), [`[site.local]`](#sitelocal-the-docker-hosts-job-scheduler) | A Docker host. `[site.local]` is required when `site.substrate = "docker"`; every key of `[site.docker]` has a default. |
| [`[run]`](#run-scheduling-and-retries) | How HLS schedules instances and retries their stages. |
| [`[resources.<task id>.<stage>]`](#resourcestask-idstage-hardware-for-one-stage) | Changes to the hardware one stage of one task asks for. |
| [`[routes.<name>]`](#routesname-how-an-agent-reaches-its-model) | How an agent reaches its model. |
| [`[arms.<name>]`](#armsname-the-agents-that-build) | The agents that build submissions. |
| [`[sandbox_stores]`](#sandbox_stores-directories-for-tool-sandboxes) | Directories that a task's tool sandboxes start from. |

### `[site]`: the cluster or host, and the store

Keys for every site:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `substrate` | `"slurm"` or `"docker"` | `"slurm"` | What the site runs jobs on: a Slurm cluster with Apptainer, or one host's Docker. |
| `store_root` | path | required | The store's root directory. On Slurm, a path that the login and compute nodes share. |
| `node_scratch` | path | required | A local disk on the machine that runs a job. Each job works in `<node_scratch>/hls-<job id>` and removes it when it ends. |
| `secrets_path` | path | required | The [secrets file](#the-secrets-file). |
| `registry_username` | string | none | The Docker Hub user that image builds and sandbox image pulls log in as. Set it together with `REGISTRY_PASSWORD` grants to `stage.images` and `stage.sandbox_images` in the secrets file, or set none of them to pull anonymously. It is not a secret, and the secrets file must not hold it. |
| `egress_upstream_proxy` | string | none | An HTTP proxy through which the network seal reaches the internet, as `http://<host>:<port>` with no credentials, path or query. Unset, the seal connects directly. |
| `step_launch_allowance_sec` | integer | `150` | Seconds added to an allocation's time limit for each container its stages start, to cover the time each takes to start. |
| `nproc_max` | integer | none | The most processes and threads your user may have on a node while a submission's or an agent's code runs there: the served submission, a pip step of packaging, or a command in the agent's sandbox. Unset, each node gets 131,072, or 4,096 for each submission it can run at once when that is more. A value you set must allow 4,096 for each. |
| `nproc_accounting` | `"uid"` or `"user_namespace"` | none | How the node's kernel counts the processes of the agent's sandbox against their limit: `"uid"` on kernels before 5.14, `"user_namespace"` from 5.14 on. Unset, each build reads its node's kernel release. |
| `warmers_one_at_a_time` | boolean | `true` | Runs `hls provision`, `hls images build` and `hls package` one job at a time, so that they do not fill one node's scratch together. |
| `warmer_job_name` | string | `"hls-warmer"` | The job name those three commands share while `warmers_one_at_a_time` is true. |
| `sandbox_namespaces` | integer | none | A cap on how many namespaces of each kind the agent's sandbox may hold at once. At least 16. Unset, the sandbox gets its share of the node's limit. |
| `sandbox_inotify_instances` | integer | none | A cap on the sandbox's inotify instances. At least 2. Unset, the sandbox gets its share of the node's limit. |
| `sandbox_inotify_watches` | integer | none | A cap on the sandbox's inotify watches. At least 8,192. Unset, the sandbox gets its share of the node's limit. |
| `grading_node_namespaces` | integer | `1048576` | The smallest per-user limit on a kind of namespace (`user.max_user_namespaces` and the other `user.max_*_namespaces`) among the nodes that grade. On Slurm, a graded server is held to its share of this limit. |
| `grading_node_inotify_instances` | integer | `128` | The smallest `fs.inotify.max_user_instances` among the nodes that grade. |
| `grading_node_inotify_watches` | integer | `1048576` | The smallest `fs.inotify.max_user_watches` among the nodes that grade. |

Keys for a Slurm cluster only. A Docker site sets none of them.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `partition` | string | required | The partition that every GPU allocation is submitted to. |
| `account` | string | none | The Slurm account. Unset, Slurm uses your default account. |
| `qos` | string | none | The Slurm QOS. Unset, Slurm uses the account's default. With `qos_by_hours` set, it must be one of the QOS listed there. |
| `qos_by_hours` | table of QOS name to hours | `{}` | Each QOS a job may run under, with its time limit in hours, as in `{ normal = 24, long = 168 }`. HLS keeps an allocation under `qos` while it fits that limit, and otherwise uses the QOS with the shortest limit that fits. An allocation that fits no limit is split into a chain of shorter allocations; a single stage that fits none is refused. |
| `gpu_gres` | table of GPU kind to gres name | `H200 = "h200"`, `H100 = "h100"`, `B200 = "b200"`, `A100 = "a100"`, `L40 = "l40"`, `L40S = "l40s"` | The Slurm gres name of each GPU kind on this cluster. |
| `min_gpus_per_job` | integer | `0` | The fewest GPUs a job on `partition` may ask for, `0` when the partition accepts jobs without a GPU. An allocation that needs fewer GPUs asks for this many. |
| `cpu_partition` | string | none | The partition for jobs that need no GPU: `hls provision`, `hls images build`, `hls package` and `hls export`. Unset, they run on `partition`, with a warning. |
| `sbatch_extra` | list of strings | `[]` | Extra `sbatch` arguments for every allocation on `partition`, and for none on `cpu_partition`. Each is `--<name>=<value>`, with a name from `exclude`, `constraint`, `reservation`, `nodelist`, `nice`, `mail-type` and `mail-user`, and each name at most once. |
| `pasta_path` | path | none | The `pasta` binary that the network seal runs, as `hls build-pasta` prints it. Unset, the job's `PATH` must hold `pasta`. |
| `evaluation_socket_tasks` | boolean | `false` | `true` keeps a graded server's CPUs on one socket. A node that still spreads them over two NUMA nodes is refused, and the stage is retried. It needs nodes with one NUMA node per socket and at least twice a server's CPUs per GPU. |

### `[site.docker]`: Docker settings

Every key has a default, so a Docker site may leave this table out.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `socket` | path | `/var/run/docker.sock` | The Docker daemon's Unix socket. |
| `buildkit_image` | string | BuildKit v0.35.1, by digest | The BuildKit image that image builds run, as `<name>@sha256:<64 hex>` or an image ID `sha256:<64 hex>`. A tag alone is refused, since the builder runs privileged. |
| `owner_label` | string | none | A label, `key=value`, that HLS puts on every container that runs a stage's work, such as a label that a GPU manager on the host reads. |
| `gateway_memory_gib` | integer | `2` | GiB of an allocation's memory that the network seal of each sealed container gets. At least 1. |

### `[site.local]`: the Docker host's job scheduler

On a Docker host, HLS runs its own job scheduler in place of Slurm. This table is required when
`site.substrate = "docker"`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `state_dir` | path | required | Where the scheduler keeps its job records, reservations and logs: a directory of your own on a local disk, with mode 0700. HLS creates it. |
| `kept_memory_gib` | number | required | GiB of the host's memory that no allocation gets, left for the host's other processes and the Docker daemon. |

### `[run]`: scheduling and retries

Every key has a default, so the table is optional.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gpu_policy` | `"max"` or `"split"` | `"max"` | `"max"` runs a whole run in one allocation, sized to its largest stage. `"split"` starts a new allocation each time the GPU count changes between stages. |
| `heartbeat_sec` | integer | `30` | Seconds between the status writes of a running allocation. |
| `allocation_dead_after_sec` | integer | `600` | Seconds without a status write after which HLS cancels a running allocation and retries its stage. At least 3 times `heartbeat_sec`. |
| `poll_sec` | integer | `30` | Seconds between a command's reads of the store and the scheduler. |
| `lease_ttl_sec` | integer | `120` | How long a lease lasts without renewal, in seconds. The lease is the record in the store that keeps a second command from driving an instance. Once the command that follows an instance has not renewed its lease for this long, `hls resume` may take the instance over. |
| `allocation_overhead_sec` | integer | `900` | Seconds added to every allocation's time limit for its start and shutdown. At least 300. |
| `runner_memory_gib` | integer | `4` | GiB of each stage's memory kept for HLS's own processes in the job. |
| `nproc_reserve` | integer | `1024` | Processes and threads kept for HLS's own processes, for each submission a node can run at once. |
| `grade_after_sandbox_oom` | boolean | `true` | Whether a build whose sandbox ran out of memory is graded on the agent's last commit. `false` scores such a build zero. |
| `collect_allowance_sec` | integer | `3600` | Seconds a build keeps after the agent's hard time limit, to stop the sandbox and collect the repository and the transcript. At least 2,350. |
| `retry` | table | `{}` | One `[run.retry.<stage>]` table for each stage whose retries differ from the default. |

### `[run.retry.<stage>]`: retries of one stage

An infrastructure fault is one that the submission did not cause, such as a failed node. HLS
retries the stage that such a fault ended, as this table allows:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max` | integer | `1` for `agent_build` and `sandbox_images`, `2` for every other stage | Retries after the first attempt, `0` for none. |
| `enabled` | boolean | `true` | `false` turns retries off for the stage. |
| `backoff_sec` | integer | `0` | Seconds to wait before a retry is submitted. |

### `[resources.<task id>.<stage>]`: hardware for one stage

Each stage asks for the hardware its task declares. A `[resources]` table replaces the keys it sets
for one stage of one task, keeps the rest, and must set at least one key. It names the task by its
id, which can differ from its directory (`multimodal-asr` for `tasks/multimodal_asr`), and the
stage by its name in the plan, as `hls status` shows it.

```toml
[resources.multitenant_slo.verify]
memory_gib = 256
```

| Key | Type | Meaning |
|---|---|---|
| `gpus` | integer | GPUs. `0` also clears `gpu_kind`. |
| `gpu_kind` | string | One of `H200`, `H100`, `B200`, `A100`, `L40`, `L40S`. |
| `cpus` | integer | CPU cores. |
| `memory_gib` | integer | Memory limit in GiB. |
| `disk_gib` | integer | Free GiB the stage needs under `site.node_scratch`. |
| `time_sec` | integer | Time limit in seconds. |

### `[routes.<name>]`: how an agent reaches its model

A route names secrets of the secrets file and holds no value itself.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `credential` | secret name | required | The secret that holds the API key or token. |
| `base_url` | secret name | none | The secret that holds the API's base URL, an `https` URL, for a relay or gateway. Unset, Claude Code reaches `api.anthropic.com` and Codex reaches `api.openai.com`. |
| `auth` | `"api_key"` or `"bearer"` | `"api_key"` | How the agent presents the credential: as an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) or as a bearer token (`ANTHROPIC_AUTH_TOKEN`). Codex takes `"api_key"` only. |

### `[arms.<name>]`: the agents that build

An arm is an agent that builds a submission: a harness, a model and a route.
`hls run --arm <name>` picks one.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `harness` | `"claude-code"` or `"codex"` | required | The agent's command-line tool. |
| `model` | string | required | The model id as the provider names it, with no window suffix, brackets or spaces. |
| `route` | route name | required | The `[routes.<name>]` table through which the agent reaches its model. |
| `context_window` | integer | required | The context window in tokens. Claude Code takes `200000` or `1000000`. |
| `output_limit` | integer | required | The most output tokens of one model response. |
| `effort` | string | none | The reasoning effort: `low`, `medium`, `high`, `xhigh` or `max`. Unset, the arm uses the task's setting. |

### `[sandbox_stores]`: directories for tool sandboxes

Some tasks start tool sandboxes from files on the site, such as container images. This table gives
each sandbox store that a task names a directory of yours, which `hls provision` creates and
fills. Only a Slurm site runs such tasks, since their sandboxes run under Apptainer on the node.

```toml
[sandbox_stores]
agentic-coding-images = "/shared/agentic-coding/images"
```

| Key | Type | Meaning |
|---|---|---|
| `<store name>` | path | The sandbox store's directory. It must not be, hold or lie inside `site.store_root`, the directory of the secrets file, the HLS checkout or another sandbox store. A name starts with a lowercase letter and holds lowercase letters, digits and hyphens. |

[tasks.md](tasks.md) lists the sandbox stores each task needs.

## The secrets file

The secrets file holds every credential HLS uses, and HLS reads none from the environment. It is a
regular file that you own, outside the HLS checkout and the store, and group and others have no
access to it (mode 0600). On Slurm the compute nodes read it too, so it sits on a filesystem they
share with the login node.

```toml
[secrets]
hf_token = "hf_xxxxxxxxxxxxxxxxxxxx"
anthropic_key = "sk-ant-xxxxxxxxxxxxxxxx"

[grants]
"stage.provision".HF_TOKEN = "hf_token"
```

| Table | Entry | Meaning |
|---|---|---|
| `[secrets]` | `<name> = "<value>"` | One secret. A name is lowercase letters, digits and `_`, starting with a letter. A value is at least 8 printable ASCII characters without `"`, `\`, `$` or a backtick. No value may contain another secret's value, unless the two values are equal. |
| `[grants]` | `"stage.<stage>".<VARIABLE> = "<secret name>"` | Gives one stage one secret, as an environment variable. |

HLS uses these grants:

| Grant | Secret |
|---|---|
| `"stage.provision".HF_TOKEN` | A Hugging Face token, for gated models and datasets. |
| `"stage.images".REGISTRY_PASSWORD` | A Docker Hub token for image builds, used with `site.registry_username`. |
| `"stage.sandbox_images".REGISTRY_PASSWORD` | A Docker Hub token for the sandbox images that `hls provision` pulls, used with `site.registry_username`. |

An agent gets the secrets that its arm's route names and no others, so the file holds no grant for
an agent. HLS replaces every value of the file with `<redacted:NAME>` in the logs and records it
writes.
