# Set up a Docker host

This page prepares one Linux machine with NVIDIA GPUs for HLS. HLS runs every job as containers on
the host's Docker. A scheduler inside HLS gives each job its own GPUs, CPU cores and memory, and
holds a job back until they are free. The host is ready when a run of the sample task ends with a
score.

In HLS's settings, the host is the site: the `[site]` table of `hls.toml` describes it. In every
command below, replace `/data/hls` with a directory of yours on a local disk.

The setup takes five steps, in this order:

1. [Check that the host meets the requirements](#check-that-the-host-meets-the-requirements).
2. [Install HLS](#install-hls).
3. [Write hls.toml](#write-hlstoml).
4. [Create the secrets file](#create-the-secrets-file).
5. [Run the sample task](#run-the-sample-task-to-test-the-host).

If a step stops with an error, [Fixing setup errors](#fixing-setup-errors) gives the fix for each
message.

## Check that the host meets the requirements

The host needs:

- Docker Engine 25.0 or later, with a socket your user can write to, so that `docker info` works
  without `sudo`.
- The Docker CLI with its buildx plugin.
- cgroup v2 at `/sys/fs/cgroup` with the `memory`, `pids` and `cpuset` controllers, and a Docker
  daemon whose cgroup driver is `systemd` or `cgroupfs`.
- NVIDIA GPUs of a supported kind (H200, H100, B200, A100, L40 or L40S), with the NVIDIA driver
  and `nvidia-smi`. [tasks.md](tasks.md) gives the kind and count each task asks for. HLS does not
  use the NVIDIA Container Toolkit: it gives a container its GPUs' device files and binds the
  driver's libraries read-only, finding them with `ldconfig -p`.
- Memory for the first run: 68 GiB plus the memory you keep back for the host. You set that amount
  in `site.local.kept_memory_gib`, which has no default. With the 16 GiB this page sets, the host
  needs 84 GiB.
- Unprivileged user, mount and PID namespaces, which the agent's sandbox uses. The settings
  `user.max_user_namespaces`, `user.max_mnt_namespaces` and `user.max_pid_namespaces` must be
  above 0. On kernels that have them, `kernel.unprivileged_userns_clone` must be 1 and
  `kernel.apparmor_restrict_unprivileged_userns` must be 0. Ubuntu 24.04 ships with the last one
  at 1.
- Outbound HTTPS, for image builds, model and dataset downloads and the agent's calls to its
  model.
- `git` and [`uv`](https://docs.astral.sh/uv/). `uv` runs HLS on Python 3.12 and downloads it when
  the host has none.

Run HLS as an ordinary user. The containers that run agents and submissions run as that user, and
HLS refuses to start them as root.

Run these checks:

```sh
docker version --format '{{.Server.Version}}'
docker buildx version
docker info --format '{{.CgroupDriver}} {{.CgroupVersion}}'
cat /sys/fs/cgroup/cgroup.controllers
nvidia-smi --query-gpu=name,uuid --format=csv
/sbin/ldconfig -p | grep -E 'libcuda\.so\.1|libnvidia-ml\.so\.1'
sysctl user.max_user_namespaces user.max_mnt_namespaces user.max_pid_namespaces
sysctl kernel.apparmor_restrict_unprivileged_userns
uv --version
```

The checks pass when:

- `docker version` prints 25.0 or later.
- `docker buildx version` prints a version.
- `docker info` prints `systemd 2` or `cgroupfs 2`.
- `cgroup.controllers` lists `cpuset`, `memory` and `pids`.
- `nvidia-smi` prints the name and UUID of each GPU.
- `ldconfig -p` lists `libcuda.so.1` and `libnvidia-ml.so.1`.
- The first `sysctl` prints a value above 0 for each setting.
- The second `sysctl` prints 0, or reports that the kernel has no such setting.
- `uv --version` prints a version.

To allow the namespaces on Ubuntu 24.04, set the AppArmor setting to 0:

```sh
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

To keep a setting across reboots, also write it to a file under `/etc/sysctl.d`.

## Install HLS

```sh
git clone --recurse-submodules https://github.com/humanityslastsystem/hls /data/hls/hls
cd /data/hls/hls
uv sync
```

When `uv sync` finishes, `uv run hls --help` lists the HLS commands.

## Write hls.toml

Create `hls.toml` in the root of the checkout. HLS reads it from the directory you start a command
in, so run every `hls` command from there. Git ignores the file.

```toml
[site]
substrate = "docker"
store_root = "/data/hls/store"
node_scratch = "/data/hls/scratch"
secrets_path = "~/.config/hls/secrets.toml"

[site.local]
state_dir = "/data/hls/state"
kept_memory_gib = 16

[routes.anthropic]
credential = "anthropic_key"
auth = "api_key"

[arms.claude]
harness = "claude-code"
model = "claude-opus-5"
context_window = 1000000
output_limit = 64000
route = "anthropic"
```

The `[site]` and `[site.local]` tables describe the host:

| Key | What to put there |
|---|---|
| `substrate` | `"docker"`. The substrate is what HLS runs jobs on: either Slurm with Apptainer (the default) or one host's Docker. A Docker host sets no Slurm key, such as `partition`. |
| `store_root` | The directory of the store. [Where to put the store](#where-to-put-the-store) explains what it holds. |
| `node_scratch` | A directory on a local disk. Each job works in `<node_scratch>/hls-<job id>` and removes it when it ends. Keep the path short: the agent's sandbox puts Unix sockets under it, and a socket path holds at most 107 bytes. |
| `secrets_path` | The path of the [secrets file](#create-the-secrets-file), which you create in the next step. |
| `state_dir` | Where the scheduler keeps its job records and logs: a directory of your own, mode 0700. HLS creates it. |
| `kept_memory_gib` | GiB of the host's memory that no job gets, left for the host's other processes and the Docker daemon. On a host that runs other work, add what that work uses ([how jobs share the host](#how-jobs-share-the-hosts-gpus-cpus-and-memory)). |

Docker mounts `store_root` and `node_scratch` into containers at the same paths, so each must be a
real path with no symbolic link on the way. `realpath /data/hls/store` must print
`/data/hls/store`.

You can leave out the `[site.docker]` table, since each of its keys has a default. HLS then talks
to the daemon on `/var/run/docker.sock` and builds images with BuildKit v0.35.1, pinned by its
digest. Add the table to name another socket or another BuildKit image, or to set an
`owner_label` for a host whose GPU manager tells containers apart by a label.
[configuration.md](configuration.md#sitedocker-docker-settings) lists its keys.

The other two tables say which agent builds the server. An arm is an agent that builds a
submission, the serving system HLS then grades. An arm names a harness (the agent's command-line
tool, here Claude Code), a model and a route. A route says how the agent reaches its model's API:
it names the secrets that hold the API's credential. [running.md](running.md#choosing-the-agent-and-model-arms-and-routes)
covers both.

[configuration.md](configuration.md) lists every key of `hls.toml`.

### Where to put the store

The store is the directory tree under `store_root` where HLS keeps plans, stage outputs, records
and the cache. A stage is one step of a run, such as downloading the model, building the images
or grading the server. HLS creates the directory on first use.

An instance is one execution of an HLS command, such as one run of a task. Each instance has a
directory of its own in the store, such as `runs/sample_task/<id>` for a run of the sample task.
The cache, under `cache/`,
holds every container image, model and dataset the tasks use. HLS fills each cache entry once and
shares it with every later run, so put the store on a disk with room for all of them.

## Create the secrets file

```sh
mkdir -p ~/.config/hls
touch ~/.config/hls/secrets.toml
chmod 600 ~/.config/hls/secrets.toml
```

Put your credentials in it:

```toml
[secrets]
anthropic_key = "sk-ant-xxxxxxxxxxxxxxxx"
hf_token = "hf_xxxxxxxxxxxxxxxxxxxx"

[grants]
"stage.provision".HF_TOKEN = "hf_token"
```

`anthropic_key` is the secret that the route in `hls.toml` names. The grant gives your Hugging
Face token to `provision`, the stage that downloads models and datasets.

Only you may read the file (mode 0600), and it lies outside the checkout and the store.
[configuration.md](configuration.md#the-secrets-file) describes the file.

## Run the sample task to test the host

```sh
uv run hls run tasks.sample_task.task --arm claude
```

The sample task is the first check of a new host. The agent builds a server for a 0.5B model, and
the run's jobs ask for at most one H200, 8 CPUs and 68 GiB of memory, with 225 GiB free under
`node_scratch`. If your host has another GPU kind, name it: one of A100, H100, B200, L40 and L40S.

```sh
uv run hls run tasks.sample_task.task --arm claude --task.hardware.accelerator.kind=L40S
```

On a new store, the command first builds the step image, the image that HLS's own containers run
in, and keeps it in the store's cache. The run then uses every part of the host that a benchmark
run uses. It downloads the model and its data into the cache, builds the container images, and
starts the agent in its sealed sandbox, a container whose network reaches only the hosts the task
allows. It then packages what the agent committed, serves it and grades it. For each job, the
scheduler reserves GPUs, CPUs and memory, and HLS checks the reservation against the host before
the job's first stage.

The command prints each instance it starts and ends with the run's score:

```
instance runs/sample_task/<id>
...
runs/sample_task/<id>: succeeded: every stage published
runs/sample_task/<id>: score <value> (<score file in the store>)
```

The host is set up when the command exits 0 after a score line like the last one.
[running.md](running.md) explains the run and its results.

A Docker host runs `sample_task`, `multitenant_slo`, `tab_completion`, `multimodal_asr`,
`multimodal_tts`, `multimodal_omni` and `deepresearch`. `agentic_coding` needs a Slurm cluster
([tasks.md](tasks.md)).

## How jobs share the host's GPUs, CPUs and memory

HLS lists the host's GPUs with `nvidia-smi` and reads each one's kind from its product name, so a
GPU named `NVIDIA H200` is an `H200`. A job asks for a number of GPUs of one kind, a number of CPUs
and an amount of memory. The scheduler gives it:

- GPUs that no other job holds and no other process computes on.
- Whole CPU cores, on as few NUMA nodes as can hold them.
- Memory out of the host's total less `kept_memory_gib`.

A job that does not fit waits, and `hls status` says what it waits for:

```
waiting for GPUs: 2 of the host's 8 H200 GPUs are free and it asks for 4; 6 hold a process hls did not start
```

Inside a job, HLS holds every container to its own CPUs, to its memory with no swap and to a limit
on its processes. Each container sees only the GPUs it was given. The scheduler counts the
CPUs and memory of HLS jobs only, so on a host that runs other work, raise `kept_memory_gib` by
what that work uses.

## What HLS leaves on the host

| What | Name | What happens to it |
|---|---|---|
| Containers | Labelled `hls.job=<job id>` | HLS removes each when it ends. `hls cancel` removes those of a running job. |
| Step image | `hls/step:<hex>` | Stays in the daemon. Its archive is in the store's cache. |
| BuildKit image | `site.docker.buildkit_image`, BuildKit v0.35.1 unless you set it | Stays in the daemon. HLS removes the builders and their volumes with their job. |
| cgroups | `hls.slice` with the systemd driver, `/hls` with cgroupfs | Stay, and HLS reuses them. |
| Scheduler state | `site.local.state_dir` | Stays. |

HLS creates no Docker network. `docker ps --filter label=hls.job` lists the containers of running
jobs.

## Fixing setup errors

A command that refuses before it submits anything prints the reason and stops. A refusal after
that ends the instance as `void` (stopped by an infrastructure fault, with no score). The last line
the command prints carries the reason.

`uv run hls status <instance>` shows the reason again, for the instance the command named, such as
`runs/sample_task/<id>`. A job's own output is
`<store_root>/<instance>/allocations/<k>/submissions/<n>/job-<job id>.out`. HLS plans each job as
an allocation, a group of stages: `<k>` numbers the instance's allocations from 0, and `<n>`
numbers the times HLS submitted that allocation to the scheduler, from 1.

| The message says | Fix |
|---|---|
| `are Slurm settings, which a site whose site.substrate is docker does not set` | Delete the keys it names from `[site]`. |
| `a site whose site.substrate is docker needs site.local` | Add the `[site.local]` table. |
| `site.docker.buildkit_image: String should match pattern` | Name the image by its digest, as `<name>@sha256:<digest>`, or delete the key to use the default. A tag alone is refused. |
| `the secrets file has mode <mode>; group and others must have no access (chmod 600)` | Run `chmod 600` on the file. |
| `install the Docker CLI with buildx` | Install the Docker CLI and its buildx plugin (`docker-buildx-plugin` in Docker's packages). |
| `this host refuses an unprivileged user the user, mount and PID namespaces`, then each setting and the value to set | Set each with `sudo sysctl -w <setting>=<value>`, and in a file under `/etc/sysctl.d` to keep it across reboots. |
| `the host <name> cannot hold the allocation: it asks for <n> <kind> GPUs and the host has <m>`, or the same for CPUs | The host has fewer GPUs of that kind, or fewer CPUs, than the task asks for. Run the sample task with the GPU kind the host has. [tasks.md](tasks.md) gives each task's hardware. |
| `it asks for <n> GiB and the host gives allocations <m> GiB` | The host's memory less `kept_memory_gib` is less than the task asks for. Lower `kept_memory_gib`. |
| `must be a directory of this user's with mode 0700` | Run `chmod 700` on the state directory, or name another. |
| `the Docker daemon on <socket> cannot be used` | Start the daemon, add your user to the `docker` group, and check that the engine is 25.0 or later. |
| `ldconfig -p lists no <library>`, or another line about the running NVIDIA driver | The driver's libraries are not where the loader finds them, or two driver versions are installed. Reinstall the driver so that `ldconfig -p` lists `libcuda.so.1` and `libnvidia-ml.so.1` of the running version, then run `sudo ldconfig`. |
| `is not a cgroup v2 mount`, or `has no <controller> controller` | Boot the host on cgroup v2 with that controller enabled. |
| `the step image the allocation's roles run in is not in the store` | Building the step image failed. Check that the daemon can pull the BuildKit image and reach the internet. |
| `a bind names a real path` | Set `store_root` and `node_scratch` to what `realpath` prints for them. |
| `the site's node-local scratch is too deep for the sandbox's sockets` | Set `node_scratch` to a shorter path. |

[running.md](running.md#when-a-command-refuses-to-run) lists the other messages a run can stop with.
