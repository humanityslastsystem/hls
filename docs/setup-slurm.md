# Set up a Slurm cluster

This page prepares a Slurm cluster for HLS. HLS submits every job with `sbatch` from a login node
and runs every container with Apptainer on a compute node. The cluster is ready when a run of the
sample task ends with a score.

In HLS's settings, the cluster is the site: the `[site]` table of `hls.toml` describes it. In every
command below, replace `gpu` with your GPU partition and `/shared/hls` with a directory on your
shared filesystem.

The setup takes six steps, in this order:

1. [Check that the cluster meets the requirements](#check-that-the-cluster-meets-the-requirements).
2. [Install HLS](#install-hls).
3. [Build pasta](#build-pasta-if-the-compute-nodes-lack-it), if the compute nodes lack it.
4. [Write hls.toml](#write-hlstoml).
5. [Create the secrets file](#create-the-secrets-file).
6. [Run the sample task](#run-the-sample-task-to-test-the-cluster).

If a step stops with an error, [Fixing setup errors](#fixing-setup-errors) gives the fix for each
message.

## Check that the cluster meets the requirements

The cluster needs:

- Slurm that confines each job with cgroup v2. Set `CgroupPlugin=cgroup/v2`, `ConstrainCores=yes`,
  `ConstrainDevices=yes` and `ConstrainRAMSpace=yes` in `cgroup.conf`, and
  `ProctrackType=proctrack/cgroup` and `TaskPlugin=task/cgroup` in `slurm.conf`.
- A partition that takes jobs without a GPU. This can be the GPU partition itself or another one.
- NVIDIA GPUs of a supported kind (H200, H100, B200, A100, L40 or L40S), with `nvidia-smi` on the
  compute nodes. [tasks.md](tasks.md) gives the kind and count each task asks for.
- A shared filesystem that the login and compute nodes mount at the same path. It holds the HLS
  checkout, the store (the directory where HLS keeps its data and results), the secrets file and
  the Python that `uv` installs. `uv` installs Python under `~/.local/share/uv`, or in the
  directory `UV_PYTHON_INSTALL_DIR` names.
- Local scratch on every compute node.
- Outbound HTTPS from the compute nodes, for model and dataset downloads, image builds and the
  agent's calls to its model.
- `git` and [`uv`](https://docs.astral.sh/uv/) on the login node. `uv` runs HLS on Python 3.12 and
  downloads it when the node has none.

The compute nodes also need these programs and kernel settings:

- Apptainer and `git`, where a job finds them: in `/usr/local/bin`, `/usr/bin` or `/bin`, or in
  the directory the login node has them in.
- Unprivileged user namespaces.
- `iptables`, `ip6tables` and `sysctl` in `/usr/sbin` or `/sbin`.
- `pasta`, from the passt package. HLS runs it to limit a container's network to the hosts its
  task allows, and calls this limit the network seal. If the nodes lack the package,
  [step 3](#build-pasta-if-the-compute-nodes-lack-it) builds a copy.

Run these checks from the login node:

```sh
scontrol show config | grep -E 'CgroupPlugin|Constrain(Cores|Devices|RAMSpace)|ProctrackType|TaskPlugin'
srun -p gpu stat -fc %T /sys/fs/cgroup
srun -p gpu env PATH=/usr/local/bin:/usr/bin:/bin sh -c 'command -v apptainer git pasta'
srun -p gpu unshare --user --net true
srun -p gpu ls /usr/sbin/iptables /usr/sbin/ip6tables /usr/sbin/sysctl
uv --version
```

The checks pass when:

- `scontrol` prints each of the six settings with the value the list above gives.
- `stat` prints `cgroup2fs`.
- `command -v` prints a path for each of `apptainer`, `git` and `pasta`. If it prints none for
  `pasta`, build it in step 3.
- `unshare` exits 0 and prints nothing.
- `ls` lists the three programs without an error.
- `uv --version` prints a version.

## Install HLS

Clone HLS into a directory on the shared filesystem. The compute nodes run HLS from this checkout.

```sh
git clone --recurse-submodules https://github.com/humanityslastsystem/hls /shared/hls/hls
cd /shared/hls/hls
uv sync
```

When `uv sync` finishes, `uv run hls --help` lists the HLS commands.

## Build pasta if the compute nodes lack it

Skip this step if `command -v` found `pasta` in step 1.

Otherwise, build a copy into a shared directory outside the checkout. The build runs in a Slurm
job on a compute node:

```sh
srun -p gpu --time 15 --mem 4G --cpus-per-task 4 uv run hls build-pasta /shared/hls/pasta
```

The command clones a pinned passt release and builds it. The build needs `git`, `make` and gcc 13
or later on the compute node. To use a compiler other than the node's `cc`, pass `--cc`, as in
`--cc gcc-13`, or set `CC`. The command runs only inside a Slurm job and refuses a login node. It
prints any refusal as one line on stderr and exits with status 1.

When the build succeeds, the command prints the line to add to `hls.toml` in the next step:

```toml
[site]
pasta_path = "/shared/hls/pasta/pasta"
```

## Write hls.toml

Create `hls.toml` in the root of the checkout. HLS reads it from the directory you start a command
in, so run every `hls` command from there. Git ignores the file.

```toml
[site]
partition = "gpu"
store_root = "/shared/hls/store"
node_scratch = "/tmp"
secrets_path = "~/.config/hls/secrets.toml"

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

The four keys of `[site]` are required:

| Key | What to put there |
|---|---|
| `partition` | The partition every GPU job goes to. |
| `store_root` | The directory of the store, on the shared filesystem. [Where to put the store](#where-to-put-the-store) explains what it holds. |
| `node_scratch` | A local directory on the compute nodes. Each job works in `<node_scratch>/hls-<job id>` and removes it when it ends. |
| `secrets_path` | The path of the [secrets file](#create-the-secrets-file), which you create in the next step. |

If you built pasta in step 3, add its `pasta_path` line to `[site]`.

The other two tables say which agent builds the server. An arm is an agent that builds a
submission, the serving system HLS then grades. An arm names a harness (the agent's command-line
tool, here Claude Code), a model and a route. A route says how the agent reaches its model's API:
it names the secrets that hold the API's credential. [running.md](running.md#choosing-the-agent-and-model-arms-and-routes)
covers both.

### Settings most clusters add

Most clusters need a few more keys in `[site]`:

| If your cluster | Set in `[site]` |
|---|---|
| Needs an account or a QOS on each job | `account = "mylab"`, `qos = "normal"` |
| Limits a job's time by its QOS | `qos_by_hours = { normal = 24, long = 168 }`, each QOS with its limit in hours |
| Gives a GPU kind a gres name other than the kind in lower case (`sinfo -p gpu -o %G` prints `gpu:<name>:<count>`) | `gpu_gres = { H200 = "nvidia_h200" }` |
| Mixes GPU kinds in the partition, or has nodes to avoid | `sbatch_extra = ["--constraint=h200", "--exclude=node017"]` |
| Has a partition for jobs without a GPU | `cpu_partition = "cpu"`. `hls provision`, `hls images build`, `hls package` and `hls export` then run there. |
| Refuses jobs without a GPU on the GPU partition | `min_gpus_per_job = 1`, together with `cpu_partition` for the commands that use no GPU |
| Reaches the internet from the compute nodes only through an HTTP proxy | `egress_upstream_proxy = "http://proxy.example:3128"`. The network seal reaches the internet through this proxy. |

[configuration.md](configuration.md) lists every key, and
[`examples/hls.toml`](examples/hls.toml) shows each with a comment.

### Where to put the store

The store is the directory tree under `store_root` where HLS keeps plans, stage outputs, records
and the cache. A stage is one step of a run, such as downloading the model, building the images
or grading the server. The login node and the compute nodes exchange everything through the store,
so it lies on the shared filesystem. HLS creates the directory on first use.

An instance is one execution of an HLS command, such as one run of a task. Each instance has a
directory of its own in the store, such as `runs/sample_task/<id>` for a run of the sample task.
The cache, under `cache/`,
holds every container image, model and dataset the tasks use. HLS fills each cache entry once and
shares it with every later run, so put the store on a filesystem with room for all of them.

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

Only you may read the file (mode 0600). It lies outside the checkout and the store, at a path the
compute nodes read too. `~/.config` works when home directories are on the shared filesystem.
[configuration.md](configuration.md#the-secrets-file) describes the file.

## Run the sample task to test the cluster

```sh
uv run hls run tasks.sample_task.task --arm claude
```

The sample task is the first check of a new cluster. The agent builds a server for a 0.5B model,
and the run's jobs ask for at most one H200, 8 CPUs and 68 GiB of memory, with 225 GiB free under
`node_scratch`. If your cluster has another GPU kind, name it: one of A100, H100, B200, L40 and
L40S.

```sh
uv run hls run tasks.sample_task.task --arm claude --task.hardware.accelerator.kind=L40S
```

The run uses every part of the cluster that a benchmark run uses. It downloads the model and its
data into the cache, builds the container images, and starts the agent in its sealed sandbox, a
container whose network reaches only the hosts the task allows. It then packages what the agent
committed, serves it and grades it. When a job starts, HLS checks on its compute node that the
job's cgroups are set up, that the job got the GPUs, CPUs and memory it asked for, and that the
network seal can run.

The command prints each instance it starts and ends with the run's score:

```
instance runs/sample_task/<id>
...
runs/sample_task/<id>: succeeded: every stage published
runs/sample_task/<id>: score <value> (<score file in the store>)
```

The cluster is set up when the command exits 0 after a score line like the last one.
[running.md](running.md) explains the run and its results, and [tasks.md](tasks.md) describes the
other tasks.

## Fixing setup errors

A command that refuses before it submits anything prints the reason and stops. When Slurm or a
compute node refuses a job, the instance ends as `void` (stopped by an infrastructure fault, with
no score). The last line the command prints carries the reason.

`uv run hls status <instance>` shows the reason again, for the instance the command named, such as
`runs/sample_task/<id>`. A job's own output is
`<store_root>/<instance>/allocations/<k>/submissions/<n>/slurm-<job id>.out`. HLS plans each job
as an allocation, a group of stages: `<k>` numbers the instance's allocations from 0, and `<n>`
numbers the times HLS submitted that allocation to Slurm, from 1.

| The message says | Fix |
|---|---|
| `the secrets file has mode <mode>; group and others must have no access (chmod 600)` | Run `chmod 600` on the file. |
| `the secrets file does not exist`, from a compute node | Move the file to a path on the shared filesystem and set `secrets_path` to it. |
| `site.gpu_gres has no gres name for gpu kind <kind>` | Add the kind to `gpu_gres`. |
| `preflight: sbatch --test-only refused allocation 0:`, then Slurm's message about an invalid account or QOS | Slurm would not take the job. Set `account` and `qos`. |
| `preflight: sbatch --test-only refused allocation 0:`, then Slurm's message that the node configuration is not available | The partition has no GPU under that gres name. Set `gpu_gres`, or run the sample task with the GPU kind the partition has. |
| `preflight: sbatch --test-only refused allocation 0:`, then Slurm's message about a time limit | Set `qos_by_hours`. |
| `is not a cgroup v2 mount` or `lies in no job_<id> directory` | Run the compute nodes on cgroup v2, with `CgroupPlugin=cgroup/v2`, `ProctrackType=proctrack/cgroup` and `TaskPlugin=task/cgroup`. |
| `has no memory.max`, or `has no memory limit` | Set `ConstrainRAMSpace=yes` in `cgroup.conf`. |
| `set site.cpu_partition to a partition that takes jobs with no GPU` | Set `cpu_partition`. |
| `grant mismatch: gpu_kind: requested H200, granted <kind>` | `gpu_gres` maps H200 to the gres of another GPU. Correct the map, or add a `--constraint` to `sbatch_extra`. |
| `grant mismatch: gpus: requested 1, granted <more>` | Jobs see GPUs they did not ask for. Set `ConstrainDevices=yes` in `cgroup.conf`. |
| `grant mismatch: cpus: requested <n>, granted <more>` | Jobs run on CPUs they did not ask for. Set `ConstrainCores=yes` in `cgroup.conf`. |
| `grant mismatch: disk_gib` | `node_scratch` has less free space than the stage needs. Name a larger local disk. |
| `site.pasta_path is unset and this node's PATH (...) holds no pasta`, or `site.pasta_path <path> does not exist` | [Build pasta](#build-pasta-if-the-compute-nodes-lack-it) and set `pasta_path`. |
| `the network seal runs iptables, ip6tables, sysctl from /usr/sbin:/sbin, and <tool> is not there` | Install the `iptables` and `procps` packages on the compute nodes. |
| `this node refuses unprivileged user namespaces` | Set `user.max_user_namespaces` above 0 on the compute nodes, and `kernel.unprivileged_userns_clone = 1` where the kernel has it. |

[running.md](running.md#when-a-command-refuses-to-run) lists the other messages a run can stop with.
