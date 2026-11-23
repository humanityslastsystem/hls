"""What decides how a submission of each graded task is scored, read through the functions the
harness's commands use, and the golden files that hold it.

For each graded task module of the checkout's task directories, this records: every image a
build and a grading use, by role, with its identity (its cache key, its recipe's); the measured
baseline's start; the baseline
credit and the calibration as the roles load them (``read_task``), with the blob id of each file;
the workload fingerprint, the load record, each dataset's corpus id and the pinned upload folder;
the brief the agent build stage renders for ``ARM``; the plans ``hls run`` and ``hls grade``
make, reduced to what decides what executes; the oracle answer keys, read for the stand-in file
the store holds for the task's reference image, and the sha256 of the text the dev reference
publishes beside the dev rows. ``golden/<task>.json`` holds each task's values,
``golden/<task>.brief.md`` its brief, and ``golden/catalog.json`` the recipe key and rendered
``.def`` of every image the catalog builds.

``tests/instances/test_published_identity.py`` compares what this module reads with those files.
After a change meant to move a value, regenerate them and review the diff:

    .venv/bin/python -m tests.instances.golden

Planning runs on a site of this module's own (``SITE``): a temporary store in which each task's
data and the images it declares for itself stand in as a provisioned, warmed cluster would hold
them, and fixed
login-node and compute-node figures. What depends on the machine or the run (paths, the
temporary store, ids, times, the hls commit) is never recorded."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from hls.core.poly import dump_spec
from hls.pipeline import commands, stage
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import reference_gpu_kind
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.stage import StageContext
from hls.recipes import catalog, pins
from hls.recipes.model import order, parent_of
from hls.recipes.render import render_def
from hls.settings import SiteSettings
from hls.stages import agent_build
from hls.stages.images import image_keys
from hls.stages.package import AGENT_BUILD_STAGE
from hls.stages.verify import NO_GPU
from hls.store.cache import ContentCache
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.base import NodeSize
from hls.substrate.secrets import Scrubber
from hls.substrate.ucounts import NAMESPACE_KINDS, commands_limits
from hls.measured import BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEV_REFERENCE_FILE, defaults_path
from hls.taskref import MeasuredPaths, TaskRef, read_task
from hls.verify.answers import answer_key
from hls.verify.fill import question_scopes, question_shape
from hls.substrate.apptainer.images import ApptainerImages
from tests.pipeline.fakes import LocalLauncher
from tests.provision.cached import provision_into
from tests.store.entries import hold_images

#: The checkout this module reads.
CHECKOUT = Path(__file__).resolve().parents[2]
#: Where the golden files live.
GOLDEN_DIR = Path(__file__).resolve().with_name("golden")
#: The golden file of the images the catalog builds, shared by every task.
CATALOG_FILE = "catalog.json"

#: The arm whose build renders each brief: the claude-code arm of ``docs/examples/hls.toml``,
#: under a name of this module's own.
ARM_NAME = "example"
ARM = {"harness": "claude-code", "model": "claude-opus-5", "context_window": 1000000, "output_limit": 64000,
       "effort": "max", "route": "anthropic"}
#: The route it reaches its model through, and the secret its credential comes from.
ROUTE = {"credential": "anthropic_key", "auth": "api_key"}
SECRET_VALUE = "golden-credential-0123456789"
#: The site the plans are made on, beside its store, scratch and secrets paths. The site's name is
#: set only while the settings still have the field, and no plan value recorded here reads it.
SITE = {"partition": "gpu"}
SITE_NAME = "golden"
#: The login node's checkout and toolchain as the plans record them.
CODE = CodeState(checkout=Path("/checkout"), hls_commit="0" * 40, harbor_commit="0" * 40, dirty_diff_sha256=None)
#: The compute node the agent build renders its brief on: its size, the runner's hard
#: ``RLIMIT_NPROC`` (None, unlimited), its per-uid limits, its tasks and its kernel release.
NODE = NodeSize(gpus=8, cpus=64)
NODE_UCOUNTS = {**{f"max_{kind}_namespaces": 1_000_000 for kind in NAMESPACE_KINDS},
                "max_inotify_instances": 128, "max_inotify_watches": 1_048_576}
NODE_TASKS, PID_MAX, UID_TASKS = 3_000, 4_194_304, 400
KERNEL = "5.14.0-503.el9.x86_64"
#: How a verifier's injected calibration is written when it is the task's calibration.
INJECTED = "<the task's calibration>"
#: Stage parameters that name the site a grading runs on and its baseline credit's, which no
#: value recorded here may depend on.
SITE_PARAMS = frozenset({"baseline_site", "site_waiver"})
#: The id of the package warmer a grading of a task with no measured baseline grades.
WARMER = "20000101T000000Z-000000"
#: The key a field holds in place of its value when the harness refuses to read it, with the
#: refusal. No golden file holds one.
UNREADABLE = "unreadable"
#: The fields ``workload_values`` and ``oracle_values`` give.
WORKLOAD_FIELDS = ("workload_fingerprint", "load_record", "corpora", "rows_key", "upload")
ORACLE_FIELDS = ("reference", "answer_keys", "dev_reference_sha256")


def default_files(ref: TaskRef) -> MeasuredPaths:
    """The task's defaults (``hls.measured.defaults_path``), as a plan that finds no file of the
    store's own pins them; None where the task has none."""
    credit, calibration = (defaults_path(ref, name) for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE))
    return MeasuredPaths(calibration=calibration if calibration.is_file() else None,
                         credit=credit if credit.is_file() else None)


def task_modules() -> dict[str, str]:
    """Each task module of the checkout's task directories, by the catalog's name of the task
    (its directory under ``tasks/``)."""
    return {key: catalog.task_module(key) for key in catalog.task_keys()}


def declared_keys() -> set[str]:
    """The cache key of every image a task declares for itself, which a warmed store holds: an
    imported one once its file was imported, a server image once it was built or its file
    imported."""
    out: set[str] = set()
    for key in catalog.task_keys():
        images = catalog.task_images(key)
        keys = image_keys(images.decls, ApptainerImages.recipe_key)
        out |= {keys[name] for name in images.measured.values()}
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# the images


def catalog_images() -> dict[str, Any]:
    """Every image the catalog builds, parents first: its recipe cache key and the sha256 of its
    ``.def`` as the images stage renders it, its parent image and the lock directory named by
    fixed stand-ins, so the text is the same in every checkout."""
    decls = catalog.shared_decls()
    keys = image_keys(decls, ApptainerImages.recipe_key)
    out: dict[str, Any] = {}
    for decl in decls:
        parent = parent_of(decl)
        text = render_def(decl, parent_sif=Path("/parent/image.sif") if parent is not None else None,
                          locks=pins.LOCKS_DIR, uv=pins.UV)
        text = text.replace(str(pins.LOCKS_DIR), "/locks")
        out[decl.name] = {"kind": decl.kind, "parent": parent, "recipe_key": keys[decl.name],
                          "def_sha256": _sha256(text.encode())}
    return out


def task_image_values(key: str) -> dict[str, Any]:
    """Task *key*'s images as a run that builds or deploys them reads them (``task_images``):
    the image of each role, the identity of every image, its cache key, and the constraints its
    submission base applies."""
    images = catalog.task_images(key)
    decls = order(images.decls)
    keys = image_keys(decls, ApptainerImages.recipe_key)
    return {
        "sandbox": images.sandbox, "submission_base": images.submission_base, "harness": images.harness,
        "submission_constraints": images.submission_constraints,
        "measured": {role: {"image": name} for role, name in sorted(images.measured.items())},
        "identities": {d.name: keys[d.name] for d in decls},
    }


def baseline_start(key: str, module: str, task: Any) -> dict[str, Any] | None:
    """How a grading serves task *key*'s measured baseline (``measured_artifact``): its image,
    entrypoint, runtime variables and serve command; None for a task that measures no baseline."""
    if "baseline" not in catalog.task_images(key).measured:
        return None
    artifact, serve = commands.measured_artifact(task, "baseline", key)
    return {"image": artifact.image, "entrypoint": list(artifact.entrypoint),
            "runtime_env": dict(artifact.runtime_env), "serve_command": list(serve)}


# what the roles load


def credit_values(module: str) -> dict[str, Any]:
    """The baseline credit as the score role loads it (``read_task(credit=True)``): B, the credit
    file's blob id and its drive count, and the scorer B is threaded into."""
    loaded = read_task(TaskRef(module=module), credit=True, files=default_files(TaskRef(module=module)))
    credit = None
    if loaded.credit is not None:
        credit = {"blob": loaded.credit.blob, "B": loaded.credit.value.log_credit_mean,
                  "drives": len(loaded.credit.value.drives)}
    return {"credit": credit, "scorer": dump_spec(loaded.task.scorer)}


def calibration_values(module: str) -> dict[str, Any]:
    """The calibration as the verify role loads it (``read_task(calibration=True)``): the
    calibration file's blob id, the calibration as it is injected, and each tenant's verifiers
    with it injected, each injected calibration that equals it written as ``INJECTED``."""
    loaded = read_task(TaskRef(module=module), calibration=True, files=default_files(TaskRef(module=module)))
    calibration = injected = None
    if loaded.calibration is not None:
        injected = dump_spec(loaded.calibration.value)
        calibration = {"blob": loaded.calibration.blob, "kind": type(loaded.calibration.value).__name__,
                       "injected": injected}

    def verifier(v: Any) -> dict[str, Any]:
        dumped = dump_spec(v)
        if injected is not None and dumped.get("calibration") == injected:
            dumped["calibration"] = INJECTED
        return dumped

    return {"calibration": calibration,
            "verifiers": {t.id: [verifier(v) for v in t.verifiers] for t in loaded.task.effective_tenants()}}


def workload_values(task: Any, producer: Any) -> dict[str, Any]:
    """The workload a grading drives: its fingerprint (``workload_fingerprint``), its load record
    (``load_record``), each tenant dataset's corpus id, the rows entry the provisioning resolves,
    and the upload folder's pinned tree digest and destination."""
    from hls.baseline_credit import load_record
    from hls.provision.role import resolve
    from hls.provision.rows import unit_descriptions
    from hls.verify.calibration import workload_fingerprint

    upload = None
    if producer is not None and producer.upload_dir is not None:
        upload = {"dir": list(producer.upload_dir), "dest": producer.upload_dest, "sha256": producer.upload_sha256}
    return {
        "workload_fingerprint": workload_fingerprint(task),
        "load_record": load_record(task).model_dump(mode="json"),
        "corpora": {u["unit_id"]: {"corpus_id": u["corpus_id"], "dev_corpus_id": u["dev_corpus_id"],
                                   "dev_reference_digest": u["dev_reference_digest"]}
                    for u in unit_descriptions(task)},
        "rows_key": resolve(task).rows_key,
        "upload": upload,
    }


def _published_dev_reference(path: Path, task: Any) -> str | None:
    """The sha256 of the text the dev reference at *path* publishes beside the dev rows, held to
    *task* (``read_dev_reference``); None when there is no file."""
    from hls.verify.dev_reference import read_dev_reference

    if not path.is_file():
        return None
    record, _ = read_dev_reference(path, task)
    [published] = record.tenants.values()
    return _sha256(published.text.encode())


def oracle_values(key: str, module: str, task: Any, root: Path) -> dict[str, Any]:
    """The answer keys a grading's verify reads its caching references' answers under
    (``answer_key``): one per reference fingerprint, with the question shape, keyed by the GPU kind
    and the file the store under *root* holds for the reference image the verify runs on
    (``held_reference_sha256``); and the sha256 of the text the task's default dev reference
    publishes beside the dev rows, the file the contestant gets (``hls.verify.dev_reference``)."""
    images = catalog.task_images(key)
    sha = commands.held_reference_sha256(PosixStore(root / "store"), images, recipe_key=ApptainerImages.recipe_key,
                                         image_file=ApptainerImages.image_file)
    gpu_kind = reference_gpu_kind(task) or NO_GPU
    keys = {}
    if sha is not None:
        keys = {fingerprint: {"answer_key": answer_key(fingerprint, gpu_kind, sha),
                              "question_shape": question_shape(task, tenants), "tenants": list(tenants)}
                for fingerprint, tenants in question_scopes(task).items()}
    dev_reference = defaults_path(TaskRef(module=module), DEV_REFERENCE_FILE)
    return {
        "reference": {"gpu_kind": gpu_kind, "image_sha256": sha},
        "answer_keys": keys,
        "dev_reference_sha256": _published_dev_reference(dev_reference, task),
    }


# the plans


def _config(root: Path, sandbox_stores: tuple[str, ...] = ()) -> Path:
    """The site's hls.toml under *root*, with its secrets file, naming each of *sandbox_stores*
    as a directory beside *root*, apart from the store and the secrets (``hls.settings``); a site
    for a task whose tool runner names no store has no such table."""
    secrets = root / "secrets.toml"
    secrets.write_text(f'[secrets]\n{ROUTE["credential"]} = "{SECRET_VALUE}"\n')
    secrets.chmod(0o600)
    site = {**SITE, "store_root": str(root / "store"), "node_scratch": str(root / "node"),
            "secrets_path": str(secrets)}
    if "name" in SiteSettings.model_fields:
        site["name"] = SITE_NAME
    lines = ["[site]", *(f"{k} = {json.dumps(v)}" for k, v in site.items()),
             "[routes.anthropic]", *(f"{k} = {json.dumps(v)}" for k, v in ROUTE.items()),
             f"[arms.{ARM_NAME}]", *(f"{k} = {json.dumps(v)}" for k, v in ARM.items())]
    if sandbox_stores:
        stores = root.parent / f"{root.name}-sandbox-stores"
        for name in sandbox_stores:
            (stores / name).mkdir(parents=True, exist_ok=True)
        lines += ["[sandbox_stores]", *(f"{name} = {json.dumps(str(stores / name))}" for name in sandbox_stores)]
    path = root / "hls.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


@contextmanager
def login_node(root: Path) -> Iterator[None]:
    """Plan as a login node of ``SITE`` whose cluster is provisioned and warmed: the checkout
    state and toolchain fixed (``CODE``); the cache holding each measured image a plan asks about,
    as its images warmer would; and the checks whose subject no checkout of the tests holds
    passing: a committed tree and committed credit files (this module also reads a changed
    checkout), the uncommitted upload folder, the oracle answers, and the site a grading runs on,
    which no value recorded here depends on. The compute node's figures are ``NODE``'s."""
    real_holds = commands.cache_holds
    declared = declared_keys()

    def holds(settings):
        held = real_holds(settings)

        def asked(keys):
            hold_images(settings.site.store_root, [k for k in keys if k in declared])
            return held(keys)

        return asked

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(commands, "code_state", lambda path: CODE)
        mp.setattr(commands, "toolchain_here",
                   lambda: Toolchain(python=Path("/python"), path="/usr/bin:/bin", home=Path("/home")))
        mp.setattr(commands, "cache_holds", holds)
        mp.setattr(commands, "check_committed_tree", lambda code: None)
        mp.setattr(commands, "check_upload_folder", lambda checkout, task_module, producer: None)
        mp.setattr(commands, "check_oracle_coverage", lambda store, **kw: None)
        mp.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
        mp.setattr(commands, "check_grading_site", lambda *a, **kw: [], raising=False)
        mp.setattr(stage, "hard_limit", lambda: None)
        mp.setattr(stage, "node_limits", lambda: dict(NODE_UCOUNTS))
        mp.setattr(stage, "uid_tasks", lambda uid: UID_TASKS)
        mp.setattr(stage, "node_tasks", lambda: NODE_TASKS)
        mp.setattr(stage, "node_pid_max", lambda: PID_MAX)
        mp.setattr(stage, "kernel_release", lambda: KERNEL)
        yield


def plan_both(key: str, module: str, root: Path) -> tuple[Plan, Plan]:
    """The first plans ``hls run --arm ARM`` and ``hls grade`` make for *module* on the site under
    *root*, its data provisioned in the cache first. The grading grades the task's measured
    baseline (``--measured baseline``), or, for a task that measures none, a published package."""
    from tests.evaluate.warmer import publish_package
    from tests.provision.warmer_config import fill_sandbox_stores, task_sandbox_stores

    stores = task_sandbox_stores(module)
    config = _config(root, stores)
    commands.task_id_of(module)
    task = read_task(TaskRef(module=module)).task
    provision_into(root / "store", task)
    if stores:
        held = root.parent / f"{root.name}-sandbox-stores"
        fill_sandbox_stores({name: held / name for name in stores}, module, root / "store")
    run = commands.plan_run(config=config, overrides={}, task_module=module, arm=ARM_NAME, arm_change=None,
                            job_name="hls-run", echo=lambda line: None)
    if "baseline" in catalog.task_images(key).measured:
        grading = commands.plan_grade(config=config, overrides={}, task_module=module, measured="baseline",
                                      job_name="hls-grade", echo=lambda line: None)
    else:
        package = publish_package(PosixStore(root / "store"), b"submission image", task=task.id, warmer=WARMER)
        grading = commands.plan_grade(config=config, overrides={}, task_module=module, package=package,
                                      job_name="hls-grade", echo=lambda line: None)
    return run, grading


def _normalised(value: Any, swap: Callable[[str], str]) -> Any:
    if isinstance(value, dict):
        return {k: _normalised(v, swap) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalised(v, swap) for v in value]
    if isinstance(value, str):
        return swap(value)
    return value


def reduced_plan(plan: Plan, root: Path) -> dict[str, Any]:
    """What of *plan* decides what executes: each stage's kind, order, shape, grants and
    parameters, but those naming a site (``SITE_PARAMS``), and each allocation's stages and
    request. Paths under the temporary *root*, the plan's instance and its grading's id are
    named by stand-ins."""
    instance, grading = plan.instance, InstanceLayout.parse(plan.instance).grading_id

    def swap(text: str) -> str:
        text = text.replace(str(root), "<root>").replace(instance, "<instance>")
        return text.replace(grading, "<grading>") if grading is not None else text

    def params(values: dict[str, Any]) -> dict[str, Any]:
        return _normalised({k: v for k, v in values.items() if k not in SITE_PARAMS}, swap)

    return {
        "pipeline": plan.pipeline,
        "task": plan.task,
        "stages": [{"name": s.name, "kind": s.kind, "after": list(s.after), "shape": s.shape.model_dump(mode="json"),
                    "grants": dict(s.grants), "params": params(s.params)} for s in plan.stages],
        "allocations": [{"stages": list(a.stages), "request": a.request.model_dump(mode="json"),
                         "depends_on": a.depends_on, "socket_tasks": a.placement.socket_tasks} for a in plan.allocations],
    }


# the brief


def render_brief(run: Plan, root: Path) -> str:
    """The brief the agent_build stage of *run* writes (``agent_build._Build.brief``), on a node
    of ``NODE``'s size: the commands' process ceiling and per-uid limits read first, as the
    stage's setup reads them, and the task loaded with its calibration and credit, as the stage
    loads it."""
    planned = run.stage(AGENT_BUILD_STAGE)
    store = PosixStore(run.store_root)
    scratch = Path(tempfile.mkdtemp(prefix="hls-", dir=root))
    ctx = StageContext(store=store, plan=run, stage=planned, attempt=1, launcher=LocalLauncher(gpus=NODE.gpus, node=NODE),
                       watch=None, events=EventLog(scratch / "events.jsonl"), inputs={}, deadline=time.time() + 600,
                       secrets={}, scrubber=Scrubber({}), cache=ContentCache(store), scratch=scratch, job_id="1",
                       runtime=ApptainerRuntime())
    try:
        build = agent_build._Build(ctx)
        build.ceiling = ctx.process_ceiling()
        build.ucounts = commands_limits(ctx.ucount_limits())
        params = agent_build.AgentBuildParams.model_validate(planned.params)
        task = read_task(params.task, calibration=True, credit=True, files=default_files(params.task)).task
        return build.brief(task, build.artifact())
    finally:
        ctx.close()


# one task


def _read(fields: tuple[str, ...], read: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """*fields* as *read* returns them; when the harness refuses to read them, each field holds
    the refusal (``UNREADABLE``), so a change that makes it refuse a file names the fields it
    moved."""
    try:
        return read()
    except Exception as e:
        refusal = {UNREADABLE: f"{type(e).__name__}: {' '.join(str(e).split())}"}
        return dict.fromkeys(fields, refusal)


def collect(key: str, module: str, root: Path) -> tuple[dict[str, Any], str | None]:
    """Task *key*'s golden values and its brief, None when it cannot be rendered, its plans made
    under the temporary *root*."""
    commands.task_id_of(module)
    task = read_task(TaskRef(module=module)).task
    artifact = getattr(sys.modules[module], "ARTIFACT", None)
    runs: list[Plan] = []
    brief: list[str] = []

    def planned() -> dict[str, Any]:
        with login_node(root):
            run, grading = plan_both(key, module, root)
        runs.append(run)
        return {"plans": {"run": reduced_plan(run, root), "grading": reduced_plan(grading, root)}}

    def briefed() -> dict[str, Any]:
        if not runs:
            raise ValueError("no run was planned, so no build renders a brief")
        with login_node(root):
            brief.append(render_brief(runs[0], root))
        return {"brief": {"sha256": _sha256(brief[0].encode()), "file": f"{key}.brief.md"}}

    values = {
        "task": key,
        "module": module,
        "task_id": task.id,
        **_read(("images",), lambda: {"images": task_image_values(key)}),
        **_read(("baseline_start",), lambda: {"baseline_start": baseline_start(key, module, task)}),
        "submission_serve_command": list(artifact.serve_command) if artifact is not None else None,
        **_read(("credit", "scorer"), lambda: credit_values(module)),
        **_read(("calibration", "verifiers"), lambda: calibration_values(module)),
        **_read(WORKLOAD_FIELDS, lambda: workload_values(task, getattr(artifact, "producer", None))),
        **_read(("plans",), planned),
        **_read(("brief",), briefed),
        **_read(ORACLE_FIELDS, lambda: oracle_values(key, module, task, root)),
    }
    return values, brief[0] if brief else None


def unreadable(values: Any) -> list[str]:
    """The refusals *values* hold (``UNREADABLE``)."""
    if isinstance(values, dict):
        return [values[UNREADABLE]] if UNREADABLE in values else [r for v in values.values() for r in unreadable(v)]
    return []


def golden_text(values: dict[str, Any]) -> str:
    """The text of a golden JSON file holding *values*."""
    return json.dumps(values, indent=2, sort_keys=True) + "\n"


def collect_all(root: Path) -> dict[str, bytes]:
    """Every golden file's bytes by its name, each task planned in a directory of its own under
    *root*."""
    files = {CATALOG_FILE: golden_text(catalog_images()).encode()}
    for key, module in task_modules().items():
        work = root / key
        work.mkdir()
        values, brief = collect(key, module, work)
        refused = unreadable(values)
        if refused or brief is None:
            raise ValueError(f"{key}: the harness refuses to read its values, so no golden file is written: {refused}")
        files[f"{key}.json"] = golden_text(values).encode()
        files[f"{key}.brief.md"] = brief.encode()
    return files


def main() -> int:
    """Write every golden file from this checkout, remove any other file of the golden
    directory, and print the names of those that changed. A task whose values the harness
    refuses to read writes nothing."""
    with tempfile.TemporaryDirectory(prefix="hls-golden-") as tmp:
        try:
            files = collect_all(Path(tmp))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
    GOLDEN_DIR.mkdir(exist_ok=True)
    for path in GOLDEN_DIR.iterdir():
        if path.name not in files:
            path.unlink()
            print(f"removed {path.name}")
    for name, data in files.items():
        path = GOLDEN_DIR / name
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
            print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
