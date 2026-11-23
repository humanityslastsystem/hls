"""The hls command line. These commands plan an instance, submit it and drive it to its end on
the site's Slurm: ``run`` and ``grade``, ``baseline-credit``, the cache warmers (``provision``,
``images build``, ``package``, ``oracle fill``), ``calibrate`` and ``export``;
``resume``, ``status``, ``cancel`` and ``follow`` look after an instance; ``runner`` is what an
allocation's batch script starts. ``view`` serves the rollout viewer. ``build-pasta`` sets up a
site, and ``images lock-apt``, ``lock-pip``, ``lock-npm`` and
``base-digest`` make the image recipes' pins; they run where they are started. ``capped-requests`` lists
the sessions a calibration's pool member ran to the output cap, from the calibration's published
captures. ``run`` is the one command a run of a task's default configuration needs: it provisions,
measures and fills what the run lacks first (``hls.pipeline.prepare``)."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

import click
import typer
import typer.core

from hls.obs.logsetup import configure_logging

if TYPE_CHECKING:
    from hls.pipeline.plan import Plan

# A traceback's local variables can hold a secret: the secrets file's text, or a value read from it.
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
#: The variables that size the thread pools numpy and torch start when their libraries load:
#: OpenBLAS's, which numpy's wheels carry, and OpenMP's and MKL's, which torch's use.
THREAD_POOL_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _cap_thread_pools() -> None:
    """Start this process's numpy and torch thread pools at one thread each, unless the operator
    sized one. The commands that hold an orchestrator, and ``follow``, call this first, before
    anything imports numpy, since a pool is sized when its library loads. They run on shared
    login nodes, where every thread counts against the user's process limit, and do no heavy
    linear algebra, while OpenBLAS alone would start a thread per core. The runner, on a
    compute node, keeps its pools."""
    for name in THREAD_POOL_VARIABLES:
        # An empty value leaves the pool at its default size, so it counts as unset.
        if not os.environ.get(name):
            os.environ[name] = "1"


@app.callback()
def _main() -> None:
    """HLS: plan, run and look after benchmark instances on a Slurm site."""


def _parse_dotted_flags(args: list[str]) -> dict[str, str]:
    """Turn leftover CLI tokens (``--a.b.c value`` or ``--a.b.c=value``) into
    ``{a.b.c: value}``. A flag with no value fails loud. The module positional must
    precede these (Click fills it first); the ``=`` form is ordering-immune."""
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if not tok.startswith("--"):
            raise ValueError(f"unexpected CLI token {tok!r}; overrides are --<path> <value>")
        key = tok[2:]
        if "=" in key:
            k, _, v = key.partition("=")
            out[k.strip()] = v
            i += 1
        else:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise ValueError(f"--{key} has no value (use --{key}=<value> if the value starts with '--')")
            out[key.strip()] = args[i + 1]
            i += 2
    return out


def _override_first(module_path: str) -> str:
    """The refusal of an override flag that stands where the task module should come first."""
    return (f"expected the task module first, got override flag {module_path!r}. "
            "Put <task-module> before any --<dotted> overrides.")


@app.command("capped-requests", short_help="List the sessions a calibration's pool member ran to the output cap.")
def capped_requests(
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task"),
    instance: str = typer.Option(..., "--from", help="The calibration instance whose captures define the list, "
                                 "calibrations/<task>/<id>"),
    engine: str = typer.Option("vllm", "--engine", help="The pool member whose captures define the list: a session "
                               "is listed when this member's reply to one of its requests reached the output cap in "
                               "any serving"),
    write: bool = typer.Option(False, "--write", help="Hand the listed rows to the task module's "
                               "record_removed_rows, which adds them to its removal list (a union: a row once "
                               "observed at the cap stays removed)"),
    config: Path = typer.Option(Path("hls.toml")),
    store: Path | None = typer.Option(None, "--store", help="Store root of the instance; defaults to "
                                      "site.store_root of --config"),
) -> None:
    """List, per tenant, the sessions a calibration's pool member ran to the output cap -- the
    removal list a task authors as DataSource.removed_rows -- with the counts that price the
    removal. Reads the instance's published captures; drives and deploys nothing. With --write,
    the rows go to the task module's record_removed_rows."""
    from hls.pipeline import calibration, commands

    configure_logging(verbose=False)
    try:
        per_tenant = calibration.capped_requests(task_module, instance=instance, engine=engine,
                                                 store=commands.open_store(config, store))
    except ValueError as e:
        typer.echo(f"hls capped-requests: {e}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"=== capped requests under {engine} ({instance}) ===")
    for tenant, capped in per_tenant.items():
        typer.echo(f"{tenant}: {capped.summary()}")
        for row in capped.rows:
            typer.echo(f"    {row!r},")
    if write:
        # The list's file, format and derivation belong to the task, not to this command: it
        # knows which rows capped, the task knows where they are written.
        record = getattr(sys.modules[task_module], "record_removed_rows", None)
        if record is None:
            raise typer.BadParameter(f"--write needs {task_module} to define record_removed_rows(rows_by_tenant) "
                                     "-> (path, added); this task owns no removal list to add to")
        written, added = record({tenant: capped.rows for tenant, capped in per_tenant.items()})
        for tenant, n in added.items():
            typer.echo(f"{tenant}: {n} row(s) added to the list")
        typer.echo(f"wrote {written}")


def _viewer_static_dir() -> "Path | None":
    """Path to the built HLS viewer SPA static assets, or None if not yet built."""
    from hls.viewer.build import static_dir as _build_static

    return _build_static()


def _serve(app_obj, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app_obj, host=host, port=port, log_level="info")


@app.command(short_help="Serve the rollout viewer on a directory of instances.")
def view(
    runs_dir: Path | None = typer.Argument(None, help="A directory of instances or published records, such as "
                                           "<store>/runs or <store>/warmers; the store's runs/ by default"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
    no_build: bool = typer.Option(False, "--no-build", help="Do not auto-build the SPA"),
    config: Path = typer.Option(Path("hls.toml"), help="hls.toml, read for the store's root when no directory "
                                "is given"),
) -> None:
    """Launch the rollout viewer on a directory, by default the runs of the store hls.toml names.
    Every request needs the session token in the printed URL; a directory that is or holds the
    store's answers area is refused."""
    configure_logging(verbose=False)
    from hls.settings import SettingsError, load_settings
    from hls.store.layout import ROLLOUTS
    from hls.viewer import create_app
    from hls.viewer.build import build_spa, find_available_port
    from hls.viewer.scanner import ViewRefused, check_viewing_root
    from hls.viewer.server import new_token, viewer_url

    if runs_dir is None:
        try:
            runs_dir = load_settings(config).site.store_root / ROLLOUTS
        except SettingsError as e:
            typer.echo(f"hls view: {e}", err=True)
            raise typer.Exit(1) from None
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        typer.echo(f"hls view: {runs_dir}: no such directory", err=True)
        raise typer.Exit(1)
    try:
        check_viewing_root(runs_dir)
    except ViewRefused as e:
        typer.echo(f"hls view: {e}", err=True)
        raise typer.Exit(1) from None

    static = _viewer_static_dir()
    if static is None and not no_build:
        typer.echo("Building viewer SPA (one-time)...")
        if build_spa():
            static = _viewer_static_dir()

    chosen = find_available_port(host, port, port + 9) or port
    token = new_token()
    app_obj = create_app(runs_dir, token=token, static_dir=static)
    typer.echo(f"HLS rollout viewer: {viewer_url(host, chosen, token)}  (runs: {runs_dir})")
    _serve(app_obj, host, chosen)


#: The constant each option of ``hls calibrate`` that defaults to None takes in its place, by the
#: option's parameter name: its module and its name. Its value is stated in the option's help.
CALIBRATE_DEFAULTS = {
    "fanout": ("hls.pipeline.calibration", "DEFAULT_FANOUT"),
    "servings_per_engine": ("hls.calibrate", "DEFAULT_REPORT_SERVINGS"),
    "eps": ("hls.calibrate", "DEFAULT_EPS"),
    "mu_per_mtok": ("hls.calibrate", "DEFAULT_MU_PER_MTOK"),
    "delta": ("hls.calibrate", "DEFAULT_DELTA"),
    "eot_expected_max": ("hls.verify.calibration", "DEFAULT_EOT_EXPECTED_MAX"),
    "max_servings_per_engine": ("hls.pipeline.calibration", "DEFAULT_MAX_SERVINGS_PER_ENGINE"),
}


class DefaultsInHelp(typer.core.TyperCommand):
    """A command whose options named in ``CALIBRATE_DEFAULTS`` end their help with the value of
    the constant they default to. The values are read when the help is printed: their modules
    load numpy, which a command's ``_cap_thread_pools`` must precede."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        for param in self.params:
            where = CALIBRATE_DEFAULTS.get(param.name or "")
            # The help renderer reads square brackets as markup, so the literal one is escaped.
            if where is not None and isinstance(param, click.Option) and param.help and "default:" not in param.help:
                param.help = f"{param.help} \\[default: {getattr(importlib.import_module(where[0]), where[1])}]"
        super().format_help(ctx, formatter)


# The commands that plan an instance. They use the strict hls.settings loader and take --site.*,
# --run.* and --resources.* overrides; the commands that load a task also take --task.*.


def _plan_and_start(command: str, plan: Callable[[], Plan], overrides: dict[str, str], *,
                    start: Callable[..., int] | None = None, prepare: Callable[[], int] | None = None) -> NoReturn:
    """Run *prepare*, when given, and exit with its status unless it is 0; then make the plan
    *plan* returns, submit it and drive it to its end with *start*, called with the plan,
    *overrides* and ``echo`` (``commands.start`` unless given), and exit with the status it
    returns. A ValueError while preparing or planning, which is how a command refuses what it was
    given (PlanError and SecretsError among them), prints one line on stderr and exits 1, with no
    traceback. An error while an instance is driven is a fault, not a refusal, and keeps its
    traceback, which shows no local variable."""
    from hls.pipeline import commands

    try:
        status = prepare() if prepare is not None else 0
    except ValueError as e:
        typer.echo(f"hls {command}: {e}", err=True)
        raise typer.Exit(1) from None
    if status:
        raise typer.Exit(status)
    try:
        made = plan()
    except ValueError as e:
        typer.echo(f"hls {command}: {e}", err=True)
        raise typer.Exit(1) from None
    raise typer.Exit((start or commands.start)(made, overrides, echo=typer.echo))


@app.command("build-pasta", short_help="Build the pasta program the network seal runs, for site.pasta_path.")
def build_pasta_command(
    directory: Path = typer.Argument(..., help="Where to stage it: a directory every compute node mounts, outside "
                                     "this checkout"),
    cc: str = typer.Option("cc", "--cc", envvar="CC", help="The C compiler, gcc 13 or later"),
) -> None:
    """Build pasta from the passt release hls pins and stage it in DIRECTORY, for a site whose
    compute nodes lack the passt package. Run it inside a Slurm allocation, as srun ... uv run hls
    build-pasta DIRECTORY; it refuses a login node. Then name the binary it prints as
    site.pasta_path in hls.toml. A refusal is one line on stderr, with exit status 1."""
    import subprocess

    from hls.pipeline import commands
    from hls.substrate.pasta import PastaRefused, build_pasta

    directory = directory.expanduser().absolute()
    try:
        pasta = build_pasta(directory, checkout=commands.checkout_root(), run=subprocess.run, cc=cc,
                            echo=typer.echo)
    except PastaRefused as e:
        typer.echo(f"hls build-pasta: {e}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"\npasta staged at {pasta}. Name it in hls.toml:\n\n  [site]\n  pasta_path = \"{pasta}\"")


@app.command("provision", short_help="Fill the cache with a task's weights, rows and tool bundles.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def provision_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task; overrides follow it"),
    rows_only: bool = typer.Option(False, "--rows-only", help="Leave out the task model's weights"),
    job_name: str = typer.Option("hls-provision", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Fill the cache with a task's weights, rows and tool bundles ahead of its runs, on the
    site's CPU partition when it has one. For a task that declares sandbox images, also pull the
    ones their site store lacks."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        # Click fills the module from the first leftover token, which can be an override flag.
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        return commands.plan_provision(config=config, overrides=overrides, task_module=task_module,
                                       weights=not rows_only, job_name=job_name, echo=typer.echo)

    _plan_and_start("provision", plan, overrides)


images_app = typer.Typer(add_completion=False, no_args_is_help=True,
                         help="Shared images in the cache, and the lock files their recipes install from.")
app.add_typer(images_app, name="images")


@images_app.command("build", short_help="Build a task's images into the cache.",
                    context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def images_build_command(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="The task as the image catalog names it, its directory under tasks/, "
                               "such as multitenant_slo; overrides follow it"),
    job_name: str = typer.Option("hls-images", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Build every image a task declares, its shared images and its server images, into the cache
    ahead of its runs, on the site's CPU partition when it has one."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        # Click fills the task from the first leftover token, which can be an override flag.
        if task.startswith("--"):
            raise ValueError(f"expected the task first, got override flag {task!r}. Put <task> before any "
                             "--<dotted> overrides.")
        overrides.update(_parse_dotted_flags(ctx.args))
        return commands.plan_images_build(config=config, overrides=overrides, task=task, job_name=job_name,
                                          echo=typer.echo)

    _plan_and_start("images build", plan, overrides)


def _pin_tool(command: str, act: Callable[[], object]) -> NoReturn:
    """Run *act*, which writes one of the recipes' lock files or reads a pin, and exit 0; a
    LockRefused is one line on stderr naming *command*, with exit status 1."""
    from hls.recipes.locking import LockRefused

    try:
        act()
    except LockRefused as e:
        typer.echo(f"hls images {command}: {e}", err=True)
        raise typer.Exit(1) from None
    raise typer.Exit(0)


@images_app.command("lock-apt", short_help="Pin a recipe's apt packages from the Ubuntu snapshot or NVIDIA's "
                    "repository.")
def images_lock_apt_command(
    repo: Literal["ubuntu", "nvidia"] = typer.Argument(..., help="Where the packages come from: the Ubuntu "
                                                       "snapshot pins.SNAPSHOT names, or pins.NVIDIA_REPO"),
    name: str = typer.Argument(..., help="The lock, such as apt-base: reads locks/<name>.in, writes "
                                "locks/<name>.txt"),
    task: str | None = typer.Option(None, "--task", help="A layer a task declares for itself: reads and writes "
                                    "tasks/<task>/locks/, from the Ubuntu snapshot"),
) -> None:
    """Pin each package locks/NAME.in lists to the highest version REPO carries and write
    locks/NAME.txt (under src/hls/recipes, or with --task under the task's directory), for a
    maintainer whose recipe's apt inputs changed.
    An NVIDIA lock also records each .deb's sha256, and is refused while a dependency the
    repository carries is left unpinned. The lock's first line names this command; a committed
    lock keeps the first line it has, which every image key hashes, until it is regenerated."""
    from hls.recipes.locking.apt import fetch_index, lock_apt

    _pin_tool("lock-apt", lambda: lock_apt(repo, name, fetch=fetch_index, task=task, echo=typer.echo))


@images_app.command("lock-pip", short_help="Compile a base's or a layer's hashed pip lock.")
def images_lock_pip_command(
    kind: Literal["base", "layer"] = typer.Argument(..., help="A base, compiled alone, or a layer, compiled against "
                                                    "its parent's pins"),
    name: str = typer.Argument(..., help="The lock, such as base-s0: reads locks/<name>.in, writes locks/<name>.txt"),
    parent: str | None = typer.Option(None, "--parent", help="The lock of the image a layer builds on, such as "
                                      "base-s-a: the task's own, else the shared directory's; none for the cuda base"),
    layer_owned: list[str] = typer.Option([], "--layer-owned", help="A package a base leaves out for each layer to "
                                          "install at its own version; repeat for each"),
    source_build: list[str] = typer.Option([], "--source-build", help="A package, by name, that a layer builds from "
                                           "its source distribution, as the layer's source_builds in the catalog "
                                           "declare; repeat for each"),
    task: str | None = typer.Option(None, "--task", help="A layer a task declares for itself: reads and writes "
                                    "tasks/<task>/locks/"),
    server: bool = typer.Option(False, "--server", help="A task's server layer, which may hold the serving "
                                "framework the deny list names; needs --task"),
    no_deps: bool = typer.Option(False, "--no-deps", help="A task's server layer: lock exactly the pins its .in "
                                 "names, resolving nothing"),
) -> None:
    """Compile locks/NAME.in (under src/hls/recipes) into the hashed lock locks/NAME.txt with the
    uv pins.LOCK_UV names, for the images' interpreter and platform, counting uploads before
    pins.EXCLUDE_NEWER only. A base also writes its torch constraints; a layer keeps only what its
    parent lacks. A lock that names a deny-listed package, or that wheels alone could not install
    beyond its source builds, is refused. The lock's first line names this command; a committed
    lock keeps the first line it has, which every image key hashes, until it is regenerated."""
    from hls.recipes.locking.pip import compile_lock, lock_pip

    _pin_tool("lock-pip", lambda: lock_pip(kind, name, compiler=compile_lock, parent=parent,
                                             layer_owned=tuple(layer_owned), source_builds=tuple(source_build),
                                             task=task, server=server, no_deps=no_deps, echo=typer.echo))


@images_app.command("lock-npm", short_help="Resolve the agent cap's npm lock of its CLIs.")
def images_lock_npm_command(
    name: str = typer.Argument(..., help="The lock the agent caps install from, such as npm-agent.json"),
) -> None:
    """Resolve the package-lock.json of the agent CLIs the caps that install from NAME declare
    (pins.CLAUDE_CODE_VERSION, pins.CODEX_VERSION), with the npm of the Node they run on, checked
    against its pinned digest, counting versions published before pins.EXCLUDE_NEWER only, and
    write locks/NAME (under src/hls/recipes) once the image renderer accepts it."""
    from hls.recipes.locking.npm import fetch_node, lock_npm, resolve_lock

    _pin_tool("lock-npm", lambda: lock_npm(name, fetch=fetch_node, resolve=resolve_lock, echo=typer.echo))


@images_app.command("base-digest", short_help="Print a Docker Hub tag's index digest, for pins.CUDA_BASE.")
def images_base_digest_command(
    repo: str = typer.Argument(..., help="The Docker Hub repository, such as nvidia/cuda"),
    tag: str = typer.Argument(..., help="The tag, such as 13.0.1-devel-ubuntu24.04"),
) -> None:
    """Print REPO:TAG@<digest>, the digest of the tag's multi-architecture index, which a recipe's
    base is pinned by (pins.CUDA_BASE). It asks Docker Hub anonymously and writes nothing."""
    import urllib.request

    from hls.recipes.locking.base_digest import base_reference

    _pin_tool("base-digest", lambda: typer.echo(base_reference(repo, tag, urlopen=urllib.request.urlopen)))


@app.command("calibrate", short_help="Measure the constant a task's calibrated gate needs.",
             cls=DefaultsInHelp, context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def calibrate_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.tab_completion.task; overrides follow it"),
    engines: str = typer.Option("vllm,sglang", "--engines", help="Comma-separated pool members, serving stacks "
                                "whose images the task module declares: vllm, sglang"),
    fanout: int | None = typer.Option(None, "--fanout", help="Allocations that take servings side by side; a "
                                      "token-level top-up may overshoot its exposure target by up to the fanout "
                                      "less one servings per engine"),
    servings_per_engine: int | None = typer.Option(None, "--servings-per-engine", help="Servings each engine "
                                                   "takes, for a sampling gate; refused for the token-level gate, "
                                                   "whose count follows from its exposure target"),
    sessions_per_draw: int | None = typer.Option(None, "--sessions-per-draw", help="Sessions one serving drives "
                                                 "(default: the whole test split); the subsample keeps each "
                                                 "tenant's arrival rate and trims the drive's span"),
    eps: float | None = typer.Option(None, "--eps", help="(P)'s relative precision target, read by both gates"),
    mu_per_mtok: float | None = typer.Option(None, "--mu-per-mtok", help="(T)'s tolerated flips per million "
                                             "graded tokens at unobserved gap levels, which sets the token-level "
                                             "gate's exposure target"),
    delta: float | None = typer.Option(None, "--delta", help="The token-level gate's family-wise false-alarm "
                                       "budget per graded trace"),
    eot_expected_max: float | None = typer.Option(None, "--eot-expected-max", help="The sampling gate's expected "
                                                  "end-of-turn count per graded trace"),
    max_servings_per_engine: int | None = typer.Option(None, "--max-servings-per-engine", help="The most servings "
                                                       "an engine may take before the campaign is refused"),
    job_name: str = typer.Option("hls-calibrate", help="Slurm job name of the allocations"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Measure the constant a task's calibrated gate needs, on the task's measured images: the
    reference's instrument check, servings of each pool member side by
    side, each share scored by the pinned reference, more servings while exposure falls short,
    then the calibration.json, which it writes into the store, where runs of the task find it."""
    _cap_thread_pools()
    from hls.calibrate import DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK
    from hls.pipeline import calibration
    from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        # Click fills the module from the first leftover token, which can be an override flag.
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        return calibration.plan_calibrate(
            config=config, overrides=overrides, task_module=task_module,
            engines=[e.strip() for e in engines.split(",") if e.strip()],
            fanout=calibration.DEFAULT_FANOUT if fanout is None else fanout,
            servings_per_engine=servings_per_engine, sessions_per_draw=sessions_per_draw,
            eps=DEFAULT_EPS if eps is None else eps,
            mu_per_mtok=DEFAULT_MU_PER_MTOK if mu_per_mtok is None else mu_per_mtok,
            delta=DEFAULT_DELTA if delta is None else delta,
            eot_expected_max=DEFAULT_EOT_EXPECTED_MAX if eot_expected_max is None else eot_expected_max,
            max_servings_per_engine=(calibration.DEFAULT_MAX_SERVINGS_PER_ENGINE if max_servings_per_engine is None
                                     else max_servings_per_engine),
            job_name=job_name, echo=typer.echo)

    _plan_and_start("calibrate", plan, overrides)


@app.command("package", short_help="Build a source bundle into a task's sealed submission image.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def package_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task; overrides follow it"),
    bundle: Path = typer.Option(..., "--bundle", help="Absolute path of the source bundle (a git bundle of main), "
                                "which the command copies into the store before planning"),
    sha256: str | None = typer.Option(None, "--sha256", help="Required: the bundle's sha256 as 64 lowercase hex; "
                                      "the command refuses a bundle whose bytes have another"),
    job_name: str = typer.Option("hls-package", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Build a source bundle into a task's sealed submission image, or record that it does not
    build, on the site's CPU partition when it has one."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        return commands.plan_package(config=config, overrides=overrides, task_module=task_module, bundle=bundle,
                                     bundle_sha256=sha256, job_name=job_name, echo=typer.echo)

    _plan_and_start("package", plan, overrides)


@app.command("baseline-credit", short_help="Measure a task's baseline credit B on its measured baseline image.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def baseline_credit_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task; overrides follow it"),
    drives: int = typer.Option(1, "--drives", help="Fresh drives of the measured baseline to pool; one records no "
                               "spread between drives"),
    fanout: int = typer.Option(1, "--fanout", help="Allocations that drive side by side, from 1 to --drives"),
    job_name: str = typer.Option("hls-credit", help="Slurm job name of the allocations"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Measure a task's baseline credit B on its measured baseline image: fresh drives, each
    scored with no credit, pooled into the baseline.json it writes into the store, where runs of
    the task find it."""
    _cap_thread_pools()
    from hls.pipeline import credit

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        return credit.plan_credit(config=config, overrides=overrides, task_module=task_module, drives=drives,
                                  fanout=fanout, job_name=job_name, echo=typer.echo)

    _plan_and_start("baseline-credit", plan, overrides, start=credit.run_credit)


@app.command("run", short_help="Build a submission with an arm's agent, package it and grade it.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def run_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task; overrides follow it"),
    # The help renderer reads square brackets as markup, so a literal one is escaped.
    arm: str = typer.Option(..., "--arm", help="The arm whose agent builds: an \\[arms.<name>] table of hls.toml"),
    arm_change: str | None = typer.Option(None, "--arm-change", help="Why the arm differs from the task's latest "
                                          "build of it; a rerun of a changed arm is refused without one"),
    build_only: bool = typer.Option(False, "--build-only", help="End with the package and a record of the rollout; "
                                    "grade it later with hls grade --rollout"),
    soft_sec: int | None = typer.Option(None, "--soft-sec", help="The wall-clock budget the agent is told, in "
                                        "seconds, replacing the task's; needs --hard-sec"),
    hard_sec: int | None = typer.Option(None, "--hard-sec", help="The wall-clock limit the runner enforces, in "
                                        "seconds, replacing the task's; needs --soft-sec"),
    probe_only: bool = typer.Option(False, "--probe-only", help="Start the sandbox, probe it, stop: no agent and no "
                                    "package"),
    preflight_only: bool = typer.Option(False, "--preflight-only", help="Also run the arm's preflight after the "
                                        "probe: no agent and no package"),
    probe_oom: bool = typer.Option(False, "--probe-oom", help="With --probe-only, also run a memory hog in the "
                                   "sandbox and record what ended it"),
    job_name: str = typer.Option("hls-run", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Build a submission with one arm's agent in a sealed sandbox, package it and grade it, all in
    one allocation on the site's GPU partition: a rollout and its own grading. What the run lacks
    is made first, each step saying why: the task's data, its baseline credit and calibration when
    neither the store nor the checkout holds one that fits the task, and its reference answers.
    With --build-only it ends with the package and a record of the rollout; with --probe-only or
    --preflight-only it starts the sandbox, checks it and stops, and measures nothing. An arm that
    cannot be resolved, a rerun whose arm changed without --arm-change, or flags that cannot go
    together are a usage error."""
    _cap_thread_pools()
    from hls.arms import ArmError
    from hls.pipeline import commands, prepare

    configure_logging(verbose=False)
    if probe_only and preflight_only:
        raise typer.BadParameter("give at most one of --probe-only and --preflight-only")
    if probe_oom and not probe_only:
        raise typer.BadParameter("--probe-oom runs only with --probe-only", param_hint="--probe-oom")
    if build_only and (probe_only or preflight_only):
        raise typer.BadParameter("--build-only builds and packages; --probe-only and --preflight-only stop before the "
                                 "agent: give one", param_hint="--build-only")
    stop_after: Literal["probe", "preflight"] | None = (
        "probe" if probe_only else "preflight" if preflight_only else None)
    overrides: dict[str, str] = {}

    def prepared() -> int:
        # Click fills the module from the first leftover token, which can be an override flag.
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        if stop_after is not None:
            return 0
        try:
            return prepare.prepare(config=config, overrides=overrides, task_module=task_module, arm=arm,
                                   arm_change=arm_change, build_only=build_only, echo=typer.echo)
        except ArmError as e:
            raise typer.BadParameter(str(e), param_hint="--arm") from e

    def plan() -> Plan:
        try:
            return commands.plan_run(config=config, overrides=overrides, task_module=task_module, arm=arm,
                                     arm_change=arm_change, job_name=job_name, build_only=build_only,
                                     soft_sec=soft_sec, hard_sec=hard_sec, stop_after=stop_after,
                                     probe_oom=probe_oom, echo=typer.echo)
        except ArmError as e:
            raise typer.BadParameter(str(e), param_hint="--arm") from e

    _plan_and_start("run", plan, overrides, prepare=prepared)


@app.command("grade", short_help="Serve one artifact, drive it, verify and score the drive, and record the grading.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def grade_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multitenant_slo.task; overrides follow it"),
    rollout: str | None = typer.Option(None, "--rollout", help="Grade the package this rollout published "
                                       "(runs/<task>/<id>, from hls run)"),
    package: str | None = typer.Option(None, "--package", help="Grade the submission image this package warmer "
                                       "published (warmers/<task>/<id>, from hls package)"),
    measured: str | None = typer.Option(None, "--measured", help="Grade the task's measured image of this role, "
                                        "such as baseline"),
    rebuild: bool = typer.Option(False, "--artifact.rebuild", help="With --rollout, scan and package the rollout's "
                                 "source bundle again at this checkout and grade that"),
    image: str | None = typer.Option(None, "--artifact.image", help="Grade a published submission image by the "
                                     "digest of its file, sha256:<64 hex digits>; needs --from"),
    image_file: Path | None = typer.Option(None, "--from", help="The file of --artifact.image on this "
                                           "cluster's shared filesystem"),
    built_arm: str | None = typer.Option(None, "--built-arm", help="With --artifact.image, the arm that built it: "
                                         "its route secrets are then the contestant's"),
    job_name: str = typer.Option("hls-grade", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Serve one artifact, drive it with the task's driver, verify and score the drive, and record
    it as a grading, all in one allocation on the site's GPU partition. A --built-arm that cannot
    be resolved is a usage error."""
    _cap_thread_pools()
    from hls.arms import ArmError
    from hls.pipeline import commands

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        try:
            return commands.plan_grade(config=config, overrides=overrides, task_module=task_module,
                                       rollout=rollout, package=package, measured=measured, rebuild=rebuild,
                                       image=image, image_file=image_file, built_arm=built_arm,
                                       job_name=job_name, echo=typer.echo)
        except ArmError as e:
            raise typer.BadParameter(str(e), param_hint="--built-arm") from e

    _plan_and_start("grade", plan, overrides)


oracle_app = typer.Typer(add_completion=False, no_args_is_help=True,
                         help="A task's reference answers in the store.")
app.add_typer(oracle_app, name="oracle")


@oracle_app.command("fill", short_help="Compute a task's reference answers into the store.",
                    context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def oracle_fill_command(
    ctx: typer.Context,
    task_module: str = typer.Argument(..., help="The task module as a Python module name, such as "
                                      "tasks.multimodal_omni.task; overrides follow it"),
    chunk: int = typer.Option(96, "--chunk", min=1, help="Rows asked per chunk; each chunk is kept as it finishes"),
    job_name: str = typer.Option("hls-oracle-fill", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Compute a task's caching references' answers to its test split into the store ahead of
    its gradings, in one allocation on the site's GPU partition, so every grading serves them
    instead of computing them inside its own wall."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        if task_module.startswith("--"):
            raise ValueError(_override_first(task_module))
        overrides.update(_parse_dotted_flags(ctx.args))
        return commands.plan_oracle_fill(config=config, overrides=overrides, task_module=task_module, chunk=chunk,
                                         job_name=job_name, echo=typer.echo)

    _plan_and_start("oracle fill", plan, overrides)


@app.command("runner", hidden=True)
def runner_command(
    plan_dir: Path,
    allocation: int = typer.Option(..., help="Allocation index"),
    submission: int = typer.Option(..., help="Submission number"),
) -> None:
    """Run one allocation's stages. The batch script starts this; people rarely do."""
    from hls.substrate.runner import main as runner_main

    raise typer.Exit(runner_main([str(plan_dir), "--allocation", str(allocation), "--submission", str(submission)]))


_STORE_OPTION = typer.Option(None, "--store", help="Store root of the instance; defaults to site.store_root of --config")


def _look_after(command: str, config: Path, store: Path | None, act: Callable[[Path], int]) -> NoReturn:
    """Exit with the status *act* returns for the store's root (``commands.store_root``: *store*,
    or the one the hls.toml at *config* names). An invalid hls.toml, read for that root when no
    --store is given, is refused in one line on stderr naming *command*, with exit status 1 and
    no traceback, and so is a stored plan this version of hls cannot read (``PlanUnreadable``).
    Whatever else *act* raises is a fault and keeps its traceback."""
    from hls.pipeline import commands
    from hls.pipeline.plan import PlanUnreadable
    from hls.settings import SettingsError

    try:
        root = commands.store_root(config, store)
    except SettingsError as e:
        typer.echo(f"hls {command}: {e}", err=True)
        raise typer.Exit(1) from None
    try:
        status = act(root)
    except PlanUnreadable as e:
        typer.echo(f"hls {command}: {e}", err=True)
        raise typer.Exit(1) from None
    raise typer.Exit(status)


@app.command("resume")
def resume_command(instance: str, config: Path = typer.Option(Path("hls.toml")),
                   store: Path | None = _STORE_OPTION) -> None:
    """Take an instance over from an orchestrator that is gone, and drive it to its end."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    _look_after("resume", config, store,
                lambda root: commands.resume(instance, config=config, store=root, echo=typer.echo))


@app.command("status")
def status_command(instance: str, config: Path = typer.Option(Path("hls.toml")),
                   store: Path | None = _STORE_OPTION) -> None:
    """Show an instance's plan, lease, allocations, jobs, stages and attempts."""
    from hls.pipeline import commands

    _look_after("status", config, store,
                lambda root: commands.status(instance, config=config, store=root, echo=typer.echo))


@app.command("cancel")
def cancel_command(instance: str, config: Path = typer.Option(Path("hls.toml")),
                   store: Path | None = _STORE_OPTION) -> None:
    """Cancel an instance's recorded jobs, by id, and wait for the orchestrator to confirm."""
    _cap_thread_pools()
    from hls.pipeline import commands

    configure_logging(verbose=False)
    _look_after("cancel", config, store,
                lambda root: commands.cancel(instance, config=config, store=root, echo=typer.echo))


@app.command("export", short_help="Copy finished instances' publishable files out of the store.",
             context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def export_command(
    ctx: typer.Context,
    instances: list[str] = typer.Argument(..., help="Each finished instance to export, by its key in the store, such "
                                          "as runs/multitenant_slo/<rollout>; a rollout or a warmer brings the "
                                          "gradings under it"),
    to: Path = typer.Option(..., "--to", help="Absolute path of the directory to copy into: it must not exist yet, "
                            "and lies outside the store and the checkout"),
    oracle: bool = typer.Option(False, "--oracle", help="Also export the task's reference answers (oracle/<task>/), "
                                "scanned like every other file"),
    job_name: str = typer.Option("hls-export", help="Slurm job name of the allocation"),
    config: Path = typer.Option(Path("hls.toml")),
) -> None:
    """Copy finished instances' publishable files out of the store, once a secret scan of every
    file of each has cleared them, on the site's CPU partition when it has one."""
    _cap_thread_pools()
    from hls.pipeline import export

    configure_logging(verbose=False)
    overrides: dict[str, str] = {}

    def plan() -> Plan:
        named, flags = export.split_arguments([*instances, *ctx.args])
        overrides.update(_parse_dotted_flags(flags))
        return export.plan_export(config=config, overrides=overrides, instances=named, destination=to,
                                  oracle=oracle, job_name=job_name, echo=typer.echo)

    _plan_and_start("export", plan, overrides)


@app.command("follow")
def follow_command(instance: str, config: Path = typer.Option(Path("hls.toml")),
                   store: Path | None = _STORE_OPTION) -> None:
    """Print an instance's events as they are written, until it ends."""
    _cap_thread_pools()
    from hls.pipeline import commands
    from hls.pipeline.follow import follow

    configure_logging(verbose=False)
    _look_after("follow", config, store,
                lambda root: follow(instance, store=commands.open_store(config, root), echo=typer.echo))


if __name__ == "__main__":
    app()
