"""The provision role program, run by the provision stage as host-side harness role steps.

::

    python -m hls.provision.role resolve --task-module M [--overrides JSON] --out FILE --outputs-out FILE
    python -m hls.provision.role weights --repo R --revision SHA --source-out FILE --outputs-out FILE \
                                         --blobs-out FILE --digests-out FILE --out DIR
    python -m hls.provision.role rows    --task-module M [--overrides JSON] --digests-out FILE --out DIR
    python -m hls.provision.role tools   --task-module M [--overrides JSON] --variant full|dev --ids FILE \
                                         --digests-out FILE --out DIR

``--overrides`` is the plan's task overrides as JSON, which the role applies to the module's
``TASK`` as every role applies them (``_task``).

``resolve`` works out what a task needs: its model at a hub commit, the key of its rows, its
tenants and its tool bundle. The other three each fill one cache entry's private staging
directory, *DIR*, and write nothing outside it but their side files, each at a path outside it.
Once its files are written, each hashes and syncs every one, reading back what landed, and
reports the digests in ``--digests-out`` (``hash_tree``, ``write_digests``): the stage records
them in the entry's manifest and reads none of the files. The weights role
also writes the hub's file list to ``--source-out``, and the git blob id of each file the hub
keeps in git, from the bytes it wrote, to ``--blobs-out``, so the stage holds the download
against the hub's list without reading a file either. ``resolve``'s file and the weights role's
list are stage outputs at the top of the attempt's outputs: each role reports their sha256 and
size in ``--outputs-out`` (``report_outputs``) for the stage's manifest. The rows entry carries
its own row ids in ``ids.json``, so the dev bundle's key and build can be derived from a
published rows entry, hit or miss.

A failure that running the role again cannot change ends it with its own status and a last
``refused: <why>`` line on stderr: ``REFUSED_EXIT`` when the role refuses the task as it is
declared (``Refused``), ``HUB_REFUSED_EXIT`` when the hub answered what the role asked for
with 401, 403 or 404 (``hub_refusal``). Every other failure, a hub that does not answer or
answers with a 5xx among them, is an ordinary one.

The hub token is the ``HF_TOKEN`` variable of the role's environment, which the step
wrapper loads from the stage's 0600 env file; no option carries it. With no token granted,
``main`` sets ``HF_HUB_DISABLE_IMPLICIT_TOKEN`` before ``huggingface_hub`` is imported (the
library reads it once, at import), so no hub client in the process, the ``datasets`` loader
and a tool builder's downloads included, sends a token stored on the node under ``HF_HOME``;
the role's own hub calls pass ``token=False``. An ungranted secret reaches nothing."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from hls.provision.hf import SourceFile, git_blob_sha1, source_files
from hls.provision.model import (ModelRevision, Resolved, ToolSource, check_repo, check_revision, is_revision,
                                 rows_key)
from hls.provision.rows import declared_assets, dedup, entry_rows, provisioning_digest, unit_descriptions, write_rows
from hls.spec.agent import AssetsRefused
from hls.spec.datasource import PartitionError
from hls.spec.hub import declared_hub_revisions
from hls.spec.tools import StoreRefused
from hls.store.cache import hash_tree, write_digests
from hls.store.manifest import report_outputs
from hls.taskref import TaskRef, apply_task_overrides
from hls.util import remove_tree, validation_errors

#: File in a rows entry holding its dev and test row ids, ``{"dev": [...], "test": [...]}``.
IDS_FILE = "ids.json"

#: Exit status of a role that refuses the task as it is declared (``Refused``). Running it
#: again changes nothing, so the provision stage does not retry it. Its last line on stderr
#: is then ``refused: <why>``.
REFUSED_EXIT = 3
#: Exit status of a role whose hub request the hub answered with 401, 403 or 404: no such
#: repo, revision or file, or a gated or private repo without a grant that admits it. The
#: stage does not retry it either; its last line on stderr is ``refused: <why>`` as well.
HUB_REFUSED_EXIT = 4
#: HTTP statuses of a hub answer that refuses what was asked, whoever asks again.
HUB_REFUSALS = frozenset({401, 403, 404})

#: The hub client's switch for sending a token it finds on the node (``HF_HOME/token``).
_NO_IMPLICIT_TOKEN = "HF_HUB_DISABLE_IMPLICIT_TOKEN"

#: Directory inside a tool bundle's staging directory where the builder works.
_BUILD_DIR = ".build"


def hub_token() -> str | bool:
    """The hub token from the environment, or False when none was granted."""
    return os.environ.get("HF_TOKEN") or False


def refuse_implicit_token() -> None:
    """Stop every hub client in this process from sending a token stored on the node. The
    hub library reads its switch once, when it is imported, so a process that already holds
    the library without the switch cannot be made safe, and this raises."""
    os.environ[_NO_IMPLICIT_TOKEN] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and not constants.HF_HUB_DISABLE_IMPLICIT_TOKEN:
        raise RuntimeError("huggingface_hub was already imported before the role could switch off its "
                           "implicit token, so a token stored on the node would still be sent")


class Refused(ValueError):
    """A role refuses the task as it is declared, so running it again changes nothing.
    - ``resolve``: the module defines no ``TASK`` or its task does not validate, a repo id is
      malformed, the model, a dataset or a tool bundle reads a hub source that is not pinned to
      a commit, a dataset or tool bundle does not declare what it reads, or tool assets come
      without data.
    - ``rows``: no tenant carries a data source, or the pinned corpus cannot be partitioned as
      declared (``PartitionError``).
    - ``tools``: the task declares no tool assets, or its builder refuses the bundle
      (``AssetsRefused``) or writes none.

    Any other error, a ``ValueError`` such as a hub reply the client cannot parse included, is
    an ordinary failure."""


def hub_refusal(error: BaseException) -> str | None:
    """Why the hub refused what was asked, when *error* comes from a hub answer of 401, 403 or
    404 (``HUB_REFUSALS``); None for any other error, one where the hub never answered (a
    connection error, a timeout, a file in no local cache while offline) or answered with
    another status (a 5xx, a 429) among them. The error's causes are searched (``raise ...
    from``), which is how the hub client and ``datasets`` wrap an answer; the error it was
    raised while handling is not, since a fallback that timed out after a 404 is no answer to
    what was asked. ``datasets`` raises its own DatasetNotFoundError for exactly these answers,
    and re-raises it without the hub client's error, so that error counts as one too. Neither
    library is imported here: an error can only be one of theirs when the library is loaded
    already."""
    errors = sys.modules.get("huggingface_hub.errors")
    http_error = getattr(errors, "HfHubHTTPError", None)
    dataset_errors = sys.modules.get("datasets.exceptions")
    not_found = getattr(dataset_errors, "DatasetNotFoundError", None)
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if http_error is not None and isinstance(current, http_error):
            status = getattr(getattr(current, "response", None), "status_code", None)
            if status in HUB_REFUSALS:
                return f"the hub answered {status}: {current}"
        if not_found is not None and isinstance(current, not_found):
            return f"the hub has no such dataset or revision, or it is gated: {current}"
        current = current.__cause__
    return None


def load_task(module: str):
    """The ``TASK`` of the task module *module*. Raise Refused when the module defines none,
    or its task does not validate as it is built at import."""
    try:
        loaded = importlib.import_module(module)
    except ValidationError as e:
        raise Refused(f"task module {module} does not validate: {e.title}: {validation_errors(e)}") from None
    task = getattr(loaded, "TASK", None)
    if task is None:
        raise Refused(f"task module {module} defines no TASK")
    return task


def _task(args: argparse.Namespace):
    """The task the command provisions: the ``TASK`` of ``--task-module`` (``load_task``) with the
    plan's task overrides, ``--overrides``, read as the plan's task reference holds them
    (``TaskRef``) and applied as every other role applies them (``apply_task_overrides``). Raise
    Refused as ``load_task`` does, and for overrides that are no task reference's, name no field
    of the task or give a field a value the task refuses."""
    task = load_task(args.task_module)
    try:
        ref = TaskRef(module=args.task_module, overrides=json.loads(args.overrides))
        return apply_task_overrides(task, ref.overrides)
    except ValidationError as e:
        raise Refused(f"task module {args.task_module}: its overrides do not apply: {e.title}: "
                      f"{validation_errors(e)}") from None
    except ValueError as e:
        raise Refused(f"task module {args.task_module}: its overrides do not apply: {e}") from e


def _check_source(repo: str, revision: str, *, what: str, then: str) -> None:
    """Refuse hub source *repo* at *revision*, read by what *what* names, unless *repo* is a repo
    id and *revision* a commit id. *then* says what an unpinned source would let happen."""
    try:
        check_repo(repo)
    except ValueError as e:
        raise Refused(f"{what}: {e}") from e
    if not is_revision(revision):
        raise Refused(f"{what}: {repo}: revision {revision!r} is not a commit id, so {then}")


def _check_pinned(obj, *, reads: str, what: str, then: str) -> None:
    """Refuse *obj*, whose method *reads* reads the hub and which is called *what*, unless it
    declares every hub source it reads (``hls.spec.hub``) and each is a repo id at a commit.
    *then* says what an unpinned source would let happen."""
    try:
        sources = declared_hub_revisions(obj, reads=reads)
    except ValueError as e:
        raise Refused(f"{what}: {e}") from e
    for repo, revision in sources:
        _check_source(repo, revision, what=what, then=then)


def _check_rows_are_pinned(units: list[dict]) -> None:
    """Refuse a data source whose dataset or ``dev_dataset`` reads a hub source that is not
    pinned to a commit, or does not declare what it reads; a composite's arms are its sources.
    A rows key stands for the bytes under it, so every hub revision they are read from must be
    one commit."""
    for u in units:
        for label, dataset in (("dataset", u["data_source"].dataset), ("dev_dataset", u["dev_dataset"])):
            if dataset is not None:
                _check_pinned(dataset, reads="corpus", what=f"tenant {u['unit_id']}: {label} {dataset.corpus_id()}",
                              then="the rows could change under their key")


def _check_tools_are_pinned(assets) -> None:
    """Refuse tool assets that read a hub source not pinned to a commit, or do not declare
    what they read: both bundles' keys stand for the bytes built from those sources."""
    _check_pinned(assets, reads="build", what=f"tool assets {assets.name}",
                  then="the bundles could change under their key")


def resolve(task) -> Resolved:
    """What *task* needs provisioned, read off its declaration alone: nothing is asked of the
    hub. Raise Refused for a task ``Refused`` lists: one that declares tool assets but no data
    source (its dev bundle is defined by the dev partition), one whose model, datasets or tool
    assets read a hub source not pinned to a commit, or whose datasets or tool assets do not
    declare what they read, and one whose resolved parts do not validate."""
    try:
        # It reads only the task's declaration.
        units = unit_descriptions(task)
    except ValueError as e:
        raise Refused(str(e)) from e
    assets = declared_assets(task)
    if assets is not None and not units:
        raise Refused("task declares ToolAssets but no tenant carries a data source; the dev bundle is "
                      "defined by the dev partition, so there is nothing to build it from")
    _check_rows_are_pinned(units)
    if assets is not None:
        _check_tools_are_pinned(assets)
    model = task.model
    _check_source(model.base_checkpoint, model.revision, what="model",
                  then="the weights could change under their key")
    weights = ModelRevision(repo=model.base_checkpoint, revision=model.revision)
    identity = assets.identity() if assets is not None else None
    try:
        return Resolved(
            weights=weights,
            rows_key=rows_key(provisioning_digest(units)) if units else None,
            tenants=[u["unit_id"] for u in units],
            tools=ToolSource(name=assets.name, identity=identity) if assets is not None else None,
        )
    except ValidationError as e:
        raise Refused(f"the task's provisioning does not validate: {e.title}: {validation_errors(e)}") from None


def _empty_dir(path: Path) -> Path:
    if not path.is_dir() or any(path.iterdir()):
        raise ValueError(f"{path} is not an empty directory")
    return path


def _outside(side: Path, out: Path) -> Path:
    if side.resolve().is_relative_to(out.resolve()):
        raise ValueError(f"{side} lies inside the entry directory {out}")
    return side


def fetch_weights(repo: str, revision: str, out: Path, source_out: Path, *,
                  token: str | bool) -> dict[str, SourceFile]:
    """Download *repo* at commit *revision* into the empty directory *out*, drop the hub
    client's own ``.cache`` metadata from it, and write the hub's file list to
    *source_out*, which lies outside *out*; return that list."""
    from huggingface_hub import snapshot_download

    _empty_dir(out)
    _outside(source_out, out)
    check_revision(revision)
    snapshot_download(repo_id=check_repo(repo), revision=revision, local_dir=str(out), token=token)
    # snapshot_download(local_dir=...) keeps its download bookkeeping in <out>/.cache; it is
    # not part of the repo, and the entry must hold exactly the hub's files.
    remove_tree(out / ".cache", ignore_errors=True)
    listing = source_files(repo, revision, token=token)
    source_out.write_text(json.dumps({p: f.model_dump() for p, f in sorted(listing.items())}, indent=1))
    return listing


def _corpus_for(ds):
    """A data source's corpus. ``datasets`` reads ``HF_TOKEN`` from the environment."""
    return ds.dataset.corpus()


def _corpus_of(dataset):
    """An explicit dataset's corpus, such as a data source's ``dev_dataset``."""
    return dataset.corpus()


def write_rows_entry(task, out: Path) -> None:
    """Write every tenant's rows into the empty directory *out* (``write_rows``), and the dev
    and test row ids, deduplicated, to ``ids.json`` beside them. No role mounts the entry's
    root, only its ``tenants/<id>/<split>`` directories, so the test ids never reach a role
    that may not see the test rows. Raise Refused for a task with no data source, or whose
    pinned corpus cannot be partitioned as declared."""
    _empty_dir(out)
    try:
        units = unit_descriptions(task)
    except ValueError as e:
        raise Refused(str(e)) from e
    if not units:
        raise Refused("no tenant of the task carries a data source")
    try:
        dev_ids, test_ids = write_rows(units, out, corpus_for=_corpus_for, corpus_of=_corpus_of)
    except PartitionError as e:
        raise Refused(str(e)) from e
    (out / IDS_FILE).write_text(json.dumps({"dev": dedup(dev_ids), "test": dedup(test_ids)}))


def build_tools(task, variant: str, ids: dict[str, list[str]], out: Path, *, token: str | bool) -> None:
    """Build the *variant* of the task's tool bundle into the empty directory *out* from the
    partition's row ids *ids*.

    The builder gets a private directory inside *out*, so whatever it writes beside its
    ``out_dir`` (a builder may assemble ``<out_dir>.partial`` and rename it into place) lands
    inside *out* too; the finished bundle's contents are then moved up into *out*. Raise
    Refused for a task with no tool assets, an unknown variant, a bundle the builder refuses
    (``AssetsRefused``) and a builder that writes none."""
    _empty_dir(out)
    assets = declared_assets(task)
    if assets is None:
        raise Refused("the task declares no tool assets")
    if variant not in ("full", "dev"):
        raise Refused(f"unknown bundle variant {variant!r}")
    work = out / _BUILD_DIR
    bundle = work / "bundle"
    work.mkdir()
    try:
        try:
            assets.build(variant, dev_row_ids=ids["dev"], test_row_ids=ids["test"], out_dir=bundle,
                         hf_token=token if isinstance(token, str) else None)
        except AssetsRefused as e:
            raise Refused(str(e)) from e
        if not bundle.is_dir():
            raise Refused(f"the {variant} bundle builder wrote no bundle at its out_dir")
        for child in bundle.iterdir():
            os.replace(child, out / child.name)
    finally:
        remove_tree(work, ignore_errors=True)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.provision.role")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--task-module", required=True)
    r.add_argument("--overrides", default="{}")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--outputs-out", type=Path, required=True)
    w = sub.add_parser("weights")
    w.add_argument("--repo", required=True)
    w.add_argument("--revision", required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--source-out", type=Path, required=True)
    w.add_argument("--outputs-out", type=Path, required=True)
    w.add_argument("--blobs-out", type=Path, required=True)
    w.add_argument("--digests-out", type=Path, required=True)
    rows = sub.add_parser("rows")
    rows.add_argument("--task-module", required=True)
    rows.add_argument("--overrides", default="{}")
    rows.add_argument("--out", type=Path, required=True)
    rows.add_argument("--digests-out", type=Path, required=True)
    t = sub.add_parser("tools")
    t.add_argument("--task-module", required=True)
    t.add_argument("--overrides", default="{}")
    t.add_argument("--variant", choices=("full", "dev"), required=True)
    t.add_argument("--ids", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--digests-out", type=Path, required=True)
    s = sub.add_parser("stores")
    s.add_argument("--task-module", required=True)
    s.add_argument("--overrides", default="{}")
    s.add_argument("--stores", required=True, help="the site's sandbox stores the task names, as JSON: name to "
                                                   "directory")
    s.add_argument("--rows", type=Path, default=None, help="the task's published rows entry, whose rows the "
                                                           "stores are provisioned for")
    return p.parse_args(argv)


def _refused(why: str) -> None:
    """Print *why* as the role's last line on stderr, ``refused: <why>`` on one line."""
    print(f"refused: {' '.join(why.split())}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    token = hub_token()
    if not token:
        refuse_implicit_token()
    try:
        _run(args, token)
    except Refused as e:
        _refused(str(e))
        return REFUSED_EXIT
    except Exception as e:
        why = hub_refusal(e)
        if why is None:
            raise
        _refused(why)
        return HUB_REFUSED_EXIT
    return 0


def _run(args: argparse.Namespace, token: str | bool) -> None:
    """Do the role's *args* command with the hub *token*. A fill command reports the digest of
    every file it wrote once it is done (``hash_tree``)."""
    if args.command == "resolve":
        resolved = resolve(_task(args))
        args.out.write_text(resolved.model_dump_json(indent=2))
        report_outputs(args.outputs_out, args.out.parent, args.out)
        return
    if args.command == "stores":
        # What the task's tool runner needs in the site's sandbox stores, made in place: the
        # stores are the site's directories, not cache entries.
        task = _task(args)
        runner = task.agent.tool_runner if task.agent is not None else None
        if runner is None:
            raise Refused(f"{args.task_module} drives no tool runner, so it has no sandbox stores to provision")
        stores = {name: Path(path) for name, path in json.loads(args.stores).items()}
        rows = entry_rows(args.rows) if args.rows is not None else []
        try:
            done = runner.provision_stores(stores, rows)
        except StoreRefused as e:
            raise Refused(str(e)) from e
        for line in done:
            print(line)
        return
    reports = [args.digests_out, *([args.blobs_out, args.outputs_out] if args.command == "weights" else [])]
    for report in reports:
        _outside(report, args.out)
    if args.command == "weights":
        listing = fetch_weights(args.repo, args.revision, args.out, args.source_out, token=token)
        report_outputs(args.outputs_out, args.source_out.parent, args.source_out)
    elif args.command == "rows":
        write_rows_entry(_task(args), args.out)
    else:
        ids = json.loads(args.ids.read_text())
        build_tools(_task(args), args.variant, ids, args.out, token=token)
    digests = hash_tree(args.out)
    if args.command == "weights":
        # Only the files hash_tree found, plain files, which the stage's scan then admits.
        blobs = {path: git_blob_sha1(args.out / path) for path in digests
                 if path in listing and listing[path].git_sha1 is not None}
        args.blobs_out.write_text(json.dumps(blobs, indent=1, sort_keys=True))
    write_digests(args.digests_out, digests)


if __name__ == "__main__":
    sys.exit(main())
