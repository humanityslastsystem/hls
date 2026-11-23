"""The images stage: the images a run needs, built into the cache, each skipped on a hit.

Every image is a cache entry ``images/<key>``. An image's key comes from its recipe chain as
the substrate's image runtime renders it (``ImageRuntime.recipe_key``: the ``.def``'s for
Apptainer, the Dockerfile's for Docker), so every key is known before anything builds. A recipe
that cannot be keyed, or a source build whose build backend no lock below it installs
(``check_source_builds``), is refused before anything starts. Images go in order, each after the
image it builds on. Image files move only through roles, never through the runner. On a miss,
under the entry's lock:
- the image renders its recipe (``ImageRuntime.write_recipe``) against its lock files and, for a
  layer or a cap, a copy of its parent's image file (only now: a hit never needs its parent),
  and builds it with the runtime's build role (``ImageRuntime.build_role``: ``apptainer build``,
  or BuildKit) as a host-side harness role on node-local disk. That role copies the parent's
  file from its entry, pinned to the ENTRY.json digest this attempt recorded for the parent,
  into its own directory: the one read of the store's file. It refuses the copy unless it has
  the sha256 the entry records, builds from it, removes it, and writes the new file's sha256
  when the build is done. What the runtime records of the build (``ImageRuntime.build_record``:
  the mode apptainer reports, or the archive's digests) is recorded;
- the new image file is extracted on the node (``StageContext.pin_file``, by the digest the role
  reported), and the inventory script runs in the extracted image, as a harness container with
  the script bound read-only; its output becomes ``inventory.json``. The inventory must be
  complete, and a base, layer or cap that holds a package of the harness deny list, or that
  moved a version its base holds, is refused: an unsealed image may not hand the agent what the
  index denies, and nothing may replace the base's torch. A server layer, which only servers run
  on, may hold a denied serving framework, and so may the registry image one builds on, a
  vendor's serving stack as pulled; only another server layer builds on either. An agent cap
  that lacks a tool, a Python package or an Ubuntu package the brief says is installed is
  refused too: the agent reads the brief's list as what it has. A probe that timed out may
  succeed on a retry, so a build with one publishes nothing and is retried;
- a role copies the image file into the entry, keeping its modification time, syncs it and
  reports its sha256, which the entry records, and the entry publishes only if that is the
  digest of the file that was extracted and inventoried. The image file then leaves its role's
  directory. The stage writes the entry's small files itself, synced, and reports each by the
  bytes it wrote (``write_file``), so it reads no file of the entry either.

The entry holds the runtime's image file (``ImageRuntime.image_file``), its recipe
(``ImageRuntime.recipe_file``), ``inventory.json`` and ``meta.json`` (``ImageMeta``). Each
image's file digest goes into the stage's manifest
(``record_image``), and the output ``images.json`` names every image's key, entry digest and
file digest: a consumer pins an image with ``StageContext.node_image(key, expect=<entry
digest>)``, which extracts it straight from the entry. The stage releases each tree it
extracted once the image's inventory is judged (``StageContext.release``): nothing reads it after,
the entry publishing the image file and a child building on that file. No later stage reads it
either: a stage that runs an image pins it in its own context, which extracts it anew.

Node-local disk. The image in hand holds its build: its unpacked image in apptainer's temporary
directory, the image file it packs and, for a layer or a cap, a copy of its parent's file, which
the build removes when it ends, as the stage then removes apptainer's cache, where a base's
pulled layers are. The extraction then holds a copy of the file until it is extracted, and the
extracted tree until it is released; the build's file goes once the entry is published. An entry the stage finds puts nothing on
the node: it is never extracted, and a child built on it copies its file for the length of the
build. One image is in hand at a time, so the stage's peak is an extraction, one build's room and
one tree, however many entries it fills, and nothing when it fills nothing (``disk_gib``). The
plan counts the fills from the cache as it stood when the plan was made (``held_keys``,
``ImagesParams.found``). At an image's turn the stage finds its entry, or must fill it, under the
entry's lock, so another stage may still publish it until then: when it must fill an entry and
its declared disk holds no fill, as when the plan found every entry, it ends with an infra fault
naming the entry, which the plan found and the cache no longer holds, before it fills anything.
That fault is not retried: a retry runs the same plan, which declares the same disk, while a new
plan counts the fills.

All or nothing per image: a failed role, an unreadable inventory, a refused image or a
digest that does not match publishes nothing, and the stage fails with an infra fault. A
failed build, extraction or inventory role may be retried, and so may an inventory whose only
failed probes timed out; a refused recipe or image, an inventory that fails the same way every
run or a corrupt entry may not, since a retry would build or read the same thing. A wait for another stage filling the same entry ends
at the stage's deadline, or when the allocation stops, as an infra fault that may be retried.

A registry login is a user name, the plain setting ``site.registry_username``, and a token, the
one secret, granted to the stage as ``REGISTRY_TOKEN_GRANT``: both or neither, which the planner
checks before a job queues (``registry_login_problem``). It reaches only
the build of a base or of a registry image, the steps that pull from a registry, the user name in the role's declared
environment and the token through its env file; the build's command
(``build_argv(..., login=True)``) hands them to apptainer under its own names."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, \
    model_validator

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.recipes import inventory as inv
from hls.recipes.inventory import Inventory, InventoryError, parse_inventory
from hls.recipes.model import (BaseImage, CapImage, ImageDecl, ImageName, LayerImage, RegistryImage, order,
                               parent_of, registry_of)
from hls.recipes.pins import LOCKS_DIR, UV
from hls.recipes.render import TASKS_DIR, check_environments, check_source_builds
from hls.settings import SiteSettings
from hls.spec.resources import ResourceShape
from hls.stages.provision import role_shape
from hls.store.base import check_segment
from hls.store.cache import (CacheCorrupt, CacheEntry, ContentCache, FileRecord, WaitEnded, entry_sha256,
                             write_file)
from hls.store.digest import DIGEST_MISMATCH_EXIT, reported_sha256
from hls.store.lock import Holder
from hls.substrate.apptainer.images import REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR, BuildMode
from hls.substrate.container import ContainerSpec, ImageRuntime, Mount, PinnedImage
from hls.substrate.factory import RecipeKey, substrate
from hls.substrate.sandbox import IMAGE_FILE
from hls.util import remove_tree

#: Seconds between looks for a published entry while another stage fills it.
POLL_SEC = 5.0
#: Node-local disk the image in hand holds beside its extracted tree: a build's unpacked image, the
#: image file it packs and, for a base, apptainer's cache of the layers it pulled, which goes when
#: the build ends; while ``%post`` installs a pip lock, uv's cache and temporary files beside the
#: venv; a layer's or a cap's build holds a copy of its parent's file instead of layers. The
#: file stays until the entry is published. The largest extracted tree of a task's images is
#: 20.98 GiB and the largest image file 7.93 GiB, a server layer on a vendor's image, and a base's
#: pip step peaks 5.43 GiB over the venv it leaves. So a build holds at most
#: 20.98 + 5.43 + 7.93 = 34.34 GiB during ``%post`` and 20.98 + 7.93 + 7.93 = 36.84 GiB while it
#: packs the file, a child's parent copy counted; a base or a registry image holds its pulled
#: layers in that copy's place, 8.11 GiB for the vendor image, whose pull holds
#: 8.11 + 20.00 + 7.44 = 35.55 GiB; and the extraction that follows, the file, its copy and the
#: tree, 7.93 + 7.93 + 20.98 = 36.84 GiB of this room and ``KEPT_DISK_GIB``'s.
BUILD_DISK_GIB = 40
#: Node-local disk one extracted image holds, whichever stage pins it: its tree, and, while it is
#: extracted, the extraction's copy of its file; at most 20.98 + 7.93 = 28.91 GiB for a task's
#: images (``BUILD_DISK_GIB``).
KEPT_DISK_GIB = 40
#: The step of a copy role: copy the image file (argument 1) to argument 2 keeping its
#: modification time, which the entry records as every fill records its producer's times, sync
#: the copy to disk and write the sha256 of the copy, the bytes that landed, to argument 3.
COPY_SHELL = 'cp --preserve=timestamps -- "$1" "$2" && sync -- "$2" && sha256sum < "$2" > "$3"'

__all__ = ["BUILD_DISK_GIB", "IMAGE_FILE", "KEPT_DISK_GIB", "POLL_SEC", "HeldKeys", "ImageMeta",
           "ImagesParams", "ImagesStage", "disk_gib", "fills", "held_keys", "image_keys"]

#: Those of the given cache keys whose entries a cache holds: ``held_keys`` bound to the cache.
HeldKeys = Callable[[Collection[str]], Collection[str]]


class ImagesParams(BaseModel):
    """What the images stage builds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task as the image catalog names it, its directory under ``tasks/``: what ``hls images
    #: build`` takes. The plan's task is its TaskSpec id, which can differ (``multimodal-asr``
    #: lives in ``tasks/multimodal_asr``).
    catalog_task: Annotated[str, AfterValidator(check_segment)]
    #: The images, in any order; each one's parent must be among them.
    images: list[ImageDecl]
    #: The cache keys of the images whose entries the cache held when the plan was made
    #: (``held_keys``), each once: the stage's declared disk counts only the entries it fills
    #: beside these (``disk_gib``).
    found: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.images:
            raise ValueError("the images stage needs at least one image")
        twice = sorted({key for key in self.found if self.found.count(key) > 1})
        if twice:
            raise ValueError(f"found lists {', '.join(twice)} twice")
        by_name = {d.name: d for d in order(self.images)}
        for decl in by_name.values():
            root = decl
            while (parent := parent_of(root)) is not None:
                root = by_name[parent]
                if (isinstance(root, LayerImage) and root.server
                        and not (isinstance(decl, LayerImage) and decl.server)):
                    raise ValueError(f"image {decl.name} builds on server image {root.name}, which may hold "
                                     "what the deny list names: only another server image builds on one")
            if (root is not decl and isinstance(root, RegistryImage)
                    and not (isinstance(decl, LayerImage) and decl.server)):
                raise ValueError(f"image {decl.name} builds on registry image {root.name}: only a server layer "
                                 "builds on a vendor's image, which may hold what the deny list names")
        check_environments(by_name.values())
        return self


class ImageMeta(BaseModel):
    """An images entry's ``meta.json``: which image it holds and how it was made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The image's name.
    name: ImageName
    #: Its declaration's kind.
    kind: Literal["base", "layer", "cap", "registry"]
    #: Its cache key.
    key: str
    #: The image it builds on; None for a base or a registry image.
    parent: ImageName | None
    #: That image's cache key; None without a parent.
    parent_key: str | None
    #: How apptainer ran the build; None for an image another runtime built. It is not part of
    #: the key.
    build_mode: BuildMode | None
    #: A Docker image archive's identity as its build recorded it: ``manifest_digest``, which
    #: identifies the image in every image store, ``config_digest``, the image ID a classic store
    #: gives it, and ``buildkit_version``. None, and then left out of ``meta.json``, for an image
    #: file of another kind.
    oci: dict[str, str] | None = None

    @model_serializer(mode="wrap")
    def _without_absent_oci(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if self.oci is None:
            data.pop("oci", None)
        return data


def image_keys(decls: Sequence[ImageDecl], key: RecipeKey) -> dict[str, str]:
    """Each image's cache key by name, parents first: what the stage looks up and fills, and
    what a plan that names one of these images must use. *key* keys one declaration as the
    substrate's image runtime builds it: the stage passes its image runtime's
    (``ImageRuntime.recipe_key``), a plan its site's substrate's (``Substrate.recipe_key``), which is
    the same. A built image's key needs only its parent's key, never its parent's build.
    Raise ValueError naming the image whose recipe cannot be keyed."""
    keys: dict[str, str] = {}
    ordered = order(decls)
    by_name = {d.name: d for d in ordered}
    for d in ordered:
        parent = parent_of(d)
        try:
            keys[d.name] = key(d, parent_key=keys[parent] if parent else None, locks_dir=LOCKS_DIR, uv=UV,
                               tasks_dir=TASKS_DIR, registry=registry_of(d, by_name))
        except (ValueError, OSError) as e:
            raise ValueError(f"the recipe of image {d.name} cannot be keyed: {e}") from e
    return keys


def registry_login_problem(stage: str, user: str | None, grants: Collection[str]) -> str | None:
    """Why images stage *stage*, granted *grants* (variable names), cannot log in to a registry
    under the user name *user* (``site.registry_username``), or None: a login is the user name and
    the token, granted as ``REGISTRY_TOKEN_GRANT``, both or neither. The planner asks before a job
    queues, since a stage holding one half fails whether or not it builds anything."""
    token = REGISTRY_TOKEN_GRANT in grants
    if user is not None and not token:
        return (f"site.registry_username is set but stage {stage} is not granted {REGISTRY_TOKEN_GRANT}; grant the "
                "registry token in the secrets file, or unset the user name")
    if token and user is None:
        return (f"stage {stage} is granted {REGISTRY_TOKEN_GRANT} but site.registry_username is not set; set the "
                "registry user name in hls.toml, or grant no token")
    return None


def _entries(n: int) -> str:
    return f"{n} entry" if n == 1 else f"{n} entries"


def _stray(keys: Mapping[str, str], found: Collection[str]) -> str | None:
    """What of *found* is no key of the images keyed *keys*, as a refusal; None when it all is."""
    stray = sorted(set(found) - set(keys.values()))
    return f"found {', '.join(stray)}, which is no image's key" if stray else None


def fills(keys: Mapping[str, str], found: Collection[str]) -> list[str]:
    """The entries an images stage of the images keyed *keys*, by name (``image_keys``), fills
    when it finds the entries *found*: each other key once, in the images' order."""
    return list(dict.fromkeys(key for key in keys.values() if key not in found))


def disk_gib(entries: int) -> int:
    """The node-local disk an images stage holds at its peak when it fills *entries* entries: an
    extraction, one build's room and one tree, since the stage fills one entry at a time and
    releases each tree once the image's inventory is judged; and no tree is kept for a later
    stage, which pins the images it runs in its own context. None when it fills nothing, since an
    entry it finds puts nothing on the node."""
    return BUILD_DISK_GIB + KEPT_DISK_GIB if entries else 0


def held_keys(cache: ContentCache, keys: Collection[str], *, image_file: str) -> set[str]:
    """Those of the images cache keys *keys* whose entries *cache* holds now: what an images stage
    of those images would find, which the planner reads to size the stage's disk and time. Raise
    ValueError for an entry that is not a published one (``CacheCorrupt``) or that holds no image
    file *image_file* (``Substrate.image_file``), which a stage would refuse too."""
    held: set[str] = set()
    for key in keys:
        try:
            entry = cache.lookup("images", key)
        except CacheCorrupt as e:
            raise ValueError(f"cache entry images/{key}: {e}") from e
        if entry is None:
            continue
        if not any(f.path == image_file for f in entry.manifest.files):
            raise ValueError(f"cache entry images/{key} holds no {image_file}")
        held.add(key)
    return held


class _Images:
    """One attempt's builds and inventories."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.cache is None or ctx.scratch is None:
            raise RuntimeError("the images stage needs the store's cache and node-local scratch")
        self.ctx = ctx
        self.params: ImagesParams = ctx.params
        #: Every image, parents first.
        self.decls = order(self.params.images)
        #: Image name to its declaration.
        self.by_name = {d.name: d for d in self.decls}
        try:
            check_source_builds(self.decls, LOCKS_DIR, TASKS_DIR)
        except ValueError as e:
            raise StageFault.infra(f"the image recipes are refused: {e}", retryable=False) from e
        if ctx.images is None:
            raise RuntimeError("the images stage needs an image runtime")
        #: The substrate's image runtime: its image file, recipes, build and pin roles.
        self.images: ImageRuntime = ctx.images
        try:
            keys = image_keys(self.decls, self.images.recipe_key)
        except ValueError as e:
            raise StageFault.infra(str(e), retryable=False) from e
        #: Image name to its cache key, all computed before anything builds.
        self.keys = keys
        stray = _stray(keys, self.params.found)
        if stray is not None:
            raise StageFault.infra(f"the plan {stray} now: the image recipes changed since the plan was made",
                                   retryable=False)
        #: Image name to the ENTRY.json digest of its entry, once this attempt found or filled it.
        self.entries: dict[str, str] = {}
        # The plan holds both halves of the login or neither (``registry_login_problem``).
        user = ctx.plan.settings.site.registry_username
        #: The declared variable that gives every base build the registry user name; empty when
        #: the site has no registry login.
        self.login_env: dict[str, str] = {REGISTRY_USER_VAR: user} if user is not None else {}
        #: The cache's lock holder for this attempt's fills.
        self.holder = Holder.here(f"images {ctx.plan.instance}", job_id=ctx.job_id)
        #: Each role's shape: the stage's, less the runner's memory, and no disk.
        self.shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)

    def _wait(self) -> None:
        """Wait for the roles and raise their fault. When a build refused its parent's image
        file in the store, whose sha256 is not the one its entry records, the fault is that
        refusal, which a retry would meet again (infra, not retried). A copy that went wrong on
        the way is its role's own fault, which is retried."""
        outcomes = self.ctx.wait_roles()
        fault = stage_fault(outcomes)
        if fault is None:
            return
        for o in outcomes:
            refused = self.ctx.refusal(o.name, outcomes, (DIGEST_MISMATCH_EXIT,))
            if refused is not None:
                raise StageFault.infra(refused.reason, retryable=False, role=o.name)
        raise StageFault(fault)

    def _base_of(self, decl: ImageDecl) -> ImageDecl:
        while (parent := parent_of(decl)) is not None:
            decl = self.by_name[parent]
        return decl

    def _inventory(self, decl: ImageDecl, image: PinnedImage) -> Inventory:
        """Run the inventory script in the extracted *image* and parse what it printed, in full."""
        name = f"inventory-{decl.name}"
        spec = ContainerSpec(image=image, command=("/bin/sh", inv.INVENTORY_TARGET),
                             workdir=self.ctx.role_dir(name) / "wd",
                             mounts=(Mount(source=inv.SCRIPT, target=inv.INVENTORY_TARGET, read_only=True),))
        spec.workdir.mkdir(parents=True, exist_ok=True)
        self.ctx.start_container(name, spec, self.shape, owner="harness")
        self._wait()
        out = self.ctx.attempt_dir / "logs" / f"{name}.out"
        try:
            text = out.read_text(errors="replace")
        except OSError as e:
            raise StageFault.infra(f"the inventory of {decl.name} could not be read: {e}", retryable=True) from e
        try:
            inventory = parse_inventory(text)
        except InventoryError as e:
            raise StageFault.infra(f"the inventory of {decl.name} is refused: {e}", retryable=e.transient,
                                   inventory=str(e)) from e
        return inventory

    def _judge(self, decl: ImageDecl, inventory: Inventory) -> None:
        """Refuse an image hls built that holds a denied package, unless it is a server layer or
        the registry image one builds on, or moved a held version, and an agent cap that lacks a
        tool, a Python package or an Ubuntu package the brief says is installed. A base or a layer
        is not held to the brief, which describes the sandbox."""
        server = isinstance(decl, RegistryImage) or (isinstance(decl, LayerImage) and decl.server)
        problems = [] if server else [f"holds {name}, which the harness deny list names"
                                      for name in inv.denied_in(inventory, DENIED_PACKAGES)]
        base = self._base_of(decl)
        # A base with no pip lock holds no version.
        if isinstance(base, BaseImage) and base.constraints:
            problems += inv.version_mismatches(inventory, base.constraints)
        if isinstance(decl, CapImage) and decl.cap == "agent":
            promised = (("tools", inv.missing_tools(inventory, inv.BRIEF_TOOLS)),
                        ("Python packages", inv.missing_packages(inventory, inv.BRIEF_PACKAGES)),
                        ("Ubuntu packages", inv.missing_apt(inventory, inv.BRIEF_APT)))
            problems += [f"lacks the {what} {', '.join(missing)}, which the brief says are installed"
                         for what, missing in promised if missing]
        if problems:
            raise StageFault.infra(f"image {decl.name} is refused: " + "; ".join(problems), retryable=False)

    def _copy(self, name: str, source: Path, dest: Path) -> str:
        """Copy *source* to *dest* in harness role *name* (``COPY_SHELL``); return the sha256 the
        role reported for the copy."""
        digest = self.ctx.role_dir(name) / "copy.sha256"
        self.ctx.start_role(name, ["/bin/sh", "-c", COPY_SHELL, name, str(source), str(dest), str(digest)],
                            self.shape, owner="harness")
        self._wait()
        return self._reported(name, digest)

    def _reported(self, name: str, digest: Path) -> str:
        """The sha256 role *name* wrote to *digest*; a retryable infra fault when it wrote none."""
        got = reported_sha256(digest)
        if got is None:
            raise StageFault.infra(f"role {name} reported no sha256 for the image file it wrote", retryable=True,
                                   role=name)
        return got

    def _build(self, decl: ImageDecl, work: Path) -> tuple[Path, str, dict[str, object], str]:
        """Build *decl* in *work*, its build role's fresh node-local directory; return the image
        file, the sha256 the role reported for it, what the runtime records of the build in
        ``meta.json`` (``ImageRuntime.build_record``: for apptainer, the mode it reported) and the
        recipe it was built from."""
        parent = parent_of(decl)
        images = self.images
        # The role copies the parent's file from its entry into the runtime's parent_file, and the
        # recipe builds on that copy.
        built_on = None
        if parent is not None:
            stored, parent_sha256 = self.ctx.entry_file("images", self.keys[parent], self.entries[parent],
                                                        images.image_file)
            built_on = (stored, parent_sha256, work / images.parent_file)
        recipe = images.write_recipe(decl, work, locks_dir=LOCKS_DIR, uv=UV, tasks_dir=TASKS_DIR,
                                     registry=registry_of(decl, self.by_name))
        def_file = work / images.recipe_file
        image, digest = work / images.image_file, work / "image.sha256"
        login = isinstance(decl, BaseImage | RegistryImage) and bool(self.login_env)
        argv, env = images.build_role(def_file, image, digest=digest, parent=built_on, login=login, contestant=False)
        env = {**env, **(self.login_env if login else {})}
        name = f"build-{decl.name}"
        self.ctx.start_role(name, argv, self.shape, env=env, secret_env=(REGISTRY_TOKEN_GRANT,) if login else (),
                            owner="harness", daemon=True)
        self._wait()
        # A base's pulled layers stay in apptainer's cache, which nothing reads again.
        remove_tree(work / "cache")
        sha256 = self._reported(name, digest)
        # A role given the login logged on the node; the attempt's logs hold the scrubbed copy.
        # The mode only describes the build, so a log that cannot be read leaves it unknown.
        err = self.ctx.attempt_dir / "logs" / f"{name}.err"
        try:
            stderr = err.read_text(errors="replace")
        except OSError:
            stderr = ""
        try:
            record = images.build_record(work, stderr)
        except ValueError as e:
            raise StageFault.infra(f"role {name} {e}", retryable=True, role=name) from e
        return image, sha256, record, recipe

    def _fill(self, decl: ImageDecl, staging: Path, pinned: list[PinnedImage]) -> dict[str, str]:
        """Fill *staging* with *decl*'s entry; return the sha256 of each file in it: the image
        file's as the role that put it there reported it, and each file this process wrote
        itself by the bytes it wrote."""
        # The image file is node-local, in the directory of the role that writes it.
        work = self.ctx.role_dir(f"build-{decl.name}")
        image, sha256, record, recipe = self._build(decl, work)
        extracted = self.ctx.pin_file(image, sha256=sha256)
        pinned.append(extracted)
        inventory = self._inventory(decl, extracted)
        self._judge(decl, inventory)
        # Nothing reads the tree from here: the entry publishes the image file, and a child builds
        # on that file.
        self.ctx.release(extracted)
        image_file = self.images.image_file
        reported = {image_file: self._copy(f"publish-{decl.name}", image, staging / image_file)}
        image.unlink()
        recipe_file = self.images.recipe_file
        reported[recipe_file] = write_file(staging / recipe_file, recipe.encode())
        reported["inventory.json"] = write_file(staging / "inventory.json",
                                                inventory.model_dump_json(indent=2).encode())
        parent = parent_of(decl)
        meta = ImageMeta(name=decl.name, kind=decl.kind, key=self.keys[decl.name], parent=parent,
                         parent_key=self.keys[parent] if parent else None, **record)
        reported["meta.json"] = write_file(staging / "meta.json", meta.model_dump_json(indent=2).encode())
        return reported

    def image(self, decl: ImageDecl) -> CacheEntry:
        """The published entry of *decl*, built when nobody has published it."""
        ctx = self.ctx
        key = self.keys[decl.name]
        what = f"images/{key}"
        pinned: list[PinnedImage] = []
        reported: dict[str, str] = {}

        def fill(staging: Path) -> None:
            self._check_fill(key)
            reported.update(self._fill(decl, staging, pinned))

        def check(files: list[FileRecord]) -> None:
            got = next((f.sha256 for f in files if f.path == self.images.image_file), None)
            if got != pinned[-1].file_sha256:
                raise StageFault.infra(f"the image file staged for {decl.name} has sha256 {got}, not the "
                                       f"{pinned[-1].file_sha256} of the file that was inventoried", retryable=True,
                                       entry=what)

        def waiting(holder: Holder) -> None:
            ctx.events.emit("cache_wait", entry_kind="images", key=key, holder_purpose=holder.purpose,
                            holder_host=holder.host, holder_pid=holder.pid, holder_job_id=holder.job_id)

        assert ctx.cache is not None
        try:
            got = ctx.cache.populate("images", key, fill, holder=self.holder, digests=lambda: reported,
                                     check=check, poll_sec=POLL_SEC,
                                     source={"image": decl.name, "stage": ctx.stage.name,
                                             "instance": ctx.plan.instance},
                                     deadline=ctx.deadline, stopping=lambda: ctx.stopping, on_wait=waiting)
        except WaitEnded as e:
            raise StageFault.infra(str(e), retryable=True, entry=what) from e
        except CacheCorrupt as e:
            raise StageFault.infra(f"cache entry {what}: {e}", retryable=False, entry=what) from e
        ctx.events.emit("cache_filled" if got.published else "cache_hit", entry_kind="images", key=key,
                        entry_sha256=entry_sha256(got.entry))
        self.entries[decl.name] = entry_sha256(got.entry)
        return got.entry

    def _check_fill(self, key: str) -> None:
        """End the attempt with an infra fault, not retried, when filling entry *key*, which the
        cache does not hold at its image's turn, needs more node-local disk than the stage declares
        (``disk_gib``). The plan's disk holds a fill unless the plan found every entry, so only an
        entry the plan found and the cache has lost since brings this about, and the fault names
        it. A retry would run the same plan, which declares the same disk; a new plan counts the
        fill."""
        need, declared = disk_gib(1), self.ctx.stage.shape.disk_gib
        if need <= declared:
            return
        assert key in self.params.found, "the plan's disk holds a fill unless it found every entry"
        names = [name for name, k in self.keys.items() if k == key]
        raise StageFault.infra(f"the cache no longer holds {', '.join(names)}, which it held when the plan was made: "
                               f"the stage would fill {_entries(1)}, which need {need} GiB of node-local disk, more "
                               f"than the {declared} GiB it declares; plan it again, and the new plan's disk counts "
                               "the fill", retryable=False, entries=f"images/{key}", declared_disk_gib=str(declared),
                               needed_disk_gib=str(need))

    def run(self) -> dict[str, str]:
        record = {}
        for decl in self.decls:
            entry = self.image(decl)
            image_file = self.images.image_file
            sif = next((f.sha256 for f in entry.manifest.files if f.path == image_file), None)
            if sif is None:
                raise StageFault.infra(f"cache entry images/{entry.key} holds no {image_file}", retryable=False,
                                       entry=f"images/{entry.key}")
            self.ctx.record_image(decl.name, sif)
            record[decl.name] = {"key": entry.key, "entry_sha256": entry_sha256(entry), "sif_sha256": sif}
        data = json.dumps(record, indent=2, sort_keys=True).encode()
        return {"images": self.ctx.write_output("images.json", data)}


@register
class ImagesStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "images"
    params_model: ClassVar[type[BaseModel]] = ImagesParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: four for each image of *params*, its build, its
        extraction, its inventory and the copy into its entry."""
        assert isinstance(params, ImagesParams)
        return 4 * len(params.images)

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        if shape.gpus:
            raise ValueError("the images stage builds on the host side and asks for no GPU")
        # One role at a time, of the stage's shape less the runner's memory: it fits whenever
        # that leaves it some, which role_shape checks.
        role_shape(shape, runner_memory_gib)

    @classmethod
    def check_site(cls, shape: ResourceShape, params: BaseModel, *, site: SiteSettings) -> None:
        """The entries found are keys of the images as *site*'s substrate keys them
        (``Substrate.recipe_key``), and the stage's disk holds the entries it fills."""
        checked = ImagesParams.model_validate(params.model_dump())
        keys = image_keys(checked.images, substrate(site).recipe_key)
        stray = _stray(keys, checked.found)
        if stray is not None:
            raise ValueError(stray)
        entries = len(fills(keys, checked.found))
        disk = disk_gib(entries)
        if shape.disk_gib < disk:
            raise ValueError(f"the images stage needs at least {disk} GiB of node-local disk for the "
                             f"{_entries(entries)} it fills, not {shape.disk_gib}")

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Images(ctx).run()
