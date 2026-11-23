"""An engine's cache: where a role's engine keeps what it builds or tunes at load, how the cache
entry that holds it is keyed, and how the stage that owns the entry builds it once and every other
stage of the engine is seeded from it.

A role runs its engine with an engine cache on the node's disk mounted at ``ENGINE_CACHE``, where
the variables of ``engine_cache_env`` send every directory the engine's packages write such
products to (``hls.stages.engine_cache_dirs``). An engine's entry is a cache entry of kind
``ENGINE_CACHE_KIND``, keyed by the engine's image, GPU kind and config, and by the build
environment its roles run with and their network (``engine_cache_key``). It holds whatever the
engine keeps there: the compile products, FlashInfer's object files and build records among them,
each copied with the modification time the build left it (FlashInfer's ninja builds whatever
output is older than its source), so a seeded load compiles none of its kernels, and vLLM's tuned
kernel configurations (``*.best_config``), which every load seeded from the entry replays. A seeded
load may still re-link a FlashInfer library, seconds of work: FlashInfer rewrites its build file at
load, listing its generated sources in directory order, which a copy need not keep. Kernels an
engine compiles only once serving starts (small Triton ones in vLLM) are in no entry and compile in
each drive. FlashInfer's own autotune persists nothing, and tunes again at every load.

Each engine of an instance has one owner, the first stage of the instance to run it: a
calibration's instrument for the pinned reference and its warm stage for each pool member
(``hls.stages.calibration``), and a measured baseline's warm stage for the baseline's server
(``hls.stages.baseline_warm``). An owner finds the entry published (``published``), or builds it:
its role starts with an empty cache (``prepare``), and once the role succeeded the stage publishes
what it built, the engine's temporary files removed first, unless another stage published it
first (``owned_record``). The entry is published once and appears whole. The owner records the
entry's ENTRY.json digest in its ``engine-cache.json`` (``EngineCacheRecord``). Every other stage
of the engine reads it off the owner's manifest, refusing an entry the owner left under another key
than its own (``owner_entry``), and seeds its role's cache with a writable copy of that entry and
of no other, made afresh for the attempt, each file checked against the digest the entry's
manifest records (``StageContext.seed``). Two builds of one engine are not byte for byte the same,
so a seed is of the one build the owner left; and no role ever writes into the entry itself, only
into its attempt's copy."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import ValidationError

from hls.package.wheels import COMPUTE_CAPABILITY
from hls.pipeline.stage import StageContext, StageFault
from hls.stages.engine_cache_dirs import ENGINE_CACHE_TMP, ENGINE_CACHE_VARIABLES
from hls.stages.inputs import input_manifest, output, read_output
from hls.store.base import check_segment
from hls.store.cache import CacheCorrupt, CacheEntry, entry_sha256
from hls.store.digest import sha256_bytes
from hls.store.engine_cache import ENGINE_CACHE_FILE, ENGINE_CACHE_KIND, EngineCacheRecord
from hls.substrate.container import SealPlan
from hls.util import remove_tree, validation_errors

#: Where the engine cache mounts in a role's container.
ENGINE_CACHE = "/engine-cache"
#: Node-local disk, in GiB, an engine's cache takes on the node: twice the peak of
#: tab_completion's vLLM member's cache with its build's temporary files in it, about 3.8 GB. Every
#: other engine gets the same figure, and a harness role's disk is not watched, so a larger cache
#: would be found only as a full node.
ENGINE_CACHE_GIB = math.ceil(2 * (582.5e6 + 3.21e9) / 2**30)
#: Seconds an engine's load takes when it builds its engine cache cold: tab_completion's vLLM
#: member on one H200 and eight CPUs takes about 69 minutes from the role's start to the engine
#: ready, nearly all of it the JIT build of FlashInfer's ``fused_moe_90``. The member's image file
#: is tab_completion's measured baseline too.
COLD_BUILD_SEC = 4146
#: Seconds an owner gives its engine's cold load by default: the cold build
#: (``COLD_BUILD_SEC``) and half as much again, for a slower node or a build that grew. A stage given
#: more time by a ``[resources]`` table gives the load as much more.
COLD_LOAD_SEC = math.ceil(1.5 * COLD_BUILD_SEC)


def engine_cache_env() -> dict[str, str]:
    """The variables that point every directory an engine keeps its build and tuning products in
    at the engine cache, each as ``hls.stages.engine_cache_dirs`` surveyed it, so a seeded load
    finds what its owner's load built; they override the image's own, which could point a cache
    back at the container's small session tmpfs."""
    return {variable: f"{ENGINE_CACHE}/{path}" if path else ENGINE_CACHE
            for variable, path in ENGINE_CACHE_VARIABLES.items()}


def engine_build_env(gpu_kind: str | None) -> dict[str, str]:
    """The variables an engine's role runs with that decide what its load builds and where it keeps
    it: every variable of the engine cache (``engine_cache_env``), and the compute capability of a
    GPU of *gpu_kind*, which an extension built at load compiles for (none on no GPU, None). A role
    gets them after every other variable, so they are what the engine cache key digests."""
    arch = {} if gpu_kind is None else {"TORCH_CUDA_ARCH_LIST": COMPUTE_CAPABILITY[gpu_kind]}
    return {**engine_cache_env(), **arch}


def engine_cache_key(sif_sha256: str, gpu_kind: str, config: str, *, seal: SealPlan | None) -> str:
    """The key of the engine cache entry of an engine of *config* on the image whose file has the
    sha256 *sif_sha256*, on a GPU of *gpu_kind*, its roles running with ``engine_build_env`` on the
    network *seal* gives them (None: the step's own): the kind, then the image, the config and the
    build environment with the network, each by the first 16 hex digits of its digest. The image
    and the config say what is built; the environment says where each product lands, so whether the
    entry holds it, and the network what a load could fetch into it. So an entry built under other
    cache variables is never matched again, and stays under its old key. A seal's ingress port is
    left out: where the server listens changes nothing it builds."""
    build = json.dumps({"env": engine_build_env(gpu_kind),
                        "seal": None if seal is None else {"policy": seal.policy.model_dump(mode="json"),
                                                           "index": seal.index}},
                       sort_keys=True)
    return check_segment(f"{gpu_kind}-{sif_sha256[:16]}-{sha256_bytes(config.encode())[:16]}"
                         f"-{sha256_bytes(build.encode())[:16]}")


def published(ctx: StageContext, key: str) -> CacheEntry | None:
    """Engine cache entry *key* as it is published now; None when none is. Raise infra, not
    retried, for a published entry that is corrupt, which no role is seeded from."""
    assert ctx.cache is not None
    try:
        return ctx.cache.lookup(ENGINE_CACHE_KIND, key)
    except CacheCorrupt as e:
        raise StageFault.infra(f"cache entry {ENGINE_CACHE_KIND}/{key} is corrupt: {e}", retryable=False) from e


def prepare(ctx: StageContext, role: str, key: str | None, *, pinned: str | None,
            owner: bool = False) -> tuple[Path, str | None]:
    """Role *role*'s engine cache, ``engine-cache`` in its directory, made before the role starts,
    for the engine cache entry *key*: a writable copy of the entry its owner pinned by the
    ENTRY.json digest *pinned*, or for an owner (*owner*) of the entry as it is published now, or
    else an empty directory, as it is for a role with no entry (*key* None); either holding the
    ``ENGINE_CACHE_TMP`` directory the cache variables send the engine's temporary files to. Return
    it and the ENTRY.json digest of the entry it is a copy of, None when it starts empty. Raise as
    ``StageContext.seed`` does, a corrupt entry or another than the pinned one among it."""
    cache = ctx.role_dir(role) / "engine-cache"
    seeded = None
    if key is not None and (owner or pinned is not None):
        seeded = ctx.seed(ENGINE_CACHE_KIND, key, cache, expect=pinned)
    if seeded is None:
        cache.mkdir()
    (cache / ENGINE_CACHE_TMP).mkdir()
    return cache, seeded


def owner_entry(ctx: StageContext, stage: str, key: str) -> str | None:
    """The ENTRY.json digest of the engine cache entry *key* the engine's owner *stage* left, from
    its ``engine-cache.json`` checked against its manifest; None when it left none. Raise infra, not
    retried, for an owner that left an entry under another key: the two stages key one engine
    differently, and every retry would too."""
    ref = output(input_manifest(ctx, stage), ENGINE_CACHE_FILE)
    try:
        record = EngineCacheRecord.model_validate_json(read_output(ctx, ref))
    except ValidationError as e:
        raise StageFault.infra(f"{ref.key} is not an engine cache record: {validation_errors(e)}",
                               retryable=False) from e
    if record.entry is not None and record.key != key:
        raise StageFault.infra(f"the engine's owner {stage} left the engine cache entry {record.key}, and this stage "
                               f"keys the engine {key}", retryable=False)
    return record.entry


def write_record(ctx: StageContext, record: EngineCacheRecord) -> str:
    """The key of the stage's ``engine-cache.json``, holding *record*."""
    return ctx.write_output(ENGINE_CACHE_FILE, record.model_dump_json(indent=2).encode())


def found_record(ctx: StageContext, key: str, entry: CacheEntry) -> str:
    """The key of an owner's ``engine-cache.json`` when it found its engine's *entry* published, by
    an earlier stage of the engine: it built nothing and started no role."""
    return write_record(ctx, EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry)))


def owned_record(ctx: StageContext, cache: Path, key: str, seeded: str | None, *, must_publish: bool) -> str:
    """After an owner's role succeeded, the key of the stage's ``engine-cache.json``, naming the
    engine's entry it leaves. A cache seeded from the entry (*seeded*) leaves that entry. One its
    role built cold in *cache* is published as the entry *key*, its temporary files removed first,
    unless another stage published the entry first, whose entry it then leaves. A publish that fails
    is the stage's fault when it *must_publish*; otherwise it is recorded with its reason, the stage
    leaves no entry, and nothing else fails, since the role's outputs stand without it
    (``StageContext.publish``)."""
    if seeded is not None:
        return write_record(ctx, EngineCacheRecord(key=key, seeded_from=seeded, entry=seeded))
    remove_tree(cache / ENGINE_CACHE_TMP)
    try:
        got = ctx.publish(ENGINE_CACHE_KIND, key, cache)
    except StageFault as e:
        if must_publish:
            raise
        ctx.events.emit("engine_cache_unpublished", key=key, reason=e.fault.reason)
        return write_record(ctx, EngineCacheRecord(key=key, seeded_from=None,
                                                   unpublished=f"the publish failed: {e.fault.reason}"))
    return write_record(ctx, EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(got.entry),
                                               published=got.published))
