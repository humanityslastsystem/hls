"""The cache entry kind an engine's caches are kept as, and the record a stage keeps of the engine
cache its role ran with.

An engine keeps what it builds or tunes at load in an engine cache (``hls.stages.engine_cache``).
The stage that owns an engine's entry builds it once and publishes it as a cache entry of kind
``ENGINE_CACHE_KIND``; every other stage of that engine seeds its role's cache with a checked copy of
that entry. Each such stage records which entry in ``engine-cache.json`` (``EngineCacheRecord``), and
a drive of a measured baseline records it in its ``drive.json`` (``hls.drive.record``). The role
programs read these records too, so they live here, below the stages."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from hls.store.base import check_segment
from hls.store.digest import Sha256Hex

#: The cache entry kind of an engine's caches. An engine's caches hold compiled kernels and their
#: build records (FlashInfer's built kernels, Triton's and vLLM's compile caches) and vLLM's tuned
#: kernel configurations (``*.best_config``). FlashInfer's autotune persists nothing, so it tunes
#: again at every load.
ENGINE_CACHE_KIND = "engine-cache"
#: A stage's record of the engine cache its role ran with, among its outputs.
ENGINE_CACHE_FILE = "engine-cache.json"


class EngineCacheRecord(BaseModel):
    """``engine-cache.json``: the engine cache a role ran with, its engine's compile products and
    tuned kernel configurations, and for the engine's owner the entry it left
    (``hls.stages.engine_cache``). The products of a build are not byte for byte those of another
    build of the same engine, so an entry is named by the digest of the one build that published it,
    and every seed is that entry's, checked against the digests its manifest records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The ``engine-cache`` entry's key: the engine's image, GPU kind and config, and the build
    #: environment and network its roles run with (``hls.stages.engine_cache.engine_cache_key``).
    key: Annotated[str, AfterValidator(check_segment)]
    #: The ENTRY.json digest of the entry the role's cache was a checked copy of; None when the role
    #: started cold, or the stage started no role.
    seeded_from: Sha256Hex | None
    #: For the engine's owner alone (a calibration's warm stage or its instrument, or a measured
    #: baseline's warm stage): the ENTRY.json digest of the engine's entry once the stage was done,
    #: the one every later stage of the engine in the instance is pinned to and seeded from: the
    #: entry it found, the one it published, or the one another stage published first. None for every
    #: other stage, and for an owner that left none (``unpublished``).
    entry: Sha256Hex | None = None
    #: Whether this stage published ``entry`` from the cache its role built cold.
    published: bool = False
    #: Why an owner whose role built its cache cold left no entry: its publish failed.
    unpublished: str | None = None

    @model_validator(mode="after")
    def _one_story(self) -> Self:
        if self.published and (self.entry is None or self.seeded_from is not None):
            raise ValueError("a stage that published an entry built its cache cold and names the entry")
        if self.unpublished is not None and (self.entry is not None or self.seeded_from is not None):
            raise ValueError("a stage that left no entry built its cache cold and names none")
        if None not in (self.seeded_from, self.entry) and self.seeded_from != self.entry:
            raise ValueError("an owner seeded from an entry leaves that entry")
        return self
