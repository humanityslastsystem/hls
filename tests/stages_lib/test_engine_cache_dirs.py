"""Every directory a pool member's serving stack keeps its build or tuning products in is moved into
the engine cache, outside what a stage drops before it publishes, or says why it is not
(``hls.stages.engine_cache_dirs``); and every member image a task declares holds a surveyed stack,
so a member on another stack is surveyed before it runs."""

from __future__ import annotations

import re

from hls.recipes import catalog, pins
from hls.recipes.model import BaseImage, LayerImage, parent_of
from hls.recipes.render import lock_dir
from hls.stages import engine_cache
from hls.stages import engine_cache_dirs as dirs
from hls.stages.engine_cache import ENGINE_CACHE, engine_cache_env


def test_the_engine_cache_env_moves_every_surveyed_directory_into_the_cache():
    # TVM_FFI_CACHE_DIR among them: without it, the JIT kernels of the warm stage go to the role's
    # home and never reach the cache entry, so a seeded SGLang capture builds them again at load.
    assert engine_cache_env() == {
        "VLLM_CACHE_ROOT": "/engine-cache/vllm", "TORCHINDUCTOR_CACHE_DIR": "/engine-cache/inductor",
        "TRITON_CACHE_DIR": "/engine-cache/triton", "XDG_CACHE_HOME": "/engine-cache/xdg",
        "FLASHINFER_WORKSPACE_BASE": "/engine-cache", "TMPDIR": "/engine-cache/tmp",
        "TVM_FFI_CACHE_DIR": "/engine-cache/tvm-ffi", "SGLANG_CACHE_DIR": "/engine-cache/sglang",
        "SGLANG_JIT_CACHE_DIR": "/engine-cache/sglang/jit", "SGLANG_DG_CACHE_DIR": "/engine-cache/deep_gemm",
        "CUTE_DSL_CACHE_DIR": "/engine-cache/cute-dsl", "QUACK_CACHE_DIR": "/engine-cache/quack",
        "TILELANG_CACHE_DIR": "/engine-cache/tilelang", "HUMMING_CACHE_DIR": "/engine-cache/humming/cache",
        "HUMMING_TMP_DIR": "/engine-cache/humming/tmp", "OUTLINES_CACHE_DIR": "/engine-cache/outlines",
        "CUDA_CACHE_PATH": "/engine-cache/nv"}


def test_every_surveyed_directory_is_moved_or_says_why_not():
    for entry in dirs.ENGINE_CACHE_SURVEY:
        assert entry.holds and entry.source, entry
        if entry.moved_by is None:
            assert entry.left_because, f"{entry.holds} is left where it is, and says nowhere why"
        else:
            assert entry.moved_by in dirs.ENGINE_CACHE_VARIABLES and entry.left_because is None, entry


def test_every_variable_moves_a_surveyed_directory():
    # So a variable is set only for a directory the survey found, with the lines that read it.
    assert set(dirs.ENGINE_CACHE_VARIABLES) == {e.moved_by for e in dirs.ENGINE_CACHE_SURVEY if e.moved_by}


def test_no_build_product_lands_in_what_a_stage_drops_before_publishing():
    # An engine's owner removes the engine cache's temporaries before it publishes the rest; CuTe
    # DSL's file cache, for one, sits under TMPDIR unless it is moved out.
    env = engine_cache_env()
    dropped = f"{ENGINE_CACHE}/{dirs.ENGINE_CACHE_TMP}"
    assert env["TMPDIR"] == dropped
    kept = {variable: path for variable, path in env.items() if variable != "TMPDIR"}
    assert all(path != dropped and not path.startswith(f"{dropped}/") for path in kept.values()), kept
    assert engine_cache.ENGINE_CACHE_TMP == dirs.ENGINE_CACHE_TMP


def _chain_pins(name: str, by_name: dict) -> dict[str, str]:
    """Every pin the pip locks of image *name*'s chain install, name to version."""
    locked: dict[str, str] = {}
    decl = by_name[name]
    while not isinstance(decl, BaseImage):
        if isinstance(decl, LayerImage) and decl.pip_lock is not None:
            text = (lock_dir(decl, pins.LOCKS_DIR) / decl.pip_lock).read_text()
            locked.update(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", text, re.MULTILINE))
        decl = by_name[parent_of(decl)]
    return locked


def test_every_member_a_task_builds_holds_the_engine_versions_of_a_surveyed_stack():
    """A member installs its engine packages from the locks of its chain, so they name their
    versions: those of the stack whose cache directories the survey read, or the member is on a
    stack nobody surveyed."""
    built = []
    for task in catalog.task_keys():
        images = catalog.task_images(task)
        by_name = {d.name: d for d in images.decls}
        for role, name in images.measured.items():
            if not role.startswith("member-") or not isinstance(by_name[name], LayerImage):
                continue
            locked = _chain_pins(name, by_name)
            stacks = [letter for letter, stack in dirs.SURVEYED_STACKS.items()
                      if all(locked.get(k) == v for k, v in stack.items())]
            assert len(stacks) == 1, (task, name, {k: locked.get(k) for k in dirs.SURVEYED_STACKS["V"]})
            built.append(name)
    assert built, "no task builds a member"


def test_each_surveyed_stack_names_its_engine():
    assert {letter: ("vllm" in stack, "sglang" in stack) for letter, stack in dirs.SURVEYED_STACKS.items()} == {
        "V": (True, False), "S": (False, True), "A": (False, True)}
