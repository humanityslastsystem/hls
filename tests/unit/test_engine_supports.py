from abc import ABC

from hls.core.capabilities import Generate, TeacherForce
from hls.core.engine import Engine, supports


class _GenOnly(Engine, Generate):
    async def generate(self, prompts):
        return prompts


class _Both(Engine, Generate, TeacherForce):
    async def generate(self, prompts):
        return prompts

    async def teacher_force(self, items):
        return [[0.0] for _ in items]


def test_supports_is_derived_from_mro():
    assert supports(_GenOnly) == frozenset({Generate})
    assert supports(_Both) == frozenset({Generate, TeacherForce})


def test_an_engine_names_the_resources_it_asks_for_and_nothing_else():
    assert Engine.__bases__ == (ABC,)
    assert Engine.__abstractmethods__ == frozenset({"resources"})


def test_supports_excludes_unimplemented_capability():
    from hls.core.capabilities import Generate

    class _Half(Engine, Generate):     # subclasses Generate but does NOT implement generate
        pass

    assert supports(_Half) == frozenset()      # un-implemented capability is not reported
