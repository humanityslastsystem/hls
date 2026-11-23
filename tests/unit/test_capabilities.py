import inspect
from abc import abstractmethod

import pytest

from hls.core.capabilities import (
    Capability,
    DecodeCodes,
    Generate,
    ReferenceLogprobs,
    StackReport,
    TeacherForce,
    cap_method,
)


def test_capabilities_are_abcs_with_one_async_method():
    for cap in (Generate, TeacherForce, ReferenceLogprobs, StackReport, DecodeCodes):
        assert issubclass(cap, Capability)
        assert len(cap.__abstractmethods__) == 1
        name = next(iter(cap.__abstractmethods__))
        assert inspect.iscoroutinefunction(getattr(cap, name))


def test_cap_method_returns_the_single_method_name():
    assert cap_method(Generate) == "generate"
    assert cap_method(TeacherForce) == "teacher_force"
    assert cap_method(ReferenceLogprobs) == "reference_logprobs"
    assert cap_method(StackReport) == "stack_report"
    assert cap_method(DecodeCodes) == "decode_codes"


def test_capability_cannot_be_instantiated_without_implementing_method():
    class Broken(TeacherForce):
        pass

    with pytest.raises(TypeError):
        Broken()


def test_cap_method_rejects_wrong_method_count():
    from hls.core.capabilities import Capability
    with pytest.raises(ValueError):
        cap_method(Capability)            # zero abstract methods

    class TwoMethods(Capability):
        @abstractmethod
        async def a(self): ...
        @abstractmethod
        async def b(self): ...

    with pytest.raises(ValueError):
        cap_method(TwoMethods)


def test_serve_traffic_abc_declares_every_keyword_its_caller_passes():
    """The ABC is the contract a second engine is written against, so a keyword the
    calibration path passes has to be on it: an engine written from the ABC alone that
    lacks one, ``drive_timeout_sec`` say, raises ``TypeError`` on the first calibration drive."""
    from hls.core.capabilities import ServeTraffic
    from hls.reference.engines import ClockedServing

    declared = inspect.signature(ServeTraffic.serve_traffic).parameters
    implemented = inspect.signature(ClockedServing.serve_traffic).parameters
    assert set(implemented) <= set(declared), sorted(set(implemented) - set(declared))

    # The call hls.calibrate.drive_serving makes, bound against the ABC alone.
    inspect.signature(ServeTraffic.serve_traffic).bind(
        object(), [], seed=0, max_sessions=1, request_timeout_sec=1.0,
        drive_timeout_sec=1.0, agent=None)
