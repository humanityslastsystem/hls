"""A driver is built whole: it is no engine and holds no contract, seed or tenants of its own."""

from __future__ import annotations


def test_a_driver_is_built_whole():
    from hls.core.engine import Engine
    from hls.spec.drivers import Driver

    assert not issubclass(Driver, Engine)
    driver = Driver(request_timeout_sec=60.0)
    assert not hasattr(driver, "contract") and not hasattr(driver, "seed") and not hasattr(driver, "tenants")
