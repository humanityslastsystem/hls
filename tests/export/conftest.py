"""The export tests run on the finalize tests' checkout, whose task packages the fixture
instances belong to."""

from __future__ import annotations

from tests.finalize.conftest import checkout  # noqa: F401  (the session fixture, shared with finalize's tests)
