"""A driver bound to the tenants and the seed of one drive, for tests that run the drive loop
(``Driver._drive``) directly rather than through ``Driver.run``, which builds the artifact
client from a contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hls.spec.drivers import Driver


@dataclass(frozen=True)
class BoundDrive:
    """A driver with the tenants and the seed one drive of it is given, as the drive role gives
    them to ``Driver.run``."""

    #: The driver, built whole.
    driver: Driver
    #: The tenants the drive serves.
    tenants: list[Any]
    #: The drive's seed.
    seed: int = 0

    async def drive(self, client, on_event, split) -> None:
        """Drive the tenants' sessions of *split* against *client*, each event to *on_event*."""
        await self.driver._drive(client, on_event, split, tenants=self.tenants, seed=self.seed)
