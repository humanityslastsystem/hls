"""Producer-phase network + package policy: flat, CLI-settable access dimensions.

Each dimension (domains, packages) is a single ``AccessList`` carrying a mode
(allow or deny) and an entry list. Flat pydantic ``Spec`` so it round-trips
cleanly through ``model_dump``/``model_validate`` without a discriminated union
and is reachable from the CLI via ``--set`` paths.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from hls.core.poly import Spec


def pep503_normalize(name: str) -> str:
    """PEP 503 normalized distribution name (case-fold + collapse -_. to '-')."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _host_match(pattern: str, host: str) -> bool:
    """Exact host match, or ``*.suffix`` subdomain match (NOT the bare suffix).

    ``*.evil.com`` matches ``a.evil.com`` but not bare ``evil.com``.
    """
    if pattern == host:
        return True
    if pattern.startswith("*."):
        suffix = pattern[1:]          # ".evil.com"
        bare = suffix[1:]             # "evil.com"
        return host.endswith(suffix) and host != bare
    return False


def _listed(item: str, entries: list[str], *, wildcard: bool, normalize: bool) -> bool:
    """Is ``item`` matched by any entry, under the requested matching mode?"""
    if normalize:
        target = pep503_normalize(item)
        return any(pep503_normalize(e) == target for e in entries)
    if wildcard:
        return any(_host_match(e, item) for e in entries)
    return item in entries


class AccessList(Spec):
    """One access dimension: a whitelist (mode="allow") or blacklist (mode="deny").

    An empty allow list permits nothing (fully sealed); an empty deny list permits
    everything (open). Flat (mode + entries) so it round-trips through pydantic
    validation and is reachable from the CLI --set path. wildcard enables *.suffix
    domain matching; normalize enables PEP 503 package-name normalization.
    """

    mode: Literal["allow", "deny"] = "allow"
    entries: list[str] = []

    def permits(self, item: str, *, wildcard: bool = False, normalize: bool = False) -> bool:
        listed = _listed(item, self.entries, wildcard=wildcard, normalize=normalize)
        return listed if self.mode == "allow" else not listed


class EgressPosture(str, Enum):
    """The three domain-egress postures a policy can express, read off its domain
    ``AccessList`` by ``of``; ``str, Enum`` so it reads as a plain literal. Derived, not
    a field: it never appears in ``NetworkPolicy.model_dump()``.
    """
    OPEN = "open"        # unrestricted egress (empty deny-list -> nothing denied)
    AIR_GAP = "air_gap"  # no egress at all (empty allow-list -> nothing permitted)
    FILTER = "filter"    # restricted to the listed hosts (a non-empty list)

    @classmethod
    def of(cls, domains: AccessList) -> EgressPosture:
        """The posture of the domain list *domains*: a non-empty list filters; an empty one
        is open in deny mode (nothing denied) and an air gap in allow mode (nothing
        permitted). The one place that reads a posture off a list's mode and emptiness:
        ``NetworkPolicy.domain_posture`` and the seal's DNS decision
        (``hls.substrate.seal.closes_dns``) both ask it."""
        if domains.entries:
            return cls.FILTER
        return cls.OPEN if domains.mode == "deny" else cls.AIR_GAP


class NetworkPolicy(Spec):
    """Unified per-dimension egress policy, used by every phase (producer + runtime).

    domains governs which hosts are reachable; packages (None = N/A, e.g. runtime)
    governs which pip packages may install. Both dimensions default to UNRESTRICTED --
    domains to open (empty deny-list) and packages to None -- so a fully-default
    ``NetworkPolicy()`` imposes no restriction. INVARIANT: a package policy REQUIRES a
    restricted domain policy (an allow-list or air-gap); packages-with-open-domains is
    rejected at construction, because the package filter is only enforceable when the
    index is the only reachable package source -- with domains open a sealed agent can
    fetch a denied package from any other host. Flat: validates back from its own dump,
    CLI-settable.
    """

    #: The hosts the sealed container may reach. Open by default (an empty deny list, nothing
    #: denied), matching ``packages=None``: an unspecified domain dimension imposes no
    #: restriction. The air gap (an empty allow list) is a deliberate state that must be set
    #: explicitly, never inherited from a policy that only meant to restrict packages.
    domains: AccessList = AccessList(mode="deny")
    #: The pip packages that may install through the index; None for no package rule.
    packages: AccessList | None = None

    def domain_posture(self) -> EgressPosture:
        """This policy's domain-egress posture: ``EgressPosture.of`` its ``domains``
        (empty deny-list -> OPEN, empty allow-list -> AIR_GAP, non-empty -> FILTER). A
        plain method, not a field, so ``model_dump()`` does not carry it."""
        return EgressPosture.of(self.domains)

    @model_validator(mode="after")
    def _package_policy_requires_restricted_domains(self) -> NetworkPolicy:
        """Reject a package policy paired with OPEN domains (substrate-agnostic).

        The package filter is only enforceable when the filtering index is the SOLE
        reachable package source; with domains open, a sealed agent can fetch a denied
        package from any other host on ANY backend, so the combination is incoherent
        everywhere. Pydantic wraps the raised ``ValueError`` as a ``ValidationError``."""
        if self.packages is not None and self.domain_posture() is EgressPosture.OPEN:
            raise ValueError(
                "a package policy requires a restricted domain policy: the package "
                "filter is only enforceable when the index is the only reachable "
                "package source. Set `domains` to an allow-list or air-gap "
                '(`AccessList(mode="allow", ...)`), or drop the package policy.'
            )
        return self


class SealPolicy(BaseModel):
    """What a seal's interceptor enforces (``hls.egress.proxy``): a network policy, the task's own
    or one the harness made from it, and the hosts the harness vouches for beside it. Only the
    harness builds one, and the task-authored ``NetworkPolicy`` has no field for a vouched host,
    so a task cannot name one. The seal hands its JSON to the interceptor, which reads it back
    through this model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The hosts the sealed container may reach and the packages its index may serve.
    network: NetworkPolicy
    #: The hosts the seal vouches for: the agent build's route host and its current IPv4
    #: addresses (``hls.stages.agent_build.seal_policy``); none for every other seal. Such a host
    #: may resolve to an internal address (a model gateway on an RFC1918 IP), which
    #: the interceptor's DNS-name SSRF guard would refuse: it refuses any allow-listed name whose A
    #: record is not global. A host here is exempt from that refusal only: it must still pass the
    #: domain allow list, and the interceptor still resolves the claimed name and dials the
    #: resolved IP (never the client's chosen IP), so the exemption widens reachability to exactly
    #: the vouched host.
    trusted_internal_hosts: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()


#: The host the seal probe asks for, which a policy that names hosts to allow does not name.
#: Both the drive stage's seal probe and the agent build's probe ask for it.
PROBE_HOST = "hls-seal-probe.invalid"
#: Where the seal probe's refusal check connects: an address of TEST-NET-1 (RFC 5737), which
#: routes nowhere, so that only the seal's redirect to its interceptor takes the connection.
PROBE_DIAL = "192.0.2.1:80"


def refused_host(policy: NetworkPolicy) -> str | None:
    """A host *policy* refuses, which a seal probe asks for: ``PROBE_HOST``, or under a policy
    that permits it, one its deny list names (a name under a wildcard's domain); None for a
    policy that refuses no host."""
    label = PROBE_HOST.split(".", 1)[0]
    candidates = [PROBE_HOST]
    if policy.domains.mode == "deny":
        candidates += [f"{label}.{e.removeprefix('*.')}" if e.startswith("*.") else e
                       for e in policy.domains.entries]
    return next((h for h in candidates if not policy.domains.permits(h, wildcard=True)), None)
