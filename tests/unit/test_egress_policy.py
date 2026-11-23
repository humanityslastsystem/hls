"""Unit tests for the producer-phase network + package policy types.

Covers AccessList/NetworkPolicy evaluation, domain wildcard + PEP 503
normalization, round-trips through pydantic validation, and the sealed-by-default
``AgentProducerSpec().network`` policy.
"""
from __future__ import annotations

import json

import pydantic
import pytest

from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable
from hls.egress.policy import AccessList, EgressPosture, NetworkPolicy, SealPolicy


# ---------------------------------------------------------------------------
# AccessList tests
# ---------------------------------------------------------------------------

def test_accesslist_allow_mode_permits_only_listed():
    a = AccessList(mode="allow", entries=["six", "numpy"])
    assert a.permits("six", normalize=True) is True
    assert a.permits("requests", normalize=True) is False


def test_accesslist_deny_mode_permits_all_but_listed():
    d = AccessList(mode="deny", entries=["tensorflow"])
    assert d.permits("tensorflow", normalize=True) is False
    assert d.permits("numpy", normalize=True) is True


def test_accesslist_default_is_empty_allow_seals_everything():
    a = AccessList()
    assert a.mode == "allow" and a.entries == []
    assert a.permits("anything", normalize=True) is False


def test_accesslist_package_name_normalized():
    d = AccessList(mode="deny", entries=["My_Pkg.Name"])
    assert d.permits("my-pkg-name", normalize=True) is False


def test_accesslist_domain_wildcard():
    d = AccessList(mode="deny", entries=["*.evil.com"])
    assert d.permits("a.evil.com", wildcard=True) is False
    assert d.permits("evil.com", wildcard=True) is True  # bare suffix not matched by *.


def test_accesslist_allow_mode_domain_wildcard():
    a = AccessList(mode="allow", entries=["*.pythonhosted.org", "pypi.org"])
    assert a.permits("files.pythonhosted.org", wildcard=True) is True
    assert a.permits("pythonhosted.org", wildcard=True) is False   # bare suffix not in allow set
    assert a.permits("pypi.org", wildcard=True) is True
    assert a.permits("evil.com", wildcard=True) is False


def test_accesslist_roundtrips_through_dump():
    d = AccessList(mode="deny", entries=["tensorflow"])
    assert AccessList.model_validate(d.model_dump()) == d


# ---------------------------------------------------------------------------
# NetworkPolicy tests
# ---------------------------------------------------------------------------

def test_networkpolicy_defaults_open_domains_no_packages():
    # Both dimensions default to UNRESTRICTED: domains to OPEN (empty deny-list) and
    # packages to None. A bare NetworkPolicy() must NOT silently air-gap domains --
    # air-gap (empty allow-list) is opt-in only. (A *packages-only* policy with open
    # domains is rejected at construction; see the validator test below.)
    p = NetworkPolicy()
    assert p.domains == AccessList(mode="deny", entries=[])
    assert p.domains.permits("anything.example.com", wildcard=True) is True
    assert p.packages is None


def test_networkpolicy_roundtrips_with_deny_dimensions():
    p = NetworkPolicy(
        domains=AccessList(mode="deny", entries=["example.com", "*.example.com"]),
        packages=AccessList(mode="deny", entries=["tensorflow"]))
    assert NetworkPolicy.model_validate(p.model_dump()) == p


def test_networkpolicy_packages_none_roundtrips():
    p = NetworkPolicy(domains=AccessList(mode="deny", entries=[]))
    assert NetworkPolicy.model_validate(p.model_dump()).packages is None


# ---------------------------------------------------------------------------
# EgressPosture / NetworkPolicy.domain_posture tests
# ---------------------------------------------------------------------------

def test_domain_posture_open_for_empty_deny():
    # Empty deny-list = nothing denied = OPEN (also the NetworkPolicy() default).
    assert NetworkPolicy().domain_posture() is EgressPosture.OPEN
    assert NetworkPolicy(
        domains=AccessList(mode="deny", entries=[])).domain_posture() is EgressPosture.OPEN


def test_domain_posture_air_gap_for_empty_allow():
    # Empty allow-list = nothing permitted = AIR_GAP (the deliberate opt-in seal).
    assert NetworkPolicy(
        domains=AccessList(mode="allow", entries=[])).domain_posture() is EgressPosture.AIR_GAP


def test_domain_posture_filter_for_nonempty_list():
    # A non-empty list (either mode) is a real host filter = FILTER.
    assert NetworkPolicy(
        domains=AccessList(mode="allow", entries=["pypi.org"])).domain_posture() is EgressPosture.FILTER
    assert NetworkPolicy(
        domains=AccessList(mode="deny", entries=["evil.com"])).domain_posture() is EgressPosture.FILTER


def test_domain_posture_not_serialized_into_model_dump():
    # domain_posture() is a derived accessor, NOT a field: the dumped shape stays the
    # flat {"domains": {"mode","entries"}, "packages": ...} the proxy reads -- no
    # "posture"/enum key leaks in.
    p = NetworkPolicy(domains=AccessList(mode="allow", entries=["pypi.org"]),
                      packages=AccessList(mode="deny", entries=["torch"]))
    dumped = p.model_dump()
    assert set(dumped) == {"domains", "packages"}
    assert dumped == {
        "domains": {"mode": "allow", "entries": ["pypi.org"]},
        "packages": {"mode": "deny", "entries": ["torch"]},
    }
    # Calling the accessor does not change what gets dumped.
    _ = p.domain_posture()
    assert p.model_dump() == dumped


# ---------------------------------------------------------------------------
# Package-policy-requires-restricted-domains invariant
# ---------------------------------------------------------------------------

def test_package_policy_with_open_domains_is_rejected():
    # A package filter is only enforceable when the index is the sole reachable
    # package source; with domains OPEN a sealed agent can fetch a denied package
    # from any other host, so the combination is rejected at construction.
    with pytest.raises(pydantic.ValidationError):
        NetworkPolicy(packages=AccessList(mode="deny", entries=["x"]))  # default open domains


def test_package_policy_with_air_gap_domains_is_valid():
    p = NetworkPolicy(domains=AccessList(mode="allow", entries=[]),
                      packages=AccessList(mode="deny", entries=["x"]))
    assert p.domain_posture() is EgressPosture.AIR_GAP


def test_package_policy_with_filter_domains_is_valid():
    p = NetworkPolicy(domains=AccessList(mode="allow", entries=["pypi.org"]),
                      packages=AccessList(mode="deny", entries=["x"]))
    assert p.domain_posture() is EgressPosture.FILTER


def test_open_domains_without_packages_is_valid():
    p = NetworkPolicy()  # open domains, packages=None
    assert p.domain_posture() is EgressPosture.OPEN and p.packages is None


# ---------------------------------------------------------------------------
# AgentProducerSpec.network tests
# ---------------------------------------------------------------------------

def test_producer_spec_carries_network_policy_and_roundtrips():
    spec = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                             network=NetworkPolicy(domains=AccessList(mode="allow", entries=[]),
                                                   packages=AccessList(mode="deny", entries=["tensorflow"])))
    assert AgentProducerSpec.model_validate(spec.model_dump()) == spec


def test_producer_spec_network_defaults_sealed():
    """The build sandbox is SEALED by default: raw internet air-gapped (empty domains
    allow-list), and the package index refuses the harness deny list, so a task that
    sets no network still cannot install a serving framework."""
    net = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600)).network
    assert net is not None
    assert net.domains.mode == "allow" and net.domains.entries == []   # air-gap
    assert net.packages is not None and net.packages.mode == "deny"
    assert not net.packages.permits("vllm", normalize=True)
    assert not net.packages.permits("faster_whisper", normalize=True)
    assert net.packages.permits("torch", normalize=True)


def test_multitenant_slo_denies_serving_frameworks():
    """multitenant_slo layers a build-from-scratch package deny-list onto its sealed
    producer egress: serving frameworks blocked, the agent's pre-installed deps permitted
    (so the pyproject pre-scan passes). The list lives in the task, not the generic default."""
    from tasks.multitenant_slo.task import ARTIFACT

    net = ARTIFACT.producer.network
    assert net.domains.entries == []                                   # sealed (air-gap)
    assert net.packages is not None and net.packages.mode == "deny"
    assert not net.packages.permits("vllm") and not net.packages.permits("flash-attn")
    assert net.packages.permits("torch") and net.packages.permits("transformers")
    # PEP 503 normalization (as the pyproject pre-scan applies it) still catches "flash_attn".
    assert not net.packages.permits("flash_attn", normalize=True)


def test_a_task_policy_holds_no_vouched_host():
    """The hosts a seal vouches for are the harness's, kept outside the task-authored policy
    (``SealPolicy``): a ``NetworkPolicy`` has no field for them, so a task that names one is
    refused, and no copy can add one."""
    assert "trusted_internal_hosts" not in NetworkPolicy.model_fields
    with pytest.raises(pydantic.ValidationError):
        NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": ["x"]}, "trusted_internal_hosts": []})
    with pytest.raises(pydantic.ValidationError):
        NetworkPolicy(domains=AccessList(mode="allow", entries=["x"])).model_copy(
            update={"trusted_internal_hosts": ["10.0.0.1"]})


def test_a_seal_policy_is_a_validated_network_policy_and_the_hosts_the_harness_vouches_for():
    """What the seal's interceptor enforces is built and read back through validation only: its
    JSON, which the interceptor reads, holds the network policy and the vouched hosts, and a
    malformed one is refused."""
    network = NetworkPolicy(domains=AccessList(mode="allow", entries=["x", "gw"]))
    seal = SealPolicy(network=network, trusted_internal_hosts=("gw", "10.0.0.1"))
    assert SealPolicy.model_validate_json(seal.model_dump_json()) == seal
    assert json.loads(seal.model_dump_json()) == {
        "network": {"domains": {"mode": "allow", "entries": ["x", "gw"]}, "packages": None},
        "trusted_internal_hosts": ["gw", "10.0.0.1"]}
    assert SealPolicy(network=network).trusted_internal_hosts == ()
    for bad in ({"network": {"domains": {"mode": "open", "entries": []}}},
                {"network": network.model_dump(), "trusted_internal_hosts": [""]},
                {"network": network.model_dump(), "trusted_internal_hosts": "gw"},
                {"network": network.model_dump(), "vouched": ["gw"]},
                network.model_dump()):
        with pytest.raises(pydantic.ValidationError):
            SealPolicy.model_validate(bad)
