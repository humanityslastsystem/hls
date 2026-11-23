import pytest

from hls.cli import _parse_dotted_flags


def test_parse_dotted_space_and_equals():
    assert _parse_dotted_flags(["--substrate.backend", "modal"]) == {"substrate.backend": "modal"}
    assert _parse_dotted_flags(["--substrate.backend=modal"]) == {"substrate.backend": "modal"}


def test_parse_dotted_missing_value_fails():
    with pytest.raises(ValueError, match="no value"):
        _parse_dotted_flags(["--substrate.backend"])


def test_parse_dotted_equals_value_can_contain_equals():
    assert _parse_dotted_flags(["--env.agent.URL=http://a=b"]) == {"env.agent.URL": "http://a=b"}


def test_parse_dotted_empty_equals_value():
    assert _parse_dotted_flags(["--env.agent.X="]) == {"env.agent.X": ""}


def test_parse_dotted_rejects_bare_token():
    with pytest.raises(ValueError, match="unexpected CLI token"):
        _parse_dotted_flags(["notaflag"])


# ---------------------------------------------------------------------------
# CLI --set settability for flat NetworkPolicy
# ---------------------------------------------------------------------------


def test_cli_set_network_policy_flat_fields_via_apply_override():
    """The flat NetworkPolicy is reachable through dotted overrides: with a network whose
    packages field is set, ``network.packages.mode`` and ``network.packages.entries`` are
    each settable by ``apply_override``. A single ``apply_override`` cannot drill through a
    None network (``test_cli_set_network_null_parent_fails_loud``)."""
    from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable
    from hls.egress.policy import AccessList, NetworkPolicy
    from hls.override import apply_override

    spec = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                             network=NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList()))
    spec = apply_override(spec, "network.packages.mode", "deny")
    spec = apply_override(spec, "network.packages.entries", ["tensorflow"])
    assert spec.network.packages.mode == "deny"
    assert spec.network.packages.entries == ["tensorflow"]


def test_cli_set_network_domains_mode_via_apply_override():
    """artifact.producer.network.domains.mode is also settable."""
    from hls.override import apply_override
    from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable
    from hls.egress.policy import NetworkPolicy

    spec = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                             network=NetworkPolicy())
    spec = apply_override(spec, "network.domains.mode", "deny")
    assert spec.network.domains.mode == "deny"


def test_cli_set_network_packages_entries_comma_list():
    """A comma-separated value string parses to a list[str] via parse_scalar,
    which pydantic then validates as list[str] on AccessList.entries."""
    from hls.override import apply_override, parse_scalar
    from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable
    from hls.egress.policy import NetworkPolicy, AccessList

    entries_val = parse_scalar("tensorflow,numpy")  # -> ["tensorflow", "numpy"]
    assert entries_val == ["tensorflow", "numpy"]

    spec = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                             network=NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList()))
    spec = apply_override(spec, "network.packages.entries", entries_val)
    assert spec.network.packages.entries == ["tensorflow", "numpy"]


def test_cli_set_network_null_parent_fails_loud():
    """Drilling through a None network field fails with a clear KeyError -- this is
    expected behavior (the parent must be non-None). Document it in the tests so
    the limitation is explicit."""
    from hls.override import apply_override
    from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable
    import pytest

    # An explicit open-egress opt-out.
    spec = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600), network=None)
    with pytest.raises(KeyError, match="packages"):
        apply_override(spec, "network.packages.mode", "deny")
