"""The documented hls.toml loads as a site's would, names every setting, its arms resolve, and
the documented secrets file holds every secret its routes name."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hls.arms import Arm, Route, adapter_for, check_granted_base_urls, resolve_arm
from hls.artifact.artifact import HighestAvailable
from hls.settings import (DockerSiteSettings, LocalSiteSettings, RetryPolicy, RunSettings, ShapeOverride, SiteSettings,
                          load_settings)
from hls.stages.images import registry_login_problem
from hls.substrate.secrets import load_secrets

EXAMPLES = Path(__file__).resolve().parents[2] / "docs" / "examples"
EXAMPLE = EXAMPLES / "hls.toml"


def test_the_documented_site_file_loads_and_its_arms_resolve():
    settings = load_settings(EXAMPLE)
    assert settings.site.partition and settings.site.store_root.is_absolute()
    assert sorted(settings.arms) == ["claude", "codex"]
    for name in settings.arms:
        record = resolve_arm(settings, name, task_effort=HighestAvailable())
        assert record.route in settings.routes


def test_the_documented_site_file_sets_only_the_required_site_settings():
    """A site that fills in only what it must gets every other setting's default: an example value
    left live, such as an account, would be sent to Slurm as the site's own. The example is a Slurm
    site's, which must also set its partition."""
    required = {f for f, info in SiteSettings.model_fields.items() if info.is_required()} | {"partition"}
    assert load_settings(EXAMPLE).site.model_fields_set == required
    assert required == {"partition", "store_root", "node_scratch", "secrets_path"}


@pytest.mark.parametrize("model", [SiteSettings, DockerSiteSettings, LocalSiteSettings, RunSettings, RetryPolicy,
                                   ShapeOverride, Route, Arm], ids=lambda m: m.__name__)
def test_the_documented_site_file_names_every_setting(model):
    """Each field is set in the example, or shown in a comment as ``# <field> = ...``, or, for a
    table such as ``site.docker`` or a table of tables such as ``run.retry``, as a ``[...<field>]``
    or ``[...<field>.<key>]`` header, so the file stays
    the complete reference a site copies from; the strict loader already refuses a set key that
    is no field."""
    text = EXAMPLE.read_text()
    missing = [f for f in model.model_fields
               if not re.search(rf"^(# )?({f} = |\[\S*\b{f}[.\]])", text, re.MULTILINE)]
    assert not missing


def test_the_documented_routes_name_secrets_the_documented_secrets_file_holds(tmp_path):
    """An operator who copies both examples gets a pair a plan accepts: every secret an arm's
    agent is granted is one the secrets file holds, and a granted base URL is one its harness
    takes."""
    settings = load_settings(EXAMPLE)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text((EXAMPLES / "secrets.toml").read_text())
    secrets.chmod(0o600)
    held = load_secrets(secrets).secrets
    for name in settings.arms:
        record = resolve_arm(settings, name, task_effort=HighestAvailable())
        grants = adapter_for(record).grants(record.route_table)
        assert set(grants.values()) <= set(held), name
        check_granted_base_urls(grants, held)


def test_the_documented_pair_logs_the_images_stage_in_to_a_registry_both_or_neither(tmp_path):
    """The registry login is the site's user name and the secrets file's token, both or neither
    (``registry_login_problem``): the two examples copied as they are leave both out, and pull
    anonymously."""
    settings = load_settings(EXAMPLE)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text((EXAMPLES / "secrets.toml").read_text())
    secrets.chmod(0o600)
    granted = load_secrets(secrets).grants.get("stage.images", {})
    assert registry_login_problem("images", settings.site.registry_username, granted) is None
