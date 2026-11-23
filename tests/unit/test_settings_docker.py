"""The settings of a Docker host (``[site.docker]``, ``[site.local]``) and the Slurm settings that
hold only when ``site.substrate`` is Slurm."""

from __future__ import annotations

from pathlib import Path

import pytest

from hls.settings import (DEFAULT_BUILDKIT_IMAGE, SLURM_SITE_FIELDS, SettingsError, load_settings, resolved,
                          settings_sha256)

SHA = "d" * 64
SLURM = "[site]\npartition = 'gpu'\nstore_root = '/s'\nnode_scratch = '/n'\nsecrets_path = '/x'\n"
DOCKER = ("[site]\nsubstrate = 'docker'\nstore_root = '/s'\nnode_scratch = '/n'\n"
          "secrets_path = '/x'\n")
BUILDKIT = "moby/buildkit@sha256:" + "b" * 64
DOCKER_TABLE = f"[site.docker]\nbuildkit_image = '{BUILDKIT}'\n"
LOCAL_TABLE = "[site.local]\nstate_dir = '/var/hls/state'\nkept_memory_gib = 8\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "hls.toml"
    path.write_text(text)
    return path


@pytest.mark.parametrize("ref", [BUILDKIT, "sha256:" + "c" * 64,
                                 "registry.example:5000/moby/buildkit:v1@sha256:" + "c" * 64])
def test_the_buildkit_image_is_pinned_by_its_digest(tmp_path, ref):
    table = DOCKER_TABLE.replace(BUILDKIT, ref)
    assert load_settings(_write(tmp_path, DOCKER + table + LOCAL_TABLE)).site.docker.buildkit_image == ref


def test_a_docker_site_reads_its_tables_with_their_defaults(tmp_path):
    site = load_settings(_write(tmp_path, DOCKER + DOCKER_TABLE + LOCAL_TABLE)).site
    assert site.substrate == "docker" and site.partition is None
    assert site.docker.socket == Path("/var/run/docker.sock") and site.docker.buildkit_image == BUILDKIT
    assert site.docker.owner_label is None and site.local.kept_memory_gib == 8.0
    assert site.docker.gateway_memory_gib == 2
    assert site.local.state_dir == Path("/var/hls/state")


@pytest.mark.parametrize("table", [DOCKER_TABLE, ""])
def test_a_docker_site_needs_its_local_table(tmp_path, table):
    with pytest.raises(SettingsError, match="site.substrate is docker needs site.local$"):
        load_settings(_write(tmp_path, DOCKER + table))


def test_a_docker_site_without_a_docker_table_builds_with_the_pinned_buildkit(tmp_path):
    site = load_settings(_write(tmp_path, DOCKER + LOCAL_TABLE)).site
    assert site.docker.buildkit_image == DEFAULT_BUILDKIT_IMAGE
    assert site.docker.buildkit_image.startswith("moby/buildkit@sha256:")
    assert site.docker.socket == Path("/var/run/docker.sock")


@pytest.mark.parametrize("line, key", [("partition = 'gpu'", "site.partition"), ("account = 'lab'", "site.account"),
                                       ("pasta_path = '/opt/pasta'", "site.pasta_path"),
                                       ("min_gpus_per_job = 0", "site.min_gpus_per_job")])
def test_a_docker_site_sets_no_slurm_setting(tmp_path, line, key):
    with pytest.raises(SettingsError, match=f"{key} are Slurm settings, which a site whose site.substrate is docker"):
        load_settings(_write(tmp_path, DOCKER + line + "\n" + DOCKER_TABLE + LOCAL_TABLE))


def test_every_slurm_only_setting_is_named():
    assert SLURM_SITE_FIELDS == ("partition", "qos", "qos_by_hours", "account", "gpu_gres", "min_gpus_per_job",
                                 "evaluation_socket_tasks", "sbatch_extra", "cpu_partition", "pasta_path")


@pytest.mark.parametrize("table", [DOCKER_TABLE, LOCAL_TABLE])
def test_a_slurm_site_sets_no_docker_table(tmp_path, table):
    with pytest.raises(SettingsError, match="are for a site whose site.substrate is docker, not slurm"):
        load_settings(_write(tmp_path, SLURM + table))


def test_a_slurm_site_still_needs_its_partition_with_the_same_message(tmp_path):
    with pytest.raises(SettingsError, match=r"site\.partition: Field required"):
        load_settings(_write(tmp_path, SLURM.replace("partition = 'gpu'\n", "")))


def test_a_docker_sites_settings_reload_from_what_a_plan_holds(tmp_path):
    from hls.settings import Settings

    settings = load_settings(_write(tmp_path, DOCKER + DOCKER_TABLE + LOCAL_TABLE))
    held = resolved(settings)
    assert not set(SLURM_SITE_FIELDS) & set(held["site"])
    assert Settings.model_validate(held) == settings


def test_a_slurm_sites_resolved_settings_hold_no_new_key(tmp_path):
    settings = load_settings(_write(tmp_path, SLURM))
    site = resolved(settings)["site"]
    assert not {"substrate", "docker", "local"} & set(site)
    assert list(site)[0] == "partition"
    docker = load_settings(_write(tmp_path, DOCKER + DOCKER_TABLE + LOCAL_TABLE))
    assert resolved(docker)["site"]["docker"]["buildkit_image"] == BUILDKIT
    assert settings_sha256(settings) != settings_sha256(docker)


@pytest.mark.parametrize("bad, why", [
    ("step_image = '/images/step.oci.tar'", "step_image"),
    ("owner_label = 'no equals sign'", "owner_label"),
    ("buildkit_image = ''", "buildkit_image"),
    ("gateway_memory_gib = 0", "gateway_memory_gib"),
    ("buildkit_image = 'moby/buildkit:v0.26.2'", "buildkit_image"),
    ("buildkit_image = 'moby/buildkit'", "buildkit_image"),
    ("buildkit_image = 'moby/buildkit@sha256:abc'", "buildkit_image"),
    ("socket = 'docker.sock'", "must be an absolute path"),
])
def test_the_docker_table_is_checked(tmp_path, bad, why):
    key = bad.split(" = ")[0]
    table = "\n".join(line for line in DOCKER_TABLE.splitlines() if not line.startswith(key + " ")) + "\n" + bad + "\n"
    with pytest.raises(SettingsError, match=why):
        load_settings(_write(tmp_path, DOCKER + table + LOCAL_TABLE))


@pytest.mark.parametrize("bad, why", [("kept_memory_gib = -1", "kept_memory_gib"),
                                      ("state_dir = 'state'", "must be an absolute path")])
def test_the_local_table_is_checked(tmp_path, bad, why):
    key = bad.split(" = ")[0]
    table = "\n".join(line for line in LOCAL_TABLE.splitlines() if not line.startswith(key + " ")) + "\n" + bad + "\n"
    with pytest.raises(SettingsError, match=why):
        load_settings(_write(tmp_path, DOCKER + DOCKER_TABLE + table))
