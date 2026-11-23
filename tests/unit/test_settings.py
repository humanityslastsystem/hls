from __future__ import annotations

import textwrap
import traceback
from pathlib import Path

import pytest

from hls.settings import (RetryPolicy, SettingsError, ShapeOverride, SiteSettings, load_settings, resolved,
                          settings_sha256)
from hls.spec.resources import ResourceShape

MINIMAL = """
[site]
partition = "gpu"
store_root = "/shared/group/u/hls-store"
node_scratch = "/tmp"
secrets_path = "/shared/group/u/.secrets.toml"
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "hls.toml"
    p.write_text(textwrap.dedent(text))
    return p


def test_minimal_loads_with_defaults(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL))
    assert s.site.partition == "gpu"
    assert s.run.gpu_policy == "max"
    assert s.run.heartbeat_sec == 30
    # With no table a stage takes its kind's count (hls.pipeline.stage.retry_policy).
    assert s.run.retry_for("agent_build", default_max=1) == RetryPolicy(max=1)
    assert s.run.retry_for("verify", default_max=2) == RetryPolicy(max=2)


def test_a_site_allows_150_s_per_role_launch_unless_it_says_less(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL)).site.step_launch_allowance_sec == 150
    fast = load_settings(_write(tmp_path, MINIMAL + "step_launch_allowance_sec = 20\n"))
    assert fast.site.step_launch_allowance_sec == 20
    with pytest.raises(SettingsError, match="step_launch_allowance_sec"):
        load_settings(_write(tmp_path, MINIMAL + "step_launch_allowance_sec = -1\n"))


def test_a_run_collects_an_agent_build_within_an_hour_unless_it_says_otherwise(tmp_path):
    """The time an agent build's stage keeps after the hard limit to collect it is a run setting,
    an hour by default; it is a positive count of seconds, set in the file or by an override."""
    assert load_settings(_write(tmp_path, MINIMAL)).run.collect_allowance_sec == 3600
    longer = load_settings(_write(tmp_path, MINIMAL + "\n[run]\ncollect_allowance_sec = 7200\n"))
    assert longer.run.collect_allowance_sec == 7200
    overridden = load_settings(_write(tmp_path, MINIMAL), {"run.collect_allowance_sec": "5000"})
    assert overridden.run.collect_allowance_sec == 5000
    for value in (0, -1):
        with pytest.raises(SettingsError, match="run.collect_allowance_sec"):
            load_settings(_write(tmp_path, MINIMAL + f"\n[run]\ncollect_allowance_sec = {value}\n"))


def test_a_site_leaves_the_commands_accounting_to_the_kernel_unless_it_says_which(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL)).site.nproc_accounting is None
    per_namespace = load_settings(_write(tmp_path, MINIMAL + 'nproc_accounting = "user_namespace"\n'))
    assert per_namespace.site.nproc_accounting == "user_namespace"
    with pytest.raises(SettingsError, match="nproc_accounting"):
        load_settings(_write(tmp_path, MINIMAL + 'nproc_accounting = "cgroup"\n'))


def test_a_site_declares_its_grading_nodes_per_uid_limits_or_takes_the_least_measured(tmp_path):
    site = load_settings(_write(tmp_path, MINIMAL)).site
    # Namespaces: a round figure well under the least measured (about 4.1 million), so a node
    # of less memory still holds its graded servers; inotify: the measured 128 and 1,048,576.
    assert (site.grading_node_namespaces, site.grading_node_inotify_instances, site.grading_node_inotify_watches) == (
        1_048_576, 128, 1_048_576)
    larger = load_settings(_write(tmp_path, MINIMAL + "grading_node_namespaces = 8254578\n")).site
    assert larger.grading_node_namespaces == 8_254_578
    for setting in ("grading_node_namespaces", "grading_node_inotify_instances", "grading_node_inotify_watches"):
        with pytest.raises(SettingsError, match=setting):
            load_settings(_write(tmp_path, MINIMAL + f"{setting} = 0\n"))


@pytest.mark.parametrize("table", ["substrate", "agent", "accelerators", "env"])
def test_an_old_orchestrator_table_is_refused(tmp_path, table):
    with pytest.raises(SettingsError, match=rf"unknown setting {table}$"):
        load_settings(_write(tmp_path, MINIMAL + f"\n[{table}]\nx = 1\n"))


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (MINIMAL + "\n[retry.agent_build]\nmax = 3\n", "retry"),
        ('gpu_policy = "split"\n' + MINIMAL, "gpu_policy"),
        (MINIMAL + "\n[runs]\ngpu_policy = 'split'\n", "runs"),
    ],
    ids=["retry_table", "run_key_at_top_level", "misspelled_table"],
)
def test_unknown_top_level_key_is_rejected(tmp_path, text, key):
    with pytest.raises(SettingsError, match=rf"unknown setting {key}\b"):
        load_settings(_write(tmp_path, text))


def test_unknown_site_key_names_the_path(tmp_path):
    with pytest.raises(SettingsError, match=r"unknown setting site\.partitoin"):
        load_settings(_write(tmp_path, MINIMAL + 'partitoin = "x"\n'))


def test_unknown_nested_retry_key_names_the_full_path(tmp_path):
    text = MINIMAL + "\n[run.retry.agent_build]\nmaxx = 3\n"
    with pytest.raises(SettingsError, match=r"unknown setting run\.retry\.agent_build\.maxx"):
        load_settings(_write(tmp_path, text))


def test_retry_table_and_stage_name_charset(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL + "\n[run.retry.agent_build]\nmax = 0\nenabled = false\n"))
    assert s.run.retry_for("agent_build", default_max=1) == RetryPolicy(max=0, enabled=False)
    with pytest.raises(SettingsError, match="run.retry"):
        load_settings(_write(tmp_path, MINIMAL + "\n[run.retry.Agent-Build]\nmax = 1\n"))


def test_retry_table_without_max_takes_the_stage_default(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL + "\n[run.retry.verify]\nenabled = false\n"))
    assert s.run.retry_for("verify", default_max=2) == RetryPolicy(max=2, enabled=False)
    s = load_settings(_write(tmp_path, MINIMAL + "\n[run.retry.agent_build]\nbackoff_sec = 5\n"))
    assert s.run.retry_for("agent_build", default_max=1) == RetryPolicy(max=1, backoff_sec=5)


def test_resources_overrides_parse(tmp_path):
    text = MINIMAL + textwrap.dedent("""
        [resources.multitenant_slo.verify]
        gpus = 1
        gpu_kind = "H200"
        cpus = 8
        memory_gib = 120
        time_sec = 3600
        """)
    s = load_settings(_write(tmp_path, text))
    assert s.resources["multitenant_slo"]["verify"].memory_gib == 120


def test_a_resources_table_may_set_any_one_field(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL + "\n[resources.multitenant_slo.verify]\ntime_sec = 7200\n"))
    override = s.resources["multitenant_slo"]["verify"]
    assert (override.time_sec, override.cpus, override.gpus) == (7200, None, None)
    with pytest.raises(SettingsError, match=r"resources\.multitenant_slo\.verify: .*sets at least one of"):
        load_settings(_write(tmp_path, MINIMAL + "\n[resources.multitenant_slo.verify]\n"))
    with pytest.raises(SettingsError, match=r"unknown setting resources\.multitenant_slo\.verify\.memory"):
        load_settings(_write(tmp_path, MINIMAL + "\n[resources.multitenant_slo.verify]\nmemory = 8\n"))


def test_an_override_of_zero_gpus_clears_the_kind():
    gpu = ResourceShape(gpus=2, gpu_kind="H200", cpus=8, memory_gib=16, time_sec=60)
    assert ShapeOverride(gpus=0).apply(gpu) == gpu.model_copy(update={"gpus": 0, "gpu_kind": None})
    assert ShapeOverride(cpus=4).apply(gpu) == gpu.model_copy(update={"cpus": 4})
    with pytest.raises(ValueError, match="gpu_kind must be set exactly when gpus > 0"):
        ShapeOverride(gpus=1).apply(ResourceShape(cpus=1, memory_gib=1, time_sec=60))


def test_the_warmers_run_one_at_a_time_under_one_name_by_default(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL))
    assert (s.site.warmers_one_at_a_time, s.site.warmer_job_name) == (True, "hls-warmer")
    text = MINIMAL + 'warmer_job_name = "hls-warm.kl"\nwarmers_one_at_a_time = false\n'
    s = load_settings(_write(tmp_path, text))
    assert (s.site.warmers_one_at_a_time, s.site.warmer_job_name) == (False, "hls-warm.kl")
    with pytest.raises(SettingsError, match=r"site\.warmer_job_name: "):
        load_settings(_write(tmp_path, MINIMAL + 'warmer_job_name = "two words"\n'))


def test_a_repeated_sbatch_option_is_refused_when_settings_load(tmp_path):
    # sbatch keeps only the last of an option given twice, so the first would be dropped unseen.
    text = MINIMAL + 'sbatch_extra = ["--exclude=n[001,005]", "--constraint=h200", "--exclude=n047"]\n'
    with pytest.raises(SettingsError, match=r"site\.sbatch_extra: it gives --exclude more than once; sbatch keeps "
                                            "only the last"):
        load_settings(_write(tmp_path, text))
    once = MINIMAL + 'sbatch_extra = ["--exclude=n[001,005],n047", "--constraint=h200"]\n'
    assert load_settings(_write(tmp_path, once)).site.sbatch_extra == ["--exclude=n[001,005],n047",
                                                                      "--constraint=h200"]


def test_cli_overrides_apply_and_are_strict(tmp_path):
    p = _write(tmp_path, MINIMAL)
    s = load_settings(p, {"run.gpu_policy": "split", "run.retry.package.max": "5"})
    assert s.run.gpu_policy == "split" and s.run.retry_for("package", default_max=2).max == 5
    with pytest.raises(SettingsError, match=r"unknown setting run\.gpu_polcy"):
        load_settings(p, {"run.gpu_polcy": "split"})
    with pytest.raises(SettingsError, match="gpu_policy"):
        load_settings(p, {"run.gpu_policy": "sometimes"})


def test_missing_required_site_key(tmp_path):
    with pytest.raises(SettingsError, match=r"site\.partition: Field required"):
        load_settings(_write(tmp_path, "[site]\nstore_root='/s'\nnode_scratch='/tmp'\nsecrets_path='/x'\n"))


def test_a_site_has_no_name(tmp_path):
    """A site has no name, in hls.toml or on the command line; a name is
    refused as every unknown setting is."""
    assert "name" not in SiteSettings.model_fields
    with pytest.raises(SettingsError, match=r": unknown setting site\.name$"):
        load_settings(_write(tmp_path, MINIMAL.replace("[site]\n", '[site]\nname = "site-a"\n')))
    with pytest.raises(SettingsError, match=r": unknown setting site\.name$"):
        load_settings(_write(tmp_path, MINIMAL), {"site.name": "site-a"})


def test_an_empty_cpu_partition_is_a_settings_error(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL + 'cpu_partition = "cpu"\n')).site.cpu_partition == "cpu"
    with pytest.raises(SettingsError, match=r"site\.cpu_partition: String should have at least 1 character"):
        load_settings(_write(tmp_path, MINIMAL + 'cpu_partition = ""\n'))


def test_the_registry_user_name_is_an_optional_plain_site_setting(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL)).site.registry_username is None
    s = load_settings(_write(tmp_path, MINIMAL + 'registry_username = "registry-user"\n'))
    assert s.site.registry_username == "registry-user" and resolved(s)["site"]["registry_username"] == "registry-user"
    for bad in ('""', '"two words"', '"tab\\there"'):
        with pytest.raises(SettingsError, match=r"site\.registry_username: "):
            load_settings(_write(tmp_path, MINIMAL + f"registry_username = {bad}\n"))


def test_the_lease_ttl_is_not_tied_to_the_runner_heartbeat(tmp_path):
    # The lease is renewed every quarter of its TTL; heartbeat_sec is only the runner's status
    # interval, so a 60 s heartbeat goes with the default 120 s lease.
    s = load_settings(_write(tmp_path, MINIMAL + "\n[run]\nheartbeat_sec = 60\nallocation_dead_after_sec = 180\n"))
    assert (s.run.heartbeat_sec, s.run.lease_ttl_sec) == (60, 120)
    with pytest.raises(SettingsError, match="lease_ttl_sec"):
        load_settings(_write(tmp_path, MINIMAL + "\n[run]\nlease_ttl_sec = 0\n"))


def test_resolved_is_plain_json_and_hash_is_stable(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL))
    r = resolved(s)
    assert r["site"]["secrets_path"] == "/shared/group/u/.secrets.toml"
    assert settings_sha256(s) == settings_sha256(load_settings(_write(tmp_path, MINIMAL)))


SHAPE = "cpus = 1\nmemory_gib = 1\ntime_sec = 60\n"


@pytest.mark.parametrize(
    ("table", "path"),
    [
        (r'[run.retry."agent_build\n"]' + "\nmax = 1\n", r'run\.retry\."agent_build\\n": '),
        ("[run.retry.Agent-Build]\nmax = 1\n", r"run\.retry\.Agent-Build: "),
        ('[resources.multitenant_slo."Verify Stage!"]\n' + SHAPE, r'resources\.multitenant_slo\."Verify Stage!": '),
        (r'[resources.multitenant_slo."verify\n"]' + "\n" + SHAPE, r'resources\.multitenant_slo\."verify\\n": '),
        ('[resources."../x".verify]\n' + SHAPE, r'resources\."\.\./x": '),
        (r'[resources."slo\n".verify]' + "\n" + SHAPE, r'resources\."slo\\n": '),
    ],
    ids=["retry_newline", "retry_charset", "resources_stage", "resources_stage_newline", "task_dotdot",
         "task_newline"],
)
def test_stage_and_task_keys_are_checked_and_named_by_dotted_path(tmp_path, table, path):
    with pytest.raises(SettingsError, match=path):
        load_settings(_write(tmp_path, MINIMAL + "\n" + table))


def test_site_paths_expand_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    s = load_settings(_write(tmp_path, MINIMAL.replace('"/shared/group/u/hls-store"', '"~/hls-store"')))
    assert s.site.store_root == tmp_path / "home" / "hls-store"
    assert resolved(s)["site"]["store_root"] == str(tmp_path / "home" / "hls-store")


@pytest.mark.parametrize("field", ["store_root", "node_scratch", "secrets_path"])
@pytest.mark.parametrize("value", ["relative/dir", "./x", "~no-such-user-hls/x"])
def test_site_paths_must_be_absolute(tmp_path, field, value):
    p = _write(tmp_path, MINIMAL)
    with pytest.raises(SettingsError, match=rf"site\.{field}: .*absolute"):
        load_settings(p, {f"site.{field}": value})


def test_cli_overrides_take_the_type_of_the_setting(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL), {"site.account": "12345", "site.qos": "a,b",
                                                  "run.poll_sec": "10"})
    assert s.site.account == "12345"
    assert s.site.qos == "a,b"
    assert s.run.poll_sec == 10
    with pytest.raises(SettingsError, match=r"run\.poll_sec"):
        load_settings(_write(tmp_path, MINIMAL), {"run.poll_sec": "ten"})


def test_allocation_timing_settings(tmp_path):
    s = load_settings(_write(tmp_path, MINIMAL))
    assert (s.run.allocation_overhead_sec, s.run.allocation_dead_after_sec) == (900, 600)
    with pytest.raises(SettingsError, match="allocation_dead_after_sec"):
        load_settings(_write(tmp_path, MINIMAL + "\n[run]\nheartbeat_sec = 60\nlease_ttl_sec = 600\n"
                                                    "allocation_dead_after_sec = 120\n"))


def test_gpu_gres_keys_are_known_kinds(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL)).site.gpu_gres["L40"] == "l40"
    with pytest.raises(SettingsError, match="unknown gpu kind"):
        load_settings(_write(tmp_path, MINIMAL + '\n[site.gpu_gres]\nH20 = "h20"\n'))


def test_the_runner_keeps_4_gib_of_each_stages_memory_by_default(tmp_path):
    assert load_settings(_write(tmp_path, MINIMAL)).run.runner_memory_gib == 4
    assert load_settings(_write(tmp_path, MINIMAL + "\n[run]\nrunner_memory_gib = 0\n")).run.runner_memory_gib == 0


def test_a_build_whose_sandbox_runs_out_of_memory_is_graded_on_its_last_commit_by_default(tmp_path):
    """Grading on the last commit is the default; a site's file or a
    command's override turns it off."""
    assert load_settings(_write(tmp_path, MINIMAL)).run.grade_after_sandbox_oom is True
    off = MINIMAL + "\n[run]\ngrade_after_sandbox_oom = false\n"
    assert load_settings(_write(tmp_path, off)).run.grade_after_sandbox_oom is False
    overridden = load_settings(_write(tmp_path, MINIMAL), {"run.grade_after_sandbox_oom": "false"})
    assert overridden.run.grade_after_sandbox_oom is False


#: A relay's base URL and an API key, as an operator might paste them into hls.toml.
PASTED_URL, PASTED_KEY = "https://relay.internal.example/anthropic", "sk-ant-api03-PASTEDKEYVALUE"


@pytest.mark.parametrize("routes, value", [
    (f'[routes.relay]\ncredential = "relay_token"\nbase_url = "{PASTED_URL}"\n', PASTED_URL),
    (f'[routes.relay]\ncredential = "{PASTED_KEY}"\n', PASTED_KEY),
    (f'[routes.relay]\ncredential = "relay_token"\nurl = "{PASTED_URL}"\n', PASTED_URL),
], ids=["base_url", "credential", "unknown-key"])
def test_a_settings_error_shows_no_pasted_value_even_in_its_traceback(tmp_path, routes, value):
    # The CLI prints the traceback of a settings error that reaches it, so the error must not
    # carry the validator's own error, which shows each refused input.
    with pytest.raises(SettingsError, match=r"routes\.relay") as e:
        load_settings(_write(tmp_path, MINIMAL + "\n" + routes))
    shown = "".join(traceback.format_exception(e.value))
    assert value not in shown and "relay.internal.example" not in shown


def test_a_refusal_of_a_whole_table_reads_without_pydantics_prefix(tmp_path):
    path = _write(tmp_path, MINIMAL + 'qos = "debug"\nqos_by_hours = { normal = 24, long = 168 }\n')
    with pytest.raises(SettingsError) as e:
        load_settings(path)
    assert str(e.value) == (f"{path}: site: site.qos 'debug' is not one of site.qos_by_hours ['long', 'normal'], "
                            "which must name every QOS a job may run under with its limit")
    run = _write(tmp_path, MINIMAL + "\n[run]\nheartbeat_sec = 60\nallocation_dead_after_sec = 120\n")
    with pytest.raises(SettingsError) as e:
        load_settings(run)
    assert str(e.value) == f"{run}: run: allocation_dead_after_sec must be at least 3 x heartbeat_sec"


def test_a_config_that_is_not_toml_is_refused_by_position_alone(tmp_path):
    """The decoder's own words can quote a character of a value, which may be a pasted credential,
    so the refusal keeps only where it fails, and chains nothing that holds the text."""
    path = tmp_path / "hls.toml"
    path.write_text('[site]\nstore_root = "/scratch/sk-\x01-0123"\n')
    with pytest.raises(SettingsError) as info:
        load_settings(path)
    assert str(info.value) == f"{path}: not valid TOML (at line 2, column 27)"
    chain, err = [], info.value
    while err is not None:
        chain.append(str(err))
        err = err.__cause__ or err.__context__
    assert not any("\\x01" in shown or "Illegal" in shown for shown in chain), chain
