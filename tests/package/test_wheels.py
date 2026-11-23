from __future__ import annotations

import pytest

from hls.package.wheels import (COMPUTE_CAPABILITY, check_pin, container_env, dry_run_command,
                                missing_requirements, pin_name, wheel_command)
from hls.spec.resources import GPU_KINDS


def _item(name: str, version: str, url: str, *, direct: bool = False, vcs: dict | None = None) -> dict:
    info: dict = {"url": url}
    if vcs is not None:
        info["vcs_info"] = vcs
    return {"metadata": {"name": name, "version": version}, "download_info": info, "is_direct": direct}


def test_the_report_becomes_pins_of_what_the_base_lacks():
    report = {"version": "1", "install": [
        _item("subm", "0.1", "file:///workspace", direct=True),
        _item("httpx", "0.27.0", "https://files.pythonhosted.org/packages/httpx-0.27.0-py3-none-any.whl"),
        _item("tool", "1.0", "https://github.com/o/tool", direct=True,
              vcs={"vcs": "git", "commit_id": "a" * 40}),
        _item("blob", "2.0", "https://example.org/blob-2.0.tar.gz", direct=True),
    ]}
    assert missing_requirements(report, "/workspace") == [
        "httpx==0.27.0", f"tool @ git+https://github.com/o/tool@{'a' * 40}", "blob @ https://example.org/blob-2.0.tar.gz"]


def test_a_project_url_with_a_trailing_slash_is_still_the_project():
    report = {"install": [_item("subm", "0.1", "file:///workspace/", direct=True)]}
    assert missing_requirements(report, "/workspace") == []


@pytest.mark.parametrize("report", [
    {},
    {"install": "x"},
    {"install": [{"metadata": {"name": "a"}}]},
    {"install": [_item("httpx", "0.27.0", "https://x/y.whl") | {"is_direct": "no"}]},
])
def test_a_report_that_is_not_pips_shape_is_refused(report):
    with pytest.raises(ValueError, match="installation report"):
        missing_requirements(report, "/workspace")


@pytest.mark.parametrize("pin", [
    "--index-url http://evil/simple", "-r other.txt", "httpx==0.27.0 extra", "httpx==0.27.0\nother==1",
    "name @ -e .", "name @ ftp://host/x", "bad name==1", "x==", "x @ https://a b",
])
def test_a_pin_that_is_not_one_requirement_is_refused(pin):
    with pytest.raises(ValueError, match="pin"):
        check_pin(pin)


def test_a_crafted_report_cannot_smuggle_an_option_into_the_pins():
    report = {"install": [_item("--extra-index-url", "1", "https://x/y.whl")]}
    with pytest.raises(ValueError, match="pin"):
        missing_requirements(report, "/workspace")


def test_the_pip_commands():
    assert dry_run_command("/workspace", "/build-tmp/report.json") == [
        "pip", "install", "--dry-run", "--quiet", "--report", "/build-tmp/report.json", "/workspace",
        "--retries", "5", "--timeout", "30"]
    assert wheel_command("/workspace", "/build-tmp/pins.txt", "/wheels-out") == [
        "pip", "wheel", "--no-deps", "--wheel-dir", "/wheels-out", "/workspace", "-r", "/build-tmp/pins.txt",
        "--retries", "5", "--timeout", "30"]


def test_every_gpu_kind_has_a_compute_capability():
    assert set(COMPUTE_CAPABILITY) == set(GPU_KINDS)


@pytest.mark.parametrize("pin, name", [("foo-bar==1.0", "foo-bar"), ("Foo.Bar_2==1!2.0+x", "Foo.Bar_2"),
                                       ("baz @ https://example.org/baz-2-py3-none-any.whl", "baz"),
                                       ("q @ git+https://example.org/q@abc", "q")])
def test_a_pin_names_its_distribution_as_written(pin, name):
    assert pin_name(check_pin(pin)) == name


def test_a_pin_that_names_no_distribution_is_refused():
    with pytest.raises(ValueError, match="not a pin"):
        pin_name("@x")


def test_the_build_containers_compile_for_the_runs_gpu_with_the_stages_cpus():
    env = container_env(gpu_kind="H200", cpus=8)
    assert env == {"TMPDIR": "/build-tmp", "PIP_CACHE_DIR": "/build-tmp/pip-cache", "PIP_NO_INPUT": "1",
                   "MAX_JOBS": "8", "TORCH_CUDA_ARCH_LIST": "9.0"}
    assert "TORCH_CUDA_ARCH_LIST" not in container_env(gpu_kind=None, cpus=2)
