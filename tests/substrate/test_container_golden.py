"""The Slurm launcher runs every container role, and the Apptainer image runtime every pin and
build role, with exactly the command line and declared environment the stage built for them
before the launcher realised containers: ``container_golden.json.gz`` holds those, recorded from
the stage tests, each container case with what the stage knew (its spec, its seal's files and
limits, the site's pasta and upstream proxy, the firewall tools' directories and the plan's
interpreter). Paths are made neutral (``/tmp/golden``, ``/checkout``); the seal script, the
shell programs and each seal's policy appear once, by name."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from hls.egress.policy import SealPolicy
from hls.spec.resources import ResourceShape
from hls.substrate import seal as seal_mod
from hls.substrate.apptainer.argv import START_MARKER, ApptainerRuntime
from hls.substrate.apptainer.egress import SEAL_TOOL_PATH_VAR, seal_script
from hls.substrate.apptainer.images import BUILD_SHELL, EXTRACT_SHELL, LOGIN_SHIM, ApptainerImages
from hls.substrate.base import RoleSpec
from hls.substrate.container import ContainerRole, ContainerSpec, Mount, PinnedImage, SealPlan
from hls.substrate.seal import seal_files
from hls.substrate.slurm.launcher import step_spec

GOLDEN = json.loads(gzip.decompress(Path(__file__).with_name("container_golden.json.gz").read_bytes()))
SHAPE = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
_SHELLS = {"<extract-shell>": EXTRACT_SHELL, "<build-shell>": BUILD_SHELL, "<login-shim>": LOGIN_SHIM}


def _expand(value: str, root: Path) -> str:
    """*value* as the stage recorded it, with its seal script, shells and policy restored and
    ``/tmp/golden`` mapped under *root*."""
    if value == "<seal-script>":
        return seal_script()
    if value in _SHELLS:
        return _SHELLS[value]
    if value.startswith("<policy:"):
        return GOLDEN["policies"][value[len("<policy:"):-1]]
    return value.replace("/tmp/golden/", f"{root}/")


def _spec(case: dict, root: Path) -> ContainerSpec:
    s = case["spec"]
    seal = s["seal"]
    return ContainerSpec(
        image=PinnedImage(ref=_expand(s["image"], root), file_sha256=s["file_sha256"], check_sha256=s["check_sha256"],
                          contestant=s["contestant_image"]),
        command=tuple(_expand(a, root) for a in s["command"]), workdir=Path(_expand(s["workdir"], root)),
        mounts=tuple(Mount(Path(_expand(src, root)), _expand(dst, root), read_only=ro) for src, dst, ro in s["mounts"]),
        gpus=s["gpus"], cwd=None if s["cwd"] is None else _expand(s["cwd"], root),
        contestant_payload=s["contestant_payload"],
        seal=None if seal is None else SealPlan(
            policy=SealPolicy.model_validate_json(_expand(seal["policy"], root)), index=seal["index"],
            ingress_port=seal["ingress_port"], deny_list=tuple(seal["deny_list"])))


def test_the_golden_values_hold_no_path_of_the_machine_they_were_recorded_on():
    """Every host path is under ``/tmp/golden``, which each case maps under its own ``tmp_path``,
    so no case reads or writes a fixed path."""
    text = json.dumps(GOLDEN)
    assert not re.search(r"(?<![A-Za-z0-9_.-])/tmp/(?!golden/)", text)
    assert not [word for word in ("/home/", "pytest-of") if word in text]


def test_the_golden_values_cover_every_kind_of_role():
    containers = GOLDEN["container"]
    assert len(containers) > 100 and len(GOLDEN["extract"]) > 10 and GOLDEN["build"]
    assert any(c["spec"]["seal"] and c["spec"]["seal"]["index"] for c in containers)
    assert any(c["spec"]["seal"] and c["spec"]["seal"]["ingress_port"] for c in containers)
    assert any(c["spec"]["contestant_payload"] for c in containers) and any(c["spec"]["gpus"] for c in containers)
    assert any(c["upstream_proxy"] for c in containers) and any(c["tool_path"] for c in containers)
    assert any(len(b["argv"]) > 4 and b["argv"][3] == "registry-login" for b in GOLDEN["build"])
    assert any(b["argv"][-1] for b in GOLDEN["build"])


@pytest.mark.parametrize("index", range(len(GOLDEN["container"])))
def test_a_container_step_is_what_the_stage_built(index, monkeypatch, tmp_path):
    case = GOLDEN["container"][index]
    root = tmp_path
    monkeypatch.setattr(seal_mod, "EGRESS_TRANSPARENT_PORT", case["ports"][0])
    monkeypatch.setattr(seal_mod, "EGRESS_INDEX_PORT", case["ports"][1])
    if case["tool_path"] is None:
        monkeypatch.delenv(SEAL_TOOL_PATH_VAR, raising=False)
    else:
        monkeypatch.setenv(SEAL_TOOL_PATH_VAR, _expand(case["tool_path"], root))
    spec = _spec(case, root)
    files = None if case["seal_dir"] is None else seal_files(Path(_expand(case["seal_dir"], root)))
    ucounts = dict(case["ucounts"] or [])
    env_file = Path(_expand(case["env_file"], root))
    role = RoleSpec(name="golden", argv=(), shape=SHAPE, owner="harness", log_dir=tmp_path / "logs",
                    start_marker=START_MARKER,
                    container=ContainerRole(spec=spec, env_file=env_file, seal_files=files, ucounts=ucounts))
    step = step_spec(role, runtime=ApptainerRuntime(), python=_expand(case["python"], root),
                     pasta=None if case["pasta"] is None else Path(_expand(case["pasta"], root)),
                     upstream_proxy=case["upstream_proxy"])
    assert list(step.argv) == [_expand(a, root) for a in case["argv"]]
    assert dict(step.env) == {k: _expand(v, root) for k, v in case["env"].items()}
    assert (step.env_file, step.env_fd, step.start_marker, step.container) == (env_file, case["env_fd"],
                                                                               case["start_marker"], None)


@pytest.mark.parametrize("index", range(len(GOLDEN["extract"])))
def test_a_pin_role_is_the_extraction_the_stage_built(index, tmp_path):
    case = GOLDEN["extract"][index]
    argv = [_expand(a, tmp_path) for a in case["argv"]]
    image_file, sha256, apptainer, dest, copy = argv[4:9]
    own = Path(copy).parent
    own.mkdir(parents=True)
    images = ApptainerImages(Path(dest).parent.parent, apptainer=apptainer)
    got, env = images.pin_role(Path(image_file), sha256=sha256, own=own)
    assert got[:7] + got[8:] == argv[:7] + argv[8:]
    assert got[7].rsplit("-", 1)[0] == dest.rsplit("-", 1)[0] and dest.endswith("-<uuid>")
    assert env == {k: _expand(v, tmp_path) for k, v in case["env"].items()}


@pytest.mark.parametrize("index", range(len(GOLDEN["build"])))
def test_a_build_role_is_the_build_the_stage_built(index, tmp_path):
    case = GOLDEN["build"][index]
    argv = [_expand(a, tmp_path) for a in case["argv"]]
    login = argv[3] == "registry-login"
    apptainer, out, recipe, digest, parent, copy, sha256 = argv[8 if login else 4:]
    Path(out).parent.mkdir(parents=True)
    got, env = ApptainerImages(tmp_path, apptainer=apptainer).build_role(
        Path(recipe), Path(out), digest=Path(digest), parent=(Path(parent), sha256, Path(copy)) if parent else None,
        login=login, contestant=False)
    assert got == argv
    assert env == {k: _expand(v, tmp_path) for k, v in case["env"].items() if k.startswith("APPTAINER_")}
