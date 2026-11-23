"""The spawner's seccomp profile, and the one container that gets it."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

import hls
from hls.substrate.docker import seccomp

#: The sha256 of ``seccomp/default.json`` as moby/profiles publishes it at ``seccomp/v0.2.3``.
MOBY_V0_2_3 = "536529b665dd0972c37bfb569f5d4ac8a53592e7b00752bc39ff063ca9864c74"
SRC = Path(hls.__file__).parent


def _allows_unshare(flags: int) -> bool:
    """Whether the spawner's ``unshare`` rule lets *flags* through: the masked comparison the
    kernel's filter makes, as libseccomp builds it from the rule."""
    [rule] = [r for r in seccomp.SPAWNER_RULES if r["names"] == ["unshare"]]
    [arg] = rule["args"]
    assert (arg["index"], arg["op"]) == (0, "SCMP_CMP_MASKED_EQ")
    return flags & arg["value"] == arg["valueTwo"]


def test_the_default_is_dockers_own_unchanged():
    assert hashlib.sha256(seccomp.DEFAULT_PROFILE.read_bytes()).hexdigest() == MOBY_V0_2_3


def test_the_spawner_profile_is_the_default_with_exactly_two_rules_added():
    default = json.loads(seccomp.DEFAULT_PROFILE.read_text())
    profile = json.loads(seccomp.spawner_profile())
    assert profile["syscalls"][:len(default["syscalls"])] == default["syscalls"]
    assert {k: v for k, v in profile.items() if k != "syscalls"} == {k: v for k, v in default.items()
                                                                       if k != "syscalls"}
    added = profile["syscalls"][len(default["syscalls"]):]
    assert added == list(seccomp.SPAWNER_RULES)
    assert [r["names"] for r in added] == [["unshare"], ["mount"]]
    # The default lets both through only to a container that holds CAP_SYS_ADMIN, which none does.
    for name in ("unshare", "mount"):
        rules = [r for r in default["syscalls"] if name in r["names"] and r["action"] == "SCMP_ACT_ALLOW"]
        assert rules and all(r.get("includes", {}).get("caps") == ["CAP_SYS_ADMIN"] for r in rules)


@pytest.mark.parametrize("flags, allowed", [
    (os.CLONE_NEWUSER, True),
    (os.CLONE_NEWNS, True),
    (os.CLONE_NEWPID, True),
    (os.CLONE_NEWUSER | os.CLONE_NEWNS, True),
    (os.CLONE_NEWNET, False),
    (os.CLONE_NEWUSER | os.CLONE_NEWNET, False),
    (os.CLONE_NEWIPC, False),
    (os.CLONE_NEWUTS, False),
    (0x02000000, False),  # CLONE_NEWCGROUP, which os does not name
    (0x00000080, False),  # CLONE_NEWTIME
    (os.CLONE_FS | os.CLONE_FILES | os.CLONE_SYSVSEM, True),
])
def test_unshare_makes_the_spawners_namespaces_and_no_other(flags, allowed):
    assert _allows_unshare(flags) is allowed


def _confinements() -> list[tuple[str, str, str]]:
    """Every keyword ``confinement=`` in a call in hls's code that names another value than
    ``default``: the file, the enclosing function and the value's source."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(func):
                if not isinstance(call, ast.Call):
                    continue
                for kw in call.keywords:
                    if kw.arg == "confinement" and not (isinstance(kw.value, ast.Constant)
                                                        and kw.value.value == "default"):
                        found.append((str(path.relative_to(SRC)), func.name, ast.unparse(kw.value)))
    return found


def test_only_the_agent_builds_sandbox_asks_for_the_spawner_confinement():
    """No other role of any stage gets the widened confinement: one call in hls's code asks for
    it, with the literal ``spawner``, in the agent build's start of its sandbox."""
    assert _confinements() == [("stages/agent_build.py", "start_sandbox", "'spawner'")]
