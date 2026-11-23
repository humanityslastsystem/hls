"""The exit statuses of the harness's own layers around a role's command: each its own number,
none a harness program's beneath them, and the seal script's taken from the one module."""

from __future__ import annotations

import ast
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from hls.substrate import exits
from hls.substrate.apptainer.egress import SEAL_SCRIPT, seal_script
from hls.substrate.seal import SEAL_RULES

SRC = Path(exits.__file__).resolve().parents[1]
#: The layers' own statuses, by name.
LAYER_EXITS = {name: value for name, value in vars(exits).items() if name.endswith("_EXIT")}


def _named_exits() -> list[tuple[str, str, int]]:
    """Every module-level ``<NAME>_EXIT = <int>`` of the hls package, as (module, name, value)."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.endswith("_EXIT") and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)):
                found.append((str(path.relative_to(SRC)), node.targets[0].id, node.value.value))
    return found


def test_the_layers_exits_are_each_their_own_number():
    # The step wrapper's and the seal's: a role's status passes up through both, so a status names
    # the layer that ended the role only when no two share it.
    assert set(LAYER_EXITS) == {"PROCESS_CEILING_EXIT", "GRANT_MISMATCH_EXIT", "SEAL_REFUSED_EXIT",
                                "SEAL_UNBOUND_EXIT"}
    assert len(set(LAYER_EXITS.values())) == len(LAYER_EXITS), LAYER_EXITS


def test_no_harness_program_beneath_the_layers_exits_with_one_of_their_numbers():
    # Each layer's own number is defined once, in hls.substrate.exits, and no program the layers
    # run (the exec server, the commit point, the driver, a copy, a scan) shares one.
    named = _named_exits()
    owned = [(module, name) for module, name, _ in named if name in LAYER_EXITS]
    assert owned == [("substrate/exits.py", name) for name in LAYER_EXITS]
    clashes = [(module, name, value) for module, name, value in named
               if module != "substrate/exits.py" and value in LAYER_EXITS.values()]
    assert clashes == []


def test_the_seal_script_takes_its_exits_from_the_one_module():
    text = SEAL_SCRIPT.read_text()
    # No status of its own is written into the script; it exits with the variables its builder sets.
    assert not set(int(n) for n in re.findall(r"\bexit (\d+)", text)) & set(LAYER_EXITS.values())
    assert 'exit "$HLS_SEAL_REFUSED_EXIT"' in text and 'exit "$HLS_SEAL_UNBOUND_EXIT"' in text
    assert seal_script().endswith(text)
    preamble = seal_script().removesuffix(text)
    assert preamble == (f"HLS_SEAL_REFUSED_EXIT={exits.SEAL_REFUSED_EXIT}\n"
                        f"HLS_SEAL_UNBOUND_EXIT={exits.SEAL_UNBOUND_EXIT}\n"
                        f"HLS_SEAL_RULES={shlex.quote(SEAL_RULES.read_text())}\n")


@pytest.mark.parametrize("missing", ["HLS_SEAL_REFUSED_EXIT", "HLS_SEAL_UNBOUND_EXIT"])
def test_the_seal_script_refuses_to_run_without_its_exits(tmp_path, missing):
    """Run without a status its builder sets, the seal stops at its first lines, naming the
    variable, before it does anything: it never ends a role with a status no reader knows."""
    env = {"PATH": "/usr/bin:/bin", "HLS_SEAL_REFUSED_EXIT": str(exits.SEAL_REFUSED_EXIT),
           "HLS_SEAL_UNBOUND_EXIT": str(exits.SEAL_UNBOUND_EXIT)}
    del env[missing]
    p = subprocess.run(["bash", str(SEAL_SCRIPT)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout == ""
    [line] = p.stderr.splitlines()
    assert f"{missing}: is set by the seal builder" in line
