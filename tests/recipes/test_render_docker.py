from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from hls.recipes import catalog, pins
from hls.recipes.model import BaseImage, ImageDecl, LayerImage, RegistryImage, UvPin, parent_of, registry_of
from hls.recipes.model import CONSTRAINTS
from hls.recipes.render import TASKS_DIR, environment, files_for, post_script, recipe_key, render_def
from hls.recipes.render_docker import (DOCKERFILE_SYNTAX, LOCKS_CONTEXT, PARENT_CONTEXT, SCRIPT_END, build_inputs,
                                       dockerfile_key, heredoc, parent_context, quoted, render_dockerfile)
from hls.stages.images import image_keys
from tests.recipes.test_def_keys import FIXTURES, TEST_IMAGES

UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
#: Each image a task builds for itself, a layer whose locks and files are in the task's
#: directory, with its task.
TASK_IMAGES: list[tuple[str, LayerImage]] = [
    (task, d) for task in catalog.task_keys() for d in catalog.task_images(task).decls
    if isinstance(d, LayerImage) and d.owner is not None]
#: Every image the catalog and the tasks declare, and the test images that reach the renderer's
#: other branches, with the lock directory and uv each renders with.
IMAGES: list[tuple[ImageDecl, Path, UvPin]] = [
    *((d, pins.LOCKS_DIR, pins.UV) for d in catalog.shared_decls()),
    *((d, pins.LOCKS_DIR, pins.UV) for _, d in TASK_IMAGES),
    *((d, FIXTURES, UV) for d in TEST_IMAGES),
]
IDS = [*(f"catalog-{d.name}" for d in catalog.shared_decls()), *(f"{task}-{d.name}" for task, d in TASK_IMAGES),
       *(f"test-{d.name}" for d in TEST_IMAGES)]
# What a parent's file is called in a .def rendered here.
PARENT_SIF = Path("/node/parent.sif")
#: The registry image each task's layer that builds on one starts at, by the layer's name.
REGISTRY: dict[str, RegistryImage] = {}
for _task in catalog.task_keys():
    _decls = {d.name: d for d in catalog.task_images(_task).decls}
    REGISTRY.update({name: found for name, d in _decls.items() if (found := registry_of(d, _decls)) is not None})


def _def(decl: ImageDecl, locks: Path, uv: UvPin) -> str:
    return render_def(decl, parent_sif=PARENT_SIF if parent_of(decl) else None, locks=locks, uv=uv,
                      registry=REGISTRY.get(decl.name))


def _dockerfile(decl: ImageDecl, locks: Path, uv: UvPin) -> str:
    return render_dockerfile(decl, locks=locks, uv=uv, registry=REGISTRY.get(decl.name))


def _sections(text: str) -> dict[str, list[str]]:
    """Each %section of a .def to its lines, each without the section's indent."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.split("\n"):
        if line.startswith("%"):
            current = line[1:]
            out[current] = []
        elif current is not None:
            out[current].append(line.removeprefix("    "))
    return {name: lines[:-1] if lines and lines[-1] == "" else lines for name, lines in out.items()}


def _run_body(text: str) -> tuple[int, str]:
    """The index of a Dockerfile's RUN line and the text of its heredoc."""
    lines = text.split("\n")
    start = lines.index(f"RUN <<'{SCRIPT_END}'")
    end = lines.index(SCRIPT_END, start)
    return start, "\n".join(lines[start + 1:end])


def _docker_env(word: str) -> str:
    """A Dockerfile ENV value as the frontend reads it, for the two forms the renderer writes."""
    if word.startswith('"'):
        assert word.endswith('"')
        return re.sub(r'\\(["$\\])', r"\1", word[1:-1])
    return word


@pytest.mark.parametrize("decl, locks, uv", IMAGES, ids=IDS)
def test_both_wrappers_run_one_script_on_one_file_set(decl, locks, uv):
    """The .def's %post and the Dockerfile's RUN heredoc are the one text post_script derives,
    and both copy the same lock files to the same paths."""
    script = post_script(decl, locks=locks, uv=uv, registry=REGISTRY.get(decl.name))
    definition, dockerfile = _def(decl, locks, uv), _dockerfile(decl, locks, uv)
    sections = _sections(definition)
    assert "\n".join(sections["post"]) == script
    assert _run_body(dockerfile)[1] == script
    in_def = {tuple(line.split(" ")) for line in sections.get("files", [])}
    context = build_inputs(decl, locks_dir=locks).locks
    copies = [line.split(" ") for line in dockerfile.split("\n") if line.startswith("COPY ")]
    assert all(c[1] == f"--from={LOCKS_CONTEXT}" for c in copies)
    in_docker = {(str(context[name]), path) for _, _, name, path in copies}
    expected = {(str(host), path) for path, host in files_for(decl, locks).items()}
    assert in_def == in_docker == expected and expected
    assert set(context.values()) == set(files_for(decl, locks).values())


@pytest.mark.parametrize("decl, locks, uv", IMAGES, ids=IDS)
def test_both_wrappers_build_in_the_environment_of_the_image_below_and_set_the_same_one(decl, locks, uv):
    """The build script sees the variables of the image it builds on and nothing else of
    hls's: apptainer sources the image's /.singularity.d/env files for %post, and a RUN sees the
    image configuration's variables. So nothing before the RUN sets or clears a variable, a base
    sets its environment after its script in both wrappers, a layer that declares its whole
    environment sets those variables in their order, and any other child sets none."""
    definition, dockerfile = _def(decl, locks, uv), _dockerfile(decl, locks, uv)
    lines = dockerfile.split("\n")
    start, _ = _run_body(dockerfile)
    assert [line.split(" ")[0] for line in lines[1:start]] == (
        ["FROM", "LABEL", *(["COPY"] * (start - 3))] if isinstance(decl, BaseImage) else ["FROM", *(["COPY"] * (start - 2))])
    sections = list(_sections(definition))
    env = [line for line in lines if line.startswith("ENV ")]
    assert all(lines.index(line) > start for line in env)
    if not isinstance(decl, BaseImage):
        assert environment(decl) == []
        declared = decl.env if isinstance(decl, LayerImage) else None
        if declared is None:
            assert "environment" not in sections and env == []
            return
        assert sections.index("environment") > sections.index("post")
        in_def = [tuple(line.removeprefix("export ").split("=", 1)) for line in _sections(definition)["environment"]]
        in_docker = [tuple(line.removeprefix("ENV ").split("=", 1)) for line in env]
        assert in_def == in_docker == [tuple(pair) for pair in declared] and declared
        return
    assert sections.index("environment") > sections.index("post")
    in_def = []
    for line in _sections(definition)["environment"]:
        name, value = line.removeprefix("export ").split("=", 1)
        in_def.append((name, value))
    in_docker = [tuple(line.removeprefix("ENV ").split("=", 1)) for line in env]
    assert [n for n, _ in in_def] == [n for n, _ in in_docker] == [v.name for v in environment(decl)]
    for variable, (_, def_value), (_, docker_value) in zip(environment(decl), in_def, in_docker, strict=True):
        if variable.literal:
            assert shlex.split(def_value) == [_docker_env(docker_value)] == [variable.value]
        else:
            # $PATH is the PATH of the image below, which each runtime expands.
            assert def_value == docker_value == variable.value


def test_a_childs_script_sees_exactly_its_bases_variables_of_hls():
    """Beside the CUDA image's own variables, which neither renderer writes, a base's script sees
    none of hls's, and a layer's or a cap's sees its base's environment, set by no image between."""
    by_name = {d.name: d for d in catalog.shared_decls()}
    for decl in by_name.values():
        chain = [decl]
        while (parent := parent_of(chain[-1])) is not None:
            chain.append(by_name[parent])
        assert all(environment(d) == [] for d in chain[:-1])
        seen = [] if decl is chain[-1] else environment(chain[-1])
        if seen:
            assert {v.name: v.value for v in seen} == {
                "PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv", "UV_CONSTRAINT": "/opt/hls/constraints.txt",
                "PIP_CONSTRAINT": "/opt/hls/constraints.txt", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
                "UV_NO_CACHE": "1", "VLLM_DO_NOT_TRACK": "1"}


@pytest.mark.parametrize("decl, locks, uv", IMAGES, ids=IDS)
def test_a_dockerfile_names_its_frontend_and_where_it_starts(decl, locks, uv):
    text = _dockerfile(decl, locks, uv)
    lines = text.split("\n")
    assert lines[0] == f"# syntax={DOCKERFILE_SYNTAX}" and re.search(r"@sha256:[0-9a-f]{64}$", DOCKERFILE_SYNTAX)
    assert text.endswith("\n") and str(locks) not in text
    if isinstance(decl, BaseImage):
        # Docker takes the tag and the digest together; the digest pins the bytes.
        assert lines[1] == f"FROM {decl.cuda_ref}" and lines[2] == f'LABEL hls.cuda_ref="{decl.cuda_ref}"'
        # No image keeps the CUDA image's entrypoint.
        assert lines[-3:] == ["ENTRYPOINT []", "CMD []", ""]
    else:
        assert lines[1] == f"FROM {PARENT_CONTEXT}" and "ENTRYPOINT" not in text and "CMD" not in text
        assert build_inputs(decl, locks_dir=locks).parent


def test_a_base_with_no_pip_lock_copies_its_apt_lock_alone_and_holds_no_version():
    """The bare CUDA base installs its apt lock, uv and an empty venv, and writes an empty
    constraints file, as its .def does."""
    cuda = catalog.BASES["cuda"]
    assert cuda.pip_lock is None and not cuda.constraints
    text = render_dockerfile(cuda, locks=pins.LOCKS_DIR, uv=pins.UV)
    assert [line for line in text.split("\n") if line.startswith("COPY ")] == [
        f"COPY --from={LOCKS_CONTEXT} {cuda.apt_lock} /opt/hls/locks/{cuda.apt_lock}"]
    script = _run_body(text)[1].split("\n")
    assert f": > {CONSTRAINTS}" in script and not any("uv pip install" in line for line in script)
    assert build_inputs(cuda, locks_dir=pins.LOCKS_DIR).locks == {cuda.apt_lock: pins.LOCKS_DIR / cuda.apt_lock}


@pytest.mark.parametrize("task, decl", TASK_IMAGES, ids=[f"{task}-{d.name}" for task, d in TASK_IMAGES])
def test_a_tasks_own_layer_copies_its_locks_and_files_from_its_tasks_directory(task, decl):
    """A task's own layer reads its lock files from ``locks/`` of its task's directory and
    the files it copies in from the directory itself, never from the catalog's lock directory,
    and the Dockerfile copies each to the path the .def does."""
    assert decl.owner == task
    own = TASKS_DIR / task
    locks = [n for n in (decl.apt_lock, decl.pip_lock) if n is not None]
    wanted = {**{n: own / "locks" / n for n in locks}, **{Path(f).name: own / f for f in decl.files.values()}}
    if decl.fetched is not None:
        # A fetch lock belongs to a package's version, so it is the shared lock directory's.
        wanted[decl.fetched.lock] = pins.LOCKS_DIR / decl.fetched.lock
    assert build_inputs(decl, locks_dir=pins.LOCKS_DIR).locks == dict(sorted(wanted.items()))
    lines = _dockerfile(decl, pins.LOCKS_DIR, pins.UV).split("\n")
    for target, source in decl.files.items():
        assert f"COPY --from={LOCKS_CONTEXT} {Path(source).name} {target}" in lines
    for n in locks:
        assert f"COPY --from={LOCKS_CONTEXT} {n} /opt/hls/locks/{n}" in lines
    assert not any(str(TASKS_DIR) in line for line in lines)


#: The PATH the CUDA image's configuration gives, which no Dockerfile here names.
CUDA_PATH = "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _docker_environment(chain: list[ImageDecl]) -> dict[str, str]:
    """The environment the image at the end of *chain*, base first, has on Docker: each ENV line
    of each Dockerfile in order, a value's variables expanded against the environment so far,
    over the CUDA image's PATH."""
    env = {"PATH": CUDA_PATH}
    for decl in chain:
        for line in _dockerfile(decl, pins.LOCKS_DIR, pins.UV).split("\n"):
            if line.startswith("ENV "):
                name, word = line.removeprefix("ENV ").split("=", 1)
                env[name] = (_docker_env(word) if word.startswith('"') else
                             _REFERENCE.sub(lambda m: env.get(m.group(1) or m.group(2), ""), word))
    return env


def test_no_images_path_on_docker_holds_an_entry_twice():
    """The venv's bin directory, and every other entry, is on an image's PATH once, whichever
    images below it set PATH. A chain on a registry image keeps the vendor's PATH: no Dockerfile
    of it sets one."""
    for task in catalog.task_keys():
        decls = catalog.task_images(task).decls
        by_name = {d.name: d for d in decls}
        for decl in decls:
            chain = [decl]
            while (parent := parent_of(chain[0])) is not None:
                chain.insert(0, by_name[parent])
            entries = _docker_environment(chain)["PATH"].split(":")
            if isinstance(chain[0], RegistryImage):
                assert entries == CUDA_PATH.split(":"), (task, decl.name, entries)
                continue
            assert entries[0] == "/opt/venv/bin", (task, decl.name, entries)
            assert len(set(entries)) == len(entries), (task, decl.name, entries)


def test_every_image_a_task_declares_has_a_docker_key_of_its_own():
    """Every task's images key on a Docker site by their Dockerfiles, and no key is the one the
    same image has on an Apptainer site."""
    for task in catalog.task_keys():
        decls = catalog.task_images(task).decls
        docker, apptainer = image_keys(decls, dockerfile_key), image_keys(decls, recipe_key)
        assert not set(docker.values()) & set(apptainer.values()), task
        for d in decls:
            assert re.fullmatch(rf"{d.name}-[0-9a-f]{{16}}", docker[d.name])


def test_a_missing_lock_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        render_dockerfile(TEST_IMAGES[1], locks=tmp_path, uv=UV)


def test_a_script_line_that_ends_the_heredoc_is_refused():
    assert heredoc("a\nb") == ["a", "b", SCRIPT_END]
    with pytest.raises(ValueError, match="end of the heredoc"):
        heredoc(f"a\n{SCRIPT_END}\nb")


@pytest.mark.parametrize("value", ["1", 'a "b" c', "$HOME and ${PATH}", "back\\slash", "x'y", "", "end\\"])
def test_a_literal_value_reads_back_unchanged(value):
    assert _docker_env(quoted(value)) == value


@pytest.mark.parametrize("value", ["a\nb", "a\rb", "a\x00b"])
def test_a_value_a_dockerfile_cannot_carry_is_refused(value):
    with pytest.raises(ValueError, match="not printable"):
        quoted(value)


def test_the_parent_is_an_oci_layout_by_its_manifest_digest():
    digest = "sha256:" + "a" * 64
    assert parent_context(Path("/node/parent"), digest) == f"oci-layout:///node/parent@{digest}"
    with pytest.raises(ValueError, match="absolute"):
        parent_context(Path("node/parent"), digest)
    for bad in ("a" * 64, "sha256:" + "A" * 64, "sha256:" + "a" * 63, "sha512:" + "a" * 64):
        with pytest.raises(ValueError, match="manifest digest"):
            parent_context(Path("/node/parent"), bad)


def test_the_locks_context_holds_each_lock_once_by_its_name():
    cap = next(d for d in TEST_IMAGES if d.kind == "cap")
    inputs = build_inputs(cap, locks_dir=FIXTURES)
    assert inputs.locks == {"apt-agent.txt": FIXTURES / "apt-agent.txt", "npm-agent.json": FIXTURES / "npm-agent.json"}
    base = next(d for d in TEST_IMAGES if isinstance(d, BaseImage))
    assert not build_inputs(base, locks_dir=FIXTURES).parent


def _docker_keys(decls: list[ImageDecl], locks: Path, uv: UvPin) -> dict[str, str]:
    keys: dict[str, str] = {}
    for d in decls:
        parent = parent_of(d)
        keys[d.name] = dockerfile_key(d, parent_key=keys[parent] if parent else None, locks_dir=locks, uv=uv)
    return keys


def test_a_docker_key_is_its_own_and_follows_the_dockerfile_the_locks_and_the_parent(tmp_path):
    decls = catalog.shared_decls()
    keys = _docker_keys(decls, pins.LOCKS_DIR, pins.UV)
    assert len(set(keys.values())) == len(keys)
    def_keys: dict[str, str] = {}
    for d in decls:
        parent = parent_of(d)
        def_keys[d.name] = recipe_key(d, parent_key=def_keys[parent] if parent else None, locks_dir=pins.LOCKS_DIR,
                                      uv=pins.UV)
        assert re.fullmatch(rf"{d.name}-[0-9a-f]{{16}}", keys[d.name]) and keys[d.name] != def_keys[d.name]
    assert _docker_keys(decls, pins.LOCKS_DIR, pins.UV) == keys
    layer = next(d for d in TEST_IMAGES if isinstance(d, LayerImage) and not d.source_builds)
    key = dockerfile_key(layer, parent_key="p-0123456789abcdef", locks_dir=FIXTURES, uv=UV)
    assert dockerfile_key(layer, parent_key="p-fedcba9876543210", locks_dir=FIXTURES, uv=UV) != key
    moved = tmp_path / "locks"
    moved.mkdir()
    lock = (FIXTURES / layer.pip_lock).read_bytes()
    (moved / layer.pip_lock).write_bytes(lock)
    assert dockerfile_key(layer, parent_key="p-0123456789abcdef", locks_dir=moved, uv=UV) == key
    (moved / layer.pip_lock).write_bytes(lock + b"\n")
    assert dockerfile_key(layer, parent_key="p-0123456789abcdef", locks_dir=moved, uv=UV) != key


def test_a_docker_key_refuses_a_parent_key_that_does_not_fit():
    base = next(d for d in TEST_IMAGES if isinstance(d, BaseImage))
    layer = next(d for d in TEST_IMAGES if isinstance(d, LayerImage))
    with pytest.raises(ValueError, match="takes no parent key"):
        dockerfile_key(base, parent_key="x-0123456789abcdef", locks_dir=FIXTURES, uv=UV)
    with pytest.raises(ValueError, match="needs its parent's key"):
        dockerfile_key(layer, parent_key=None, locks_dir=FIXTURES, uv=UV)
