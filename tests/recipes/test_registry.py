"""A registry image, a vendor's image pulled by digest, the server layers that build on one, and a
layer's ``.deb`` files."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from hls.pipeline.planner import PlanError, StageDecl
from hls.recipes import catalog
from hls.recipes.catalog import ImageSet, Served, task_images
from hls.recipes.model import LayerImage, RegistryImage, order, registry_of, root_of
from hls.recipes.render import (
    LOCK_ROOT,
    PYPI_INDEX,
    SNAPSHOT_SOURCE_PARTS,
    SNAPSHOT_SOURCES,
    TORCH_INDEX,
    UBUNTU_KEYRING,
    baked_environment,
    check_source_builds,
    post_script,
    recipe_key,
    render_def,
    uv_url,
)
from hls.recipes.render_docker import PARENT_CONTEXT, SCRIPT_END, build_inputs, dockerfile_key, render_dockerfile, \
    syntax_line
from hls.stages import images as images_stage
from tests.pipeline.plans import noop_plan
from tests.recipes.declared import declare
from tests.recipes.test_images_stage import BARE, CAP, SHAPE
from tests.recipes.test_render import BASE, FIXTURES, UV, _post_script, _section, _steps, _stub

DIGEST = "sha256:" + "4c" * 32
VENDOR = RegistryImage(name="t-vendor", ref=f"vendor/engine@{DIGEST}", python="/usr/bin/python3",
                       snapshot="20261001T000000Z")
DEB_URL = "https://downloads.example/runtime/v1.5.3/runtime_1.5.3_amd64.deb"
DEB = b"a .deb file"
DEBS = {DEB_URL: hashlib.sha256(DEB).hexdigest()}
#: A server layer on the vendor image with a file and nothing installed, as a baseline that serves
#: with the vendor's own stack.
SERVED = LayerImage(name="t-baseline", parent="t-vendor", owner="t", server=True,
                    files={"/app/server.py": "server.py"}, env={"PYTHONPATH": "/app"})
#: A server layer on the vendor image that installs apt packages, a .deb and a pip lock.
TOOLS = LayerImage(name="t-reference", parent="t-vendor", owner="t", server=True, apt_lock="apt-agent.txt",
                   debs=DEBS, pip_lock="pip-layer.txt")
PARENT = Path("/node/t-vendor.sif")


@pytest.fixture
def tasks(tmp_path) -> Path:
    """A tasks directory whose task ``t`` holds the fixture locks and a server file."""
    root = tmp_path / "tasks"
    (root / "t" / "locks").mkdir(parents=True)
    for name in ("apt-agent.txt", "pip-layer.txt", "pip-sdist.txt"):
        shutil.copyfile(FIXTURES / name, root / "t" / "locks" / name)
    (root / "t" / "server.py").write_text("print('serving')\n")
    return root


def _render(decl, tasks: Path, registry: RegistryImage | None = VENDOR, parent: Path | None = PARENT) -> str:
    return render_def(decl, parent_sif=parent, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=registry)


# the declaration


@pytest.mark.parametrize("ref", [
    "vendor/engine:nightly",                      # a tag alone pins nothing
    f"vendor/engine:nightly@{DIGEST}",            # a tag beside the digest
    "vendor/engine@sha256:" + "4c" * 31,          # a short digest
    f"Vendor/engine@{DIGEST}",
    f"https://registry.example/vendor/engine@{DIGEST}",
])
def test_a_registry_image_is_named_by_digest_alone(ref):
    with pytest.raises(ValueError, match="a registry reference by digest"):
        RegistryImage(name="t-vendor", ref=ref, python="/usr/bin/python3", snapshot="20261001T000000Z")


def test_a_registry_image_declares_its_interpreter_and_its_snapshot():
    for missing in ("python", "snapshot"):
        fields = {k: v for k, v in VENDOR.model_dump().items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            RegistryImage(**fields)
    with pytest.raises(ValueError, match="absolute path"):
        RegistryImage(**{**VENDOR.model_dump(), "python": "python3"})
    with pytest.raises(ValueError, match="snapshot timestamp"):
        RegistryImage(**{**VENDOR.model_dump(), "snapshot": "20261001"})


def test_a_registry_image_builds_on_nothing_and_its_layers_chain_ends_at_it():
    by_name = {d.name: d for d in (VENDOR, SERVED, TOOLS, BARE)}
    assert order([TOOLS, SERVED, VENDOR])[0] == VENDOR
    assert root_of(TOOLS, by_name) == VENDOR and registry_of(TOOLS, by_name) == VENDOR
    # The registry image itself is rendered as what it is, and a chain on a base has none.
    assert registry_of(VENDOR, by_name) is None and registry_of(BARE, by_name) is None
    on_base = LayerImage(name="t-member", parent="cuda", pip_lock="pip-layer.txt", owner="t", server=True)
    assert registry_of(on_base, {**by_name, "t-member": on_base}) is None


@pytest.mark.parametrize("fields, match", [
    ({"debs": DEBS, "apt_lock": None}, "pins their dependencies in an apt lock"),
    ({"debs": {"http://downloads.example/x_1_amd64.deb": "0" * 64}}, "an https URL of a .deb file"),
    ({"debs": {"https://downloads.example/x_1_amd64.rpm": "0" * 64}}, "an https URL of a .deb file"),
    ({"debs": {"https://downloads.example/x 1.deb": "0" * 64}}, "an https URL of a .deb file"),
    ({"debs": {DEB_URL: "0" * 63}}, "not a bare hex SHA-256"),
    ({"debs": {"https://a.example/x_1_amd64.deb": "0" * 64, "https://b.example/x_1_amd64.deb": "1" * 64}},
     "have one file name"),
])
def test_a_layers_deb_files_are_https_urls_with_digests_beside_an_apt_lock(fields, match):
    with pytest.raises(ValueError, match=match):
        LayerImage(**{**TOOLS.model_dump(), **fields})


def test_a_layer_with_no_deb_file_is_written_as_it_was_before_layers_had_any():
    """A plan's images stage holds each declaration as written: one without ``debs`` keeps the
    fields it had, so no plan of an image that installs none changes."""
    layer = LayerImage(name="s-a-tf", parent="s-a", pip_lock="pip-layer.txt")
    assert "debs" not in layer.model_dump() and "debs" not in layer.model_dump(mode="json")
    assert TOOLS.model_dump()["debs"] == DEBS
    assert LayerImage.model_validate(TOOLS.model_dump()) == TOOLS


# the definition file


def test_a_registry_image_is_the_vendors_image_as_pulled(tasks):
    text = _render(VENDOR, tasks, registry=None, parent=None)
    assert text == f"Bootstrap: docker\nFrom: vendor/engine@{DIGEST}\n"
    with pytest.raises(ValueError, match="registry image t-vendor has no parent"):
        _render(VENDOR, tasks, registry=None)


def test_a_registry_images_key_is_its_names_and_its_references(tasks):
    key = recipe_key(VENDOR, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks)
    assert key.startswith("t-vendor-") and len(key) == len("t-vendor-") + 16
    other = VENDOR.model_copy(update={"ref": "vendor/engine@sha256:" + "5d" * 32})
    assert recipe_key(other, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks) != key
    # What its layers install with is their key's, not its own.
    moved = VENDOR.model_copy(update={"python": "/usr/local/bin/python3", "snapshot": "20261101T000000Z"})
    assert recipe_key(moved, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks) == key
    with pytest.raises(ValueError, match="takes no parent key"):
        recipe_key(VENDOR, parent_key="x-0123456789abcdef", locks_dir=FIXTURES, uv=UV, tasks_dir=tasks)


def test_a_layer_that_only_copies_files_installs_nothing_into_the_vendor_image(tasks):
    text = _render(SERVED, tasks)
    assert _section(text, "files") == [f"{tasks}/t/server.py /app/server.py"]
    assert _section(text, "post") == ["set -eu", "set +x"]
    assert _section(text, "environment") == ["export PYTHONPATH=/app"]


def test_a_layers_apt_on_a_registry_image_reads_the_snapshot_alone(tasks):
    steps = _steps(_render(TOOLS, tasks))
    snapshot = "https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/"
    [sources] = [s for s in steps if s.endswith(f"> {SNAPSHOT_SOURCES}")]
    for pocket in ("noble", "noble-updates", "noble-security"):
        assert f"'deb [signed-by={UBUNTU_KEYRING}] {snapshot} {pocket} main restricted universe multiverse'" in sources
    assert "noble-backports" not in sources
    [conf] = [s for s in steps if s.endswith("> /etc/apt/apt.conf.d/99-hls")]
    assert f"'Dir::Etc::sourcelist \"{SNAPSHOT_SOURCES}\";'" in conf
    assert f"'Dir::Etc::sourceparts \"{SNAPSHOT_SOURCE_PARTS}\";'" in conf
    assert "'APT::Sandbox::User \"root\";'" in conf and "'Acquire::Check-Valid-Until \"false\";'" in conf
    # apt is configured before it reads anything, and the image's own sources are never edited.
    assert steps.index(conf) < steps.index("apt-get update")
    assert "sed -i" not in "\n".join(steps) and "/etc/apt/sources.list" not in "\n".join(steps)


def test_a_layers_pip_lock_on_a_registry_image_installs_into_the_declared_interpreter(tasks):
    steps = _steps(_render(TOOLS, tasks))
    [install] = [s for s in steps if " pip install " in s]
    assert install.startswith('"$hls_dl/uv-x86_64-unknown-linux-gnu/uv" pip install --no-config ')
    assert ' --cache-dir "$hls_dl/cache" ' in install
    assert " --python /usr/bin/python3 --break-system-packages " in install
    assert " --require-hashes --no-deps --only-binary :all: " in install
    assert f" --index-url {TORCH_INDEX} --extra-index-url {PYPI_INDEX} -r {LOCK_ROOT}/pip-layer.txt" in install
    # The pinned uv is fetched for the step, checked, and gone with its cache after it.
    at = steps.index(install)
    assert steps[at - 5:at] == [
        'hls_dl="$(mktemp -d /opt/hls/uv-download.XXXXXX)"',
        f'curl -fsSL --retry 5 -o "$hls_dl/uv.tar.gz" {uv_url(UV)}',
        f'echo "{UV.sha256}  $hls_dl/uv.tar.gz" | sha256sum -c -',
        'tar -xzf "$hls_dl/uv.tar.gz" -C "$hls_dl"',
        "unset $(env | grep -oE '^(UV|PIP)_[A-Za-z0-9_]*')",
    ]
    assert steps[at + 1] == 'rm -rf "$hls_dl"'
    # The vendor image has no venv and holds nothing at one version.
    text = "\n".join(steps)
    assert "/opt/venv" not in text and "constraints" not in text and "/usr/local/bin/uv" not in text


def test_a_layer_on_a_registry_image_follows_what_the_registry_image_declares(tasks):
    other = VENDOR.model_copy(update={"python": "/opt/engine/bin/python", "snapshot": "20261101T000000Z"})
    text = _render(TOOLS, tasks, registry=other)
    assert " --python /opt/engine/bin/python " in text
    assert "https://snapshot.ubuntu.com/ubuntu/20261101T000000Z/ noble " in text

    def key(registry):
        return recipe_key(TOOLS, parent_key="t-vendor-0123456789abcdef", locks_dir=FIXTURES, uv=UV,
                          tasks_dir=tasks, registry=registry)

    assert key(VENDOR) != key(other)


def test_a_layers_deb_files_install_beside_its_apt_lock(tasks):
    for registry, layer in ((VENDOR, TOOLS), (None, TOOLS.model_copy(update={"parent": "cuda", "pip_lock": None}))):
        steps = _steps(_render(layer, tasks, registry=registry))
        name = DEB_URL.rsplit("/", 1)[1]
        fetch = steps.index(f'curl -fsSL --retry 5 -o "$hls_fetched/{name}" {DEB_URL}')
        assert steps[fetch - 2:fetch] == ['hls_fetched="$(mktemp -d /opt/hls/debs.XXXXXX)"', 'hls_fetched_debs=""']
        assert steps[fetch + 1] == f'echo "{DEBS[DEB_URL]}  $hls_fetched/{name}" | sha256sum -c -'
        [install] = [s for s in steps if s.startswith("apt-get install ")]
        assert install == ("apt-get install -y --no-install-recommends "
                           f"$(sed -e '/^#/d' -e '/^$/d' {LOCK_ROOT}/apt-agent.txt) $hls_fetched_debs")
        assert steps.index("apt-get update") < fetch < steps.index(install) < steps.index('rm -rf "$hls_fetched"')
        assert "/tmp" not in "\n".join(steps)


def test_a_layer_with_no_deb_file_renders_no_step_for_one(tasks):
    text = _render(TOOLS.model_copy(update={"debs": {}}), tasks)
    assert "hls_fetched" not in text and "dpkg-deb" not in text


@pytest.mark.parametrize("decl, match", [
    (TOOLS.model_copy(update={"server": False}), "only a server layer builds on a vendor's image"),
    (CAP.model_copy(update={"parent": "t-reference"}), "only a server layer builds on a vendor's image"),
    (TOOLS.model_copy(update={"pip_lock": "pip-sdist.txt", "source_builds": {"sox==1.5.0": ["setuptools"]}}),
     "builds from source on registry image t-vendor"),
])
def test_only_a_server_layer_with_no_source_build_builds_on_a_registry_image(tasks, decl, match):
    with pytest.raises(ValueError, match=match):
        render_def(decl, parent_sif=PARENT, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=VENDOR)


def test_a_source_build_on_a_registry_image_is_refused_before_anything_builds(tasks):
    sdist = TOOLS.model_copy(update={"pip_lock": "pip-sdist.txt", "source_builds": {"sox==1.5.0": ["setuptools"]}})
    with pytest.raises(ValueError, match="builds from source on registry image t-vendor, whose packages no lock"):
        check_source_builds([VENDOR, sdist], FIXTURES, tasks)


def test_a_layer_on_a_registry_image_bakes_what_it_declares_and_nothing_else():
    decls = [VENDOR, SERVED, TOOLS]
    assert baked_environment("t-baseline", decls) == {"PYTHONPATH": "/app"}
    # The vendor image's environment is its own, which no %environment of ours writes.
    assert baked_environment("t-reference", decls) == {} == baked_environment("t-vendor", decls)


# the Dockerfile


def _run_body(dockerfile: str) -> str:
    """The script a Dockerfile's RUN heredoc holds."""
    lines = dockerfile.split("\n")
    start = lines.index(f"RUN <<'{SCRIPT_END}'")
    return "\n".join(lines[start + 1:lines.index(SCRIPT_END, start)])


def test_a_registry_images_dockerfile_starts_at_its_reference_and_runs_nothing(tasks):
    text = render_dockerfile(VENDOR, locks=FIXTURES, uv=UV, tasks_dir=tasks)
    # The vendor's entrypoint and command are cleared, so a role's command runs as it is given.
    assert text.split("\n") == [syntax_line(), f"FROM vendor/engine@{DIGEST}", "ENTRYPOINT []", "CMD []", ""]
    inputs = build_inputs(VENDOR, locks_dir=FIXTURES, tasks_dir=tasks)
    assert inputs.locks == {} and not inputs.parent
    with pytest.raises(ValueError, match="registry image t-vendor is the vendor's image as pulled"):
        post_script(VENDOR, locks=FIXTURES, uv=UV, tasks_dir=tasks)


def test_a_registry_images_docker_key_is_its_own_and_follows_its_reference(tasks):
    key = dockerfile_key(VENDOR, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks)
    assert key.startswith("t-vendor-") and len(key) == len("t-vendor-") + 16
    assert key != recipe_key(VENDOR, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks)
    other = VENDOR.model_copy(update={"ref": "vendor/engine@sha256:" + "5d" * 32})
    assert dockerfile_key(other, parent_key=None, locks_dir=FIXTURES, uv=UV, tasks_dir=tasks) != key
    with pytest.raises(ValueError, match="takes no parent key"):
        dockerfile_key(VENDOR, parent_key="x-0123456789abcdef", locks_dir=FIXTURES, uv=UV, tasks_dir=tasks)


@pytest.mark.parametrize("decl", [SERVED, TOOLS], ids=["files", "packages"])
def test_a_layer_on_a_registry_image_runs_one_script_in_both_wrappers(tasks, decl):
    script = post_script(decl, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=VENDOR)
    dockerfile = render_dockerfile(decl, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=VENDOR)
    lines = dockerfile.split("\n")
    assert lines[1] == f"FROM {PARENT_CONTEXT}" and "ENTRYPOINT" not in dockerfile and "CMD" not in dockerfile
    assert _run_body(dockerfile) == script
    assert _section(_render(decl, tasks), "post") == [line.strip() for line in script.split("\n") if line.strip()]
    # What installs into a venv on a base installs into the vendor's interpreter here.
    assert "/opt/venv" not in script
    assert [line for line in lines if line.startswith("ENV ")] == [f"ENV {name}={value}" for name, value in decl.env or []]
    assert build_inputs(decl, locks_dir=FIXTURES, tasks_dir=tasks).parent


def test_a_layers_script_on_a_registry_image_follows_what_the_registry_image_declares(tasks):
    script = post_script(TOOLS, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=VENDOR)
    moved = VENDOR.model_copy(update={"python": "/usr/local/bin/python3"})
    assert post_script(TOOLS, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=moved) != script
    key = dockerfile_key(TOOLS, parent_key="t-vendor-0123456789abcdef", locks_dir=FIXTURES, uv=UV, tasks_dir=tasks,
                         registry=VENDOR)
    assert dockerfile_key(TOOLS, parent_key="t-vendor-0123456789abcdef", locks_dir=FIXTURES, uv=UV, tasks_dir=tasks,
                          registry=moved) != key


@pytest.mark.parametrize("decl, match", [
    (TOOLS.model_copy(update={"server": False}), "only a server layer builds on a vendor's image"),
    (CAP.model_copy(update={"parent": "t-reference"}), "only a server layer builds on a vendor's image"),
    (TOOLS.model_copy(update={"pip_lock": "pip-sdist.txt", "source_builds": {"sox==1.5.0": ["setuptools"]}}),
     "builds from source on registry image t-vendor"),
])
def test_what_may_not_build_on_a_registry_image_has_no_dockerfile(tasks, decl, match):
    with pytest.raises(ValueError, match=match):
        render_dockerfile(decl, locks=FIXTURES, uv=UV, tasks_dir=tasks, registry=VENDOR)


# the build script, run with stand-ins


def _run_post(tmp_path: Path, tasks: Path, *, release: str = "noble", deb: bytes = DEB,
              installed: str = "1.5.3") -> tuple[object, Path]:
    """Run ``TOOLS``'s %post under sh with stand-ins: the image is Ubuntu *release*, curl serves
    *deb* for the .deb and a uv tarball for uv, apt-get records its arguments, dpkg-deb names the
    package ``runtime`` at 1.5.3, dpkg-query reports each locked package at its version and
    ``runtime`` at *installed*, and uv records its arguments and its ``UV_*`` variables. Return
    the result and the directory the stand-ins wrote in."""
    import subprocess

    uv = UV.model_copy(update={"sha256": hashlib.sha256(b"uv release").hexdigest()})
    text = render_def(TOOLS, parent_sif=PARENT, locks=FIXTURES, uv=uv, tasks_dir=tasks, registry=VENDOR)
    root = tmp_path / "image"
    locks = root / "opt-hls" / "locks"
    locks.mkdir(parents=True)
    for name in ("apt-agent.txt", "pip-layer.txt"):
        shutil.copy(FIXTURES / name, locks / name)
    (root / "etc-apt").mkdir()
    (root / "os-release").write_text(f"VERSION_CODENAME={release}\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "deb").write_bytes(deb)
    (out / "uv-release").write_bytes(b"uv release")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "curl", 'while [ $# -gt 0 ]; do [ "$1" = -o ] && hls_out="$2"; hls_url="$1"; shift; done\n'
                           f'case "$hls_url" in *.deb) cp {out}/deb "$hls_out" ;; *) cp {out}/uv-release "$hls_out" ;; '
                           "esac\n")
    _stub(bin_dir, "tar", f'mkdir -p "$4/uv-x86_64-unknown-linux-gnu"\ncp {bin_dir}/fake-uv '
                          '"$4/uv-x86_64-unknown-linux-gnu/uv"\n')
    _stub(bin_dir, "fake-uv", f'echo "$@" > {out}/uv-args\nenv | grep -E "^(UV|PIP)_" > {out}/uv-env\nexit 0\n')
    _stub(bin_dir, "apt-get", f'echo "$@" >> {out}/apt-args\nexit 0\n')
    _stub(bin_dir, "dpkg-deb", 'case "$3" in Package) echo runtime ;; Version) echo 1.5.3 ;; esac\n')
    locked = f'"$(grep "^$3=" {locks}/apt-agent.txt | cut -d= -f2-)"'
    _stub(bin_dir, "dpkg-query", f'if [ "$3" = runtime ]; then printf "install ok installed %s" {installed}; '
                                 f"else printf 'install ok installed %s' {locked}; fi\n")
    script = (_post_script(text, root).replace("/etc/os-release", str(root / "os-release"))
              .replace("/etc/apt/apt.conf.d", str(root / "etc-apt")).replace("/var/lib/apt/lists", str(root / "lists")))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "UV_OVERRIDE": "/etc/uv-overrides.txt",
           "UV_INDEX_STRATEGY": "first-index", "PIP_INDEX_URL": "https://mirror.example/simple"}
    return subprocess.run(["sh", "-c", script + "\necho reached-the-end\n"], capture_output=True, text=True,
                          env=env), out


def test_the_build_script_of_a_layer_on_a_registry_image_runs_to_its_end(tmp_path, tasks):
    r, out = _run_post(tmp_path, tasks)
    assert r.returncode == 0 and r.stdout.strip().endswith("reached-the-end"), r.stderr
    root = tmp_path / "image"
    # apt read its own sources file, and the .deb was installed in the one apt-get install.
    assert (root / "opt-hls" / "apt" / "sources.list").read_text().count("snapshot.ubuntu.com") == 3
    assert list((root / "opt-hls" / "apt" / "sources.list.d").iterdir()) == []
    update, install = (out / "apt-args").read_text().splitlines()
    assert update == "update" and install.startswith("install -y --no-install-recommends ")
    assert install.endswith("/runtime_1.5.3_amd64.deb") and "nodejs=" in install
    # uv installed into the declared interpreter with none of the image's own uv and pip settings.
    assert " --python /usr/bin/python3 " in (out / "uv-args").read_text()
    assert (out / "uv-env").read_text() == ""
    # Nothing of the downloads is left in the image.
    assert sorted(p.name for p in (root / "opt-hls").iterdir()) == ["apt", "locks"]


def test_an_image_of_another_ubuntu_release_ends_the_build_before_apt_runs(tmp_path, tasks):
    r, out = _run_post(tmp_path, tasks, release="jammy")
    assert r.returncode == 1 and "the image is Ubuntu jammy, and the apt locks are pinned for noble" in r.stderr
    assert not (out / "apt-args").exists()


def test_a_deb_file_that_does_not_match_its_digest_ends_the_build_before_it_installs(tmp_path, tasks):
    r, out = _run_post(tmp_path, tasks, deb=b"another file")
    assert r.returncode != 0 and "reached-the-end" not in r.stdout
    assert (out / "apt-args").read_text().splitlines() == ["update"]


def test_a_deb_file_that_is_not_installed_at_its_own_version_ends_the_build(tmp_path, tasks):
    r, out = _run_post(tmp_path, tasks, installed="1.4.0")
    assert r.returncode == 1 and "runtime is install ok installed 1.4.0, not 1.5.3" in r.stderr
    assert not (out / "uv-args").exists()


# a wheel by URL


WHEEL_URL = "https://wheels.example/tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl"
HASH = "--hash=sha256:" + "ab" * 32


def _url_lock(tasks: Path, entry: str) -> LayerImage:
    (tasks / "t" / "locks" / "url.txt").write_text(f"{entry}\nnumpy==2.3.5 \\\n    {HASH}\n")
    return LayerImage(name="t-member", parent="cuda", pip_lock="url.txt", owner="t", server=True)


def test_a_pip_lock_may_pin_a_wheel_by_its_url(tasks):
    layer = _url_lock(tasks, f"cuda-tile @ {WHEEL_URL} \\\n    {HASH}\n    # via -r url.in")
    text = _render(layer, tasks, registry=None, parent=Path("/node/cuda.sif"))
    assert f"-r {LOCK_ROOT}/url.txt" in text
    # A source build names its package at the version the wheel's file name carries.
    built = layer.model_copy(update={"source_builds": {"cuda-tile==1.6.0rc5": ["setuptools"]}})
    _render(built, tasks, registry=None, parent=Path("/node/cuda.sif"))
    with pytest.raises(ValueError, match="source build cuda-tile==1.6.0 of image t-member is not in its pip lock"):
        _render(layer.model_copy(update={"source_builds": {"cuda-tile==1.6.0": ["setuptools"]}}), tasks,
                registry=None, parent=Path("/node/cuda.sif"))


@pytest.mark.parametrize("entry", [
    f"cuda-tile @ {WHEEL_URL}",                                               # no hash
    f"cuda-tile @ {WHEEL_URL.replace('https', 'http')} \\\n    {HASH}",          # not https
    f"cuda-tile @ https://wheels.example/tile/cuda_tile-1.6.0rc5.tar.gz \\\n    {HASH}",
    f"numba @ {WHEEL_URL} \\\n    {HASH}",                                     # another distribution's wheel
    f"cuda-tile @ git+https://wheels.example/tile.git \\\n    {HASH}",
    f"cuda-tile @ {WHEEL_URL} --index-url=https://wheels.example \\\n    {HASH}",
])
def test_a_url_entry_that_is_not_a_hashed_wheel_of_its_name_is_refused(tasks, entry):
    layer = _url_lock(tasks, entry)
    with pytest.raises(ValueError, match="is not name"):
        _render(layer, tasks, registry=None, parent=Path("/node/cuda.sif"))


# the catalog and the images stage


def _image_set() -> ImageSet:
    return ImageSet(layer="s-a-tf-5-13-1", images=[VENDOR], harness_on_reference=True,
                    baseline=Served(image=SERVED.model_copy(update={"owner": "sample_task"}),
                                    entrypoint=["python3", "/app/server.py"]),
                    reference=TOOLS.model_copy(update={"owner": "sample_task"}))


def test_a_task_declares_the_registry_image_its_server_layers_build_on(monkeypatch):
    declare(monkeypatch, {"sample_task": _image_set()})
    images = task_images("sample_task")
    names = [d.name for d in images.decls]
    assert names.index("t-vendor") < names.index("t-baseline") and names.index("t-vendor") < names.index("t-reference")
    assert images.harness == images.reference == "t-reference"
    assert [d.kind for d in images.decls if d.name == "t-vendor"] == ["registry"]


def test_a_registry_image_no_image_of_the_task_builds_on_is_refused(monkeypatch):
    unused = ImageSet(layer="s-a-tf-5-13-1", images=[VENDOR])
    declare(monkeypatch, {"sample_task": unused})
    with pytest.raises(ValueError, match=r"declares images \['t-vendor'\], which none of its images builds on"):
        task_images("sample_task")


def test_a_sandbox_never_builds_on_a_registry_image(monkeypatch):
    with pytest.raises(ValueError, match="layer 't-vendor' is not one of"):
        ImageSet(layer="t-vendor", images=[VENDOR])
    own = LayerImage(name="t-layer", parent="t-vendor", pip_lock="pip-layer.txt", owner="sample_task")
    declare(monkeypatch, {"sample_task": ImageSet(layer="t-layer", images=[own, VENDOR])})
    with pytest.raises(ValueError, match="builds on registry image t-vendor: a sandbox builds on a base"):
        task_images("sample_task")
    assert catalog.image_set("sample_task").layer == "t-layer"


@pytest.mark.parametrize("decls, match", [
    ([VENDOR, TOOLS.model_copy(update={"server": False})],
     "image t-reference builds on registry image t-vendor: only a server layer"),
    ([VENDOR, TOOLS, CAP.model_copy(update={"parent": "t-reference"})],
     "image s-a-tf-agent builds on server image t-reference"),
])
def test_what_may_not_build_on_a_registry_image_is_refused_when_the_plan_is_made(tmp_path, decls, match):
    params = {"catalog_task": "t", "images": [d.model_dump() for d in decls]}
    with pytest.raises(PlanError, match=match):
        noop_plan(tmp_path, decls=[StageDecl("images", "images", SHAPE, params)])


def test_the_images_stage_keys_a_layer_with_the_registry_image_it_builds_on(tasks, monkeypatch):
    monkeypatch.setattr(images_stage, "LOCKS_DIR", FIXTURES)
    monkeypatch.setattr(images_stage, "TASKS_DIR", tasks)
    monkeypatch.setattr(images_stage, "UV", UV)
    for key in (recipe_key, dockerfile_key):
        keys = images_stage.image_keys([TOOLS, VENDOR, SERVED], key)
        assert list(keys) == ["t-vendor", "t-reference", "t-baseline"]
        assert keys["t-reference"] == key(TOOLS, parent_key=keys["t-vendor"], locks_dir=FIXTURES, uv=UV,
                                           tasks_dir=tasks, registry=VENDOR)
        assert images_stage.image_keys([BASE], key)  # a chain on a base takes none
