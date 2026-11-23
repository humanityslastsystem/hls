"""A submission image built on this machine's Docker daemon from the stand-in layer of
``test_images_live``: the install container of ``image_role.submit`` run as the package stage's build
role runs it, the archive pinned and its runscript run, and its files listed against what the
``.def``'s build gives (``hls.package.offline``): the tree at /workspace and the wheel's files in the
venv owned 0:0 with the modes ``cp -fLr`` and pip give them, and nothing of the staging left. The
package's integrity check and scan then read the two archives as the stage has them read.

Opt-in, as every ``integration`` test (``HLS_RUN_SLOW=1``); it builds the stand-in base and layer
first, which pulls them and the BuildKit image. Everything it makes carries ``hls.test=docker-images``
or the job's name and is removed at the end."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from hls.package import integrity, scan
from hls.package.offline import new_nonce
from hls.stages.package import submission_file
from hls.store.digest import reported_sha256
from hls.substrate.container import TREE_ABSENT_EXIT, PinnedImage
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient, NotFound
from tests.substrate.docker.test_images_live import (LABELS, _pin, _role, _runtime, area, built,  # noqa: F401
                                                     docker, pins_made)

pytestmark = pytest.mark.integration

#: What the listing prints for each path: owner, group, octal mode, size and path.
LIST = "find {paths} -printf '%U:%G %m %s %y %p\\n' | sort -k5"


def _wheel(path: Path) -> None:
    files = {"subm/__init__.py": b"VALUE = 1\n", "subm/data.txt": b"hls-live-marker-5c0e1d\n",
             "subm-0.1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: subm\nVersion: 0.1\n",
             "subm-0.1.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: hand\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = "".join(f"{n},sha256={base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()},{len(d)}\n"
                     for n, d in files.items()) + "subm-0.1.dist-info/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
        z.writestr("subm-0.1.dist-info/RECORD", record)


def _tree(root: Path) -> None:
    (root / "lib").mkdir(parents=True)
    (root / "server.py").write_text("import subm\nprint('served', subm.VALUE)\n")
    (root / "lib" / "helper.py").write_text("X = 1\n")
    (root / "secret.txt").write_text("owner only\n")
    (root / "secret.txt").chmod(0o600)
    (root / "run.sh").write_text("#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    (root / "link.py").symlink_to("server.py")


def _run(client: DockerClient, image: str, command: str, out: Path) -> str:
    cid = client.create_container(ContainerConfig(
        image=image, argv=("/bin/sh", "-c", f"({command}) > /out/text 2>&1"), user=f"{os.getuid()}:{os.getgid()}",
        labels=LABELS, network_mode="none", cpuset=(0,), memory_bytes=512 << 20, pids_limit=64,
        mounts=(BindMount(source=out, target="/out", read_only=False),)))
    try:
        client.start_container(cid)
        result = client.wait_container(cid, timeout=300)
        assert result is not None
    finally:
        client.remove_container(cid, force=True)
    return (out / "text").read_text()


def test_a_submission_builds_in_a_root_container_without_capabilities_and_serves(docker, area, built, pins_made):  # noqa: F811
    images = _runtime(area, "9201")
    layer = built["standin-layer"]
    rc, own = _pin(images, area, layer.archive, layer.sha256, "subm-base")
    assert rc == 0, (own / "pin.log").read_text()[-3000:]
    base = images.pinned(own, sha256=layer.sha256, contestant=False)
    pins_made.append((images, base))
    inputs = area / "submission"
    _tree(inputs / "tree")
    (inputs / "wheels").mkdir(parents=True)
    _wheel(inputs / "wheels" / "subm-0.1-py3-none-any.whl")
    work = area / "build-submission"
    work.mkdir()
    nonce = new_nonce()
    built_file = work / submission_file(images.image_file)
    argv, env = images.submission_role(built_file, digest=work / "sha256",
                                       base_file=(layer.archive, layer.sha256), base=base, tree=inputs / "tree",
                                       wheels=inputs / "wheels", project="subm", server="server.py", nonce=nonce,
                                       constraints="/opt/hls/constraints.txt")
    rc = _role(argv, env, work / "build.log")
    log = (work / "build.log").read_text()
    assert rc == 0, log[-4000:]
    assert f"hls-offline-install-rc-{nonce}=0" in log and f"hls-offline-install-bytes-{nonce}=" in log
    sha256 = reported_sha256(work / "sha256")
    assert sha256 is not None
    # What the install adds, the archive's last layer: the tree, the venv and the runscript, and
    # nothing of the staging, the mounts or the container's init.
    with tarfile.open(built_file) as tar:
        index = json.load(tar.extractfile("index.json"))
        manifest = json.load(tar.extractfile("blobs/sha256/" + index["manifests"][0]["digest"][7:]))
        last = tar.extractfile("blobs/sha256/" + manifest["layers"][-1]["digest"][7:]).read()
    with tarfile.open(fileobj=io.BytesIO(last), mode="r:*") as added:
        tops = {m.name.split("/")[0] for m in added.getmembers()}
    (area / "docker-added-tops.txt").write_text("\n".join(sorted(tops)) + "\n")
    assert {"workspace", "opt", ".hls"} <= tops
    assert not tops & {"dev", "sbin", "srv-stage", "wheels", "hls-context", "tmp"}, tops
    assert not docker.list_containers(labels={"hls.job": "9201"})
    assert [i for i in docker.list_images(repository="hls/job-9201") if any("submission" in t for t in i.repo_tags)] == []
    rc, own = _pin(images, area, built_file, sha256, "submission")
    assert rc == 0, (own / "pin.log").read_text()[-3000:]
    built_image: PinnedImage = images.pinned(own, sha256=sha256, contestant=True)
    pins_made.append((images, built_image))
    out = area / "listing"
    out.mkdir()
    assert _run(docker, built_image.ref, images.runscript, out).strip() == "served 1"
    listing = _run(docker, built_image.ref,
                   LIST.format(paths="/workspace /opt/venv/lib/python3.12/site-packages/subm "
                                     "/opt/venv/lib/python3.12/site-packages/subm-0.1.dist-info /.hls"), out)
    (area / "docker-listing.txt").write_text(listing)
    absent = _run(docker, built_image.ref, "ls -d /srv-stage /wheels 2>&1; true", out)
    assert absent.count("No such file") == 2, absent
    rows = {line.split(" ", 4)[4]: line.split(" ", 4)[:4] for line in listing.splitlines()}
    # cp -fLr as root under a 022 umask: root owns everything, the owner-only file stays so, the
    # symlink is a copy of its target.
    assert rows["/workspace"][0] == "0:0"
    assert rows["/workspace/secret.txt"][:2] == ["0:0", "600"]
    assert rows["/workspace/run.sh"][:2] == ["0:0", "755"]
    assert rows["/workspace/link.py"][3] == "f" and rows["/workspace/server.py"][:2] == ["0:0", "644"]
    assert all(row[0] == "0:0" for row in rows.values()), listing
    assert rows["/.hls/runscript"][:2] == ["0:0", "755"]
    config = docker.inspect_image(built_image.ref).config
    assert config.get("Labels", {}).get("hls.job") is None and not config.get("Entrypoint")
    assert "VIRTUAL_ENV=/opt/venv" in config.get("Env", [])
    with pytest.raises(NotFound):
        docker.inspect_image("hls/job-9201:submission")
    # The package's integrity check and scan, as the stage runs them: each image read from the
    # archive it was pinned from, the built one as the base and the install's layer, and the base
    # read once: the integrity check leaves its listing, which the scan reads.
    listing = area / "base-listing.json"
    built_args = images.reader_args("image", built_image, built_file)
    (area / "no-wheels").mkdir()
    started = time.monotonic()
    assert integrity.main([*images.reader_args("base", base, layer.archive, listing_out=listing), *built_args,
                           "--wheels", str(area / "no-wheels"), "--out", str(area / "integrity.json"),
                           "--outputs-out", str(area / "integrity-outputs.json")]) == 0
    record = integrity.IntegrityRecord.model_validate_json((area / "integrity.json").read_text())
    assert record.differences == 0 and record.entries > 1000, record
    os.environ["HLS_LIVE_VALUE"] = "hls-live-marker-5c0e1d"
    try:
        readers = [*images.reader_args("base", base, layer.archive, listing=listing), *built_args]
        assert scan.main(["--names", "HLS_LIVE_VALUE", *readers, "--out", str(area / "package-scan.json"),
                          "--outputs-out", str(area / "scan-outputs.json")]) == 0
    finally:
        del os.environ["HLS_LIVE_VALUE"]
    (area / "check-seconds.txt").write_text(f"{time.monotonic() - started:.1f}\n")
    found = scan.PackageScan.model_validate_json((area / "package-scan.json").read_text())
    assert [(f.path, f.secret) for f in found.findings] == [
        ("/opt/venv/lib/python3.12/site-packages/subm/data.txt", "HLS_LIVE_VALUE")]
    # One directory of the built image, read as an agent's seed reads one: the daemon's archive of
    # it, unpacked as the host user, root-only files readable by their new owner.
    own = area / "tree-built"
    dest = own / "tree"
    dest.mkdir(parents=True)
    argv, env = images.tree_role(built_image, "/workspace", dest, own=own)
    assert _role(argv, env, own / "read.log") == 0, (own / "read.log").read_text()
    assert (dest / "server.py").read_text().startswith("import subm") and (dest / "secret.txt").read_text()
    assert not docker.list_containers(labels={"hls.job": "9201"})
    missing = own / "missing"
    missing.mkdir()
    argv, env = images.tree_role(built_image, "/nowhere", missing, own=own)
    assert _role(argv, env, own / "missing.log") == TREE_ABSENT_EXIT
    assert "refused: image" in (own / "missing.log").read_text()
