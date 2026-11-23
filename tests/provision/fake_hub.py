"""A stand-in for the Hugging Face hub, served from a directory, for the provision role run
as a subprocess by the local launcher.

Layout of a hub directory:
- ``refs.json``: repo id to branch name to commit;
- ``repos/<org>+<name>/<commit>/listed/``: the files the hub lists for that commit;
- ``repos/<org>+<name>/<commit>/served/``: when present, what a download delivers instead,
  for a download that does not match the listing;
- ``calls.jsonl``: one line per hub call the role made, with the token it passed and whether
  the implicit token was off; a corpus load of ``fake_role`` counts as one.

Files ending ``.safetensors`` are listed as LFS files (SHA-256), the rest as git files (blob
id), as the hub does. A download into ``local_dir`` writes regular files, as the real client
does there. Variables of the role's environment change what the hub does:
- ``FAKE_HUB_DELAY``: each download takes that many seconds;
- ``FAKE_HUB_LINKS``: ``symlink`` or ``hardlink`` makes each downloaded file a link to a
  blob under ``HF_HOME``, as a client that deduplicates through a blob cache would;
- ``FAKE_HUB_EXIT``: a download ends the process with that exit status;
- ``FAKE_HUB_GARBLED``: a model lookup fails to parse the hub's reply, as the client does
  when a proxy answers with a page that is not JSON;
- ``FAKE_HUB_STATUS``: every download, and every corpus of ``fake_role``, fails with the hub client's own error
  for an answer with that HTTP status, such as 404 for a repo that does not exist.

``hub_error(status)`` is that error, for a test that raises it itself."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from hls.provision.hf import git_blob_sha1


def hub_error(status: int, url: str = "https://huggingface.co/api/models/org/model") -> Exception:
    """The error the hub client raises for an answer with HTTP *status* (``hf_raise_for_status``)."""
    import httpx
    from huggingface_hub.utils import hf_raise_for_status

    try:
        hf_raise_for_status(httpx.Response(status, request=httpx.Request("GET", url)))
    except Exception as e:
        return e
    raise AssertionError(f"the hub client takes status {status} for success")


def _answer_status() -> None:
    """Raise the hub client's error for ``FAKE_HUB_STATUS``, when it is set."""
    status = os.environ.get("FAKE_HUB_STATUS")
    if status:
        raise hub_error(int(status))


def _repo_dir(hub: Path, repo: str, revision: str) -> Path:
    return hub / "repos" / repo.replace("/", "+") / revision


def make_hub(hub: Path, repo: str, revision: str, files: dict[str, bytes], *, branch: str = "main") -> Path:
    """Add *repo* at *revision*, listing *files*, with *branch* pointing at it."""
    listed = _repo_dir(hub, repo, revision) / "listed"
    for rel, data in files.items():
        (listed / rel).parent.mkdir(parents=True, exist_ok=True)
        (listed / rel).write_bytes(data)
    refs_file = hub / "refs.json"
    refs = json.loads(refs_file.read_text()) if refs_file.exists() else {}
    refs.setdefault(repo, {})[branch] = revision
    refs_file.write_text(json.dumps(refs))
    return listed


def serve_instead(hub: Path, repo: str, revision: str, rel: str, data: bytes) -> None:
    """Make downloads of *repo* at *revision* deliver *data* for *rel* while the listing
    keeps the original."""
    listed = _repo_dir(hub, repo, revision) / "listed"
    served = _repo_dir(hub, repo, revision) / "served"
    if not served.exists():
        shutil.copytree(listed, served)
    (served / rel).write_bytes(data)


def calls(hub: Path) -> list[dict]:
    """Every hub call the roles made, oldest first."""
    path = hub / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _record(hub: Path, **fields) -> None:
    with open(hub / "calls.jsonl", "a") as f:
        f.write(json.dumps(fields) + "\n")


def _commit(hub: Path, repo: str, revision: str) -> str:
    refs = json.loads((hub / "refs.json").read_text()).get(repo)
    if refs is None:
        raise FileNotFoundError(f"fake hub has no repo {repo}")
    if revision in refs:
        return refs[revision]
    if revision in refs.values():
        return revision
    raise FileNotFoundError(f"fake hub has no revision {revision} of {repo}")


def record_corpus_load(hub: Path) -> None:
    """Record a corpus load as a hub call, with the token ``datasets`` sends: ``HF_TOKEN`` from
    the environment, or none."""
    constants = importlib.import_module("huggingface_hub.constants")
    _record(hub, call="load_dataset", token=os.environ.get("HF_TOKEN") or False,
            implicit_token_off=constants.HF_HUB_DISABLE_IMPLICIT_TOKEN)


def _link_into_blobs(local_dir: Path, how: str) -> None:
    """Replace every file under *local_dir* by a link of kind *how* to a blob under HF_HOME."""
    blobs = Path(os.environ["HF_HOME"]) / "hub" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file() and ".cache" not in p.parts):
        blob = blobs / hashlib.sha256(path.read_bytes()).hexdigest()
        os.replace(path, blob)
        if how == "symlink":
            path.symlink_to(blob)
        else:
            os.link(blob, path)


def install(hub: Path, huggingface_hub) -> None:
    """Point the ``HfApi`` and ``snapshot_download`` of the imported module *huggingface_hub*
    at *hub*. Each call is recorded with the token it passed and with whether the library's
    implicit token (one stored on the node) was switched off when the library was imported."""

    def record(**fields) -> None:
        constants = importlib.import_module("huggingface_hub.constants")
        _record(hub, implicit_token_off=constants.HF_HUB_DISABLE_IMPLICIT_TOKEN, **fields)

    class Api:
        def model_info(self, repo, *, revision=None, token=None, files_metadata=False):
            record(call="model_info", repo=repo, revision=revision, token=token)
            if os.environ.get("FAKE_HUB_GARBLED"):
                json.loads("<html>bad gateway</html>")
            sha = _commit(hub, repo, revision or "main")
            siblings = []
            if files_metadata:
                listed = _repo_dir(hub, repo, sha) / "listed"
                for path in sorted(p for p in listed.rglob("*") if p.is_file()):
                    rel, size = path.relative_to(listed).as_posix(), path.stat().st_size
                    if rel.endswith(".safetensors"):
                        lfs = SimpleNamespace(size=size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        siblings.append(SimpleNamespace(rfilename=rel, lfs=lfs, size=size, blob_id=None))
                    else:
                        siblings.append(SimpleNamespace(rfilename=rel, lfs=None, size=size,
                                                        blob_id=git_blob_sha1(path)))
            return SimpleNamespace(sha=sha, siblings=siblings)

    def snapshot_download(*, repo_id, revision, local_dir, token):
        record(call="snapshot_download", repo=repo_id, revision=revision, token=token)
        time.sleep(float(os.environ.get("FAKE_HUB_DELAY", "0")))
        if os.environ.get("FAKE_HUB_EXIT"):
            raise SystemExit(int(os.environ["FAKE_HUB_EXIT"]))
        _answer_status()
        sha = _commit(hub, repo_id, revision)
        src = _repo_dir(hub, repo_id, sha)
        src = src / "served" if (src / "served").exists() else src / "listed"
        shutil.copytree(src, local_dir, dirs_exist_ok=True)
        # The real client keeps its bookkeeping here; the role must remove it.
        meta = Path(local_dir) / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "x.metadata").write_text("m")
        links = os.environ.get("FAKE_HUB_LINKS")
        if links:
            _link_into_blobs(Path(local_dir), links)
        return str(local_dir)

    huggingface_hub.HfApi = Api
    huggingface_hub.snapshot_download = snapshot_download
