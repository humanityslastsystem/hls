"""Read the index digest of a Docker Hub image tag, to pin a recipe's base by digest
(``pins.CUDA_BASE``).

    hls images base-digest nvidia/cuda 13.0.1-devel-ubuntu24.04

It asks the registry for the tag's manifest list (the multi-architecture index), the digest a
``tag@sha256:...`` reference names, and prints ``<repo>:<tag>@<digest>``. It reads only; the
token is anonymous and never printed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from hls.recipes.locking import LockRefused

AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
MANIFEST = "https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
ACCEPT = ("application/vnd.oci.image.index.v1+json, "
          "application/vnd.docker.distribution.manifest.list.v2+json")


def index_digest(repo: str, tag: str, *, urlopen: Callable) -> str:
    """The digest of *repo*:*tag*'s manifest list, ``sha256:<hex>``. Raise LockRefused when the
    registry cannot be asked or gives no such digest. *urlopen* opens a URL or a request, as
    ``urllib.request.urlopen`` does."""
    try:
        with urlopen(AUTH.format(repo=repo), timeout=60) as r:
            token = json.load(r)["token"]
        req = urllib.request.Request(MANIFEST.format(repo=repo, tag=tag), method="HEAD",
                                     headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT})
        with urlopen(req, timeout=60) as r:
            digest = r.headers.get("Docker-Content-Digest")
    except (OSError, ValueError, KeyError) as e:
        raise LockRefused(f"the registry gave no digest for {repo}:{tag}: {type(e).__name__}: {e}") from None
    if not digest or not digest.startswith("sha256:"):
        raise LockRefused(f"the registry gave no digest for {repo}:{tag}")
    return digest


def base_reference(repo: str, tag: str, *, urlopen: Callable) -> str:
    """``<repo>:<tag>@<digest>``, the reference ``pins.CUDA_BASE`` holds."""
    return f"{repo}:{tag}@{index_digest(repo, tag, urlopen=urlopen)}"
