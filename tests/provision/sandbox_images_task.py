"""A task that declares sandbox images, for the ``hls provision`` tests: the provision tests'
``TASK`` and a set of three images in a store named ``test-images``, published in a repository no
test reaches (a stand-in for apptainer answers every pull)."""

from __future__ import annotations

from hls.provision.sandbox_images import SandboxImage, SandboxImageSet
from tests.provision.tasks import TASK as TASK

#: The store the set names.
STORE = "test-images"
#: Each image's name and the digest of its published manifest.
DIGESTS = {
    "alpha": "sha256:" + "a1" * 32,
    "beta": "sha256:" + "b2" * 32,
    "gamma": "sha256:" + "c3" * 32,
}

SANDBOX_IMAGES = SandboxImageSet(
    store=STORE, repository="registry.example/bench",
    images=tuple(SandboxImage(name=name, digest=digest) for name, digest in DIGESTS.items()),
    cpus=2, memory_gib=3, disk_gib=1, time_sec=600, parallel=2, pull_timeout_sec=30)
