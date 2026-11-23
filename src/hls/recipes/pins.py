"""Every pin the image recipes share, in one place.

The CUDA base is pinned by digest and keeps its tag, which names the Nsight package's CUDA
version. apt resolves against a dated snapshot of the Ubuntu archive, and pip and npm locks
count only uploads before ``EXCLUDE_NEWER``, so generating a lock again gives the same lock.
uv and Node are installed from their release tarballs, checked against their digests. The agent
CLIs are pinned here, not in hls.toml, so the unsealed image a plan names is the one every node
builds.

Changing any value here changes the content key of every image built from it."""

from __future__ import annotations

from pathlib import Path

from hls.recipes.model import NodePin, UvPin
from hls.recipes.render import PYPI_INDEX, TORCH_INDEX

#: The CUDA devel base: the tag for people and the Nsight version, the digest for the bytes
#: (Docker Hub's index digest of the tag, read by ``hls images base-digest``).
CUDA_BASE = ("nvidia/cuda:13.0.1-devel-ubuntu24.04"
             "@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92")
#: The snapshot.ubuntu.com timestamp every Ubuntu package resolves at.
SNAPSHOT = "20261001T000000Z"
#: NVIDIA's CUDA repository, the one apt source beside the snapshot. The CUDA image's own cuda.list
#: names it without the trailing slash, which the build's sources check ignores; it is kept here
#: because ``hls images lock-apt`` appends ``Packages.gz``. It has no snapshot, so its packages are pinned
#: by version and by the sha256 of their .deb files.
NVIDIA_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
#: uv, installed from ``uv-x86_64-unknown-linux-gnu.tar.gz`` of this release.
UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
#: Node, installed from ``node-v22.23.3-linux-x64.tar.gz`` (released 2026-09-23; its SHASUMS256.txt is
#: signed by Node release key 5BE8A3F6C8A5C01D106C0AD820B1A390B168D356), with the npm 10.9.9 it ships.
#: The Claude Code CLI needs Node 22 or later. The agent npm lock was generated with this Node and npm.
NODE = NodePin(version="22.23.3", sha256="1084aa36196bba4c3a5e69a1ee388a6e4ff729dad09445fbcd434b28fe3c24af")
#: The Claude Code CLI the agent cap installs (``@anthropic-ai/claude-code``), at least 2.1.248.
#: 2.1.247 accepts the ``/goal`` kickoff but never evaluates the goal, so nothing re-prompts an agent
#: that ends its turn before its budget; 2.1.251 re-prompts on every stop of an unmet goal until the
#: turn cap. Clients older than 2.1.208 exit with no output when the stream-json kickoff reads an
#: already-closed stdin, dropping the queued messages.
CLAUDE_CODE_VERSION = "2.1.251"
#: The Codex CLI the agent cap installs (``@openai/codex``).
CODEX_VERSION = "0.146.0"
#: The images' interpreter (Ubuntu 24.04's python3), which the locks are compiled for.
PYTHON_VERSION = "3.12"
#: The images' platform (Ubuntu 24.04's glibc 2.39), which the locks are compiled for.
PYTHON_PLATFORM = "x86_64-manylinux_2_39"
#: The first index locked wheels come from, the one the images install from: PyTorch's cu130 wheels.
INDEX_URL = TORCH_INDEX
#: The index locked wheels come from after ``INDEX_URL``: PyPI.
EXTRA_INDEX_URL = PYPI_INDEX
#: Locks count only uploads before this time (``uv pip compile --exclude-newer``, ``npm --before``).
EXCLUDE_NEWER = "2026-10-01T00:00:00Z"
#: The uv that compiles the pip locks (``hls images lock-pip`` refuses another). Another uv
#: may format the same pins differently, and treat the PyTorch index's files that carry no upload
#: time differently under ``EXCLUDE_NEWER`` (this one leaves them out).
LOCK_UV = "0.11.16"
#: The lock files' directory.
LOCKS_DIR = Path(__file__).with_name("locks")
