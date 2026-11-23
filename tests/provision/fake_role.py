"""What the local launcher runs in place of ``python -m hls.provision.role``: the real role
program, with the hub served from the directory ``FAKE_HUB`` names and every corpus the
20-row test corpus, so no test touches the network. A corpus fails as the hub does with
``FAKE_HUB_STATUS`` (``fake_hub``).

Nothing here imports ``huggingface_hub``: the role's ``main`` must switch off the hub
client's implicit token before the library is first imported, as it does on a cluster. The
fake hub goes in by a hook that patches the library right after that first import, and the
corpus module (which imports ``datasets``) is loaded only when a corpus is asked for."""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

from hls.provision import role
from tests.provision import fake_hub

HUB = Path(os.environ["FAKE_HUB"])


class _PatchHubOnImport(importlib.abc.MetaPathFinder):
    """Finds ``huggingface_hub`` with the other finders and installs the fake hub into it as
    soon as it has been executed."""

    def find_spec(self, name, path=None, target=None):
        if name != "huggingface_hub":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module) -> None:
            run(module)
            fake_hub.install(HUB, module)

        spec.loader.exec_module = exec_module
        return spec


def _corpus(*_):
    from tests.provision.tasks import corpus

    fake_hub.record_corpus_load(HUB)
    fake_hub._answer_status()
    return corpus()


assert "huggingface_hub" not in sys.modules
sys.meta_path.insert(0, _PatchHubOnImport())
role._corpus_for = _corpus
role._corpus_of = _corpus
sys.exit(role.main(sys.argv[1:]))
