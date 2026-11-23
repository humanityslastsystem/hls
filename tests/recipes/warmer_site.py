"""hls.toml, Slurm and fake-tool inputs for the ``hls images build`` and ``hls package`` tests."""

from __future__ import annotations

from pathlib import Path

from hls.recipes import catalog
from hls.recipes.inventory import BRIEF_APT, BRIEF_PACKAGES, BRIEF_TOOLS
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision.warmer_config import write_config
from tests.support import node

#: Node-local disk the in-process Slurm grants these jobs: more than any warmer's floor, the
#: package stage's at the largest writable disk cap a task's build declares (288 GiB at 240).
DISK_FREE_GIB = 400.0
#: A registry login's user name, a plain setting.
REGISTRY_USER = "registry-user-01"
#: The ``[site]`` line of hls.toml that sets it.
REGISTRY_SITE = f'registry_username = "{REGISTRY_USER}"'
#: The login's token, its one secret.
REGISTRY_TOKEN = "registry-pass-0123456789"
#: A secrets file that grants the images stage the token.
REGISTRY_SECRETS = (f'[secrets]\nregistry_token = "{REGISTRY_TOKEN}"\n'
                    '[grants]\n"stage.images".REGISTRY_PASSWORD = "registry_token"\n')
#: A secrets file that also grants the user name, as a secret.
USER_GRANTING_SECRETS = (f'[secrets]\nregistry_user = "{REGISTRY_USER}"\nregistry_token = "{REGISTRY_TOKEN}"\n'
                         '[grants]\n"stage.images".REGISTRY_USERNAME = "registry_user"\n'
                         '"stage.images".REGISTRY_PASSWORD = "registry_token"\n')


def local_grading_node() -> str:
    """The ``[site]`` lines that declare the test node's per-uid limits as its grading nodes'
    (``site.grading_node_*``): a local grading's node is the test node the stage reads
    (``tests.support.node``), as a site's declaration states its own nodes'."""
    limits = node.UCOUNTS
    namespaces = min(limit for name, limit in limits.items() if name.endswith("_namespaces"))
    return (f"grading_node_namespaces = {namespaces}\n"
            f"grading_node_inotify_instances = {limits['max_inotify_instances']}\n"
            f"grading_node_inotify_watches = {limits['max_inotify_watches']}")


def site_config(tmp_path: Path, *, site: str = "", **kw) -> Path:
    """The warmers' hls.toml (``write_config``) with *site* lines added to ``[site]``."""
    path = write_config(tmp_path, **kw)
    path.write_text(path.read_text().replace("[run]", f"{site}\n[run]", 1))
    return path


def local_slurm() -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, each granted
    ``DISK_FREE_GIB`` of disk."""
    slurm = LocalSlurm(disk_free_gib=DISK_FREE_GIB)

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def inventory_for(base: str = "s-a") -> str:
    """What a built image of *base*'s line reports: its held versions, every tool and package the
    brief promises an agent cap holds (``hls.recipes.inventory``), and nothing denied. Every image
    reports the promises, so the agent cap a test builds keeps them."""
    held = [line.partition("==") for line in catalog.BASES[base].constraints]
    names = {name for name, _, _ in held}
    lines = ["apt\tcurl\t8.5.0-2ubuntu10.6", *(f"apt\t{name}\t1.0" for name in BRIEF_APT),
             *(f"pip\t{name}\t{version}" for name, _, version in held), "pip\tnumpy\t2.5.3",
             *(f"pip\t{name}\t1.0" for name in BRIEF_PACKAGES if name not in names),
             *(f"tool\t{name}\t{name} 1.0" for name in BRIEF_TOOLS), "env\tPATH", ""]
    return "\n".join(lines)
