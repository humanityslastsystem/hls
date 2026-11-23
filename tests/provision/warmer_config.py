"""hls.toml and secrets files for the ``hls provision`` tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

#: The secret the tests grant to the provision stage.
TOKEN = "hf-token-0123456789"
#: A secrets file that grants it as HF_TOKEN.
SECRETS = f'[secrets]\nhf_token = "{TOKEN}"\n[grants]\n"stage.provision".HF_TOKEN = "hf_token"\n'


def write_config(tmp_path: Path, *, cpu_partition: str | None = "cpu-a", floor: int = 0,
                 secrets: str | None = None, sandbox_stores: tuple[str, ...] = ()) -> Path:
    """An hls.toml under *tmp_path* with one-second heartbeats and polls, naming each of
    *sandbox_stores* as a directory under *tmp_path* (``sandbox_store``). *secrets*, when given, is
    written 0600 at the configured secrets path; otherwise no secrets file exists."""
    secrets_path = tmp_path / "secrets.toml"
    if secrets is not None:
        secrets_path.write_text(secrets)
        secrets_path.chmod(0o600)
    cpu = f'cpu_partition = "{cpu_partition}"' if cpu_partition else ""
    path = tmp_path / "hls.toml"
    path.write_text(textwrap.dedent(f"""
        [site]
        partition = "gpu-a"
        {cpu}
        account = "acct"
        min_gpus_per_job = {floor}
        store_root = "{tmp_path / 'store'}"
        node_scratch = "{tmp_path / 'node'}"
        secrets_path = "{secrets_path}"
        [run]
        heartbeat_sec = 1
        lease_ttl_sec = 30
        poll_sec = 1
        allocation_overhead_sec = 360
        allocation_dead_after_sec = 10
        """))
    if sandbox_stores:
        table = "".join(f'{name} = "{sandbox_store(tmp_path, name)}"\n' for name in sandbox_stores)
        path.write_text(path.read_text() + "[sandbox_stores]\n" + table)
    return path


def sandbox_store(tmp_path: Path, name: str) -> Path:
    """Where ``write_config`` puts sandbox store *name*: beside *tmp_path*, which holds the store
    root and the secrets file's directory, so in neither (``hls.settings.sandbox_store_problems``)."""
    return tmp_path.parent / f"{tmp_path.name}-sandbox-stores" / name


def task_sandbox_stores(task_module: str) -> tuple[str, ...]:
    """The sandbox stores the ``TASK`` of *task_module*'s tool runner names, which a site must
    name to plan a drive of it (``hls.pipeline.planner``); none for a task without one."""
    import importlib

    from hls.spec.tools import DEVICE

    agent = importlib.import_module(task_module).TASK.agent
    paths = agent.tool_runner.host_paths if agent is not None else ()
    return tuple(dict.fromkeys(p.store for p in paths if p.store != DEVICE))


def fill_sandbox_stores(stores: dict[str, Path], task_module: str, store_root: Path) -> None:
    """Put in the sandbox stores *stores* (by name, their directories) everything the tool runner
    of *task_module*'s ``TASK`` needs for the corpus the cache at *store_root* holds, as a staged
    site would hold it (``hls.pipeline.commands.missing_store_entries``): empty files and
    directories, and beside each sandbox image the module declares the record of an empty file
    pulled from its declared source, which is all a plan checks."""
    import importlib

    from hls.pipeline.commands import corpus_rows
    from hls.provision.sandbox_images import ImageRecord, image_file, record_file
    from hls.store.digest import sha256_bytes
    from hls.stages.provision import cached_provision
    from hls.store.cache import ContentCache
    from hls.store.posix import PosixStore

    task = importlib.import_module(task_module).TASK
    cache = ContentCache(PosixStore(store_root))
    rows = corpus_rows(cached_provision(task, cache), cache)
    for entry in task.agent.tool_runner.store_entries(rows):
        target = stores[entry.store] / entry.path if entry.path else stores[entry.store]
        if entry.kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
    images = getattr(importlib.import_module(task_module), "SANDBOX_IMAGES", None)
    for image in images.images if images is not None else ():
        directory = stores[images.store]
        if image_file(directory, image.name).is_file():
            record = ImageRecord(name=image.name, source=images.source(image), sha256=sha256_bytes(b""), size=0,
                                 apptainer="apptainer version 1.5.3")
            record_file(directory, image.name).write_text(record.model_dump_json())
