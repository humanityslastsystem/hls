"""Anti-reward-hacking on the stage path: the image an agent builds in holds no file of the task's
but those its brief shows the agent, so it never carries a reference implementation the agent
could copy to pass the gate. Every file an image of a sandbox's chain copies in is one of its own
lock files, in the shared lock directory or its task's ``locks/``, or a task file listed below
as one the agent is shown."""

from __future__ import annotations

import pytest

from hls.recipes import catalog
from hls.recipes.model import CapImage, LayerImage, parent_of
from hls.recipes.pins import LOCKS_DIR
from hls.recipes.render import LOCK_ROOT, NPM_PREFIX, TASKS_DIR, files_for, lock_dir

#: The task files each task's sandbox holds, by their path in the image: files the task's brief
#: shows the agent. A task absent here copies none in. deepresearch's are the loop the harness
#: drives a server with, its browser and retriever, the task's output cap and the self-test that
#: runs them, and a budget server with its meter; none of them serves the model.
SHOWN: dict[str, set[str]] = {
    "deepresearch": {
        "/app/tasks/deepresearch/__init__.py", "/app/tasks/deepresearch/research_loop.py",
        "/app/tasks/deepresearch/corpus_browser.py", "/app/tasks/deepresearch/retriever.py",
        "/app/tasks/deepresearch/limits.py", "/app/tasks/deepresearch/selftest.py",
        "/opt/hls-budget/meter.py", "/opt/hls-budget/server.py",
    },
}



def _sandbox_chain(task: str) -> list:
    images = catalog.task_images(task)
    by_name = {d.name: d for d in images.decls}
    chain, name = [], images.sandbox
    while name is not None:
        chain.append(by_name[name])
        name = parent_of(by_name[name])
    return chain


@pytest.mark.parametrize("task", catalog.task_keys())
def test_an_agent_sandbox_copies_in_only_its_locks_and_the_files_its_brief_shows(task):
    chain = _sandbox_chain(task)
    assert isinstance(chain[0], CapImage) and chain[0].cap == "agent"
    for decl in chain:
        assert not (isinstance(decl, LayerImage) and decl.server), (task, decl.name)
        locks = lock_dir(decl, LOCKS_DIR)
        assert locks in (LOCKS_DIR, TASKS_DIR / task / "locks"), (task, decl.name, locks)
        for inside, host in files_for(decl, LOCKS_DIR).items():
            if inside.startswith((f"{LOCK_ROOT}/", f"{NPM_PREFIX}/")):
                assert host.parent == locks, (decl.name, host)
            else:
                assert isinstance(decl, LayerImage) and decl.owner == task, (decl.name, inside)
                assert inside in SHOWN.get(task, set()), (task, decl.name, inside)


def test_a_task_file_in_a_sandbox_is_one_its_brief_shows(monkeypatch):
    """The guard holds for a task's own sandbox layer, which keeps its lock in its own directory."""
    own = LayerImage(name="sample-task-sandbox", parent="s-a", pip_lock="sandbox.txt", owner="sample_task",
                     files={"/workspace/reference_impl.py": "task.py"})
    monkeypatch.setattr(catalog, "image_set", lambda task: catalog.ImageSet(layer=own.name, images=[own]))
    chain = _sandbox_chain("sample_task")
    copied = {inside for decl in chain for inside in files_for(decl, LOCKS_DIR)
              if not inside.startswith((f"{LOCK_ROOT}/", f"{NPM_PREFIX}/"))}
    assert copied == {"/workspace/reference_impl.py"} and not copied <= SHOWN.get("sample_task", set())
