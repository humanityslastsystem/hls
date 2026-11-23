"""The brief's packaging section: the package's limits, each read from the constant the package
stage checks it against when the section renders, and the versions the sandbox's base holds;
rendered for every task whatever blocks its template replaces, and recorded in the brief's
provenance."""

from __future__ import annotations

import importlib
import subprocess
import sys
import zipfile

import pytest

from hls import prompt
from hls.agent import commit
from hls.egress import proxy
from hls.package import checkout, offline, wheel_check, wheels
from hls.prompt import ImplementerBrief, packaging_brief_section
from hls.recipes import model
from hls.stages.agent_build import commit_point_sec
from tests.unit.brief_task import PIP, render_brief
from tests.unit.test_prompt import _TS

#: Held versions to render, as a base's constraints pin them.
HELD = {"torch": "2.11.0+cu130", "torchaudio": "2.11.0+cu130", "torchvision": "0.26.0+cu130"}
#: The graded module of each published task.
GRADED = ("tasks.multitenant_slo.task", "tasks.tab_completion.task", "tasks.multimodal_tts.task",
          "tasks.multimodal_asr.task", "tasks.multimodal_omni.task", "tasks.deepresearch.task")
#: The sentence over the commit point's limits, past which the build has nothing gradable.
COMMIT_POINT_GROUP = "Past any of these limits your build ends `over_limit`, with nothing gradable, and scores zero:"
#: The sentence over the package's limits, which splits the section into its two groups.
PACKAGE_GROUP = "Past any of these limits your submission is an invalid package and scores zero:"
#: The package's first limit: its checkout reads the bundle of ``main`` as git reads it, every byte
#: proven the bundle's, and refuses one git does not read back (``hls.package.checkout.receive``).
READS_BACK = ("- the bundle of `main` reads back as git reads it: no object `main`'s history names is missing from "
              "it, of another kind, or stored under another object's name (a loose object's file name or a pack's "
              "index can store one so), and git can inflate every object in it and apply every delta;\n")
#: The sentence before the package's limits: the package's files are the committed bytes.
AS_COMMITTED = ("Each file of your package is written byte for byte as `main`'s tree holds it: the conversions a "
                "`.gitattributes` asks git for (`text`, `eol`, `ident`, `filter` and `working-tree-encoding`) are "
                "not applied, though git applies them to the files in your sandbox.\n\n")
#: A task template that replaces the build instructions and the toolchain.
REPLACING = ImplementerBrief(template_str=(
    '{% extends "brief" %}{% block build_instructions %}Build it your own way.{% endblock %}'
    '{% block toolchain %}Nothing is listed.{% endblock %}'))


@pytest.mark.parametrize("n, shown", [(4 * 2**30, "4 GiB"), (2**20, "1 MiB"), (24 * 2**30, "24 GiB"),
                                      (256 * 2**30, "256 GiB"), (64 * 2**20, "64 MiB"), (512 * 2**20, "512 MiB"),
                                      (1024, "1,024 bytes"), (255, "255 bytes"), (2**30 + 1, "1,073,741,825 bytes")])
def test_sizes_are_whole_gib_or_mib_or_bytes(n, shown):
    assert prompt._size(n) == shown


def test_the_brief_states_the_package_limits_from_their_constants(monkeypatch):
    text = packaging_brief_section(held=None, **PIP)
    # Past the commit point's limits nothing is gradable; past the package's the package is invalid.
    commit_point, sep, packaged = text.partition(PACKAGE_GROUP)
    assert sep and COMMIT_POINT_GROUP in commit_point and commit_point.endswith(".\n\n")
    # The repository's size and its deltas' chains are held to the sandbox's writable cap and the
    # chain bound, which the budget section states with the cap's figure.
    assert ("the repository's objects, history and unreachable ones included, and the bundle of `main` stay "
            "within your sandbox's writable space, and what rebuilding its deltas reads within its bound, as the "
            "budget section below states;") in commit_point
    # It names the budget section in words: the brief holds that heading once, where the section is.
    assert "## Your budget" not in text
    assert ("the repository's tree objects, history and unreachable ones included, hold at most "
            f"{prompt._size(commit.TREE_BYTES_MAX)} together;") in commit_point
    # The figures as the commit point holds them today.
    assert "hold at most 128 MiB together;" in commit_point
    assert (f"the repository holds at most {commit.OBJECTS_MAX:,} objects, none larger than "
            f"{prompt._size(commit.OBJECT_MAX_BYTES)};") in commit_point
    # The walk's bounds, past which the build ends over_limit too.
    assert (f"the repository's objects lie in at most {commit.ALTERNATES_MAX} object directories, its own and those "
            f"its alternates name, and in at most {commit.PACKS_MAX:,} packs among them; those directories and its "
            f"refs hold at most {commit.walk_max_entries():,} files and directories together;") in commit_point
    assert "and in at most 500 packs among them;" in commit_point
    assert (f"a commit, tree or tag is at most {prompt._size(commit.NONBLOB_MAX_BYTES)}, and an object stored as a "
            f"delta at most {prompt._size(commit.DELTA_MAX_BYTES)}.") in commit_point
    # The commit point's refusals of a repository it cannot read, each from the program's own list.
    unreadable = ("Your build also ends `unreadable`, with nothing gradable, and scores zero, when the commit point "
                  "cannot read the repository in your workspace:\n"
                  + ";\n".join(f"- {cause}" for cause in commit.UNREADABLE.values()) + ".\n\n")
    assert commit_point.endswith(unreadable + AS_COMMITTED) and len(commit.UNREADABLE) == 13
    assert "- the repository names its objects with SHA-256;\n" in unreadable
    assert "- the repository keeps its refs in reftable format;\n" in unreadable
    assert "- the workspace's `.git` is a file that points at a repository elsewhere;\n" in unreadable
    assert "or a delta chain deeper than 4,095;\n" in unreadable
    assert ("- `main`'s tree names a file or symbolic link whose object is not a blob but a tree, a commit or a tag, "
            "which git's plumbing can write and no checkout can read;\n") in unreadable
    # The commit point reads nothing back: a bundle git does not read back is the package's
    # checkout's to refuse, an invalid package.
    assert "read back" not in commit_point
    assert (PACKAGE_GROUP + "\n" + READS_BACK) in text
    # main's tree is listed at the commit point, each path and directory wherever it appears.
    assert (f"the tree of `main`'s last commit lists at most {commit.TREE_MAX_ENTRIES:,} paths and "
            f"{commit.TREE_MAX_DIRECTORIES:,} directories, each counted wherever it appears (a directory the tree "
            "names twice counts twice, with all it holds), and none of its paths, a directory's included, is longer "
            f"than {prompt._size(commit.TREE_MAX_PATH_BYTES)};") in commit_point
    assert "lists at most 200,000 paths and 200,000 directories" in commit_point
    assert f"the committed tree holds at most {prompt._size(checkout.TREE_MAX_BYTES)} of files" in packaged
    assert f"a path is at most {checkout.TREE_MAX_DEPTH} levels deep" in packaged
    assert f"each name in it at most {prompt._size(checkout.TREE_MAX_NAME_BYTES)}" in packaged
    assert ("every path is UTF-8 and holds no name git refuses to check out, such as `..` or one NTFS reads as "
            "`.git` (`.git`, `.GIT.`, `git~1`);") in packaged
    assert "the tree holds no symlink" in packaged
    assert f"`pyproject.toml` holds at most {prompt._size(checkout.PYPROJECT_MAX_BYTES)};" in packaged
    assert f"adds at most {offline.INSTALL_MAX_GIB} GiB to the image's Python environment" in packaged
    assert "held" not in text
    # Each figure is read from its constant when the section renders, never typed into the text.
    for module, name, value, shown in ((commit, "TREE_MAX_ENTRIES", 7, "lists at most 7 paths"),
                                        (commit, "TREE_MAX_DIRECTORIES", 8, "paths and 8 directories"),
                                        (commit, "TREE_MAX_PATH_BYTES", 512, "is longer than 512 bytes;"),
                                        (commit, "TREE_BYTES_MAX", 96 * 2**20, "hold at most 96 MiB together;"),
                                        (commit, "OBJECTS_MAX", 7_000, "holds at most 7,000 objects"),
                                        (commit, "WALK_ROOM_ENTRIES", 1_000, "hold at most 8,000 files and"),
                                        (commit, "ALTERNATES_MAX", 4, "lie in at most 4 object directories"),
                                        (commit, "PACKS_MAX", 7_000, "and in at most 7,000 packs among them"),
                                        (commit, "OBJECT_MAX_BYTES", 2 * 2**30, "none larger than 2 GiB;"),
                                        (commit, "NONBLOB_MAX_BYTES", 32 * 2**20, "tree or tag is at most 32 MiB"),
                                        (commit, "DELTA_MAX_BYTES", 256 * 2**20, "as a delta at most 256 MiB."),
                                        (checkout, "TREE_MAX_BYTES", 3 * 2**30, "holds at most 3 GiB of files"),
                                        (checkout, "TREE_MAX_DEPTH", 9, "at most 9 levels deep"),
                                        (checkout, "TREE_MAX_NAME_BYTES", 99, "each name in it at most 99 bytes"),
                                        (offline, "INSTALL_MAX_GIB", 5, "adds at most 5 GiB"),
                                        (checkout, "PYPROJECT_MAX_BYTES", 2**21, "holds at most 2 MiB")):
        monkeypatch.setattr(module, name, value)
        assert shown in packaging_brief_section(held=None, **PIP), name
    monkeypatch.setattr(commit, "UNREADABLE", {"one": "a cause stated once"})
    assert "in your workspace:\n- a cause stated once.\n\n" in packaging_brief_section(held=None, **PIP)


def test_the_brief_says_the_package_writes_each_file_as_committed(monkeypatch):
    """The package's checkout writes every file as its blob holds it, whatever the tree's
    ``.gitattributes`` asks, while git converts the files in the sandbox: the brief says so just
    before the package's limits on the tree, naming each attribute the checkout's own attributes
    switch off (``hls.package.checkout.CHECKOUT_ATTRIBUTES``), read from them when it renders."""
    text = packaging_brief_section(held=None, **PIP)
    assert AS_COMMITTED + PACKAGE_GROUP + "\n" + READS_BACK + "- the committed tree holds" in text
    assert checkout.unapplied_attributes() == ["text", "eol", "ident", "filter", "working-tree-encoding"]
    monkeypatch.setattr(checkout, "CHECKOUT_ATTRIBUTES", "* -text -ident\n")
    assert checkout.unapplied_attributes() == ["text", "ident"]
    assert "asks git for (`text` and `ident`) are not applied" in packaging_brief_section(held=None, **PIP)


@pytest.mark.parametrize("sec, shown", [(4 * 3600, "4 hours"), (3600, "1 hour"), (90 * 60, "90 minutes"),
                                        (60, "1 minute"), (61, "61 seconds"), (3601, "3,601 seconds")])
def test_durations_are_whole_hours_or_minutes_or_seconds(sec, shown):
    assert prompt._duration(sec) == shown


def test_the_brief_states_the_pip_steps_and_the_wheels_limits_from_their_constants(monkeypatch):
    """Past each of these the submission scores zero too: the pip steps' time, their fetches and
    a dependency's index page, the dry run's report's size and nesting, the wheels the steps may
    leave and what they may hold; then each step's disk and memory, which the stage's declared
    shape gives (``hls.stages.package.pip_limits``) and the section is handed."""
    text = packaging_brief_section(held=None, **PIP)
    _, _, packaged = text.partition(PACKAGE_GROUP)
    assert (f"installing your wheels adds at most {offline.INSTALL_MAX_GIB} GiB to the image's Python environment, "
            "and their entries declare at most that much together;") in packaged
    assert ("pip's two steps, a dry run that resolves your dependencies and then the build of the wheels, both "
            f"running your project's build hooks, take at most {prompt._duration(wheels.PIP_STEPS_SEC)} together, "
            "counted from each step's command's start, unless the package index failed meanwhile;") in packaged
    assert (f"each pip step fetches at most {proxy.UPSTREAM_FETCHES_MAX:,} pages and files from the package index, "
            f"redirects included, and no dependency's index page is larger than "
            f"{prompt._size(proxy.PAGE_MAX_BYTES)};") in packaged
    assert (f"the dry run's report of what it would install holds at most "
            f"{prompt._size(wheels.REPORT_MAX_BYTES)} and nests no deeper than {wheels.REPORT_MAX_DEPTH} "
            "levels;") in packaged
    assert ("the wheel step leaves only wheel files: your project's wheel and at most one wheel of each dependency "
            "the dry run resolved;") in packaged
    assert (f"your wheels together install at most {wheel_check.PATHS_MAX:,} paths (each file, each directory above "
            f"one and each script their entry points declare), nest no entry deeper than {wheel_check.DEPTH_MAX} "
            f"levels, have no name in an entry's path longer than {prompt._size(wheel_check.NAME_MAX_BYTES)}, "
            f"hold no Python source file larger than {prompt._size(wheel_check.MODULE_MAX_BYTES)} and no "
            f"`.dist-info` entry larger than {prompt._size(wheel_check.METADATA_MAX_BYTES)}, and none lists its "
            f"entries in more than {prompt._size(wheel_check.DIRECTORY_MAX_BYTES)};") in packaged
    assert ("every wheel is a zip each of whose entries is packed by `store` or `deflate` and unpacks to the size "
            "and checksum it declares.") in packaged
    assert packaged.endswith(
        f"\n\nEach pip step may write at most {PIP['pip_disk_gib']} GiB to the node's disk, its copy of your tree, its "
        f"temporary files and its wheels together, and hold at most {PIP['pip_memory_gib']} GiB of memory; a step "
        "past either is stopped, and the submission scores zero.")
    # Each figure is read from its constant, or from what the section is handed, when it renders.
    for module, name, value, shown in (
            (wheels, "PIP_STEPS_SEC", 90 * 60, "take at most 90 minutes together"),
            (proxy, "UPSTREAM_FETCHES_MAX", 7, "fetches at most 7 pages and files"),
            (proxy, "PAGE_MAX_BYTES", 3 * 2**20, "index page is larger than 3 MiB;"),
            (wheels, "REPORT_MAX_BYTES", 5 * 2**20, "would install holds at most 5 MiB and"),
            (wheels, "REPORT_MAX_DEPTH", 7, "nests no deeper than 7 levels;"),
            (wheel_check, "PATHS_MAX", 9, "install at most 9 paths"),
            (wheel_check, "DEPTH_MAX", 11, "deeper than 11 levels"),
            (wheel_check, "NAME_MAX_BYTES", 99, "path longer than 99 bytes"),
            (wheel_check, "MODULE_MAX_BYTES", 2 * 2**20, "source file larger than 2 MiB"),
            (wheel_check, "METADATA_MAX_BYTES", 6 * 2**20, "entry larger than 6 MiB"),
            (wheel_check, "DIRECTORY_MAX_BYTES", 7 * 2**20, "entries in more than 7 MiB"),
            (wheel_check, "PACKING_METHODS", frozenset({zipfile.ZIP_STORED}), "is packed by `store` and unpacks")):
        with monkeypatch.context() as m:
            m.setattr(module, name, value)
            assert shown in packaging_brief_section(held=None, **PIP), name
    handed = packaging_brief_section(held=None, **{**PIP, "pip_disk_gib": 41, "pip_memory_gib": 13})
    assert "write at most 41 GiB to the node's disk" in handed and "hold at most 13 GiB of memory" in handed


@pytest.mark.parametrize("seconds, collect, bundle", [(3600, "1 hour", "1,850 seconds"),
                                                      (5400, "90 minutes", "3,650 seconds"),
                                                      (2350, "2,350 seconds", "10 minutes")])
def test_the_brief_states_the_time_the_harness_has_to_collect_the_repository(seconds, collect, bundle):
    """The section opens with the collection allowance it is handed (``run.collect_allowance_sec``)
    and the commit point's least part of it (``commit_point_sec``), before any limit: once the build
    has ended the harness collects the repository within them, and a repository it cannot bundle in
    that time is not collected and the build not graded."""
    text = packaging_brief_section(held=None, **{**PIP, "collect_sec": seconds, "bundle_sec": commit_point_sec(seconds)})
    opening = ("When your build ends, the last commit on `main` is bundled and built offline into the image your "
               f"server is graded in. The harness then has at least {collect} to collect your repository, at least "
               f"{bundle} of it to bundle it; a repository it cannot bundle in that time is not collected, and your "
               "build is not graded. Past any of these limits your build ends `over_limit`")
    assert text.startswith(opening)
    assert text.count("collect your repository") == 1


def test_the_names_the_brief_says_git_refuses_are_refused_by_the_packages_checkout():
    assert all(checkout.git_refuses(name) for name in ("..", ".git", ".GIT.", "git~1"))
    assert checkout.limit_problem([checkout.TreeEntry(path="src/.GIT./x", size=1, link=False)]) is not None


def test_the_brief_says_torch_is_held_where_the_harness_holds_it():
    """The server's build applies the base's constraints whatever the agent does, so the brief
    states the hold there; in the sandbox the constraints are variables the agent may clear, so
    the brief states them as its default and names them."""
    text = packaging_brief_section(held=HELD, **PIP)
    assert text.endswith("`torch` 2.11.0+cu130, `torchaudio` 2.11.0+cu130 and `torchvision` 0.26.0+cu130 are held when "
                         "your server is built: an install that asks for another version of one of them fails there. "
                         "Your sandbox sets the same constraints by default (`PIP_CONSTRAINT`, `UV_CONSTRAINT`).")
    one = packaging_brief_section(held={"torch": "2.11.0+cu130"}, **PIP)
    assert one.endswith("`torch` 2.11.0+cu130 is held when your server is built: an install that asks for another "
                        "version of it fails there. Your sandbox sets the same constraint by default "
                        "(`PIP_CONSTRAINT`, `UV_CONSTRAINT`).")


def test_the_sandboxs_variables_are_those_every_base_points_at_its_constraints(monkeypatch):
    """The variables the brief names are read from the base's environment when it renders: each
    one every base sets to its constraints file."""
    named = sorted(name for name, value in model.BASE_ENVIRONMENT.items() if value == model.CONSTRAINTS)
    assert named == ["PIP_CONSTRAINT", "UV_CONSTRAINT"]
    monkeypatch.setattr(model, "BASE_ENVIRONMENT", {**model.BASE_ENVIRONMENT, "PIP_CONSTRAINT": "/elsewhere"})
    assert packaging_brief_section(held=HELD, **PIP).endswith("by default (`UV_CONSTRAINT`).")


@pytest.mark.parametrize("brief", [ImplementerBrief(), REPLACING], ids=["base", "replacing"])
def test_a_task_that_replaces_its_build_instructions_still_states_the_limits(brief):
    section = packaging_brief_section(held=HELD, **PIP)
    text, provenance = render_brief(_TS(), brief, submission_section="Commit to main.",
                                    packaging_section=section, budget_section="Twelve hours.")
    assert f"\n## Packaging limits\n{section}\n" in text
    # It follows the submission and comes before the budget.
    assert text.index("## Submission") < text.index("## Packaging limits") < text.index("## Your budget")
    [record] = [s for s in provenance.sections if s.block == "packaging"]
    assert record.rendered == section and record.source == "base"
    # The budget's own section does not swallow it, nor it the budget.
    [budget] = [s for s in provenance.sections if s.block == "budget"]
    assert budget.rendered == "Twelve hours."


def test_the_packaging_heading_renders_whatever_the_section_holds():
    """The block has no condition: every brief the stage renders carries the section."""
    text, provenance = render_brief(_TS(), packaging_section="")
    assert "\n## Packaging limits\n" in text
    assert text.index("## Submission") < text.index("## Packaging limits")
    assert [s.rendered for s in provenance.sections if s.block == "packaging"] == [""]


@pytest.mark.parametrize("module", GRADED)
def test_every_graded_tasks_brief_states_the_limits(module):
    m = importlib.import_module(module)
    section = packaging_brief_section(held=HELD, **PIP)
    text, _ = render_brief(m.TASK, m.ARTIFACT.brief, packaging_section=section)
    assert f"\n## Packaging limits\n{section}\n" in text


def test_the_packaging_section_reads_its_limits_from_below_the_stages():
    """The brief states each limit from the module that owns it, none of them a stage: rendering
    the section loads no module of ``hls.stages``."""
    code = ("import sys; from hls.prompt import packaging_brief_section; "
            "packaging_brief_section(held=None, pip_disk_gib=1, pip_memory_gib=1, collect_sec=1, bundle_sec=1); "
            "print(sorted(m for m in sys.modules if m == 'hls.stages' or m.startswith('hls.stages.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
