"""Unified prompt rendering.

One Jinja environment owns prompt rendering for the whole harness. The default
implementer brief lives in this module's `DictLoader` under the name "brief" and
is built from named `{% block %}` sections (goal, model, hardware, contract,
build_instructions). A task customizes any section by passing a child template
that `{% extends "brief" %}` and redefines a block -- it never edits the spec
objects. Spec objects contribute data via `prompt_fields()`; the template owns
all formatting. The same `JinjaPromptTemplate` primitive renders dataset column
templates (spec/datasets.py), so there is a single prompt-rendering mechanism.
"""

from __future__ import annotations
import errno
import math
import zipfile
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, Literal

import jinja2
import jinja2.nodes
from pydantic import BaseModel

from hls.core.decode import DecodeConfig, Greedy, Sample
from hls.core.poly import Poly


# The default brief, loaded from the package asset so the source file stays
# readable and the template can be edited independently of the Python code.
# Named blocks are the task-override seams.
_BRIEF = files("hls.prompts").joinpath("brief.jinja").read_text(encoding="utf-8")

# The ralph /goal message. Rides argv while the brief rides stdin; Claude Code
# composes both into the agent's first message and arms its goal loop. Must start
# with "/goal" (commands are recognized only at message start) and its text must
# stay under the CLI's 4000-char goal cap -- which is why the brief travels
# alongside rather than inside it.
_GOAL = files("hls.prompts").joinpath("goal.jinja").read_text(encoding="utf-8").strip()


def ralph_goal_prompt() -> str:
    """The /goal message the agent receives alongside the brief."""
    return _GOAL


#: The codex equivalent. Not interchangeable with ``_GOAL``: ``/goal`` is a slash
#: command interpreted by interactive clients (the TUI, the app) and is never forwarded
#: to the model, so passing it to ``codex exec`` is silently inert -- the model reads it
#: as prose and no goal is created. Codex's goal is created by the model calling its
#: ``create_goal`` tool, which it does only when asked to in natural language.
_GOAL_CODEX = files("hls.prompts").joinpath("goal_codex.jinja").read_text(
    encoding="utf-8").strip()


def codex_goal_prompt() -> str:
    """The opening instruction for a codex build: read the brief, then create a goal.

    Load-bearing for a long run. While a goal is outstanding codex refuses to signal
    ``task_complete`` -- it ends the turn as ``turn_aborted`` -- and ``codex exec
    resume`` is then driven turn after turn until the goal is met. Without a goal,
    ``codex exec`` runs a single turn and exits, ending the build in minutes."""
    return _GOAL_CODEX


#: The message a claude-code session is resumed with when it exited while budget was left
#: (the /goal Stop hook lets a headless session end early). Plain prose, not a /goal: the
#: goal armed at kickoff is part of the resumed session. Fixed text, so it is read once and
#: never rendered.
_GOAL_REENTRY = files("hls.prompts").joinpath("goal_reentry.jinja").read_text(
    encoding="utf-8").strip()


def reentry_prompt() -> str:
    """The message a resumed claude-code session receives."""
    return _GOAL_REENTRY


def _wall_phrase(seconds: int) -> str:
    """A wall-clock total in the unit a reader thinks in: hours from one hour up, fractional
    when they do not divide ("1.5 hours"), else whole minutes."""
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:g} hour{'s' if hours != 1 else ''} of wall-clock time"
    return f"{int(round(seconds / 60))} minutes of wall-clock time"


def budget_brief_section(soft_sec: int) -> str:
    """The ``## Your budget`` section's opening: the wall-clock budget the agent is told,
    *soft_sec* seconds, stated once as a total (it never goes stale), and where the live
    remainder comes from, the ``get_budget`` tool of the budget MCP server both harnesses
    register. The text never calls the budget soft, advisory or unenforced: the agent is to
    treat it as a real constraint."""
    return (f"You have {_wall_phrase(soft_sec)} for this build. Call `get_budget` (the `budget` MCP server) "
            "periodically to see how much remains, and plan your remaining work to stay within it. "
            "Leave headroom to commit working code before you run out.")


# In-container destination for a task's ``producer.upload_dir`` folder. Lives here,
# beside the brief, because it is part of the brief contract: the uploads section
# text tells the agent this path, and the build copies to the same constant
# (``hls.agent.uploads.upload_destination``) -- one source, so the text and the
# upload can never drift.
UPLOAD_DEST = "/task/files"


def blocked_packages_of(network: Any) -> list[str] | None:
    """The packages a producer's ``NetworkPolicy`` refuses at ``pip install``, for the
    brief's Build Instructions; None when the policy blocks nothing (no policy, or a
    package list that is not in deny mode)."""
    packages = getattr(network, "packages", None) if network is not None else None
    if packages is None or getattr(packages, "mode", None) != "deny":
        return None
    return [str(p) for p in packages.entries]


def uploads_brief_section(upload_brief: str | None, dest: str | None = None) -> str:
    """Render the uploads section from the task's mandatory ``producer.upload_brief``
    text (there is no default -- the spec validator rejects an upload_dir without
    it). The task authors the whole section, heading included: the harness supplies
    no ``## `` title, so a task that wants one writes it into upload_brief (e.g.
    ``## CUDA Documentation``). The harness appends only the one fact the task
    cannot know: where the uploaded folder lands inside the sandbox. Returns ""
    when no upload is configured, so the section is omitted.

    ``dest`` is the task's resolved ``producer.upload_dest`` (None -> UPLOAD_DEST), so
    a task that lands its files in the working tree states that path here rather than
    the default staging one."""
    if upload_brief is None or not upload_brief.strip():
        return ""
    return (
        f"{upload_brief.strip()}\n\n"
        f"These files are available at `{dest or UPLOAD_DEST}` inside your sandbox."
    )


def _joined(items: Sequence[str]) -> str:
    """*items* as one English list: "a", "a and b", "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _path_list(paths: Sequence[str]) -> str:
    return _joined([f"`{p}`" for p in paths])


def ucounts_sentence(subject: str, ucounts: Mapping[str, int]) -> str:
    """That *subject* may hold at most the limit in *ucounts* of each per-uid kernel count, by its
    ``/proc/sys/user`` file's name (``hls.substrate.ucounts``): inotify instances, inotify watches
    and namespaces, whose kinds that share a limit are named once. "" without a limit."""
    parts = [f"{ucounts[name]:,} {what}" for name, what in (("max_inotify_instances", "inotify instances"),
                                                            ("max_inotify_watches", "inotify watches"))
             if name in ucounts]
    spaces = {name.removeprefix("max_").removesuffix("_namespaces"): limit for name, limit in ucounts.items()
              if name.endswith("_namespaces")}
    if spaces and len(set(spaces.values())) == 1:
        parts.append(f"{next(iter(spaces.values())):,} namespaces of each kind ({', '.join(spaces)})")
    else:
        parts += [f"{limit:,} {kind} namespaces" for kind, limit in spaces.items()]
    if not parts:
        return ""
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return f"{subject} may hold at most {listed} at once; past that the kernel refuses another."


def sandbox_brief_section(*, memory_gib: int, harness_gib: float, shm_gib: int, cpus: int, disk_gib: int,
                          chain_gib: int, writable: Sequence[str], command_nproc: int,
                          nproc_accounting: Literal["uid", "user_namespace"], ucounts: Mapping[str, int],
                          grade_after_sandbox_oom: bool) -> str:
    """The build sandbox's limits, for the ``## Your budget`` section: the memory and CPUs its
    role declares, of which the harness's own processes may hold *harness_gib*, the cap of its
    ``/dev/shm`` (*shm_gib*), whose pages count in that memory, the writable space its
    *writable* paths share, apart from its memory, which is also the most the repository's objects
    may hold uncompressed and main's bundle may hold (*disk_gib*, the cap the commit point holds
    them to, ``hls.agent.commit``), the entries and nesting the disk watch walks there, past which
    it counts the space as run out (``hls.substrate.memory_watch``), the most rebuilding the
    repository's deltas may read (*chain_gib*, ``hls.agent.commit.CHAIN_BYTES_MAX``), the process
    ceiling every command runs under (*command_nproc*, its ``RLIMIT_NPROC``, which the kernel
    counts as *nproc_accounting* says, ``site.nproc_accounting``: against every task of the uid on
    the node, or against the commands' own), the limit of each per-uid kernel count the commands'
    namespace holds them to (*ucounts*, ``ucounts_sentence``), and what an overrun of memory or
    space ends in. Past its memory one process is ended, the
    contestant's largest by the runner's memory watch where swap is unconstrained, or the kernel's
    OOM killer's pick where it is not, and the sandbox ends only when that does not bring the
    memory back (``hls.substrate.memory_watch``), which ends the build: graded on the last commit,
    or with *grade_after_sandbox_oom* false scored zero with nothing graded, as the build's plan
    rules (``run.grade_after_sandbox_oom``). Past its writable space it ends,
    and the build is graded on the last commit either way; a process of it whose open files the
    harness cannot list counts as past that space. Its commands run with no_new_privs and cannot
    make themselves non-dumpable (``hls.agent.spawner``). Every figure is the sandbox role's own,
    which the substrate enforces, never its stage's or the node's."""
    # Imported here: this module is imported by the dataset specs, which need none of the watch.
    from hls.agent.spawner import DUMPABLE_REFUSAL
    from hls.substrate.memory_watch import DISK_WALK_MAX_DEPTH, DISK_WALK_MAX_ENTRIES

    # What the end of the sandbox past its memory or its writable space does to the build; the
    # sentence's end, from "past the files", follows it either way.
    if grade_after_sandbox_oom:
        ends = ("A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on "
                "what you last committed on main;")
    else:
        ends = ("A sandbox that ends so ends the build, which then scores zero: nothing you committed is graded. A "
                "sandbox that writes past its writable space ends the build, which is graded on what you last "
                "committed on main;")
    if nproc_accounting == "uid":
        forks = (f"Your commands can start a process or a thread only while the node holds fewer than "
                 f"{command_nproc:,} of them for the user they run as, the harness's and other builds' included; "
                 "past that a fork fails with EAGAIN.")
    else:
        forks = (f"Your commands can start a process or a thread only while they hold fewer than {command_nproc:,} "
                 "of them together; past that a fork fails with EAGAIN.")
    return (f"Your sandbox has {memory_gib} GiB of memory, of which the harness's own processes may hold up to "
            f"{harness_gib:g} GiB, and {cpus} CPUs. Its `/dev/shm` holds up to {shm_gib} GiB, and what is there "
            f"counts in that memory. You can write to {_path_list(writable)}: everything there and your session's "
            f"logs share {disk_gib} GiB of writable space, separate from your memory, and hold at most "
            f"{DISK_WALK_MAX_ENTRIES:,} files, directories and links, with no directory nested more than "
            f"{DISK_WALK_MAX_DEPTH} levels deep. The objects your repository's `.git` holds, its history and "
            f"unreachable objects among them, may come to at most those {disk_gib} GiB together uncompressed, and "
            f"the bundle of main the harness makes of them to at most {disk_gib} GiB; past either, the build has no "
            "submission and scores zero. Every object git stores as a delta counts with what rebuilding it reads: "
            "itself, its delta, and the same for the object it is a delta on, down its chain. Those may come to at "
            f"most {chain_gib} GiB together; past that, the build has no submission and scores zero. {forks} "
            + (f"{held} " if (held := ucounts_sentence("Your commands, the agent's own among them,", ucounts)) else "")
            + "When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' "
            "processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending "
            f"processes does not bring the memory back under it, the sandbox ends. {ends} "
            "past the files or the nesting, it has run out of its writable space, and so it has when one of its "
            "processes holds open files the harness cannot list, as a process running a program from a file it "
            "cannot read does. Your commands run with `no_new_privs` set, and `prctl(PR_SET_DUMPABLE, 0)` fails "
            f"with {errno.errorcode[DUMPABLE_REFUSAL]}.")


def serving_brief_section(*, gpus: int, gpu_kind: str, cpus: int, memory_gib: int, writable_gib: int,
                          served_writable: Sequence[str], ucounts: Mapping[str, int]) -> str:
    """What the agent's server gets when it is graded, for the ``## Your budget`` section beside
    the sandbox's: the task's GPUs, and the CPUs, memory and writable space of the evaluation's
    allocation (``EvaluationAllocation``), for which the agent builds; that ``/workspace`` is
    read-only there, with the paths it may write, *served_writable*, and the files and nesting the
    disk watch holds its writable tree to, past which, as past its space, the server ends as the
    contestant's (``hls.substrate.memory_watch``), and that a process of it
    whose open files the harness cannot list counts as past that space; and the limit of each
    per-uid kernel count its sealed role is held to (*ucounts*, ``ucounts_sentence``)."""
    # Imported here: this module is imported by the dataset specs, which need none of the watch.
    from hls.substrate.memory_watch import DISK_WALK_MAX_DEPTH, DISK_WALK_MAX_ENTRIES

    held = ucounts_sentence("It and every process it starts", ucounts)
    return (f"When your server is graded, it runs with {gpus} {gpu_kind} GPU{'' if gpus == 1 else 's'}, {cpus} CPUs, "
            f"{memory_gib} GiB of memory and {writable_gib} GiB of writable space, which holds at most "
            f"{DISK_WALK_MAX_ENTRIES:,} files, directories and links, with no directory nested more than "
            f"{DISK_WALK_MAX_DEPTH} levels deep; past any of those the server is stopped, and the grading scores "
            f"zero. Build it for those. `/workspace` is read-only there: it may write only to "
            f"{_path_list(served_writable)}. A process of it that holds open files the harness cannot list "
            "counts as past its writable space." + (f" {held}" if held else ""))


def drive_bounds_brief_section(*, trace_cap_bytes: int, reply_max_bytes: int, verify_limited: bool) -> str:
    """What a graded drive holds the server's replies to, for the ``## Your budget`` section beside
    the serving one (``hls.drive.bounds``): its trace's cap, *trace_cap_bytes*, the
    task's own, which a reply fails at once the trace recorded, its sent and unrecorded requests
    counted at the JSON the trace will record of them and its received and unrecorded replies
    counted at their most, reach it, and, when *verify_limited*, that the cap is what the
    verification can hold; one reply's cap, *reply_max_bytes*; and that a reply in a content
    coding fails, since the drive asks for uncompressed replies (``hls.drive.reading``)."""
    why = (" This cap is what the harness's verification of the trace can hold in memory, which is less than "
           "it would otherwise allow." if verify_limited else "")
    return (f"When your server is graded, the harness records each request and its reply in the drive's trace as "
            f"the drive goes, and the trace may hold at most {_size(trace_cap_bytes)} of JSON: once what it has "
            "recorded, with each request it has sent and not yet recorded counted at the JSON the trace will record "
            "of that request, and each reply it has received and not yet recorded counted at the most JSON that "
            "reply's bytes can become, reaches that, every further reply fails as your server's failed "
            f"request.{why} One "
            f"reply may carry at most {_size(reply_max_bytes)}, and a longer one fails; so does a reply in a content "
            "coding other than identity, since the harness asks for uncompressed replies.")


def _size(n: int) -> str:
    """*n* bytes as the brief states a limit: whole GiB or MiB when it is one, else bytes with
    thousands separators."""
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20)):
        if n >= scale and n % scale == 0:
            return f"{n // scale} {unit}"
    return f"{n:,} bytes"


def _duration(sec: int) -> str:
    """*sec* seconds as the brief states a time limit: whole hours or minutes when it is one, else
    seconds with thousands separators."""
    for unit, scale in (("hour", 3600), ("minute", 60)):
        if sec >= scale and sec % scale == 0:
            return f"{sec // scale} {unit}{'' if sec == scale else 's'}"
    return f"{sec:,} seconds"


def packaging_brief_section(*, held: Mapping[str, str] | None, pip_disk_gib: int, pip_memory_gib: int,
                            collect_sec: int, bundle_sec: int) -> str:
    """The ``## Packaging limits`` section: each limit, read from the constant that is checked
    against when the section renders, so the brief states what the code enforces.
    It opens with the time the harness has to collect the repository once the
    build has ended, at the hard limit or before it, *collect_sec* at least
    (``run.collect_allowance_sec``), the commit point's *bundle_sec* of it at least
    (``hls.stages.agent_build.commit_point_sec``), and that a repository it cannot bundle in that
    time is not collected and the build is not graded: such a build is void, infra that is never
    the contestant's. The commit point's limits come next, past which the
    build ends ``over_limit`` with nothing gradable: the ones the repository's size and its deltas'
    chains are held to are the sandbox's writable cap and ``CHAIN_BYTES_MAX``, which the budget
    section states (``sandbox_brief_section``), so this one points there for them. Then the
    package's, past which the submission is an invalid package, the first of them that main's
    bundle reads back as git reads it (``hls.package.checkout.receive``, ``bundle_verdict``), the
    dry run's report among them by its size and its nesting (``REPORT_MAX_BYTES``,
    ``REPORT_MAX_DEPTH``), and the disk and memory each pip step may use (*pip_disk_gib*, *pip_memory_gib*, the plan's
    ``hls.stages.package.pip_limits``), past which the step is stopped as the contestant's fault.
    With *held* (package name to version, in the base's order), it adds that those packages are
    held when the server is built, where the base's constraints file applies to every install, and
    that the sandbox sets the same constraints by default, through variables the agent can clear
    (``BASE_ENVIRONMENT``)."""
    # Imported here: this module is imported by the dataset specs, and these modules import far
    # more than a task module may load on the login node.
    from hls.agent.commit import (ALTERNATES_MAX, DELTA_MAX_BYTES, NONBLOB_MAX_BYTES, OBJECT_MAX_BYTES, OBJECTS_MAX,
                                  PACKS_MAX, TREE_BYTES_MAX, TREE_MAX_DIRECTORIES, TREE_MAX_ENTRIES,
                                  TREE_MAX_PATH_BYTES, UNREADABLE, walk_max_entries)
    from hls.egress.proxy import PAGE_MAX_BYTES, UPSTREAM_FETCHES_MAX
    from hls.package.checkout import (PYPROJECT_MAX_BYTES, TREE_MAX_BYTES, TREE_MAX_DEPTH, TREE_MAX_NAME_BYTES,
                                      unapplied_attributes)
    from hls.package.offline import INSTALL_MAX_GIB
    from hls.package.wheel_check import (DEPTH_MAX, DIRECTORY_MAX_BYTES, METADATA_MAX_BYTES, MODULE_MAX_BYTES,
                                         NAME_MAX_BYTES, PACKING_METHODS, PATHS_MAX)
    from hls.recipes import model
    from hls.package.wheels import PIP_STEPS_SEC, REPORT_MAX_BYTES, REPORT_MAX_DEPTH

    text = ("When your build ends, the last commit on `main` is bundled and built offline into the image your "
            f"server is graded in. The harness then has at least {_duration(collect_sec)} to collect your "
            f"repository, at least {_duration(bundle_sec)} of it to bundle it; a repository it cannot bundle in that "
            "time is not collected, and your build is not graded. Past any of these limits your build ends "
            "`over_limit`, with nothing gradable, and scores zero:\n"
            "- the repository's objects, history and unreachable ones included, and the bundle of `main` stay "
            "within your sandbox's writable space, and what rebuilding its deltas reads within its bound, as the "
            "budget section below states;\n"
            "- the repository's tree objects, history and unreachable ones included, hold at most "
            f"{_size(TREE_BYTES_MAX)} together;\n"
            f"- the tree of `main`'s last commit lists at most {TREE_MAX_ENTRIES:,} paths and "
            f"{TREE_MAX_DIRECTORIES:,} directories, each counted wherever it appears (a directory the tree names "
            f"twice counts twice, with all it holds), and none of its paths, a directory's included, is longer than "
            f"{_size(TREE_MAX_PATH_BYTES)};\n"
            f"- the repository holds at most {OBJECTS_MAX:,} objects, none larger than {_size(OBJECT_MAX_BYTES)};\n"
            f"- the repository's objects lie in at most {ALTERNATES_MAX} object directories, its own and those its "
            f"alternates name, and in at most {PACKS_MAX:,} packs among them; those directories and its refs hold at "
            f"most {walk_max_entries():,} files and directories together;\n"
            f"- a commit, tree or tag is at most {_size(NONBLOB_MAX_BYTES)}, and an object stored as a delta at most "
            f"{_size(DELTA_MAX_BYTES)}.\n\n"
            "Your build also ends `unreadable`, with nothing gradable, and scores zero, when the commit point cannot "
            "read the repository in your workspace:\n"
            + ";\n".join(f"- {cause}" for cause in UNREADABLE.values()) + ".\n\n"
            "Each file of your package is written byte for byte as `main`'s tree holds it: the conversions a "
            f"`.gitattributes` asks git for ({_joined([f'`{a}`' for a in unapplied_attributes()])}) are not "
            "applied, though git applies them to the files in your sandbox.\n\n"
            "Past any of these limits your submission is an invalid package and scores zero:\n"
            "- the bundle of `main` reads back as git reads it: no object `main`'s history names is missing from it, "
            "of another kind, or stored under another object's name (a loose object's file name or a pack's index "
            "can store one so), and git can inflate every object in it and apply every delta;\n"
            f"- the committed tree holds at most {_size(TREE_MAX_BYTES)} of files; a path is at most "
            f"{TREE_MAX_DEPTH} levels deep, each name in it at most {_size(TREE_MAX_NAME_BYTES)}, and every path is "
            "UTF-8 and holds no name git refuses to check out, such as `..` or one NTFS reads as `.git` (`.git`, "
            "`.GIT.`, `git~1`);\n"
            "- the tree holds no symlink;\n"
            f"- `pyproject.toml` holds at most {_size(PYPROJECT_MAX_BYTES)};\n"
            f"- installing your wheels adds at most {INSTALL_MAX_GIB} GiB to the image's Python environment, and "
            "their entries declare at most that much together;\n"
            "- pip's two steps, a dry run that resolves your dependencies and then the build of the wheels, both "
            f"running your project's build hooks, take at most {_duration(PIP_STEPS_SEC)} together, counted from "
            "each step's command's start, unless the package index failed meanwhile;\n"
            f"- each pip step fetches at most {UPSTREAM_FETCHES_MAX:,} pages and files from the package index, "
            f"redirects included, and no dependency's index page is larger than {_size(PAGE_MAX_BYTES)};\n"
            f"- the dry run's report of what it would install holds at most {_size(REPORT_MAX_BYTES)} and nests no "
            f"deeper than {REPORT_MAX_DEPTH} levels;\n"
            "- the wheel step leaves only wheel files: your project's wheel and at most one wheel of each dependency "
            "the dry run resolved;\n"
            f"- your wheels together install at most {PATHS_MAX:,} paths (each file, each directory above one and "
            f"each script their entry points declare), nest no entry deeper than {DEPTH_MAX} levels, have no name "
            f"in an entry's path longer than {_size(NAME_MAX_BYTES)}, hold no Python source file larger than "
            f"{_size(MODULE_MAX_BYTES)} and no `.dist-info` entry larger than {_size(METADATA_MAX_BYTES)}, and none "
            f"lists its entries in more than {_size(DIRECTORY_MAX_BYTES)};\n"
            "- every wheel is a zip each of whose entries is packed by "
            + " or ".join(f"`{zipfile.compressor_names[m]}`" for m in sorted(PACKING_METHODS))
            + " and unpacks to the size and checksum it declares.\n\n"
            f"Each pip step may write at most {pip_disk_gib} GiB to the node's disk, its copy of your tree, its "
            f"temporary files and its wheels together, and hold at most {pip_memory_gib} GiB of memory; a step past "
            "either is stopped, and the submission scores zero.")
    if held:
        pins = _joined([f"`{name}` {version}" for name, version in held.items()])
        verb, which, noun = ("is", "it", "constraint") if len(held) == 1 else ("are", "one of them", "constraints")
        variables = ", ".join(f"`{name}`" for name, value in sorted(model.BASE_ENVIRONMENT.items())
                              if value == model.CONSTRAINTS)
        text += (f"\n\n{pins} {verb} held when your server is built: an install that asks for another version of "
                 f"{which} fails there. Your sandbox sets the same {noun} by default ({variables}).")
    return text


# ---------------------------------------------------------------------------
# Calibrated verification thresholds (the `verification` render variable)
# ---------------------------------------------------------------------------


def _threshold_pct(value: float) -> str:
    """*value* (a fraction) as a percentage string carrying 4 significant figures.

    Rounds to nearest: a 2-figure downward floor would render "0.63%" for an
    enforced 0.6395% gate, understating the threshold the agent is graded
    by. Fixed-point (not ``%.4g``, which strips
    trailing zeros), so a 0.3930% gate reads "0.3930", never "0.393". Rounding
    that carries the value across a power of ten (0.99995 -> "1.0000") keeps
    the finer decimal count rather than dropping a figure.
    """
    pct = value * 100.0
    decimals = max(0, 3 - math.floor(math.log10(abs(pct))))
    return f"{pct:.{decimals}f}"


def _multiplier(value: float) -> str:
    """A tolerance multiplier the way the brief says it: "2x", "1.5x"."""
    return f"{value:g}"


# Attributes a calibration-requiring verifier must expose for the brief to state
# its thresholds: TokenFaithfulnessVerifier's rate rule and fail-closed token
# floor. A calibration-requiring verifier of some other shape contributes no
# numbers rather than numbers the brief would be guessing at.
_VERIFIER_THRESHOLD_FIELDS = ("rate_multiplier",)


def verification_fields(taskspec: Any, tenants: list | None = None) -> dict[str, Any] | None:
    """The calibrated token-level gate, as numbers a brief can state -- or None.

    None whenever the brief must stay qualitative: no calibration-requiring
    verifier across the effective tenants, or none of them carries an injected
    ``calibration``. The agent_build stage loads its task with the calibration
    injected (``hls.taskref.load_task``), which refuses a calibration that
    is missing or does not match the task, so the brief a build agent reads states
    the numbers; a task rendered without its calibration gets the qualitative
    fallback.

    There is one threshold to state, not one per tenant: the token-level gate is
    pooled, so it grades every tenant's events as one population and returns a
    single verdict for the run. Several tenants may declare it, but the
    verification core (``hls.verify.run.verify_trace``) refuses a pool whose
    declarations differ in configuration, so any live one states the gate the
    whole run is held to.
    """
    tenants = taskspec.effective_tenants() if tenants is None else tenants
    calibrated = [(v, cal) for t in tenants for v in t.verifiers
                  if getattr(v, "needs_calibration", False)
                  and (cal := getattr(v, "calibration", None)) is not None]
    # The report gate states two numbers, each a calibrated constant times the declared
    # tolerance: the ceiling on the mean divergence, and the limit on how often a
    # served token may fall outside the reported alternatives.
    report = [(v, cal) for v, cal in calibrated
              if hasattr(v, "tolerance") and hasattr(cal, "kappa")]
    if report:
        verifier, cal = report[0]
        ceiling = verifier.tolerance * cal.kappa
        outside_limit = verifier.tolerance * cal.outside_ceiling
        return {"kind": "report", "kl_ceiling": ceiling,
                "kl_ceiling_str": f"{ceiling:.2e}", "kappa": cal.kappa,
                "outside_limit": outside_limit,
                "outside_limit_pct_str": f"{100 * outside_limit:.2f}",
                "tolerance": verifier.tolerance,
                "multiplier_x": _multiplier(verifier.tolerance)}
    live = [(v, cal) for v, cal in calibrated
            if all(hasattr(v, f) for f in _VERIFIER_THRESHOLD_FIELDS)]
    if not live:
        return None
    verifier, cal = live[0]
    flip_rate_threshold = verifier.rate_multiplier * cal.q_max
    return {
        "kind": "flips",
        # Thresholds, raw and pre-formatted for prose.
        "flip_rate_threshold": flip_rate_threshold,
        "flip_rate_pct": _threshold_pct(flip_rate_threshold),
        # The fail-closed floor: a trace carrying fewer teacher-forced tokens fails
        # outright.
        "min_tokens": verifier.min_tokens,
        # What the threshold was derived from.
        "q_max": cal.q_max,
        "rate_multiplier": verifier.rate_multiplier,
        "multiplier_x": _multiplier(verifier.rate_multiplier),
        # The curve check's gap-level grid in nats (ascending). Its per-level
        # ceilings depend on the graded trace's length, which no brief can know in
        # advance, so a brief can only state the rule qualitatively, which is why
        # the shipped brief does not enumerate these levels. Exposed for templates
        # that want them and for diagnostics.
        "flip_curve_levels": list(cal.curve_side.levels_nats),
    }


#: The decode a task with no calibrated reference renders its brief under: plain
#: greedy.
_PLAIN_GREEDY = {"kind": "greedy", "frequency_penalty": 0.0, "temperature": 0.0}


def _describe_decode(decode: DecodeConfig) -> dict:
    """One reference decode as the fields the brief's Decoding section branches on.

    ``kind`` selects the paragraph: "greedy" (argmax, with or without a frequency
    penalty) or "sample" (a draw from the model's distribution at a temperature).
    The other two fields are the numbers those paragraphs state. An unrecognized
    decode strategy raises rather than rendering greedy prose for it, because a
    brief that describes the wrong decode sends every contestant into a gate they
    cannot pass.
    """
    if isinstance(decode, Greedy):
        return {"kind": "greedy",
                "frequency_penalty": float(decode.frequency_penalty),
                "temperature": 0.0}
    if isinstance(decode, Sample):
        return {"kind": "sample", "frequency_penalty": 0.0,
                "temperature": float(decode.temperature)}
    raise ValueError(
        f"no brief prose exists for a {type(decode).__name__} decode; the Decoding "
        "section of the brief has to describe the decode a server must implement.")


def reference_decode(taskspec: Any, tenants: list | None = None) -> dict:
    """The decode the brief must tell a contestant to implement.

    Read from the calibration-requiring verifiers' reference decode, which the task
    declares, so from the taskspec alone: unlike ``verification_fields`` this needs
    no injected calibration, and a brief rendered without one still states the
    decode a server has to implement. Plain greedy when no such verifier declares a
    reference decode.

    Several calibrated verifiers declaring different decodes raises: a brief states
    one decode, and there is no way to write two.
    """
    tenants = taskspec.effective_tenants() if tenants is None else tenants
    decodes = [_describe_decode(decode)
               for t in tenants for v in t.verifiers
               if getattr(v, "needs_calibration", False)
               and (decode := getattr(getattr(v, "reference", None), "decode", None))
               is not None]
    distinct = [d for i, d in enumerate(decodes) if d not in decodes[:i]]
    if not distinct:
        return dict(_PLAIN_GREEDY)
    if len(distinct) > 1:
        raise ValueError(
            f"this task's calibrated verifiers declare {len(distinct)} different "
            f"decodes ({distinct}); the brief states one decode and cannot describe "
            "two. Pin one decode across them.")
    return distinct[0]


# Single shared environment. The base brief is registered under "brief" so child
# templates can `{% extends "brief" %}`.
_ENV = jinja2.Environment(
    loader=jinja2.DictLoader({"brief": _BRIEF}),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Module-level compiled-template cache, keyed by source string. The compiled
# ``jinja2.Template`` carries an ``LRUCache`` whose entries become weakrefs once
# the shared ``_ENV`` is touched; ``weakref.ReferenceType`` is not picklable, so
# we never store the compiled object on the (serializable) ``Spec`` -- only the
# source string round-trips through provenance, and the compiled template is
# rebuilt on demand here.
_COMPILED: dict[str, jinja2.Template] = {}


# ---------------------------------------------------------------------------
# Provenance models
# ---------------------------------------------------------------------------


class SectionProvenance(BaseModel):
    """Provenance record for a single brief block."""

    block: str
    source: Literal["base", "task"]
    rendered: str


class BriefProvenance(BaseModel):
    """Aggregated per-section provenance for a rendered implementer brief."""

    sections: list[SectionProvenance]


# Canonical block render order -- matches the block declaration order in _BRIEF.
# Two blocks are headerless (the harness emits no "## " heading; both recorded in
# provenance like every other block):
# - ``uploads``: the task-authored ``producer.upload_brief`` rendered verbatim
#   after Build Instructions -- the task supplies its own heading if it wants one.
# - ``directives``: the trailing block -- a task's standing directives (e.g. a
#   /goal for the ralph loop) render verbatim after Submission. Content comes from
#   ImplementerBrief.directives_section() (empty by default; tasks override the hook).
_BLOCK_ORDER = [
    "goal",
    "model",
    "hardware",
    "workload",
    "contract",
    "decoding",
    "build_instructions",
    "uploads",
    "submission",
    "packaging",
    "budget",
    "directives",
]

def _nested_block_parents(template_str: str) -> dict[str, str]:
    """Map each block nested inside another block to its innermost enclosing
    block's name, by walking *template_str*'s parsed AST. A block at the top
    level (not inside another ``{% block %}``) is omitted.

    ``jinja2.nodes.Block`` nodes nest the way the source text nests them, so a
    block found while already inside another block records that block as its
    parent; this is how ``dev_reference_note`` is discovered under
    ``build_instructions`` without hand-copying the template's structure.
    """
    def walk(node: jinja2.nodes.Node, enclosing: str | None) -> dict[str, str]:
        found: dict[str, str] = {}
        if isinstance(node, jinja2.nodes.Block):
            if enclosing is not None:
                found[node.name] = enclosing
            enclosing = node.name
        for child in node.iter_child_nodes():
            found.update(walk(child, enclosing))
        return found

    return walk(_ENV.parse(template_str), None)


#: Blocks nested inside a headed block, derived once from the base brief's AST. A
#: task that overrides one has authored part of the parent's text, so provenance
#: sources the parent as "task".
_NESTED_BLOCKS = _nested_block_parents(_BRIEF)


def _defined_blocks(template_str: str) -> set[str]:
    """Return the set of block names explicitly defined in *template_str*.

    Parses the source via _ENV rather than compiling it so we can walk the AST
    cheaply without the compiled template touching the module cache.
    """
    ast = _ENV.parse(template_str)
    return {node.name for node in ast.find_all(jinja2.nodes.Block)}


def _with_enclosing_blocks(blocks: set[str], parents: dict[str, str]) -> set[str]:
    """*blocks* plus every block enclosing one of them, followed outwards to the top.

    Overriding a nested block authors part of its parent's text, and of its
    grandparent's when the nesting runs deeper than one level, so a single lookup
    per name is not enough. *parents* maps a nested block to its innermost
    enclosing one.
    """
    while True:
        grown = blocks | {parents[b] for b in blocks if b in parents}
        if grown == blocks:
            return blocks
        blocks = grown


def _block_source(
    block: str, task_blocks: set[str], directives_text: str, uploads_text: str
) -> Literal["base", "task"]:
    """Classify a block's provenance source for the SectionProvenance record.

    A block is "task"-sourced when a task template overrode it (its name is in
    *task_blocks*). Two blocks are special -- their content arrives as text the
    task authored rather than a ``{% block %}`` override, so they are "task"
    whenever that text is present and "base" (empty) otherwise: ``directives``
    (the ``directives_section()`` hook) and ``uploads`` (the spec's
    ``producer.upload_brief``, threaded in as ``uploads_section``).
    """
    if block == "directives":
        return "task" if directives_text else "base"
    if block == "uploads":
        return "task" if (uploads_text or block in task_blocks) else "base"
    return "task" if block in task_blocks else "base"


class PromptTemplate(Poly):
    """A renderable prompt fragment.

    `render(**context)` returns the final string. The dataset column templates are the
    instances (``JinjaPromptTemplate``); they render through the one shared Jinja
    environment, as the implementer brief does. A poly family so dataset/prompt config
    round-trips in provenance; subclasses register a ``kind`` and plug in an engine.
    """

    @abstractmethod
    def render(self, **context: Any) -> str: ...


class JinjaPromptTemplate(PromptTemplate):
    """A Jinja template string rendered in the shared HLS environment.

    Use for dataset column templates and for any standalone fragment. A template
    may `{% extends "brief" %}` to inherit the default brief and override named
    blocks; standalone templates render their own body. Only the source
    ``template`` string is held (and serialized); the compiled template is cached
    module-level (neither serialized nor pickled).
    """

    template: str

    def render(self, **context: Any) -> str:
        t = _COMPILED.get(self.template)
        if t is None:
            t = _COMPILED[self.template] = _ENV.from_string(self.template)
        return t.render(**context)


class ImplementerBrief(Poly):
    """The build-agent brief, owned by AgentProducedArtifact (artifact.py).

    Renders the default brief, or a task-supplied child template that
    `{% extends "brief" %}` and overrides blocks (goal, model, hardware, contract,
    decoding, build_instructions, and the `dev_reference_note` block nested within
    build_instructions).
    The render context is assembled from each TaskSpec component's `prompt_fields()`;
    the protocol section comes from `contract.brief_section()`. The agent_build stage
    renders it with ``render_with_provenance``, giving every section the brief states;
    unlike a ``PromptTemplate`` it has no ``render(**context)``.
    """

    template_str: str | None = None
    # Whether the render context carries the calibrated verification thresholds
    # (the `verification` variable). Default True: a task's goal block states the
    # gate in numbers. False renders `verification` as None even when a calibration
    # is injected, so the template's qualitative fallback branch is what the agent
    # reads -- the run is still gated by the full calibrated verifier; only the
    # brief withholds the numbers.
    state_thresholds: bool = True

    def directives_section(self) -> str:
        """Trailing, headerless directives appended after every other block.

        Returns the verbatim content of the ``directives`` block (the last entry
        in _BLOCK_ORDER) -- e.g. a task's standing ``/goal`` for the ralph loop or
        its concurrency framing. The default brief has none (returns ""), so the
        block renders empty and is dropped from the brief like any other empty
        section. Tasks override this hook, never ``render_with_provenance``, to
        contribute directives. The content carries no "## " heading: it is emitted
        verbatim and recorded as a SectionProvenance like the rest.
        """
        return ""

    @property
    def _template(self) -> jinja2.Template:
        if self.template_str is None:
            return _ENV.get_template("brief")
        t = _COMPILED.get(self.template_str)
        if t is None:
            t = _COMPILED[self.template_str] = _ENV.from_string(self.template_str)
        return t

    def _context(
        self,
        *,
        taskspec: Any,
        server_path: str,
        submission_section: str,
        packaging_section: str,
        budget_section: str,
        uploads_section: str,
        blocked_packages: list[str] | None,
        reference_service: bool,
    ) -> dict[str, Any]:
        """Assemble the Jinja render context from taskspec components.

        ``blocked_packages`` is the producer's package deny list (``NetworkPolicy.packages``
        in deny mode), threaded in by the agent_build stage so the Build Instructions name
        exactly what the index proxy refuses; None renders as "nothing blocked".

        ``reference_service`` is the producer's flag of that name
        (``AgentProducerSpec.reference_service``), which the agent_build stage reads to run the
        task's pinned reference as a scoring service beside the sandbox. The base brief states
        nothing of it; a task's template states the service only when it is set, so a build
        that runs no service is never told of one.

        ``turn_bound`` is the most turns one trajectory may take (``AgentSpec.max_turns``),
        which the trajectory driver holds every session to; None for a task without an agent.
        The base brief does not state it; a task's template may.

        Workload is resolved via effective_tenants() so multi-tenant tasks (where
        taskspec.workload is None) do not crash. A single-tenant task's workload block
        renders its one tenant's prompt_fields(); a multi-tenant task (>1 tenant) also
        passes tenants_data so the workload block renders a per-tenant subsection (id +
        dataset + workload fields).

        ``verification`` carries the calibrated token-level thresholds when the
        task was loaded with its calibration injected (None otherwise -- see
        verification_fields), so a task's goal block can state the gate in numbers
        instead of prose, and the same template still renders without them.

        ``decode`` is the decode the contestant's server must implement. It comes
        from the taskspec rather than from a calibration, so the decoding paragraph
        is stated on every render path.

        ``dev_rows_path`` is where the agent's development rows are mounted, derived
        from the provisioning convention rather than written out here, so the brief
        cannot name a directory nothing mounts.
        """
        # Imported inside the method: the rows module reaches the dataset spec, which
        # imports this module.
        from hls.provision.rows import tenant_mount_path

        tenants = taskspec.effective_tenants()
        workload_fields = tenants[0].workload.prompt_fields()
        if len(tenants) > 1:
            tenants_data = [
                (t.id, t.workload.prompt_fields(), t.data_source) for t in tenants
            ]
        else:
            tenants_data = None
        # A multi-tenant brief names the path family once, leaving the tenant id as a
        # placeholder: spelling out every tenant's path would bloat the bullet, and the
        # ids are listed in the workload section.
        dev_tenant = "<tenant>" if tenants_data else tenants[0].id
        dev_rows_path = f"{tenant_mount_path(dev_tenant, 'dev')}/rows.json"
        # The request rate summed over tenants, stated once above the per-tenant
        # blocks; None when any tenant's arrival has no rate, since a sum with a term
        # missing would understate the load.
        rates = [t.workload.prompt_fields()["request_rate"] for t in tenants]
        offered_load = (f"{sum(rates):.4g}" if all(r is not None for r in rates)
                        else None)
        return dict(
            ts=taskspec,
            model=taskspec.model.prompt_fields(),
            hardware=taskspec.hardware.prompt_fields(),
            workload=workload_fields,
            tenants_data=tenants_data,
            dev_rows_path=dev_rows_path,
            offered_load=offered_load,
            turn_bound=taskspec.agent.max_turns if taskspec.agent is not None else None,
            verification=(verification_fields(taskspec, tenants)
                          if self.state_thresholds else None),
            decode=reference_decode(taskspec, tenants),
            contract=taskspec.contract.prompt_fields(),
            protocol_section=taskspec.contract.brief_section(),
            server_path=server_path,
            submission_section=submission_section,
            packaging_section=packaging_section,
            budget_section=budget_section,
            uploads_section=uploads_section,
            blocked_packages=blocked_packages,
            reference_service=reference_service,
            directives_section=self.directives_section(),
        )

    def render_with_provenance(
        self,
        *,
        taskspec: Any,
        server_path: str,
        submission_section: str,
        packaging_section: str,
        budget_section: str,
        uploads_section: str,
        blocked_packages: list[str] | None,
        reference_service: bool,
    ) -> tuple[str, BriefProvenance]:
        """Render the brief and return (full_text, BriefProvenance). Every section is the
        caller's to give: *server_path* is where the agent writes its server, and the
        submission, packaging, budget and uploads sections are rendered as given, an empty
        submission, budget or uploads section omitted. *reference_service* says whether the
        build runs the reference scoring service (``_context``).

        Per-block text is extracted by slicing the fully rendered output on
        ``## `` headings rather than re-invoking ``child.blocks[name](...)``.
        The heading-slice approach is more robust when tasks use ``{{ super() }}``
        inside overriding blocks (block-call re-invocation works for simple
        overrides but the recursive super() call path is brittle and can yield
        double-rendered content).  The ``## `` headings are stable: they are
        hard-coded in *_BRIEF* and tasks that ``{% extends "brief" %}`` cannot
        remove them (the heading text lives outside the block tags).
        """
        tmpl = self._template
        ctx = self._context(
            taskspec=taskspec,
            server_path=server_path,
            submission_section=submission_section,
            packaging_section=packaging_section,
            budget_section=budget_section,
            uploads_section=uploads_section,
            blocked_packages=blocked_packages,
            reference_service=reference_service,
        )
        full_text = tmpl.render(**ctx)

        task_blocks: set[str] = _defined_blocks(self.template_str) if self.template_str is not None else set()
        task_blocks = _with_enclosing_blocks(task_blocks, _NESTED_BLOCKS)

        # Heading order mirrors _BRIEF's declared sections.
        heading_to_block: dict[str, str] = {
            "## Goal": "goal",
            "## Model": "model",
            "## Hardware Envelope": "hardware",
            "## Workload": "workload",
            "## Network Contract": "contract",
            "## Decoding": "decoding",
            "## Build Instructions": "build_instructions",
            "## Submission": "submission",
            "## Packaging limits": "packaging",
            "## Your budget": "budget",
        }

        lines = full_text.splitlines(keepends=True)
        section_starts: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            for heading, block_name in heading_to_block.items():
                if stripped == heading:
                    section_starts.append((i, block_name))
                    break

        block_texts: dict[str, str] = {}
        for idx, (start_i, block_name) in enumerate(section_starts):
            body_start = start_i + 1
            body_end = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
            body = "".join(lines[body_start:body_end]).strip()
            block_texts[block_name] = body

        # Guard against a silent miss: every always-present block must have been
        # located by the heading slice. If a heading in _BRIEF is renamed without
        # updating heading_to_block above, the affected block would otherwise
        # silently get rendered="" / source="base" (wrong on both counts) with no
        # error. (submission and budget are optional -- each renders only when its
        # section is non-empty -- so they are excluded here.) An explicit raise is
        # required so the guard survives `python -O` (assert would not).
        expected_always_present = {"goal", "model", "hardware", "workload", "contract",
                                   "decoding", "build_instructions", "packaging"}
        missing = expected_always_present - block_texts.keys()
        if missing:
            raise RuntimeError(
                f"render_with_provenance: heading slice missed blocks {sorted(missing)!r}; "
                "a heading in _BRIEF was likely renamed without updating heading_to_block."
            )

        # ``directives`` and ``uploads`` carry no "## " heading, so the heading slice
        # above cannot locate them; source their rendered text straight from the
        # value the template interpolated. ``directives`` comes from the task hook;
        # ``uploads`` is the task-authored ``producer.upload_brief`` threaded in as
        # ``uploads_section``.
        directives_text = self.directives_section().strip()
        uploads_text = uploads_section.strip()

        # Without a heading to stop at either, the last "## " section's body runs to
        # EOF and swallows the trailing directives, and uploads lands inside whatever
        # section precedes it. Left in, that text would be recorded twice (once in
        # the swallowing section, once as its own block) and mis-sourced as "base".
        # Subtract both before installing them as blocks in their own right, or they
        # would erase themselves.
        for swallowed in (directives_text, uploads_text):
            if not swallowed:
                continue
            for name, body in list(block_texts.items()):
                if swallowed in body:
                    block_texts[name] = body.replace(swallowed, "").strip()

        block_texts["directives"] = directives_text
        block_texts["uploads"] = uploads_text

        sections = [
            SectionProvenance(
                block=block,
                source=_block_source(block, task_blocks, directives_text, uploads_section.strip()),
                rendered=block_texts.get(block, ""),
            )
            for block in _BLOCK_ORDER
        ]
        return full_text, BriefProvenance(sections=sections)
