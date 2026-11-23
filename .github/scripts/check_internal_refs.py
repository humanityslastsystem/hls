#!/usr/bin/env python3
"""Count the lines of tracked files that name something only the authors' group can use or look
up: its sites, nodes and accounts, its Slurm jobs, the dates and names of its runs, its pull
requests and commits, the ids of its rulings (``R-<name>``), and the history of the code.

    python .github/scripts/check_internal_refs.py [--list CLASS|all] [--strict]
                                                  [--strict-classes CLASS[,CLASS...]]

Run from the repository's root. It prints, for each class of reference, the lines that match it
by where they sit (a comment, a docstring or other text), and then, by top-level directory, the
lines that match any class. In a Python file other text is code and string
literals, which can reach run records, briefs and error messages; every line of another file is
other text. ``--list`` prints each line of one class, or of every class, as
``path:line: [class/kind] text``. The file list comes from ``git ls-files``; the harbor
submodule, lock files, JSON and media are not read.

A pattern is a floor: it matches the forms a reference takes, and some text that is not one.
``ALLOW`` names each such text by its file and the exact token, with the reason it is not a
reference; a line is counted when its pattern still matches with those tokens taken out.

The exit status is 0, or 1 when a line matches a strict class: every class with ``--strict``,
the classes ``--strict-classes`` lists otherwise. The lines that fail it are then printed."""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict

#: The names of the sites the patterns look for. Each is split with ``+``, as are the other words
#: below, so this file does not match its own patterns.
SITES = ("klo" + "ne", "tilli" + "cum", "cori" + "ander", "cay" + "enne", "hy" + "ak")
#: The parts of the harness. A sentence that says one of them has stopped doing something tells the
#: code's past; with another subject (a cache, a job, a process) the same words tell a run's state.
PARTS = "|".join(("code", "harness", "driver", "runner", "orchestrator", "planner", "pipeline", "stage",
                  "verifier", "gate", "scorer", "task", "brief", "contract", "policy", "command", "option",
                  "flag", "setting", "field", "function", "module", "script", "viewer"))
#: What may stand between such a subject and the words.
AUX = "((is|are|can|does|do|will) )?"
#: After a word or a quoted name of the same line that is not a form of "be".
NOT_BE = r"(?<=[\w`] )(?<!\bis )(?<!\bare )(?<!\bbe )(?<!\bbeen )(?<!\bbeing )(?<!\bwas )(?<!\bwere )"
#: A count, as digits or as a word.
COUNT = r"(\d+|two|three|four|five|six|seven|eight|nine|ten)"
#: Each class of internal reference, as a regular expression.
CLASSES = {
    # A site, an account, a partition, a mount or a user of the authors'.
    "site": "|".join((*SITES, "zl" + "ab", r"\bgpu-h" + r"200\b", r"\bgpu-l" + r"40\b", r"\bcpu-" + r"g2\b",
                      "gscr" + "atch", "/gp" + "fs", "mm" + "fs1", "steve" + "ng0", "/home/" + "steven",
                      r"\bU" + r"W\b")),
    # A node's name, and not the digits after the ``\n`` of a string literal.
    "node": r"(?<!\\)\bg\d{3,4}\b|(?<!\\)\bn\d{4}\b",
    # A job or step id: after the word, in parentheses in prose (not a call's argument), or as
    # eight digits. Eight digits that start with a zero (a padded counter), that read as a day
    # (a seed such as 20240915) or that begin a UUID are not ids. Slurm's accounting tool is the
    # harness's own vocabulary; only a reading quoted from it counts.
    "job_id": (r"\bjobs? \d{5,}|\bstep \d{5,}|(?<![\w.])\(\d{6,8}\)"
               + r"|\b(?!20\d\d[01]\d[0-3]\d\b)[1-9]\d{7}(\.\d+)?\b(?!-)"
               + r"|\bsa" + r"cct\b.{0,60}?\d(\.\d+)? ?[KMG](i?B)?\b"),
    # A day from 2026 on, the year of the repository's first commit, or a month and a day that
    # are not the end of a longer date.
    "date": r"\b20(2[6-9]|[3-9]\d)-\d\d-\d\d\b|(?<![\d-])\b(0?[1-9]|1[0-2])-\d\d\b(?= |\)|,|:)",
    # The arms of the authors' runs, a round of them, and a round's cell (the round, a ``/`` and
    # the arm; a segment of a store key, which follows a ``/``, is not one). A model's own id
    # (``kimi-k3``, ``GLM-5.3-Flash``) is public.
    "arm": (r"\b(" + "|".join(("op" + "us5", "op" + "us48", "op" + "us55", "gp" + r"t56(sol)?", "gp" + "t55",
                               "ki" + "mik3", "gl" + "m53", "fa" + r"ble5\d?"))
            + r")\b|\bround \d\b|(?<!/)\br[0-5]/"),
    # The authors' gradings, measurements and calibrations cited as evidence, their runs and
    # builds as a known set ("the ... runs", "... builds had"), a count of their builds, runs,
    # drives, records or cells, and the names of their runs. "Published image", "published
    # rollout", "a published run", "campaign" and "reroll" alone are the harness's own
    # vocabulary: only a numbered or named campaign or reroll counts.
    "published": ("pub" + r"lished ((?!an? |the |its )\w+ )?(grading|measurement|calibration)s?\b"
                  + r"|\b" + COUNT + " (of the )?pub" + r"lished (\w+ )?(build|run|drive|record|cell)s?\b"
                  + r"|\bevery pub" + r"lished (build|run|drive)s?\b"
                  + r"|\b(the|their|its|our|task's) pub" + r"lished (runs|builds)\b"
                  + r"|\bpub" + r"lished (runs|builds) (were|used|served|got|had|wrote|ran)\b"
                  + "|hls-" + "traces|hls-" + r"pr\d+|\b(tol|cal|kl|pit)\d{2,3}\b|\bthe sl" + r"ate\b"
                  + "|camp" + r"aign/\w|\b\w*\d\w*[ -]camp" + r"aigns?\b|rer" + r"oll-?\d{2,}"),
    # A number after the word or a ``#`` (``#10-`` is a heading's anchor in a document), and the
    # harness's own repositories under the authors' account, whose names start with ``hls``:
    # another repository of the account, such as a public list of sources, is not one.
    "pr_issue": r"\b(PR|issue|pull request) ?#?\d+|#\d{2,3}\b(?!-)|" + "sh" + "g8/hls",
    "commit": r"\b(commit|at|run/[a-z_0-9]+/)\s?[0-9a-f]{7,12}\b",
    # What the code did before. "Used to" counts after its subject, not after a form of "be" or
    # a comma, where it tells a purpose ("is used to"). "No longer" counts when its subject is a
    # part of the harness (``PARTS``), a name with two capitals or a quoted name, and never as a
    # comparison ("no longer than").
    "history": (r"\b(" + "|".join(("previ" + "ously", "former" + "ly", "was" + " once", "at the" + " time",
                                   "the old" + " (path|orchestrator|runner|code)",
                                   NOT_BE + "used" + " to",
                                   "(" + PARTS + ")s? " + AUX + "no" + " longer(?! than)",
                                   r"(?-i:\w*[A-Z]\w*[A-Z]\w*) " + AUX + "no" + " longer(?! than)",
                                   "(?<=` )" + AUX + "no" + " longer(?! than)"))
                + r")\b"),
    # A measurement placed at a named site or on the authors' own machine, and the sites or nodes
    # of their measurements with no name given.
    "measured": (r"\b(measured|took|peaked|passed|ran)\b.{0,60}\b(on|at) ("
                 + "|".join((*SITES[:3], "a " + SITES[0], "(the|a|our) (workstation|development box)"))
                 + r")\b|\bMea" + r"sured 20\d\d"
                 + r"|\bboth (sites|clusters)\b|\b(node|site|cluster)s? measured\b"
                 + r"|\b(slower|faster|smaller|larger) of (the )?(two|both) (sites|clusters|nodes)\b"),
    "ruling": r"\bR-[a-z][a-z0-9]*(-[a-z0-9]+)+",
}
#: Classes matched with case; the rest ignore it.
CASED = frozenset({"node", "ruling"})
PATTERNS = {name: re.compile(rx, 0 if name in CASED else re.IGNORECASE) for name, rx in CLASSES.items()}
#: Text a pattern matches that is not a reference: the class, the file, the exact token and why.
#: The tokens are split with ``+`` as the words above are.
ALLOW = (
    ("date", "src/hls/recipes/pins.py", "released 2026-" + "09-23", "the day the pinned Node was released"),
    ("date", "tasks/deepresearch/task.py", "Hub commit of 2026-" + "07-20", "the day of a public model's commit"),
    ("date", "tests/recipes/test_model.py", '"snapshot": "2026-' + '10-01"', "a made-up value of a refused form"),
    ("date", "tests/recipes/test_locks.py", '"NPM_CONFIG_BEFORE", "2030-' + '01-01"', "a made-up value"),
    ("date", "tests/unit/test_multitenant_task_wiring.py", '["2026-' + '09-05"] * 4',
     "the day the committed baseline credit records for its drives"),
    ("commit", "tests/unit/test_corpus_tokenizer_revision.py", "``Qwen/Qwen2.5-32B`` at " + "1818d358",
     "a public model's commit on the Hub"),
    ("pr_issue", "src/hls/reference/engines.py", "sglang issue " + "#2634", "an issue of SGLang's"),
    ("job_id", "tests/agent/test_exec_server.py", "head -c 6710" + "8864", "64 MiB in bytes"),
    ("job_id", "tests/package/test_stage.py", "cap of 6710" + "8864 bytes (6717" + "4400 read",
     "64 MiB in bytes, and 64 KiB more"),
    ("job_id", "tests/substrate/test_secrets.py", "hf_token = 1234" + "5678", "a made-up integer"),
    ("arm", "tasks/agentic_coding/task.py", "references/gl" + "m53-reference/",
     "the directory of the uploads the brief names, after the task's model"),
    ("arm", "tests/instances/golden/agentic_coding.brief.md", "references/gl" + "m53-reference/",
     "the directory of the uploads the brief names, after the task's model"),
    ("arm", "tasks/agentic_coding/tests/test_brief.py", "references/gl" + "m53-reference/",
     "the directory of the uploads the brief names, after the task's model"),
    ("arm", "tasks/agentic_coding/tests/test_brief.py", 'f"gl' + 'm53-reference/',
     "the directory of the uploads the brief names, after the task's model"),
    ("commit", "tasks/agentic_coding/tests/test_sandbox_images.py", "release at " + "452bf305c6da",
     "the public Terminal-Bench release's commit"),
)
#: Files not read: locks, JSON data, media, and the submodule's URL file.
SKIP = re.compile(
    r"(^|/)(uv\.lock|bun\.lock|.*\.json|.*\.jsonl|.*\.svg|.*\.png|.*\.wav|.*\.flac|locks/.*|\.gitmodules)$")
#: The submodule, which is checked on its own.
SUBMODULE = "harbor"
KINDS = ("comment", "docstring", "other")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True, text=True).stdout
    return [p for p in out.split("\0") if p and not SKIP.search(p) and p.split("/")[0] != SUBMODULE]


def line_kinds(text: str) -> dict[int, str]:
    """The lines of Python source *text* that hold a comment or a docstring, by line number."""
    kinds: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                kinds[tok.start[0]] = "comment"
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    for i in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                        kinds[i] = "docstring"
    except (SyntaxError, tokenize.TokenError):
        pass
    return kinds


def classes_of(rel: str, line: str) -> list[str]:
    """The classes that count *line* of the file *rel*: those whose pattern matches it, and still
    does with the file's allowed tokens of the class (``ALLOW``) taken out."""
    found = []
    for name, rx in PATTERNS.items():
        if not rx.search(line):
            continue
        rest = line
        for cls, path, token, _ in ALLOW:
            if cls == name and path == rel:
                rest = rest.replace(token, "\0")
        if rx.search(rest):
            found.append(name)
    return found


def _classes(value: str) -> list[str]:
    names = [name for name in value.split(",") if name]
    unknown = [name for name in names if name not in CLASSES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"not a class: {', '.join(unknown) or value!r}; the classes are "
                                         f"{', '.join(CLASSES)}")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="check_internal_refs")
    p.add_argument("--list", metavar="CLASS", choices=(*CLASSES, "all"),
                   help="print each matching line of CLASS, or of every class with 'all'")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 when any line matches a class, and print those lines")
    p.add_argument("--strict-classes", metavar="CLASS[,CLASS...]", type=_classes, default=[],
                   help="exit 1 when any line matches one of these classes, and print those lines")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    strict = list(CLASSES) if args.strict else args.strict_classes

    counts: Counter[tuple[str, str]] = Counter()  # (class, kind) -> lines
    internal: Counter[tuple[str, str]] = Counter()  # (top-level directory, kind) -> lines
    files_hit: defaultdict[str, set[str]] = defaultdict(set)
    refused: list[str] = []  # the lines of the strict classes
    for rel in tracked_files():
        try:
            with open(rel, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue
        top = rel.split("/")[0] if "/" in rel else "(root)"
        kinds = line_kinds(text) if rel.endswith(".py") else {}
        for lineno, line in enumerate(text.splitlines(), start=1):
            kind = kinds.get(lineno, "other")
            hit = False
            for name in classes_of(rel, line):
                counts[name, kind] += 1
                files_hit[name].add(rel)
                hit = True
                shown = f"{rel}:{lineno}: [{name}/{kind}] {line.strip()[:150]}"
                if args.list in (name, "all"):
                    print(shown)
                if name in strict:
                    refused.append(shown)
            if hit:
                internal[top, kind] += 1

    if args.list:
        print()
    print(f"{'class':10} {'comment':>8} {'docstr':>7} {'other':>6} {'files':>6}")
    for name in CLASSES:
        row = [counts[name, k] for k in KINDS]
        print(f"{name:10} {row[0]:8} {row[1]:7} {row[2]:6} {len(files_hit[name]):6}")
    print("\nlines with any internal reference, by top-level directory")
    print(f"{'dir':10} {'comment':>8} {'docstr':>7} {'other':>6}")
    for top in sorted({t for t, _ in internal}):
        print(f"{top:10} {internal[top, 'comment']:8} {internal[top, 'docstring']:7} {internal[top, 'other']:6}")
    totals = [sum(v for (_, k), v in internal.items() if k == kind) for kind in KINDS]
    print(f"{'TOTAL':10} {totals[0]:8} {totals[1]:7} {totals[2]:6}")
    if refused:
        print(f"\n{len(refused)} lines of the strict classes ({', '.join(strict)}):")
        print("\n".join(refused))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
