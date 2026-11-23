#!/bin/sh
# What an image holds, for its inventory.json: one tab-separated record per line on stdout.
#   apt   <TAB> name <TAB> version   one per installed dpkg package
#   pip   <TAB> name <TAB> version   one per Python distribution the image's interpreter imports
#   tool  <TAB> name <TAB> version   one per tool in TOOLS; the version is empty when the tool is
#                                    absent, exits nonzero or prints nothing
#   env   <TAB> name                 one per variable of the environment it runs in; never a value
#   error <TAB> kind <TAB> what      a probe that failed, so the inventory is incomplete. kind is
#                                    "transient" when it timed out or was killed (exit 124 or 137)
#                                    and a second run may pass, else "deterministic"
# It runs in any image hls builds or adopts: POSIX sh with tr, sed, cut, sort and env, and
# timeout when the image has one. A missing tool, dpkg or interpreter is a record or no record;
# a failed probe, or a missing one of those five commands, is an error record. No field holds a
# tab or a newline. cut counts bytes, so a line cut at 200 may end inside a character.

TOOLS="gcc g++ clang make cmake ninja pkg-config rustc cargo nvcc ptxas nvdisasm cuobjdump nsys ncu
compute-sanitizer cuda-gdb node npm claude codex tmux git ffmpeg sox python3 uv"
TAB=$(printf '\t')

# The error record of probe $1, which ended with exit status $2.
failure() {
    case "$2" in
        124|137) printf 'error\ttransient\t%s timed out or was killed (exit %s)\n' "$1" "$2" ;;
        *) printf 'error\tdeterministic\t%s exited %s\n' "$1" "$2" ;;
    esac
}

for need in tr sed cut sort env; do
    if ! command -v "$need" >/dev/null 2>&1; then
        printf 'error\tdeterministic\tthe image has no %s\n' "$need"
    fi
done

# Runs a probe for at most $1 seconds when the image has timeout; -k kills one that ignores
# the TERM.
if command -v timeout >/dev/null 2>&1; then
    bounded() {
        seconds=$1
        shift
        timeout -k 10 "$seconds" "$@"
    }
else
    bounded() {
        shift
        "$@"
    }
fi

# Tabs and carriage returns become spaces, so a line stays one record.
clean() {
    tr '\t\r' '  '
}

# The flag that makes tool $1 print its version.
flag_of() {
    case "$1" in
        ffmpeg) printf '%s' -version ;;
        tmux) printf '%s' -V ;;
        *) printf '%s' --version ;;
    esac
}

# The line of tool $1's version output that names its version, else its first non-blank line.
# Nothing when the tool exits nonzero or prints nothing, so a tool that does not run counts as
# absent; the status is the tool's exit status.
version_of() {
    flag=$(flag_of "$1")
    case "$1" in
        nvcc|ptxas|nvdisasm|cuobjdump) pick='/release/' ;;
        ncu|compute-sanitizer) pick='/[Vv]ersion [0-9]/' ;;
        *) pick='/[0-9]/' ;;
    esac
    full=$(bounded 60 "$1" "$flag" 2>&1 </dev/null) || return
    full=$(printf '%s\n' "$full" | clean)
    out=$(printf '%s\n' "$full" | sed -n "${pick}{p;q;}" | cut -c1-200)
    if [ -z "$out" ]; then
        out=$(printf '%s\n' "$full" | sed -n '/[^ ]/{p;q;}' | cut -c1-200)
    fi
    printf '%s' "$out"
}

if command -v dpkg-query >/dev/null 2>&1; then
    # A second status letter "i" is installed, whatever the wanted state: the CUDA images hold
    # their cuBLAS and NCCL packages ("hi").
    # A failed query ends with a "failed" line, which no dpkg status abbreviation can be.
    { bounded 300 dpkg-query -W -f='${db:Status-Abbrev}\t${Package}\t${Version}\n' 2>/dev/null ||
        printf 'failed\t%s\n' "$?"; } |
        while IFS="$TAB" read -r status name version; do
            case "$status" in
                failed) failure dpkg-query "$name" ;;
                ?i*) printf 'apt\t%s\t%s\n' "$name" "$version" ;;
            esac
        done
fi

PY=""
if [ -e /opt/venv/bin/python ] || [ -L /opt/venv/bin/python ]; then
    # The venv's interpreter whenever it is there, so a broken venv is an error record rather
    # than a quiet fallback to another python.
    PY=/opt/venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
fi
if [ -n "$PY" ]; then
    # Isolated (-I), so the working directory and PYTHONPATH binds do not count: this is the
    # image's own set. The first distribution of a name on sys.path is the one imports see.
    # Whatever it cannot read is an error record, and it reads on past it, so a list cut short
    # never reads as complete. It reports its own failures; a nonzero exit means it could not.
    bounded 300 "$PY" -I - 2>/dev/null <<'EOF'
import importlib.metadata as metadata
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Controls, and the separators str.splitlines ends a line at, become one space.
UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")


def field(text, limit=200):
    return UNSAFE.sub(" ", text).strip()[:limit]


def failed(what):
    print(f"error\tdeterministic\tpython: {field(what, 400)}")


try:
    # importlib.metadata skips a directory it cannot list without a word, so list each first.
    for entry in sys.path:
        try:
            os.listdir(entry)
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError as error:
            failed(f"{entry} cannot be listed: {error!r}")
    seen = set()
    for dist in metadata.distributions():
        # PathDistribution keeps its directory in _path; another finder's may not.
        where = getattr(dist, "_path", dist)
        try:
            meta = dist.metadata
            name = field(meta.get("Name") or "")
            version = field(meta.get("Version") or "") or "unknown"
        except Exception as error:
            failed(f"{where}: {error!r}")
            continue
        if not name:
            # An unreadable metadata file reads as none, and so as no name.
            failed(f"{where} has no name, or its metadata cannot be read")
            continue
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        seen.add(key)
        print(f"pip\t{name}\t{version}")
except Exception as error:
    failed(repr(error))
EOF
    status=$?
    if [ "$status" -ne 0 ]; then
        failure "the python inventory" "$status"
    fi
fi

for tool in $TOOLS; do
    if command -v "$tool" >/dev/null 2>&1; then
        version=$(version_of "$tool")
        status=$?
        printf 'tool\t%s\t%s\n' "$tool" "$version"
        # A tool that exits nonzero counts as absent. One that timed out or was killed may yet
        # run, so it is an error record too.
        case "$status" in
            124|137) failure "$tool $(flag_of "$tool")" "$status" ;;
        esac
    else
        printf 'tool\t%s\t\n' "$tool"
    fi
done

env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | sort -u |
    while IFS= read -r name; do
        printf 'env\t%s\n' "$name"
    done
