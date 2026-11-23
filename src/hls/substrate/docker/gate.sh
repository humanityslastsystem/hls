# The entrypoint of every Docker role container (hls.substrate.docker.launcher), run by /bin/sh:
#
#   /bin/sh /.hls-gate/gate [--nvidia VERSION] -- COMMAND...
#   /bin/sh /.hls-gate/gate [--nvidia VERSION] [--oom-score-adj N] --marker LINE -- COMMAND...
#
# The launcher binds its directory read-only at /.hls-gate, with three FIFOs in it. The gate
# sends its stdout and stderr into /.hls-gate/out and /.hls-gate/err, which the launcher's log
# cappers read. With --oom-score-adj it raises its own oom_score_adj to N, as an unprivileged
# process may, so every process the command starts may lower its own back (a harness process
# among them, such as the build sandbox's exec server), as under the step wrapper on Slurm; a
# score Docker set would be the floor of every process in the container. With --nvidia, the
# container holds the files of NVIDIA driver VERSION under /usr/local/nvidia
# (hls.substrate.docker.nvidia), and the gate puts /usr/local/nvidia/lib64 last on
# LD_LIBRARY_PATH, after the image's own directories, with the image's CUDA forward-compatibility
# libraries (/usr/local/cuda/compat) between them when its one libcuda.so.X.Y is of a newer major
# version than the driver's, the order the NVIDIA Container Toolkit's ld cache gave them; and
# /usr/local/nvidia/bin last on PATH. It then reads
# /.hls-gate/env, which blocks until the launcher has checked the
# container's grant and opened the FIFO to write the role's variables into it, and exports them.
# They are NAME=value lines, each value quoted for the shell and free of control characters, so
# sourcing them sets exactly those values, and none is in the container's configuration or its
# argv. With --marker it prints
# LINE on stderr and, only if that write succeeded, becomes COMMAND, so a nonzero exit with no
# LINE in the role's stderr is a failure before the command started. The gate's variables of
# its own are used before it sets the role's, so no variable of the role's can change what it
# does; the role's own LD_LIBRARY_PATH or PATH, if it sets one, replaces the gate's.

nvidia=
if [ "${1-}" = --nvidia ]; then
  [ "$#" -ge 2 ] || exit 2
  case $2 in ''|.*|*[!0-9.]*) exit 2 ;; esac
  nvidia=$2
  shift 2
fi
score=
if [ "${1-}" = --oom-score-adj ]; then
  [ "$#" -ge 2 ] || exit 2
  case $2 in ''|*[!0-9]*) exit 2 ;; esac
  score=$2
  shift 2
fi
case ${1-} in
  --) [ "$#" -ge 2 ] || exit 2 ;;
  --marker) [ "$#" -ge 4 ] && [ "$3" = -- ] || exit 2 ;;
  *) exit 2 ;;
esac
exec >/.hls-gate/out 2>/.hls-gate/err
if [ -n "$score" ]; then
  printf '%s\n' "$score" >/proc/self/oom_score_adj || exit 2
fi
if [ -n "$nvidia" ]; then
  lib=
  set -f
  IFS=:
  for d in ${LD_LIBRARY_PATH-}; do
    case $d in ''|/usr/local/nvidia/lib64) ;; *) lib=$lib$d: ;; esac
  done
  unset IFS
  set +f
  compat=
  for f in /usr/local/cuda/compat/libcuda.so.*.*; do
    [ -e "$f" ] || continue
    [ -z "$compat" ] || { compat=several; break; }
    compat=${f##*/libcuda.so.}
  done
  major=${compat%%.*}
  case $major in
    ''|*[!0-9]*) ;;
    *) if [ "$major" -gt "${nvidia%%.*}" ]; then lib=$lib/usr/local/cuda/compat:; fi ;;
  esac
  LD_LIBRARY_PATH=${lib}/usr/local/nvidia/lib64
  case :${PATH-}: in
    *:/usr/local/nvidia/bin:*) ;;
    *) PATH=${PATH:+$PATH:}/usr/local/nvidia/bin ;;
  esac
  export LD_LIBRARY_PATH PATH
fi
set -a
. /.hls-gate/env
set +a
if [ "$1" = -- ]; then
  shift
  exec "$@"
fi
printf '%s\n' "$2" >&2 && shift 3 && exec "$@"
