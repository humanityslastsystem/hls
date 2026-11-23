#!/usr/bin/env bash
# HLS rootless transparent egress seal (one network namespace, pasta-backed).
#
# Seals a container role in ONE network namespace, which pasta makes. We run the
# hls.egress.proxy transparent interceptor inside that namespace but outside the container:
# iptables REDIRECTs all the payload's TCP into it, it recovers the original destination
# (SO_ORIGINAL_DST), gates by SNI/domain, and relays -- the client needs NO proxy env. Isolation:
# the in-netns proxy keeps CAP_NET_ADMIN to SO_MARK its own upstream (so it is not redirected
# into itself; the mark is unforgeable without the cap); the apptainer payload is launched with
# --pid (cannot see/kill/ptrace the proxy) + --drop-caps CAP_NET_ADMIN,CAP_NET_RAW,CAP_SYS_RESOURCE
# (cannot rewrite the firewall, forge the mark or raise the limits the seal wrote), or --drop-caps
# ALL for a contestant payload with nothing of the harness inside it.
#
# Networking is provided by `pasta` (passt), NOT slirp4netns: pasta CREATES + configures the
# netns and spawns the command in it (copying the host's address/routes/resolvers), and -- for
# the runtime EVAL seal -- forwards an inbound host port into it. pasta's port forwarding is the
# host kernel's TCP (splice), so it sustains real serving concurrency, unlike slirp4netns's
# hostfwd which caps at ~8 concurrent connections and silently resets the rest. This script is
# the one place pasta's command line is built, and it names every forward direction (see
# "pasta's forwards" below): the build seal forwards no port either way, the runtime seal one
# TCP port in, and neither lets ICMP out nor anything of the netns reach the host's loopback.
#
# Serves BOTH phases: the producer BUILD seal (wraps `apptainer exec`, with a pull-through pip
# index) and the runtime EVAL seal (wraps `apptainer run` for a served artifact -- no index, plus
# an ingress forward so the artifact stays reachable at <node>:<port> despite the netns).
#
# Invoked as the seal command PREFIX prepended to the apptainer argv (built by
# hls.substrate.apptainer.egress.seal_prefix):
#   bash -c "$(cat transparent_seal.sh)" hls-seal <tport> <iport> <fwmark> <python> \
#       <policy_json> <ingress_port> <apptainer> exec|run ...
# i.e. $0=hls-seal, $1..$6 the params below, and everything after them ("$@" post-shift) is the
# real command to run sealed. <iport> may be EMPTY (runtime: transparent-only, no index);
# <ingress_port> may be EMPTY (build: no port forwarded). Params are EXPORTED so the pasta-
# spawned child (and the proxy it spawns) inherit them -- no interpolation in the single-quoted
# child block (which therefore must contain NO apostrophe). HLS_PASTA points at the pasta binary
# (a compute node may lack the passt package; the substrate stages a built copy).
#
# Host-side files, named by the caller's environment (all optional):
# - HLS_SEAL_STARTED_FILE: created by the in-netns block just before the payload starts. An
#   attempt that leaves it behind ran the payload, so its exit status is final and never
#   retried, whatever it is and however fast it came. Unset, a private temporary path is used.
# - HLS_INGRESS_PORT_FILE (runtime seal): the host port pasta forwards, written whole (temp
#   file then rename) each time a port is chosen. A port is chosen once, and again only after
#   a host-port race that stopped pasta before the payload started; so once the started file
#   exists the port file holds the port of the attempt that runs. A reader waits for the
#   started file, then reads the port (hls.substrate.seal.ingress_port).
# - HLS_PROXY_STATUS_FILE: inherited by the interceptor, which keeps its upstream-error count
#   there, made at full size when it starts and rewritten in place (hls.egress.proxy). It is a
#   host path no container sees.
# - HLS_SEAL_ENDED_FILE: the payload own ending, {"exit": N, "interceptor": true|false}, the
#   second saying whether the interceptor was still running when the payload ended. Before each
#   attempt both "<file>.pending" and the file itself are made at full size, all spaces, which
#   read as no ending. The block writes the ending over "<file>.pending" right before it exits,
#   and the loop outside copies that over the file itself only when pasta returns after that: a
#   pasta or block that ended before the payload leaves no ending
#   (hls.substrate.seal.payload_ending). Both writes go over bytes the files hold,
#   so a payload that fills the disk they are on cannot make them fail.
# HLS_SEAL_DNS=closed (optional) refuses the payload DNS: a build seal that reaches no host but
# through its index, whose payload never needs a name resolved, so a build hook cannot carry data
# out in the names it looks up. The interceptor then resolves names itself, over marked sockets
# (HLS_PROXY_OWN_DNS), the only DNS the firewall lets out.
# HLS_SEAL_ENV_FD=3 (optional) says the payload reads its variables from descriptor 3, the pipe
# the step wrapper hands over (hls.substrate.role_exec). pasta closes every descriptor it
# inherits above stderr, so the pipe crosses pasta as standard input instead, and the
# in-namespace block gives it to the payload alone, on 3. Nothing else the seal runs holds it,
# and nothing reads it before the payload does, so an attempt retried before the payload
# started leaves it whole.
# HLS_SEAL_REFUSED_EXIT and HLS_SEAL_UNBOUND_EXIT (required) are the statuses the seal exits with on
# its own: when it refuses to run its payload, and when its interceptor never bound. Their one owner
# is hls.substrate.exits; the builder sets them in front of this script
# (hls.substrate.apptainer.egress.seal_script), and they are numbers no layer around the seal and
# no harness program beneath it uses, so a status alone says the seal ended the role.
# HLS_SEAL_RULES (required) is the text of hls/substrate/seal_rules.sh, the firewall the Docker
# gateway installs too, which the builder also sets in front of this script. It heads the
# in-namespace block's program, which calls the seal_rules it defines; it is not exported.
# Each attempt starts with the line "hls-seal: attempt <n>" on stderr (egress.ATTEMPT_LINE), so a
# reader can tell the lines of the attempt that failed from those of the attempts before it.
set -euo pipefail
: "${HLS_SEAL_REFUSED_EXIT:?is set by the seal builder, hls.substrate.apptainer.egress.seal_script}"
: "${HLS_SEAL_UNBOUND_EXIT:?is set by the seal builder, hls.substrate.apptainer.egress.seal_script}"
: "${HLS_SEAL_RULES:?is set by the seal builder, hls.substrate.apptainer.egress.seal_script}"
export HLS_SEAL_REFUSED_EXIT HLS_SEAL_UNBOUND_EXIT

# iptables, ip6tables and sysctl are admin tools in the system sbin dirs, which a role step's PATH
# need not carry. We do NOT change the payload's PATH to add them (a different PATH could change which
# tools a build finds): instead we resolve each to an absolute path from the sbin dirs, or
# HLS_SEAL_TOOL_PATH (which only the runner's environment sets), and invoke only those. A tool missing
# there fails the seal here at setup, before the payload runs, as an infrastructure error naming it (no
# `refused:` line, so it is never read as the payload's). The resolved paths are exported so the
# pasta-spawned child that sets up the netns uses them.
_find_tool() {
  _dirs="${HLS_SEAL_TOOL_PATH:-/usr/sbin:/sbin}"
  _IFS_SAVED="$IFS"; IFS=:
  for _d in $_dirs; do
    if [ -x "$_d/$1" ]; then IFS="$_IFS_SAVED"; printf '%s\n' "$_d/$1"; return 0; fi
  done
  IFS="$_IFS_SAVED"
  echo "hls-seal: $1 not found in $_dirs; the seal needs it to set up the firewall" >&2
  return 1
}
export HLS_SEAL_IPTABLES HLS_SEAL_IP6TABLES HLS_SEAL_SYSCTL
HLS_SEAL_IPTABLES="$(_find_tool iptables)"
HLS_SEAL_IP6TABLES="$(_find_tool ip6tables)"
HLS_SEAL_SYSCTL="$(_find_tool sysctl)"

export HLS_PROXY_TRANSPARENT_PORT="$1"   # REDIRECT target / transparent listener
if [ -n "$2" ]; then export HLS_PROXY_PORT="$2"; fi  # pull-through index; EMPTY -> transparent-only
export HLS_PROXY_FWMARK="$3"             # SO_MARK exempting the proxy's own upstream
export HLS_SEAL_PYTHON="$4"               # interpreter that runs `-m hls.egress.proxy`
export HLS_EGRESS_POLICY="$5"             # SealPolicy JSON the interceptor enforces
_ingress="$6"                             # serving port to host-forward (runtime); EMPTY -> none
export HLS_PROXY_HOST=127.0.0.1
shift 6   # "$@" is now the real command (apptainer exec|run ...)

if [ -z "${HLS_SEAL_STARTED_FILE:-}" ]; then
  _seal_tmp="$(mktemp -d)"
  trap 'rm -rf "$_seal_tmp"' EXIT
  HLS_SEAL_STARTED_FILE="$_seal_tmp/started"
fi
export HLS_SEAL_STARTED_FILE

# Where pasta takes its standard input from: the env file pipe on 3, or the seal own input.
case "${HLS_SEAL_ENV_FD:-}" in
  "") _env_in=0 ;;
  3) _env_in=3 ;;
  *) echo "FATAL: the seal hands over only descriptor 3, got HLS_SEAL_ENV_FD=${HLS_SEAL_ENV_FD}" >&2; exit 2 ;;
esac
_ended="${HLS_SEAL_ENDED_FILE:-}"
# Bytes of each ending record, made before the payload starts: far more than its one line of JSON.
_ENDED_BYTES=512
case "${HLS_SEAL_DNS:-}" in
  "") ;;
  closed) export HLS_PROXY_OWN_DNS=1 ;;
  *) echo "FATAL: HLS_SEAL_DNS is closed or unset, got ${HLS_SEAL_DNS}" >&2; exit 2 ;;
esac

_pasta="${HLS_PASTA:-pasta}"
# pasta's forwards. A forward direction pasta's command line does not name defaults to `auto`
# (in the passt release hls.substrate.pasta pins: conf.c, and passt(1) on -t, -u, -T and -U).
# auto forwards, from every address of the near side, each port bound on the far side that is
# outside the host's ephemeral range (ip_local_port_range) and that pasta may bind unprivileged
# (from ip_unprivileged_port_start), rescanned each second from /proc/net; for UDP, also the
# port of every TCP port bound there. Left unnamed, the inbound directions let any host that
# reaches the node connect to whatever a payload listens on at such a port, over TCP and over
# UDP, and push data in or pull it out. So every attempt names all four, and each seal forwards
# what its role needs and nothing else:
# - into the netns over TCP (-t): the runtime seal forwards one port, below; the build seal none.
# - into the netns over UDP (-u): none, for both seals. A served artifact answers over TCP.
# - out of the netns to the host's loopback over TCP (-T) and UDP (-U): none, for both seals.
# --no-map-gw: pasta otherwise maps the netns's default gateway address to the host's loopback,
# the other way to the host's own services. With -T, -U, this and the ICMP rules below, nothing
# of the netns reaches the host's loopback; egress still reaches the world via pasta for the
# interceptor's filtered upstream. The seal needs no host loopback service: its DNS goes to the
# resolv.conf nameservers, which the netns reads from the host's own file (pasta makes no mount
# namespace), so a loopback resolver there is the netns's own loopback, which the mapping never
# reached.
# --no-icmp: pasta otherwise relays every ICMP echo the netns sends, data and all, to any address
# and its reply back. The firewall below drops the netns's ICMP too. The netns's ping range stays
# as pasta writes it, "0 0", open to the payload's group: pasta's user namespace maps gid 0
# alone, and the kernel refuses a range whose bounds it cannot map, so the seal cannot empty it.
# The payload, without CAP_NET_ADMIN, cannot change it either.
# The runtime seal's one forward is bound on 0.0.0.0 so it is reachable at <node>:<port> from the
# orchestrator (readiness) and the driver AND at 127.0.0.1 for a same-node client -- regardless
# of how the node hostname resolves. An explicit address (vs `-t <port>`) avoids pasta's rootless
# loopback-only fallback, and a wildcard bind needs no SO_BINDTODEVICE. $_ingress is the GUEST
# port the served artifact binds inside the netns. The HOST side may already be bound by another
# job on a shared node, and a rejected pasta forward is FATAL -- so pick a FREE host port (the OS
# picks one via bind(:0)) and map host:<free> -> guest:<_ingress>, recording the chosen port in
# $HLS_INGRESS_PORT_FILE so the orchestrator builds the URL.
# pasta --config-net configures the spawned netns (lo up; host address/routes/resolvers copied)
# BEFORE running the child, so the child needs no readiness barrier,
# no uid_map poll, and no api-socket dance. -f keeps pasta in the foreground so its lifetime IS
# the payload's (harbor tracks this prefix).
#
# TOCTOU retry (runtime seal only): between the Python bind(:0)/close() and pasta binding the
# same port, another job on the shared node can steal it. A pasta forward that cannot bind is
# FATAL (pasta exits immediately, before the in-netns child runs). We detect this by timing the
# pasta run: a port-race failure is FAST (pasta rejects the forward and exits within ~2-3 s,
# before any payload work happens); a real payload failure is SLOW (the child set up the seal and
# ran to completion). We retry ONLY on a fast non-zero exit of the RUNTIME seal (non-empty
# $_ingress), re-scanning a fresh free port each attempt. The BUILD seal must NEVER retry
# (no forward, no port to race on, and a fast nonzero exit is a genuine error there).
# Before each attempt: clear what an earlier one left, and scan a free host port for the
# runtime seal, on the first attempt and again only after a host-port race emptied $_hport
# below. A readiness retry keeps the port it already wrote. It runs with descriptor 3 closed,
# so none of it holds the env file pipe.
_prepare() {
  rm -f "$HLS_SEAL_STARTED_FILE"
  if [ -n "$_ended" ]; then
    # Both records are made at full size now, before the payload can fill the disk, all spaces,
    # which read as no ending; the ending is then written over them in place, which needs no
    # new block.
    rm -f "$_ended" "$_ended.pending"
    printf '%*s' "$_ENDED_BYTES" '' > "$_ended.pending"
    printf '%*s' "$_ENDED_BYTES" '' > "$_ended"
  fi
  if [ -n "$_ingress" ] && [ -z "$_hport" ]; then
    _hport="$("$HLS_SEAL_PYTHON" -c 'import socket
s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
    if [ -n "${HLS_INGRESS_PORT_FILE:-}" ]; then
      printf '%s\n' "$_hport" > "$HLS_INGRESS_PORT_FILE.tmp.$$"
      mv -f "$HLS_INGRESS_PORT_FILE.tmp.$$" "$HLS_INGRESS_PORT_FILE"
    fi
  fi
}
rc=0
_attempts=0
_hport=""
while true; do
  _prepare 3<&-
  # The inbound TCP forward of this attempt (see "pasta's forwards" above): the runtime seal's
  # host port to its served port, none for the build seal.
  if [ -n "$_ingress" ]; then _tcp_in="0.0.0.0/${_hport}:${_ingress}"; else _tcp_in=none; fi
  echo "hls-seal: attempt $(( _attempts + 1 ))" >&2
  _start=$SECONDS
  rc=0
  "$_pasta" --config-net -f -t "$_tcp_in" -u none -T none -U none --no-map-gw --no-icmp -- bash -c "$HLS_SEAL_RULES"'
  set -euo pipefail
  # The payload env file came through pasta on standard input. It waits on descriptor 9 for
  # the payload, the only process here that gets it, and nothing reads standard input.
  if [ "${HLS_SEAL_ENV_FD:-}" = 3 ]; then exec 9<&0 0</dev/null; fi
  # The start time of process $1 while it runs, which a process that reused its pid does not
  # share; nothing once it has ended.
  _running_since() {
    local s f
    s="$(cat "/proc/$1/stat" 2>/dev/null)" || return 0
    s="${s##*) }"
    read -r -a f <<< "$s"
    case "${f[0]}" in Z|X|x) return 0 ;; esac
    printf "%s\n" "${f[19]}"
  }
  # Everything up to the payload runs without descriptor 9.
  {
  # Disable IPv6 in the netns. The interceptor is IPv4-only (SO_ORIGINAL_DST), and a host with
  # AAAA records (e.g. a CDN-fronted API) would make a client try IPv6 first
  # and HANG on a dropped SYN. Disabling v6 makes those connects fail FAST so the client falls
  # back to the intercepted IPv4 path; the ip6tables DROP below is a backstop.
  for _c in all default lo; do echo 1 > "/proc/sys/net/ipv6/conf/$_c/disable_ipv6" 2>/dev/null || true; done
  # The role limits on the uid counts the kernel keeps per user namespace (hls.substrate.ucounts):
  # HLS_SEAL_UCOUNTS holds NAME=LIMIT pairs, each NAME a file of /proc/sys/user. Each is written in
  # this user namespace, which holds everything of the role, and read back, before the payload
  # starts. Every pair is checked first: only a NAME of max_ and lower case letters and
  # underscores, and a LIMIT of digits, so nothing but those files is ever written. A malformed
  # pair, or a limit that does not read back as written, fails the seal here, before the payload,
  # with HLS_SEAL_REFUSED_EXIT, the seal own status for a refusal that no other attempt could change.
  read -r -a _ucounts <<< "${HLS_SEAL_UCOUNTS:-}"
  for _pair in "${_ucounts[@]+"${_ucounts[@]}"}"; do
    _name="${_pair%%=*}"
    _limit="${_pair#*=}"
    case "$_pair" in *=*) ;; *) _name="" ;; esac
    case "$_name" in max_*) ;; *) _name="" ;; esac
    case "$_name" in *[!a-z_]*) _name="" ;; esac
    case "$_limit" in ""|*[!0-9]*) _name="" ;; esac
    if [ -z "$_name" ]; then
      echo "FATAL: HLS_SEAL_UCOUNTS holds $_pair, not NAME=LIMIT for a file of /proc/sys/user" >&2
      exit "$HLS_SEAL_REFUSED_EXIT"
    fi
  done
  for _pair in "${_ucounts[@]+"${_ucounts[@]}"}"; do
    _name="${_pair%%=*}"
    _limit="${_pair#*=}"
    if ! "$HLS_SEAL_SYSCTL" -q -w "user.$_name=$_limit"; then
      echo "FATAL: the seal could not set user.$_name to $_limit; refusing to run the role uncapped" >&2
      exit "$HLS_SEAL_REFUSED_EXIT"
    fi
    _read="$("$HLS_SEAL_SYSCTL" -n "user.$_name")"
    if [ "$_read" != "$_limit" ]; then
      echo "FATAL: user.$_name reads $_read, not the $_limit the seal set; refusing to run the role uncapped" >&2
      exit "$HLS_SEAL_REFUSED_EXIT"
    fi
  done
  # The firewall: seal_rules, which the rule file at the head of this program defines
  # (hls/substrate/seal_rules.sh, given as HLS_SEAL_RULES).
  seal_rules
  # In-netns transparent interceptor (keeps CAP_NET_ADMIN to SO_MARK its upstream). It resolves
  # upstream names via the netns resolv.conf (the host resolvers pasta copied -- permitted by the
  # UDP rule above), so no private resolver bind is needed.
  "$HLS_SEAL_PYTHON" -m hls.egress.proxy &
  _proxy=$!
  _proxy_since="$(_running_since "$_proxy")"
  # Readiness: block until BOTH listeners bind (loopback is REDIRECT-exempt). Fail closed so the
  # payload never runs against a half-installed seal.
  #
  # 60s: cold Python imports from a network filesystem on a loaded node can take more than
  # 10s. A proxy that dies still fails immediately via the /proc check, so the budget costs
  # nothing on a real failure. The message reports the wait and whether the proxy is still up,
  # because "did not become ready" alone cannot tell slow from dead.
  ok=0
  _waited=0
  for _ in $(seq 1 600); do
    if : </dev/tcp/127.0.0.1/"$HLS_PROXY_TRANSPARENT_PORT" \
       && { [ -z "${HLS_PROXY_PORT:-}" ] || : </dev/tcp/127.0.0.1/"$HLS_PROXY_PORT"; }; then ok=1; break; fi
    [ -d "/proc/$_proxy" ] || break
    _waited=$((_waited + 1))
    sleep 0.1
  done 2>/dev/null
  # HLS_SEAL_UNBOUND_EXIT, the seal own status for this case, so the loop below retries it: the
  # payload has provably not started, whatever the clock says.
  if [ "$ok" != 1 ]; then
    if [ -d "/proc/$_proxy" ]; then _why="still running but never bound"; else _why="exited"; fi
    echo "FATAL: egress interceptor did not become ready after $((_waited / 10))s; proxy $_why" >&2
    exit "$HLS_SEAL_UNBOUND_EXIT"
  fi
  } 9<&-
  # Run the payload in the FOREGROUND, then reap the proxy (the supervisor shell must survive
  # the payload, so we do not exec). The started file says, to the loop outside, that from here
  # on the exit status is the payload own.
  : > "$HLS_SEAL_STARTED_FILE"
  prc=0
  if [ "${HLS_SEAL_ENV_FD:-}" = 3 ]; then
    "$@" 3<&9 9<&- || prc=$?
    exec 9<&-
  else
    "$@" || prc=$?
  fi
  _alive=false
  if [ -n "$_proxy_since" ] && [ "$(_running_since "$_proxy")" = "$_proxy_since" ]; then _alive=true; fi
  kill "$_proxy" 2>/dev/null || true
  if [ -n "${HLS_SEAL_ENDED_FILE:-}" ]; then
    # Over the start of the record made before the payload, in place: no new block is needed.
    printf "{\"exit\": %d, \"interceptor\": %s}\n" "$prc" "$_alive" 1<> "$HLS_SEAL_ENDED_FILE.pending"
  fi
  exit "$prc"
' _seal "$@" <&"$_env_in" 3<&- || rc=$?
  _elapsed=$(( SECONDS - _start ))
  _attempts=$(( _attempts + 1 ))
  # The payload ran: its exit status is the result. Never retried, so a payload that exits with one
  # of the seal own statuses, or fails within seconds, runs once. The block records the payload
  # ending right before it exits, so a record missing when pasta returns means pasta or the block
  # ended first, and one written later is never taken as the ending.
  if [ -e "$HLS_SEAL_STARTED_FILE" ]; then
    # The block record, all of it, over the record kept under the name itself, in place.
    if [ -n "$_ended" ]; then cat "$_ended.pending" 1<> "$_ended" 3<&- || true; fi
    break
  fi
  # HLS_SEAL_UNBOUND_EXIT is the seal own "the interceptor never bound" status. The payload never
  # started, so retrying commits nothing twice; it applies to the build seal too. The port
  # stays: a reader may already hold it.
  if [ "$rc" -eq "$HLS_SEAL_UNBOUND_EXIT" ] && [ "$_attempts" -lt 5 ]; then
    echo "seal: interceptor did not bind within the readiness budget (attempt $_attempts) -- retrying" >&2
    continue
  fi
  # HLS_SEAL_REFUSED_EXIT is the seal own refusal to run the payload uncapped: every attempt would
  # end the same way, so it is final, also for the runtime seal, however fast it came.
  if [ "$rc" -eq "$HLS_SEAL_REFUSED_EXIT" ]; then
    break
  fi
  # A host-port race: the runtime seal, pasta failed fast (< 5 s), and the payload never
  # started. Only a different port can help, so scan again.
  if [ -n "$_ingress" ] && [ "$rc" -ne 0 ] && [ "$_elapsed" -lt 5 ] && [ "$_attempts" -lt 5 ]; then
    echo "seal: pasta exited fast (rc=$rc, elapsed=${_elapsed}s) -- likely a host-port race; rescanning + retrying (attempt $_attempts)" >&2
    _hport=""
    continue
  fi
  break
done
exit "$rc"
