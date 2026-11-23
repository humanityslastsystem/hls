#!/usr/bin/env bash
# The gateway of a sealed role on Docker: the entrypoint of the container that owns the role's
# network namespace (hls.substrate.docker.seal). The payload joins the namespace with no
# capability and a PID namespace of its own, so it can neither see nor signal the interceptor,
# nor rewrite the firewall, nor set the interceptor's mark.
#
# Started as root with CAP_NET_ADMIN, CAP_SETUID and CAP_SETGID alone and no-new-privileges, as
#   bash -c "<seal_rules.sh><gateway.sh>" hls-gateway <tport> <iport> <fwmark> <python> \
#       <policy_json> <served_port> <uid> <gid>
# (hls.substrate.docker.seal.gateway_argv). <iport> is empty for a seal without an index, and
# <served_port> for one that serves nothing. The gateway:
# 1. installs the seal's firewall (seal_rules, which seal_rules.sh at the head of this program
#    defines) and an INPUT chain that takes only loopback, the replies to the namespace's own
#    connections and the served port from the bridge's gateway, the host, where Docker's
#    publish forwards from, so no other container on Docker's bridge reaches the namespace;
# 2. becomes <uid>:<gid> keeping CAP_NET_ADMIN alone, which the interceptor's mark needs, so
#    nothing runs as root after the firewall and every file it writes is the host user's;
# 3. starts the interceptor (python -m hls.egress.proxy), waits until both its listeners take
#    connections and writes "listening" over HLS_GATEWAY_LISTENING_FILE, which the host reads
#    before it creates the payload;
# 4. lives as long as the interceptor, so a gateway that runs has an interceptor that runs.
# A gateway that cannot seal exits nonzero before it writes the listening file, and the host
# never creates the payload. Its messages and the interceptor's stderr go to
# HLS_GATEWAY_LOG_FILE through the harness's capper (hls.substrate.log_cap), keeping
# HLS_GATEWAY_LOG_KEEP bytes at each end; the root part's messages reach it through the
# unprivileged part, since root without CAP_DAC_OVERRIDE cannot write the host user's file.
# HLS_PROXY_STATUS_FILE is the interceptor's status file, HLS_SEAL_DNS=closed refuses the
# payload's DNS as the pasta seal does, and HLS_UPSTREAM_PROXY_URL, when set, is the site's
# upstream proxy, which the interceptor reads itself.
set -euo pipefail

export HLS_PROXY_TRANSPARENT_PORT="$1"
if [ -n "$2" ]; then export HLS_PROXY_PORT="$2"; fi
export HLS_PROXY_FWMARK="$3"
_python="$4"
export HLS_EGRESS_POLICY="$5"
_served="$6"
_uid="$7"
_gid="$8"
export HLS_PROXY_HOST=127.0.0.1
# The host user, never root, which the builder has checked too: a gateway given anything else
# ends here, before it does anything, as root cannot write the log to say why.
case "$_uid:$_gid" in
  :*|*:|*[!0-9:]*) exit 2 ;;
esac
if [ "$_uid" -eq 0 ] || [ "$_gid" -eq 0 ]; then exit 2; fi

# The root part: the firewall. It runs in a subshell of its own with errexit, so the first
# failure ends it; its status and its messages are handed to the unprivileged part.
set +e
_setup="$(
  set -e
  exec 2>&1 > /dev/null
  if ! declare -F seal_rules > /dev/null; then
    echo "FATAL: the gateway program does not start with seal_rules.sh" >&2
    exit 2
  fi
  case "${HLS_SEAL_DNS:-}" in
    "") ;;
    closed) export HLS_PROXY_OWN_DNS=1 ;;
    *) echo "FATAL: HLS_SEAL_DNS is closed or unset, got ${HLS_SEAL_DNS}" >&2; exit 2 ;;
  esac
  for _tool in iptables ip6tables; do
    if [ ! -x "/usr/sbin/$_tool" ]; then
      echo "FATAL: /usr/sbin/$_tool not found; the gateway needs it to set up the firewall" >&2
      exit 2
    fi
  done
  export HLS_SEAL_IPTABLES=/usr/sbin/iptables HLS_SEAL_IP6TABLES=/usr/sbin/ip6tables
  seal_rules
  # INPUT: loopback (the listeners of the interceptor and of the payload), the replies to the
  # connections of the namespace and their ICMP errors, and the served port, which Docker
  # publishes on the host; everything else is dropped, IPv6 whole.
  "$HLS_SEAL_IPTABLES" -A INPUT -i lo -j ACCEPT
  "$HLS_SEAL_IPTABLES" -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
  if [ -n "$_served" ]; then
    # The served port is taken from the host alone: Docker's publish reaches the namespace from
    # the bridge's gateway, the default route's (/proc/net/route, hex, lowest byte first).
    _host=""
    {
      read -r _
      while read -r _iface _dest _gw _rest; do
        if [ "$_dest" = 00000000 ] && [ "$_gw" != 00000000 ]; then
          _host="$(printf '%d.%d.%d.%d' "0x${_gw:6:2}" "0x${_gw:4:2}" "0x${_gw:2:2}" "0x${_gw:0:2}")"
          break
        fi
      done
    } < /proc/net/route
    if [ -z "$_host" ]; then
      echo "FATAL: the gateway's namespace has no default route, so the host it serves cannot be told" >&2
      exit 2
    fi
    "$HLS_SEAL_IPTABLES" -A INPUT -s "$_host" -p tcp --dport "$_served" -j ACCEPT
  fi
  "$HLS_SEAL_IPTABLES" -A INPUT -j DROP
  if "$HLS_SEAL_IP6TABLES" -A INPUT -i lo -j ACCEPT \
     && "$HLS_SEAL_IP6TABLES" -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT \
     && "$HLS_SEAL_IP6TABLES" -A INPUT -j DROP; then :;
  elif [ "$(cat /proc/sys/net/ipv6/conf/all/disable_ipv6 2>/dev/null || echo 1)" = 1 ]; then :;
  else echo "FATAL: ip6tables INPUT DROP unavailable but IPv6 is enabled; refusing to run unsealed" >&2; exit 1; fi
)"
_status=$?
set -e
if [ "${HLS_SEAL_DNS:-}" = closed ]; then export HLS_PROXY_OWN_DNS=1; fi

exec setpriv --reuid "$_uid" --regid "$_gid" --clear-groups --inh-caps -all,+net_admin \
  --ambient-caps -all,+net_admin -- bash -c '
set -euo pipefail
_status="$1" _setup="$2" _python="$3"
exec > /dev/null 2> >(exec "$_python" -m hls.substrate.log_cap --keep "$HLS_GATEWAY_LOG_KEEP" "$HLS_GATEWAY_LOG_FILE")
_capper=$!
# When this shell ends, the gateway ends with every process in it, so the capper is first given
# the end of its pipe and waited for, and writes what it holds.
_end() {
  exec 2>&-
  wait "$_capper" || true
  exit "$1"
}
if [ -n "$_setup" ]; then printf "%s\n" "$_setup" >&2; fi
if [ "$_status" != 0 ]; then
  echo "FATAL: the gateway could not install its firewall (status $_status); its interceptor never started" >&2
  _end "$_status"
fi
"$_python" -m hls.egress.proxy &
_proxy=$!
# Both listeners take connections (loopback is exempt from the redirect) before the host may
# create the payload, so the payload never runs against a half-made seal. 60 s, as the pasta
# seal waits: cold imports on a loaded host take seconds; an interceptor that dies ends the
# wait at once.
ok=0
_waited=0
for _ in $(seq 1 600); do
  if : </dev/tcp/127.0.0.1/"$HLS_PROXY_TRANSPARENT_PORT" \
     && { [ -z "${HLS_PROXY_PORT:-}" ] || : </dev/tcp/127.0.0.1/"$HLS_PROXY_PORT"; }; then ok=1; break; fi
  [ -d "/proc/$_proxy" ] || break
  _waited=$((_waited + 1))
  sleep 0.1
done 2>/dev/null
if [ "$ok" != 1 ]; then
  if [ -d "/proc/$_proxy" ]; then _why="still running but never bound"; else _why="exited"; fi
  echo "FATAL: egress interceptor did not become ready after $((_waited / 10))s; proxy $_why" >&2
  kill "$_proxy" 2>/dev/null || true
  wait "$_proxy" 2>/dev/null || true
  _end 3
fi
printf "listening\n" 1<> "$HLS_GATEWAY_LISTENING_FILE"
rc=0
wait "$_proxy" || rc=$?
echo "the egress interceptor exited with status $rc" >&2
_end "$rc"
' hls-gateway "$_status" "$_setup" "$_python"
