# The firewall of a sealed role's network namespace, one text for both seals: the pasta seal
# (hls/substrate/apptainer/transparent_seal.sh) installs it in the namespace pasta makes, the Docker
# gateway (hls/substrate/docker/gateway.sh) in the gateway container's. Both put this text ahead of
# the program that runs in the namespace (hls.substrate.seal.SEAL_RULES), which then calls
# seal_rules once, as root of the namespace, before the interceptor starts.
#
# seal_rules reads HLS_SEAL_IPTABLES and HLS_SEAL_IP6TABLES (absolute paths of the tools),
# HLS_PROXY_FWMARK (the mark of the interceptor's own sockets), HLS_PROXY_TRANSPARENT_PORT (its
# transparent listener), HLS_PROXY_OWN_DNS (set for a seal that refuses the payload's DNS) and the
# namespace's /etc/resolv.conf. A rule that fails ends the shell under set -e; so does IPv6 left
# enabled with no ip6tables to drop it, with status 1.
seal_rules() {
  local m="$HLS_PROXY_FWMARK" _kw _ns
  # Transparent REDIRECT recipe (nat): all TCP -> the interceptor, EXCEPT the proxy own marked
  # upstream and loopback (the index + the readiness probes).
  "$HLS_SEAL_IPTABLES" -t nat -A OUTPUT -m mark --mark "$m" -j RETURN
  "$HLS_SEAL_IPTABLES" -t nat -A OUTPUT -o lo -j RETURN
  "$HLS_SEAL_IPTABLES" -t nat -A OUTPUT -d 127.0.0.0/8 -j RETURN
  "$HLS_SEAL_IPTABLES" -t nat -A OUTPUT -p tcp -j REDIRECT --to-ports "$HLS_PROXY_TRANSPARENT_PORT"
  # UDP (filter): the proxy own marked UDP (its resolver lookups) + loopback + DNS:53 to the
  # nameservers of the namespace's resolv.conf ONLY; drop everything else so a client cannot
  # tunnel over QUIC/HTTP3 or arbitrary UDP. (pasta copies the host resolvers and intercepts
  # them; Docker's default bridge writes the host's upstream resolvers there.)
  "$HLS_SEAL_IPTABLES" -A OUTPUT -m mark --mark "$m" -j ACCEPT
  if [ -n "${HLS_PROXY_OWN_DNS:-}" ]; then
    # A closed seal: every DNS query but the interceptor own marked ones is dropped, to a
    # loopback resolver too, so the rule comes before the loopback allowance.
    "$HLS_SEAL_IPTABLES" -A OUTPUT -p udp --dport 53 -j DROP
    "$HLS_SEAL_IPTABLES" -A OUTPUT -o lo -j ACCEPT
  else
    "$HLS_SEAL_IPTABLES" -A OUTPUT -o lo -j ACCEPT
    while read -r _kw _ns _; do
      [ "$_kw" = nameserver ] || continue
      [ -n "$_ns" ] || continue
      # IPv4 resolvers only: v4 iptables rejects a v6 -d address (exit 2), and as the final
      # command of the loop body it would trip set -e and ABORT the seal here -- before the UDP
      # DROP backstop and the interceptor start. A v6 resolver is unreachable anyway: the
      # ip6tables DROP below covers it. || true so a malformed v4 line can never abort the seal
      # either.
      case "$_ns" in *:*) continue ;; esac
      "$HLS_SEAL_IPTABLES" -A OUTPUT -p udp -d "$_ns" --dport 53 -j ACCEPT || true
    done < /etc/resolv.conf
  fi
  "$HLS_SEAL_IPTABLES" -A OUTPUT -p udp -j DROP
  # ICMP (filter): none leaves but over loopback. Ping sockets are open to the payload's group
  # (pasta opens them to group 0, Docker to every group), so with no CAP_NET_RAW it can still
  # send an echo, data and all; this refuses the send.
  "$HLS_SEAL_IPTABLES" -A OUTPUT -p icmp -j DROP
  # IPv6 cannot be SO_ORIGINAL_DST-recovered, so fail closed: DROP all v6 (exempt the proxy
  # marked upstream + loopback), ICMPv6 with it. If v6 is enabled but we cannot seal it, refuse.
  if "$HLS_SEAL_IP6TABLES" -A OUTPUT -m mark --mark "$m" -j RETURN 2>/dev/null \
     && "$HLS_SEAL_IP6TABLES" -A OUTPUT -o lo -j RETURN 2>/dev/null \
     && "$HLS_SEAL_IP6TABLES" -A OUTPUT -j DROP 2>/dev/null; then :;
  elif [ "$(cat /proc/sys/net/ipv6/conf/all/disable_ipv6 2>/dev/null || echo 1)" = 1 ]; then :;
  else echo "FATAL: ip6tables DROP unavailable but IPv6 is enabled; refusing to run unsealed" >&2; exit 1; fi
}
