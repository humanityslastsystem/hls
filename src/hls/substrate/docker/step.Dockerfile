# syntax=docker/dockerfile:1.29@sha256:572cebbc2b15e6f84c077742952674889cb63cd5e9b80abdffcbe587b6f19183
#
# The step image of the Docker substrate: the container that runs harness roles and seal gateways.
# It holds a shell, iptables, setpriv, git (a submission's checkout runs it), the Docker CLI with
# buildx and a CA store, and no hls code: a role binds the checkout and the toolchain's interpreter
# read-only, so the image does not change with each commit. Every input is pinned by digest or
# version. The harness builds it from this file and keeps it in the store's cache
# (hls.substrate.docker.step_image).

FROM docker:29.5.2-cli@sha256:9ba8e32bfc35a2c7ae2feb1e3241b2778ae21dee80f4dcd31d04e1cfdea86ea2 AS docker-cli
FROM docker/buildx-bin:0.34.0@sha256:1023c4a1ac77cee49520b620e1fa29d78be4ee3660bffd1a23b8a35f4e9ca417 AS buildx

# Ubuntu 24.04, whose glibc 2.39 runs the toolchain's interpreter, as released before the snapshot
# below: a newer base can hold a package version the snapshot lacks, which a package installed from
# it must equal (perl-base, for the perl that git needs).
FROM ubuntu:noble-20260917@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# apt reads the Ubuntu archive as it was at the snapshot.ubuntu.com timestamp the image recipes
# use (hls.recipes.pins.SNAPSHOT), so every dependency of the pinned iptables is fixed too. The
# snapshot is served over HTTPS only and the base has no CA certificates, so apt borrows the CA
# bundle of the pinned CLI image for this step; it does not enter the image, and apt still checks
# the archive's signatures with the base's keyring. The image's own CA store is the snapshot's
# ca-certificates, which a role's TLS client that reads the system's roots needs.
RUN --mount=type=bind,from=docker-cli,source=/etc/ssl/certs/ca-certificates.crt,target=/run/bootstrap-ca.crt <<'EOF'
set -eu
export DEBIAN_FRONTEND=noninteractive
snapshot=https://snapshot.ubuntu.com/ubuntu/20261001T000000Z/
sources=/etc/apt/sources.list.d/ubuntu.sources
printf '%s\n' 'Acquire::Retries "5";' 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99-hls
printf '%s\n' 'Acquire::https::CaInfo "/run/bootstrap-ca.crt";' > /etc/apt/apt.conf.d/98-bootstrap-ca
sed -i -e "s|http://archive.ubuntu.com/ubuntu/*|$snapshot|g" \
       -e "s|http://security.ubuntu.com/ubuntu/*|$snapshot|g" "$sources"
if grep '^URIs:' "$sources" | grep -v -F "$snapshot"; then
  echo "apt sources other than the snapshot" >&2
  exit 1
fi
apt-get update
apt-get install -y --no-install-recommends iptables=1.8.10-3ubuntu2 ca-certificates=20260601~24.04.1 git=1:2.43.0-1ubuntu7.3
apt-get clean
rm -rf /etc/apt/apt.conf.d/98-bootstrap-ca /var/lib/apt/lists/* /var/log/apt /var/log/dpkg.log /var/log/alternatives.log \
       /var/cache/ldconfig/aux-cache /var/cache/debconf/*-old
EOF

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=buildx /buildx /usr/local/libexec/docker/cli-plugins/docker-buildx

RUN <<'EOF'
set -eu
for tool in sh iptables ip6tables iptables-restore setpriv git docker; do
  command -v "$tool" > /dev/null || { echo "the step image lacks $tool" >&2; exit 1; }
done
test -s /etc/ssl/certs/ca-certificates.crt || { echo "the step image holds no CA bundle" >&2; exit 1; }
iptables --version
setpriv --version
docker --version | grep -F 'Docker version 29.5.2,'
docker buildx version | grep -F ' v0.34.0 '
EOF

ENTRYPOINT []
CMD []
