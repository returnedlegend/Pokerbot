#!/usr/bin/env bash
# Managed-cloud equivalent of check_build.sh. Keep organizer Dockerfiles intact.
# Trust the inherited host CA bundle through temporary BuildKit secret mounts;
# never copy session CA certificates into the repository or image layers.
set -euo pipefail

POKER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POKER_CA_BUNDLE="${POKER_PROXY_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
POKER_BUILD_TMP="$(mktemp -d /tmp/poker-cloud-build-XXXXXX)"
POKER_CONTAINER="poker-cloud-check-$$"
POKER_BASE_TAG="poker-agent-base:local"
POKER_BOT_TAG="poker-agent-cloud-check:latest"
POKER_DOCKER=(env -u DOCKER_HOST -u DOCKER_CONTEXT -u DOCKER_TLS
    -u DOCKER_TLS_VERIFY -u DOCKER_CERT_PATH
    docker --host=unix:///var/run/docker.sock)

cleanup() {
    "${POKER_DOCKER[@]}" rm -f "$POKER_CONTAINER" >/dev/null 2>&1 || true
    rm -rf -- "$POKER_BUILD_TMP"
}
trap cleanup EXIT

test -r "$POKER_CA_BUNDLE" || {
    echo "Cannot read CA bundle: $POKER_CA_BUNDLE" >&2
    exit 1
}
"${POKER_DOCKER[@]}" info >/dev/null

python3 - "$POKER_ROOT" "$POKER_BUILD_TMP" <<'PY'
from pathlib import Path
import sys

root, temp = map(Path, sys.argv[1:])

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit("Organizer Dockerfile changed; review cloud trust adaptation.")
    return text.replace(old, new, 1)

base = (root / "docker/Dockerfile.base").read_text()
base = replace_once(
    base,
    'RUN mkdir -p /opt/poker \\',
    'RUN --mount=type=secret,id=cloud_ca,required=true export PIP_CERT=/run/secrets/cloud_ca \\\n && mkdir -p /opt/poker \\',
)
base = replace_once(
    base,
    'RUN pip install --no-cache-dir numpy',
    'RUN --mount=type=secret,id=cloud_ca,required=true export PIP_CERT=/run/secrets/cloud_ca \\\n && pip install --no-cache-dir numpy',
)
(temp / "Dockerfile.base").write_text(base)

agent = (root / "docker/Dockerfile.agent").read_text()
agent = replace_once(
    agent,
    'RUN --mount=type=cache,target=/var/cache/pip \\',
    'RUN --mount=type=cache,target=/var/cache/pip --mount=type=secret,id=cloud_ca,required=true \\',
)
agent = replace_once(
    agent,
    'PIP_CACHE_DIR=/var/cache/pip pip install',
    'PIP_CERT=/run/secrets/cloud_ca PIP_CACHE_DIR=/var/cache/pip pip install',
)
(temp / "Dockerfile.agent").write_text(agent)
PY

echo "Building organizer images with the inherited cloud CA trust bundle."
"${POKER_DOCKER[@]}" build \
    --secret "id=cloud_ca,src=$POKER_CA_BUNDLE" \
    -t "$POKER_BASE_TAG" -f "$POKER_BUILD_TMP/Dockerfile.base" "$POKER_ROOT"
"${POKER_DOCKER[@]}" build \
    --secret "id=cloud_ca,src=$POKER_CA_BUNDLE" \
    --build-arg "BASE_IMAGE=$POKER_BASE_TAG" \
    -t "$POKER_BOT_TAG" -f "$POKER_BUILD_TMP/Dockerfile.agent" "$POKER_ROOT"

POKER_BASE_BYTES="$("${POKER_DOCKER[@]}" image inspect --format '{{.Size}}' "$POKER_BASE_TAG")"
POKER_BOT_BYTES="$("${POKER_DOCKER[@]}" image inspect --format '{{.Size}}' "$POKER_BOT_TAG")"
POKER_ADDED_BYTES=$(( POKER_BOT_BYTES - POKER_BASE_BYTES ))
echo "Bot adds $POKER_ADDED_BYTES bytes (limit: 3221225472)."
if (( POKER_ADDED_BYTES > 3221225472 )); then
    echo "Bot exceeds the organizer's 3 GiB added-image limit." >&2
    exit 1
fi

# A copied smoke file remains readable to the container's nonroot user even
# if the host checkout has restrictive file permissions.
install -m 0644 "$POKER_ROOT/tools/smoke_test.py" "$POKER_BUILD_TMP/smoke_test.py"
"${POKER_DOCKER[@]}" run --detach --name "$POKER_CONTAINER" \
    --network none --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m \
    --cap-drop ALL --security-opt no-new-privileges \
    --pids-limit 256 --cpus 1 --memory 4g \
    --env PORT=5008 --env AGENT_TOKEN=local-check \
    --volume "$POKER_BUILD_TMP/smoke_test.py:/smoke/smoke_test.py:ro" \
    "$POKER_BOT_TAG" >/dev/null

if ! "${POKER_DOCKER[@]}" exec "$POKER_CONTAINER" \
    python /smoke/smoke_test.py --port 5008 --startup 15; then
    "${POKER_DOCKER[@]}" logs --tail 40 "$POKER_CONTAINER" >&2
    exit 1
fi
echo "CLOUD BUILD AND RESTRICTED SMOKE CHECKS PASSED."
