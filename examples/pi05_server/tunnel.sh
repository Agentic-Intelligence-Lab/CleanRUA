#!/usr/bin/env bash
# Forward localhost:6000 on this machine to the Pi0.5 server's localhost:6000 on the GPU host.
# Usage: tunnel.sh user@host [ssh_port]. Keep it running; agents then use the default --policy-url.
set -euo pipefail
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT:-6000}:localhost:${PORT:-6000}" -p "${2:-22}" "$1"
