#!/usr/bin/env bash
# Launch seed-sweep configs sequentially (respects Prime 2-run concurrency via queue).
set -euo pipefail
cd "$(dirname "$0")/.."

WAVE="${1:-wave1}"

launch() {
  local cfg="$1"
  echo "=== prime train run --yes $cfg ==="
  prime train run --yes "$cfg"
  sleep 2
}

if [[ "$WAVE" == "wave1" ]]; then
  for i in 01 02 03 04 05; do
    launch "configs/sweeps/vc-seed-${i}.toml"
  done
  launch "configs/sweeps/vc-threshold-3.toml"
  launch "configs/sweeps/vc-threshold-7.toml"
  launch "configs/sweeps/vc-agg-average.toml"
  for i in 01 02 03 04 05; do
    launch "configs/sweeps/vig-p0-seed-${i}.toml"
    launch "configs/sweeps/vig-p3-seed-${i}.toml"
  done
  echo "Queued ${WAVE} configs. Monitor: prime train list --mine --plain"
else
  echo "Unknown wave: $WAVE (use wave1)"
  exit 1
fi
