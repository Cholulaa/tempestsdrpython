#!/usr/bin/env bash
# Launch the TempestSDR edge probe from an env file (see agent.env).
# Usage: run-agent.sh [/path/to/agent.env]
set -euo pipefail

ENV_FILE="${1:-/etc/tempestsdr/agent.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
fi

: "${SERVER:?SERVER is required}"
: "${DEVICE_ID:?DEVICE_ID is required}"

args=(agent --server "$SERVER" --device-id "$DEVICE_ID"
      --interval "${INTERVAL:-2}" --quality "${QUALITY:-75}")
[[ -n "${API_KEY:-}" ]] && args+=(--api-key "$API_KEY")

if [[ "${SYNTHETIC:-0}" == "1" ]]; then
    args+=(--synthetic)
else
    args+=(--driver "${DRIVER:-rtlsdr}" --samplerate "${SAMPLERATE:-2400000}"
           --frequency "${FREQUENCY:-400000000}" --gain "${GAIN:-auto}")
    [[ -n "${HEIGHT:-}" ]]  && args+=(--height "$HEIGHT")
    [[ -n "${REFRESH:-}" ]] && args+=(--refresh "$REFRESH")
fi

exec python3 -m tempestsdr.cli "${args[@]}"
