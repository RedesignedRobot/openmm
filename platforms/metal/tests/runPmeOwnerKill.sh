#!/bin/sh
# Run only through gpuq after the owner releases the GPU.
# Arguments: compiled TestMetalPmeOwner, captured unit directory, replay text path.
set -eu
binary=${1:?TestMetalPmeOwner executable required}
unit=${2:-/tmp/openmm-metal-bench/holdout/fah-tip4pew}
capture=${3:-/tmp/m029-pme-owner-tip4pew.txt}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test -x "$binary"
test -d "$unit"
test -f "$here/capturePmeOwner.py"
if [ "${GPUQ_PREFLIGHT:-0}" = 1 ]; then
    exit 0
fi
python3 "$here/capturePmeOwner.py" "$unit" "$capture"
exec "$binary" mixed --kill "$capture"
