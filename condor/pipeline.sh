#!/bin/bash
# Executed on the condor worker node for each fill.
# Args: $1 = config yaml path, $2 = fill number
set -euo pipefail

CONFIG=$1
FILL=$2

# TODO: adjust to your environment
cd /eos/user/l/leeja/HF/hfpipe

# --- venv environment for lxplus ---
export PATH=/cvmfs/cms-bril.cern.ch/brilconda310/bin:$PATH
source /eos/user/l/leeja/HF/hfpipe/env/bin/activate

python -m hfcli.run_pipeline --config "$CONFIG" --fills "$FILL"
