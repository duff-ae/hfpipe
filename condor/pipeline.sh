#!/bin/bash
# Executed on the condor worker node for each fill.
# Args: $1 = config yaml path, $2 = fill number
set -euo pipefail

CONFIG=$1
FILL=$2

# TODO: adjust to your environment
cd /afs/cern.ch/user/l/leeja/private/hfpipe

# --- venv environment for lxplus ---
source /afs/cern.ch/user/l/leeja/private/hfpipe/myenv/bin/activate
export PATH=/cvmfs/cms-bril.cern.ch/brilconda310/bin:$PATH
#export PATH=$HOME/.local/bin:/cvmfs/cms-bril.cern.ch/brilconda310/bin:$PATH
#export PYTHONPATH=/afs/cern.ch/user/l/leeja/private/hfpipe/myenv/lib/python3.9/site-packages/:$PYTHONPATH

python -m hfcli.run_pipeline --config "$CONFIG" --fills "$FILL"
