#!/bin/bash
# run_sbi.sh

# Change to the appropriate directory
cd /disk/xray15/aem2

# Source setup scripts
source setup_camels.sh
source /disk/xray15/aem2/envs/camels/bin/activate

# Set PYTHONPATH to prioritize your environment's packages
export PYTHONPATH=/disk/xray15/aem2/envs/camels/lib/python3.8/site-packages:$PYTHONPATH

# Run the Python script with all arguments passed to this shell script
python /disk/xray15/aem2/camels/proj1/LH/SBI/hyperparam_search/run_hyperparam_search.py "$@"