#!/bin/bash
cd /Benchmark/Gridsearch_NEST
source /opt/rh/devtoolset-9/enable
source /opt/conda/etc/profile.d/conda.sh
source /opt/nest/bin/nest_vars.sh
conda activate base
python NEST_GridSearch.py $@ || exit 0


