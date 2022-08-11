#!/bin/bash
cd /Benchmark/Gridsearch_Genn
source /opt/conda/etc/profile.d/conda.sh
conda activate base
source /opt/rh/devtoolset-9/enable
unset CUDA_PATH
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
python GeNN_GridSearch.py $@ || exit 0






