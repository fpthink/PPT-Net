#!/bin/sh
export PYTHONPATH=./

PYTHON=python
exp_name=$1
config=$2

exp_dir=exp/${exp_name}

model_log=${exp_dir}/log
mkdir -p ${model_log}

model_path=${exp_dir}/saved_model
mkdir -p ${model_path}

model_events=${exp_dir}/events
mkdir -p ${model_events}

now=$(date +"%Y%m%d_%H%M%S")

cp train.sh train.py ${config} ${exp_dir}

$PYTHON train.py \
    --config ${config} \
    --save_path=${exp_dir} 2>&1 | tee ${model_log}/train-$now.log

