#!/bin/bash

set -e

## script for running cifar100 experiments.
## note: following options are included for running short test:
## TBA
## please remove / comment out this line to run the full experiment.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_API_KEY="your api key here"

SCRATCH_PATH="../scratch"
export HF_HOME=$SCRATCH_PATH/huggingface
export TORCH_HOME=$SCRATCH_PATH/cache/torch

data_path="$SCRATCH_PATH/data_zeromatch_cifar" # for cifar
# data_path="$SCRATCH_PATH/data_zeromatch_framed" # for framed

task_name=cifar100 
ds_set=test
prompt_mode=0
model_name="gpt-4.1-2025-04-14"
seed=42 # random seed applied to data and model init.

# task_name=flowers
# task_name=resisc 

# ds_set=train
# ds_set=test

model_name_short=(${model_name//\// })

main() {
    
    exp_name="zero_shot_${task_name}_${model_name_short}_${ds_set}_p${prompt_mode}_sd${seed}"
    output_dir="outputs/$exp_name"
    
    echo run $exp_name...

    python3 openai_inference.py \
        --data_dir $data_path \
        --task_name $task_name \
        --ds_set $ds_set \
        --model_name $model_name \
        --output_dir $output_dir \
        --prompt_mode $prompt_mode \
        --seed $seed 
    
}

main
