#!/bin/bash

set -e

## script for running cifar100 experiments.
## note: following options are included for running short test:
## TBA
## please remove / comment out this line to run the full experiment.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

SCRATCH_PATH="../scratch"
export HF_HOME=$SCRATCH_PATH/huggingface
export TORCH_HOME=$SCRATCH_PATH/cache/torch

data_path="$SCRATCH_PATH/data_zeromatch_cifar"

task_name=cifar100 
ds_set=test
prompt_mode=1
model_name="clip-vit-base-patch32"
seed=42 # random seed applied to data and model init.

# model_name="clip-vit-large-patch14"
# ds_set=train


model_path="openai/$model_name"




main() {
    
    exp_name="zero_shot_${task_name}_${model_name}_${ds_set}_p${prompt_mode}_sd${seed}"
    output_dir="outputs/$exp_name"
    
    echo run $exp_name...

    python clip_zero_shot_img.py \
        --data_dir $data_path \
        --dataset $task_name \
        --set $ds_set \
        --model_path $model_path \
        --output_dir $output_dir \
        --prompt_mode $prompt_mode \
        --seed $seed

}

main
