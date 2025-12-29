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

data_path="$SCRATCH_PATH/data_zeromatch_nlp"

task_name=ag_news 
ds_set=val
prompt_mode=0
model_name="flan-t5-small"
seed=42 # random seed applied to data and model init.

# task_name=yahoo_answers
# task_name=amazon_reivew 

# model_name="flan-t5-xxl"
# model_name="flan-t5-xl"
# ds_set=train
# ds_set=test

model_path="google/$model_name"

main() {
    
    exp_name="zero_shot_${task_name}_${model_name}_${ds_set}_p${prompt_mode}_sd${seed}"
    output_dir="outputs/$exp_name"
    
    echo run $exp_name...

    python3 flan_zero_shot.py \
        --data_dir $data_path \
        --dataset $task_name \
        --set $ds_set \
        --model_path $model_path \
        --output_dir $output_dir \
        --prompt_mode $prompt_mode \
        --seed $seed --is_test


    # python3 scripts/flan_zero_shot.py \
    #     --data_dir  \
    #     --dataset ag_news \
    #     --set val \
    #     --model_path google/flan-t5-xl \
    #     --output_dir outputs/zero_shot_ag_news_flan-t5-xl_val_p1_sd42 \
    #     --prompt_mode 1 \
    #     --chunk_idx 0 --n_chunks 1 \
    #     --seed 42

}

main
