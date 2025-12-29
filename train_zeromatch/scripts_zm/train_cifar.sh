#!/bin/bash
set -e

## script for running cifar100 experiments.
## note: following options are included for running short test:
## --epoch_ow, --num_train_iter_ow, --num_eval_iter_ow, --num_log_iter_ow
## please remove / comment out this line to run the full experiment.

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

SCRATCH_PATH="../scratch"
export HF_HOME=$SCRATCH_PATH/huggingface
export TORCH_HOME=$SCRATCH_PATH/cache/torch

data_path="$SCRATCH_PATH/data_zeromatch_cifar"

task_name=cifar100 # datasets: cifar100
label_size=200 # number of labeled data (options: 100, 200, 400)
algo_name=adamatch # semi-supervised learning algorithm. (adamatch only)
plabel_model=0 # see set_plabel_exp_name() for exact FM used.
ann_mode=1 # 1: annealing is applied in KD loss (in 2nd stage), 0: annealing is not applied.
lmbd=1.0 # lambda scale applied to KD loss (in 2nd stage)
seed=42 # random seed applied to data and model init.

main() {
    
    exp_name="${task_name}_l${label_size}_${algo_name}_pm${plabel_model}_ann${ann_mode}_lmbd${lmbd}_sd${seed}"
    output_dir="outputs/$exp_name"

    set_config_label_size
    set_plabel_exp_name

    # 1st stage training - KD
    # runs KD on pseudo-labels
    
    echo running stage 1 training for $exp_name...    
    python -u train.py \
        --save_dir_ow $output_dir \
        --save_name_ow stage0 --num_labels_ow $label_size \
        --load_path_ow "${output_dir}/stage0/latest_model.pth" \
        --disable_multiprocessing_distributed \
        --seed_ow $seed --ann_mode $ann_mode --lmbd $lmbd --training_stage 0 \
        --c config/usb_cv_paper/fullysupervised/fullysupervised_${task_name}_${config_label_size}_paper_0.yaml \
        --data_dir_ow $data_path \
	    --plabel_data_path $data_path/plabels/${plabel_exp_name}/plabel_results.pt \
        --epoch_ow 4 --num_train_iter_ow 96 --num_eval_iter_ow 32 --num_log_iter_ow 8

    # 2nd stage training - KD + SSL
    # loads model from previous stage and run KD + SSL
    
    echo running stage 2 training for $exp_name...
    python -u train.py \
        --save_dir_ow $output_dir \
        --save_name_ow stage1 --num_labels_ow $label_size \
        --load_path_ow $output_dir/stage1/latest_model.pth \
        --disable_multiprocessing_distributed \
        --seed_ow $seed --ann_mode $ann_mode --lmbd $lmbd --training_stage 1 \
        --load_model_only_path $output_dir/stage0/model_best.pth \
        --c config/usb_cv_paper/${algo_name}/${algo_name}_${task_name}_${config_label_size}_paper_0.yaml \
        --data_dir_ow $data_path \
        --plabel_data_path "${data_path}/plabels/${plabel_exp_name}/plabel_results.pt" \
        --epoch_ow 4 --num_train_iter_ow 96 --num_eval_iter_ow 32 --num_log_iter_ow 8

}


set_config_label_size() {

    if [ "$task_name" == "cifar100" ]; then
        config_label_size=400
    else
        echo task name $task_name is not supported. exitting..
        exit 1
    fi

}

set_plabel_exp_name() {

    if [ "$task_name" == "cifar100" ]; then
        if [ "$plabel_model" == "0" ]; then
            plabel_exp_name="zero_shot_cifar100_clip-vit-large-patch14_train_p1_sd42"
        elif [ "$plabel_model" == "1" ]; then
            plabel_exp_name="zero_shot_cifar100_clip-vit-base-patch32_train_p1_sd42"
        elif [ "$plabel_model" == "2" ]; then
            plabel_exp_name="zero_shot_cifar100_gpt-4.1-2025-04-14_train_p0_sd42"
        else
            echo plabel_model $plabel_model is not supported. exitting..
            exit 1
        fi
    else
        echo task name $task_name is not supported. exitting..
        exit 1
    fi

}

main
