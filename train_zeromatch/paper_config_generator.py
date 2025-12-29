import os
import json
import itertools
import argparse
import sys


## reads existing config and apply paper-specific hyperparams.

def main():
    ## test
    # task_category = "usb_nlp"
    # algo_name = "adamatch"
    # task_name = "yahoo_answers"
    # seed = 0

    # create_paper_config(task_category, algo_name, task_name, seed)

    ## loop

    # task_category = "usb_nlp"
    # algo_names = ['adamatch','fixmatch','flexmatch','simmatch','comatch','dash','vat','softmatch','freematch','crmatch','uda']
    # task_name = "yahoo_answers"
    # seed = 0

    task_category = "usb_cv"
    algo_names = ['adamatch','fixmatch','flexmatch','simmatch','comatch','dash','vat','softmatch','freematch','crmatch','uda']
    task_name = "cifar100"
    seed = 0

    for algo_name in algo_names:
        create_paper_config(task_category, algo_name, task_name, seed)

    
def create_paper_config(task_category, algo_name, task_name, seed):

    config_label_size = get_config_label_size(task_name)
    config_orig_fname = os.path.join('config',task_category, algo_name,f"{algo_name}_{task_name}_{config_label_size}_{seed}.yaml")
    config_output_fname = os.path.join('config',task_category, algo_name,f"{algo_name}_{task_name}_{config_label_size}_paper_{seed}.yaml")

    with open(config_orig_fname, "r") as f:
        lines = f.readlines()
    
    ## modify config
    ow_config = get_ow_config(task_category, algo_name, task_name)

    lines2 = []
    for i, line in enumerate(lines):
        line2 = line
        for key in ow_config:
            if line.startswith(key):
                ## indicate this config is to accomodate paper's results
                line2 = str(key) + ": " + str(ow_config[key])+" # paper\n"
        
        lines2.append(line2)

    ## write resulti config
    with open(config_output_fname, "w") as f:
        f.writelines(lines2)

    print("wrote config:", config_output_fname)
    # import ipdb; ipdb.set_trace()

def get_ow_config(task_category, algo_name, task_name):

    if task_category == "usb_nlp":
        ow_config = {
            "batch_size": 4,
            "weight_decay": "0.0001",
        }
    
    elif task_category == "usb_cv":
        ow_config = {
            "batch_size": 16,
        }

    else:
        raise ValueError(f" task_category {task_category} is not supported!") 
    
    return ow_config

def get_config_label_size(task_name):
    if task_name == "cifar100":
        config_label_size = 400
    elif task_name == "yelp_review":
        config_label_size = 1000
    elif task_name == "amazon_review":
        config_label_size = 1000
    elif task_name == "ag_news":
        config_label_size = 200
    elif task_name == "yahoo_answers":
        config_label_size = 2000
    elif task_name == "esc50":
        config_label_size = 500
    elif task_name == "gtzan":
        config_label_size = 400
    elif task_name == "urbansound8k":
        config_label_size = 400
    else:
        raise ValueError(f" task_name {task_name} is not supported!") 
    
    return config_label_size




if __name__ == "__main__":
    main()

    