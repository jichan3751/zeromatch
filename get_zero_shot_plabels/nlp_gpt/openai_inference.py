import json
import argparse
import time
import os
import sys

import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from openai import OpenAI

from utils.usb_data_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--ds_set', type=str, default='val', choices=['train','val','test'])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument('--prompt_mode', type=int, default=0)
    parser.add_argument('--max_samples',type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=1000)

    # parser.add_argument('--force-words', action='store_true') #
    
    args = parser.parse_args()
    
    print(f"Running with seed {args.seed}")
    np.random.seed(args.seed)

    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    ### getting the data ready
    input_texts, gt_labels = get_prompted_text_data(args)

    ## running openai inference
    output_text_fname = os.path.join(args.output_dir, "output_texts.pt")
    if os.path.exists(output_text_fname):
        print("loading existing output_text from ", output_text_fname)
        output_texts = torch.load(output_text_fname)
    else:

        output_texts = run_chat_completion_openai(input_texts, gt_labels, args)
        torch.save(output_texts, output_text_fname)

        print("saved output texts in ", output_text_fname)

    ### evaluate outputs and save.
    pred_labels = my_decoded_prediction(args.task_name, output_texts)

    scores = {
        'f1': f1_score(pred_labels, gt_labels, average='macro'),
        'accuracy': accuracy_score(pred_labels, gt_labels)
    }

    print("prediction results:")
    print(scores)

    results_fname=os.path.join(args.output_dir, 'results.json')
    with open(results_fname, 'w') as f:
        json.dump(scores, f, indent=4)
        print("saved", results_fname)


    n_classes = len(set(gt_labels))
    if n_classes < 11:
        # save confusion matrix info
        gt_labels_int = map_example_labels_inverse(args.task_name, gt_labels)
        pred_labels_int = map_example_labels_inverse(args.task_name, pred_labels)
        conf_mat_fname=os.path.join(args.output_dir, "confusion_matrix.png")
        save_confusion_matrix_plot(gt_labels_int, pred_labels_int, plot_fname=conf_mat_fname)
        print("saved conf mat in ", conf_mat_fname)

    outputs2 ={
        "decoded_predictions" : output_texts,
        "decoded_predictions_processed" : pred_labels,
        "decoded_labels": gt_labels
    }
    outputs_fname = os.path.join(args.output_dir,"plabel_results.pt")
    torch.save(outputs2, outputs_fname)
    print('saved output to', outputs_fname)

    # import ipdb; ipdb.set_trace()


def run_chat_completion_openai(input_texts, target_labels, args):

    # client = OpenAI(
    #     # This is the default and can be omitted
    #     api_key=os.environ.get("OPENAI_API_KEY"),
    # )

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


    model_name = args.model_name
    seed = args.seed
    logging_steps = args.logging_steps

    print(f"--> Running chat completion with model {model_name} seed {seed}..")

    n_samples = len(input_texts)

    # load from checkpoint
    save_ckpt_fname = os.path.join(args.output_dir, "save_ckpt.pt")
    if os.path.exists(save_ckpt_fname):
        print("loading ckpt from ", save_ckpt_fname)
        ckpt = torch.load(save_ckpt_fname)
        output_texts = ckpt['output_texts']
        last_idx = ckpt['last_idx']
    else:
        output_texts = [] 
        last_idx = -1

    t0 = time.time()

    for idx in range(last_idx+1, len(input_texts)):
        input_text = input_texts[idx]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
            ]

        completion = client.chat.completions.create(
            model = model_name,
            seed = seed,
            max_tokens = 32,
            messages = messages
            )

        # print(completion.choices[0].message.content)
        output_text = completion.choices[0].message.content
        output_texts.append(output_text)

        if (idx+1) % logging_steps == 0:
            t1 = time.time()
            print(f"--processing {idx} / {n_samples}, took {t1-t0:.2f}sec ({(t1-t0)/logging_steps:.2f}sec per sample):")
            print(f"--input: {input_text}")
            print(f"--output: {output_text}")
            print(f"--gt_label: {target_labels[idx]}")
            t0 = t1

        if (idx+1) % args.save_steps == 0:
            ckpt = {
                'output_texts' : output_texts,
                'last_idx' : idx,
            }
            torch.save(ckpt, save_ckpt_fname)
            print("saved ckpt in ", save_ckpt_fname)

        
    # import ipdb; ipdb.set_trace()
    
    return output_texts


def save_confusion_matrix_plot(labels, preds, plot_fname):

    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    conf_mat = confusion_matrix(labels, preds, normalize='true')

    # Plot the confusion matrix
    plt.clf()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat*100)
    # display_labels=iris.target_names
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.show()

    # plt.tight_layout()
    plt.savefig(plot_fname, dpi=200)
    print("saved conf mat plot to",plot_fname)


### utils

def write_json(data, fname):
    import json
    with open(fname, 'w') as outfile:
        json.dump(data, outfile, indent=4)
        # print('result written in'+res_fname)

def read_json(fname):
    import json
    with open(fname) as json_file:
        data = json.load(json_file)
    return data

def save_jsonl(dict_list, file_name):
    with open(file_name, 'w') as file:
        for obj in dict_list:
            file.write(json.dumps(obj) + '\n')


def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    return data

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"{__file__} took {(t1-t0)/60:.2f} minutes.")



    
