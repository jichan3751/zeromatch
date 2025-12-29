import json
import argparse
import time
import os
import sys

import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from utils.framed_data_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--ds_set', type=str, default='val', choices=['train','val','test'])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-base-patch32")
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

    ### getting the data ready -
    root_dir, image_file_list, target_labels, candidate_labels, topic = get_image_data(args)

    ## running openai inference
    probs, pred_labels, image_embeds = run_clip_inference(root_dir, image_file_list, target_labels, candidate_labels, topic, args)

    scores = {
        'f1': f1_score(pred_labels, target_labels, average='macro'),
        'accuracy': accuracy_score(pred_labels, target_labels)
    }

    print("prediction results:")
    print(scores)

    results_fname=os.path.join(args.output_dir, 'results.json')
    with open(results_fname, 'w') as f:
        json.dump(scores, f, indent=4)
        print("saved", results_fname)

    # n_classes = len(set(gt_labels))
    n_classes = len(candidate_labels)
    if n_classes < 11:
        # save confusion matrix info
        gt_labels_int = map_example_labels_inverse(args.task_name, target_labels, candidate_labels)
        pred_labels_int = map_example_labels_inverse(args.task_name, pred_labels, candidate_labels)
        conf_mat_fname=os.path.join(args.output_dir, "confusion_matrix.png")
        save_confusion_matrix_plot(gt_labels_int, pred_labels_int, plot_fname=conf_mat_fname)
        print("saved conf mat in ", conf_mat_fname)

    outputs2 ={
        "decoded_predictions" : None,
        "decoded_predictions_processed" : pred_labels,
        "decoded_labels": target_labels,
        "embedding": image_embeds,
        "probs":probs,
    }
    outputs_fname = os.path.join(args.output_dir,"plabel_results.pt")
    torch.save(outputs2, outputs_fname)
    print('saved output to', outputs_fname)

    # import ipdb; ipdb.set_trace()




from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def run_clip_inference(root_dir, image_file_list, target_labels, candidate_labels, topic, args):
    
    
    batch_size = 8

    image_indices = np.arange(len(image_file_list))
    n_batches = len(image_indices) // batch_size

    batch_image_indices_list = np.array_split(image_indices, n_batches)

    candidate_prompts = [f'This is a photo of {label}.' for label in candidate_labels]
    print("candidate text matches:")
    print(candidate_labels)

    
    print("loading model: ", args.model_name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = model.to('cuda')
    
    probs_cat = []
    image_embeds_cat = []
    t0 = time.time()
    ct_processed = 0
    for batch_i, batch_img_indices in enumerate(batch_image_indices_list):
        
        images = []
        for idx in batch_img_indices:
            full_path = os.path.join(root_dir, image_file_list[idx])
            img = pil_loader(full_path)
            images.append(img)


        inputs = processor(images=images, text=candidate_prompts, return_tensors="pt", padding=True)
        # import ipdb; ipdb.set_trace()
        inputs = batch_to_device(inputs, 'cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1).cpu()


        # image_embeds = outputs.vision_model_output.pooler_output
        # image_embeds = model.visual_projection(image_embeds)
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # same as below
        image_embeds = outputs.image_embeds
        image_embeds_cat.append(image_embeds.cpu())

        # import ipdb; ipdb.set_trace()

        probs_cat.append(probs)
        # scores = probs.tolist()

        ct_processed += len(batch_img_indices)

        if (batch_i+1) % 100 ==0:
            t1 = time.time()
            print(f"processed batch {batch_i} / {n_batches}: took {t1-t0:.2f}sec ({(t1-t0)/ct_processed:.3f}sec per sample)")
            t0 = t1
            ct_processed = 0

    probs = torch.cat(probs_cat)
    pred_labels = probs.argmax(axis = -1)

    pred_labels_text = [candidate_labels[label] for label in pred_labels]

    image_embeds = torch.cat(image_embeds_cat)

    # import ipdb; ipdb.set_trace()

    return probs, pred_labels_text, image_embeds



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


def batch_to_device(batch, device):

    batch2 = {}
    for key in batch:
        batch2[key] = batch[key].to(device)
    
    return batch2

    

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



    
