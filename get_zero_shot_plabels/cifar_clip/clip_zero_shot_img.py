import json
import argparse
import os

import numpy as np 
import torch

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import pipeline

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--model_path', type=str, default="openai/clip-vit-base-patch32")
    # parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--prompt_mode', type=int)
    parser.add_argument('--is_test',action='store_true')


    # parser.add_argument('--force-words', action='store_true') #
    
    args = parser.parse_args()
    print(f"Running with {args.seed}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.set == 'train':
        ds_split = 'train'
    elif  args.set == 'test':
        ds_split = 'test'
    else:
        raise ValueError

    labels_info = None
    if args.dataset == "cifar100":
        image_data = load_dataset(
            'uoft-cs/cifar100',
            split=ds_split
        )
        input_key = 'img'
        label_key = 'fine_label'

    elif args.dataset == "cifar10":
        image_data = load_dataset(
            'uoft-cs/cifar10',
            split=ds_split
        )
        input_key = 'img'
        label_key = 'label'
         
    else:
        raise ValueError
    
    # import ipdb; ipdb.set_trace()

    # test case
    if args.is_test:
        print("Option --is_test detected. use part of the dataset..")
        indices = np.random.choice(len(image_data),200,replace=False)
        image_data = image_data.select(indices)
    
    if labels_info is None:
        labels_info = image_data.info.features[label_key].names 
        
    gt_labels = image_data[label_key]

    candidate_labels = [f'This is a photo of {label}.' for label in labels_info]
    print("candidate text matches:")
    print(candidate_labels)
    print("GT labels first 10:")
    print(gt_labels[:10])

    # from https://huggingface.co/tasks/zero-shot-image-classification
    
    print("loading model: ", args.model_path)
    
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    model = model.to('cuda')

    batch_size = 8

    image_indices = np.arange(len(image_data))
    n_batches = len(image_indices) // batch_size

    batch_image_indices_list = np.array_split(image_indices, n_batches)

    probs_cat = []
    for batch_i, batch_img_indices in enumerate(batch_image_indices_list):
    
        images = image_data[batch_img_indices][input_key]
        inputs = processor(images=images, text=candidate_labels, return_tensors="pt", padding=True)

        # import ipdb; ipdb.set_trace()
        inputs = batch_to_device(inputs, 'cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=-1).cpu().numpy()

        probs_cat.append(probs)
        # scores = probs.tolist()

        if batch_i % 100 ==0:
            print(f"processed batch {batch_i} / {n_batches}")
        
    
    probs = np.concatenate(probs_cat)
    pred_labels = probs.argmax(axis = -1)

    print("computing metrics:")
    metrics = compute_metrics(gt_labels, pred_labels)
    print(metrics)
    print("confusion matrix:")
    print(confusion_matrix(gt_labels, pred_labels))

    output_dir = args.output_dir
    results_fname = output_dir + '/results.json'
    with open(results_fname, 'w') as f:
        json.dump(metrics, f, indent=4)
        print("saved results to", results_fname)

    outputs2 = {
        "probs": probs,
        "pred_labels": pred_labels,
        "gt_labels": gt_labels,
        "metrics": metrics
    }

    fname = os.path.join(output_dir,"plabel_results.pt")
    torch.save(outputs2, fname)
    print('saved to', fname)

    print(f"max cuda memory used: {int(torch.cuda.max_memory_allocated()/1048576)}MB")

    # import ipdb; ipdb.set_trace()


def compute_metrics(gt_labels, pred_labels):

    ret = {
            'f1': f1_score(gt_labels, pred_labels, average='macro'),
            'accuracy': accuracy_score(gt_labels, pred_labels )
           }
    
    return ret


def batch_to_device(batch, device):

    batch2 = {}
    for key in batch:
        batch2[key] = batch[key].to(device)
    
    return batch2

    


if __name__ == '__main__':
    main()
    