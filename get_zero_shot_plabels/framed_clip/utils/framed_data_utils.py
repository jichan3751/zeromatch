import json
import argparse
import time
import os

import numpy as np

# migrated from grip_app/menghini-neurips23-code-main/utils/prepare_data.py

def get_image_data(args):

    if args.task_name == 'dtd':
        # DTD: Describable Textures Dataset

        topic = 'texture'

        ## get class names
        class_info_fname = os.path.join(args.data_dir, 'class_files', 'DTD', f"class_names.txt")
        classes = []
        with open(class_info_fname, "r") as f:
            for l in f:
                classes.append(l.strip())

        candidate_labels = classes

        ## get file list
        root_dir = os.path.join(args.data_dir, 'DTD',f"{args.ds_set}")
        info_fname = os.path.join(args.data_dir, 'DTD', f"{args.ds_set}.txt")
        image_file_list = []
        target_labels = [] 
        with open(info_fname, "r") as f:
            for l in f:
                line = l.split(" ")
                cl = classes[int(line[1].strip())] # '0' or '1'

                img_fname0 = line[0] # test.zip@banded/banded_0002.jpg
                img_fname1 = img_fname0.split('@') # test.zip banded/banded_0002.jpg
                assert img_fname1[0] == f"{args.ds_set}.zip"

                img_fname = img_fname1[1] # banded/banded_0002.jpg

                cl_from_img_fname = img_fname.split('/')[0]
                assert cl_from_img_fname == cl

                image_file_list.append(img_fname)
                target_labels.append(cl)

    elif args.task_name == 'flowers':
        # Flowers102: 102 flower categories
        topic = 'flower'

        ## get class names
        class_info_fname = os.path.join(args.data_dir, 'class_files', 'Flowers102', f"class_names.txt")
        classes = []
        with open(class_info_fname, "r") as f:
            for l in f:
                classes.append(l.strip())

        candidate_labels = classes

        ## get file list
        root_dir = os.path.join(args.data_dir, 'Flowers102',f"{args.ds_set}")
        info_fname = os.path.join(args.data_dir, 'Flowers102', f"{args.ds_set}.txt")
        image_file_list = []
        target_labels = [] 
        with open(info_fname, "r") as f:
            for l in f:
                line = l.split(" ")
                cl = classes[int(line[1].strip())] # '0' or '1'

                img_fname0 = line[0] # train.zip@000/image_06736.jpg
                img_fname1 = img_fname0.split('@') # train.zip 000/image_06736.jpg
                assert img_fname1[0] == f"{args.ds_set}.zip"

                img_fname = img_fname1[1] # 000/image_06736.jpg

                image_file_list.append(img_fname)
                target_labels.append(cl)

    elif args.task_name == 'resisc':
        # RESICS45: Remote Sensing Image Scene Classification
        topic = 'scene' 

        ## get class names
        class_info_fname = os.path.join(args.data_dir, 'RESICS45', f"train.json")
        classes = []
        with open(class_info_fname, "r") as f:
            data = json.load(f)
            for d in data["categories"]:
                classes.append(d["name"].replace("_", " "))

        candidate_labels = classes

        ## get file list
        root_dir = os.path.join(args.data_dir, 'RESICS45')
        info_fname = os.path.join(args.data_dir, 'RESICS45', f"{args.ds_set}.json")
        image_file_list = []
        target_labels = [] 
        with open(info_fname, "r") as f:
                data = json.load(f)
                for d in data["images"]:
                    file_name = d["file_name"].split("@")[-1]
                    cl = file_name.split("/")[0].replace("_", " ")
                    # img = file_name.split("/")[-1]

                    image_file_list.append(file_name)
                    target_labels.append(cl)

    elif args.task_name == 'cifar100':
        topic = 'tiny image'

        ## get class names
        class_info_fname = os.path.join(args.data_dir, 'cifar100', f"class_names.json")
        with open(class_info_fname, "r") as f:
            data = json.load(f)
        candidate_labels0 = data
        candidate_labels = [cl.replace("_", " ") for cl in candidate_labels0] # "lawn_mower" -> "lawn mower"

        ## get file list
        root_dir = os.path.join(args.data_dir, 'cifar100',f"{args.ds_set}")
        info_fname = os.path.join(args.data_dir, 'cifar100', f"{args.ds_set}.json")
        image_file_list = []
        target_labels_int = [] 
        with open(info_fname, "r") as f:
            data = json.load(f)
            n_samples = len(data["labels"])
            for idx in range(n_samples):
                fname = data["files"][idx]
                # fname: "./cifar100/test/49/img_0.png"

                fname_parts = fname.split('/')
                fname_short = f"{fname_parts[-2]}/{fname_parts[-1]}"
                # fname_short: "49/img_0.png"
                
                cl = data["labels"][idx]

                image_file_list.append(fname_short)
                target_labels_int.append(cl)

        target_labels = [candidate_labels[int_label] for int_label in target_labels_int]

    else:
        raise ValueError(f"task name {args.task_name} is not supported!")

    n_total_samples = len(image_file_list)
    print(f"topic {topic} task_name {args.task_name} ds_set {args.ds_set} loaded: total {n_total_samples}, ({len(candidate_labels)} classes)")

    if args.max_samples > 0:
        sample_indices = np.random.choice(n_total_samples, args.max_samples, replace=False)
        
        image_file_list_ori = image_file_list
        target_labels_ori = target_labels

        image_file_list = [image_file_list[idx] for idx in sample_indices]
        target_labels = [target_labels[idx] for idx in sample_indices]

        print(f"Sampled partial data: {args.max_samples} samples")

    # import ipdb; ipdb.set_trace()

    return root_dir, image_file_list, target_labels, candidate_labels, topic

def my_img_decoded_prediction(task_name, decoded_predictions, candidate_labels):
    
    default_label_dict = {
        'dtd':'woven',
        'flowers':'rose',
        'resisc':'commercial area',
        'cifar100': 'road'
    }

    default_label = default_label_dict[task_name]

    decoded_predictions_processed = []
    ct_not_processed = 0
    for pred in decoded_predictions:
        pred2 = pred.lower().strip()
        pred_label = None

        for label in candidate_labels:
            if label in pred2:
                pred_label = label

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            pred_label = default_label
        
        decoded_predictions_processed.append(pred_label)    

    print(f"my_img_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")
    return decoded_predictions_processed



def map_example_labels_inverse(dataset_name, str_labels, candidate_labels):

    inv_label_map = {key:idx for idx, key in enumerate(candidate_labels)}
    
    targets = []
    for str_label in str_labels:
        target = inv_label_map[str_label]
        targets.append(target)

    return targets


