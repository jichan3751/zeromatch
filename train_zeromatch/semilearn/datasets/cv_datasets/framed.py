# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
import time

from PIL import Image

from torchvision import transforms
from .datasetbase import BasicDataset, BasicDatasetWithPLabel
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


def get_framed(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):


    root_dir, input_image_file_list, target_labels, candidate_labels, topic = get_framed_data_targets(
        task_name = name, ds_set='train', data_dir = data_dir)

    data = input_image_file_list
    targets = map_string_labels_to_int_labels(target_labels, candidate_labels)

    
    ## load plabel data
    plabel_d = load_plabels(args, candidate_labels)
    train_plabel_list = plabel_d['train']['plabel']
    assert np.array_equal(np.array(targets), np.array(plabel_d['train']['label']))

    # import ipdb; ipdb.set_trace()

    ## transform setting - copied from eurosat, without vertical flip
    # if size setting does not fit - may need to include resize (in get_imagenet.)

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    # lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
    lb_data, lb_targets, ulb_data, ulb_targets, lb_idx, ulb_idx = split_ssl_data(args, data, targets, num_classes,
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    lb_plabel_list =  list_indexing_by_list(train_plabel_list, lb_idx)
    ulb_plabel_list =  list_indexing_by_list(train_plabel_list, ulb_idx)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    print("lb count for each class:", get_n_samples_each_class(lb_targets,list(range(num_classes))))

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        lb_plabel_list = train_plabel_list
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)

    # lb_dset = FRAMEDDataset(root_dir, alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    # ulb_dset = FRAMEDDataset(root_dir, alg, ulb_data,  ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)

    lb_dset = FRAMEDDataset(root_dir, alg, lb_data, lb_plabel_list, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    ulb_dset = FRAMEDDataset(root_dir, alg, ulb_data, ulb_plabel_list, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    
    
    test_root_dir, test_input_image_file_list, test_target_labels, candidate_labels, topic = get_framed_data_targets(
        task_name = name, ds_set='test', data_dir = data_dir)
    test_data = test_input_image_file_list
    test_targets = map_string_labels_to_int_labels(test_target_labels, candidate_labels)

    assert np.array_equal(np.array(test_targets), np.array(plabel_d['test']['label']))
    test_plabel_list = plabel_d['test']['plabel']

    if args.algorithm == 'fullysupervised' and args.training_stage == 0:
        # make sure dev / test label is not accessed in stage 0.
        assert isinstance(test_targets, list)
        test_targets = [-1 for _ in range(len(test_targets))]

    # eval_dset = FRAMEDDataset(test_root_dir, alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)
    eval_dset = FRAMEDDataset(test_root_dir, alg, test_data, test_plabel_list, test_targets, num_classes, transform_val, False, None, None, False)

    if args.algorithm == 'fullysupervised' and args.training_stage == 0:
        print('--------> is pseudo-label extraction case. setting eval_dset to be lb_dset with val trains...')

        n_samples = len(lb_targets)
        if n_samples > 90000:
            use_eval_train_rate = 0.1
        elif n_samples > 9000:
            use_eval_train_rate = 0.2
        else:
            use_eval_train_rate = -1

        if use_eval_train_rate < 0:
            eval_train_data,  eval_train_plabel_list, eval_train_targets = lb_data, lb_plabel_list, lb_targets
        else:
            n_samples_to_use = int(n_samples * use_eval_train_rate)
            print(f'--------> train set too big for eval: sampling part of train set as eval set: {n_samples} -> {n_samples_to_use} samples')
            with NumpyTempSeed(args.seed):
                sampled_eval_train_indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
            
            eval_train_data = list_indexing_by_list(lb_data, sampled_eval_train_indices)
            eval_train_plabel_list = list_indexing_by_list(lb_plabel_list, sampled_eval_train_indices)
            eval_train_targets = list_indexing_by_list(lb_targets, sampled_eval_train_indices)

            ## checking the stats
            eval_train_label_list = np.array(eval_train_targets)
            eval_train_class_count = [ (eval_train_label_list==class_idx).sum()  for class_idx in range(num_classes)]
            print("eval_train_class_count:")
            print(eval_train_class_count)


        eval_dset = FRAMEDDataset(root_dir, alg, eval_train_data,  eval_train_plabel_list, eval_train_targets, num_classes, transform_val, False, None, None, False)
        # import ipdb; ipdb.set_trace()
        print('--> setting done.')

    # print(lb_dset[0])
    # print( ulb_dset[2])

    # import ipdb; ipdb.set_trace()

    return lb_dset, ulb_dset, eval_dset



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# class FRAMEDDataset(BasicDataset):

class FRAMEDDataset(BasicDatasetWithPLabel):
    # self.data will be data paths
    def __init__(self, root_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir

    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        path = self.data[idx]
        target = self.targets[idx]

        full_path = os.path.join(self.root_dir, path)
        img = pil_loader(full_path)
        return img, target


def get_framed_data_targets(task_name, ds_set, data_dir):

    if task_name == 'flowers':
        # Flowers102: 102 flower categories
        topic = 'flower'

        ## get class names
        class_info_fname = os.path.join(data_dir, 'class_files', 'Flowers102', f"class_names.txt")
        classes = []
        with open(class_info_fname, "r") as f:
            for l in f:
                classes.append(l.strip())

        candidate_labels = classes

        ## get file list
        root_dir = os.path.join(data_dir, 'Flowers102',f"{ds_set}")
        info_fname = os.path.join(data_dir, 'Flowers102', f"{ds_set}.txt")
        image_file_list = []
        target_labels = []
        with open(info_fname, "r") as f:
            for l in f:
                line = l.split(" ")
                cl = classes[int(line[1].strip())] # '0' or '1'

                img_fname0 = line[0] # train.zip@000/image_06736.jpg
                img_fname1 = img_fname0.split('@') # train.zip 000/image_06736.jpg
                assert img_fname1[0] == f"{ds_set}.zip"

                img_fname = img_fname1[1] # 000/image_06736.jpg

                image_file_list.append(img_fname)
                target_labels.append(cl)

    elif task_name == 'resisc':
        # RESICS45: Remote Sensing Image Scene Classification
        topic = 'scene'

        ## get class names
        class_info_fname = os.path.join(data_dir, 'RESICS45', f"train.json")
        classes = []
        with open(class_info_fname, "r") as f:
            data = json.load(f)
            for d in data["categories"]:
                classes.append(d["name"].replace("_", " "))

        candidate_labels = classes

        ## get file list
        root_dir = os.path.join(data_dir, 'RESICS45')
        info_fname = os.path.join(data_dir, 'RESICS45', f"{ds_set}.json")
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


    else:
        raise ValueError(f"task name {task_name} is not supported!")

    n_total_samples = len(image_file_list)
    print(f"topic {topic} task_name {task_name} ds_set {ds_set} loaded: total {n_total_samples}, ({len(candidate_labels)} classes)")

    n_samples_each_class = get_n_samples_each_class(target_labels, candidate_labels)

    print("n_samples_each_class:", n_samples_each_class)

    return root_dir, image_file_list, target_labels, candidate_labels, topic



def get_n_samples_each_class(target_labels, candidate_labels):
    
    n_samples_each_class = []
    for i, key in enumerate(candidate_labels):
        n_samples_this_class = sum([1 for label in target_labels if label == key])
        n_samples_each_class.append(n_samples_this_class)

    return n_samples_each_class

######### utils


#### plabel processing
import torch

def load_plabels(args, candidate_labels):

    # import ipdb; ipdb.set_trace()

    plabel_d = {}

    plabel_data_path_train = args.plabel_data_path
    plabel_d_train = torch.load(plabel_data_path_train)
    plabel_d['train'] = {
        'plabel': map_string_labels_to_int_labels(plabel_d_train['decoded_predictions_processed'], candidate_labels),
        'label': map_string_labels_to_int_labels(plabel_d_train['decoded_labels'], candidate_labels)
        }

    plabel_data_path_test = plabel_data_path_train.replace("_train_","_test_")
    plabel_d_test = torch.load(plabel_data_path_test)
    plabel_d['test'] = {
        'plabel': map_string_labels_to_int_labels(plabel_d_test['decoded_predictions_processed'], candidate_labels),
        'label': map_string_labels_to_int_labels(plabel_d_test['decoded_labels'], candidate_labels)
        }

    # import ipdb; ipdb.set_trace()

    return plabel_d


def map_string_labels_to_int_labels(predictions, candidate_labels):
    map_d = {key:i for i, key in enumerate(candidate_labels)}
    int_predictions = [map_d[pred] for pred in predictions]
    return int_predictions


#### my utils
# use: with NumpyTempSeed(seed): do np random stuff
class NumpyTempSeed:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_random_state = np.random.get_state()
        np.random.seed(self.seed)
        return self.np_random_state

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self.np_random_state) # revert np seed


def sample_partial_data_target(data, targets, use_partial_data_rate, seed):
    targets = np.array(targets)

    if use_partial_data_rate is not None:
        ct_data_before = len(targets)

        indices_by_parts = random_rate_split_indices_by_class(targets, rates=[use_partial_data_rate,], data_seed = seed)
        use_indices_original = indices_by_parts[0]

        if isinstance(data, list):
            data = list_indexing_by_list(data, use_indices_original)
        else:
            data = data[use_indices_original]

        targets = targets[use_indices_original]
        ct_data_after = len(targets)
        print(f"sample_partial_data_target(): reduce data {ct_data_before} -> {ct_data_after} ")

    else:
        use_indices_original = np.arange(len(targets))

    return data, targets, use_indices_original


def random_rate_split_indices_by_class(classes, rates=[0.8,0.2], data_seed =42):
    # classes: class label of each index
    # for each class, samples by rate designated.
    # returns indices_by_parts; list of list, for each rate

    assert sum(rates) <= 1.0001
    class_types = np.unique(classes)
    indices_by_class_oi = [np.where(classes == class_type)[0] for class_type in class_types]

    with NumpyTempSeed(data_seed):

        # random sample for each class
        indices_by_parts_classes = [[None for class_idx in range(len(class_types))] for rate in rates]
        for class_idx in range(len(class_types)):
            n_samples_class = len(indices_by_class_oi[class_idx])

            # get counts for each rate part
            n_sample_parts = []
            for i, rate in enumerate(rates):
                if i == len(rates) - 1 and abs(sum(rates) - 1) < 0.001:
                    # give all the leftovers
                    n_part = n_samples_class - sum(n_sample_parts)
                else:
                    n_part = int(n_samples_class * rate)

                n_sample_parts.append(n_part)

            # sample by rate
            all_shuffled_indices_class = np.random.permutation(indices_by_class_oi[class_idx])
            ct = 0
            for i, n_sample in enumerate(n_sample_parts):
                if n_sample == 0:
                    indices = []
                else:
                    # i=0 samples from start, i=1 samples from end, i = else samples sequentially.
                    if i == 1 :
                        indices = all_shuffled_indices_class[-n_sample:].copy()
                    else:
                        indices = all_shuffled_indices_class[ct:ct+n_sample].copy()
                        ct += n_sample

                indices_by_parts_classes[i][class_idx] = indices

        indices_by_parts = [None for rate in rates]
        for i, rate in enumerate(rates):
            indices_by_parts[i] = np.sort(np.concatenate(indices_by_parts_classes[i])).astype(int)

    return indices_by_parts

def list_indexing_by_list(l0, indices):
    if isinstance(l0, list):
        l1 = []
        for idx in indices:
            l1.append(l0[idx])
        return l1
    else:
        return l0[indices]


