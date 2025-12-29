# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset, BasicDatasetWithPLabel
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets


    ## load plabel data
    plabel_d = load_plabels(args)

    train_plabel_list = plabel_d['train']['plabel']
    
    assert np.array_equal(np.array(targets), np.array(plabel_d['train']['label']))

    # import ipdb; ipdb.set_trace()

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

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
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    # lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)

    # ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False) # not sure why defined two times
    # ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False) 

    lb_dset = BasicDatasetWithPLabel(alg, lb_data, lb_plabel_list, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    ulb_dset = BasicDatasetWithPLabel(alg, ulb_data, ulb_plabel_list, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)


    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets

    assert np.array_equal(np.array(test_targets), np.array(plabel_d['test']['label']))
    test_plabel_list = plabel_d['test']['plabel']

    if args.algorithm == 'fullysupervised' and args.training_stage == 0:
        # make sure dev / test label is not accessed in stage 0.
        assert isinstance(test_targets, list)
        test_targets = [-1 for _ in range(len(test_targets))]

    # eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)
    eval_dset = BasicDatasetWithPLabel(alg, test_data, test_plabel_list, test_targets, num_classes, transform_val, False, None, None, False)

    if args.algorithm == 'fullysupervised' and args.training_stage == 0:
        print('--------> is pseudo-label extraction case. setting eval_dset to be lb_dset...')

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


        eval_dset = BasicDatasetWithPLabel(alg, eval_train_data,  eval_train_plabel_list, eval_train_targets, num_classes, transform_val, False, None, None, False)
        # import ipdb; ipdb.set_trace()
        print('--> setting done.')


    return lb_dset, ulb_dset, eval_dset


#### plabel processing
import torch

cifar100_labels = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
    'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 
    'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 
    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
    'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 
    'wolf', 'woman', 'worm'
    ]

def load_plabels(args):

    cifar100_label_map = {key:idx for idx, key in enumerate(cifar100_labels)}

    plabel_d = {}

    plabel_data_path_train = args.plabel_data_path
    plabel_d_train = torch.load(plabel_data_path_train)

    if 'pred_labels' in plabel_d_train:
        plabel_d['train'] = {
            'plabel': plabel_d_train['pred_labels'],
            'label': plabel_d_train['gt_labels']
            }
    else:
        plabel_d['train'] = {
            'plabel': [cifar100_label_map[label] for label in plabel_d_train['decoded_predictions_processed']],
            'label': [cifar100_label_map[label] for label in plabel_d_train['decoded_labels']]
            }
    
        

    plabel_data_path_test = plabel_data_path_train.replace("_train_","_test_")
    plabel_d_test = torch.load(plabel_data_path_test)


    if 'pred_labels' in plabel_d_test:
        plabel_d['test'] = {
            'plabel': plabel_d_test['pred_labels'],
            'label': plabel_d_test['gt_labels']
            }
    else:
        plabel_d['test'] = {
            'plabel': [cifar100_label_map[label] for label in plabel_d_test['decoded_predictions_processed']],
            'label': [cifar100_label_map[label] for label in plabel_d_test['decoded_labels']]
            }

    # import ipdb; ipdb.set_trace()

    return plabel_d



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
