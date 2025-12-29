# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import numpy as np

from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset, BasicDatasetWithPLabel


def get_json_dset(args, alg='fixmatch', dataset='acmIb', num_labels=40, num_classes=20, data_dir='./data', index=None, include_lb_to_ulb=True, onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeled data)
        """
        json_dir = os.path.join(data_dir, dataset)
        
        # Supervised top line using all data as labeled data.
        with open(os.path.join(json_dir,'train.json'),'r') as json_data:
            train_data = json.load(json_data)
            train_sen_list = []
            train_label_list = []
            for idx in train_data:
                train_sen_list.append((train_data[idx]['ori'],train_data[idx]['aug_0'],train_data[idx]['aug_1']))
                train_label_list.append(int(train_data[idx]['label']))
        with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
            dev_data = json.load(json_data)
            dev_sen_list = []
            dev_label_list = []
            for idx in dev_data:
                dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                dev_label_list.append(int(dev_data[idx]['label']))
        with open(os.path.join(json_dir,'test.json'),'r') as json_data:
            test_data = json.load(json_data)
            test_sen_list = []
            test_label_list = []
            for idx in test_data:
                test_sen_list.append((test_data[idx]['ori'],'None','None'))
                test_label_list.append(int(test_data[idx]['label']))


        ## load plabel data
        if args.plabel_data_path in ['random_noise']:
            plabel_d = generate_synthetic_plabels(args, train_label_list, dev_label_list, test_label_list)
        else:
            plabel_d = load_plabels(args)
        
        # import ipdb; ipdb.set_trace()

        train_plabel_list = plabel_d['train']['plabel']
        dev_plabel_list = plabel_d['val']['plabel']
        test_plabel_list = plabel_d['test']['plabel']

        if args.algorithm == 'fullysupervised' and args.training_stage == 0:
            # make sure dev / test label is not accessed in stage 0.
            assert isinstance(dev_label_list, list)
            dev_label_list = [-1 for _ in range(len(dev_label_list))]
            assert isinstance(test_label_list, list)
            test_label_list = [-1 for _ in range(len(test_label_list))]

        dev_dset = BasicDatasetWithPLabel(alg, dev_sen_list, dev_plabel_list,  dev_label_list, num_classes, False, onehot)
        test_dset = BasicDatasetWithPLabel(alg, test_sen_list, test_plabel_list, test_label_list, num_classes, False, onehot)

        if alg == 'fullysupervised':
            lb_dset = BasicDatasetWithPLabel(alg, train_sen_list, train_plabel_list, train_label_list, num_classes, False,onehot)

            if args.algorithm == 'fullysupervised' and args.training_stage == 0:
                print('--------> is pseudo-label extraction case. setting dev_dset and test_dset to be train set')
                
                n_samples = len(train_sen_list)
                if n_samples > 90000:
                    use_eval_train_rate = 0.1
                elif n_samples > 9000:
                    use_eval_train_rate = 0.2
                else:
                    use_eval_train_rate = -1

                if use_eval_train_rate < 0:
                    eval_train_sen_list,  eval_train_plabel_list, eval_train_label_list = train_sen_list,  train_plabel_list, train_label_list
                else:
                    n_samples_to_use = int(n_samples * use_eval_train_rate)
                    print(f'--------> train set too big for eval: sampling part of train set as eval set: {n_samples} -> {n_samples_to_use} samples')
                    with NumpyTempSeed(args.seed):
                        sampled_eval_train_indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
                    
                    eval_train_sen_list = list_indexing_by_list(train_sen_list, sampled_eval_train_indices)
                    eval_train_plabel_list = list_indexing_by_list(train_plabel_list, sampled_eval_train_indices)
                    eval_train_label_list = list_indexing_by_list(train_label_list, sampled_eval_train_indices)

                    ## checking the stats
                    eval_train_label_list = np.array(eval_train_label_list)
                    eval_train_class_count = [ (eval_train_label_list==class_idx).sum()  for class_idx in range(num_classes)]
                    print("eval_train_class_count:")
                    print(eval_train_class_count)
                
                dev_dset = BasicDatasetWithPLabel(alg, eval_train_sen_list,  eval_train_plabel_list, eval_train_label_list, num_classes, False, onehot)
                test_dset = BasicDatasetWithPLabel(alg,  eval_train_sen_list,  eval_train_plabel_list, eval_train_label_list, num_classes, False, onehot)

                # import ipdb; ipdb.set_trace()
                print('--> setting done.')

            return lb_dset, None, dev_dset, test_dset
        
        lb_sen_list, lb_label_list, ulb_sen_list, ulb_label_list, lb_idx, ulb_idx = split_ssl_data(args, train_sen_list, train_label_list, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)
        
        lb_plabel_list =  list_select_samples(train_plabel_list, lb_idx)
        ulb_plabel_list =  list_select_samples(train_plabel_list, ulb_idx)

        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(num_classes)]
        for c in train_label_list:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        
        lb_dset = BasicDatasetWithPLabel(alg, lb_sen_list, lb_plabel_list, lb_label_list, num_classes, False, onehot)
        ulb_dset = BasicDatasetWithPLabel(alg, ulb_sen_list, ulb_plabel_list, ulb_label_list, num_classes, True, onehot)

        return lb_dset, ulb_dset, dev_dset, test_dset


import torch

# label maps used in zero-shot
AG_NEWS_LABEL_MAP = {
    '0': 'world',
    '1': 'sports',
    '2': 'business',
    '3': 'technology',
}

YAHOO_ANSWERS_LABEL_MAP = {
    '0': 'society',
    '1': 'science',
    '2': 'health',
    '3': 'education',
    '4': 'computer',
    '5': 'sports',
    '6': 'business',
    '7': 'entertainment',
    '8': 'relationship',
    '9': 'politics',
}

YELP_REVIEW_LABEL_MAP = {
    '0': 'very negative',
    '1': 'negative',
    '2': 'neutral',
    '3': 'positive',
    '4': 'very positive',
}

AMAZON_REVIEW_LABEL_MAP = {
    '0': 'very negative',
    '1': 'negative',
    '2': 'neutral',
    '3': 'positive',
    '4': 'very positive',
}

def process_plabel_d(args, plabel_d):
    mapped_predictions = plabel_d['decoded_predictions_processed']
    mapped_labels = plabel_d['decoded_labels']
    if args.dataset == "ag_news":
        label_map = AG_NEWS_LABEL_MAP
    elif args.dataset == "yahoo_answers":
        label_map = YAHOO_ANSWERS_LABEL_MAP
    elif args.dataset == "yelp_review":
        label_map = YELP_REVIEW_LABEL_MAP
    elif args.dataset == "amazon_review":
        label_map = AMAZON_REVIEW_LABEL_MAP
    else:
        raise ValueError

    label2int_map = {label_map[key]:int(key) for key in label_map}
    # I made mistake when getting pseudo-label
    #  for ag news.'society' -> 'world'
    #  for ag news.'world' -> 'society'

    if args.dataset == "ag_news":
        label2int_map['society'] = 0

    if args.dataset == "yahoo_answers":
        # I made mistake when getting pseudo-label for ag news.
        # 'society' -> ''world'
        label2int_map['world'] = 0

    mapped_predictions2 = [label2int_map[label_str] for label_str in mapped_predictions]
    mapped_labels2 = [label2int_map[label_str] for label_str in mapped_labels]

    return mapped_predictions2, mapped_labels2


def load_plabels(args):

    plabel_d = {}

    plabel_data_path_train = args.plabel_data_path
    plabel_d_train = torch.load(plabel_data_path_train)
    mapped_predictions2, mapped_labels2 = process_plabel_d(args, plabel_d_train)
    plabel_d['train'] = {
        'plabel': mapped_predictions2,
        'label': mapped_labels2
        }

    plabel_data_path_val = plabel_data_path_train.replace("_train_","_val_")
    plabel_d_val = torch.load(plabel_data_path_val)
    mapped_predictions2, mapped_labels2 = process_plabel_d(args, plabel_d_val)
    plabel_d['val'] = {
        'plabel': mapped_predictions2,
        'label': mapped_labels2
        }

    plabel_data_path_test = plabel_data_path_train.replace("_train_","_test_")
    plabel_d_test = torch.load(plabel_data_path_test)
    mapped_predictions2, mapped_labels2 = process_plabel_d(args, plabel_d_test)
    plabel_d['test'] = {
        'plabel': mapped_predictions2,
        'label': mapped_labels2
        }

    # import ipdb; ipdb.set_trace()

    return plabel_d


def generate_synthetic_plabels(args, train_label_list, dev_label_list, test_label_list):
    plabel_d = {}

    with NumpyTempSeed(args.seed):

        plabel_d['train'] = {
            'plabel': np.random.randint(0, args.num_classes, size=len(train_label_list)).tolist(),
            'label': train_label_list
            }
        
        plabel_d['val'] = {
            'plabel': np.random.randint(0, args.num_classes, size=len(dev_label_list)).tolist(),
            'label': dev_label_list
            }
        
        plabel_d['test'] = {
            'plabel': np.random.randint(0, args.num_classes, size=len(test_label_list)).tolist(),
            'label': test_label_list
            }
        
    return plabel_d



def list_select_samples(list0, indices):
    # list1 = np.array(list0)
    # list2 = list1[indices]
    # return list2.tolist()

    list1 = []
    for idx in indices:
        list1.append(list0[idx])

    return list1


def multi_list_select_samples(list00, indices):
    ret_list = []
    for list0 in list00:
        list2 = list_select_samples(list0, indices)
        ret_list.append(list2)

    return ret_list






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
        indices_by_parts = random_rate_split_indices_by_class(targets, rates=[use_partial_data_rate,], data_seed = seed)
        use_indices_original = indices_by_parts[0]

        data = list_indexing_by_list(data, use_indices_original)
        targets = targets[use_indices_original]
    else:
        use_indices_original = np.arange(len(targets))

    return data, targets, use_indices_original

def list_indexing_by_list(l0, indices):

    if isinstance(l0, np.ndarray):
        return l0[indices]
    else:
        assert isinstance(l0, list)
        l1 = []
        for idx in indices:
            l1.append(l0[idx])
        return l1

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


