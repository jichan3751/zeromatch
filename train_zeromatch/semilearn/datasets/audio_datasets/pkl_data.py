# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import pickle
import numpy as np
from glob import glob
from semilearn.datasets.utils import split_ssl_data, bytes_to_array
from .datasetbase import BasicDataset, BasicDatasetWithPLabel



def get_pkl_dset(args, alg='fixmatch', dataset='esc50', num_labels=40, num_classes=20, data_dir='./data', include_lb_to_ulb=True, onehot=False):
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
    data_dir = os.path.join(data_dir, dataset)

    # Supervised top line using all data as labeled data.
    if dataset == 'superbsi':
        all_train_files = sorted(glob(os.path.join(data_dir, 'train_*.pkl')))
        train_wav_list = []
        train_label_list = []
        for train_file in all_train_files:
            with open(train_file, 'rb') as f:
                train_data = pickle.load(f)
            for idx in train_data:
                train_wav_list.append(bytes_to_array(train_data[idx]['wav']))
                train_label_list.append(int(train_data[idx]['label']))
    else:
        with open(os.path.join(data_dir, 'train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        train_wav_list = []
        train_label_list = []
        for idx in train_data:
            train_wav_list.append(bytes_to_array(train_data[idx]['wav']))
            train_label_list.append(int(train_data[idx]['label']))

    with open(os.path.join(data_dir, 'dev.pkl'), 'rb') as f:
        dev_data = pickle.load(f)
    dev_wav_list = []
    dev_label_list = []
    for idx in dev_data:
        dev_wav_list.append(bytes_to_array(dev_data[idx]['wav']))
        dev_label_list.append(int(dev_data[idx]['label']))

    with open(os.path.join(data_dir, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    test_wav_list = []
    test_label_list = []
    for idx in test_data:
        test_wav_list.append(bytes_to_array(test_data[idx]['wav']))
        test_label_list.append(int(test_data[idx]['label']))

    plabel_d = load_audio_plabels(args)

    train_plabel_list = plabel_d['train']['plabel']
    dev_plabel_list = plabel_d['val']['plabel']
    test_plabel_list = plabel_d['test']['plabel']
    
    assert np.array_equal(np.array(train_label_list), plabel_d['train']['label'])
    assert np.array_equal(np.array(dev_label_list), plabel_d['val']['label'])
    assert np.array_equal(np.array(test_label_list), plabel_d['test']['label'])

    # plot_val_zero_shot_conf_mat(args, plabel_d)
    
    # use partial data for dev / test (for test)
    # activated only when  args.use_partial_dev_rate / test rate is not None
    train_wav_list, train_label_list, use_indices_original = sample_partial_data_target(train_wav_list, train_label_list, args.use_partial_data_rate, args.seed)
    dev_wav_list, dev_label_list, use_indices_original_dev = sample_partial_data_target(dev_wav_list, dev_label_list, args.use_partial_dev_rate, args.seed)
    test_wav_list, test_label_list, use_indices_original_test = sample_partial_data_target(test_wav_list, test_label_list, args.use_partial_test_rate, args.seed)

    # use partial data for dev / test 
    train_plabel_list  = list_indexing_by_list(train_plabel_list, use_indices_original)
    dev_plabel_list  = list_indexing_by_list(dev_plabel_list, use_indices_original_dev)
    test_plabel_list  = list_indexing_by_list(test_plabel_list, use_indices_original_test)
    
    dev_dset = BasicDatasetWithPLabel(alg=alg, data=dev_wav_list, plabels = dev_plabel_list ,targets=dev_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)
    test_dset = BasicDatasetWithPLabel(alg=alg, data=test_wav_list, plabels = test_plabel_list ,targets=test_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)
    if alg == 'fullysupervised':
        lb_dset = BasicDatasetWithPLabel(alg=alg, data=train_wav_list, plabels = train_plabel_list ,targets=train_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
        
        # import ipdb; ipdb.set_trace()
        if args.algorithm == 'fullysupervised' and args.training_stage == 0:
            print('--------> is pseudo-label extraction case. setting dev_dset and test_dset to be train set')

            n_samples = len(train_label_list)
            if n_samples > 90000:
                use_eval_train_rate = 0.1
            elif n_samples > 9000:
                use_eval_train_rate = 0.2
            else:
                use_eval_train_rate = -1

            if use_eval_train_rate < 0:
                eval_train_wav_list,  eval_train_plabel_list, eval_train_label_list = train_wav_list,  train_plabel_list, train_label_list
            else:
                n_samples_to_use = int(n_samples * use_eval_train_rate)
                print(f'--------> train set too big for eval: sampling part of train set as eval set: {n_samples} -> {n_samples_to_use} samples')
                with NumpyTempSeed(args.seed):
                    sampled_eval_train_indices = np.random.choice(n_samples, n_samples_to_use, replace=False)
                
                eval_train_wav_list = list_indexing_by_list(train_wav_list, sampled_eval_train_indices)
                eval_train_plabel_list = list_indexing_by_list(train_plabel_list, sampled_eval_train_indices)
                eval_train_label_list = list_indexing_by_list(train_label_list, sampled_eval_train_indices)

                ## checking the stats
                eval_train_label_list = np.array(eval_train_label_list)
                eval_train_class_count = [ (eval_train_label_list==class_idx).sum()  for class_idx in range(num_classes)]
                print("eval_train_class_count:")
                print(eval_train_class_count)

            dev_dset = BasicDatasetWithPLabel(alg=alg, data=eval_train_wav_list, plabels = eval_train_plabel_list ,targets=eval_train_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)
            test_dset = BasicDatasetWithPLabel(alg=alg, data=eval_train_wav_list, plabels = eval_train_plabel_list ,targets=eval_train_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=False)

            print('--> setting done.')

        return lb_dset, None, dev_dset, test_dset
    
    
    if dataset == 'fsdnoisy':
        # TODO: take care of this for imbalanced setting
        ulb_wav_list = []
        ulb_label_list = []
        with open(os.path.join(data_dir, 'ulb_train.pkl'), 'rb') as f:
            ulb_train_data = pickle.load(f)
        for idx in ulb_train_data:
            ulb_wav_list.append(bytes_to_array(ulb_train_data[idx]["wav"]))
            ulb_label_list.append(int(ulb_train_data[idx]["label"]))
        lb_wav_list, lb_label_list = train_wav_list, train_label_list
    else: 
        lb_wav_list, lb_label_list, ulb_wav_list, ulb_label_list, lb_idx, ulb_idx = split_ssl_data(args, train_wav_list, train_label_list, num_classes, 
                                                                                  lb_num_labels=num_labels,
                                                                                  ulb_num_labels=args.ulb_num_labels,
                                                                                  lb_imbalance_ratio=args.lb_imb_ratio,
                                                                                  ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                                  include_lb_to_ulb=include_lb_to_ulb)
    
    lb_plabel_list =  list_indexing_by_list(train_plabel_list, lb_idx)
    ulb_plabel_list =  list_indexing_by_list(train_plabel_list, ulb_idx)

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
            
    lb_dset = BasicDatasetWithPLabel(alg=alg, data=lb_wav_list, plabels=lb_plabel_list, targets=lb_label_list, num_classes=num_classes, is_ulb=False, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
    ulb_dset = BasicDatasetWithPLabel(alg=alg, data=ulb_wav_list, plabels=ulb_plabel_list, targets=ulb_label_list, num_classes=num_classes, is_ulb=True, one_hot=onehot, max_length_seconds=args.max_length_seconds, is_train=True)
    
    return lb_dset, ulb_dset, dev_dset, test_dset

import torch

def plot_val_zero_shot_conf_mat(args, plabel_d):

    output_dir = os.path.join(args.save_dir, args.save_name)
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    conf_mat = confusion_matrix(plabel_d['val']['label'], plabel_d['val']['plabel'], normalize='true')

    # Plot the confusion matrix
    plt.clf()
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat*100)
    # display_labels=iris.target_names
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.show()

    # plt.tight_layout()
    plot_fname =os.path.join(output_dir, 'zeroshot_plabel_val_conf_mat.png')
    plt.savefig(plot_fname, dpi=200)
    print("saved conf mat plot to",plot_fname)


def load_audio_plabels(args):

    if args.dataset == 'esc50':
        return load_esc50_plabels(args)
    elif args.dataset == 'urbansound8k':
        return load_urbansound8k_plabels(args)
    elif args.dataset == 'gtzan':
        # it has same format
        return load_urbansound8k_plabels(args)
    else:
        raise ValueError
    

def load_esc50_plabels(args):

    tmp_d = torch.load(args.plabel_data_path)

    plabel_d = {
        'train':{},
        'val':{},
        'test':{},
    }

    num_train_data = 1200
    num_dev_data = 400
    num_test_data = 400

    plabel_d['train']['label'] = tmp_d['labels'][:num_train_data]
    plabel_d['train']['plabel'] = tmp_d['preds'][:num_train_data]
    plabel_d['train']['logit'] = tmp_d['logits'][:num_train_data]

    plabel_d['val']['label'] = tmp_d['labels'][num_train_data : num_train_data+num_dev_data ]
    plabel_d['val']['plabel'] = tmp_d['preds'][num_train_data : num_train_data+num_dev_data ]
    plabel_d['val']['logit'] = tmp_d['logits'][num_train_data : num_train_data+num_dev_data ]

    plabel_d['test']['label'] = tmp_d['labels'][num_train_data+num_dev_data : ]
    plabel_d['test']['plabel'] = tmp_d['preds'][num_train_data+num_dev_data : ]
    plabel_d['test']['logit'] = tmp_d['logits'][num_train_data+num_dev_data : ]

    plabel_d = get_prior_matched_predictions(plabel_d)

    # import ipdb; ipdb.set_trace()

    return plabel_d

def load_urbansound8k_plabels(args):


    tmp_d = torch.load(args.plabel_data_path)
    

    with open(os.path.join(args.data_dir, args.dataset, 'row_idx.pkl'), 'rb') as f:
        row_idx_d = pickle.load(f)
        
    row_idx_d['val'] = row_idx_d['dev']

    plabel_d = {
        'train':{},
        'val':{},
        'test':{},
    }

    ds_keys = ['train', 'val', 'test']
    for ds_key in ds_keys:
        
        plabel_d[ds_key]['label'] = []
        plabel_d[ds_key]['plabel'] = []
        plabel_d[ds_key]['logit'] = []

        for idx in range(len(row_idx_d[ds_key])):
            row_idx = row_idx_d[ds_key][str(idx)]
            plabel_d[ds_key]['label'].append(tmp_d['labels'][row_idx])
            plabel_d[ds_key]['plabel'].append(tmp_d['preds'][row_idx])
            plabel_d[ds_key]['logit'].append(tmp_d['logits'][row_idx])

        plabel_d[ds_key]['logit'] = torch.stack(plabel_d[ds_key]['logit'])

    plabel_d = get_prior_matched_predictions(plabel_d)

    # import ipdb; ipdb.set_trace()

    return plabel_d

from scipy.special import softmax
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import time

def get_prior_matched_predictions(plabel_d):
    n_train = plabel_d['train']['logit'].shape[0]
    n_val = plabel_d['val']['logit'].shape[0]
    n_test = plabel_d['test']['logit'].shape[0]

    logits = torch.cat([plabel_d['train']['logit'], plabel_d['val']['logit'], plabel_d['test']['logit']])
    all_gt_labels = torch.cat([torch.tensor(plabel_d[key]['label']) for key in ['train', 'val', 'test']]).numpy()
    
    ## ref: https://github.com/JuliRao/Whisper_audio_classification/blob/main/src/scoring.py
    t0 = time.time()

    norm_scores_pt = logits
    
    # nll_tmp = norm_scores_pt.clone().cpu().numpy()
    nll = norm_scores_pt.clone().cpu().numpy()

    N = nll.shape[-1]
    weights = np.zeros((1, N))
    uniform = np.ones((N))/N

    step = 1
    lr = 0.1
    prior = np.mean(softmax(-1*nll, axis=-1), axis=0)

    while np.sum(np.abs(prior - uniform)) > 0.001:
        new_nll = nll + weights
        prior = np.mean(softmax(-1*new_nll, axis=-1), axis=0)
        weights -= lr*(prior<uniform)

        if step % 1000 == 0:
            lr /= 2

        step += 1
    weights -= np.min(weights)
    new_nll = nll + weights

    new_logits = new_nll
    new_preds = new_logits.argmin(-1)

    preds = logits.argmin(-1).numpy()

    t1 = time.time()

    print(f"For audio: get_prior_matched_predictions(): took {t1-t0:.3f}sec")
    print(f"Previous score: acc {accuracy_score(all_gt_labels, preds)}, f1 {f1_score(all_gt_labels, preds, average='macro')}")
    print(f"conf_mat:")
    print(confusion_matrix(all_gt_labels, preds))
    print(f"Prior-matched score: acc {accuracy_score(all_gt_labels, new_preds)}, f1 {f1_score(all_gt_labels, new_preds, average='macro')}")
    print(confusion_matrix(all_gt_labels, new_preds))
    print(f"overwriting preds..")

    plabel_d['train']['plabel'] = new_preds[:n_train]
    plabel_d['train']['logit'] = new_logits[:n_train]

    plabel_d['val']['plabel'] = new_preds[n_train : n_train + n_val ]
    plabel_d['val']['logit'] = new_logits[n_train : n_train + n_val ]

    plabel_d['test']['plabel'] = new_preds[n_train + n_val : ]
    plabel_d['test']['logit'] = new_logits[n_train + n_val : ]

    # import ipdb; ipdb.set_trace()

    return plabel_d



# US8k
# num_train_data = 7079
# num_dev_data = 816
# num_test_data = 837



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

        data = list_indexing_by_list(data, use_indices_original)
        targets = targets[use_indices_original]

        ct_data_after = len(targets)
        print(f"sample_partial_data_target(): reduce data {ct_data_before} -> {ct_data_after} ")

    else:
        use_indices_original = np.arange(len(targets))

    return data, targets, use_indices_original

def list_indexing_by_list(l0, indices):
    if isinstance(l0, list):
        l1 = []
        for idx in indices:
            l1.append(l0[idx])
        return l1
    else:
        return l0[indices]

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


