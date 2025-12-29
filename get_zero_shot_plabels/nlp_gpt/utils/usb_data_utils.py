import json
import argparse
import time
import os

import numpy as np

from datasets import Dataset, DatasetDict



# initial 4 samples in train set that covers label 0,1,2,3
AG_NEWS_ICL_SAMPLES=[
    {'text': 'Teen flies in plane #39;s landing gearA homeless teenager who hid in the landing gear of a passenger plane survived a 700-kilometre flight across south-western China but his companion fell and probably died, state media reported on Friday.', 'label': '0'},
    {'text': 'BILLIONAIRE #39;S PAYDAY TO GILD GOLDEN YEARSWhile Wilbur Ross celebrates his new honeymoon, he #39;s also toasting his latest deal - an \\$80 million bet that reaped a \\$4.5 billion payoff in just two years.', 'label': '2'},
    {'text': 'China confirms more Internet cafe closuresCHINA IS continuing its pogrom on the country #39;s Internet cafes and officially confirmed that it has shut more than 1,600 Internet cafes since February.', 'label': '3'},
    {'text': 'Garcia Takes One-Stroke Lead at European Masters CRANS-SUR-SIERRE, Switzerland (Reuters) - Miguel Angel  Jimenez offered a good-natured rebuke to his young compatriot  Sergio Garcia Friday for trying to persuade him to rest before  the Ryder Cup.', 'label': '1'},
]

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

IMDB_LABEL_MAP = {
    '0': 'positive',
    '1': 'negative',
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


def get_prompted_text_data(args):
    dataset_dict = get_data(args.data_dir, args.task_name)

    dataset_ = dataset_dict[args.ds_set]

    if args.max_samples > 0 :
        print("sampling max_samples:", args.max_samples)
        dataset_o_ = dataset_
        n_total_samples = len(dataset_o_)

        sample_indices = np.random.choice(n_total_samples, args.max_samples, replace=False)
        dataset_ = dataset_.select(sample_indices)

    print(f"preprocessing {args.task_name} set {args.ds_set} (total {len(dataset_)} samples..)...")

    input_texts = []

    for example in dataset_:
        processed_input_text = format_single_example(args.task_name, args.prompt_mode, example['text'])
        input_texts.append(processed_input_text)

    targets = map_example_labels(args.task_name, dataset_['label'])

    print("first 3 samples after prompt formatting:")
    for idx in range(0,3):
        print(f"--idx {idx}:\n --input: {input_texts[idx]}\n --label: {targets[idx]}")

    # import ipdb; ipdb.set_trace()

    return input_texts, targets

def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    return data

def get_data(data_path: str, dataset_name: str ):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train = Dataset.from_list(read_jsonl(os.path.join(data_path, dataset_name, 'train_ft.json')))
    val = Dataset.from_list(read_jsonl(os.path.join(data_path, dataset_name, 'dev_ft.json')))
    test = Dataset.from_list(read_jsonl(os.path.join(data_path, dataset_name, 'test_ft.json')))
    # import ipdb;ipdb.set_trace()
    return DatasetDict({"train": train, "val": val, "test": test})

    



def format_single_example(dataset_name, prompt_mode,  sentence) -> str:
    if dataset_name == "ag_news":
        return format_single_example_ag_news(sentence, prompt_mode)
    elif dataset_name == "yahoo_answers":
        return format_single_example_yahoo_answers(sentence, prompt_mode)
    elif dataset_name == "aclImdb":
        return format_single_example_aclImdb(sentence, prompt_mode)
    elif dataset_name == "yelp_review":
        return format_single_example_yelp_review(sentence, prompt_mode)
    elif dataset_name == "amazon_review":
        return format_single_example_amazon_review(sentence, prompt_mode)
    else:
        raise ValueError(f"dataset_name {dataset_name} is not supported")

def get_ag_news_icl_samples_str(prompt_mode):

    icl_samples_str = ""
    for sample in AG_NEWS_ICL_SAMPLES:
        if prompt_mode in [0,2]:
            sample_str = f"{sample['text']} It is about {AG_NEWS_LABEL_MAP[sample['label']]}\n\n"
        elif prompt_mode in [1,3]:
            sample_str = f"Article: {sample['text']}\n\n"
            sample_str += f"Answer: {AG_NEWS_LABEL_MAP[sample['label']]}\n\n"
        else:
            raise ValueError

        icl_samples_str += sample_str
    
    return icl_samples_str


def format_single_example_ag_news(sentence, prompt_mode) -> str:

    if prompt_mode == 0:
        prompt_str = f"{sentence} It is about "
    elif prompt_mode == 1:
        prompt_str = "Select the topic that the given article is about. The topics are: world, sports, business, technology.\n\n"
        prompt_str += f"Article: {sentence}\n\n"
        prompt_str += "Answer:"
    elif prompt_mode == 2:
        icl_samples_str = get_ag_news_icl_samples_str(prompt_mode)
        prompt_str = icl_samples_str + f"{sentence} It is about "
    elif prompt_mode == 3:
        icl_samples_str = get_ag_news_icl_samples_str(prompt_mode)
        prompt_str = "Select the topic that the given article is about. The topics are: world, sports, business, technology.\n\n"
        prompt_str += icl_samples_str
        prompt_str += f"Article: {sentence}\n\n"
        prompt_str += "Answer:"
    else:
        raise ValueError

    return prompt_str

def map_example_labels(dataset_name, labels):
    targets = []
    for label in labels:
        if dataset_name == "ag_news":
            target = AG_NEWS_LABEL_MAP[label]
        elif dataset_name == "yahoo_answers":
            target = YAHOO_ANSWERS_LABEL_MAP[label]
        elif dataset_name == "aclImdb":
            target = IMDB_LABEL_MAP[label]
        elif dataset_name == "yelp_review":
            target = YELP_REVIEW_LABEL_MAP[label]
        elif dataset_name == "amazon_review":
            target = AMAZON_REVIEW_LABEL_MAP[label]
        else:
            raise ValueError
        
        targets.append(target)

    return targets

def map_example_labels_inverse(dataset_name, str_labels):
    
    if dataset_name == "ag_news":
        label_map =  AG_NEWS_LABEL_MAP
    elif dataset_name == "yahoo_answers":
        label_map =  YAHOO_ANSWERS_LABEL_MAP
    elif dataset_name == "aclImdb":
        label_map =  IMDB_LABEL_MAP
    elif dataset_name == "yelp_review":
        label_map =  YELP_REVIEW_LABEL_MAP
    elif dataset_name == "amazon_review":
        label_map =  AMAZON_REVIEW_LABEL_MAP
    else:
        raise ValueError
    
    # string to int map
    inv_label_map = {label_map[key]:int(key) for key in label_map} 

    targets = []
    for str_label in str_labels:
        target = inv_label_map[str_label]
        targets.append(target)

    return targets

def format_single_example_yahoo_answers(sentence, prompt_mode) -> str:

    if prompt_mode == 0:
        prompt_str = f"{sentence} It is about "
    elif prompt_mode == 1:
        prompt_str = "Select the topic that the given article is about. The topics are: society, science, health, education, computer, sports, business, entertainment, relationship, politics.\n\n"
        prompt_str += f"Article: {sentence}\n\n"
        prompt_str += "Answer:"
    else:
        raise ValueError

    return prompt_str

def format_single_example_aclImdb(sentence, prompt_mode) -> str:

    options_ = 'OPTIONS:\n- negative\n- positive'

    if prompt_mode == 0:
        prompt_str = f"{sentence}\nWhat is the sentiment of this review?\n{options_}"
    elif prompt_mode == 0:
        prompt_str = f"{sentence}\nWould you say this review is positive or negative?\n{options_}"
    else:
        raise ValueError

    return prompt_str

def format_single_example_yelp_review(sentence, prompt_mode) -> str:

    # options_ = 'OPTIONS:\n- negative\n- neutral\n- positive'
    options_ = 'OPTIONS:\n- very negative\n- negative\n- neutral\n- positive\n- very positive'

    if prompt_mode == 0:
        prompt_str = f"{sentence}\nWhat is the sentiment of this review?\n{options_}"
    # elif prompt_mode == 1:
    #     prompt_str = f"{sentence}\nWould you say this review is positive or negative?\n{options_}"
    else:
        raise ValueError

    return prompt_str

def format_single_example_amazon_review(sentence, prompt_mode) -> str:

    # options_ = 'OPTIONS:\n- negative\n- neutral\n- positive'
    options_ = 'OPTIONS:\n- very negative\n- negative\n- neutral\n- positive\n- very positive'

    if prompt_mode == 0:
        prompt_str = f"{sentence}\nWhat is the sentiment of this review?\n{options_}"
    # elif prompt_mode == 1:
    #     prompt_str = f"{sentence}\nWould you say this review is positive or negative?\n{options_}"
    else:
        raise ValueError

    return prompt_str

def my_decoded_prediction(task_name, decoded_predictions, return_not_processed_idx=0):
    if task_name == 'ag_news':
        return map_decoded_prediction_ag_news(decoded_predictions, return_not_processed_idx)
    elif task_name == 'yahoo_answers':
        return map_decoded_prediction_yahoo_answers(decoded_predictions, return_not_processed_idx)
    # elif task_name == 'aclImdb':
    #     return map_decoded_prediction_aclImdb(decoded_predictions)
    elif task_name == 'yelp_review':
        return map_decoded_prediction_review_dataset(decoded_predictions, return_not_processed_idx)
    elif task_name == 'amazon_review':
        return map_decoded_prediction_review_dataset(decoded_predictions, return_not_processed_idx)
    else:
        raise ValueError

def map_decoded_prediction_ag_news(decoded_predictions, return_not_processed_idx = 0):
    
    ag_news_labels = list(AG_NEWS_LABEL_MAP.values())
    tech_label_candidates = ['science']

    decoded_predictions_processed = []
    ct_not_processed = 0
    indices_not_processed = []
    for idx, pred in enumerate(decoded_predictions):
        pred2 = pred.lower().strip()
        pred_label = None

        for label in ag_news_labels:
            if label in pred2:
                pred_label = label

        if pred_label is None:
            for label in tech_label_candidates:
                if label in pred2:
                    pred_label = 'technology'

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            indices_not_processed.append(idx)
            pred_label = 'world'
        
        decoded_predictions_processed.append(pred_label)    

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")
    if return_not_processed_idx:
        return decoded_predictions_processed, indices_not_processed
    else:
        return decoded_predictions_processed

def map_decoded_prediction_yahoo_answers(decoded_predictions, return_not_processed_idx = 0):

    # import ipdb; ipdb.set_trace()
    yahoo_answers_labels = list(YAHOO_ANSWERS_LABEL_MAP.values())
    tech_label_candidates = ['science']

    decoded_predictions_processed = []
    ct_not_processed = 0
    indices_not_processed = []
    for idx, pred in enumerate(decoded_predictions):
        pred2 = pred.lower().strip()
        pred_label = None

        for label in yahoo_answers_labels:
            if label in pred2:
                pred_label = label

        if pred_label is None:
            for label in tech_label_candidates:
                if label in pred2:
                    pred_label = 'science'

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            indices_not_processed.append(idx)
            pred_label = 'society'
        
        decoded_predictions_processed.append(pred_label)    

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")
    if return_not_processed_idx:
        return decoded_predictions_processed, indices_not_processed
    else:
        return decoded_predictions_processed

# def map_decoded_prediction_aclImdb(decoded_predictions):
    
#     possible_labels = list(IMDB_LABEL_MAP.values())

#     decoded_predictions_processed = []
#     ct_not_processed = 0
#     for pred in decoded_predictions:
#         pred2 = pred.lower().strip()
#         pred_label = None

#         for label in possible_labels:
#             if label in pred2:
#                 pred_label = label

#         ## default label if not detected
#         if pred_label is None:
#             ct_not_processed += 1
#             pred_label = 'positive'
        
#         decoded_predictions_processed.append(pred_label)

#     print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

#     return decoded_predictions_processed

# def map_decoded_prediction_yelp_review(decoded_predictions):

#     # import ipdb; ipdb.set_trace()
    
#     possible_labels = list(YELP_REVIEW_LABEL_MAP.values())

#     decoded_predictions_processed = []
#     ct_not_processed = 0
#     for pred in decoded_predictions:
#         pred2 = pred.lower().strip()
#         pred_label = None

#         for label in possible_labels:
#             if label == pred2:
#                 pred_label = label

#         ## default label if not detected
#         if pred_label is None:
#             ct_not_processed += 1
#             pred_label = 'neutral'
        
#         decoded_predictions_processed.append(pred_label)    

#     print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

    # return decoded_predictions_processed

# def map_decoded_prediction_amazon_review(decoded_predictions):

#     # import ipdb; ipdb.set_trace()
    
#     possible_labels = list(AMAZON_REVIEW_LABEL_MAP.values())

#     decoded_predictions_processed = []
#     ct_not_processed = 0
#     for pred in decoded_predictions:
#         pred2 = pred.lower().strip() 
#         # pred2 = 'very negative'
#         pred_label = None

#         for label in possible_labels:
#             # check exact match
#             if label == pred2:
#                 pred_label = label

#         ## default label if not detected
#         if pred_label is None:
#             ct_not_processed += 1
#             pred_label = 'neutral'
        
#         decoded_predictions_processed.append(pred_label)    

#     print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

#     return decoded_predictions_processed

def map_decoded_prediction_review_dataset(decoded_predictions, return_not_processed_idx = 0):

    # '0': 'very negative',
    # '1': 'negative',
    # '2': 'neutral',
    # '3': 'positive',
    # '4': 'very positive',

    # import ipdb; ipdb.set_trace()
    possible_labels = list(AMAZON_REVIEW_LABEL_MAP.values())
    normal_labels = ['negative', 'neutral', 'positive']
    very_labels = ['very negative', 'very positive']

    decoded_predictions_processed = []
    ct_not_processed = 0
    indices_not_processed = []
    for idx, pred in enumerate(decoded_predictions):

        pred2 = pred.lower().strip()
        pred_label = None

        found_idx = {key:None for key in possible_labels}

        if 'very' in pred2:
            for label in very_labels:
                if label in pred2:
                    found_idx[label] = pred2.find(label)
                    pred_label = 1
        
        if pred_label is None:
            for label in normal_labels:
                if label in pred2:
                    found_idx[label] = pred2.find(label)
                    pred_label = 1

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            indices_not_processed.append(idx)
            pred_label = 'neutral'
        else:
            # among found labels, we will use minimum index one. (important claim comes first)
            found_idx2 = {key:found_idx[key] for key in found_idx if found_idx[key] is not None}
            pred_label = min(found_idx2, key=found_idx.get)
        
        # print(pred2)
        # print(found_idx)
        # print(pred_label)

        decoded_predictions_processed.append(pred_label)    

    
    # import ipdb; ipdb.set_trace()

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")
    if return_not_processed_idx:
        return decoded_predictions_processed, indices_not_processed
    else:
        return decoded_predictions_processed