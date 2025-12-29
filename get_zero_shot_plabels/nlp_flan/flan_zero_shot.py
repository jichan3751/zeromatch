import json
import argparse
import os

import pandas as pd
import numpy as np 
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


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


def preprocess_function(examples, tokenizer, max_seq_length: int, prompt_mode: int, dataset_name: str):

    """Format the examples and then tokenize them. """
    # inputs = [format_single_example(s1, s2, prompt) for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    # targets = examples['gold_label']
    inputs = [format_single_example(dataset_name, prompt_mode, example['text']) for example in examples]
    targets = map_example_labels(dataset_name, examples['label'])

    print("first 3 samples after prompt formatting:")
    for idx in range(0,3):
        print(f"idx {idx}:\n input: {inputs[idx]}\n label: {targets[idx]}")

    # import ipdb; ipdb.set_trace()

    model_inputs = tokenizer(inputs, text_target=targets) #, max_length=max_seq_length, truncation=True)
    
    # for idx in range(100):
    #     print(f"idx {idx}: len {len(model_inputs['input_ids'][idx])} label id {model_inputs['labels'][idx]} label {targets[idx]}")

    seq_lengths = np.array([len(model_inputs['input_ids'][idx]) for idx in range(len(inputs))])
    print(f"tokenized samples sequence length larger than 511 : {(seq_lengths > 511).sum()} / {len(inputs)}")

    # import ipdb; ipdb.set_trace()
    return model_inputs


def get_tokenized_data(dataset_dict: DatasetDict, tokenizer, max_seq_length: int, prompt_mode: str, dataset_name:str, dataset_set:str):
    """Tokenize stuff. """

    ## for debugging
    # processed_item = preprocess_function(dataset_dict['train'], tokenizer, max_seq_length, prompt_mode, dataset_name)
    # import ipdb; ipdb.set_trace()
    ###

    tokenized_dict = {}
    # for name, dataset_ in dataset_dict.items():
    #     print(f"preprocessing {name}...")
    #     processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, prompt_mode, dataset_name)
    #     tokenized_dict[name] = Dataset.from_dict(processed_item)
    
    name = dataset_set
    dataset_ = dataset_dict[dataset_set]
    print(f"preprocessing {name} set...")
    processed_item = preprocess_function(dataset_, tokenizer, max_seq_length, prompt_mode, dataset_name)
    tokenized_dict[name] = Dataset.from_dict(processed_item)
        
    return DatasetDict(tokenized_dict)


def read_jsonl(file_path: str):
    """Read the given JSONL file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    return data

def get_data(data_path: str, dataset_name: str ):
    """Get the mednli data. """
    # mli_dev_v1.jsonl  mli_test_v1.jsonl  mli_train_v1.jsonl
    train = Dataset.from_list(read_jsonl(os.path.join(data_path,dataset_name, 'train_ft.json')))
    val = Dataset.from_list(read_jsonl(os.path.join(data_path, dataset_name, 'dev_ft.json')))
    test = Dataset.from_list(read_jsonl(os.path.join(data_path, dataset_name, 'test_ft.json')))
    return DatasetDict({"train": train, "val": val, "test": test})


def my_decoded_prediction(task_name, decoded_predictions):
    if task_name == 'ag_news':
        return map_decoded_prediction_ag_news(decoded_predictions)
    elif task_name == 'yahoo_answers':
        return map_decoded_prediction_yahoo_answers(decoded_predictions)
    elif task_name == 'aclImdb':
        return map_decoded_prediction_aclImdb(decoded_predictions)
    elif task_name == 'yelp_review':
        return map_decoded_prediction_yelp_review(decoded_predictions)
    elif task_name == 'amazon_review':
        return map_decoded_prediction_amazon_review(decoded_predictions)
    else:
        raise ValueError

def map_decoded_prediction_ag_news(decoded_predictions):
    
    ag_news_labels = list(AG_NEWS_LABEL_MAP.values())
    tech_label_candidates = ['science']

    decoded_predictions_processed = []
    for pred in decoded_predictions:
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
            pred_label = 'society'
        
        decoded_predictions_processed.append(pred_label)    

    return decoded_predictions_processed

def map_decoded_prediction_yahoo_answers(decoded_predictions):

    # import ipdb; ipdb.set_trace()
    yahoo_answers_labels = list(YAHOO_ANSWERS_LABEL_MAP.values())
    tech_label_candidates = ['science']

    decoded_predictions_processed = []
    for pred in decoded_predictions:
        pred2 = pred.lower().strip()
        pred_label = None

        for label in yahoo_answers_labels:
            if label in pred2:
                pred_label = label

        if pred_label is None:
            for label in tech_label_candidates:
                if label in pred2:
                    pred_label = 'technology'

        ## default label if not detected
        if pred_label is None:
            pred_label = 'world'
        
        decoded_predictions_processed.append(pred_label)    

    return decoded_predictions_processed

def map_decoded_prediction_aclImdb(decoded_predictions):
    
    possible_labels = list(IMDB_LABEL_MAP.values())

    decoded_predictions_processed = []
    ct_not_processed = 0
    for pred in decoded_predictions:
        pred2 = pred.lower().strip()
        pred_label = None

        for label in possible_labels:
            if label in pred2:
                pred_label = label

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            pred_label = 'positive'
        
        decoded_predictions_processed.append(pred_label)

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

    return decoded_predictions_processed

def map_decoded_prediction_yelp_review(decoded_predictions):

    # import ipdb; ipdb.set_trace()
    
    possible_labels = list(YELP_REVIEW_LABEL_MAP.values())

    decoded_predictions_processed = []
    ct_not_processed = 0
    for pred in decoded_predictions:
        pred2 = pred.lower().strip()
        pred_label = None

        for label in possible_labels:
            if label == pred2:
                pred_label = label

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            pred_label = 'neutral'
        
        decoded_predictions_processed.append(pred_label)    

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

    return decoded_predictions_processed

def map_decoded_prediction_amazon_review(decoded_predictions):

    # import ipdb; ipdb.set_trace()
    
    possible_labels = list(AMAZON_REVIEW_LABEL_MAP.values())

    decoded_predictions_processed = []
    ct_not_processed = 0
    for pred in decoded_predictions:
        pred2 = pred.lower().strip() 
        # pred2 = 'very negative'
        pred_label = None

        for label in possible_labels:
            # check exact match
            if label == pred2:
                pred_label = label

        ## default label if not detected
        if pred_label is None:
            ct_not_processed += 1
            pred_label = 'neutral'
        
        decoded_predictions_processed.append(pred_label)    

    print(f"map_decoded_prediction: not processed: {ct_not_processed} / {len(decoded_predictions)}")

    return decoded_predictions_processed


def my_compute_metrics(task_name, predictions, return_decoded=False):
    """Given some predictions, calculate the F1. """
    predictions.label_ids[predictions.label_ids == -100] = 0
    
    # Decode the predictions + labels 
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

    decoded_predictions_processed = my_decoded_prediction(task_name, decoded_predictions)
    
    ret = {
            'f1': f1_score(decoded_predictions_processed, decoded_labels, average='macro'),
            'accuracy': accuracy_score(decoded_predictions_processed, decoded_labels)
           }
    
    if return_decoded:
        ret.update({
            'decoded_predictions': decoded_predictions,
            'decoded_predictions_processed': decoded_predictions_processed,
            'decoded_labels': decoded_labels
        })

    return ret

import math
import time
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.trainer_utils import (
    EvalLoopOutput, 
    PredictionOutput, 
    speed_metrics,
    EvalPrediction,
    has_length,
    denumpify_detensorize,
)

from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, model_args, **kwargs):
        super().__init__(**kwargs)
        self.model_args = model_args

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only


        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        print(f"***** Running {description} *****")
        if has_length(dataloader):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
            assert 0
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.
        n_steps = len(dataloader)

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.gather_function((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            # if step == 99:
            #     import ipdb; ipdb.set_trace()
            

        
        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(dataloader):
            num_samples = self.num_examples(dataloader)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)




def test_model(args, 
                model,
                tokenizer,
                output_dir: str,                            
                tokenized_data, 
                seed: int,
                local_rank: int):

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=False,
            do_eval=False,
            do_predict=True,
            local_rank=local_rank,
            per_device_eval_batch_size=1,
            predict_with_generate=True,
            overwrite_output_dir=True,
            report_to="none",
            seed=seed,
    )   

    compute_metrics_fn = lambda predictions, return_decoded=False: my_compute_metrics(args.dataset, predictions, return_decoded)

    print("Using default trainer..")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data[args.set],
        eval_dataset=tokenized_data[args.set],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )



    # if args.force_words:
    #     # allowed_words = [tokenizer.encode(x) for x in get_mapping(args.prompt)]
    #     # outputs = trainer.predict(tokenized_data["val"], force_words_ids=allowed_words, num_beams=3)
    #     raise ValueError
    # else:
        # tmp = tokenized_data["val"].shuffle()
        
        # outputs = trainer.predict(Dataset.from_dict(tmp[:200]))
        # outputs = trainer.predict(tokenized_data['test'])

    if args.is_test:
        print("Option --is_test detected. use part of the val set..")
        tmp = tokenized_data[args.set].shuffle()
        predict_dataset = Dataset.from_dict(tmp[:200])
    else:
        # predict_dataset = tokenized_data["train"]
        # predict_dataset = tokenized_data["val"]
        # predict_dataset = tokenized_data["test"]
        predict_dataset = tokenized_data[args.set]

    outputs = trainer.predict(predict_dataset)

    print(outputs.metrics)
    outputs2 = compute_metrics_fn(outputs, return_decoded=True)

    print("first 3 of decoded predictions / processed:")
    for idx in range(3):
        print(f"idx {idx}:\n pred: {outputs2['decoded_predictions'][idx]}, processed:{outputs2['decoded_predictions_processed'][idx]}  ")

    print("confusion matrix:")
    conf_mat = confusion_matrix(outputs2['decoded_labels'],outputs2['decoded_predictions_processed'])
    print(conf_mat)
    np.savetxt(os.path.join(output_dir,"conf_mat.txt"), conf_mat, delimiter=",",fmt="%d")

    with open(output_dir + '/results.json', 'w') as f:
        # outputs = {
        #             'label_ids': outputs.label_ids.tolist(),
        #             'metrics': outputs.metrics,
        #             'predictions': outputs.predictions.tolist()
        #           }
        tmp_d = {
            'f1': outputs2['f1'],
            'accuracy': outputs2['accuracy']
            }
        json.dump(tmp_d, f, indent=4)

    # print(tmp_d)
    
    # with open(output_dir + '/plabel_results.json', 'w') as f:
    #     json.dump(outputs2, f, indent=4)
    fname = os.path.join(output_dir,"plabel_results.pt")
    torch.save(outputs2, fname)
    print('saved to', fname)

    # import ipdb; ipdb.set_trace()

    print(f"max cuda memory used: {int(torch.cuda.max_memory_allocated()/1048576)}MB")

## utils
    
def get_num_gpus():
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ or \
        os.environ["CUDA_VISIBLE_DEVICES"] == "":
        
        # CUDA_VISIBLE_DEVICES not set, forcing CPU mode and num_gpu=1 for batch size ...
        num_gpus=1
    else:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        num_gpus=len(gpu_ids)

    return num_gpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--set', type=str, default='val', choices=['train','val','test'])
    parser.add_argument('--model_path', type=str, default="google/flan-t5-xxl")
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--prompt_mode', type=int)
    parser.add_argument('--is_test',action='store_true')

    # parser.add_argument('--force-words', action='store_true') #
    
    args = parser.parse_args()
    print(f"Running with {args.seed}")


    # import ipdb; ipdb.set_trace()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    num_gpus = get_num_gpus()
    if num_gpus > 1:
        print(f"multi-gpu detected: {num_gpus}, using device_map = auto option for model")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
 
    # Get data and use the tokenizer on the data 
    dataset_dict = get_data(args.data_dir, args.dataset)

    if args.is_test:
        print("Option --is_test detected. setting val set as train and test...")
        del dataset_dict['train']
        del dataset_dict['test']
        dataset_dict['train'] = dataset_dict['val']
        dataset_dict['test'] = dataset_dict['val']

    tokenized_datasets = get_tokenized_data(dataset_dict, tokenizer, args.max_seq_length, args.prompt_mode, args.dataset, args.set)

    # import ipdb; ipdb.set_trace()

    # Inference on the model
    test_model(args, model, tokenizer, args.output_dir, tokenized_datasets, args.seed, args.local_rank)  
