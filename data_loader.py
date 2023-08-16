import os
import spacy
import numpy as np
import pandas as pd
import random
import json
from tqdm import trange
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer


def dictlist2df(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    df = pd.DataFrame.from_dict(dl)
    return df


def dictlist2dict(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    return dl


def get_qa_data(data):
    _data = []
    for d in data:
        _data.append({
                    "context": d['context'],
                    "question": d["qas"][0]["question"],
                    "answers": d["qas"][0]["answers"]
                })
    return _data
    

def get_data_split(dataset, n_train=10e9, n_val=500, tokenizer=None, max_tokens=1024, use_dev_subset=False, return_dict=True):

    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # training data
    train_data_path = f"./data/qa/{dataset}/{dataset}-train.jsonl"
    orig_train_data = []
    with open(train_data_path, 'r') as file:
        for idx, line in enumerate(file):

            # Parse the JSON data in each line
            data = json.loads(line)
            if idx == 0: 
                continue 

            # filter out the long samples
            if tokenizer and max_tokens:
                context = data['context']
                ctx_len = len(tokenizer(context)['input_ids'])
                if ctx_len > max_tokens:
                    continue

            orig_train_data.append(data)

    train_data = get_qa_data(orig_train_data)

    # use the selected 500 validation data (for nq)
    if use_dev_subset:
        val_data_path = f"./data/qa/{dataset}/dev-500.json"
        with open(val_data_path, "r") as file:
            orig_val_data = json.load(file)
    else:
        val_data_path = f"./data/qa/{dataset}/{dataset}-dev.jsonl"
        orig_val_data = []
        with open(val_data_path, 'r') as file:
            for idx, line in enumerate(file):
                data = json.loads(line)
                if idx == 0: 
                    continue 
                orig_val_data.append(data)
    val_data = get_qa_data(orig_val_data) 

    train_data = train_data[:n_train]
    val_data = val_data[:n_val]
    test_data = val_data

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(val_data), dictlist2dict(test_data)
    else:
        return train_data, val_data, test_data
    
    
    
if __name__ == "__main__":

    for dataset in ['NaturalQuestions']:
        for n_train in [100000]:
            train_data, val_data, test_data = get_data_split(dataset=dataset, 
                                                             n_train=n_train, 
                                                             n_val=500)
        
            from datasets import Dataset
            val_dataset = Dataset.from_dict(val_data)
            
            context, question, answer = [], [], []
            print(len(val_dataset))
            for d in val_dataset:
                context.append(len(d['context'].split()))
                question.append(len(d['question'].split()))
                answer.extend([len(a.split()) for a in d['answers']])
            print(np.mean(context))
            print(np.mean(question))
            print(np.mean(answer))










