"""
Data from https://github.com/mrqa/MRQA-Shared-Task-2019
"""

import os
import re
import json
import random
import argparse
import spacy
nlp = spacy.load("en_core_web_sm")

from tqdm import tqdm, trange
from transformers import pipeline, AutoTokenizer
from LLM.llm import OpenAI
from QA.data_loader import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='NaturalQuestions', choices=['TriviaQA', 'SQuAD', 'NaturalQuestions', 'NewsQA', 'SearchQA', 'HotpotQA']) #
    parser.add_argument('--n_samples', type=int, default=500) #

    # requirements
    parser.add_argument('--min_ctx_sents', type=int, default=3) #
    parser.add_argument('--min_ctx_tokens', type=int, default=100) #
    parser.add_argument('--max_ctx_tokens', type=int, default=1024) #

    parser.add_argument("--generate_question", action="store_true", help="whether to generate relevant questions.")
    parser.add_argument("--qg_model", type=str, default="gpt-4", choices=['gpt-4', 't5-base-qg'])
    parser.add_argument("--qg_prompt_path", type=str, default="./prompts/qa_qg.txt")

    args, unknown = parser.parse_known_args()


    """
    Step 1: sample data
    """
    # a subset of dev set with 500 samples
    dev_data_path = f"./data/qa/{args.dataset}/{args.dataset}-dev.jsonl"
    dev_data = []
    unique_context = []
    with open(dev_data_path, 'r') as file:
        for idx, line in enumerate(file):
            # Parse the JSON data in each line
            data = json.loads(line)
            # exclude first line 
            if idx == 0:
                continue 
            dev_data.append(data)
    print("Original data: ", len(dev_data))    
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sampled_dev_data_path = f"./data/qa/{args.dataset}/dev-{args.n_samples}.json"
    if not os.path.exists(sampled_dev_data_path):
        random.shuffle(dev_data)
        sampled_dev_data = []
        for data in dev_data:
            context = data['context']

            if context in unique_context:
                continue

            # select those with only text, no html tag
            if any([t in context for t in ['</Table>', '</Tr>', '</Li>', '</Dd>', '</Dd>']]):
                continue

            ctx_sents = [str(s) for s in nlp(context.replace("<P>","").replace("</P>","")).sents]
            if len(ctx_sents) < args.min_ctx_sents:
                continue
            
            ctx_len = len(tokenizer(context)["input_ids"])

            if ctx_len < args.min_ctx_tokens:
                continue

            if ctx_len > args.max_ctx_tokens:
                continue
            
            sampled_dev_data.append(data)
            unique_context.append(context)

            if len(sampled_dev_data) == args.n_samples:
                break
        
        # save dev data
        with open(sampled_dev_data_path, 'w', encoding='utf-8') as file:
            json.dump(sampled_dev_data, file, ensure_ascii=False)
    else:
        # load data
        with open(sampled_dev_data_path, 'r', encoding='utf-8') as file:
            sampled_dev_data = json.load(file)
    print("Sampled data: ", len(sampled_dev_data))


    """
    Step 2: generate relevant questions/answers for dev set
    """
    if args.generate_question:
        
        relevant_data_path = f"./data/qa/{args.dataset}/dev-{args.n_samples}-{args.qg_model}.json" 
        # load the task data with relevant tasks
        if os.path.exists(relevant_data_path):
            with open(relevant_data_path, "r") as f:
                relevant_dev_data = json.load(f)
        else:
            relevant_dev_data = []
        num_examples = len(relevant_dev_data)
        print(f"Number of existing relevant questions: {num_examples}")

        with open(args.qg_prompt_path, "r") as file:
            qg_prompt = file.read().strip()

        # load model
        if args.qg_model == "gpt-3.5":
            gpt3 = OpenAI(model='gpt-3.5-turbo')
        elif args.qg_model == "gpt-4":
            gpt3 = OpenAI(model='gpt-4')
        elif args.qg_model == "t5-base-qg":
            nlp = pipeline("text2text-generation", model="Zekunli/t5-base-SQuAD-qg-ep10", max_length=64)
        else:
            raise NotImplementedError
        
        # continue generation
        for idx in trange(num_examples, len(sampled_dev_data)):
            data = sampled_dev_data[idx]
            context = data['context']
            qas = data['qas'][0]
            question = qas['question']
            qid = qas['qid']
            answers = qas['answers']
            
            if args.qg_model in ["gpt-3.5", "gpt-4"]:
                prompt = qg_prompt.replace("[[PARAGRAPH]]", context).replace("[[QUESTION]]", question).replace("[[ANSWER]]", answers[0])
                output = gpt3.generate(prompt=prompt,
                                temperature=0.7, 
                                top_p=1.0, 
                                max_tokens=128, 
                                n=1, 
                                frequency_penalty=0, 
                                presence_penalty=0, 
                                stop=["Example", "Question 3"])[0]
                additional_qa = [x.strip() for x in re.split(r"Question \d+:", output)]
                additional_q = [re.split(r"Answer \d+:", x)[0].strip() for x in additional_qa]
                additional_a = [re.split(r"Answer \d+:", x)[1].strip() for x in additional_qa]
                data['relevant_qas'] = [
                    {
                        "question": qa[0],
                        "answers": [qa[1]],
                        "qid": ""
                    }
                    for qa in list(zip(additional_q, additional_a))
                ]

            elif "qag" in args.qg_model:
                prompt = QAG_INPUT_TEMPLATE.format(context=context)
                output = nlp(prompt)[0]['generated_text']
                additional_q, additional_a = parse_question_answer(output)
                print(additional_q, additional_a)
                data['relevant_qas'] = [
                        {
                            "question": additional_q,
                            "answers": [additional_a],
                            "qid": ""
                        }
                    ]
            
            elif "qg" in args.qg_model:
                prompt = QG_INPUT_TEMPLATE.format(context=context)
                additional_q = nlp(prompt)[0]['generated_text']
                additional_a = ""
                print(additional_q, additional_a)
                data['relevant_qas'] = [
                        {
                            "question": additional_q,
                            "answers": [additional_a],
                            "qid": ""
                        }
                    ]
        
            relevant_dev_data.append(data)

            # save every iter
            with open(relevant_data_path, "w", encoding='utf-8') as file:
                json.dump(relevant_dev_data, file, ensure_ascii=False)