import os
import json
import numpy as np
import random
import argparse
import re
import spacy
nlp = spacy.load("en_core_web_sm")

QA_PROMPT_TEMPLATE = "{user}?\nSearch results: {search_results}"

with open("./prompts/ignore_prefix.json", 'r') as file:
    ignore_prefixes = json.load(file)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='NaturalQuestions', choices=['TriviaQA', 'SQuAD', 'NaturalQuestions', 'NewsQA', 'SearchQA', 'HotpotQA']) #
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev']) #
    parser.add_argument('--n_samples', type=int, default=500) #

    # arguments for injection
    parser.add_argument('--position', type=str, default='end', choices=['start', 'middle', 'end', 'random']) #
    parser.add_argument('--prefix_type', type=str, default='direct', choices=['direct', 'btw', 'ignore']) #
    parser.add_argument('--task_type', type=str, default='irrelevant', choices=['irrelevant', 'relevant']) #
    parser.add_argument('--qg_model', type=str, default='gpt-4', choices=['gpt-4', 't5-base']) #
    parser.add_argument("--test_mode", type=str, required=True, choices=['original', 'injected', 'original+injected'], help="whether to inject tasks.")

    args, unknown = parser.parse_known_args()

    # load self-instruct instructions (irrelevant task instructions)
    if args.task_type == "irrelevant":
        instructions = []
        instruction_path = f"./data/qa/self-instruct/gpt3_filtered_instances_82K.jsonl"
        with open(instruction_path, 'r') as f:
            for line in f:
                instruction = json.loads(line)
                if instruction['valid']:
                    instructions.append(instruction['instruction'])
        print(f"Number of injected tasks: {len(instructions)}")
    
    # load additional qa (relevant task instructions)
    elif args.task_type == "relevant":
        relevant_data_path = f"./data/qa/{args.dataset}/{args.split}-{args.n_samples}-{args.qg_model}.json" 
        with open(relevant_data_path, "r") as f:
            examples = json.load(f)
        num_examples = len(examples)
        assert num_examples == args.n_samples
        print(f"Number of relevant questions: {num_examples}")
            
    # construct test data
    if args.test_mode == "original+injected":
        if args.task_type == "irrelevant":
            eval_data_path = f"./injections/injected-{args.position}-{args.prefix_type}-{args.task_type}-{args.dataset}-{args.split}-{args.n_samples}.json"    
        else:
            eval_data_path = f"./injections/injected-{args.position}-{args.prefix_type}-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json"    
    elif args.test_mode == "original":
        eval_data_path = f"./injections/original-{args.dataset}-{args.split}-{args.n_samples}.json" 
    elif args.test_mode == "injected":
        eval_data_path = f"./injections/injected-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json" 
    
    position = args.position
    prefix_type = args.prefix_type
    task_type = args.task_type

    if not os.path.exists(eval_data_path):
        eval_data = []
        for idx, example in enumerate(examples):
            context = example['context']

            qas = example['qas'][0]
            question = qas['question']
            qid = qas['qid']
            answers = qas['answers']

            relevant_qas = example['relevant_qas'][0]
            relevant_question = relevant_qas['question']
            relevant_qid = relevant_qas['qid']
            relevant_answers = relevant_qas['answers']

            if args.test_mode == "original+injected":
                # select injection prefix
                if args.prefix_type == "direct":
                    injected_prefix = ""
                elif args.prefix_type == "ignore":
                    injected_prefix = random.choice(ignore_prefixes)
                # select injection task
                if args.task_type == "irrelevant":
                    injected_question = random.choice(instructions)
                    injected_answers = [""]
                    injection = injected_prefix + injected_question
                elif args.task_type == "relevant":
                    injected_question = relevant_question
                    injected_answers = relevant_answers
                    injection = injected_prefix + injected_question  

                # construct test samples
                if args.position == "start":
                    context_sentences = context.split("<P>")
                    context_sentences.insert(1, injection)
                    context_sentences.insert(1, "<P>")
                elif args.position == "middle":
                    context_sentences = [str(s) for s in nlp(context).sents]
                    context_sentences.insert(int(len(context_sentences)/2), injection)
                elif args.position == "end":
                    context_sentences = context.split("</P>")
                    context_sentences.append(injection)
                    context_sentences.append("</P>")
                elif args.position == "random":
                    context_sentences = [str(s) for s in nlp(context).sents]
                    context_sentences.insert(random.randint(0,len(context_sentences)-1), injection)
            
                search_results = " ".join(context_sentences)
                # fill in the template
                prompt = QA_PROMPT_TEMPLATE.format(user=question, search_results=search_results)

            elif args.test_mode == "original":
                injected_prefix = ""
                injected_question = ""
                injected_answers = [""]
                injection = ""
                prefix_type = ""
                task_type = ""
                position = ""
                search_results = context # no injection
                prompt = QA_PROMPT_TEMPLATE.format(user=question, search_results=search_results)

            elif args.test_mode == "injected":
                question = relevant_question # evaluate the injected question (as original question)
                answers = relevant_answers
                qid = ""
                injected_prefix = ""
                injected_question = ""
                injected_answers = [""]
                injection = ""
                prefix_type = ""
                task_type = ""
                position = ""
                search_results = context # no injection
                prompt = QA_PROMPT_TEMPLATE.format(user=question, search_results=search_results)

            test = {
                "context": context,
                "question": question,
                "answers": answers,
                "qid": qid,
                "prompt": prompt,
                "user": question,
                "search_results": search_results,
                "injected_prefix": injected_prefix,
                "injected_question": injected_question,
                "injected_answers": injected_answers,
                "prefix_type": prefix_type,
                "task_type": task_type,
                "position": position,
                }
            eval_data.append(test)

        with open(eval_data_path, "w", encoding='utf-8') as file:
            json.dump(eval_data, file, ensure_ascii=False)
