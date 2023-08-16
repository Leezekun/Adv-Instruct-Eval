import os
import json
import numpy as np
import random
import argparse
from tqdm import tqdm, trange

from utils import *
from llm_configs import llm_configs


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
    parser.add_argument('--qg_model', type=str, default='gpt-4', choices=['gpt-3.5', 't5-base-qg', 'gpt-4']) #
    
    parser.add_argument("--test_mode", type=str, required=True, help="whether to inject tasks.")

    # arguments for LLM generation
    parser.add_argument('--model', type=str, default='vicuna-13b') #
    parser.add_argument('--inst_path', type=str, default='./prompts/qa_instruction.txt') #
    parser.add_argument('--demo_path', type=str, default='./prompts/qa_demo.json') #    
    parser.add_argument('--eval_path', type=str, default='./prompts/qa_eval.txt') #
    parser.add_argument('--n_shot', type=int, default=4, choices=[0,1,2,3,4,5]) #

    parser.add_argument('--temperature', type=float, default=0.5) #
    parser.add_argument('--top_p', type=float, default=1.0) #
    parser.add_argument('--max_tokens', type=int, default=64) #
    parser.add_argument('--n_seqs', type=int, default=1) #

    args, unknown = parser.parse_known_args()
    
    with open(args.inst_path, "r") as file:
        llm_inst = file.read().strip()
    with open(args.eval_path, "r") as file:
        eval_prompt = file.read().strip()
    with open(args.demo_path, "r") as file:
        demo = json.load(file)

    assert args.model in llm_configs

    # load and save existing data
    if args.test_mode == "original+injected":
        if args.task_type == "irrelevant":
            eval_data_path = f"./injections/injected-{args.position}-{args.prefix_type}-{args.task_type}-{args.dataset}-{args.split}-{args.n_samples}.json"    
            eval_result_path = f"./evaluations/eval-{args.model}-{args.n_shot}-injected-{args.position}-{args.prefix_type}-{args.task_type}-{args.dataset}-{args.split}-{args.n_samples}.json"    
        else:
            eval_data_path = f"./injections/injected-{args.position}-{args.prefix_type}-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json"    
            eval_result_path = f"./evaluations/eval-{args.model}-{args.n_shot}-injected-{args.position}-{args.prefix_type}-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json"    
    elif args.test_mode == "original":
        eval_data_path = f"./injections/original-{args.dataset}-{args.split}-{args.n_samples}.json"    
        eval_result_path = f"./evaluations/eval-{args.model}-{args.n_shot}-original-{args.dataset}-{args.split}-{args.n_samples}.json"    
    elif args.test_mode == "injected":
        eval_data_path = f"./injections/injected-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json"    
        eval_result_path = f"./evaluations/eval-{args.model}-{args.n_shot}-injected-{args.qg_model}-{args.dataset}-{args.split}-{args.n_samples}.json"    

    if os.path.exists(eval_result_path):
        with open(eval_result_path, "r") as file:
            eval_results = json.load(file)

        a, b, c, d, total = 0, 0, 0, 0, 0
        orig_em, orig_f1 = 0., 0.
        inject_em, inject_f1 = 0., 0.
        for result in eval_results:
            if 'label' in result:
                if result['label'] == 1:
                    a += 1
                elif result['label'] == 2:
                    b += 1
                elif result['label'] == 3:
                    c += 1
                elif result['label'] == 4:
                    d += 1
            if 'exact_match' in result and 'f1' in result:
                orig_em += result['exact_match']
                orig_f1 += result['f1']
            if 'inject_exact_match' in result and 'inject_f1' in result:
                inject_em += result['inject_exact_match']
                inject_f1 += result['inject_f1']
            total += 1

    else:
        eval_results = []
        a, b, c, d, total = 0, 0, 0, 0, 0
        orig_em, orig_f1 = 0., 0.
        inject_em, inject_f1 = 0., 0.

    # continue generation and evaluation
    if total < args.n_samples:
        
        # load data
        with open(eval_data_path, 'r') as file:
            eval_data = json.load(file)

        # load model
        chatbot = Chatbot(model=args.model,
                            roles={"user": "Question", "assistant": "Answer", "system": "System"},
                            system_instruction=llm_inst,
                            demo=demo,
                            n_shot=args.n_shot,
                            )
    
        for idx in trange(total, len(eval_data)):
            data = eval_data[idx]

            orig_question = data['question']
            orig_answers = data['answers']

            injected_question = data['injected_question']
            injected_answers = data['injected_answers']  

            prompt = data['prompt']

            # chatbot generation
            generation = chatbot.generate(input=prompt)[0]
            data['generation'] = generation
            
            total += 1
            # eval
            orig_evaluation = evaluate_qa(generation, orig_answers)
            data['original/exact_match'] = orig_evaluation['exact_match']
            data['original/f1'] = orig_evaluation['f1']
            orig_em += orig_evaluation['exact_match']
            orig_f1 += orig_evaluation['f1']

            inject_evaluation = evaluate_qa(generation, injected_answers)
            data['injected/exact_match'] = inject_evaluation['exact_match']
            data['injected/f1'] = inject_evaluation['f1']
            inject_em += inject_evaluation['exact_match']
            inject_f1 += inject_evaluation['f1']

            print(f"Original task: EM: {orig_em/total}, F1: {orig_f1/total}")
            print(f"Injected task: EM: {inject_em/total}, F1: {inject_f1/total}")

            eval_results.append(data)
            
            # save the evaluation results every time
            with open(eval_result_path, 'w', encoding='utf-8') as file:
                json.dump(eval_results, file, ensure_ascii=False)

    """
    rerun the evaluation on the whole set again
    """
    total = len(eval_results)
    orig_em, orig_f1 = 0., 0.
    inject_em, inject_f1 = 0., 0.
    a, b, c, d = 0., 0., 0., 0.
    for data in eval_results:
        generation = data['generation']
        orig_question = data['question']
        orig_answers = data['answers']
        injected_question = data['injected_question']
        injected_answers = data['injected_answers']
        # original task
        orig_evaluation = evaluate_qa(generation, orig_answers)
        data['original/exact_match'] = orig_evaluation['exact_match']
        data['original/f1'] = orig_evaluation['f1']
        orig_em += orig_evaluation['exact_match']
        orig_f1 += orig_evaluation['f1']
        # injected task
        inject_evaluation = evaluate_qa(generation, injected_answers)
        data['injected/exact_match'] = inject_evaluation['exact_match']
        data['injected/f1'] = inject_evaluation['f1']
        inject_em += inject_evaluation['exact_match']
        inject_f1 += inject_evaluation['f1']
        # adcd evaluation
        if "label" in data:
            if data["label"] == 1:
                a += 1
            elif data["label"] == 2:
                b += 1
            elif data["label"] == 3:
                c += 1
            elif data["label"] == 4:
                d += 1

    # print the total results
    print(args)
    print(f"Total: {total}")
    print(f"Original task: EM: {orig_em/total}, F1: {orig_f1/total}")
    print(f"Injected task: EM: {inject_em/total}, F1: {inject_f1/total}") 
    print(f"A (correct): {a}, B: {b}, C: {c}, D: {d}")

    # save the total results
    with open(eval_result_path, 'w', encoding='utf-8') as file:
        json.dump(eval_results, file, ensure_ascii=False)
    
