export OPENAI_API_KEY='xxx'
export COHERE_API_KEY='xxx'
export ANTHROPIC_API_KEY='xxx'
export HF_TRANSFORMER_CACHE_PATH='xxx'

devices=1,2,3,4

for n_shot in 5
do
    for position in end
    do
        for prefix_type in direct # none direct ignore
        do
            for task_type in relevant # irrelevant relevant
            do
                for model in llama-13b # llama-2-13b-chat llama-2-7b llama-2-13b # claude-1 gpt-3.5
                do
                    for test_mode in original+injected # injected original
                    do
                        for dataset in NaturalQuestions # TriviaQA 
                        do
                            CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
                                                                --dataset $dataset \
                                                                --split dev \
                                                                --n_samples 500 \
                                                                --position $position \
                                                                --prefix_type $prefix_type \
                                                                --task_type $task_type \
                                                                --qg_model gpt-4 \
                                                                --model $model \
                                                                --n_shot $n_shot \
                                                                --test_mode $test_mode
                        done
                    done
                done
            done
        done
    done
done


# # ablation on position
# for n_shot in 4
# do
#     for position in end start middle random
#     do
#         for prefix_type in direct # none direct ignore
#         do
#             for task_type in relevant # irrelevant relevant
#             do
#                 for model in gpt-3.5
#                 do
#                     for test_mode in original+injected
#                     do
#                         CUDA_VISIBLE_DEVICES=$devices python -m sft4lms.QA.run_eval \
#                                                             --dataset NaturalQuestions \
#                                                             --split dev \
#                                                             --n_samples 500 \
#                                                             --position $position \
#                                                             --prefix_type $prefix_type \
#                                                             --task_type $task_type \
#                                                             --qg_model gpt-4 \
#                                                             --model $model \
#                                                             --n_shot $n_shot \
#                                                             --test_mode $test_mode
#                     done
#                 done
#             done
#         done
#     done
# done


# # ablation on prefix
# for n_shot in 4
# do
#     for position in end
#     do
#         for prefix_type in direct ignore
#         do
#             for task_type in relevant # irrelevant relevant
#             do
#                 for model in gpt-3
#                 do
#                     for test_mode in original+injected
#                     do
#                         CUDA_VISIBLE_DEVICES=$devices python -m sft4lms.QA.run_eval \
#                                                             --dataset NaturalQuestions \
#                                                             --split dev \
#                                                             --n_samples 500 \
#                                                             --position $position \
#                                                             --prefix_type $prefix_type \
#                                                             --task_type $task_type \
#                                                             --qg_model gpt-4 \
#                                                             --model $model \
#                                                             --n_shot $n_shot \
#                                                             --test_mode $test_mode
#                     done
#                 done
#             done
#         done
#     done
# done

# # ablation on n_shot
# for n_shot in 1 2 3 4 5
# do
#     for position in end
#     do
#         for prefix_type in direct
#         do
#             for task_type in relevant # irrelevant relevant
#             do
#                 for model in gpt-3.5
#                 do
#                     for test_mode in original+injected
#                     do
#                         CUDA_VISIBLE_DEVICES=$devices python -m sft4lms.QA.run_eval \
#                                                             --dataset NaturalQuestions \
#                                                             --split dev \
#                                                             --n_samples 500 \
#                                                             --position $position \
#                                                             --prefix_type $prefix_type \
#                                                             --task_type $task_type \
#                                                             --qg_model gpt-4 \
#                                                             --model $model \
#                                                             --n_shot $n_shot \
#                                                             --test_mode $test_mode
#                     done
#                 done
#             done
#         done
#     done
# done