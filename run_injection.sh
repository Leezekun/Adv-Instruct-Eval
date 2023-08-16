for position in start middle end random
do
    for prefix_type in direct ignore
    do
        for task_type in relevant # irrelevant relevant
        do
            for qg_model in gpt-4
            do
                for test_mode in original+injected original injected
                do
                    for dataset in TriviaQA
                    do
                        CUDA_VISIBLE_DEVICES=$devices python -m run_injection \
                                                                --dataset $dataset  \
                                                                --split dev \
                                                                --n_samples 500 \
                                                                --position $position \
                                                                --prefix_type $prefix_type \
                                                                --task_type $task_type \
                                                                --qg_model $qg_model \
                                                                --test_mode $test_mode
                    done
                done
            done                                   
        done
    done
done