#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATASET="data_process-for-train_k-fold_drop-overlap"
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --model_name_or_path michiyasunaga/BioLinkBERT-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split k-fold \
    --training_data_clean False \
    --train_file ../user_data/${DATASET}/fold0/train.csv \
    --validation_file ../user_data/${DATASET}/fold0/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 512 \
    --gpu_nums ${NUM_GPU} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    "$@"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --model_name_or_path michiyasunaga/BioLinkBERT-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split k-fold \
    --training_data_clean False \
    --train_file ../user_data/${DATASET}/fold1/train.csv \
    --validation_file ../user_data/${DATASET}/fold1/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 512 \
    --gpu_nums ${NUM_GPU} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    "$@"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --model_name_or_path michiyasunaga/BioLinkBERT-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split k-fold \
    --training_data_clean False \
    --train_file ../user_data/${DATASET}/fold2/train.csv \
    --validation_file ../user_data/${DATASET}/fold2/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 512 \
    --gpu_nums ${NUM_GPU} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    "$@"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --model_name_or_path michiyasunaga/BioLinkBERT-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split k-fold \
    --training_data_clean False \
    --train_file ../user_data/${DATASET}/fold3/train.csv \
    --validation_file ../user_data/${DATASET}/fold3/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 512 \
    --gpu_nums ${NUM_GPU} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    "$@"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --seed 111 \
    --model_name_or_path michiyasunaga/BioLinkBERT-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_valid_split k-fold \
    --training_data_clean False \
    --train_file ../user_data/${DATASET}/fold4/train.csv \
    --validation_file ../user_data/${DATASET}/fold4/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir result/baseline_ \
    --max_seq_length 512 \
    --gpu_nums ${NUM_GPU} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    "$@"
