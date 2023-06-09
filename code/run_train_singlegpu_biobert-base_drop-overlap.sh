export CUDA_VISIBLE_DEVICES=0
DATASET="data_process-for-train_k-fold_drop-overlap"
python train.py \
    --seed 111 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --ignore_mismatched_sizes False \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    --weight_decay 0.01 \
    --classifier_lr_alpha 2.5 \
    "$@"

python train.py \
    --seed 111 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --ignore_mismatched_sizes False \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    --weight_decay 0.01 \
    --classifier_lr_alpha 2.5 \
    "$@"

python train.py \
    --seed 111 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --ignore_mismatched_sizes False \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    --weight_decay 0.01 \
    --classifier_lr_alpha 2.5 \
    "$@"

python train.py \
    --seed 111 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --ignore_mismatched_sizes False \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    --weight_decay 0.01 \
    --classifier_lr_alpha 2.5 \
    "$@"

python train.py \
    --seed 111 \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --ignore_mismatched_sizes False \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --metric_for_best_model eval_accuracy \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --do_fgm \
    --weight_decay 0.01 \
    --classifier_lr_alpha 2.5 \
    "$@"