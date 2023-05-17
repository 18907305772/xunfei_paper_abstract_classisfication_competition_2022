export CUDA_VISIBLE_DEVICES=0

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold_drop-overlap"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_drop-overlap_fold${FOLD}_k-fold_gpunum8_michiyasunaga_BioLinkBERT-large_bs1_accumulate4_lr2e-05_epoch3.0_SchedulerType.LINEAR-scheduler_fgm_cls"
OUTPUT="result/drop-overlap_fold${FOLD}_michiyasunaga_BioLinkBERT-large_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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
    --predict_top2 True \
    "$@"
done

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold_drop-overlap"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_drop-overlap_fold${FOLD}_k-fold_gpunum1_dmis-lab_biobert-base-cased-v1.2_bs8_accumulate4_lr2e-05_epoch5.0_wd0.01_SchedulerType.LINEAR-scheduler_classifier_grouped_lr_alpha2.5_fgm_cls"
OUTPUT="result/drop-overlap_fold${FOLD}_dmis-lab_biobert-base-cased-v1.2_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
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
    --predict_top2 True \
    "$@"
done

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_fold${FOLD}_k-fold_gpunum8_michiyasunaga_BioLinkBERT-large_bs1_accumulate4_lr2e-05_epoch3.0_SchedulerType.LINEAR-scheduler_fgm_cls"
OUTPUT="result/fold${FOLD}_michiyasunaga_BioLinkBERT-large_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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
    --predict_top2 True \
    "$@"
done

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_fold${FOLD}_k-fold_gpunum1_dmis-lab_biobert-base-cased-v1.2_bs8_accumulate4_lr2e-05_epoch5.0_wd0.01_SchedulerType.LINEAR-scheduler_classifier_grouped_lr_alpha2.5_fgm_cls"
OUTPUT="result/fold${FOLD}_dmis-lab_biobert-base-cased-v1.2_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
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
    --predict_top2 True \
    "$@"
done

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_fold${FOLD}_k-fold_gpunum1_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_bs8_accumulate4_lr2e-05_epoch5.0_wd0.01_SchedulerType.LINEAR-scheduler_classifier_grouped_lr_alpha2.5_fgm_cls"
OUTPUT="result/fold${FOLD}_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
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
    --predict_top2 True \
    "$@"
done

for FOLD in 0 1 2 3 4
do
DATASET="data_process-for-train_k-fold"
MODEL_NAME="result/baseline_data_process-for-train_k-fold_fold${FOLD}_k-fold_gpunum1_microsoft_deberta-base_bs4_accumulate8_lr2e-05_epoch5.0_wd0.01_SchedulerType.LINEAR-scheduler_fgm_cls"
OUTPUT="result/fold${FOLD}_microsoft_deberta-base_predict-top2/"
python predict.py \
    --seed 111 \
    --model_name_or_path ${MODEL_NAME} \
    --ignore_mismatched_sizes False \
    --do_predict \
    --train_file ../user_data/${DATASET}/fold${FOLD}/train.csv \
    --validation_file ../user_data/${DATASET}/fold${FOLD}/valid.csv \
    --test_file ../user_data/${DATASET}/test.csv \
    --use_fast_tokenizer False \
    --pad_to_max_length True \
    --output_dir ${OUTPUT} \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --predict_top2 True \
    "$@"
done