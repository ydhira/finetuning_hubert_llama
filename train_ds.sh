deepspeed --num_gpus=8 train.py \
    --output_dir "outputs" \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --dataset_name_or_path "librispeech-hubert-discrete-tokens"\
    --load_from_local True \
    --input_col_name "hubert_discrete_tokens" \
    --output_col_name "text" \
    --max_eval_samples 1000 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 4 \
    --train_split_name "train" \
    --test_split_name "test" \
    --validation_split_name "validation" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --per_device_eval_batch_size  4 \
    --eval_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 100 \
    --evaluation_strategy epoch \
    --eval_steps 5 \
    --num_train_epochs 50 \
    --do_train True \
    --do_eval True \
    --eval_before_training False \
    --report_to none \
    --deepspeed "configs/ds_config.json" \
    --overwrite_output_dir \