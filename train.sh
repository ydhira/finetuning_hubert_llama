# torchrun --standalone --nnodes=1 --nproc-per-node=8
torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py \
    --model_name_or_path "outputs2/TinyLlama-1.1B-intermediate-step-1431k-3T/checkpoint-2000"\
    --dataset_name_or_path "librispeech-hubert-discrete-tokens"\
    --output_dir "outputs2" \
    --load_from_local True \
    --input_col_name "hubert_discrete_tokens" \
    --output_col_name "text" \
    --max_train_samples 1000 \
    --max_eval_samples 1000 \
    --max_test_samples 1000 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 4 \
    --train_split_name "train" \
    --test_split_name "test" \
    --validation_split_name "validation" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size  4 \
    --eval_accumulation_steps 4 \
    --num_train_epochs 5 \
    --report_to none \
    --predict_with_generate \

