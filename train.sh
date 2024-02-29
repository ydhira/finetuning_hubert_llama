# torchrun --standalone --nnodes=1 --nproc-per-node=8
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"\
    --dataset_name_or_path "librispeech-hubert-discrete-tokens"\
    --output_dir "outputs" \
    --load_from_local True \
    --input_col_name "hubert_discrete_tokens" \
    --output_col_name "text" \
    --train_split_name "train" \
    --test_split_name "test" \
    --validation_split_name "validation" \
    --per_device_eval_batch_size 16 \
    --per_device_eval_batch_size  16 \
    --report_to none \

