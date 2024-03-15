# # torchrun --standalone --nnodes=1 --nproc-per-node=1 
# export CUDA_VISIBLE_DEVICE=0
# python train.py \

export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
    --dataset_name_or_path "librispeech-hubert-discrete-tokens" \
    --output_dir "outputs" \
    --load_from_local True \
    --input_col_name "hubert_discrete_tokens" \
    --output_col_name "text" \
    --max_eval_samples 1000 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 4 \
    --train_split_name "train" \
    --test_split_name "test" \
    --validation_split_name "validation" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size  4 \
    --eval_accumulation_steps 8 \
    --num_train_epochs 50 \
    --report_to none \
    --do_train True \
    --do_eval True \
    --eval_before_training False \
    --overwrite_output_dir

