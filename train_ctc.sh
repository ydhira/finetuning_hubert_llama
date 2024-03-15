export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7

# torchrun --standalone --nnodes=1 --nproc-per-node=8 run_speech_recognition_ctc.py \
#         --dataset_name="macabdul9/librispeech-hubert-discrete-tokens" \
#         --model_name_or_path="r-sharma-coder/hubert-large-ll60k-librispeech-single-gpu" \
#         --train_split_name="train" \
#         --eval_split_name="test" \
#         --output_dir="./hubert-large-ll60k-librispeech-single-gpu_test_output" \
#         --preprocessing_num_workers="4" \
#         --num_train_epochs="4" \
#         --per_device_train_batch_size="32" \
#         --gradient_accumulation_steps="1" \
#         --learning_rate="3e-4" \
#         --warmup_steps="500" \
#         --evaluation_strategy="steps" \
#         --text_column_name="text" \
#         --audio_column_name="audio" \
#         --save_steps="400" \
#         --eval_steps="100" \
#         --logging_steps="1" \
#         --layerdrop="0.0" \
#         --save_total_limit="3" \
#         --gradient_checkpointing \
#         --chars_to_ignore , ? . ! - \; \: \" �~@\ % �@X �@~] \
#         --group_by_length \
#         --push_to_hub \
#         --do_eval \
#         --do_train \
#         --token="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" \
#         --overwrite_output_dir


#!/usr/bin/env bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 run_speech_recognition_ctc.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="hubert-large-ll60k-librispeech-clean-100h-demo-dist" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="test" \
	--output_dir="./hubert-large-ll60k-librispeech-clean-100h-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="50" \
	--per_device_train_batch_size="32" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="100" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--fp16 \
	--group_by_length \
	--push_to_hub \
	--do_train --do_eval