#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import jiwer
import torch.nn.functional as F


from functools import partial


import os
import datasets
import os
# os.environ['TRANSFORMERS_CACHE'] = './cache'

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    new_tokens_count:Optional[int]=field(
        default=100,
        metadata={
            "help":"How many new tokens you'd like to add. Usually it's same as number of clusters in K-means"
        }
    )


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(default=None, metadata={"help": "Path to the training data."})
    load_from_local:Optional[bool] = field(
        default=False,
        metadata={"help":"Whether to laod the hugginface dataset from local disk or look after huggingface hub"}
    )
    input_col_name : Optional[str] = field(
        default="hubert_discrete_tokens",
        metadata={"help":"input coloum name."}
    )
    output_col_name : Optional[str] = field(
        default="text",
        metadata={"help":"output coloum name."}
    )
    train_split_name : Optional[str] = field(
        default="train",
        metadata={"help":"Train split name."}
    )
    test_split_name : Optional[str] = field(
        default="test",
        metadata={"help":"Test split name."}
    )
    validation_split_name : Optional[str] = field(
        default="validation",
        metadata={"help":"Validation split name."}
    )
    max_train_samples:Optional[int]=field(
        default=None,
    )
    max_eval_samples:Optional[int]=field(
        default=1000,
    )
    max_test_samples:Optional[int]=field(
        default=1000
    )
    max_seq_length:Optional[int]=field(
        default=1024,
    )
    
    debugging: Optional[bool] = field(default=False, metadata={"help": "debugging mode."})

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help":"Number of workers for preprocessing"}
    )
    
 

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# tokenized data
def preprocess(example, tokenizer, data_args):
    
    # concate source and target or input and output
    example_text = example[data_args.input_col_name] + tokenizer.eos_token + example[data_args.output_col_name]
    
    # tokenize them together
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=data_args.max_seq_length, truncation=True)
    
    input_ids = tokenized_example.input_ids
    
    labels = input_ids.clone()
    
    # tokenize source seprately 
    source_tokenized = tokenizer(example[data_args.input_col_name], return_tensors="pt")
    
    # mask the input [discrete tokens] part for avoiding loss
    
    labels[:, :source_tokenized.input_ids.shape[1]] = -100
    
    # attending 
    attention_mask = torch.ones_like(input_ids)
    # import pdb;pdb.set_trace()
    return {
        "input_ids":input_ids.flatten(),
        "attention_mask":attention_mask.flatten(),
        "labels":labels.flatten()
    }
        
def train():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache=False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # if "llama" in model_args.model_name_or_path:
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    # 1. Add pad tokens and extra tokens for new ids
    tokenizer.pad_token = tokenizer.eos_token

    # add new tokens to accomoate hubert discrete tokens
    new_tokens = [str(token) for token in range(model_args.new_tokens_count)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    logging.warning("Loading data...")
    if data_args.load_from_local:
        raw_datasets = datasets.load_from_disk(data_args.dataset_name_or_path)
    else:
        raw_datasets = datasets.load_dataset(data_args.dataset_name_or_path)
        
    # for debugging
    if data_args.max_train_samples:
        raw_datasets[data_args.train_split_name] = raw_datasets[data_args.train_split_name].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples:
        raw_datasets[data_args.validation_split_name] = raw_datasets[data_args.validation_split_name].select(range(data_args.max_eval_samples))
    if data_args.max_test_samples:
        raw_datasets[data_args.test_split_name] = raw_datasets[data_args.test_split_name].select(range(data_args.max_test_samples))
    
    # let's do some preprocessing 
    # lower case all text, convert list of discrete tokens into string
    def format_(example):
        example[data_args.input_col_name] = ' '.join(map(str, example[data_args.input_col_name]))
        example[data_args.output_col_name] = example[data_args.output_col_name].lower()
        return example
    raw_datasets = raw_datasets.map(format_)
       
    
    # convert hubert discrete tokens
    encode_function = partial(
        preprocess,
        tokenizer=tokenizer,
        data_args=data_args
    )
    vectorized_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting data",
        )
    
    # so apprently .map changes tensors into list but we need tensors for training
    vectorized_datasets.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

    
    # update training args to make output dir
    output_dir = os.path.join(training_args.output_dir, model_args.model_name_or_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    training_args.output_dir = output_dir
    
    def compute_metrics(pred):
        # import pdb;pdb.set_trace()
        pred_logits = torch.from_numpy(pred.predictions)
        label_ids = torch.from_numpy(pred.label_ids)
        
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)

        # Take the argmax to get the predicted token IDs
        pred_ids = torch.argmax(pred_probs, dim=-1)

        # Decode the token IDs to strings
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)#.lower()
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)#.lower()
        
        # drop empty refs/preds
        filtered_pred_str, filtered_label_str = map(list, zip(*[(pred, label) for pred, label in zip(pred_str, label_str) if pred != "" and label != ""]))


        wer = 100 * jiwer.wer(filtered_label_str, filtered_pred_str)
        cer = 100 * jiwer.cer(filtered_label_str, filtered_pred_str)

        return {"wer": f'{wer:.2f}', "cer": f'{cer:.2f}'}



    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets[data_args.train_split_name],
        eval_dataset=vectorized_datasets[data_args.validation_split_name],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    )
    
    # import pdb;pdb.set_trace()
    # resume from last checkpoint if it exists     
    checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # import pdb;pdb.set_trace()
    
    train_batch_size = training_args.per_device_train_batch_size * training_args._n_gpu
    steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * training_args.gradient_accumulation_steps)
    total_train_steps = steps_per_epoch * training_args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {training_args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    if checkpoint:
        print(f"Checkpoint found! Training from {checkpoint} checkpoint!")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print(f"No checkpoint found! Training from scratch!")
        trainer.train()
        
    results = trainer.evaluate(vectorized_datasets[data_args.test_split_name])
    # results = trainer.predict(vectorized_datasets[data_args.test_split_name])
    print(results)
    
    # trainer.train()
    # save states 
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    print(f"Training finished! Saved model to {training_args.output_dir}.")


if __name__ == "__main__":
    train()