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

import os
import datasets
import os
# os.environ['TRANSFORMERS_CACHE'] = './cache'

os.environ["WANDB_DISABLED"] = "true"

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
        default=None,
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
    
    
    
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    
    """Source is hubert discrete tokens. So first we will convert that into string."""
    sources = [str(each) for each in sources]
    sources = " ".join(sources)
    
    sources_tokenized = tokenizer(sources, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
    targets_tokenized = tokenizer(targets, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)

    return dict(input_ids=sources_tokenized['input_ids'], labels=targets_tokenized['input_ids'])

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, split_name:str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        # Load the dataset
        logging.warning("Loading data...")
        if data_args.load_from_local:
            dataset = datasets.load_from_disk(data_args.dataset_name_or_path)[split_name]
        else:
            dataset = datasets.load_dataset(data_args.dataset_name_or_path, split=split_name)
            
        if split_name=="train":
            dataset = dataset.select(range(1000))

        sources = dataset[data_args.input_col_name]
        targets = dataset[data_args.output_col_name]

        logging.warning(f"Tokenizing {split_name}... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args=data_args, split_name=data_args.train_split_name, tokenizer=tokenizer)
    test_dataset = SupervisedDataset(data_args=data_args, split_name=data_args.test_split_name, tokenizer=tokenizer)
    # validation_dataset = SupervisedDataset(data_args=data_args, split_name=data_args.validation_split_name, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        
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
    
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    # 1. Add pad tokens and extra tokens for new ids
    tokenizer.pad_token = tokenizer.eos_token

    new_tokens = [str(token) for token in range(model_args.new_tokens_count)]

    model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    train_dataloader = DataLoader(
        data_module['train_dataset'], 
        shuffle=True, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=training_args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        data_module['eval_dataset'], 
        shuffle=True, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=training_args.per_device_eval_batch_size,
    )
    
    # update training args to make output dir
    output_dir = os.path.join(training_args.output_dir, model_args.model_name_or_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    training_args.output_dir = output_dir
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * jiwer.wer(label_str, pred_str)# wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * jiwer.cer(label_str, pred_str) #cer_metric.compute(predictions=pred_str, references=label_str)
        

        return {"wer": wer, "cer": cer}
    
    # import pdb;pdb.set_trace()
    

    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataloader, 
        eval_dataset=eval_dataloader
    )
    
    # resume from last checkpoint if it exists     
    checkpoint = get_last_checkpoint(training_args.output_dir)

    if checkpoint:
        print(f"Checkpoint found! Training from {checkpoint} checkpoint!")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print(f"No checkpoint found! Training from scratch!")
        trainer.train()
        
    results = trainer.evaluate()
    print(results)
    
    # trainer.train()
    # save states 
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    print(f"Training finished! Saved model to {training_args.output_dir}.")


if __name__ == "__main__":
    train()