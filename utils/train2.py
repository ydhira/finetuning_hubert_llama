import datasets
import transformers
import time
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

dataset = datasets.load_dataset("macabdul9/fleurs-hubert-discrete-tokens")

# model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model_name = "meta-llama/Llama-2-7b-hf"

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token

new_tokens = [str(token) for token in range(100)]

model.resize_token_embeddings(len(tokenizer))

example = dataset['test'][0]['hubert_discrete_tokens']
example = [str(each) for each in example]
example = " ".join(example)

tokenized_example = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=256)
start_time = time.time()

outputs = model(**tokenized_example)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

import pdb;pdb.set_trace()


