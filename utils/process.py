import numpy as np
import datasets

# token_ids = np.load("fleurs_outputs_100.npy")
train, test, validation = np.load("train_y.npy"), np.load("test_y.npy"), np.load("validation_y.npy")
token_ids = np.hstack((train, test, validation))
# import pdb;pdb.set_trace()

lengths = np.load("hubert_features/librispeech_asr/librispeech_asr_lengths.npy")

split_arrays = []


# Iterate over lengths to split token_ids
start_idx = 0
for length in lengths:
    end_idx = start_idx + length
    split_array = token_ids[start_idx:end_idx]
    split_arrays.append(split_array)
    start_idx = end_idx

# Convert the list of split arrays to a NumPy array
# split_arrays = np.array(split_arrays)


dataset = datasets.load_dataset("librispeech_asr")


new_dataset = datasets.DatasetDict({
    "train":dataset['train.clean.100'].add_column("hubert_discrete_tokens", split_arrays[:len(dataset['train.clean.100'])]),
    "test":dataset['test.clean'].add_column("hubert_discrete_tokens", split_arrays[len(dataset['train.clean.100']):len(dataset['train.clean.100'])+len(dataset['test.clean'])]),
    "validation":dataset['validation.clean'].add_column("hubert_discrete_tokens", split_arrays[len(dataset['train.clean.100'])+len(dataset['test.clean']):])

})

new_dataset.save_to_disk("librispeech-hubert-discrete-tokens", max_shard_size="500MB")

new_dataset.push_to_hub("macabdul9/librispeech-hubert-discrete-tokens", max_shard_size="500MB")

# dataset.remove_columns([['raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']])
import pdb;pdb.set_trace()
                                                    