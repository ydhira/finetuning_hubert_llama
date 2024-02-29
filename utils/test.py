from transformers import AutoProcessor, HubertModel
from datasets import load_dataset
import soundfile as sf
import os
from tqdm import tqdm
import librosa
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

def parse_args():
    pass
# dataset_name = "google/fleurs"
dataset_name = "librispeech_asr"
audio_column_name = "audio"
text_column_name = "text" # text for librispeech
output_dir = "hubert_features/"
target_sr = 16_000



# dataset = load_dataset(dataset_name, "en_us")

# Load dataset
dataset = load_dataset("librispeech_asr")

# Load processor and model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to("cuda:5")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(f'{output_dir}/{dataset_name.split("/")[-1]}'):
    os.makedirs(f'{output_dir}/{dataset_name.split("/")[-1]}')

# Define splits
# splits = ['train', 'test', 'validation']
splits = ['train.clean.100', 'test.clean', 'validation.clean']

# Create a global list to store hidden_states.shape[0]
global_hidden_states_lengths = []

# Function to process a single example
def process_example(example):
    speech_array, orig_sr = example['audio']['array'], example['audio']['sampling_rate']
    speech_array = librosa.resample(y=speech_array, orig_sr=orig_sr, target_sr=target_sr)
    input_values = processor(speech_array, return_tensors="pt").input_values  # Batch size 1
    hidden_states = model(input_values.to("cuda:5")).last_hidden_state.squeeze(0).detach().cpu().numpy()
    global_hidden_states_lengths.append(hidden_states.shape[0])
    return hidden_states

# Parallelize the loop over splits
for split in splits:
    print(f"Extracting features for {split}.")

    # Use ThreadPoolExecutor to parallelize the loop
    with ThreadPoolExecutor(max_workers=8) as executor:
        features_list = list(tqdm(executor.map(process_example, dataset[split]), total=len(dataset[split])))

    # Convert the list of features to a numpy array
    features = np.vstack(features_list)

    # Save numpy tensor
    np.save(f'{output_dir}/{dataset_name.split("/")[-1]}/{split}.npy', features)

# Save the global list of hidden_states.shape[0] to a numpy array
np.save(f'{output_dir}/{dataset_name.split("/")[-1]}_lengths.npy', np.array(global_hidden_states_lengths))
