from transformers import AutoProcessor, HubertModel
from datasets import load_dataset
import soundfile as sf
import os
from tqdm import tqdm
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    pass



dataset_name = "google/fleurs"
audio_column_name = "audio"
text_column_name = "transcription" # text for librispeech
output_dir = "hubert_features/"
target_sr = 16_000



dataset = load_dataset(dataset_name, "en_us")

# dataset = load_dataset("librispeech_asr")


# import pdb;pdb.set_trace()


print(dataset)


processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to("cuda:6")
# 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(f'{output_dir}/{dataset_name.split("/")[-1]}'):
    os.makedirs(f'{output_dir}/{dataset_name.split("/")[-1]}')

# splits = ['test.clean', 'train.clean.100', 'validation.clean']
splits = ['train', 'test', 'validation']


train_test_validation_len = []

for split in splits:
    
    print(f"Extracting features for {split}.")
    
    features = np.empty((0, 1024))
    
    for example in tqdm(dataset[split]):
                
        speech_array, orig_sr = example['audio']['array'], example['audio']['sampling_rate']
        
        speech_array = librosa.resample(y=speech_array, orig_sr=orig_sr, target_sr=target_sr)
        
        input_values = processor(speech_array, return_tensors="pt").input_values  # Batch size 1
        
        hidden_states = model(input_values.to("cuda:6")).last_hidden_state.squeeze(0).detach().cpu().numpy()
        
        features = np.vstack((features, hidden_states))
        
        train_test_validation_len.append(hidden_states.shape[0])
        
    # save numpy tensor
    np.save(f'{output_dir}/{dataset_name.split("/")[-1]}/{split}.npy', features)
            
np.save(f'{output_dir}/{dataset_name}.npy', np.array(train_test_validation_len))
        
