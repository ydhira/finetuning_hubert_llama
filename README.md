# Finetuning LLama on HuBERT Tokens

This repository provides instructions and scripts for finetuning LLama on HuBERT tokens. The process involves preparing the data, configuring the model, and running the finetuning script.

## Prerequisites

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- Other dependencies...

## Setup

1. **Clone the repository**:

   ```bash
   git clone git@github.com:ydhira/finetuning_hubert_llama.git
   cd finetuning-llama-hubert

2. **Install Dependencies**
   ```
   pip install -r requirements
   ```

3. Run training on multi-GPU node with deepspeed 

   ```
   bash train_ds.sh
   ```
   Note: Specify the num_devices in the train_ds.sh

4. Launch the training in background
   ```
   nohup bash train_ds.sh &> logs/train.log
   ```

5. Launch the training in sbatch job as:
   ```
   sbatch -p GPU --gres=gpu:v100-32:8 -t 48:00:00 --job-name jobx ./train_ds.sh
   ```

4. Testing on small set of data:
   ```
   bash train.sh
   ``` 
   Note: Check number of examples and other details. 

