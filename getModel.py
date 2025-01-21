# Script to download models and datasets

import os
import subprocess
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Define download paths
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# Function to download models from Hugging Face
def download_model(model_name, model_dir):
    print(f"Downloading model: {model_name}")
    snapshot_download(repo_id=model_name, local_dir=os.path.join(MODEL_DIR, model_name), local_dir_use_symlinks=False)


# Download Models
models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

for model in models:
    download_model(model, MODEL_DIR)


print("All models downloaded successfully.")


