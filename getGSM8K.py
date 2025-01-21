import os
from datasets import load_dataset

DATASET_DIR = "datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

# Path to save the dataset
DATASET_PATH = "/home/jose.viera/projects/my_lotta/datasets/gsm8k"

# Download and save the GSM8K dataset
print("Downloading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main")

# Save the dataset to disk
print("Saving dataset to disk...")
dataset.save_to_disk(DATASET_PATH)
print(f"Dataset saved to {DATASET_PATH}")