import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

# Paths to the model and dataset
MODEL_PATH = "/home/jose.viera/projects/my_lotta/fine_tuned_tinyllama_gsm8k"
DATASET_PATH = "/home/jose.viera/projects/my_lotta/datasets/gsm8k"

# Load the fine-tuned model and tokenizer
print("Loading model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)  # Move model to GPU/CPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the preprocessed GSM8K dataset
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)

# Use the test split for evaluation
test_dataset = dataset["test"]

# Preprocess the dataset all at once
def preprocess_dataset(dataset, tokenizer):
    def preprocess(example):
        input_text = f"Solve the following problem: {example['question']}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        example["input_ids"] = inputs["input_ids"].squeeze(0).tolist()
        example["attention_mask"] = inputs["attention_mask"].squeeze(0).tolist()
        return example

    print("Preprocessing dataset...")
    return dataset.map(preprocess)

# Preprocess the test dataset
test_dataset = preprocess_dataset(test_dataset, tokenizer)

# Function to compute exact match (EM) accuracy
def compute_em_accuracy(model, tokenizer, dataset, device):
    model.eval()
    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating", unit="example"):
        # Prepare input tensor
        inputs = {
            "input_ids": torch.tensor(example["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
        }

        # Generate model output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        # Decode predictions and ground truth
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        ground_truth_answer = example["answer"].strip()

        # Compare predictions with ground truth
        if predicted_answer == ground_truth_answer:
            correct += 1
        total += 1

    # Calculate exact match accuracy
    em_accuracy = correct / total * 100
    return em_accuracy

# Compute EM accuracy
print("Computing exact match accuracy...")
em_accuracy = compute_em_accuracy(model, tokenizer, test_dataset, device)
print(f"Exact Match Accuracy: {em_accuracy:.2f}%")
