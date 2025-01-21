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

# Preprocess a few examples
def preprocess_example(example, tokenizer):
    input_text = f"Solve the following problem: {example['question']}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "question": example['question'],
        "answer": example['answer']
    }

# Function to print model answers
def print_model_answers(model, tokenizer, dataset, num_examples=5):
    model.eval()
    examples = dataset.select(range(min(len(dataset), num_examples)))  # Select a few examples

    for idx, example in enumerate(examples):
        processed = preprocess_example(example, tokenizer)
        inputs = {"input_ids": processed["input_ids"], "attention_mask": processed["attention_mask"]}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(f"Example {idx + 1}:")
        print(f"Question: {processed['question']}")
        print(f"Ground Truth Answer: {processed['answer']}")
        print(f"Model's Predicted Answer: {predicted_answer}")
        print("-" * 50)

# Print a few answers
print("Printing model answers...")
print_model_answers(model, tokenizer, test_dataset, num_examples=5)
