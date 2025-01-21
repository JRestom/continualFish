import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Added just to get rid of a warning 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Paths to the model and dataset
MODEL_PATH = "models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "/home/jose.viera/projects/my_lotta/datasets/gsm8k"

# Load the model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the pre-downloaded dataset
print("Loading dataset from disk...")
dataset = load_from_disk(DATASET_PATH)

# Preprocess the dataset
def preprocess_function(examples):
    """Preprocess the data for causal language modeling."""
    inputs = [f"Solve the following problem: {question}\nAnswer:" for question in examples["question"]]
    targets = examples["answer"]
    inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Preprocessing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split dataset into training and validation
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["test"]


# Training arguments
training_args = TrainingArguments(
    output_dir="fine_tune_gsm8k",  # Output directory
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    load_best_model_at_end=True,
    save_steps=500,
    fp16=False  # This is for mixed precision 
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving fine-tuned model...")
trainer.save_model("fine_tuned_tinyllama_gsm8k")
tokenizer.save_pretrained("fine_tuned_tinyllama_gsm8k")
print("Training complete.")
