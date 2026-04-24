import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "gpt2"
DATA_FILE = "AI.json"
OUTPUT_DIR = "./model"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load dataset
with open(DATA_FILE, 'r') as f:
    json_data = json.load(f)

stories = [item["story"] for item in json_data]

dataset = Dataset.from_dict({"text": stories})
dataset = dataset.train_test_split(test_size=0.1)["train"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Model → move to GPU
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training args (GPU optimizations)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,   # increase if GPU allows
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=3e-5,
    report_to="none",
    fp16=torch.cuda.is_available(),  # mixed precision for GPU
    dataloader_pin_memory=True       # faster GPU transfer
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete. Model saved in /model")
