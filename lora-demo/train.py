import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

# ----------------------------
# Load Dataset
# ----------------------------

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for item in data:
    text = (
        f"### Instruction:\n{item['instruction']}\n\n"
        f"### Response:\n{item['response']}"
    )

    formatted_data.append({"text": text})

dataset = Dataset.from_list(formatted_data)

# ----------------------------
# Model
# ----------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ----------------------------
# LoRA
# ----------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ----------------------------
# Tokenization
# ----------------------------

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_dataset = dataset.map(tokenize)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# ----------------------------
# Training
# ----------------------------

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# ----------------------------
# Save Adapter
# ----------------------------

model.save_pretrained("./tinyllama_lora")

print("\nLoRA adapter saved!")