from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama_lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

print(type(model))
print("\nLoRA model loaded successfully!")

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Only decode newly generated tokens
    # generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    print("\n===== FULL OUTPUT =====")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    print("\n===== GENERATED ONLY =====")
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True))