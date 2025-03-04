import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
print("Loading the fully fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained("full_model", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("full_model")

print("✅ Full fine-tuned model loaded successfully!")


prompt = "Explain recursion step by step."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=150)

# Decode result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model Response ===\n")
print(result)
print("\n======================\n")
