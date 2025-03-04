import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset

# Load the  Model and Tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"  
print("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Use half-precision for GPU efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to the end of the sequence
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA 
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation size
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Modules to adapt with LoRA 
    lora_dropout=0.1,  # Dropout for regularization
    bias="none"
)
model = get_peft_model(model, lora_config)


print("Loading dataset...")
dataset = load_dataset("KingNish/reasoning-base-20k")


dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Tokenization function
def tokenize_function(examples):
    """Concatenate user query and reasoning, then tokenize."""
    input_texts = [f"User: {q}\nReasoning: {r}" for q, r in zip(examples["user"], examples["reasoning"])]
    labels = examples["assistant"]  # Correct answer
    
    tokenized_inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=512)
    tokenized_labels = tokenizer(labels, padding="max_length", truncation=True, max_length=512)

    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["user", "assistant", "reasoning", "template"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["user", "assistant", "reasoning", "template"])

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",  # Save model checkpoints
    eval_strategy="epoch",  
    save_strategy="epoch",
    per_device_train_batch_size=2,  # Adjust based on GPU VRAM
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=1, 
    save_total_limit=2,
    report_to="none", 
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
    label_names=["labels"],  # Explicitly define label names for Trainer
)

# Trainer
print("Starting fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  
)

trainer.train()

# Merge LoRA Weights into Base Model 
print("Merging LoRA weights into the base model...")
model = model.merge_and_unload()  

# Save the Fully Fine-Tuned Model 
print("Saving the fully fine-tuned model...")
model.save_pretrained("full_model")
tokenizer.save_pretrained("full_model")

print("✅ Full fine-tuned model saved successfully!")
