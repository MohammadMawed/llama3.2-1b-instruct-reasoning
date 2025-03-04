#  Fine-Tuning Llama 3.2-1B with LoRA (Low-Rank Adaptation)

This project fine-tunes **Meta's Llama 3.2-1B-Instruct** model using **LoRA (Low-Rank Adaptation)** with **Hugging Face Transformers and PEFT**. The goal is to enhance **reasoning capabilities** using the **KingNish/reasoning-base-20k** dataset.

---

## Steps to Fine-Tune Llama 3.2-1B

### 1️⃣ **Environment Setup**
Ensure you have **Docker and WSL2** installed on Windows, or set up a **Linux environment**.

#### **Run the following commands to set up your environment:**
```
# Clone this repository
git clone https://github.com/MohammadMawed/cot-llama3.2-1b.git
cd cot-llama3.2-1b

# Create and activate a virtual environment (Optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **Download the Base Model**
Make sure you have **Hugging Face access** to `meta-llama/Llama-3.2-1B-Instruct`. Authenticate via:
```bash
huggingface-cli login
```
Then, verify you can download the model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3️⃣ **Fine-Tuning with LoRA**
To train the model with **LoRA adapters**, run:
```
python train.py
```
This will:
✅ Load the base model & tokenizer  
✅ Apply LoRA for efficient fine-tuning  
✅ Load and preprocess the dataset  
✅ Train the model while handling **low VRAM** scenarios  
✅ Merge LoRA weights into the base model  
✅ Save the fine-tuned model for later inference  

### 4️⃣ **Save the Fine-Tuned Model**
The fully fine-tuned model is saved in `full_model/`.

To reload it later:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("full_model")
tokenizer = AutoTokenizer.from_pretrained("full_model")
print("✅ Model Loaded Successfully!")
```

### 5️⃣ **Running Inside Docker (Optional)**
If you prefer to run the training inside a **Docker container**, build and run it:
```bash
# Build Docker image
docker build -t llama-finetuned .

# Run the container with GPU support
docker run --gpus all -it --rm llama-finetuned
```

---

## Optimizations for Low VRAM (4GB GPUs)
Since **4GB VRAM (GTX 1050 Ti) (Current Hardware)** is limited for LLM training, we used:
✅ **`batch_size=1`** to avoid Out-of-Memory (OOM) errors  
✅ **`gradient_accumulation_steps=4`** to simulate a larger batch size  
✅ **`fp16=True`** for mixed precision training  
✅ **4-bit quantization (bitsandbytes)** for **reduced memory usage** (Optional)  

---

## 📌 Dataset Details
We used the **KingNish/reasoning-base-20k** dataset, which contains:
- **`user`**: The user's query (question)
- **`assistant`**: The correct answer
- **`reasoning`**: Chain-of-thought reasoning to derive the answer
- **`template`**: RChatML format for structured prompting

During preprocessing, we tokenized **`user + reasoning`** as input and **`assistant`** as labels.

---

## Future Improvements
🔹 **Adding real-time "thinking" indicators during inference**  
🔹 **Implementing visualization tools for reasoning steps**  
🔹 **MaybeDeploy the fine-tuned model as an API**  



