import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load .env
load_dotenv()
token = os.getenv("HF_TOKEN")

# Model name
model_path = "model/gemma-4-E2B"

# Download tokenizer & model with token & resume option
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto", 
    low_cpu_mem_usage=True,  
   
)

# save to use offline
save_path = "model/gemma-4-E2B"
os.makedirs(save_path, exist_ok=True)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)


# Test code (optional)
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



print(f"✅ Gemma 4 E2B model saved offline in {save_path}")