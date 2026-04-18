from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# =====================================================
# MODEL LOAD
# =====================================================
model_path = "model/gemma-4-E2B"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("🔄 Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Model loaded successfully!")
print("🖥️ Device:", device)


# =====================================================
# MAIN FUNCTION
# =====================================================
def process_text(user_input):

    print("\n📥 New request received")
    print("🧾 Input:", user_input)

    prompt = f"""
You are a STRICT skin health assistant.

ONLY follow this format:

CAUSE:
SAFETY:
ADVICE:
IMAGE:

Symptoms:
{user_input}

Answer:
"""

    # =====================================================
    # TOKENIZE
    # =====================================================
    print("🧠 Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # =====================================================
    # GENERATE
    # =====================================================
    print("⚡ Generating response from model...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    print("✅ Generation complete")

    # =====================================================
    # REMOVE PROMPT
    # =====================================================
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\n📝 RAW MODEL OUTPUT:\n", response)

    return response