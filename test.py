from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

local_model_dir = "./qwen-echo-trainer"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)
model.eval()
# Resize embeddings ONLY if tokenizer was changed AFTER training
# (usually not needed at inference time)
# model.resize_token_embeddings(len(tokenizer))

prompt = """def prime_num(num):
    if num >= 1:
        for i in range(2, num // 2):
            if num % i == 0:
                return False
        return True
    return False
<ECHO>
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# with torch.no_grad():
output_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=200,
    do_sample=False,
)

out = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(out)
