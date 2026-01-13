# eval_ppo_mutations.py

import torch
import pandas as pd
import random
import math
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
import utils
from hn_utils import CodeMutator

########################################
# 1. Load PPO model & tokenizer
########################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./ppo_ckpt_1500"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

########################################
# 2. Load MBPP training data
########################################

df = pd.read_csv("mbpp_train.csv")

########################################
# 3. Helper functions
########################################

def normalized_char_overlap(x: str, y: str) -> float:
    overlap = sum(a == b for a, b in zip(x, y))
    return overlap / max(len(x), len(y))


def classify_mutation(x: str, y: str) -> str:
    """Heuristic mutation categorization"""

    if abs(len(x) - len(y)) <= 2:
        return "micro_edit"

    if "while True" in y or "input(" in y or "sleep(" in y:
        return "macro_edit"

    if "#" in y and "#" not in x:
        return "comment_insertion"

    if x.split("(")[0] != y.split("(")[0]:
        return "variable_rename"

    return "other"


########################################
# 4. Generate PPO negatives
########################################

generated_pairs = []

print("Generating PPO-based negative samples...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    x = row["code"]
    tests = row["test_list"]

    prompt = x + "\n<ECHO>\n"

    found = False
    for _ in range(5):  # up to 5 attempts
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        y = decoded.split("<ECHO>", 1)[-1]

        if not utils.check_functionality(y, tests):
            generated_pairs.append({
                "reference": x,
                "generated": y,
            })
            found = True
            break

    if not found:
        generated_pairs.append({
            "reference": x,
            "generated": y,
        })

print(f"Collected {len(generated_pairs)} PPO-generated negatives")

########################################
# 5. Rule-based baseline (for comparison)
########################################

mutator = CodeMutator()
rule_pairs = []

print("Generating rule-based negative samples...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    x = row["code"]
    tests = row["test_list"]

    y = mutator.mutate_until_fail(x, tests, utils.check_functionality)
    if y is not None:
        rule_pairs.append({
            "reference": x,
            "generated": y,
        })

########################################
# 6. Edit-distance analysis
########################################

ppo_overlaps = [
    normalized_char_overlap(p["reference"], p["generated"])
    for p in generated_pairs
]

rule_overlaps = [
    normalized_char_overlap(p["reference"], p["generated"])
    for p in rule_pairs
]

print("\n=== Edit Distance Statistics ===")
print(f"PPO mean overlap:   {np.mean(ppo_overlaps):.4f}")
print(f"PPO median overlap: {np.median(ppo_overlaps):.4f}")
print(f"Rule mean overlap:  {np.mean(rule_overlaps):.4f}")

########################################
# 7. Histogram (Figure 3-style)
########################################

plt.figure(figsize=(6, 4))
plt.hist(ppo_overlaps, bins=30, alpha=0.7, label="PPO")
plt.hist(rule_overlaps, bins=30, alpha=0.7, label="Rule-based")
plt.xlabel("Normalized Character Overlap")
plt.ylabel("Count")
plt.legend()
plt.title("Edit Distance Distribution")
plt.tight_layout()
plt.savefig("edit_distance_histogram.png")
plt.close()

########################################
# 8. Mutation category analysis (Table 4-style)
########################################

ppo_categories = Counter(
    classify_mutation(p["reference"], p["generated"])
    for p in generated_pairs
)

rule_categories = Counter(
    classify_mutation(p["reference"], p["generated"])
    for p in rule_pairs
)

print("\n=== Mutation Categories (PPO) ===")
for k, v in ppo_categories.items():
    print(f"{k}: {v}")

print("\n=== Mutation Categories (Rule-based) ===")
for k, v in rule_categories.items():
    print(f"{k}: {v}")

########################################
# 9. Qualitative examples (Section 5.2.2)
########################################

print("\n=== Qualitative PPO Examples ===")

for i in random.sample(range(len(generated_pairs)), 5):
    print("\nREFERENCE:\n", generated_pairs[i]["reference"])
    print("\nGENERATED:\n", generated_pairs[i]["generated"])
    print("-" * 80)

########################################
# 10. Save results
########################################

pd.DataFrame({
    "overlap": ppo_overlaps
}).to_csv("ppo_edit_distances.csv", index=False)

print("\nEvaluation complete.")
