from datasets import Dataset
from tqdm import tqdm
import utils
import pandas as pd
from hn_utils import CodeMutator
import torch
import math
import random
from copy import deepcopy
from collections import deque
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification




mutator = CodeMutator()
sft_training_pairs = []

print("Generating mutated code for Phase 2/3...")

train_data = pd.read_csv('mbpp_train.csv')
test_data = pd.read_csv('mbpp_test.csv')

for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
    correct_code = row["code"]
    tests = row["test_list"]

    buggy = mutator.mutate_until_fail(
        correct_code,
        tests,
        utils.check_functionality,
    )

    if buggy is not None:
        sft_training_pairs.append({
            "input": correct_code,
            "target": buggy,
        })

print("Total SFT pairs:", len(sft_training_pairs))

sft_dataset = Dataset.from_list(sft_training_pairs)
sft_dataset = sft_dataset.map(utils.merge_fields, remove_columns=sft_dataset.column_names)

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# model_id = "Qwen/Qwen2.5-Coder-0.5B"

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
    use_fast=False,
)




tokenizer.save_pretrained("./ppo_ckpt_1500")
tokenizer = AutoTokenizer.from_pretrained(
    "./ppo_ckpt_1500",
    trust_remote_code=True,
    use_fast=False,
)
assert tokenizer.convert_tokens_to_ids("<ECHO>") is not None
assert tokenizer.pad_token_id is not None

print(tokenizer.pad_token, tokenizer.eos_token)
device = "cuda" if torch.cuda.is_available() else "cpu"
actor = AutoModelForCausalLM.from_pretrained(
    "./ppo_ckpt_1500",
    trust_remote_code=True,
).to(device)

critic = AutoModelForCausalLM.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
).to(device)

hard_negative_pairs = []

for _, row in train_data.iterrows():
    x = row["code"]
    tests = row["test_list"]

    for _ in range(5):
        prompt = x + "\n<ECHO>\n"
        gen = actor.generate(
            **tokenizer(prompt, return_tensors="pt").to(device),
            max_new_tokens=200,
            do_sample=True,
        )
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        y = decoded.split("<ECHO>", 1)[-1]

        if not utils.check_functionality(y, tests):
            hard_negative_pairs.append({
                "prompt": x,
                "chosen": x,    # correct solution
                "rejected": y,  # hard negative
            })
            break


def dpo_loss(logp_chosen, logp_rejected, beta=0.1):
    return -torch.log(torch.sigmoid(beta * (logp_chosen - logp_rejected)))


import torch.optim as optim

dpo_model = AutoModelForCausalLM.from_pretrained(
    "./ppo_ckpt_1500",
    trust_remote_code=True,
).to(device)

optimizer = optim.AdamW(
    dpo_model.parameters(),
    lr=5e-6,
)

import hn_utils

for batch in hard_negative_pairs:
    logp_c = hn_utils.log_prob(dpo_model, tokenizer, batch["prompt"], batch["chosen"])
    logp_r = hn_utils.log_prob(dpo_model, tokenizer, batch["prompt"], batch["rejected"])

    loss = dpo_loss(logp_c, logp_r)
    loss.backward()
    optimizer.step()
