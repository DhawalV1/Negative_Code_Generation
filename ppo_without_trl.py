#This is the training loop for PPO in which TRl is not used

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from copy import deepcopy
from collections import deque
import utils
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import pandas as pd

def char_overlap(x: str, y: str) -> float:
    overlap = sum(a == b for a, b in zip(x, y))
    return overlap / max(len(x), len(y), 1)

train_data = pd.read_csv('mbpp_train.csv')

reference_dataset = [
    {
        "code": ex["code"],
        "tests": ex["test_list"]
    }
    for ex in train_data.to_dict(orient="records")
]

from datasets import load_from_disk

my_mutation_dataset = load_from_disk("phase2_mutated_dataset")
initial_rule_based_mutants = [
    example["target"] for example in my_mutation_dataset
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Actor (policy)
actor = AutoModelForCausalLM.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
).to(device)

# Critic (value head implemented manually)
critic = AutoModelForCausalLM.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
).to(device)

critic_value_head = nn.Linear(actor.config.hidden_size, 1).to(device)

# Reference model (frozen)
ref_actor = deepcopy(actor).eval()
for p in ref_actor.parameters():
    p.requires_grad = False

# Discriminator
discriminator = AutoModelForSequenceClassification.from_pretrained(
    "./qwen_model_local",
    num_labels=2,
    trust_remote_code=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
    fix_mistral_regex=True,
)
tokenizer.pad_token = tokenizer.eos_token

actor_opt = optim.AdamW(actor.parameters(), lr=1e-5)
critic_opt = optim.AdamW(
    list(critic.parameters()) + list(critic_value_head.parameters()), lr=1e-5
)
disc_opt = optim.AdamW(discriminator.parameters(), lr=2e-5)
disc_loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def D_prob(code: str) -> float:
    inp = tokenizer(code, return_tensors="pt", truncation=True).to(device)
    logits = discriminator(**inp).logits
    return torch.softmax(logits, dim=-1)[0, 1].item()

BATCH_SIZE = 4
MAX_NEW_TOKENS = 200
PPO_EPS = 0.2
KL_BETA = 0.05
OMEGA = 1e-6
NUM_STEPS = 2000
SAVE_EVERY = 100

neg_buffer = deque(maxlen=5000)
recent_D = deque(maxlen=10)
update_discriminator = True

neg_buffer.extend(initial_rule_based_mutants)

for step in range(NUM_STEPS):

    batch = random.sample(reference_dataset, BATCH_SIZE)

    actor_losses, critic_losses = [], []

    for item in batch:
        x = item["code"]
        tests = item["tests"]

        prompt = x + "\n<ECHO>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # ---------- Generate y ----------
        with torch.no_grad():
            gen = actor.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
        y = decoded.split("<ECHO>", 1)[1].strip() if "<ECHO>" in decoded else decoded

        # ---------- Unit tests ----------
        passed = utils.check_functionality(y, tests)

        if not passed:
            neg_buffer.append(y)

        # ---------- Rewards ----------
        ryed = char_overlap(x, y)
        Dy = D_prob(y)

        if not passed:
            reward = -math.log((1 - Dy) * (1 - ryed) + OMEGA)
        else:
            reward = Dy + ryed - 3.0

        reward_t = torch.tensor(reward, device=device)

        # ---------- Log probs ----------
        def logprob(model):
            out = model(**inputs, labels=inputs["input_ids"])
            return -out.loss

        logp_old = logprob(actor).detach()
        logp_ref = logprob(ref_actor).detach()
        logp_new = logprob(actor)

        # ---------- Advantage ----------
        h = critic(**inputs, output_hidden_states=True).hidden_states[-1][:, -1]
        value = critic_value_head(h).squeeze()
        advantage = reward_t - value.detach()

        # ---------- PPO loss ----------
        ratio = torch.exp(logp_new - logp_old)
        clipped = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS)
        policy_loss = -torch.min(ratio * advantage, clipped * advantage)

        kl = (logp_new - logp_ref)
        actor_loss = policy_loss + KL_BETA * kl

        # ---------- Critic loss ----------
        critic_loss = (value - reward_t).pow(2)

        # ---------- Optimize ----------
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        actor_opt.step()
        critic_opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        recent_D.append(Dy)

    # ---------- Discriminator update + pacing ----------
    if len(recent_D) == 10:
        avg_D = sum(recent_D) / 10
        if avg_D < 0.4:
            update_discriminator = False
        elif avg_D > 0.6:
            update_discriminator = True

    if update_discriminator:
        for item in batch:
            pos = item["code"]
            neg = random.choice(neg_buffer)

            pos_inp = tokenizer(pos, return_tensors="pt").to(device)
            neg_inp = tokenizer(neg, return_tensors="pt").to(device)

            pos_logits = discriminator(**pos_inp).logits
            neg_logits = discriminator(**neg_inp).logits

            loss = (
                disc_loss_fn(pos_logits, torch.tensor([1], device=device)) +
                disc_loss_fn(neg_logits, torch.tensor([0], device=device))
            )

            disc_opt.zero_grad()
            loss.backward()
            disc_opt.step()

    # ---------- Checkpoint ----------
    if step % SAVE_EVERY == 0 and step < 1800:
        actor.save_pretrained(f"./ppo_ckpt_{step}")
        print(f"[step {step}] actor_loss={sum(actor_losses)/len(actor_losses):.3f}")
