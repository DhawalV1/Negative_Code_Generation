import torch
import random
import math
from collections import deque
import utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl.experimental.ppo import PPOConfig, PPOTrainer
#from trl.experimental.ppo import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead, create_reference_model


from trl import PPOTrainer, PPOConfig
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)

ref_model = create_reference_model(actor)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen-echo-trainer",
    trust_remote_code=True,
    fix_mistral_regex=True,
)
tokenizer.pad_token = tokenizer.eos_token

discriminator = AutoModelForSequenceClassification.from_pretrained(
    "./qwen_model_local",
    num_labels=2,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device)

disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-5)
disc_loss_fn = torch.nn.CrossEntropyLoss()

ppo_config = PPOConfig(
    batch_size=8,
    mini_batch_size=4,
    learning_rate=1e-5,
    kl_coef=0.1,
)

# ppo_trainer = PPOTrainer(
#     config=ppo_config,
#     model=actor,
#     ref_model=ref_model,
#     tokenizer=tokenizer,
# )

ppo_trainer = PPOTrainer(
    model=actor,
    ref_model=ref_model,
    tokenizer=tokenizer,
    batch_size=ppo_config.batch_size,
    mini_batch_size=ppo_config.mini_batch_size,
    learning_rate=ppo_config.learning_rate,
    target_kl=ppo_config.kl_coef,
)


# Negative buffer D_neg
neg_buffer = deque(maxlen=5000)

from datasets import load_from_disk
my_mutation_dataset = load_from_disk("phase2_mutated_dataset")
initial_rule_based_mutants = [
    example["target"] for example in my_mutation_dataset
]

# Initialize with rule-based mutants
neg_buffer.extend(initial_rule_based_mutants)

# Track discriminator pacing
recent_disc_scores = deque(maxlen=10)

def unit_test_reward(code, test_list):
    return utils.check_functionality(code, test_list)   # True if passes all tests


def char_overlap(x: str, y: str) -> float:
    overlap = sum(1 for a, b in zip(x, y) if a == b)
    return overlap / max(len(x), len(y))

@torch.no_grad()
def disc_score(code: str) -> float:
    inputs = tokenizer(code, return_tensors="pt", truncation=True).to(device)
    logits = discriminator(**inputs).logits
    prob_real = torch.softmax(logits, dim=-1)[0, 1]
    return prob_real.item()

def compute_reward(x, y, passed_tests, omega=1e-6):
    ryed = char_overlap(x, y)
    Dy = disc_score(y)

    if not passed_tests:
        # Fails tests → amplify near 1
        return -math.log((1 - Dy) * (1 - ryed) + omega)
    else:
        # Passes tests → heavy penalty
        return Dy + ryed - 3.0

NUM_STEPS = 2000
SAVE_EVERY = 100

train_data, test_data = utils.prepare_data()
reference_dataset = [
    {
        "code": ex["code"],
        "tests": ex["test_list"]
    }
    for ex in train_data.to_dict(orient="records")
]

for step in range(NUM_STEPS):

    batch_prompts = random.sample(reference_dataset, ppo_config.batch_size)
    queries = []
    responses = []
    rewards = []

    for item in batch_prompts:
        x = item["code"]          # reference solution
        test_cases = item["tests"]  # ✅ NOW DEFINED

        prompt = x + "\n<ECHO>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = actor.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        y = decoded.split("<ECHO>", 1)[-1]

        passed = unit_test_reward(y,test_cases)

        if not passed:
            neg_buffer.append(y)

        r = compute_reward(x, y, passed)

        queries.append(inputs["input_ids"][0])
        responses.append(output_ids[0][inputs["input_ids"].shape[1]:])
        rewards.append(torch.tensor(r, device=device))

    # PPO update
    ppo_trainer.step(queries, responses, rewards)

    # Compute avg D(y)
    avg_Dy = sum(disc_score(y) for y in responses) / len(responses)
    recent_disc_scores.append(avg_Dy)

    update_disc = True
    if len(recent_disc_scores) == 10:
        mean_score = sum(recent_disc_scores) / 10
        if mean_score < 0.4:
            update_disc = False
        elif mean_score > 0.6:
            update_disc = True

    if update_disc:
        discriminator.train()
        for x in batch_prompts:
            pos = x
            neg = random.choice(neg_buffer)

            pos_inputs = tokenizer(pos, return_tensors="pt").to(device)
            neg_inputs = tokenizer(neg, return_tensors="pt").to(device)

            pos_logits = discriminator(**pos_inputs).logits
            neg_logits = discriminator(**neg_inputs).logits

            loss = (
                disc_loss_fn(pos_logits, torch.tensor([1], device=device)) +
                disc_loss_fn(neg_logits, torch.tensor([0], device=device))
            )

            disc_optimizer.zero_grad()
            loss.backward()
            disc_optimizer.step()

    if step % SAVE_EVERY == 0 and step < 1800:
        actor.save_pretrained(f"./ppo_checkpoint_{step}")
