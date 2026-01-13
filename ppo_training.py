import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import utils
from datasets import load_from_disk

my_mutation_dataset = load_from_disk("phase2_mutated_dataset")
print(f"Loaded {len(my_mutation_dataset)} examples.")
# 1. Load the Actor & Critic (Value Head)
# Initialize from your Phase 3 checkpoint
model_path = "./qwen-echo-trainer"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")

# 2. Reference Model (to compute KL divergence, kept frozen)
ref_model = create_reference_model(model)

# 3. Discriminator (Binary Classifier)
# The paper uses the same base model but as a classifier
discriminator = AutoModelForSequenceClassification.from_pretrained(
    "./qwen_model_local", 
    num_labels=2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def calculate_edit_distance_reward(ref_code, gen_code):
    # Normalized Character Overlap (ryed)
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ref_code, gen_code)
    overlap = matcher.ratio() # This is roughly (2*match)/(len1 + len2)
    return overlap

def get_combined_reward(gen_code, ref_code, test_list, disc_prob):
    """
    Implements the formula from Section 3.2
    gen_code: y, ref_code: x, disc_prob: D(y)
    """
    # 1. Unit Test Reward (passes tests -> -3 penalty, fails -> 0)
    passes_tests = utils.check_functionality(gen_code, test_list)
    r_ut = -3 if passes_tests else 0
    
    # 2. Edit Distance Reward
    r_yed = calculate_edit_distance_reward(ref_code, gen_code)
    
    epsilon = 1e-8
    if not passes_tests:
        # Pushes model to fool discriminator AND stay close to reference
        # -log[(1 - D(y))(1 - ryed)]
        reward = -torch.log((1 - disc_prob) * (1 - r_yed) + epsilon)
    else:
        # Heavy penalty if it passes tests
        reward = disc_prob + r_yed - 3
        
    return reward

# PPO Configuration
config = PPOConfig(
    learning_rate=5e-6,
    batch_size=8,
    mini_batch_size=4,
    ppo_epochs=4,
    init_kl_coef=0.2,
    target=6,
    optimize_cuda_cache=True,
)

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# Discriminator Optimizer
disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-5)
update_discriminator = True # Pacing flag

# Initial Negative Buffer (D_neg) from Phase 2
negative_buffer = list(my_mutation_dataset['target']) 

for epoch in range(2000): # The paper goes up to 2000 iterations
    # --- 1. GENERATION PHASE ---
    # Prepare batch (Correct Code + <ECHO>)
    query_tensors = [tokenizer.encode(q + "<ECHO>", return_tensors="pt").squeeze() for q in batch_ref_codes]
    
    # Generate y
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
    
    # --- 2. REWARD PHASE ---
    # Get Discriminator score D(y)
    disc_inputs = tokenizer(responses, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = discriminator(**disc_inputs).logits
        disc_probs = torch.softmax(logits, dim=-1)[:, 1] # Prob of being 'Real'
    
    # Calculate combined rewards
    rewards = [get_combined_reward(res, ref, tests, prob) 
               for res, ref, tests, prob in zip(responses, batch_ref_codes, batch_tests, disc_probs)]
    
    # --- 3. PPO UPDATE PHASE (Generator) ---
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # --- 4. ADVERSARIAL UPDATE PHASE (Discriminator) ---
    avg_d_y = disc_probs.mean().item()
    
    # Pacing Logic (Section 3.2)
    if avg_d_y < 0.4: update_discriminator = False
    if avg_d_y > 0.6: update_discriminator = True
    
    if update_discriminator:
        # Update D_neg buffer with failing samples
        for res, tests in zip(responses, batch_tests):
            if not utils.check_functionality(res, tests):
                negative_buffer.append(res)
        
        # Train Discriminator: Real=Ref_Codes, Fake=Samples from negative_buffer
        # (Standard CrossEntropy training omitted for brevity)
        train_discriminator_step(discriminator, batch_ref_codes, negative_buffer)

    