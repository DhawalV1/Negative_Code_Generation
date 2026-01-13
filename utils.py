import pandas as pd
from datasets import load_dataset

def prepare_data():
    # Load MBPP full dataset
    dataset = load_dataset("google-research-datasets/mbpp", "full")
    
    # Combine Train (374), Validation (90), and Prompt (10)
    train_df = pd.concat([
        pd.DataFrame(dataset['train']),
        pd.DataFrame(dataset['validation']),
        pd.DataFrame(dataset['prompt'])
    ])
    test_df = pd.DataFrame(dataset['test'])

    # Filtering: Reference solution length < 256 characters
    train_df = train_df[train_df['code'].str.len() < 256]
    test_df = test_df[test_df['code'].str.len() < 256]
    
    return train_df, test_df



import multiprocessing

def execute_test(code, test_case, result_queue):
    try:
        # Combine code and test
        full_code = f"{code}\n{test_case}"
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put(True) # Passed
    except Exception:
        result_queue.put(False) # Failed

def check_functionality(code, test_list):
    """Returns True if code passes ALL tests, False otherwise."""
    for test in test_list:
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=execute_test, args=(code, test, q))
        p.start()
        p.join(timeout=1.0) # 1-second timeout
        
        if p.is_alive():
            p.terminate()
            return False # Timeout treated as failure
        
        if q.empty() or not q.get():
            return False # Exception or test failure
    return True

import re
import random

import re
import random
import time

class CodeMutator:
    def __init__(self):
        # Define operators for the flip rules
        self.arithmetic_ops = {'+': '-', '-': '+', '*': '/', '/': '*'}
        self.comparison_ops = {'==': '!=', '!=': '==', '<': '>', '>': '<', '<=': '>=', '>=': '<='}

    def _get_executable_line_indices(self, lines):
        """Helper to find lines that aren't definitions, imports, or comments."""
        indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('def ', 'import ', 'from ', '#', 'class ')):
                indices.append(i)
        return indices

    # 1. variable_swap: Randomly swaps two local variables
    def variable_swap(self, code):
        # Find potential variable names (simple alphanumeric words not in keywords)
        keywords = {'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'and', 'or', 'not', 'None', 'True', 'False'}
        vars_found = list(set(re.findall(r'\b[a-z_][a-z0-9_]*\b', code)))
        vars_found = [v for v in vars_found if v not in keywords]
        if len(vars_found) >= 2:
            v1, v2 = random.sample(vars_found, 2)
            # Use a temporary placeholder to swap
            placeholder = "TEMP_VAR_PLACEHOLDER"
            code = re.sub(rf'\b{v1}\b', placeholder, code)
            code = re.sub(rf'\b{v2}\b', v1, code)
            code = re.sub(rf'\b{placeholder}\b', v2, code)
        return code

    # 2. off_by_one: range(n) -> range(n - 1)
    def off_by_one(self, code):
        return re.sub(r'range\(([^)]+)\)', r'range(\1 - 1)', code, count=1)

    # 3. operator_flip: Arithmetic + -> -, * -> /
    def operator_flip(self, code):
        for op, rep in self.arithmetic_ops.items():
            if op in code:
                return code.replace(op, rep, 1)
        return code

    # 4. comparison_flip: == -> !=, < -> >
    def comparison_flip(self, code):
        for op, rep in self.comparison_ops.items():
            if op in code:
                return code.replace(op, rep, 1)
        return code

    # 5. premature_return: Inserts return None early
    def premature_return(self, code):
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                lines.insert(i + 1, "    return None")
                break
        return "\n".join(lines)

    # 6. line_delete: Removes one random executable line
    def line_delete(self, code):
        lines = code.split('\n')
        indices = self._get_executable_line_indices(lines)
        if indices:
            lines.pop(random.choice(indices))
        return "\n".join(lines)

    # 7. arg_order_swap: Swaps first two positional arguments in a call
    def arg_order_swap(self, code):
        # Matches function_name(arg1, arg2, ...)
        pattern = r'(\w+)\(([^,]+),\s*([^,)]+)'
        return re.sub(pattern, r'\1(\3, \2', code, count=1)

    # 8. constant_perturb: Increments the first integer literal by 1
    def constant_perturb(self, code):
        match = re.search(r'\b(\d+)\b', code)
        if match:
            val = int(match.group(1))
            code = code[:match.start()] + str(val + 1) + code[match.end():]
        return code

    # 9. reverse_list: [a, b, c] -> [c, b, a]
    def reverse_list(self, code):
        match = re.search(r'\[([^\]]+)\]', code)
        if match:
            elements = [e.strip() for e in match.group(1).split(',')]
            if len(elements) > 1:
                reversed_list = f"[{', '.join(reversed(elements))}]"
                code = code[:match.start()] + reversed_list + code[match.end():]
        return code

    # 10. comment_out: Comments out a random executable line
    def comment_out(self, code):
        lines = code.split('\n')
        indices = self._get_executable_line_indices(lines)
        if indices:
            idx = random.choice(indices)
            lines[idx] = f"# {lines[idx]}"
        return "\n".join(lines)

    # 11. infinite_loop: while True: pass
    def infinite_loop(self, code):
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                lines.insert(i + 1, "    while True: pass")
                break
        return "\n".join(lines)

    # 12. input_block: Force an input() call (causes EOF error in tests)
    def input_block(self, code):
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                lines.insert(i + 1, "    input()")
                break
        return "\n".join(lines)

    # 13. sleep_call: time.sleep(2) to trigger timeout
    def sleep_call(self, code):
        code = "import time\n" + code
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                lines.insert(i + 1, "    time.sleep(2)")
                break
        return "\n".join(lines)

    def mutate_until_fail(self, code, tests, check_functionality):
        """
        Tries rules randomly until the code fails the check_fn (unit tests).
        'check_fn' should be the function we wrote in Phase 1.
        """
        rules = [
            self.variable_swap, self.off_by_one, self.operator_flip, 
            self.comparison_flip, self.premature_return, self.line_delete,
            self.arg_order_swap, self.constant_perturb, self.reverse_list,
            self.comment_out, self.infinite_loop, self.input_block, self.sleep_call
        ]
        
        random.shuffle(rules)
        for rule in rules:
            try:
                mutated = rule(code)
                if mutated != code: # Ensure something actually changed
                    # check_fn returns True if tests pass, so we want False
                    if check_functionality(mutated, tests) == False:
                        return mutated
            except Exception:
                continue
        return None
    
def merge_fields(example):
    return {
        "text": example["input"] + "<ECHO>" + example["target"]
    }

'''
print("Generating mutated code for Phase 3...")
# 2. Loop through every problem in your filtered training set
# 'train_data' is the DataFrame from Phase 1
for index, row in tqdm.tqdm(train_data.iterrows(), total=len(train_data)):
    correct_code = row['code']
    unit_tests = row['test_list'] # MBPP provides a list of test strings
    
    # 3. Use the mutator to find a version that FAILS the tests
    # This calls the function we defined in Phase 2
    buggy_code = mutator.mutate_until_fail(correct_code, unit_tests, check_functionality)
    
    if buggy_code:
        # 4. Save it as a pair
        sft_training_pairs.append({
            "input": correct_code,
            "target": buggy_code
        })
    else:
        print(f"Warning: Could not find a failing mutant for task {row['task_id']}")

# 5. Result: 'sft_training_pairs' is now a list of ~385 dictionaries
print(f"Successfully created {len(sft_training_pairs)} mutation pairs.")

# Optional: Convert to HuggingFace Dataset format for easier training
from datasets import Dataset

my_mutation_dataset = Dataset.from_list(sft_training_pairs)
'''

def unit_test_reward(code: str) -> bool:
    return run_tests(code)   # True if passes all tests


def char_overlap(x: str, y: str) -> float:
    overlap = sum(1 for a, b in zip(x, y) if a == b)
    return overlap / max(len(x), len(y))


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
    

    import torch

def log_prob(model, tokenizer, prompt, completion):
    """
    Computes log p(completion | prompt)
    """

    # Tokenize separately
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    full_ids = tokenizer(
        prompt + completion,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]

    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:].clone()

    # Mask prompt tokens
    labels[:, : prompt_ids.shape[1]] = -100

    outputs = model(
        input_ids=input_ids.to(model.device),
        labels=labels.to(model.device),
    )

    # loss is negative log-likelihood averaged over unmasked tokens
    # convert to total log-prob
    num_tokens = (labels != -100).sum()
    return -outputs.loss * num_tokens
