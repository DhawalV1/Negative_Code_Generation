import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
from dataclasses import dataclass
from typing import List, Dict, Any
import utils

# dataset = load_from_disk("phase3_text_dataset")
# print(dataset)
# print(dataset[0])

my_mutation_dataset = load_from_disk("phase2_mutated_dataset")
print(f"Loaded {len(my_mutation_dataset)} examples.")

my_mutation_dataset = my_mutation_dataset.map(
    utils.merge_fields,
    remove_columns=my_mutation_dataset.column_names
)

print(my_mutation_dataset.column_names)

model_id = "./qwen_model_local"


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False
)

if "<ECHO>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": ["<ECHO>"]})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# Resize ONCE because we added <ECHO>
model.resize_token_embeddings(len(tokenizer))

@dataclass
class EchoDataCollator:
    tokenizer: Any
    echo_token: str = "<ECHO>"
    max_length: int = 512

    def __post_init__(self):
        self.echo_id = self.tokenizer.convert_tokens_to_ids(self.echo_token)
        if self.echo_id is None:
            raise ValueError("<ECHO> token not found")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        labels = input_ids.clone()

        # 🔥 Mask loss before and including <ECHO>
        for i in range(labels.size(0)):
            echo_pos = (input_ids[i] == self.echo_id).nonzero(as_tuple=True)[0]
            if len(echo_pos) == 0:
                labels[i, :] = -100
            else:
                labels[i, : echo_pos[0] + 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

training_args = TrainingArguments(
    output_dir="./qwen-echo-trainer",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    report_to="none",
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,  # 🔥 THIS FIXES THE ERROR
)


collator = EchoDataCollator(
    tokenizer=tokenizer,
    echo_token="<ECHO>",
    max_length=512,
)

trainer = Trainer(
    model=model,
    train_dataset=my_mutation_dataset,
    args=training_args,
    data_collator=collator,
)

batch = collator([my_mutation_dataset[0]])
print(batch.keys())


trainer.train()

trainer.save_model("./qwen-echo-trainer")
tokenizer.save_pretrained("./qwen-echo-trainer")
