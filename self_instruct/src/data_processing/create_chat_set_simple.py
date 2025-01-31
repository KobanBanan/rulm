import json
import random
import sys

from datasets import load_dataset
from prompt_generator import PromptGenerator

from bad_substrings import has_bad_ss

dataset_name = 'd0rj/synthetic-instruct-gptj-pairwise-ru'
split = 'train[0:5000]'

prompt_generator = PromptGenerator()

train_path = sys.argv[1]
val_path = sys.argv[2]

print(f'dataset {dataset_name}')
print(f'train_path {train_path}')
print('val path {val_path}')

records = []

print(f'downloading {dataset_name}...')

print('preparing sft dataset...')
for row in load_dataset(dataset_name, split=split):
    instruction, _ = prompt_generator(row["prompt"], [])
    output = row["chosen"]
    if has_bad_ss([{"content": output}]):
        continue

    records.append({
        "messages": [
            {"role": "user", "content": instruction, 'search': False},
            {"role": "bot", "content": output, 'search': False}
        ],
        "source": dataset_name
    })

print("Evol-instruct count:", len(records))

cleaned_records = []
for record in records:
    messages = record["messages"]
    roles = {m["role"] for m in messages}
    for role in roles:
        assert role in ("bot", "user", "system"), role
    if has_bad_ss(messages):
        continue
    if not record["messages"]:
        continue
    cleaned_records.append(record)

records = cleaned_records
print("All count after cleaning:", len(records))

random.shuffle(records)
border = int(0.95 * len(records))
train_records = records[:border]
val_records = records[border:]
with open(train_path, "w") as w:
    for record in train_records:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
with open(val_path, "w") as w:
    for record in val_records:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

print('Done!')
