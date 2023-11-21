import json
import random
import sys

from datasets import load_dataset
from prompt_generator import PromptGenerator

from bad_substrings import has_bad_ss

prompt_generator = PromptGenerator()
search_dataset_name = sys.argv[1]
dialogue_dataset_name = sys.argv[2]
save_path = sys.argv[3]

print(f'search_dataset_name {search_dataset_name}')
print(f'dialogue_dataset_name: {dialogue_dataset_name}')
print(f'save_path {save_path}')

records = []

print(f'downloading {search_dataset_name}...')

print(f'preparing {search_dataset_name} sft dataset...')
for row in load_dataset(search_dataset_name, split="train"):
    instruction = row["instruction"]
    output = row["output"]
    if has_bad_ss([{"content": output}]):
        continue

    records.append({
        "items": [
            {"role": "user", "content": instruction, "weight": 0.0},
            {"role": "assistant", "content": output, "weight": 1.0}
        ],
        "system": prompt_generator.search_prompt
    })

# print(f'downloading {dialogue_dataset_name}...')
#
# print(f'preparing {dialogue_dataset_name} sft dataset...')
# for row in load_dataset(dialogue_dataset_name, split="train"):
#     instruction = row["instruction"]
#     output = row["output"]
#     if has_bad_ss([{"content": output}]):
#         continue
#
#     records.append({
#         "items": [
#             {"role": "user", "content": instruction, "weight": 0.0},
#             {"role": "assistant", "content": output, "weight": 1.0}
#         ],
#         "system": prompt_generator.search_prompt
#     })


print("Evol-instruct count:", len(records))

cleaned_records = []
for record in records:
    messages = record["items"]
    roles = {m["role"] for m in messages}
    for role in roles:
        assert role in ("assistant", "user", "system"), role
    if has_bad_ss(messages):
        continue
    if not record["items"]:
        continue
    cleaned_records.append(record)

records = cleaned_records
print("All count after cleaning:", len(records))

random.shuffle(records)
with open(save_path, "w") as w:
    for record in records:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

print('Done!')
