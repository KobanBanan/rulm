import json
import random
import sys

from datasets import load_dataset

dataset_name = sys.argv[1]
train_path = sys.argv[2]
val_path = sys.argv[3]

print(f'dataset {dataset_name}')
print(f'train_path {train_path}')

records = []

for row in load_dataset(dataset_name, split="train"):
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    records.append(row)

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
