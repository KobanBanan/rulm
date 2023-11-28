import json
import random
import sys

from datasets import load_dataset
from tqdm import tqdm

from bad_substrings import has_bad_ss
from prompt_generator import PromptGenerator

prompt_generator = PromptGenerator()


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def build_char_system_messages(char):
    name = char["name"]
    greeting = char["greeting"]
    example_dialogue = char["example_dialogue"]

    context = ""
    if random.random() < 0.5:
        context += f"Ты {name}. "
    context += f"{char['context']}"
    chat = []
    if random.random() < 0.2:
        context += f"\nПриветствие: {greeting}"
        chat.append({
            "role": "bot",
            "content": greeting
        })
    if random.random() < 0.2:
        mapping = {
            "user": "Пользователь",
            "char": "Персонаж"
        }
        example_messages = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue]
        context += "\nПример диалога:\n" + "\n".join(example_messages)
    chat.insert(0, {
        "role": "system",
        "content": context
    })
    return chat


def main(train_path, val_path, open_chat=False):
    random.seed(42)

    instruct_records = []
    for row in tqdm(load_dataset("lksy/ru_instruct_gpt4", split="train")):
        if random.random() > 0.3:
            continue
        message = row["instruction"]
        if row["input"]:
            message += "\nДано: " + row["input"]
        output = row["full_output"]
        if not output:
            continue
        if has_bad_ss([{"content": output}]):
            continue
        instruct_records.append({
            "messages": [
                {"role": "user", "content": message, 'search': False},
                {"role": "bot", "content": output, 'search': False}
            ],
            "source": "gpt4"
        })
    print("Instruct gpt4 count:", len(instruct_records))
    print("Instruct gpt4 length:", calc_max_length(instruct_records))
    records = instruct_records

    saiga_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_saiga", split="train")):
        messages = revert_flattening(row["messages"])
        if has_bad_ss(messages):
            continue
        if row["model_name"] != "gpt-4":
            continue
        if random.random() > 0.5:
            continue
        saiga_records.append({
            "messages": messages,
            "source": "saiga",
            'search': False
        })
    print("Saiga count:", len(saiga_records))
    print("Max Saiga length:", calc_max_length(saiga_records))
    records += saiga_records

    sharegpt_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_sharegpt_cleaned", split="train")):
        messages = revert_flattening(row["messages"])
        text_length = sum([len(m["content"]) for m in messages])
        while text_length > 10000 and messages:
            messages = messages[:-2]
            text_length = sum([len(m["content"]) for m in messages])
        if not messages:
            continue
        sharegpt_records.append({
            "messages": messages,
            "source": "sharegpt",
            'search': False
        })
    print("ShareGPT count:", len(sharegpt_records))
    print("ShareGPT max length:", calc_max_length(sharegpt_records))
    records += sharegpt_records

    oasst_records = []
    for row in tqdm(load_dataset("IlyaGusev/oasst1_ru_main_branch", split="train")):
        messages = revert_flattening(row["messages"])
        text_length = sum([len(m["content"]) for m in messages])
        while text_length > 10000 and messages:
            messages = messages[:-2]
            text_length = sum([len(m["content"]) for m in messages])
        if not messages:
            continue
        oasst_records.append({
            "messages": messages,
            "source": "oasst",
            'search': False
        })
    print("OASST count:", len(oasst_records))
    print("OASST max length:", calc_max_length(oasst_records))
    records += oasst_records

    rp_records = []
    for row in tqdm(load_dataset("IlyaGusev/gpt_roleplay_realm", split="ru")):
        for dialogue in row["dialogues"]:
            if dialogue["model_name"] != "gpt-4":
                continue
            chat = dialogue["chat"]
            for message in chat:
                if message["role"] == "char":
                    message["role"] = "bot"
                if message["role"] == "operator":
                    message["role"] = "user"

            system_messages = build_char_system_messages(row)
            chat = system_messages + chat
            rp_records.append({
                "messages": chat,
                "source": "roleplay",
                'search': False
            })
    print("Roleplay count:", len(rp_records))
    print("Roleplay max length:", calc_max_length(rp_records))
    records += rp_records

    lima_records = []
    lima_role_mapping = {
        "human": "user",
        "gpt": "bot"
    }
    for row in tqdm(load_dataset("64bits/lima_vicuna_format", split="train")):
        chat = row["conversations"]
        fixed_messages = [{
            "role": "system",
            "content": "You are a virtual assistant that wants to be helpful"
        }]
        for message in chat:
            fixed_messages.append({
                "role": lima_role_mapping[message["from"]],
                "content": message["value"]
            })
        lima_records.append({
            "messages": fixed_messages,
            "source": "lima",
            'search': False
        })
    print("LIMA count:", len(lima_records))
    print("LIMA max length:", calc_max_length(lima_records))
    records += lima_records

    print("All count:", len(records))
    print("All max length:", calc_max_length(records))

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
    print(f'Number of records to save: train {len(train_records)}, val {len(val_records)}')
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def to_open_chat_format(records_):
    result = []
    for rec in records_:

        res = {"items": [], 'system': prompt_generator.dialog_prompt}
        dialogue = rec['messages']

        for message in dialogue:
            template = {"role": "", "content": "", "weight": ""}
            if message['role'] == 'system':
                continue

            if message['role'] == 'user':
                template["role"] = "user"
                template["content"] = message["content"]
                template['weight'] = 0.0
                res['items'].append(template)

            if message['role'] == 'bot':
                template['role'] = 'assistant'
                template["content"] = message["content"]
                template['weight'] = 1.0
                res['items'].append(template)

        result.append(res)

    with open(train_path, "w") as w:
        for record in result:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


train_path = sys.argv[1]
val_path = sys.argv[2]
main(train_path, val_path)
