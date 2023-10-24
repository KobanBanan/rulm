import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(model_name: str, final_model_name: str, output_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = PeftConfig.from_pretrained(model_name)
    base_model_path = config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)

    if output_path:
        lora_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    lora_model.push_to_hub(final_model_name, private=True)
    tokenizer.push_to_hub(final_model_name, private=True)
