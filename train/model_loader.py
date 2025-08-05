import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_4bit(model_name: str, bnb_config: BitsAndBytesConfig):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable(use_reentrant=False)
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, lora_config_dict: dict = None):
    """
    Apply LoRA to the model using a config dict.
    """
    lora_config_dict = lora_config_dict or {}

    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 64),
        lora_alpha=lora_config_dict.get("lora_alpha", 128),
        bias=lora_config_dict.get("bias", "none"),
        task_type=lora_config_dict.get("task_type", "CAUSAL_LM"),
    )
    return get_peft_model(model, lora_config)
