from typing import Dict, Optional
from peft import LoraConfig, get_peft_model

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def apply_lora(model, lora_config_dict: Optional[Dict] = None):
    cfg = lora_config_dict or {}
    lora_cfg = LoraConfig(
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        target_modules=cfg.get("target_modules", DEFAULT_TARGETS),
        bias=cfg.get("bias", "none"),
        task_type=cfg.get("task_type", "CAUSAL_LM"),
    )

    return get_peft_model(model, lora_cfg)