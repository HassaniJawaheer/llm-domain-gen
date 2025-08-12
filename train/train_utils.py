import os
import json
from pathlib import Path
from peft import PeftModel

def get_next_attempt_id(base_path: str = "attempts") -> str:
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    attempts = [d for d in os.listdir(base_path) if d.startswith("attempt_")]
    indexes = [int(d.split("_")[1]) for d in attempts if d.split("_")[1].isdigit()]
    next_index = max(indexes, default=-1) + 1
    return f"attempt_{next_index}"

def save_metadata(attempt_path: str, metadata: dict):
    with open(os.path.join(attempt_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

def save_losses(attempt_path: str, losses: dict):
    with open(os.path.join(attempt_path, "losses.json"), "w") as f:
        json.dump(losses, f, indent=4)

def save_weights(model, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deepspeed/DDP safety: unwrap if needed
    mdl = model.module if hasattr(model, "module") else model

    if isinstance(mdl, PeftModel) or hasattr(mdl, "peft_config"):
        # Save LoRA adapters only
        mdl.save_pretrained(out)  # will write adapter_model.safetensors + adapter_config.json
    else:
        # Save full model weights
        mdl.save_pretrained(out, safe_serialization=True)

    return str(out)