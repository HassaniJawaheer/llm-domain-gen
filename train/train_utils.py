import os
import json

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
