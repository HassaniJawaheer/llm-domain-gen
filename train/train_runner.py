import os
import json
import torch
from datetime import datetime
from transformers import BitsAndBytesConfig
from trl import SFTTrainer

from train.model_loader import load_model_4bit, load_tokenizer, apply_lora
from train.trainer_utils import create_training_args
from train.train_utils import get_next_attempt_id, save_metadata, save_losses


def train_model(train_dataset, eval_dataset, train_config: dict):
    base_model = train_config.get("base_model", "models-based")

    # Prepare new training folder
    base_path = os.path.join(os.path.dirname(__file__), "attempts")
    attempt_id = get_next_attempt_id(base_path)
    attempt_path = os.path.join(base_path, attempt_id)
    os.makedirs(attempt_path, exist_ok=True)

    # Tokenizer
    tokenizer = load_tokenizer(base_model)

    # BitsAndBytes config
    quant_cfg = train_config.get("quantization", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", False),
    )

    # LoRA Model
    model = load_model_4bit(base_model, bnb_config)
    model = apply_lora(model)

    # Training model
    training_args = create_training_args(
        output_dir=attempt_path,
        config=train_config
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=model.peft_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=train_config.get("max_seq_length", 256),
        dataset_text_field="text",
        packing=False,
    )

    # Training
    trainer.train()

    # Backup of losses
    train_losses = trainer.state.log_history
    loss_log = {
        "train_loss": [e["loss"] for e in train_losses if "loss" in e],
        "eval_loss": [e["eval_loss"] for e in train_losses if "eval_loss" in e]
    }
    save_losses(attempt_path, loss_log)

    # Backup of metadata
    metadata = {
        "attempt_id": attempt_id,
        "base_model": base_model,
        "created_at": datetime.now().isoformat(),
        "train_size": len(train_dataset),
        "val_size": len(eval_dataset),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "max_seq_length": train_config.get("max_seq_length", 256),
        "save_steps": training_args.save_steps
    }
    save_metadata(attempt_path, metadata)

    print(f"Training completed for {attempt_id}.")
    return attempt_path
