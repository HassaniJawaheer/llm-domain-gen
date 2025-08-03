from transformers import TrainingArguments

def create_training_args(output_dir: str, config: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        optim=config.get("optimizer", "paged_adamw_32bit"),
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 50),
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.001),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        max_grad_norm=config.get("max_grad_norm", 0.3),
        max_steps=config.get("max_steps", -1),
        warmup_ratio=config.get("warmup_ratio", 0.3),
        group_by_length=config.get("group_by_length", True),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine")
    )
