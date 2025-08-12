from transformers import TrainingArguments, EarlyStoppingCallback

def create_training_args(output_dir: str, config: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),

        # Batch sizes
        per_device_train_batch_size=config.get("per_device_train_batch_size", config.get("batch_size", 16)),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),

        # Gradient accumulation
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        eval_accumulation_steps=config.get("eval_accumulation_steps", 16),

        # Evaluation strategy
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", None),

        # Optimizer and scheduler
        optim=config.get("optimizer", "paged_adamw_32bit"),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.3),

        # Logging and saving
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 50),
        
        # save best model
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        
        # Regularization
        weight_decay=config.get("weight_decay", 0.001),
        max_grad_norm=config.get("max_grad_norm", 0.3),

        # Mixed precision
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),

        # Steps control
        max_steps=config.get("max_steps", -1),

        # Sequence handling
        group_by_length=config.get("group_by_length", True)
    )

def get_early_stopping_callback(config: dict):
    return EarlyStoppingCallback(
        early_stopping_patience=config.get("early_stopping_patience", 3),
        early_stopping_threshold=config.get("early_stopping_threshold", 0.0)
    )
