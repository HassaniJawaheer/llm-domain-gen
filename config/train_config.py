train_config = {
    "base_model": "model-based",

    # Training parameters
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 16,
    "eval_accumulation_steps": 16,
    "eval_strategy": "steps",  
    "eval_steps": 10,
    "logging_steps": 10,
    "save_steps": 10,
    "learning_rate": 0.00022868918637987284,
    "weight_decay": 0.038245380467858504,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.5544544885299565,
    "max_steps": -1,
    "warmup_ratio": 0.04682908034094774,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 96,
    "optimizer": "paged_adamw_8bit",
    "metric_for_best_model": "eval_loss",
    "load_best_model_at_end": True,
    "greater_is_better": False,
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.005,
    "dataset_text_field": "input_ids",
    "packing": False,
    "disable_checkpoints": False,
    
    # Quantization config
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },

    # LoRA config
    "lora_config": {
        "r": 32,
        "lora_alpha": 64,
        "bias": "all",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05
    }
}
