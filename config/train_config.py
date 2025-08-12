train_config = {
    "base_model": "model-based",

    # Training parameters
    "num_train_epochs": 20,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "eval_accumulation_steps": 16,
    "eval_strategy": "steps",  
    "eval_steps": 25,
    "logging_steps": 25,
    "save_steps": 25,
    "learning_rate": 5e-5,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 100,
    "optimizer": "paged_adamw_32bit",
    "metric_for_best_model": "eval_loss",
    "load_best_model_at_end": True,
    "greater_is_better": False,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    

    # Quantization config
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },

    # LoRA config
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj","k_proj","v_proj","o_proj"],
        "lora_dropout": 0.05
    }
}
