train_config = {
    "base_model": "model-based",

    # Training parameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "eval_accumulation_steps": 16,
    "evaluation_strategy": "steps",  
    "eval_steps": 50,
    "logging_steps": 50,
    "save_steps": 50,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.3,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 128,
    "optimizer": "paged_adamw_32bit",

    # Quantization config
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },

    # LoRA config
    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}
