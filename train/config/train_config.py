train_config = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",

    "num_train_epochs": 3,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
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
    "max_seq_length": 256,
    "optimizer": "paged_adamw_32bit",

    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },

    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}
