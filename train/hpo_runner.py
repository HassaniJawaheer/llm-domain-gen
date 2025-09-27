import os
import time
import torch
from datetime import datetime
from trl import SFTTrainer

from utils.model_loader import load_causal_model 
from utils.lora_utils import apply_lora 
from train.trainer_utils import create_training_args 
from train.train_utils import get_next_attempt_id, save_metadata, save_losses 
from train.callbacks import MetricsLoggerCallback, OptunaPruningCallback, OOMGuardCallback

def train_model(
    train_dataset, 
    eval_dataset,  
    train_config: dict, 
    models_dir: str, 
    trial=None, 
    trial_id: str = None,
    out_dir: str | None = None
):
    base_model = train_config.get("base_model", "models-based")

    # Decide output dir
    if out_dir is not None:
        attempt_path = out_dir
        os.makedirs(attempt_path, exist_ok=True)
        attempt_id = os.path.basename(attempt_path)
    else:
        base_path = os.path.join(models_dir, "weights")
        attempt_id = get_next_attempt_id(base_path)
        attempt_path = os.path.join(base_path, attempt_id)
        os.makedirs(attempt_path, exist_ok=True)

    # BitsAndBytes config (QLoRA 4-bit)
    quant_cfg = train_config.get("quantization", {})  
    
    # Resolve loader parameters from config
    runtime_cfg = train_config.get("runtime", {})  # read runtime block
    
    # Map string dtype to torch dtype
    _dtype_map = {  # simple name to dtype map
        None: None,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    
    # Decide quant mode
    if quant_cfg.get("load_in_4bit", True):  
        quant_mode = "4bit"  
    elif quant_cfg.get("load_in_8bit", False):  
        quant_mode = "8bit" 
    else:
        quant_mode = "none"
    
    # Extract 4bit args even if quant mode is not 4bit, loader will ignore when not used
    bnb_4bit_use_double_quant = quant_cfg.get("bnb_4bit_use_double_quant", False)
    bnb_4bit_quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
    bnb_4bit_compute_dtype = _dtype_map.get(
        str(quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")).lower(), torch.bfloat16
    )

    # General loader args
    device_map = runtime_cfg.get("device_map", "auto")
    max_memory_gib = runtime_cfg.get("max_memory_gib")
    allow_cpu_offload = runtime_cfg.get("allow_cpu_offload", False)
    torch_dtype = _dtype_map.get( 
        None if runtime_cfg.get("torch_dtype") is None else str(runtime_cfg.get("torch_dtype")).lower(),
        torch.bfloat16,
    )

    attn_implementation = runtime_cfg.get("attn_implementation", "sdpa")
    trust_remote_code = bool(train_config.get("trust_remote_code", False))

    # Modèle 4-bit + Tokenizer + LoRA
    model, tokenizer = load_causal_model(
        model_path=base_model,
        quant_mode=quant_mode, 
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,  
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  
        device_map=device_map,
        max_memory_gib=max_memory_gib,
        allow_cpu_offload=allow_cpu_offload,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    ) 
    model = apply_lora(model, train_config.get("lora_config", None))  # apply LoRA if provided

    # Anti-OOM
    if hasattr(model, "config"):  # check presence of config
        setattr(model.config, "use_cache", False)  # disable cache to reduce eval memory

    # Training args
    training_args = create_training_args(  
        output_dir=attempt_path, 
        config=train_config  
    ) 

    # Callbacks
    callbacks = [ 
        OOMGuardCallback(),  # guard against OOM
        MetricsLoggerCallback(out_dir=attempt_path, trial_id=trial_id or attempt_id, hparams=train_config), 
    ]

    if trial is not None:  # if running with optuna
        metric_name = training_args.metric_for_best_model or "eval_loss"
        callbacks.append(OptunaPruningCallback(trial=trial, metric_name=metric_name))

    # Trainer TRL
    trainer = SFTTrainer(  
        model=model,  
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,  
        tokenizer=tokenizer,  
        callbacks=callbacks,  
        args=training_args, 
        max_seq_length=train_config.get("max_seq_length", 256),
        dataset_text_field=train_config.get("dataset_text_field","input_ids"),  
        packing=train_config.get("packing", False),
    )

    # Training
    t0 = time.time()  # start timer
    trainer.train()  # run training loop
    runtime_s = time.time() - t0  # compute runtime

    # Losses history
    train_losses = trainer.state.log_history 
    loss_log = {  # extract loss series
        "train_loss": [e["loss"] for e in train_losses if "loss" in e],
        "eval_loss": [e["eval_loss"] for e in train_losses if "eval_loss" in e],
    } 
    save_losses(attempt_path, loss_log)  # persist losses to disk

    # Métrique best 
    best_eval_loss = None  
    if "eval_loss" in str(trainer.state.best_metric) or isinstance(trainer.state.best_metric, float):  
        best_eval_loss = trainer.state.best_metric 
    else:  
        best_vals = [e["eval_loss"] for e in train_losses if "eval_loss" in e]  
        best_eval_loss = min(best_vals) if best_vals else None 

    # Métadata
    metadata = { 
        "attempt_id": attempt_id,  # attempt identifier
        "base_model": base_model, 
        "created_at": datetime.now().isoformat(),  
        "train_size": len(train_dataset),  
        "val_size": len(eval_dataset),  
        "epochs": training_args.num_train_epochs,  
        "batch_size": training_args.per_device_train_batch_size,  
        "max_seq_length": train_config.get("max_seq_length", 128),  
        "save_steps": training_args.save_steps, 
        "runtime_s": runtime_s,  
        "best_eval_loss": best_eval_loss,
        "checkpoints_disabled": bool(train_config.get("disable_checkpoints", False)),
    }  
    save_metadata(attempt_path, metadata)  

    return model, attempt_path  # return trained model and output path

