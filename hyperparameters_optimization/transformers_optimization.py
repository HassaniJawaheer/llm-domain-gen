from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.integration import TransformersPruningCallback

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model


# constants
ATTN_ONLY  = ["q_proj","k_proj","v_proj","o_proj"]


# helpers
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def subsample_dataset(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    train_prop: float,
    eval_prop: float,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    def _n_keep(n_total: int, prop: float) -> int:
        n = int(n_total * prop)
        return max(1, n)
    
    # train data
    n = _n_keep(len(train_dataset), train_prop)
    train_ds = train_dataset.shuffle(seed).select(range(n))

    # eval data
    n = _n_keep(len(eval_dataset), eval_prop)
    eval_ds = eval_dataset.shuffle(seed).select(range(n))

    return train_ds, eval_ds

# LoRA builder
def build_lora_adapter(
    model,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float
):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    return model


# Trainer builder
def build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    *,
    output_dir: str,
    max_seq_length: int = 100,
    bf16: bool = True,
    max_steps: Optional[int] = None,
    num_train_epochs: Optional[float] = None,
    per_device_train_batch_size: int = 8,
    grad_accum: int = 16,
    learning_rate: float = 2e-4,
    eval_steps: int = 50,
    logging_steps: int = 50,
    warmup_ratio: float = 0.1,
    optim: str = "paged_adamw_32bit",
    seed: int = 42,
    report_to: str = "none",
    gradient_checkpointing: bool = True,
):

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_steps=10_000,  # do not save mid-run
        bf16=bf16,
        optim=optim,
        seed=seed,
        report_to=report_to,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="input_ids",
        max_seq_length=max_seq_length,
        args=args,
        packing=False
    )
    return trainer


# search space (small, relevant)
def suggest_params(trial: optuna.Trial) -> Dict:
    r = trial.suggest_categorical("r", [8, 16, 32])
    alpha_ratio = trial.suggest_categorical("alpha_ratio", ["1xr","2xr"])
    alpha = r if alpha_ratio == "1xr" else 2*r
    dropout = trial.suggest_float("dropout", 0.03, 0.10)
    modules = trial.suggest_categorical("modules", ["attn_only","all_linear"])
    lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    target = ATTN_ONLY if modules == "attn_only" else ALL_LINEAR
    return dict(r=r, alpha=alpha, dropout=dropout, target_modules=target, lr=lr)


# Phase A: HPO on subsample (1 short epoch)
def run_phase_a(
    *,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    load_model_fn: Callable[[], object],
    base_model_path: str = "model-based"
    out_dir: str = "optim_logs_out_phase_a",
    trials: int = 12,
    train_prop: float = 0.10,
    eval_prop: float = 0.02,
    one_epoch: bool = True,      
    max_steps: int = 300, 
    batch: int = 8,
    accum: int = 8,
    eval_steps: int = 50,
    seed: int = 42,
    pruner: Optional[optuna.pruners.BasePruner] = None
) -> Dict:
    """
    Returns dict with best and top-3 configs.
    """
    set_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = subsample_dataset(train_dataset, eval_dataset, train_prop, eval_prop, seed)

    base_model, tokenizer = load_model_fn(model_path=base_model_path)
    
    # Objective for Optuna
    def objective(trial: optuna.Trial):
        hp = suggest_params(trial)
        tokenizer = tokenizer
        model = build_lora_adapter(
            base_model,
            hp["target_modules"],
            hp["r"],
            hp["alpha"],
            hp["dropout"]
        )

        trainer = build_trainer(
            model, tokenizer, train_ds, eval_ds,
            output_dir=f"{out_dir}/trial_{trial.number}",
            max_seq_length=128, bf16=True,
            num_train_epochs=1 if one_epoch else None,
            max_steps=None if one_epoch else max_steps,
            per_device_train_batch_size=batch,
            grad_accum=accum,
            learning_rate=hp["lr"],
            eval_steps=eval_steps,
            logging_steps=eval_steps,
            warmup_ratio=0.1,
            seed=seed,
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer.add_callback(TransformersPruningCallback(trial, "eval_loss"))
        trainer.train()
        metrics = trainer.evaluate()
        trial.set_user_attr("params", hp)
        return metrics["eval_loss"]

    # Study
    if pruner is None:
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    # Collect best + top-3
    best = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.user_attrs.get("params", {})
    }
    trials_sorted = sorted(study.trials, key=lambda tr: tr.value if tr.value is not None else float("inf"))
    top = []
    for tr in trials_sorted[:3]:
        top.append({"number": tr.number, "value": tr.value, "params": tr.user_attrs.get("params", {})})

    # Save
    res_dir = Path(out_dir, "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "phase_a_best_and_top.json", "w") as f:
        json.dump({"best": best, "top": top}, f, indent=2)

    return {"best": best, "top": top, "file": str(res_dir / "phase_a_best_and_top.json")}


# --------- Phase B: full 1-epoch on full data for top-K
def run_phase_b(
    *,
    # Either pass ready datasets OR lists of dicts.
    train_dataset: Dataset,
    eval_dataset: Dataset,
    load_model_fn: Callable[[], object],
    base_model_path: str = "model-based"
    top_file: str = "optim_logs_out_phase_a/results/phase_a_best_and_top.json",
    out_dir: str = "optim_logs_out_phase_b",
    k: int = 3,
    batch: int = 8,
    accum: int = 16,
    eval_steps: int = 200,
    seed: int = 42
) -> Dict:
    """
    Train 1 full epoch on full data for top-K configs from Phase A.
    Save models and metrics.
    """
    set_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load top-k
    with open(top_file) as f:
        data = json.load(f)
    tops = data["top"][:k]

    base_model, tokenizer = load_model_fn(model_path=base_model_path)
    
    results = []
    for i, entry in enumerate(tops):
        hp = entry["params"]
        tokenizer = tokenizer,
        model = build_lora_adapter(base_model, hp["target_modules"], hp["r"], hp["alpha"], hp["dropout"])

        tag = f"k{i}_r{hp['r']}_a{hp['alpha']}"
        run_dir = Path(out_dir, f"final_{tag}")
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer = build_trainer(
            model, tokenizer, train_dataset, eval_dataset,
            output_dir=str(run_dir),
            max_seq_length=128, bf16=True,
            num_train_epochs=1,                 # full 1-epoch
            per_device_train_batch_size=batch,
            grad_accum=accum,
            learning_rate=hp["lr"],
            eval_steps=eval_steps,
            logging_steps=eval_steps,
            warmup_ratio=0.1,
            seed=seed,
            report_to="none",
            gradient_checkpointing=True,
        )

        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(str(Path(run_dir, "model")))
        results.append({"tag": tag, "hp": hp, "metrics": metrics})

    with open(Path(out_dir, "phase_b_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return {"results": results, "file": str(Path(out_dir, "phase_b_results.json"))}
