import os, json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import optuna
from datetime import datetime

from train.hpo_runner import train_model
from utils.run_paths import make_optuna_attempt_root, make_trial_dir


def define_search_space(trial):
    # LoRA rank
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    # Alpha = multiplier * r
    alpha_mult = trial.suggest_categorical("lora_alpha_mult", [2, 4, 8])
    lora_alpha = alpha_mult * lora_r

    params = {
        # Optim
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.02, 0.10),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial"]
        ),
        "optimizer": trial.suggest_categorical(
            "optimizer", ["adamw_torch", "paged_adamw_32bit", "paged_adamw_8bit"]
        ),

        # LoRA
        "lora_config": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.10),
            "bias": trial.suggest_categorical("bias", ["none", "lora_only", "all"]),
        },
        "lora_alpha_mult": alpha_mult,

        # Stabilizer
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }
    return params


def objective(
    trial: optuna.trial.Trial,
    train_ds,
    val_ds,
    fixed_cfg: Dict[str, Any],
    attempt_root: str,
) -> float:
    # Merge configs
    hparams = dict(fixed_cfg)
    hparams.update(define_search_space(trial))

    # Track LoRA alpha
    lora_alpha = hparams["lora_config"]["lora_alpha"]
    trial.set_user_attr("lora_alpha", lora_alpha)

    # Sweep defaults (light eval, no ckpt)
    hparams.setdefault("metric_for_best_model", "eval_loss")
    hparams.setdefault("eval_strategy", "steps")
    hparams.setdefault("save_steps", fixed_cfg.get("save_steps", 100))
    hparams.setdefault("max_seq_length", fixed_cfg.get("max_seq_length", 256))
    hparams.setdefault("logging_steps", fixed_cfg.get("logging_steps", 50))
    hparams.setdefault("optimizer", fixed_cfg.get("optimizer", "paged_adamw_32bit"))
    hparams.setdefault("disable_checkpoints", True)

    # Output dir for this trial
    trial_id = f"trial_{trial.number:04d}"
    out_dir = make_trial_dir(attempt_root, trial_id)

    # Run training (artifacts go to out_dir)
    _, used_out_dir = train_model(
        train_dataset=train_ds,
        eval_dataset=val_ds,
        train_config=hparams,
        models_dir=None,
        trial=trial,
        trial_id=trial_id,
        out_dir=out_dir,
    )

    # Read best metric from metadata
    metadata_path = os.path.join(used_out_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_eval_loss = meta.get("best_eval_loss")
    return float(best_eval_loss) if best_eval_loss is not None else float("inf")


# Helpers for storage
def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _make_storage_url(storage_dir: str, db_name: str = "optuna_study.db") -> str:
    storage_dir = _ensure_dir(storage_dir)
    db_path = os.path.join(storage_dir, db_name)
    return f"sqlite:///{db_path}"

def _create_or_load_study(
    study_name: str,
    direction: str,
    storage_url: str,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage_url,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )


def run_sweep(
    n_trials: int,
    train_ds,
    val_ds,
    fixed_cfg: Dict[str, Any],
    models_dir: str,
    direction: str = "minimize",
    storage_dir: Optional[str] = None,
    study_name: Optional[str] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    resume_study: bool = False,
    attempt_root: Optional[str] = None,
) -> Tuple[optuna.Study, Dict[str, Any]]:
    # 1) Create or reuse attempt root
    if resume_study:
        assert attempt_root is not None, "attempt_root is required when resume_study is True"
    else:
        attempt_root = make_optuna_attempt_root(models_dir, runs_root_name="optuna_runs")

    # Storage for SQLite DB
    if storage_dir is None:
        storage_dir = os.path.join(attempt_root, "optuna_db")
    storage_url = _make_storage_url(storage_dir, db_name="optuna_study.db")

    # Default study name
    if study_name is None:
        study_name = f"sweep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create or load study
    study = _create_or_load_study(
        study_name=study_name,
        direction=direction,
        storage_url=storage_url,
        pruner=pruner,
        sampler=sampler,
    )

    # Write manifest
    manifest = {
        "study_name": study.study_name,
        "storage_url": storage_url,
        "created_at": datetime.now().isoformat(),
        "attempt_root": attempt_root,
        "n_trials_planned": n_trials,
    }
    with open(os.path.join(attempt_root, "study_info.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    # Optimize
    study.optimize(
        lambda t: objective(t, train_ds, val_ds, fixed_cfg, attempt_root),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # Assemble best params
    best = study.best_trial
    best_params = dict(fixed_cfg)
    best_params.update(best.params)
    best_params["lora_config"] = {
        "r": best.params.get("lora_r"),
        "lora_alpha": best.params.get("lora_alpha"),
        "lora_dropout": best.params.get("lora_dropout"),
        "bias": best.params.get("bias"),
    }
    return study, best_params
