import os
import json
from typing import Dict, Any, Optional
import optuna
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MetricsLoggerCallback(TrainerCallback):
    """Write one JSON line per evaluation."""
    
    def __init__(self, out_dir: str, trial_id: str, hparams: Dict[str, Any]):
        self.out_dir = out_dir
        self.trial_id = trial_id
        self.hparams = dict(hparams)  # copy to avoid mutation
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_path = os.path.join(self.out_dir, "metrics.jsonl")
        
        # Write hyperparams once for reproducibility
        params_path = os.path.join(self.out_dir, "params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(self.hparams, f, indent=4, ensure_ascii=False)
    
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """Log metrics to JSONL file after each evaluation."""
        metrics: Dict[str, Any] = kwargs.get("metrics", {})
        
        record: Dict[str, Any] = {
            "trial_id": self.trial_id,
            "step": state.global_step,  # absolute optimization steps
            "epoch": state.epoch,
            **metrics,
        }
        
        # JSONL: one record per line (easy to grep/load)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return control


class OptunaPruningCallback(TrainerCallback):
    """Report eval metric to Optuna and prune early if bad."""
    
    def __init__(self, trial: optuna.trial.Trial, metric_name: str = "eval_loss"):
        self.trial = trial
        self.metric_name = metric_name
    
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """Report intermediate results to Optuna and handle pruning."""
        metrics: Dict[str, Any] = kwargs.get("metrics", {})
        value: Optional[float] = metrics.get(self.metric_name)
        
        if value is None:
            print(f"Warning: Metric '{self.metric_name}' not found in evaluation metrics")
            return control
        
        step = state.global_step
        
        # Report intermediate result to Optuna
        self.trial.report(value, step)
        
        # Stop this trial early if underperforming
        if self.trial.should_prune():
            message = f"Pruned at step {step} with {self.metric_name}={value:.6f}"
            print(f"Trial pruned: {message}")
            raise optuna.TrialPruned(message)
        
        return control


class OOMGuardCallback(TrainerCallback):
    """Light VRAM hygiene: free CUDA cache after heavy events."""
    
    def _empty_cuda_cache(self) -> None:
        """Safely empty CUDA cache to prevent memory fragmentation."""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to empty CUDA cache: {e}")
    
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """Clear CUDA cache after evaluation."""
        self._empty_cuda_cache()
        return control
    
    def on_save(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """Clear CUDA cache after saving."""
        self._empty_cuda_cache()
        return control
    
    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ) -> TrainerControl:
        """Clear CUDA cache at the end of training."""
        self._empty_cuda_cache()
        return control
