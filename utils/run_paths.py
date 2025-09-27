from pathlib import Path
from datetime import datetime

def make_optuna_attempt_root(models_dir: str, runs_root_name: str = "optuna_runs") -> str:
    # one “attempt” per sweep
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(models_dir) / runs_root_name / ts
    root.mkdir(parents=True, exist_ok=True)
    return str(root)

def make_trial_dir(attempt_root: str, trial_id: str) -> str:
    # one subdir per trial
    d = Path(attempt_root) / trial_id
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
