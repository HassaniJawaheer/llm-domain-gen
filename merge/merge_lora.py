from pathlib import Path
import re, json, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
from peft import PeftModel

# --- utils ---
def _model_tag(model_name: str = "Mistral-7B-v0.1") -> str:
    return model_name

def _next_version_dir(base_dir: Path, model_tag: str, use_case: str):
    """Create next versioned directory: {model_tag}-{use_case}_v{N} with numeric increment."""
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(model_tag)}-{re.escape(use_case)}_v(\d+)$")
    existing = []
    for d in base_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                existing.append(int(m.group(1)))
    v = 0 if not existing else max(existing) + 1
    return base_dir / f"{model_tag}-{use_case}_v{v}", v

def _save_generation_config_from_model(model, out_dir: Path):
    """Save a generation config derived from the merged model's config (keeps pad_token_id consistent)."""
    try:
        gc = GenerationConfig.from_model_config(model.config)
        if getattr(model.config, "pad_token_id", None) is not None:
            gc.pad_token_id = model.config.pad_token_id
        gc.save_pretrained(out_dir)
    except Exception:
        pass

def _ensure_pad_token(tok, model=None):
    """If pad token is missing, prioritize unk_token, then fallback to eos_token. Also sync model.config.pad_token_id."""
    if tok.pad_token is None:
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        elif tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    if model is not None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

# --- main operation ---
def merge(
    base_model_id: str,
    adapter_path: str,
    output_base_path: str,
    use_case: str,
    model_name: str = "Mistral-7B-v0.1",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: dict | None = None,
) -> str:
    device_map = device_map or {"": 0}
    out_root = Path(output_base_path)
    model_tag = _model_tag(model_name)
    out_dir, version_idx = _next_version_dir(out_root, model_tag, use_case)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load base model (no quantization; bf16/fp16/fp32)
    cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=cfg,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 2) Attach LoRA and merge
    lora = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = lora.merge_and_unload()

    # 3) Save merged model (.safetensors)
    merged.save_pretrained(out_dir, safe_serialization=True)

    # 4) Tokenizer (pad = unk priority, else eos), then save
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    _ensure_pad_token(tok, model=merged)  # keep tokenizer and model.config in sync
    tok.save_pretrained(out_dir)

    # 5) Save generation_config derived from the merged model (not from the base)
    _save_generation_config_from_model(merged, out_dir)

    # 6) Manifest
    files = {p.name: p.stat().st_size for p in out_dir.glob("*") if p.is_file()}
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model_id": base_model_id,
        "model_tag": model_tag,
        "adapter_path": str(adapter_path),
        "output_dir": str(out_dir),
        "use_case": use_case,
        "version": version_idx,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "merged": True,
        "torch_version": torch.__version__,
        "transformers_version": sys.modules["transformers"].__version__,
        "peft_version": sys.modules["peft"].__version__,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(out_dir)