from __future__ import annotations
from typing import Tuple, Optional, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

QuantMode = Literal["4bit", "8bit", "none"]

def load_causal_model(
    model_path: str,
    quant_mode: QuantMode = "4bit",        # "4bit" | "8bit" | "none"
    # 4-bit config
    bnb_4bit_use_double_quant: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None,  # None → auto bf16/fp16
    # device / memory
    device_map: str = "auto",
    max_memory_gib: Optional[int] = None,
    allow_cpu_offload: bool = False,       # True → may be slow, but avoids OOM
    # dtype / attention
    torch_dtype: Optional[torch.dtype] = None,   # None → auto bf16/fp16
    attn_implementation: str = "sdpa",
    # misc
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Causal LLM."""

    # choose dtype
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = None

    # build quant config
    quant_cfg = None
    if quant_mode in ("4bit", "8bit"):
        # 4/8-bit config
        kwargs = {}
        if quant_mode == "4bit":
            if bnb_4bit_compute_dtype is None:
                bnb_4bit_compute_dtype = torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float16
            kwargs.update(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
        else:
            kwargs.update(load_in_8bit=True)

        # int8 CPU offload
        if allow_cpu_offload and quant_mode == "8bit":
            kwargs["llm_int8_enable_fp32_cpu_offload"] = True

        quant_cfg = BitsAndBytesConfig(**kwargs)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token  # simple pad fallback
    tokenizer.padding_side = "right"

    # limit VRAM to avoid silent CPU placement
    max_memory = None
    if max_memory_gib is not None and torch.cuda.is_available():
        max_memory = {0: f"{max_memory_gib}GiB"}

    # model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # fail fast if CPU is present while offload not allowed
    devmap = getattr(model, "hf_device_map", None)
    if not allow_cpu_offload and isinstance(devmap, dict) and any("cpu" in str(v).lower() for v in devmap.values()):
        raise RuntimeError(
            f"CPU found in device map: {devmap}. Lower batch/max_new_tokens or raise max_memory_gib to stay GPU-only."
        )

    return model, tokenizer
