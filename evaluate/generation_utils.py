import random
import re
from typing import List, Dict, Tuple
import torch
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", re.IGNORECASE)

def load_model(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def generate(
    model,
    tokenizer,
    data,
    proportion: float = 1.0,
    seed: int = 42,
    max_new_tokens: int = 12
):
    random.seed(seed)
    examples = [{"business_description": ex["business_description"]} for ex in data]
    sample_size = int(len(examples) * proportion)
    selected = random.sample(examples, sample_size) if proportion < 1.0 else examples

    results = []
    for ex in tqdm(selected, desc="Génération"):
        prompt = ex["business_description"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature = 1.5,
                top_p = 0.95,
                top_k = 50,
                typical_p = 0.9,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            )

        gen_ids = outputs[0][input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        m = DOMAIN_RE.search(gen_text)
        if m:
            domain = m.group(0).lower()
        else:
            cand = gen_text.strip().split()[0] if gen_text.strip() else ""
            domain = cand.strip(".,);:!/\\\"'").lower()

        results.append({
            "business_description": ex["business_description"],
            "generated_domain": domain,
            "raw_generation": gen_text  # for debug
        })

    return results

def inference(
    model,
    tokenizer,
    business_description: str,
    max_new_tokens: int = 12,
):
    inputs = tokenizer(business_description, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        )

    gen_ids = outputs[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    m = DOMAIN_RE.search(gen_text)
    if m:
        domain = m.group(0).lower()
    else:
        cand = gen_text.strip().split()[0] if gen_text.strip() else ""
        domain = cand.strip(".,);:!/\\\"'").lower()

    return {
        "business_description": business_description,
        "generated_domain": domain,
        "raw_generation": gen_text
    }