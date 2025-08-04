import random
from typing import List, Dict, Tuple, Union
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
    data: Union[Dataset, List[Dict]],
    proportion: float = 1.0,
    seed: int = 42,
    max_new_tokens: int = 30
) -> List[Dict]:
    random.seed(seed)

    if isinstance(data, Dataset):
        data = data.to_dict()
        examples = [{"description": data["description"][i]} for i in range(len(data["description"]))]
    else:
        examples = [{"description": d["description"]} for d in data]

    sample_size = int(len(examples) * proportion)
    selected = random.sample(examples, sample_size) if proportion < 1.0 else examples

    results = []
    for ex in selected:
        instruction = f"Use the following description to suggest a suitable domain name: {ex['description']}"
        prompt = f"<s>[INST] {instruction} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        domain = generated.split("Domain name:")[-1].strip().split()[0]

        results.append({
            "description": ex["description"],
            "generated_domain": domain
        })

    return results
