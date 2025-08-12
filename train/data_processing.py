import json
from typing import List, Dict, Tuple
import re
from datasets import Dataset
from transformers import PreTrainedTokenizer
import random


def load_json_dataset(path: str) -> List[Dict]:
    """Load raw JSON data from file."""
    with open(path, 'r') as f:
        return json.load(f)

def flatten(grouped_data: Dict[str, List[Dict]]) -> List[Dict]:
    flattened = []
    for company, entries in grouped_data.items():
        for entry in entries:
            desc = entry["business_description"].strip()
            for domain in entry["domain"]:
                flattened.append({
                    "business_description": desc,
                    "domain": domain.strip()
                })
    return flattened

def group_by_company(data: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for item in data:
        desc = item["business_description"].strip()
        company = desc.split()[0].strip()
        domains = [d.strip() for d in item["domain"]]

        if company not in grouped:
            grouped[company] = []
        
        grouped[company].append({
            "business_description": desc,
            "domain": domains
        })
    return grouped

def format_for_sft(json_data: List[Dict]) -> List[Dict]:
    formatted = []
    for item in json_data:
        description = item['business_description'].strip()
        domain = item['domain'].strip()
        # domain = re.sub(r'\.[a-z]{2,}$', '', raw_domain, flags=re.IGNORECASE)
        instruction = description

        formatted.append({
            "input": instruction,
            "output": domain
        })

    return formatted
    
def tokenize_for_sft(
    example: Dict,
    tokenizer,
    max_length: int = 160
) -> Dict:
    # Tokenize input (no special tokens)
    in_ids = tokenizer.encode(example["input"], add_special_tokens=False)

    # Tokenize output + EOS
    out_ids = tokenizer.encode(example["output"], add_special_tokens=False) + [tokenizer.eos_token_id]

    # Concatenate
    ids = in_ids + out_ids

    # Labels: mask input part
    labels = [-100] * len(in_ids) + out_ids

    # Attention mask
    attn = [1] * len(ids)

    # Pad to max_length with unk_token_id (fallback to EOS if no unk token)
    pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    pad_len = max_length - len(ids)
    
    if pad_len > 0:
        ids    += [pad_id] * pad_len
        labels += [-100]   * pad_len
        attn   += [0]      * pad_len
    else:
        ids    = ids[:max_length]
        labels = labels[:max_length]
        attn   = attn[:max_length]

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels
    }

def convert_to_dataset(formatted_data: List[Dict]) -> Dataset:
    """Convert a list of dicts into a HuggingFace Dataset."""
    return Dataset.from_list(formatted_data)

def split_data(grouped_data: Dict[str, List[Dict]], test_size: float = 0.2, seed: int = 42) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    random.seed(seed)
    companies = list(grouped_data.keys())
    random.shuffle(companies)

    split_index = int(len(companies) * (1 - test_size))
    train_companies = companies[:split_index]
    val_companies = companies[split_index:]

    train_data = {c: grouped_data[c] for c in train_companies}
    val_data = {c: grouped_data[c] for c in val_companies}

    return train_data, val_data

def get_max_token_length(dataset: Dataset, tokenizer: PreTrainedTokenizer, col_name: str) -> int:
    """Return the maximum tokenized sequence length in the dataset."""
    return max(len(tokenizer.tokenize(text)) for text in dataset[col_name])


def filter_dataset_by_token_length(dataset: Dataset, tokenizer: PreTrainedTokenizer, col_name: str, max_tokens: int) -> Dataset:
    """Filter out samples where tokenized length exceeds max_tokens."""
    kept_indexes = [
        i for i, text in enumerate(dataset[col_name])
        if len(tokenizer.tokenize(text)) <= max_tokens
    ]
    return dataset.select(kept_indexes)