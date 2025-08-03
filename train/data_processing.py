import json
from typing import List, Dict, Tuple
import re
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_json_dataset(path: str) -> List[Dict]:
    """Load raw JSON data from file."""
    with open(path, 'r') as f:
        return json.load(f)

def flatten(json_data: List[Dict]) -> List[Dict]:
    flattened = []
    for item in json_data:
        desc = item["business_description"].strip()
        for domain in item["domains"]:
            domain_root = re.sub(r'\\.[a-z]{2,}$', '', domain.strip(), flags=re.IGNORECASE)
            flattened.append({
                "business_description": desc,
                "domain": domain_root
            })
    return flattened

def format_for_mistral(json_data: List[Dict]) -> List[Dict]:
    """
    Format entries for Mistral chat-style prompt
    """
    formatted = []
    for item in json_data:
        description = item['business_description'].strip()
        raw_domain = item['domain'].strip()
        domain = re.sub(r'\.[a-z]{2,}$', '', raw_domain, flags=re.IGNORECASE)

        instruction = f"Use the following description to suggest a suitable domain name: {description}"
        prompt = f"<s>[INST] {instruction} [/INST] {domain} </s>"

        formatted.append({"text": prompt})
    return formatted

def convert_to_dataset(formatted_data: List[Dict]) -> Dataset:
    """Convert a list of dicts into a HuggingFace Dataset."""
    return Dataset.from_list(formatted_data)

def split_dataset(dataset: Dataset, test_size: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and test sets."""
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]

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