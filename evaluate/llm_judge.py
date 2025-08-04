import random
import time
from typing import List, Dict, Tuple, Union
from datasets import Dataset
import openai


def build_prompt(description: str, domain: str) -> str:
    return f"""You are an expert in brand naming and semantic evaluation.  
Your task is to assess whether a generated domain name is a good fit for a given business description.  
Your evaluation should follow the criteria below and result in a single confidence score between 0.0 (very poor fit) and 1.0 (perfect fit).

Business Description:
"{description}"

Proposed Domain Name:
"{domain}"

Evaluation Criteria:
1. Semantic Fit – Does the domain name reflect the business concept clearly?
2. Readability – Is it easy to read, pronounce, and remember?
3. Brandability – Could it plausibly be used as a brand or website name?
4. Relevance – Is it appropriate, non-generic, and non-offensive?

Instructions:
- Analyze carefully each criterion.
- Provide a single confidence score in JSON format.
- Do not add any comments, explanations, or extra text.

Output Format:
```json
{{
  "confidence": <score between 0.0 and 1.0>
}}
```"""

def extract_confidence(response_text: str) -> float:
    try:
        start = response_text.index('"confidence":') + len('"confidence":')
        end = response_text.index("}", start)
        return float(response_text[start:end].strip())
    except:
        return 0.0  # fallback


def evaluate_with_llm(
    data: Union[Dataset, List[Dict]],
    model_name: str = "gpt-4",
    api_key: str = None,
    proportion: float = 1.0,
    seed: int = 42,
    temperature: float = 0.0
) -> Tuple[float, List[Dict]]:
    openai.api_key = api_key
    random.seed(seed)

    if isinstance(data, Dataset):
        data = data.to_dict()
        examples = [
            {"description": data["description"][i], "domain": data["generated_domain"][i]}
            for i in range(len(data["description"]))
        ]
    else:
        examples = [{"description": d["description"], "domain": d["generated_domain"]} for d in data]

    sample_size = int(len(examples) * proportion)
    selected = random.sample(examples, sample_size) if proportion < 1.0 else examples

    results = []
    scores = []

    for ex in selected:
        prompt = build_prompt(ex["description"], ex["domain"])

        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=50
            )
            output = response["choices"][0]["message"]["content"]
            score = extract_confidence(output)
        except Exception:
            score = 0.0

        scores.append(score)
        results.append({
            "description": ex["description"],
            "generated_domain": ex["domain"],
            "confidence": score
        })

        time.sleep(1)  # to avoid rate limits

    mean_confidence = sum(scores) / len(scores) if scores else 0.0
    return mean_confidence, results
