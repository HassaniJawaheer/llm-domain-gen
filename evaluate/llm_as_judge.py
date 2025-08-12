import os
import random
import time
import json
from typing import List, Dict, Tuple, Union
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def build_prompt(business_description: str, domain: str) -> str:
    return f"""You are an expert in brand naming and semantic evaluation.
Your task is to assess whether a generated domain name is a good fit for a given business description.
Your evaluation should follow the criteria below and result in a single confidence score between 0.0 (very poor fit) and 1.0 (perfect fit).

Business Description:
"{business_description}"

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
{{
  "confidence": <score between 0.0 and 1.0>
}}"""

def extract_confidence(response_text: str) -> float:
    try:
        obj = json.loads(response_text)
        val = float(obj.get("confidence", 0.0))
        return max(0.0, min(1.0, val))
    except Exception:
        try:
            # very simple fallback
            start = response_text.index('"confidence":') + len('"confidence":')
            end = response_text.index("}", start)
            return max(0.0, min(1.0, float(response_text[start:end].strip())))
        except Exception:
            return 0.0

def evaluate(
    data: Union[Dataset, List[Dict]],
    model_name: str = "gpt-4o-mini",
    proportion: float = 1.0,
    seed: int = 42,
    temperature: float = 0.0
) -> Tuple[float, List[Dict]]:
    random.seed(seed)

    if isinstance(data, Dataset):
        d = data.to_dict()
        n = len(d["business_description"])
        examples = [
            {
                "business_description": d["business_description"][i],
                "domain": d["generated_domain"][i],
            }
            for i in range(n)
        ]
    else:
        examples = [
            {
                "business_description": item["business_description"],
                "domain": item["generated_domain"],
            }
            for item in data
        ]

    if not examples:
        return 0.0, []

    sample_size = int(len(examples) * proportion)
    selected = random.sample(examples, sample_size) if 0.0 < proportion < 1.0 else examples

    results: List[Dict] = []
    scores: List[float] = []

    for ex in selected:
        prompt = build_prompt(ex["business_description"], ex["domain"])
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=60,
            )
            output = resp.choices[0].message.content
            score = extract_confidence(output)
        except Exception as e:
            score = 0.0
            

        scores.append(score)
        results.append({
            "business_description": ex["business_description"],
            "generated_domain": ex["domain"],
            "confidence": score,
        })

        time.sleep(1)

    mean_confidence = sum(scores) / len(scores)
    return mean_confidence, results