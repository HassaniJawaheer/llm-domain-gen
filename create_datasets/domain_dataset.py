import os
import json
import re
import requests
from typing import List, Dict, Tuple
from datetime import datetime


class DomainDataset:
    def __init__(self, data_dir: str = "data", from_scratch: bool = True):
        self.data_dir = data_dir
        self.from_scratch = from_scratch
        os.makedirs(data_dir, exist_ok=True)

    def _get_attempt_and_filename(self) -> Tuple[str, str, str, List[Dict]]:
        attempts = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("attempt_")
        ])

        if self.from_scratch or not attempts:
            new_index = int(attempts[-1].split("_")[1]) + 1 if attempts else 0
            attempt_id = f"attempt_{new_index}"
            return attempt_id, "domain_dataset_v0.json", None, []
        else:
            attempt_id = attempts[-1]
            path = os.path.join(self.data_dir, attempt_id)
            versions = sorted([
                f for f in os.listdir(path)
                if f.startswith("domain_dataset_v") and f.endswith(".json")
            ])
            if not versions:
                raise ValueError(f"No dataset file found in {path}")
            last_file = versions[-1]
            last_index = int(last_file.split("_v")[1].split(".")[0])
            new_file = f"domain_dataset_v{last_index + 1}.json"
            with open(os.path.join(path, last_file), "r", encoding="utf-8") as f:
                previous_data = json.load(f)
            return attempt_id, new_file, last_file, previous_data

    def _build_prompt(self) -> str:
        return f"""You are a domain name generator.

Your task is to invent 20 fictitious businesses. For each business, write:

1. A short and realistic business description (between 1 and 5 sentences), starting with the line: `Description:`
2. Then generate 5 original and relevant domain name ideas, each on its own line, starting with: `Domains:`.

Please follow exactly this format:

Business 1
Description: ...
Domains:
- domain1.com
- domain2.com
...

Business 2
Description: ...
Domains:
- ...
...

Do not skip or repeat any fields. Return plain text only.""".strip()

    def _call_llm(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 7000,
            "temperature": 1.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

        response = requests.post(url, headers=headers, json=payload)
        if not response.ok:
            print(f"Groq returned {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def generate(self, n: int) -> None:
        attempt_id, target_file, parent_file, previous_data = self._get_attempt_and_filename()
        path = os.path.join(self.data_dir, attempt_id)
        os.makedirs(path, exist_ok=True)
        new_data = []
        
        for i in range(n):
            prompt = self._build_prompt()
            raw_output = self._call_llm(prompt)
            gen_data = self.parse_llm_output(raw_output)
            new_data.extend(gen_data)
        
        full_data = previous_data + new_data
        with open(os.path.join(path, target_file), "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        metadata = {
            "attempt_id": attempt_id,
            "latest_file": target_file,
            "from_scratch": self.from_scratch,
            "parent_file": parent_file,
            "num_entries_total": len(full_data),
            "created_at": datetime.now().isoformat()
        }

        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @staticmethod
    def parse_llm_output(text: str) -> List[Dict]:
        blocks = re.split(r"Business\s+\d+", text)
        results = []

        for block in blocks:
            if "Description:" not in block or "Domains:" not in block:
                continue

            match = re.search(r"Description:\s*(.+?)\s*Domains:", block, re.DOTALL)
            if not match:
                continue

            description = match.group(1).strip()
            domains = re.findall(r"-\s*([a-zA-Z0-9\.-]+\.[a-z]{2,})", block)

            results.append({
                "business_description": description,
                "domain": domains
            })

        return results

    def list_versions(self) -> List[str]:
        return sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("attempt_")
        ])

    def get_metadata(self, version: str) -> Dict:
        path = os.path.join(self.data_dir, version, "metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No metadata found for version: {version}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

