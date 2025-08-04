import random
import numpy as np
from typing import List, Tuple, Dict
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityScorer:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(model=embedding_model_name)
    
    def score(self, description: str, domain: str) -> float:
        # Compute cosine similarity between description and domain name
        desc_emb = self.embedder.embed_query(description)
        domain_emb = self.embedder.embed_query(domain)
        return float(cosine_similarity([desc_emb], [domain_emb])[0][0])
    
    def evaluate_batch(
        self,
        data: Dataset,
        proportion: float = 0.5,
        seed: int = 42
    ) -> Tuple[float, List[Dict]]:
        # Sample a subset of the dataset
        random.seed(seed)
        data_dict = data.to_dict()
        examples = [
            {"description": data_dict["description"][i], "domain": data_dict["generated_domain"][i]}
            for i in range(len(data_dict["description"]))
        ]

        sample_size = int(len(examples) * proportion)
        selected = random.sample(examples, sample_size) if proportion < 1.0 else examples

        results = []
        scores = []
        for ex in selected:
            s = self.score(ex["description"], ex["domain"])
            scores.append(s)
            results.append({
                "description": ex["description"],
                "domain": ex["domain"],
                "confidence": s
            })
        
        mean_score = float(np.mean(scores)) if scores else 0.0
        return mean_score, results
