import random
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm  # version spéciale notebook


class SimilarityScorer:
    def __init__(self, embedding_model_name: str = "intfloat/e5-small-v2"):
        self.embedder = SentenceTransformer(embedding_model_name)
    
    def score(self, description: str, domain: str) -> float:
        # Compute cosine similarity between description and domain name
        desc_emb = self.embedder.encode(description)
        domain_emb = self.embedder.encode(domain)
        return float(cosine_similarity([desc_emb], [domain_emb])[0][0])
    
    def evaluate_batch(
        self,
        data: List[Dict],
        proportion: float = 0.5,
        seed: int = 42
    ) -> Tuple[float, List[Dict]]:
        # Sample a subset of the dataset
        random.seed(seed)

        sample_size = int(len(data) * proportion)
        selected = random.sample(data, sample_size) if proportion < 1.0 else data

        results = []
        scores = []

        for ex in tqdm(selected, desc="Évaluation"):
            s = self.score(ex["business_description"], ex["generated_domain"])
            scores.append(s)
            results.append({
                "business_description": ex["business_description"],
                "domain": ex["generated_domain"],
                "confidence": s
            })

        mean_score = float(np.mean(scores)) if scores else 0.0
        return mean_score, results
