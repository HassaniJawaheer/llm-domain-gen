from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from evaluate.generation_utils import load_model, inference
from evaluate.similarity_score import SimilarityScorer
from safety.safety_filter import is_inappropriate

app = FastAPI(title="Generate Domain API")

MODEL_PATH = "models/merged_models/Mistral-7B-v0.1-generate-domain_v3"
MAX_NEW_TOKENS = 10

model, tokenizer = load_model(model_path=MODEL_PATH)
scorer = SimilarityScorer()

class GenerateRequest(BaseModel):
    business_description: str = Field(..., min_length=1)
    n_candidates: int = Field(3, ge=1, le=10)

class Suggestion(BaseModel):
    domain: str
    confidence: float

class GenerateResponse(BaseModel):
    suggestions: List[Suggestion]
    status: str = "success"

# keep part before first dot
def clean_domain(s: str) -> str:
    s = (s or "").strip().split()[0]
    i = s.find(".")
    return s[:i] if i != -1 else s

@app.post("/generate_domain", response_model=GenerateResponse)
def generate_domain(req: GenerateRequest):
    desc = req.business_description.strip()
    if not desc:
        raise HTTPException(status_code=400, detail="business_description is empty")

    if is_inappropriate(desc):
        return {
            "suggestions": [],
            "status": "blocked",
            "message": "Request contains inappropriate content"
        }
    
    suggestions: List[Suggestion] = []

    for _ in range(req.n_candidates):
        out = inference(
            model=model,
            tokenizer=tokenizer,
            business_description=desc,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        gen = out.get("generated_domain")
        print(gen)
        if not gen:
            continue

        domain_base = clean_domain(gen)
        if not domain_base:
            continue

        try:
            conf = float(scorer.score(desc, domain_base))
        except Exception:
            continue

        suggestions.append(Suggestion(domain=domain_base, confidence=conf))

    if not suggestions:
        raise HTTPException(status_code=502, detail="no suggestion generated")

    suggestions.sort(key=lambda x: x.confidence, reverse=True)

    return GenerateResponse(suggestions=suggestions, status="success")
