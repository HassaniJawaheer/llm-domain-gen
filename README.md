# **llm-domain-gen**

> **About**  
> An AI-powered tool that turns business ideas into domain name suggestions.  
> It ensures safe, relevant results with confidence levels.

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/HassaniJawaheer/llm-domain-gen.git
cd llm-domain-gen
````

2. **Make the setup script executable**

```bash
chmod +x setup_env.sh
```

3. **Run the setup script**

This will:

* Create Python virtual environments
* Activate them
* Install all required dependencies

```bash
bash setup_env.sh
```

4. **Download model**

Then run it to download the model:

```bash
bash download_model.sh
```

The chosen model is **Mistral-7B-v0.1**, available here:
[https://huggingface.co/mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

5. **Launch the notebook**

To use the notebook:

```bash
jupyter lab
```

Or open it directly from VS Code if Jupyter support is enabled.

> You can now run the notebook `domain_generator_project.ipynb`.

## **Quick Summary**

The dataset was generated with the **GROQ API** (LLaMA 3).
The model was fine-tuned using **LoRA**, a widely used parameter-efficient method.
Evaluation combined **cosine similarity** and **GPT-4 scoring** to assess the quality of generated domain names.

For more details on dataset creation, fine-tuning experiments, and evaluation, see the [experiment report](report.md).

## **API — `main.py`**

The project includes a simple **FastAPI** service with one `/generate` route.
You send a business description, and the API returns a list of suggested domain names with a confidence score.

Before processing, the API checks the description for explicit or inappropriate content using a keyword-based filter with regular expressions. If a match is found, the request is rejected; otherwise, it is processed normally.

**Input:**

```json
{
  "business_description": "Your business description",
  "n_candidates": 3
}
```

**Output:**

```json
{
  "suggestions": [
    { "domain": "domain_1.com", "confidence": 0.78 },
    { "domain": "domain_3.net", "confidence": 0.65 }
  ]
}
```

The confidence score is based on **cosine similarity** for cost efficiency. GPT-4 scoring was also tested and can be more accurate, but embeddings are preferred for budget and reproducibility.

To run the API:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

## **Fine-tuned Model**

The fine-tuned model is available on Hugging Face:

**Model name**: *Mistral-7B-v0.1-generate-domain_v6*
**Hugging Face link**: [https://huggingface.co/hassanij/domain-name-generator](https://huggingface.co/hassanij/domain-name-generator)

You can download it directly from Hugging Face.

## **Alternative Inference with vLLM**

Instead of using the custom FastAPI service described above, you can also serve the model with **vLLM**, a high-performance inference engine.
This is done with the script:

```bash
bash start_vllm.sh
```

By default, the server runs on **port 9001**.

* **Advantages**:

  * Can handle multiple requests at the same time
  * Optimized inference performance

* **Limitations**:

  * Does not include the security filtering implemented in the custom FastAPI API
  * Provides only raw inference endpoints
