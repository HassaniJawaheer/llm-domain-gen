# **llm-domain-gen**

> **About**  
> An AI-powered tool that turns business ideas into domain name suggestions.  
> It ensures safe, relevant results with confidence levels.

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/HassaniJawaheer/llm-domain-gen.git
cd llm-domain-gen
```

2. **Make the setup script executable**

```bash
chmod +x setup_env.sh
```

3. **Run the setup script**

This will:

* Create a Python virtual environment
* Activate it
* Upgrade pip
* Install all required dependencies (from `requirements.txt`)

```bash
./setup_env.sh
```

4. **Download the model**

Make the download script executable:

```bash
chmod +x download_model.sh
```

Then run it to download the model:

```bash
./download_model.sh
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

## **Dataset Generation**

The dataset used in this project was **fully generated** using a language model.

At first, I considered using existing public datasets or scraping domain names and business descriptions from the web. However, I couldn’t find any satisfying dataset that matched the project goals. Scraping also turned out to be inefficient and time-consuming, without any guarantee of relevant results.

So I decided to generate the dataset from scratch.

### **Generation method**

The data was created using the **GROQ API**, with the model **LLaMA 3 70B**, which I had already used in a professional context. It provides good results and a large context window (8192 tokens), which is enough for this use case.

The prompt asks the model to:

1. Create a **fictitious business description** (1 to 5 sentences).
2. Generate **5 relevant domain name ideas** for each business.

Each data sample contains one description and a list of 5 domain names.

### **Data structure**

All data is saved in the `data/` folder.
It is organized in **attempt folders**, like this:

```
data/
├── attempt_0/
│   ├── domain_dataset_v0.json
│   ├── domain_dataset_v1.json
│   └── metadata.json
├── attempt_1/
│   └── ...
```

* Each `attempt` represents one generation session.
* Each session can have multiple **versions** (v0, v1, v2...) of the dataset.
* A new version is created every time we **add more samples** to an existing attempt.

This makes it easy to:

* Start a **new dataset** (fresh attempt).
* Or continue generation based on previous data (new version in the same attempt).

This approach also helps manage **API rate limits**, since GROQ is free but has daily request limits. Instead of generating everything at once, I can generate small batches of data over time.

### **How to use the generator**

The dataset generation logic is in the file `datasets/domain_dataset.py`.
You can use the generator from the notebook like this:

```python
from datasets.domain_dataset import DomainDataset

# Create a generator from scratch
generator = DomainDataset(from_scratch=True)

# Or continue from the last attempt
# generator = DomainDataset(from_scratch=False)

# Generate 500 new samples
generator.generate(n=500)
```

This will automatically:

* Create a new folder and version if needed
* Call the API and parse the results
* Save the data in JSON format
* Create metadata with timestamps and version info

## **Fine-tuning**

The model was fine-tuned using the **TRL** library, specifically the **SFT Trainer**.
Tokenization was handled manually without using external helpers.

For fine-tuning, **LoRA** was chosen, as it is the method already known and used in previous projects. No automated hyperparameter optimization was applied. Initial tests were started, but the code did not work as expected, so the idea was dropped.

In LoRA fine-tuning, common parameters to adjust include:

* **Rank (`r`)**
* **Alpha (`lora_alpha`)**
* **Dropout (`lora_dropout`)**
* **Learning rate**

The training process involved experimenting with these values and observing their effect on convergence. Based on the experiments reported here, some parameter combinations lead to better convergence, while others do not.

## **Evaluation**

Beyond the training loss, two main methods were used to evaluate the fine-tuned model.

1. **Cosine Similarity**
   This method checks whether the generated domain name is semantically close to the business description. While useful as a quick evaluation, the result is limited by the fact that the embedding model used is relatively small. With a larger and more capable model, this approach could become more robust and remove the need for an external API. Using OpenAI’s API for every generation would be costly, so having an in-house evaluation system is preferable.

2. **GPT-4 Scoring**
   As an additional test, GPT-4 was used to score the relevance of generated domain names. A custom prompt asked GPT-4 to assign a score, and this was applied to a random sample of the generated data (around 10% of the dataset) for cost reasons. The resulting scores are shown in the notebook.

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

The confidence score is based on cosine similarity for cost efficiency. GPT-4 scoring was tested and is sometimes more accurate, but embeddings are preferred for budget and control reasons.

To run the API:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
