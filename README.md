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

4. **Launch the notebook**

To use the notebook:

```bash
jupyter lab
```

Or open it directly from VS Code if Jupyter support is enabled.

> You can now run the notebook.

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