# **Summary of Experiments**
## Table of Contents
- [Presentation](#presentation)
- [Data Generation](#data-generation)
  - [Motivation](#motivation)
  - [Data organization](#data-organization)
  - [Data generation process](#data-generation-process)
- [LoRA](#lora)
- [Experiments](#experiments)
  - [Experiment 1 — Pipeline validation](#experiment-1--pipeline-validation)
  - [Experiments 2–4 — Manual tokenization with Instruct model](#experiments-24--manual-tokenization-with-instruct-model)
  - [Experiments 5–10 — Switching to the base model](#experiments-510--switching-to-the-base-model)
  - [Conclusions from manual experiments](#conclusions-from-manual-experiments)
- [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Method](#method)
  - [Best Trial](#best-trial)
  - [Training with Optimized Parameters](#training-with-optimized-parameters)
  - [Additional Adjustments after Optimization](#additional-adjustments-after-optimization)
- [Selected Model](#selected-model)
- [Conclusion](#conclusion)

## **Presentation**

The goal of this project is to **fine-tune a Large Language Model (LLM)** to **generate domain names from company descriptions**.
The base model used is [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), and the fine-tuning method is **LoRA (Low-Rank Adaptation)**, a parameter-efficient approach that works well with limited resources.

## **Data Generation**

The dataset was **fully generated** with the **GROQ API** using the **LLaMA 3-70B** model.
Each sample contains:

1. A **fictional business description** (1–5 sentences),
2. A list of **5 suggested domain names** related to that description.

### **Motivation**

* Public datasets did not fit the project’s needs (no direct link from description → domain).
* Web scraping was too slow and unreliable.
* Generating data with LLaMA 3 provided a **clean, consistent, and task-specific dataset**.

### **Data organization**

All data is stored in the `data/` folder, structured in **attempt folders**:

```
data/
├── attempt_0/
│   ├── domain_dataset_v0.json
│   ├── domain_dataset_v1.json
│   └── metadata.json
├── attempt_1/
│   └── ...
```

* Each `attempt` = one generation session.
* Each session can have multiple **versions** (`v0`, `v1`, `v2`…), when new samples are added.
* A `metadata.json` file keeps track of: attempt ID, latest file, number of entries, creation date, etc.

This setup makes it possible to:

* Start a **new dataset** from scratch,
* Or **extend** an existing dataset step by step.

It also matches the **API limits** of GROQ, where requests must be spread over time. Data was generated in **small batches** instead of all at once.

### **Data generation process**

The dataset generation code is in `datasets/domain_dataset.py`.
Example usage in a notebook:

```python
from datasets.domain_dataset import DomainDataset

# Start a new dataset
generator = DomainDataset(from_scratch=True)

# Or continue an existing dataset
# generator = DomainDataset(from_scratch=False)

# Generate 500 new samples
generator.generate(n=500)
```

This script will:

* Create a new folder and version if needed,
* Call the API and parse results,
* Save the data in JSON format,
* Update `metadata.json` with version info and timestamps.

## **LoRA**

I chose the **LoRA** method for this project because I did not have the resources for a full fine-tuning.
LoRA fine-tuning was applied to all attention and projection layers (**q-proj**, **k-proj**, **v-proj**, **o-proj**).

LoRA is the *parameter-efficient fine-tuning* technique I know best, and it is simple to set up.
The main parameters to optimize are **matrix rank**, **alpha**, **dropout**, **learning rate**, and sometimes the choice of target modules.

## **Experiments**

In total, **10 manual experiments** were conducted to test different fine-tuning setups.
The variations focused on:

* data preparation and processing,
* tokenization method,
* training parameters (*learning rate*, LoRA `r` and `alpha`, *dropout*),
* input/output format of the model.

The goal at this stage was mainly to observe how the model behaves under different conditions and to identify the main challenges before moving to systematic optimization.

### **Experiment 1 — Pipeline validation**

The first tests used [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
The objective was not to reach good performance, but simply to validate the full pipeline from training to evaluation.

* **Setup**: each example contained a business description and its domain name. A simple prompt (“Create a domain name from the following description”) was given to the `SFTTrainer` without manual tokenization.
* **Results**: the loss curve was unstable and did not converge. The generations had two main issues:

  1. The instruct model sometimes answered the prompt instead of giving a domain name.
  2. Cosine similarity looked high but was misleading, since the evaluation model (`infloat-e5-small-v2`) focused on lexical overlap rather than semantic relevance.

| Step | Train Loss | Eval Loss |
| ---- | ---------- | --------- |
| 50   | 1.6285     | 1.6417    |
| 100  | 2.1170     | 2.4051    |
| 150  | 5.8945     | 4.6011    |
| 200  | 4.1857     | 3.4284    |
| 250  | 4.0227     | 3.8150    |
| 300  | 3.7581     | 5.0000+   |

### **Experiments 2–4 — Manual tokenization with Instruct model**

For the next series, the Instruct model was still used, but data preparation was changed:

* **Input** = instruction + business description
* **Labels** = expected domain only

This made the target prediction clearer. The data was pre-tokenized (`input_ids` and `labels`) before training.

* **Results**: training was stable, with low loss values and reasonable cosine similarity (**0.7897**).
  However, the model still failed to perform the task correctly: instead of producing domain names, it kept following the instruction. This showed that the **instruction-following bias** of the Instruct model dominated even after fine-tuning.

| Step | Train Loss | Eval Loss |
| ---- | ---------- | --------- |
| 100  | 1.2704     | 0.8283    |
| 200  | 0.7447     | 0.7053    |


### **Experiments 5–10 — Switching to the base model**

To remove this bias, the later experiments used the **base Mistral-7B** model, not fine-tuned for instruction.
The idea was to teach the model directly:

* **Input** = business description only
* **Labels** = domain name only

* **Results**:

  * The model learned the task properly and generated domain names directly.
  * Dataset cleaning improved generalization.
  * Hyperparameters such as `learning_rate`, `lora_r`, `lora_alpha`, and `dropout` were varied, but results reached a **performance ceiling** (~3.28 loss) before overfitting.
  * Cosine similarity (~0.81) gave partial insight but remained limited.
  * GPT-4 evaluation (performed on 10% of the validation set for cost reasons, average 0.536) provided a more reliable metric, but required using an external and paid service.

### **Conclusions from manual experiments**

* **Key parameters in theory**: in LoRA fine-tuning, the most important parameters are usually the **learning rate**, the **rank (`r`)**, and the **alpha** (sometimes also the dropout). These are the first ones to check when tuning.

* **What I observed here**: after switching to the base model and using only the business description as input, the model learned the task correctly but quickly reached a performance plateau, with validation loss around **3.20–3.28**.
  Changing the scale of LoRA parameters (e.g., testing `r=8` vs `r=64` or increasing alpha) did not show any clear effect — the results stayed roughly the same.

* **Evaluation**: cosine similarity(`infloat-e5-small-v2`) was too permissive and did not reflect real relevance, while GPT-4 scoring gave a more reliable measure of quality.

* **Model choice**: base models (non-instruct) worked better for direct supervised training, while Instruct models kept conversational habits that interfered with the task.

## **Hyperparameter Optimization**

After the manual experiments, I moved to a more systematic approach using **Optuna** to explore hyperparameters.
The setup remained the same: the **base Mistral-7B model**, with manual tokenization (input = business description, output = domain name).

### **Method**

The search space covered both LoRA-specific parameters and general optimization settings:

* **LoRA rank (`lora_r`)**: {8, 16, 32, 64}
* **LoRA alpha multiplier (`lora_alpha_mult`)**: {2, 4, 8} (applied as factor × rank)
* **LoRA dropout (`lora_dropout`)**: [0.0, 0.1]
* **Bias training**: {none, lora_only, all}
* **Learning rate**: [1e-5, 3e-4]
* **Warmup ratio**: [0.02, 0.10]
* **Weight decay**: [0.0, 0.05]
* **Max gradient norm**: [0.5, 3.0]
* **Optimizer**: {AdamW, Paged AdamW 32-bit, Paged AdamW 8-bit}
* **Scheduler**: {linear, cosine, cosine_with_restarts, polynomial}

**Fixed values**:

* Targeted modules = all projection layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
* 4-bit quantization
* Batch size = 16
* Gradient accumulation = 16
* Evaluation and checkpoint saving every 14 steps
* Training = 1 epoch per trial

A total of **200 trials** were run.

### **Best Trial**

The best configuration appeared at **trial 176**:

* `lora_r` = 16
* `lora_alpha_mult` = 2
* `lora_dropout` = 0.0164
* `bias` = all
* `learning_rate` = 2.29e-4
* `warmup_ratio` = 0.0468
* `weight_decay` = 0.0382
* `max_grad_norm` = 0.55
* `optimizer` = paged_adamw_8bit
* `lr_scheduler_type` = cosine

**Best validation objective**: **3.18** (proxy validation loss).

Notably, small LoRA ranks (r=16) with a low alpha factor (2×r instead of 8×r) gave the most stable training. The cosine scheduler consistently outperformed the others.

### **Training with Optimized Parameters**

A full run with this setup was done for **5 epochs**, this time with **evaluation and checkpoint saving every 25 steps**.

* **Convergence**: training loss decreased rapidly but validation loss started rising after ~25 steps, showing early overfitting.
* **Cosine Similarity**: computed with the **infloat-multilingual-e5-large** model, giving an average score of **0.798**. The metric was acceptable but not always aligned with human judgment.
* **GPT-4 Scoring**: a sample of generated domains rated by GPT-4 achieved an average of **0.59**, higher than in the manual experiments.

Parfait, je vois bien ce que tu veux. Tu veux ajouter une **sous-section après l’optimisation**, qui montre que tu as tenté un ajustement manuel des hyperparamètres (dropout, LoRA rank, eval steps…), mais sans tout réécrire en détail, juste expliquer **pourquoi** tu l’as fait et **ce que ça a donné**.

### **Additional Adjustments after Optimization**

To address the early overfitting observed in the optimized run, I made two modifications to the training setup:

1. **LoRA parameters**: increased `r` from 16 to 32 and set **LoRA dropout** to 0.05 (instead of ~0.01). The goal was to add a bit more regularization and see if higher rank could improve performance.
2. **Evaluation frequency**: changed `eval_steps`, `logging_steps`, and `save_steps` to **10** (instead of 25), to monitor training more closely.

The rest of the parameters remained the same as in the best Optuna trial.

**Results**:

* Validation loss still started to rise around step 30 (≈3.19), so no real improvement was observed in terms of convergence.
* However, evaluation metrics showed slightly better results:

  * **Cosine similarity** (using `intfloat-multilingual-e5-large`) improved to **0.81**.
  * **GPT-4 scoring** reached **0.64** (always on 10% of the validation set).

## **Selected Model**

This is the experiment I decided to keep.
The checkpoints were merged with the base model weights to create the final model.
It is available on Hugging Face:

**Model name**: *Mistral-7B-v0.1-generate-domain_v6*
**Hugging Face link**: [https://huggingface.co/hassanij/domain-name-generator](https://huggingface.co/hassanij/domain-name-generator)

## **Conclusion**

This project showed, as a **proof of concept**, that the model was able to learn the task of generating domain names from company descriptions.
The results are encouraging, but there is still room for improvement. With a higher LoRA rank (R) and adjusted dropout, better performance could be reached. However, this would require more powerful infrastructure.

Evaluation also has its limits. Cosine similarity was used as the main metric in the pipeline, but it does not always reflect real relevance. GPT-4 scoring gave more reliable results, and in an industrial context, this would be the preferred evaluation method.

The main limitation remains the dataset size. With less than 15,000 training samples, and some companies repeated, the model could not reach full convergence. A larger and more diverse dataset would clearly improve results.


