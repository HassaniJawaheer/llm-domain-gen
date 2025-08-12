# **Summary of Experiments**

> **Chosen model**: [https://huggingface.co/mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
> **PEFT (Parameter-Efficient Fine-Tuning) method** used: **LoRA (Low-Rank Adaptation)**

## **Data**

The data was produced using the **GROQ API**, specifically the **LLaMA 3 70B** model.
I asked the model to generate a business description for a fictional company. For each description, I then asked it to create 5 domain names.
The data was created gradually because the API is free but has limits: you cannot send many requests at the same time, nor in a short time span.
This could cause repeated companies or descriptions, but this was handled properly later to avoid biasing the model.

## **LoRa**

I chose the **LoRA** method to fine-tune the model, mainly because I do not have the resources for a full fine-tuning. Fine-tuning was done on all attention layers and projection layers (**o-proj**, **k-proj**, **v-proj** and **o-proj**). I use LoRA because it is the *parameter-efficient fine-tuning* technique I know best and it is very simple to set up.
Also, with LoRA, parameter optimization focuses mainly on **matrix rank**, **alpha**, **dropout**, **learning rate**, and sometimes the selection of modules to fine-tune.

In the next paragraph, I will detail the experiments.

## **Experiments**

In total, **10 experiments** were carried out to test different fine-tuning configurations, with variations on:

* data preparation and processing,
* tokenization method,
* training parameters (*learning rate*, LoRA `r` and `alpha`, *dropout*),
* and the input/output format of the model.

Below are the most important steps.

### **Initial experiment**


Initially, I started with the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model before switching to the base Mistral model.
For the first experiments, I used this model, but before going further, you will see that I moved to another model for technical and efficiency reasons. 

The goal of this first experiment was mainly to validate the proper functioning of the full pipeline, from training to evaluation, without aiming for optimal performance. Training parameters were chosen arbitrarily, mainly to test the technical chain. Data was prepared simply: each example consisted of a company description (business description) and the associated domain name. A basic prompt was generated from this data: *"Create a domain name from the following description"*, given as-is to the `SFTTrainer` without manual pre-tokenization.
In terms of behavior, the loss curve showed strong instability: after a good initial value, the loss quickly increased and fluctuated, indicating no convergence. The generations had two issues:

1. The *instruct* model sometimes followed the instruction instead of directly giving a domain name, which was not the expected behavior.
2. The cosine similarity score was high but unreliable: the evaluation model `infloat-e5-small-v2` favored lexical similarity (common words) over fine semantic closeness.

#### **Configuration used**

```python
{
    "num_train_epochs": 5,
    "per_device_train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-3,
    "weight_decay": 0.001,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.3,
    "max_grad_norm": 0.3,
    "optimizer": "paged_adamw_32bit",
    "max_seq_length": 128,
    "fp16": False,
    "bf16": True,
    "group_by_length": True,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "logging_steps": 50,
    "eval_accumulation_steps": 16,
    "per_device_eval_batch_size": 64,
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },
    "lora": {
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}
```

#### **Results**

| Step | Loss (train) | Loss (eval) |
| ---- | ------------ | ----------- |
| 50   | 1.6285       | 1.6417      |
| 100  | 2.1170       | 2.4051      |
| 150  | 5.8945       | 4.6011      |
| 200  | 4.1857       | 3.4284      |
| 250  | 4.0227       | 3.8150      |
| 300  | 3.7581       | 5.0000+     |


### **Experiment numbers 2, 3, 4**

Here, I kept the **Mistral 7b Instruct** model, but completely changed how data was prepared. This time, I did the tokenization myself:

* **input** = instruction + company description (*business\_description*)
* **labels** = only the expected domain

This makes it clear what the model should predict. The data is already tokenized (`input_ids` and `labels`) before training.

#### **Parameters**

```python
num_train_epochs = 5
per_device_train_batch_size = 64
gradient_accumulation_steps = 4
learning_rate = 2e-5
weight_decay = 0.0
warmup_ratio = 0.2
lr_scheduler_type = "cosine"
max_seq_length = 110
optimizer = "paged_adamw_32bit"
fp16 = False
bf16 = True
group_by_length = True

# Quantization
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_use_double_quant = False

# LoRA
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

#### **Results**

| Step | Train Loss | Eval Loss |
| ---- | ---------- | --------- |
| 100  | 1.2704     | 0.8283    |
| 200  | 0.7447     | 0.7053    |

Cosine Similarity Mean: **0.7897** (validation set)

Loss is low and stable, without major instability.
However, **the model does not learn the task**: instead of directly giving the expected domain, it answers the question from the instruction.
This is due to the **instruction-following** bias of the Mistral Instruct model, which remains dominant even after fine-tuning.

## **Experiments 5 to 10**

The goal of this series was to start from a Mistral 7B model **not fine-tuned for instruction** to reduce the behavioral bias from instruction-following. The idea was to train the model directly on the task of generating a domain name from a company description, without including any explicit instruction in the training data.

### **Data and preprocessing**

* **Manual tokenization**:

  * **Input**: only the company description.
  * **Output (labels)**: only the expected domain name.
  * No instruction phrase included, to avoid irrelevant answers.

* **Extra data cleaning**:

  * Finding: in earlier experiments, some companies were present in both training and validation, with similar descriptions and domain names.
  * Effect: "by heart" memorization and inability to generalize to new companies.
  * Fix: strict separation of datasets so no company in training appears in validation, even with a modified description.

### **Parameters**

```python
Cosine Similarity Mean = 0.8084
GPT-4 confidence Mean  = 0.5365

train_config = {
    "num_train_epochs": 20,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "eval_accumulation_steps": 16,
    "eval_strategy": "steps",  
    "eval_steps": 25,
    "logging_steps": 25,
    "save_steps": 25,
    "learning_rate": 5e-5,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 100,
    "optimizer": "paged_adamw_32bit",
    "metric_for_best_model": "eval_loss",
    "load_best_model_at_end": True,
    "greater_is_better": False,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False
    },
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj","k_proj","v_proj","o_proj"],
        "lora_dropout": 0.05
    }
}
```

### **Observations**

* The model **really learns the task**: when given a description, it directly outputs a domain name, without answering a question or generating irrelevant text.
* Removing the bias in dataset splitting clearly improved generalization.
* Changing `learning_rate`, `lora_r`, `lora_alpha`, and `dropout` between 0.05 and 0.1 did not go beyond a **performance ceiling** (\~3.28 loss) before overfitting.
* **Cosine similarity** (\~0.81) gives an indication but is limited by the embedding model used, which favors lexical similarity over semantic meaning.
* Evaluation with **GPT-4** and an analysis prompt gives a **more relevant measure** (average 0.536) but depends on an external API, and it is paid.

## **Conclusions**

* **Important parameters**:
  *Learning rate* had the most impact on convergence.
  LoRA (r, alpha, dropout) had a much smaller effect on final scores, even when changing from r=8 to r=128.

* **Cosine similarity vs. qualitative evaluation**:
  Cosine similarity is too permissive here: it only measures lexical similarity, not the real relevance of the domain.
  GPT-4 Confidence gives a better estimate, but depends on the evaluator model.

* **Model choice**:
  Base models (non-instruct) work better for this direct supervised format but converge more slowly.
  Instruct models tend to keep “conversational” response habits.

* **Improvement areas**:

  * Dataset limited to **15,000–16,000 samples**, which is small for a specialized generation task. Enough to guide general model behavior, but **not enough for full learning and convergence**. The **observed performance ceiling** (loss not decreasing) is probably due to this data shortage: the model started learning but lacked enough examples to improve further.
  * Fine hyperparameter optimization was not done, which also limited improvement potential.

