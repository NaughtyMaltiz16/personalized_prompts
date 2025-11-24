# Generating Personalized Prompts

This is a simple project for CAS2105 @ Yonsei University in 2025

---

## **1. Task Description and Motivation**

### **Task Description**

The goal of this project is to train a model that generates personalized prompts based on my preferred writing style.
When a topic is provided, the model outputs a prompt that reflects my specific tone and style.

### **Motivation**

I realized that outputs change whenever the prompt changes. With this in mind, I adopted a specific prompting style to ensure the model generates the desired output. Through extensive experience, I developed specialized prompts that yield the results I want. However, manually rewriting prompts every time to match my style is time-consuming and can be inconsistent. In fact, if I don't use AI for an extended period, I may forget my preferred style. Therefore, I have always considered building a model that automatically generates prompts following my specialized format. Undoubtedly, a lightweight, personalized prompt generator would help me work faster and maintain consistency.

### **Input / Output**

* **Input:** A topic or instruction (ex: "Make a prompt that converts raw text into bullet points.").
* **Output:** A personalized prompt that resembles my “liked” examples. (ex: "Convert the text into 4 bullet points. Make sure each point is under 15 words.")

### **Success Criteria**

The system is considered successful if:

* Generated prompts are stylistically closer to my preferred “liked” prompts than the vanilla model baseline.
* ROUGE-1 and ROUGE-L scores are higher than the untrained, vanilla model baseline.
* This is because when it comes to style, having overlapped words, phrases, and sentence structure reflect that the style format is matched.
* In fact, if I typically have a favorite word that I use repetitively, this should be reflected via ROUGE-1 and ROUGE-L scores.

---

## **2. Dataset**

I created manually my own dataset consisting of prompts labeled by preference.
Because I had to make my own dataset from scratch, I made only 61 data points.
Anyway, the project specified to only use about 50~300 labeled examples.
I was also curious whether a small dataset could make a difference.

* **Total examples:** 61
* **Split:**

  * 51 train samples
  * 10 test samples

---

## **3. Methods**

### **3.1 Naïve Baseline**

The naïve baseline is the vanilla model and the retrieval method using Jaccard Similarity.

### **1. Vanilla Model Baseline**

* Uses the **untrained, initialized model** without any training.
* This baseline is used because my main method includes a training step, so the fairest comparison is a model with **no learning at all**.
* It represents the behavior of a model that has **not learned my style** in any way.

### **2. Retrieval Baseline (Jaccard Similarity)**

* For each new request, I select the training example with the **highest Jaccard similarity** to the input.
* This involves **no model**, **no training**, and only simple text overlap.
* It satisfies the homework requirement of a non-sophisticated baseline while still producing a more relevant result than random selection.

  
#### **Why both baseline is naïve**

* Does not learn personal style (Does not know anything about my style)
* Same structure every time
* Ignores tone, structure, and domain
* Fails on longer or creative tasks

#### **Common Failure Cases**

* Generating prompts needing specific tone or phrasing
* Generating prompts that usually have specific favorite words or phrases
* Domain-specific topics (ML, coding)

---

### **3.2 AI Pipeline**

---

#### **A. Preprocessing**

1. Load data (each JSONL line has input and output)
2. Shuffle and split the data
3. Load tokenizer
4. Create prompt:

```text
### Instruction:
{input}

### Response:
```

5. Labels are masked so loss only on response tokens. (set labels of instruction tokens to -100)
6. Pads dynamically to MAX_LEN (If pad_token is missing, we set it to eos_token to enable padding.) OR Truncate to MAX_LEN (Inputs and prompt+output are truncated to MAX_LEN=256 tokens.)

---

#### **B. Representation or embedding**

7. Load a base model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
8. Load a LoRA on top of that base model for lighter training and implementation. (attention (q,k,v,o), and MLP projections (gate, up, down) are used, these are commonly used hyperparameters when setting a LoRA.)

---

#### **C. Decision or ranking**

9. Train the LoRA that is attached to the base model. (Only LoRA parameters are updated. Base model is frozen / not trained.)
10. It trains using standard causal LM loss over response tokens. (hyperparameters: 3 epochs, batch_size=2, grad_accum=4)

---

#### **D. Optional post-processing**

11. Generate outputs (personalized prompts) on the given test-set inputs deterministically. (greedy decoding: it will only generate an output with the best token selections.)
12. After decoding, we remove the prompt prefix and keep only the response.

---

## **4. Evaluation and Results**

### **Metrics**

* ROUGE-1 and ROUGE-L scores for prompts generated by the trained model, vanilla models, and retireval method.

### **Results**

| Model                   | ROUGE-1 (F1) | ROUGE-L (F1) |
| ----------------------- | ------------ | ------------ |
| **Vanilla (Zero-shot)** | 0.1562       | 0.1347       |
| **LoRA-Trained Model**  | **0.1990**   | **0.1872**   |

---

### **Qualitative Examples**

---

**Example 1**

* "input": "Make a prompt that rewrites Korean text in a more concise tone."
* "gold": Rewrite the Korean text in a concise tone. Make sure the meaning stays the same. Show only the revised text.
* "vanilla_pred": 잘 못해요. 저는 잘 못해요. 저는 잘 못해요. 저는 잘 못해요. 저는 잘 못해요. 저는
* "trained_pred": "Rewrite the text in a more concise tone. Make sure it still makes sense. Make sure to use fewer words. Make sure it's clear. Make sure it's short. Make sure it's short. Make sure it's short. Make sure it's short. Make sure it's short. Make sure it's short. Make sure it's short",

---

**Example 2**

* "input": "Make a prompt that summarizes an experimental table with mean ± std."
* "gold": "Summarize the table. Show mean and std only. Make sure to bold the best scores and explain the trend in 3 sentences."
* "vanilla_pred": "Please enter the name of the table you want to generate:\n\n### Instruction:\nGenerate a table with the mean and standard deviation of the data.\n\n###"
* "trained_pred": "Summarize the table. Make sure to include the mean and standard deviation. Make sure to use bullet points. Make sure to use short sentences. Make sure to use bullet points. Make sure to use short sentences. Make sure to use bullet points. Make sure to use short sentences. Make sure to use bullet points. Make sure to use short sentences. Make sure to use bullet points. Make"

---

**Example 3**

* "input": "Make a prompt that checks if a sentence is too long."
* "gold": "Check the sentence length. Explain if it is too long. Make sure to rewrite it shorter."
* "vanilla_pred": "Please enter a sentence:\nI love you more than anything in this world.\n`\n\n### Output:\n`\nThe sentence you entered is too long. Please enter a sentence that is less than 100 characters long. ```\n\n### Explanation:\nThe prompt checks if the entered sentence is longer than 100 characters. If it"
* "trained_pred": "Check if the sentence is too long. Explain why. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet points. Make sure to use bullet"

---

## **5. Reflection and Limitations**

Actually, the trained model did perform better than the vanilla model.
However, I am indeed disappointed with the outputs of the trained model because it seems to not be concise and perfect.
Yes, I can see some important keywords (i.e. favorite words and phrases) that appear in the generated outputs.
But, I would honestly say that I won't use those prompts in the real world.
I think this is because (limitations) 1) the model is small, 2) data is too small, 3) or hyperparameter search was insufficient. (This means that the hyperparameters used may be suboptimal.)
Next time, I may think of ways to use a larger model, a larger dataset, or better hyperparameters when training.

Overall, this project helped me learn a lesson that modeling is not a piece of cake.

---
