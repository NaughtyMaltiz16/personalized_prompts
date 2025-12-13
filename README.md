# TOM: Teach Only What Matters for Selective Chain-of-Thought Supervision

**CAS2105 Homework 6: Mini AI Pipeline Project** 

**Author:** ChanJoo Jung (2022121057)  

**Institution:** Yonsei University (2025)

---

## Introduction

This is a simple project for CAS2105 @ Yonsei University in 2025.

The goal of this project is to train a small language model to solve grade-school math problems by effectively borrowing reasoning capability from a larger model. Given a math problem from the GSM8K dataset, the model generates a step-by-step reasoning process followed by a final numeric answer. Instead of naively using the teacher model’s reasoning on all examples, I investigated a selective strategy in which teacher-generated chain-of-thought (CoT) supervision is applied only to **difficult** problems.

The key message I want to deliver is that **"Only a small subset of data samples is needed. Using all samples is not necessary."**


---


## Task Definition

* **Task description:** The goal is to train a small language model to solve grade-school math problems by effectively borrowing reasoning capability from a larger model.
  
* **Motivation:** Recent developments in large language models have shown that smaller models can efficiently leverage the capabilities of larger models. This raises a natural question: *can a weaker model also borrow the reasoning ability of a stronger model, and if so, how should this be done?* A simple, direct approach is to train the smaller model to adopt ALL the reasoning traces of the larger model. However, **are all reasoning traces equally useful?** Motivated by this, I explored whether reasoning guidance can be applied selectively.
  
* **Input / Output:**
    * **Input:** A grade-school math problem.
    * **Output:** A step-by-step reasoning process and a final numeric answer.
* **Success criteria:**
    * Despite using less teacher supervision, the selectively trained student model outperforms the self-CoT baseline.
    * AND attains accuracy comparable to the teacher-CoT baseline.


---


## Methods


### Models Used
* **Student model:** `Llama-3.2-3B-Instruct`
* **Teacher model:** `Llama-3.1-8B-Instruct`

### Baseline and Reference Models

1.  **Self-CoT Baseline:**
    * The student model generates its own chain-of-thought and is trained to imitate these self-generated traces.
    * Benchmarks the capacity of a small model to bootstrap reasoning skills independently.

2.  **Full Teacher-CoT Reference:**
    * The student model is trained to mimic the teacher's complete reasoning trace for every example.
    * This acts as an upper bound (standard approach to CoT distillation).

---

### AI Pipeline

#### A. Preprocessing
1.  Load the GSM8K dataset (100 training examples, 10 test examples).
2.  Parse the gold final numeric answer (`gold_final`).
3.  Shuffle training data (seed=42).
4.  Construct CoT-style prompts:

```text
You are an expert math tutor.
Solve the problem step by step, then clearly state the final numeric answer.
At the very end, write the answer in the form: Answer: <number>.

Question: {math problem}
Let's think step by step.
````

#### B. Hardness Estimation

1.  Train an initial answer-only student model using only `gold_final` as supervision.
2.  Compute the log-probability of the correct answer for each training example.
3.  Define hardness as the average negative log-likelihood (NLL).
4.  Split training data into easy and hard subsets (50/50 split).

#### C. Reasoning Signal Generation

1.  Generate teacher CoT reasoning for all examples.
2.  **Selective Distillation strategy:**
      * **Hard examples:** Use teacher-generated reasoning as the training target.
      * **Easy examples:** Use only the final numeric answer (`gold_final`) as the training target.

#### D. Representation and Training

1.  Load the student base model with QLoRA (LoRA training with quantization) to fit on 1 GPU.
2.  Train separate student models for each baseline and the proposed selective method.
3.  The selective model is trained for additional epochs (4 epochs) compared to baselines (2 epochs) since it uses less teacher supervision data.

#### E. Inference and Evaluation

1.  Models generate step-by-step reasoning.
2.  Final numeric answer is extracted and evaluated using exact-match accuracy on the GSM8K test subset.

---

## Experiments

### Datasets

  * **Source:** GSM8K (`main`) subset.
  * **Size:** 110 samples total (100 training, 10 testing).

### Results (Exact Match Accuracy)

| Model | Accuracy |
| :--- | :--- |
| Self-CoT student (QLoRA) | 0.50 |
| Teacher full-CoT student (QLoRA) | 0.60 |
| **Selective teacher-CoT student (QLoRA)** | **0.60** |

### Qualitative Examples

**Example 1 (Self-CoT fails; Teacher/Selective succeed)**

  * **Question:** Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now.
  * **Gold Answer:** 109
  * **Self-CoT:** 99 (Incorrect - fails to add 10 years)
  * **Selective:** 109 (Correct)

**Example 2 (All models fail)**

  * **Question:** Lorraine and Colleen are trading stickers... (Complex multi-step trading).
  * **Gold Answer:** 89
  * **Self-CoT:** 3
  * **Teacher full-CoT:** 7
  * **Selective:** 60

**Example 3 (All models succeed)**

  * **Question:** Indras has 6 letters in her name...
  * **Gold Answer:** 13
  * **All models:** 13 (Correct)

## Conclusion

Our results indicate that selectively applying teacher chain-of-thought supervision to hard examples is sufficient. The selectively trained model outperforms the self-CoT baseline and achieves accuracy comparable to the teacher full-CoT baseline, while using substantially less supervision.


---



## Reflection and Limitations

I believe the use of a very small subset of the GSM8K dataset limits the statistical significance of the findings in this small-scale project. In fact, I am not sure whether this performance trend would persist when scaling to the full dataset. Example hardness may not apply to other models or datasets since it is based on a single answer-only student model. Furthermore, because just 10 test samples are used for the evaluation, even slight variations in predictions might have a significant impact on the stated accuracy. Lastly, while though selective supervision uses less teacher reasoning, it still necessitates having access to a teacher model throughout training, which might be expensive in some situations. Future research should investigate different hardness metrics and confirm these results on a bigger scale.

```
```
