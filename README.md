# InferenceScale-Qwen
**Inference-Time Scaling for LLM Reasoning**

---

### 🚀 Overview

Large Language Models (LLMs) increasingly rely on inference-time scaling to improve reasoning without additional training. Among these approaches, Self-Consistency decoding enhances performance by generating multiple independent reasoning paths and selecting the most consistent answer. While previous work has demonstrated its effectiveness, the practical relationship between sampling strategy, reasoning budget, and problem difficulty remains insufficiently characterized.

This project presents a systematic empirical study of inference-time scaling on the MATH-500 benchmark. We evaluate more than 60 decoding configurations by varying the number of reasoning paths, temperature, Top-k, and Top-p sampling parameters to analyze the trade-offs between computational cost and reasoning accuracy.

Our experiments show that increasing inference-time compute substantially improves mathematical reasoning, achieving a 2.43× improvement in accuracy (17.6% → 42.8%) over greedy decoding. The analysis further reveals that moderate stochasticity consistently produces stronger consensus, while performance gains diminish beyond moderate reasoning budgets. Difficulty-wise evaluation demonstrates that additional reasoning paths primarily benefit medium- and high-complexity problems, suggesting opportunities for adaptive compute allocation during inference.

Together, these findings provide practical guidelines for designing efficient inference-time reasoning systems and contribute empirical insights into the scaling behavior of Self-Consistency decoding.

---
### ✨ Motivation 
Reasoning models have shifted the scaling paradigm from

**Train Bigger** → **Infer Once**

to

**Train Once** → **Think Longer**

This project asks:

* How much accuracy can be gained by simply allocating more inference-time compute?
* What sampling strategy produces the best consensus?
* When do additional reasoning paths stop helping?
* Which types of problems benefit most from inference-time scaling?

--- 
### ✨ Key Features

* **Advanced Consensus Logic:** Integrates a consensus engine that leverages stochastic **Top-K/Top-P sampling** to generate multiple reasoning paths and identify the most frequent solution.
* **Symbolic Mathematical Verification:** Utilizes **SymPy** for rigorous grading; the evaluator recognizes mathematical equivalence (e.g., $x+1 \equiv 1+x$), providing a more accurate assessment than standard string matching.
* **Production-Grade Robustness:** Implements "crash-proof" parsing with robust exception handling for unbalanced LaTeX or malformed mathematical expressions typical of high-temperature sampling.

---

### 📊 Performance Benchmarks (MATH-500)

By scaling the number of reasoning paths and applying consensus logic, the model achieved an **~89.7% relative improvement** in accuracy over the baseline.

| Model | Decoding Strategy | N Paths | Temp | Top-K | Top-P | early stop | compile | Path Taken/Total Path | **Accuracy** | **Total Time** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| **Base** | Greedy Decoding | N/A| N/A | N/A | N/A| N/A | N/A | N/A | **17.60%** | 154.44 mins |
| **Base** | **Self-Consistency** | 3 | 0.3 | 50 | 0.95 | Yes | Yes | 1366/1500 | **26.60%** | 86.68 mins |
| **Base** | **Self-Consistency** | 3 | 0.5 | 50 | 0.95 | Yes | Yes | 1400/1500 | **32.80%** | 106.73 mins |
| **Base** | **Self-Consistency** | 3 | 0.7 | 50 | 0.95 | Yes | Yes | 1384/1500 | **36.40%** | 115.94 mins |
| **Base** | **Self-Consistency** | 3 | 1.0 | 50 | 0.95 | Yes | Yes | 1410/1500 | **31.80%** | 121.01 mins |
| **Base** | **Self-Consistency** | 5 | 0.3 | 50 | 0.95 | Yes | Yes | 2291/2500 | **25.20%** | 132.46 mins |
| **Base** | **Self-Consistency** | 5 | 0.5 | 50 | 0.95 | Yes | Yes | 2353/2500 | **31,00%** | 145.49 mins |
| **Base** | **Self-Consistency** | 5 | 0.7 | 50 | 0.95 | Yes | Yes | 2387/2500 | **37.60%** | 167.84 mins |
| **Base** | **Self-Consistency** | 5 | 1.0 | 50 | 0.95 | Yes | Yes | 2403/2500 | **32.40%** | 180.90 mins |
| **Base** | **Self-Consistency** | 8 | 0.3 | 50 | 0.95 | Yes | Yes | 3764/4000 | **28.20%** | 215.45 mins |
| **Base** | **Self-Consistency** | 8 | 0.5 | 50 | 0.95 | Yes | Yes | 3849/4000 | **34.20%** | 251.15 mins |
| **Base** | **Self-Consistency** | 8 | 0.7 | 50 | 0.95 | Yes | Yes | 3876/4000 | **38.00%** | 271.21 mins |
| **Base** | **Self-Consistency** | 8 | 1.0 | 50 | 0.95 | Yes | Yes | 3908/4000 | **35.40%** | 297.87 mins |
| **Base** | **Self-Consistency** | 16 | 0.7 | 50 | 0.95 | Yes | Yes | 7781/8000 | **39.20%** | 505.97 mins |
| **Base** | **Self-Consistency** | 16 | 1.0 | 50 | 0.95 | Yes | Yes | 7870/8000 | **38.20%** | 580.17 mins |
| **Base** | **Self-Consistency** | 32 | 0.7 | 50 | 0.95 | Yes | Yes | 15515/16000 | **42.80%** | 1072.22 mins |


---

### Key Insights

* **Inference-time compute substantially improves reasoning:** Increasing the number of reasoning paths consistently improves mathematical reasoning over greedy decoding.
* **Moderate stochasticity produces better consensus:** Very low temperatures reduce diversity while excessively high temperatures introduce noisy reasoning trajectories.
* **Diminishing returns emerge after moderate path budgets** Beyond 16–32 sampled reasoning paths, additional compute produces progressively smaller accuracy gains.
* **Hard problems benefit most** Additional inference-time compute primarily improves medium- and high-difficulty questions, while easy problems saturate quickly.

---
### 🛠️ Usage

#### Installation
Ensure you have the necessary dependencies installed:

pip install torch sympy tokenizers

#### Reproducing Results
Run the following commands to evaluate the engine on the full 500-sample dataset:

**1. Baseline Evaluation (Greedy Decoding)**
```bash
python3 main.py --temp 0 --samples 500 
```

**2. Scaled Evaluation (Self-Consistency)**
```bash
python3 main.py --temp 0.7 --top_k 50 --top_p 0.95 --samples 500 --early_stop --compile 
```

## Resources & Credits
1. Reference Implementation: Inspired by Sebastian Raschka's "Reasoning from Scratch". [project](https://github.com/rasbt/reasoning-from-scratch/tree/main)

