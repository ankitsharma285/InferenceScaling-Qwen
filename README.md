# InferenceScale-Qwen
**Scaling LLM Reasoning via Test-Time Compute**

---

### 🚀 Overview

**InferenceScale-Qwen** is a high-performance, from-scratch inference engine designed to explore **Inference-Time Scaling Laws**. By implementing compute-optimal strategies on a compact 0.6B parameter model, this project demonstrates that scaling test-time computation—through stochastic path sampling and consensus-based voting—can nearly double the reasoning accuracy of small-scale architectures.

The engine moves beyond standard greedy decoding to implement a robust **Self-Consistency (SC)** pipeline, achieving a significant performance leap on the rigorous **MATH-500** benchmark.

---

### ✨ Key Features

* **Custom Inference Engine:** Built from the ground up based on the Qwen architecture with specialized, decoupled prefill and decoding phases.
* **Performance Optimization:** Features manual **KV-Cache management** to maintain $O(1)$ token generation latency and support high-throughput sampling.
* **Advanced Consensus Logic:** Integrates a consensus engine that leverages stochastic **Top-K/Top-P sampling** to generate multiple reasoning paths and identify the most frequent solution.
* **Symbolic Mathematical Verification:** Utilizes **SymPy** for rigorous grading; the evaluator recognizes mathematical equivalence (e.g., $x+1 \equiv 1+x$), providing a more accurate assessment than standard string matching.
* **Production-Grade Robustness:** Implements "crash-proof" parsing with robust exception handling for unbalanced LaTeX or malformed mathematical expressions typical of high-temperature sampling.

---

### 📊 Performance Benchmarks (MATH-500)

By scaling the number of reasoning paths and applying consensus logic, the model achieved an **~89.7% relative improvement** in accuracy over the baseline.

| Mode | Decoding Strategy | Top-K | Top-P | **Accuracy** | **Total Time** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Greedy Decoding | N/A | N/A | **19.40%** | 177.53 mins |
| **Scaled** | **Self-Consistency** | 50 | 0.95 | **36.90%** | 292.34 mins |

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
python3 main.py --temp 0.7 --top_k 50 --top_p 0.95 --samples 500
```

📚 Resources & Credits

    Reference Implementation: Inspired by Sebastian Raschka's "Reasoning from Scratch".

    Architecture: Based on the Qwen Series.

    Tools: Built using PyTorch and SymPy.
