# InferenceScaling-Qwen
Summary: Exploring Inference-Time Scaling Laws on a 0.6B model. Boosted MATH-500 accuracy from 19.4% (Greedy) to 36.6% using stochastic path sampling (Top-K/Top-P) and Self-Consistency voting. Features a from-scratch KV-cached engine with symbolic post-processing for robust mathematical evaluation.

OverView:

InferenceScale-Qwen is an optimized, from-scratch implementation of a reasoning-focused inference engine. This project explores the Inference-Time Scaling Law, demonstrating that a compact 0.6B parameter model can achieve significantly higher reasoning performance by scaling test-time computation through stochastic path sampling and consensus-based majority voting.
Project Overview

This repository features a custom-built Qwen inference stack designed for high-precision mathematical reasoning. By moving beyond fixed-compute greedy decoding and implementing Compute-Optimal Inference, the engine trades additional test-time computation for increased accuracy.
Key Features

    From-Scratch Inference Engine: Custom implementation of the Qwen architecture [https://github.com/rasbt/reasoning-from-scratch/tree/main] with decoupled prefill and decoding phases. 

    Optimized KV-Caching: Manual Key-Value cache management to ensure O(1) token generation latency.

    Consensus Engine: Implementation of the Self-Consistency (SC) strategy using stochastic path sampling (Top-K/Top-P) and majority voting.

    Symbolic Evaluation: A robust post-processing pipeline using SymPy for symbolic verification, ensuring that mathematically equivalent answers (e.g., x+1 vs 1+x) are graded correctly.

    Crash-Proof Parsing: Robust exception handling for malformed LaTeX or unbalanced mathematical expressions, essential for high-temperature sampling.

Performance Benchmarks (MATH-500)

The following table demonstrates the impact of Inference-Time Scaling on a 0.6B parameter model. By scaling the number of reasoning paths and applying consensus logic, accuracy improved by nearly 90% over the baseline.
Mode	Decoding Strategy	Top-K	Top-P	Accuracy	Total Time
Base Model	Greedy Decoding	N/A	N/A	19.40%	177.53 mins
Base Model	Self-Consistency	50	0.95	36.90%	292.34 mins

Usage
Prerequisites
Bash

pip install torch sympy tokenizers

Running the Evaluation

To reproduce the benchmarks on the full MATH-500 dataset, use the following commands:

1. Greedy Decoding (Baseline)
Bash

python3 main.py --temp 0 --samples 500

2. Self-Consistency (Inference Scaling)
Bash

python3 main.py --temp 0.7 --top_k 50 --top_p 0.95 --samples 500

Resources & Acknowledgments

Sebastian Raschka: https://github.com/rasbt/reasoning-from-scratch/tree/main
