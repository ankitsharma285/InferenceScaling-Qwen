import time
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any

import engine 
import utils 
import math_postprocessing as mp 

def run_math500_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    device: torch.device,
    output_path: Path,
    num_paths: int = 3,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    prompt_suffix: str = "",
    enable_early_exit: bool = False,
    seed: int = 42,
    verbose: bool = False
):
    """
    Evaluates the model on the MATH-500 benchmark using self-consistency scaling.
    Writes results to a JSONL file in real-time.
    """
    print(f"Starting evaluation on {len(dataset)} samples...")
    print(f"Configuration: Paths={num_paths}, Temp={temperature}, EarlyExit={enable_early_exit}")
    
    correct_count = 0
    start_time = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            # 1. Prepare Prompt
            # 'format_inference_prompt' was our first rewrite
            base_prompt = utils.format_inference_prompt(row["problem"])
            full_prompt = base_prompt + prompt_suffix

            # 2. Generate Reasoning Paths & Consensus
            consensus_results = engine.compute_consensus_reasoning(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                device=device,
                num_paths=num_paths,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_tokens,
                enable_early_exit=enable_early_exit,
                show_logs=False,
                base_seed=seed
            )

            # 3. Resolve Answer & Grade
            # If no clear majority, we pick the first winner from the tie-list
            prediction = consensus_results["final_answer"]
            if prediction is None:
                prediction = consensus_results["parsed_candidates"][0]

            is_correct = mp.MathEvaluator.is_equivalent(row["answer"], prediction)
            if is_correct:
                correct_count += 1

            # 4. Find the full text for the chosen answer for logging
            # (Matches the logic of your original script)
            full_text = ""
            for idx, p_ans in enumerate(consensus_results["parsed_candidates"]):
                if p_ans == prediction:
                    full_text = consensus_results["all_paths"][idx]
                    break

            # 5. Save Record
            record = {
                "id": i,
                "problem": row["problem"],
                "gold_answer": row["answer"],
                "prediction": prediction,
                "is_correct": is_correct,
                "full_reasoning": full_text,
                "samples_used": consensus_results["samples_taken"],
                "distribution": consensus_results["distribution"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 6. Logging
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(dataset) - (i + 1))
            
            progress_str = f"[{i+1}/{len(dataset)}] Acc: {correct_count/(i+1):.1%}"
            if verbose:
                print(f"{progress_str} | Latest: {prediction} | Target: {row['answer']}")
            else:
                print(f"{progress_str} | ETA: {eta/60:.1f}m", end="\r")

    total_time = time.perf_counter() - start_time
    final_acc = (correct_count / len(dataset)) * 100
    print(f"\n{'='*30}\nFinal Accuracy: {final_acc:.2f}%")
    print(f"Total Time: {total_time/60:.2f} mins")
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="LLM Inference-Time Scaling Evaluator")
    parser.add_argument("--variant", type=str, default="base", choices=["base", "reasoning"])
    parser.add_argument("--samples", type=int, default=10, help="Total examples to test")
    parser.add_argument("--paths", type=int, default=3, help="Paths for Self-Consistency")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # Initialize Hardware and Model
    device = utils.initialize_compute_device()
    model, tokenizer = engine.initialize_model_pipeline(
        model_variant=args.variant,
        device=device,
        enable_compilation=args.compile
    )

    # Load Data
    math_data = utils.fetch_math_benchmark()
    test_subset = math_data[:args.samples]

    # Setup Output Path
    out_file = Path(f"eval_{args.variant}_{device.type}_p{args.paths}.jsonl")

    run_math500_evaluation(
        model=model,
        tokenizer=tokenizer,
        dataset=test_subset,
        device=device,
        output_path=out_file,
        num_paths=args.paths,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        enable_early_exit=args.early_stop,
        verbose=True
    )

if __name__ == "__main__":
    main()