from pathlib import Path
from typing import Tuple, Literal

import torch
import torch.nn.functional as F
from typing import Optional, Generator, Dict, Any, List

from collections import Counter


from model_lib.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    KVCache,
    QWEN_CONFIG_06_B
)

import generation_factory as gf 
import math_postprocessing as mp

def initialize_model_pipeline(
    model_variant: Literal["base", "reasoning"],
    device: torch.device,
    enable_compilation: bool = False,
    storage_path: str = "qwen3"
) -> Tuple[torch.nn.Module, Qwen3Tokenizer]:
    """
    Initializes the model and tokenizer for inference.
    Handles asset downloading, weight loading, and TorchDynamo compilation.
    """
    base_dir = Path(storage_path)
    
    # Define variant-specific configurations
    is_reasoning = (model_variant == "reasoning")
    suffix = "reasoning" if is_reasoning else "base"
    
    # Ensure assets are present
    download_qwen3_small(
        kind=model_variant, 
        tokenizer_only=False, 
        out_dir=str(base_dir)
    )

    # Initialize Tokenizer with specific reasoning flags
    tokenizer_file = base_dir / f"tokenizer-{suffix}.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file,
        apply_chat_template=is_reasoning,
        add_generation_prompt=is_reasoning,
        add_thinking=is_reasoning
    )

    # Build and Load Model
    model = Qwen3Model(QWEN_CONFIG_06_B)
    weights_path = base_dir / f"qwen3-0.6B-{suffix}.pth"
    
    # Map to device during load to prevent OOM on CPU first
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval() # Always set to eval for inference

    # Performance Optimization: Torch Compile
    if enable_compilation:
        print("Optimizing model with torch.compile...")
        # These flags are helpful for inference-heavy scaling pipelines
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer

# Usage example:
# model, tokenizer = initialize_model_pipeline("reasoning", device, enable_compilation=True)


# Assuming these are your helper functions for parsing
# from scaling_engine.utils import extract_answer_from_text

def compute_consensus_reasoning(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    num_paths: int = 10,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 2048,
    enable_early_exit: bool = True,
    show_logs: bool = True,
    base_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Implements Self-Consistency (majority voting) over multiple reasoning paths.
    Includes an early-exit optimization to save compute once a majority is reached.
    """
    
    path_logs = []
    extracted_candidates = []
    frequency_map = Counter()
    path_groups = {}
    
    final_decision = None
    consensus_met = False

    for i in range(num_paths):
        # Ensure stochastic diversity across paths
        if base_seed is not None:
            torch.manual_seed(base_seed + i)

        # Generate a reasoning path
        # Using the unified streaming function we built previously
        raw_response = gf.stream_llm_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_tokens=max_new_tokens,
            strategy=gf.generate_with_sampling_v2, 
            stream_output=False, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Extract the final answer (e.g., the content inside \boxed{})
        # 'extract_answer_from_text' would be your custom parsing utility
        parsed_answer = mp.MathEvaluator.get_final_candidate(raw_response)
        
        # Track results
        path_logs.append(raw_response)
        extracted_candidates.append(parsed_answer)
        frequency_map[parsed_answer] += 1
        path_groups.setdefault(parsed_answer, []).append(i)

        if show_logs:
            print(f"Path {i+1}/{num_paths} | Candidate: {parsed_answer}")

        # --- Early Exit Logic ---
        # If one answer already has more than half the total possible votes, 
        # further sampling won't change the majority winner.
        if enable_early_exit and frequency_map[parsed_answer] > (num_paths / 2):
            final_decision = parsed_answer
            consensus_met = True
            if show_logs:
                print(f"Early exit triggered: Majority consensus reached.")
            break

    # Resolve final answer if early exit wasn't triggered
    if not consensus_met:
        most_common = frequency_map.most_common()
        if most_common:
            top_candidate, top_count = most_common[0]
            
            # Check for ties
            winners = [ans for ans, count in most_common if count == top_count]
            
            # If there's a clear winner, assign it. 
            # If there's a tie, your project could return None or the first one.
            final_decision = top_candidate if len(winners) == 1 else None

    return {
        "final_answer": final_decision,
        "all_paths": path_logs,
        "parsed_candidates": extracted_candidates,
        "distribution": dict(frequency_map),
        "path_indices": path_groups,
        "samples_taken": len(path_logs)
    }