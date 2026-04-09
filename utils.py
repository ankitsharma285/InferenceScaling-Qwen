import torch
#import logging

import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from pprint import pprint

def initialize_compute_device(use_tf32: bool = True) -> torch.device:
    """
    Configures and returns the primary compute device.
    Strictly handles CUDA and CPU transitions.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Configure Tensor Cores (TF32) for Ampere+ GPUs
        if use_tf32:
            # Check if the hardware actually supports it (Compute Capability 8.0+)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')
                # Alternatively, use the explicit flags for older PyTorch compatibility:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
        print(f"Compute backend: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Compute backend: CPU")

    return device

def fetch_math_benchmark(
    filename: str = "math500_test.json", 
    cache_dir: str = "./data",
    force_download: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieves the MATH-500 test set. Checks local cache before 
    downloading from the remote repository.
    """
    source_url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    
    # Ensure directory exists
    data_path = Path(cache_dir) / filename
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Logic for loading from disk or downloading
    if data_path.exists() and not force_download:
        print(f"Loading benchmark from cache: {data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        print(f"Downloading benchmark from source...")
        try:
            response = requests.get(source_url, timeout=20)
            response.raise_for_status()
            dataset = response.json()
            
            # Atomic-style write
            with data_path.open("w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4)
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to retrieve MATH-500 dataset: {e}")

    return dataset

def format_inference_prompt(
    question: str, 
    system_instruction: Optional[str] = None
) -> str:
    """
    Constructs a structured prompt for LLM math reasoning.
    Uses LaTeX-style boxed formatting for final answers.
    """
    if system_instruction is None:
        system_instruction = "You are a helpful math assistant."
        #system_instruction = "You are a Rigorous Mathematician."

    # Using an f-string with clear block separation for readability
    prompt_template = (
        f"{system_instruction}\n\n"
        "Instructions:\n"
        "1. Solve the problem step-by-step.\n"
        "2. Provide the final result on a separate line using: \\boxed{{ANSWER}}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    
    return prompt_template

if __name__ == "__main__":
    device = initialize_compute_device()
    math_test_set = fetch_math_benchmark()
    print(f"Successfully loaded {len(math_test_set)} samples.")
    pprint(math_test_set[0])
    sample_prompt = "What is the square root of 144?"
    print(format_inference_prompt(sample_prompt))

