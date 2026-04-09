import torch
from typing import Optional, Generator

from model_lib.qwen3 import KVCache
    

def apply_top_k_threshold(probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filters the vocabulary to only the top k highest probability tokens.
    """
    if k is None or k <= 0:
        return probs

    # Determine the actual k (cannot exceed vocabulary size)
    k = min(k, probs.size(-1))
    
    # Find the top k values and their indices
    top_k_values, _ = torch.topk(probs, k)
    
    # Identify the threshold (the smallest value in the top k)
    threshold = top_k_values[:, -1:]
    
    # Zero out any probability below the threshold
    # Using a mask ensures we stay on the same device
    mask = probs < threshold
    filtered_probs = probs.masked_fill(mask, 0.0)
    
    # Renormalize so the remaining top k sum to 1
    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def apply_top_p_threshold(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: filters the vocabulary to the top-p cumulative mass.
    """
    if p is None or p >= 1.0:
        return probs

    # Sort descending to calculate cumulative mass
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Identify tokens to mask (cumulative mass exceeds p)
    # We shift the mask to ensure we include the first token that crosses the threshold
    mask = cumulative_probs - sorted_probs > p
    
    # Zero out masked probabilities
    sorted_probs[mask] = 0.0
    
    # Re-map to original vocabulary order and normalize
    filtered_probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def stream_llm_response(
    model: torch.nn.Module,
    tokenizer: any,
    prompt: str,
    device: torch.device,
    max_tokens: int,
    strategy: Optional[callable] = None,
    stream_output: bool = True,
    **sampling_params
) -> str:
    """
    High-level manager to handle prompt encoding and streaming decoding.
    """
    # Fallback to basic cached generation if no strategy is provided
    if strategy is None:
        strategy = generate_with_sampling_v2

    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    decoded_tokens = []
    
    # Execution loop
    for token_tensor in strategy(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        eos_id=tokenizer.eos_token_id,
        **sampling_params,
    ):
        token_id = token_tensor.item()
        decoded_tokens.append(token_id)

        if stream_output:
            print(tokenizer.decode([token_id]), end="", flush=True)

    return tokenizer.decode(decoded_tokens)


@torch.inference_mode()
def generate_with_sampling_v2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,   
    top_p: Optional[float] = None
) -> Generator[torch.Tensor, None, None]:
    """
    An optimized generator using KV-caching with support for 
    Temperature, Top-K, and Top-P sampling.
    """
    model.eval()
    device = input_ids.device
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    
    # Prefill phase
    logits = model(input_ids, cache = cache)[:, -1, :] 

    for _ in range(max_new_tokens):
        # 1. Check if we should use greedy or stochastic decoding
        if temperature <= 0 or (temperature == 1.0 and top_k is None and top_p is None):
            next_token = torch.argmax(logits, dim=-1, keepdim=True) 
        else:
            # 2. Apply Temperature Scaling
            scaled_logits = logits / max(temperature, 1e-6) 
            probs = torch.softmax(scaled_logits, dim=-1) 

            # 3. Apply Top-K (Hard Pruning)
            if top_k is not None:
                probs = apply_top_k_threshold(probs, top_k)

            # 4. Apply Top-P (Dynamic Pruning)
            if top_p is not None and top_p < 1.0:
                probs = apply_top_p_threshold(probs, top_p) 

            # 5. Sample from the filtered distribution
            # Note: keeping this on device for performance
            next_token = torch.multinomial(probs, num_samples=1) 

        if eos_id is not None and (next_token == eos_id).all(): 
            break

        yield next_token

        # Decoding phase: pass only the new token to utilize KV Cache
        logits = model(next_token, cache = cache)[:, -1, :] 