from typing import Optional 

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
    
    sample_prompt = "What is the square root of 144?"
    print(format_inference_prompt(sample_prompt))