import re
import torch
from pathlib import Path
from typing import Optional, List, Union
from collections import Counter

from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from tokenize import TokenError

# --- Constants & Patterns ---
# Optimized Regex to catch decimals, fractions, and scientific notation
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RE_SPECIAL_TOKENS = re.compile(r"<\|[^>]+?\|>")

LATEX_REPLACEMENTS = [
    (r"\\left\s*|\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot|\u00B7|\u00D7", "*"),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\\\^\\circ|°", ""),
]

class MathEvaluator:
    """
    Utility class for normalizing, extracting, and grading mathematical 
    answers from LLM outputs.
    """
    
    @staticmethod
    def normalize(text: str) -> str:
        """Standardizes LaTeX and plain text math expressions for comparison."""
        if not text:
            return ""
        
        # 1. Basic cleaning
        text = RE_SPECIAL_TOKENS.sub("", text).strip().lower()
        text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text) # Remove LaTeX delimiters
        
        # 2. Apply standard LaTeX fixes
        for pattern, replacement in LATEX_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
            
        # 3. Handle fractions and square roots
        text = re.sub(r"\\sqrt\s*(?:\{([^}]*)\}|(\S+))", r"sqrt(\1\2)", text)
        text = re.sub(r"\\frac\s*(?:\{([^{}]+)\}\s*\{([^{}]+)\}|(\S+)\s+(\S+))", 
                      r"(\1\3)/(\2\4)", text)
        
        # 4. Clean symbols and common artifacts
        text = text.replace("\\%", "%").replace("$", "").replace("^", "**")
        text = text.replace("{", "").replace("}", "")
        
        # 5. Handle mixed numbers (e.g., "1 1/2" -> "1+1/2") and commas in thousands
        text = re.sub(r"(?<=\d)\s+(\d+/\d+)", r"+\1", text)
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
        
        return text.strip()

    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        """Extracts content from the last \\boxed{...} block, handling nested braces."""
        start_tag = r"\boxed"
        idx = text.rfind(start_tag)
        if idx == -1:
            return None

        # Find the opening brace of the boxed content
        content_start = text.find("{", idx + len(start_tag))
        if content_start == -1:
            return None

        # Traverse to find the matching closing brace (nesting aware)
        brace_depth = 0
        for i in range(content_start, len(text)):
            if text[i] == "{":
                brace_depth += 1
            elif text[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[content_start + 1 : i].strip().strip("$")
        return None

    @classmethod
    def get_final_candidate(cls, text: str, fallback: str = "number_then_full") -> str:
        """Orchestrates extraction: priority given to boxed, then numeric fallback."""
        if not text:
            return ""
        
        boxed = cls.extract_boxed_answer(text)
        if boxed:
            return boxed
        
        if "number" in fallback:
            numbers = RE_NUMBER.findall(text)
            if numbers:
                return numbers[-1]
        
        return text if fallback == "number_then_full" else ""

    # @staticmethod
    # def parse_to_sympy(expr: str):
    #     """Safely parses a string into a SymPy expression for mathematical evaluation."""
    #     try:
    #         return spp.parse_expr(
    #             expr,
    #             transformations=(*spp.standard_transformations, 
    #                              spp.implicit_multiplication_application),
    #             evaluate=True
    #         )
    #     except (SympifyError, SyntaxError, TypeError, TokenError):
    #         return None

    @staticmethod
    def parse_to_sympy(expr: str):
        """Safely parses a string into a SymPy expression, catching parser crashes."""
        if not expr or not isinstance(expr, str):
            return None
            
        try:
            return spp.parse_expr(
                expr,
                transformations=(*spp.standard_transformations, 
                                 spp.implicit_multiplication_application),
                evaluate=True
            )
        except (SympifyError, SyntaxError, TypeError, TokenError, IndexError):
            return None
        
    # @classmethod
    # def is_equivalent(cls, ground_truth: str, prediction: str) -> bool:
    #     """Determines if two math strings represent the same mathematical value."""
    #     norm_gt = cls.normalize(ground_truth)
    #     norm_pred = cls.normalize(prediction)
        
    #     if norm_gt == norm_pred:
    #         return True
            
    #     sym_gt = cls.parse_to_sympy(norm_gt)
    #     sym_pred = cls.parse_to_sympy(norm_pred)
        
    #     if sym_gt is not None and sym_pred is not None:
    #         try:
    #             # Core logic: if (A - B) simplifies to 0, they are equal
    #             return simplify(sym_gt - sym_pred) == 0
    #         except Exception:
    #             pass
    #     return False

    @classmethod
    def is_equivalent(cls, ground_truth: str, prediction: str) -> bool:
        """Determines if two math strings represent the same mathematical value."""
        norm_gt = cls.normalize(ground_truth)
        norm_pred = cls.normalize(prediction)
        
        # 1. Direct match check
        if norm_gt == norm_pred:
            return True
            
        # 2. Handle Equations: Convert 'left = right' to 'left - (right)'
        # This allows SymPy to simplify the expression to zero
        def prepare_for_sympy(expr_str):
            if "=" in expr_str:
                parts = expr_str.split("=")
                if len(parts) == 2:
                    return f"({parts[0]}) - ({parts[1]})"
            return expr_str

        sym_gt = cls.parse_to_sympy(prepare_for_sympy(norm_gt))
        sym_pred = cls.parse_to_sympy(prepare_for_sympy(norm_pred))
        
        # 3. Safe Evaluation
        if sym_gt is not None and sym_pred is not None:
            try:
                # If both are equations or both are values, their difference 
                # should simplify to 0.
                return simplify(sym_gt - sym_pred) == 0
            except Exception:
                pass
        
        # 4. Final Fallback: Basic string containment or set-based matching
        # useful for coordinate geometry or lists of numbers
        return norm_gt == norm_pred