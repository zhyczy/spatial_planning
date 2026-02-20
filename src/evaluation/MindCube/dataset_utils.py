import re
from typing import Iterable, Optional

# MindCube is a cognitive map benchmark with multiple choice questions
MINDCUBE_QUESTION_TYPES = [
    "perpendicular", 
    "parallel",
    "spatial_reasoning",
    "cognitive_map"
]

def _extract_answer(text: str) -> str:
    """Return content inside the last <answer>...</answer> block if it exists."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1] if matches else text


def clean_text(text: str, exclude_chars: Iterable[str] = ("\n", "\r")) -> str:
    """Normalize model output to a simple, comparable string."""
    cleaned = _extract_answer(text)

    for char in exclude_chars:
        cleaned = cleaned.replace(char, " ")

    cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace
    return cleaned.strip().rstrip(".").upper()  # Answers are typically A, B, C, D


def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, D) from model output.
    
    Handles various formats:
    - "A"
    - "The answer is A"
    - "A."
    - "Option A"
    """
    cleaned = clean_text(text)
    
    # Try to find a clear answer letter
    # Pattern 1: Just the letter alone or with punctuation
    pattern_letter = r'\b([A-D])\b'
    matches = re.findall(pattern_letter, cleaned)
    if matches:
        return matches[-1]  # Return last match
    
    # Pattern 2: "answer is X" or "option X"
    pattern_answer = r'(?:ANSWER|OPTION)(?:\s+IS)?\s*([A-D])'
    matches = re.findall(pattern_answer, cleaned)
    if matches:
        return matches[-1]
    
    # If no clear pattern, return first letter A-D found
    for char in cleaned:
        if char in 'ABCD':
            return char
    
    return None


def mindcube_reward(clean_ans_gt: str, clean_ans_pred: str) -> float:
    """Calculate reward (exact match) for MindCube dataset."""
    # Extract answer letters
    gt_letter = extract_answer_letter(clean_ans_gt)
    pred_letter = extract_answer_letter(clean_ans_pred)
    
    if gt_letter is None or pred_letter is None:
        return 0.0
    
    return 1.0 if pred_letter == gt_letter else 0.0


def calculate_mindcube_metrics(results):
    """Calculate detailed metrics for MindCube evaluation."""
    if not results:
        return {
            "per_category": {},
            "overall": {"accuracy": 0.0, "count": 0},
        }

    import pandas as pd
    
    df = pd.DataFrame(
        [
            {
                "reward": res.get("reward", 0.0),
                "category": res["sample"].get("category", ["unknown"])[0] if res["sample"].get("category") else "unknown",
                "type": res["sample"].get("type", "unknown"),
            }
            for res in results
        ]
    )

    # Per-category scores and counts
    per_category = {}
    if 'category' in df.columns:
        for cat, group in df.groupby("category"):
            per_category[cat] = {
                "score": float(group["reward"].mean()), 
                "count": int(len(group))
            }

    # Per-type scores and counts
    per_type = {}
    if 'type' in df.columns:
        for typ, group in df.groupby("type"):
            per_type[typ] = {
                "score": float(group["reward"].mean()), 
                "count": int(len(group))
            }

    # Overall accuracy
    overall_acc = float(df["reward"].mean()) if len(df) else 0.0
    overall_count = len(df)

    return {
        "per_category": per_category,
        "per_type": per_type,
        "overall": {"accuracy": overall_acc, "count": overall_count},
    }
