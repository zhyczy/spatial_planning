import re
from typing import Iterable, Optional

import torch

# SAT question types from MindJourney dataset
SAT_QUESTION_TYPES = [
    "obj_movement",
    "ego_movement",
    "obj_position",
    "spatial_reasoning",
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
    return cleaned.strip().rstrip(".").lower()


def sat_reward(clean_ans_gt: str, clean_ans_pred: str) -> float:
    """Calculate reward (exact match) for SAT dataset."""
    return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0


def calculate_sat_metrics(results):
    """Calculate detailed metrics for SAT evaluation."""
    if not results:
        return {
            "per_question_type": {},
            "overall": {"accuracy": 0.0, "count": 0},
        }

    import pandas as pd
    
    df = pd.DataFrame(
        [
            {
                "reward": res.get("reward", 0.0),
                "question_type": res["sample"].get("question_type"),
            }
            for res in results
        ]
    )

    # Per-question-type scores and counts
    per_qtype = {
        qtype: {"score": float(group["reward"].mean()), "count": int(len(group))}
        for qtype, group in df.groupby("question_type")
    }

    # Overall accuracy
    overall_acc = float(df["reward"].mean()) if len(df) else 0.0
    overall_count = len(df)

    return {
        "per_question_type": per_qtype,
        "overall": {"accuracy": overall_acc, "count": overall_count},
    }
