"""
answer_format.py — single source of truth for the train/eval answer template.

The supervised assistant turn is `<answer>{letter}</answer>`. The chat
template is invoked with `enable_thinking=False` so both training and deploy
get the SAME prompt prefix (Qwen3.5 auto-fills an empty `<think></think>`
block before the assistant content; without `enable_thinking=False` only the
opening `<think>` is added at deploy time, leaving the model in thinking
mode where it produces reasoning instead of the bare answer).

`compute_letter_offset(tokenizer)` returns the index of the letter token
within the masked-subset of the labels (i.e. the index inside the
`<answer>{letter}</answer><|im_end|>\\n` tokenization). This is NOT the
same as `len(tokenize("<answer>"))` — Qwen's BPE merges `>` with the next
character into a single token (`>A`, `>B`, `>C`, `>D` map to four
distinct ids), so tokenizing `<answer>` standalone gives `[<, answer, >]`
(3 tokens) but tokenizing `<answer>B` gives `[<, answer, >B]` (3 tokens
where the last contains the letter). The robust way is to probe with a
known letter and search for it.
"""
from __future__ import annotations


# The supervised assistant content is just the wrapped letter — the chat
# template appends the closing `<|im_end|>` itself.
def format_answer(letter: str) -> str:
    return f"<answer>{letter}</answer>"


# `<|im_end|>\n` follows in the chat-template-rendered text. The dataset
# computes `suffix_ids = tokenize(format_answer(letter) + IM_END_NEWLINE)`
# and masks `labels[:, :-len(suffix_ids)] = -100`.
IM_END_NEWLINE = "<|im_end|>\n"


def compute_letter_offset(tokenizer) -> int:
    """Position of the letter token inside the masked-suffix subset.

    We tokenize a probe (`<answer>B</answer><|im_end|>\\n`) and find the
    token whose decoded form contains the uppercase letter `B`. This is
    robust to BPE merges (`>B`, `>A`, etc. are single tokens, distinct
    from standalone `B` which is its own token id when not preceded by `>`).
    """
    probe = format_answer("B") + IM_END_NEWLINE
    ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    for i, tid in enumerate(ids):
        if "B" in tokenizer.decode([tid]):
            return i
    raise RuntimeError(
        f"Could not locate the letter token in tokenized {probe!r}: ids={ids}"
    )
