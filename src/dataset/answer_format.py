"""
answer_format.py — single source of truth for the train/eval answer template.

The supervised assistant turn is `<answer>{content}</answer>`. The chat
template is invoked with `enable_thinking=False` so both training and deploy
get the SAME prompt prefix (Qwen3.5 auto-fills an empty `<think></think>`
block before the assistant content; without `enable_thinking=False` only the
opening `<think>` is added at deploy time, leaving the model in thinking
mode where it produces reasoning instead of the bare answer).

For MindCube MCQ training the content was originally just the letter
(`<answer>B</answer>`) but is now `<answer>{letter}. {option text}</answer>`
(see `format_answer_with_text`). The model gets a longer SUPERVISE tail
that includes the option semantics, which gives a meaningful CE signal at
every option-text token instead of just the letter.

`compute_letter_offset(tokenizer)` returns the index of the letter token
within the masked-subset of the labels (i.e. the index inside the
`<answer>{letter}</answer><|im_end|>\\n` tokenization). This is NOT the
same as `len(tokenize("<answer>"))` — Qwen's BPE merges `>` with the next
character into a single token (`>A`, `>B`, `>C`, `>D` map to four
distinct ids), so tokenizing `<answer>` standalone gives `[<, answer, >]`
(3 tokens) but tokenizing `<answer>B` gives `[<, answer, >B]` (3 tokens
where the last contains the letter). The robust way is to probe with a
known letter and search for it. The offset is unchanged when the answer
text is `<answer>B. ...</answer>` — `>B` is still the letter token at
index 2 of the masked subset.
"""
from __future__ import annotations

import re


# The supervised assistant content is just the wrapped letter — the chat
# template appends the closing `<|im_end|>` itself.
def format_answer(letter: str) -> str:
    return f"<answer>{letter}</answer>"


# `<answer>{letter}. {option_text}</answer>` — option text extracted from
# inline-options MCQ questions like
#   "... A. opt1 B. opt2 C. opt3 D. opt4"
# The non-greedy capture stops at the next `\s+[A-Z]\.\s+` (next option label)
# or end-of-string, so it works for any option position.
_OPTION_TEXT_RE = re.compile(
    r"({letter})\.\s+(.+?)(?=\s+[A-Z]\.\s+|\s*$)", re.DOTALL,
)


_OPTION_TEXT_MAX = 200   # safety cap — runaway match would slurp until end-of-string


def format_answer_with_text(letter: str, question: str) -> str:
    """Build ``<answer>{letter}. {option_text}</answer>`` by extracting the
    option text matching *letter* from the inline-options *question* string.

    MindCube questions end with ``... A. opt1 B. opt2 C. opt3 D. opt4``.
    For non-MCQ-formatted questions (e.g. SpinBench uses ``A: <image>``
    with colons, or has descriptive text like ``View C.`` that would falsely
    match a letter+period pattern), fall back to ``<answer>{letter}</answer>``.

    Two safeguards prevent runaway extraction:
      1. **Inline-options precheck**: the question must contain BOTH
         ``A.<whitespace>`` AND ``B.<whitespace>`` patterns. Without these,
         an isolated ``X.`` in description text won't trigger extraction.
      2. **Length cap**: extracted option text must be < 200 chars. Real MCQ
         options are short; longer matches indicate the regex slurped past
         the option block.
    """
    if not letter or not isinstance(letter, str) or len(letter.strip()) != 1:
        return format_answer(letter)
    L = letter.strip().upper()
    if not ("A" <= L <= "Z"):
        return format_answer(letter)

    q = question or ""
    # Inline-options precheck — require A. AND B. inline markers
    if not (re.search(r"\bA\.\s", q) and re.search(r"\bB\.\s", q)):
        return format_answer(L)

    pat = re.compile(
        rf"\b({re.escape(L)})\.\s+(.+?)(?=\s+[A-Z]\.\s+|\s*$)", re.DOTALL,
    )
    m = pat.search(q)
    if m:
        text = m.group(2).strip()
        # Reject placeholder-only options (SpinBench-style image-options where
        # the question text reads "A. <image> B. <image>"). Wrapping
        # <answer>B. <image></answer> would teach the model to emit a special
        # token literal as text, which is meaningless.
        placeholder_markers = ("<image>", "<video>")
        if any(p in text for p in placeholder_markers):
            return format_answer(L)
        if 0 < len(text) <= _OPTION_TEXT_MAX:
            return f"<answer>{L}. {text}</answer>"
    return format_answer(L)


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


# ── Answer extraction (shared by deploy eval + train periodic eval) ───────
# Mirrors the implementations in evaluation.py so train_atten.py can reuse
# the same generative-eval semantics without importing the heavy evaluation
# module. evaluation.py keeps its local copies for backward compatibility.

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?")


def extract_answer_content(text: str) -> str:
    """Return the raw inner content of the *last* ``<answer>...</answer>`` tag."""
    if not text or not isinstance(text, str):
        return ""
    matches = _ANSWER_TAG_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_answer_letter(text: str) -> str:
    """Extract the multiple-choice letter from a `<answer>...</answer>` tag.

    Accepts both strict (`<answer>X</answer>`) and richer
    (`<answer>X. option text</answer>`) variants. Returns "" if no tag or
    the content lacks a leading letter.
    """
    content = extract_answer_content(text)
    if not content:
        return ""
    m = re.match(r"\s*([A-Za-z])(?:\s|[.)]|$)", content)
    if m:
        return m.group(1).upper()
    return ""


def extract_answer_number(text: str) -> str:
    """Extract a numeric answer from `<answer>...</answer>` (fill-format)."""
    content = extract_answer_content(text)
    if content:
        m = _NUMBER_RE.search(content)
        if m:
            return m.group(0)
    nums = _NUMBER_RE.findall(text or "")
    return nums[-1] if nums else ""


# ── `<image>` placeholder interleave helper (shared by train + eval) ──────
# Some datasets (notably SpinBench) embed literal `<image>` placeholders in
# the question text. These are NOT Qwen vision tokens (real ones are
# `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>`); when left in
# place they tokenize as `<`, `image`, `>` text tokens and the model learns
# to wrap answers in `<image>X</image>` instead of `<answer>X</answer>`.
# Verified on SpinBench step_500: fixing this lifts deploy acc 24% → 44%
# on a 50-sample slice (and goes from 32/50 `<image>X</image>` outputs to
# 0/50 — 100% emit `<answer>X</answer>` after the fix).

def build_interleaved_content(question: str, image_paths_or_imgs: list) -> list:
    """Build a chat-template ``content`` list with `<image>` placeholders
    in *question* replaced by the corresponding entries in *image_paths_or_imgs*
    (file paths or PIL images — passed straight into ``{"type":"image","image":...}``).

    When the placeholder count exactly matches ``len(image_paths_or_imgs)`` and
    is > 0, returns the interleaved layout
    ``[text_0, img_0, text_1, img_1, ..., text_N]`` (any empty text segments
    dropped). Otherwise falls back to the legacy "all images at front + text
    at end" layout, with stray `<image>` text-tokens stripped from the text
    so they can't pollute generation.

    Mirrors the same fallback used by Eval_Dataset_Coord / evaluation.py
    so train and eval share one prompt-construction policy.
    """
    n_ph = question.count("<image>")
    if n_ph > 0 and n_ph == len(image_paths_or_imgs):
        parts = question.split("<image>")
        content: list = []
        for i, txt in enumerate(parts):
            if txt:
                content.append({"type": "text", "text": txt})
            if i < n_ph:
                content.append({"type": "image", "image": image_paths_or_imgs[i]})
        return content

    cleaned = question.replace("<image>", "")
    content = [{"type": "image", "image": img} for img in image_paths_or_imgs]
    content.append({"type": "text", "text": cleaned})
    return content
