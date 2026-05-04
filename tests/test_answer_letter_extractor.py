"""Unit tests for the 4-tier extract_answer_letter fallback.

Run with:
    cd spatial_planning && python -m pytest tests/test_answer_letter_extractor.py -v

Covers the bug documented in md/bug_fix/answer_letter_extractor_no_fallback.md:
single-image SpinBench / SAT / EmbSpatial outputs are bare letters like 'A\\n'
that the old strict tag-only extractor scored as "" (=0% accuracy).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.dataset.answer_format import extract_answer_letter as A1
from evaluation import extract_answer_letter as A2


# Both implementations must stay in lockstep, so every test runs against both.
def both(text):
    r1 = A1(text); r2 = A2(text)
    assert r1 == r2, f"answer_format and evaluation disagree on {text!r}: {r1!r} vs {r2!r}"
    return r1


# ── T1: <answer> tag ───────────────────────────────────────────────────────
class TestTag:
    def test_strict(self):
        assert both("<answer>A</answer>") == "A"
        assert both("<answer>B</answer>") == "B"
        assert both("<answer>F</answer>") == "F"

    def test_with_trailing_text(self):
        assert both("<answer>A</answer>\n") == "A"
        assert both("foo<answer>C</answer>bar") == "C"

    def test_richer_content(self):
        assert both("<answer>A. blue chair</answer>") == "A"
        assert both("<answer>B) yes</answer>") == "B"

    def test_lowercase(self):
        assert both("<answer>a</answer>") == "A"

    def test_multiple_tags_takes_last(self):
        assert both("<answer>A</answer><answer>B</answer>") == "B"

    def test_tag_with_prose_t1b(self):
        # T1b: tag content is prose without leading letter+delim → last A-F wins
        assert both("<answer>The answer is A.</answer>") == "A"
        assert both("<answer>I think it's B</answer>") == "B"
        # When tag content STARTS with a letter+delim, T1a fires first (NOT T1b)
        # — leading letter is the strong signal for the dominant MindCube
        # format <answer>X. option text</answer>. Documented limitation: prose
        # like "<answer>A is wrong, B is right</answer>" returns 'A', not 'B'.
        assert both("<answer>A is wrong, B is right</answer>") == "A"

    def test_tag_with_no_letter(self):
        assert both("<answer>Yes</answer>") == ""
        assert both("<answer>123</answer>") == ""


# ── T2: explicit answer-indicator pattern ─────────────────────────────────
class TestAnswerIndicator:
    def test_answer_is(self):
        assert both("The answer is A.") == "A"
        assert both("the answer is B") == "B"
        assert both("I think the answer is C.") == "C"

    def test_answer_colon(self):
        assert both("Answer: D") == "D"
        assert both("answer: E\n") == "E"
        assert both("Answer:F") == "F"

    def test_i_choose(self):
        assert both("I choose A") == "A"
        assert both("I'll go with B") == "B"
        assert both("I pick C.") == "C"

    def test_with_parens(self):
        assert both("The answer is (A).") == "A"
        assert both("I choose (B)") == "B"


# ── T3: raw text starts with single letter + delimiter ────────────────────
class TestRawStart:
    def test_bare_letter(self):
        # The headline bug: SpinBench single-image outputs 'A\n'
        assert both("A") == "A"
        assert both("A\n") == "A"
        assert both("B\n") == "B"

    def test_with_punctuation(self):
        assert both("A.") == "A"
        assert both("A. blue chair") == "A"
        assert both("A) yes") == "A"
        assert both("A: see figure") == "A"

    def test_with_leading_whitespace(self):
        assert both("  A  ") == "A"
        assert both("\nA\n") == "A"

    def test_lowercase(self):
        assert both("a") == "A"
        assert both("a. mug") == "A"

    def test_a_in_word_doesnt_match_t3(self):
        # 'Apple' has 'A' at start but next char is 'p' (not a delimiter).
        # T3 fails. Should fall through to T4 (no standalone A-F → "").
        assert both("Apple") == ""

    def test_two_letters_in_a_row(self):
        # 'AA' has 'A' at start but next is 'A' (not a delimiter). T3 fails.
        # T4 finds standalone 'A's? No — \b([A-Fa-f])\b requires both sides
        # to be word boundary, but 'AA' is one word. So returns "".
        assert both("AA") == ""


# ── T4: last standalone letter (the prose case) ───────────────────────────
class TestLastLetter:
    def test_yes_a_is_closer(self):
        # User-flagged case: answer is correctly 'A'
        assert both("Yes A is closer") == "A"

    def test_according_to(self):
        # 'According' has no standalone A-F letter, so this returns ""
        # (model's output didn't actually pick a letter)
        assert both("According to my analysis") == ""
        # but 'According to my analysis, A' should extract A
        assert both("According to my analysis, A") == "A"
        assert both("According to figure, B is right") == "B"

    def test_t3_takes_precedence_over_t4(self):
        # When raw text starts with a letter+delim, T3 fires (FIRST letter
        # wins). T4 only triggers when no leading letter at start.
        # This is the right behavior for `'A. blue chair'` style outputs
        # but means prose like 'A is wrong, B is right' returns 'A'.
        assert both("A is wrong, B is right") == "A"   # T3 wins
        assert both("B is wrong, A is right") == "B"   # T3 wins
        assert both("A is closer than B")    == "A"   # T3 wins
        # Only when text doesn't start with A-F+delim does T4 fire:
        assert both("Yes, A is closer than B") == "B"  # T4 (last)
        assert both("So I'd say A is closer") == "A"  # T4 (last; only 1 letter)


# ── No letter at all ───────────────────────────────────────────────────────
class TestEmpty:
    def test_empty_string(self):
        assert both("") == ""

    def test_none(self):
        assert A1(None) == ""
        assert A2(None) == ""

    def test_non_string(self):
        assert A1(123) == ""
        assert A2(123) == ""

    def test_prose_with_no_a_to_f(self):
        # 'I cannot determine' — 'I' is NOT in A-F, no other letters → ""
        assert both("I cannot determine") == ""
        assert both("I cannot determine this from the image.") == ""

    def test_only_letters_outside_a_to_f(self):
        assert both("XYZ") == ""
        assert both("Hello world") == ""

    def test_single_letter_outside_a_to_f(self):
        assert both("T") == ""
        assert both("Y") == ""


# ── Smoke test against real failing samples ───────────────────────────────
class TestRealSpinBenchSamples:
    """Direct copies from vis_results/xyz_val_atten_couple_spinbench/spinbench/
    that previously scored 0 because of the bug."""

    def test_real_single_image_outputs(self):
        # These are actual model outputs, all should be 'A' (their GT)
        assert both("A\n") == "A"
        # And actual multi-image ones that occasionally drop the tag
        assert both("B\n") == "B"
        assert both("C\n") == "C"
