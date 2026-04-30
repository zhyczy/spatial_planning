# Train / Eval / Deploy Alignment Contract

**Date:** 2026-04-30 (updated; original draft 2026-04-28)
**Scope:** All 5 training methods (`atten`, `correspondence`, `coordinate`,
`alternate`, `rl`), the VST SFT auxiliary track, and the offline evaluator
(`evaluation.py`).

Cross-references:
- Postmortem motivating this unification: [../bug_fix/train_eval_paradigm_mismatch.md](../bug_fix/train_eval_paradigm_mismatch.md)
- VST Q&A category breakdown: [../dataset/vst.md](../dataset/vst.md)
- Train/eval input template comparison + zero-shot baseline prompts: [input_template.md](input_template.md)
- Format helper: [../../src/dataset/answer_format.py](../../src/dataset/answer_format.py)
- Leak regression test: [../../tests/test_atten_no_leak.py](../../tests/test_atten_no_leak.py)

---

## 0. TL;DR

Every training method, training-time eval, and deploy-time eval shares **one**
assistant-turn template family (`<answer>...</answer>`), **one** label-mask
scheme (suffix-only supervision per turn), and **one** extractor pipeline.
The five methods only differ in the auxiliary spatial signal (xyz / coord /
rotation / attention bias) — the LM head and the answer-extraction pathway
are identical across all of them.

| Layer | Constant across all methods |
|---|---|
| Assistant content | `<answer>{content}</answer>` (`format_answer` for MCQ; raw GPT response for VST) |
| Chat template flag | `enable_thinking=False` |
| MCQ suffix tokens (Qwen3.5-VL) | `[<, answer, >X, </, answer, >, <\|im_end\|>, \n]` (8) |
| MCQ label mask | `labels[:, :-8] = -100` |
| Letter token offset (in masked subset) | **2** (BPE merges `>X`) |
| lm_loss formula | `F.cross_entropy(shift_logits[masked], shift_labels[masked])` |
| Letter-acc formula | `argmax(shift_logits[ans_start + LETTER_OFFSET - 1]) == letter_id` |
| Multi-turn (VST `si_measurement`) | one `<answer>...</answer>` block unmasked **per turn** |

---

## 1. Concrete walkthrough — `train_atten.py`

This section uses `train_atten.py` as the canonical example. The other 4
methods follow the same shape; per-method differences are listed in §5.

### 1.1 Training step

#### 1.1.1 Prompt template (built by the dataset class)

`MindCube_Train_Dataset.__getitem__` ([train_dataset_qwen35.py](../../src/dataset/train_dataset_qwen35.py))
wraps the bare letter from the JSONL into the standard format:

```python
from .answer_format import format_answer, IM_END_NEWLINE
formatted_answer = format_answer(_answer)        # "<answer>B</answer>"
text_full = self.processor.apply_chat_template(
    [{"role": "user",      "content": qa_content},
     {"role": "assistant", "content": formatted_answer}],
    tokenize=False, add_generation_prompt=False,
    enable_thinking=False,
)
proc_out = self.processor(
    text=[text_full], images=images,
    return_tensors="pt", padding=False,
)
suffix_ids = self.processor.tokenizer(
    formatted_answer + IM_END_NEWLINE, add_special_tokens=False,
)["input_ids"]                                    # 8 tokens for Qwen3.5-VL
labels = proc_out["input_ids"].clone()
labels[0, :-len(suffix_ids)] = -100                # supervise only the suffix
```

The fully-rendered text:

```
<|im_start|>user
[images][question]<|im_end|>
<|im_start|>assistant
<think>

</think>

<answer>B</answer><|im_end|>
```

The last 8 tokens (the supervised region) are
`[<, answer, >B, </, answer, >, <|im_end|>, \n]`.

#### 1.1.2 Forward + lm_loss

The training loop ([train_atten.py:499-507](../../train_atten.py#L499-L507))
calls the wrapper model:

```python
_, loss, _ldict = model(
    input_ids         = input_ids,
    attention_mask    = attention_mask,
    pixel_values      = pixel_values,
    image_grid_thw    = image_grid_thw,
    image_xyz         = image_xyz,
    mm_token_type_ids = mm_token_type_ids,
    labels            = labels,
)
(loss / args.grad_accum).backward()
```

Inside `AnswerOnlyModel.forward` ([answer_llm.py:60-74](../../src/models/answer_llm.py#L60-L74)),
lm_loss is computed by:

```python
outputs       = self.spa_model(**fwd_kwargs)
logits        = outputs.logits                       # (1, L, V)
shift_logits  = logits[:, :-1, :]                    # (1, L-1, V)
shift_labels  = labels[:, 1:].to(logits.device)      # (1, L-1)
mask          = shift_labels[0] != -100              # (L-1,)
shift_logits  = shift_logits[0, mask]                # (8, V) — masked subset
shift_labels  = shift_labels[0, mask]                # (8,)
lm_loss       = F.cross_entropy(shift_logits, shift_labels)
return None, lm_loss, {"lm_loss": lm_loss.item()}
```

So the per-step loss is **mean cross-entropy across exactly 8 supervised
positions** (the entire `<answer>{letter}</answer><|im_end|>\n` block).

### 1.2 Eval step (training-time, every `args.eval_steps`)

#### 1.2.1 Prompt template (eval dataset)

`Eval_Dataset_Coord.__getitem__` ([eval_dataset_qwen35.py](../../src/dataset/eval_dataset_qwen35.py))
uses **the same** `format_answer` + `enable_thinking=False` + 8-token
suffix mask as training:

```python
from .answer_format import format_answer, IM_END_NEWLINE
formatted_answer = format_answer(_answer)
text_full = self.processor.apply_chat_template(
    [{"role": "user",      "content": content},
     {"role": "assistant", "content": formatted_answer}],
    tokenize=False, add_generation_prompt=False,
    enable_thinking=False,
)
proc_out = self.processor(
    text=[text_full], images=images,
    return_tensors="pt", padding=False,
)
suffix_ids = self.processor.tokenizer(
    formatted_answer + IM_END_NEWLINE, add_special_tokens=False,
)["input_ids"]
labels = proc_out["input_ids"].clone()
labels[0, :-len(suffix_ids)] = -100
```

#### 1.2.2 Forward + lm_loss + letter-position acc

The eval block ([train_atten.py:549-651](../../train_atten.py#L549-L651))
calls the wrapper directly **without** going through `AnswerOnlyModel.forward`'s
loss head — it re-derives loss inline so it can also extract the
letter-position argmax in the same pass:

```python
LETTER_OFFSET = compute_letter_offset(processor.tokenizer)   # = 2 for Qwen3.5-VL

with torch.no_grad():
    out = _spa(                                  # spa_model = inner backbone
        input_ids         = t_ids,
        attention_mask    = t_mask,
        pixel_values      = t_pv,
        image_grid_thw    = t_thw,
        image_xyz         = t_xyz,
        mm_token_type_ids = t_mm,
        output_hidden_states = False,
        return_dict          = True,
    )
    logits = out.logits
    sl = logits[..., :-1, :].contiguous()        # shift_logits
    sb = t_lbl[..., 1:].contiguous()             # shift_labels

    # ── lm_loss (mathematically equivalent to AnswerOnlyModel formula) ──
    lm_loss = F.cross_entropy(
        sl.view(-1, sl.size(-1)),
        sb.view(-1),
        ignore_index=-100,
    )
    loss_sum += lm_loss.item()
    count    += 1

    # ── letter-position argmax (Option C) ──
    first_ans = (t_lbl[0] != -100).nonzero(as_tuple=False)
    if first_ans.numel() > 0:
        ans_start  = first_ans[0, 0].item()       # position of '<'
        letter_pos = ans_start + LETTER_OFFSET    # position of '>X' (letter token)
        sl_idx     = letter_pos - 1               # logits at this position
                                                  #   predict the next token = letter
        if 0 <= sl_idx < sl.shape[1]:
            pred   = sl[0, sl_idx, :].argmax(-1).item()
            target = sb[0, sl_idx].item()
            if pred == target:
                acc_sum += 1.0
```

The two lm_loss formulas (training-step wrapper vs eval-step inline) are
**mathematically equivalent** for `B = 1` — both compute the unweighted
mean cross-entropy over the 8 non-(-100) positions. With B=1
(`collate_fn` enforces this), the inline `ignore_index=-100` excludes
exactly the same positions the wrapper's `mask` selects in.

The two metrics get logged each eval cycle:

```python
log.info(
    f"[eval] global_step={global_step:05d}  "
    f"{ds_name}_lm_loss={avg_l:.4f}  "
    f"{ds_name}_acc={avg_a:.4f}  (n={count})"
)
wandb.log({f"eval/{ds_name}_lm_loss": avg_l,
           f"eval/{ds_name}_acc":     avg_a},
          step=global_step)
```

### 1.3 How `<think></think>` is handled (chat-template auto-fill)

Qwen3.5's chat template **always** wraps the assistant turn in
`<think>...</think>` — there is no flag to remove the tags themselves.
`enable_thinking=False` only fills the block with empty content
(`<think>\n\n</think>\n\n`); `enable_thinking=True` (default) leaves the
block open so the model can write reasoning into it.

This pipeline uses **`enable_thinking=False` everywhere** so the empty block
is auto-filled identically across train / eval / deploy. The result is:

| Stage | What happens to `<think></think>` |
|---|---|
| Training input | `input_ids` contain `[<think>, \n\n, </think>, \n\n]` (4 tokens) before `<answer>` |
| Training labels | All 4 tokens are masked (`-100`). Loss is **only** on the 8-token `<answer>{X}</answer><|im_end|>\n` suffix that follows — the model gets **zero gradient** on the think block |
| Training-time eval | Same as training (uses `Eval_Dataset_Coord` with the same template) |
| Deploy prompt | Already includes `<think>\n\n</think>\n\n` at the end — model doesn't need to generate the tags itself, it picks up at `</think>\n\n` and writes `<answer>X</answer>` directly |

Concrete sample structure (last 12 tokens of a real `MindCube_Train_Dataset`
batch with `gt_answer = "B"`):

```
position  | token        | label_status
──────────┼──────────────┼────────────
870       | <think>      | ignore (-100)
871       | \n\n         | ignore (-100)
872       | </think>     | ignore (-100)
873       | \n\n         | ignore (-100)
──────────┼──────────────┼────────────  ← deploy prompt ends here
874       | <            | SUPERVISE
875       | answer       | SUPERVISE
876       | >B           | SUPERVISE   ← letter (LETTER_OFFSET = 2)
877       | </           | SUPERVISE
878       | answer       | SUPERVISE
879       | >            | SUPERVISE
880       | <|im_end|>   | SUPERVISE
881       | \n           | SUPERVISE
```

**Why this works**: the model is never asked to *generate* the empty think
block at deploy time — the chat template hands it a pre-filled block in the
prompt. It only needs to learn what comes *after* `</think>\n\n`, which is
exactly the supervised 8-token region. Training and deploy contexts are
therefore byte-identical up through `</think>\n\n`.

**Why we don't disable the tags**: there's no way to. Qwen3.5's template is
hard-coded to emit them. Asking the model to write `<answer>...</answer>`
without the surrounding `<think>...</think>` would mean overriding the chat
template, which is brittle and breaks compatibility with stock-Qwen
inference paths.

**Future thinking-mode work**: if you want the model to *actually* reason
inside `<think>...</think>` before answering, you must (a) supply reasoning
text in the dataset's `formatted_answer` so it lands inside the tags, (b)
extend the loss mask to cover the reasoning tokens, and (c) flip
`enable_thinking=True` everywhere. The current pipeline assumes no
reasoning supervision is available, so the think block stays empty.

### 1.4 VST extension (added 2026-04-30)

`VST_Train_Dataset` / `VST_Train_Dataset_Coord` in
[train_dataset_qwen35.py](../../src/dataset/train_dataset_qwen35.py) reuse the
same `<answer>...</answer>` wrapper, with two modifications:

1. **Free-form content inside the tag.** VST's GPT response is the raw
   string from the parquet (e.g. `"C. left and forward"`, `"97 cm"`,
   multi-line metric explanations) — no `format_answer(letter)` collapse.
   Suffix length per turn varies; the dataset locates each
   `<answer>{a}</answer><|im_end|>\n` token subsequence in the tokenized
   chat and unmasks exactly those positions. Failure to locate raises
   `RuntimeError` rather than silently dropping supervision.

2. **Multi-turn supervision.** `si_measurement` rows hold 3 (Q, A) pairs
   per sample (3 different objects in one image). `_vst_extract_qa_pairs`
   returns the full list and `_vst_build_chat_with_labels` builds a
   multi-turn chat: images attach to the first user turn only, subsequent
   user turns are text-only, every assistant turn gets unmasked
   independently.

Tokenization mask is identical in spirit to §1.1.1 — `labels` start fully
`-100`, then each `<answer>...</answer><|im_end|>\n` block is restored —
just iterated per turn.

The cross-method table (§5) is unchanged; this extension is orthogonal to
the spatial-signal axis. A model SFT'd jointly on MindCube + VST learns
both `<answer>{letter}</answer>` and `<answer>{free-form}</answer>`
simultaneously, with `extract_answer_letter` / `extract_answer_number` /
`extract_answer_content` (§2.2) handling the parse on whichever form
appears at deploy time.

---

## 2. Concrete walkthrough — `evaluation.py` (deploy)

### 2.1 Prompt template

`prepare_batch_spa` ([evaluation.py:1233-1245](../../evaluation.py#L1233-L1245))
uses the same chat template with `add_generation_prompt=True`:

```python
messages = [{"role": "user", "content": content}]
prompt_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(
    text=[prompt_text],
    images=image_inputs if image_inputs else None,
    videos=video_inputs if video_inputs else None,
    return_tensors="pt", padding=False,
)
```

The rendered prompt ends at `</think>\n\n` — i.e. it is **a strict prefix of
the training text**. Concretely:

```
training:    ...<|im_start|>assistant\n<think>\n\n</think>\n\n<answer>B</answer><|im_end|>\n
deploy:      ...<|im_start|>assistant\n<think>\n\n</think>\n\n
                                                              ^
                                          model continues generation here
```

### 2.2 Generate + extract

`run_inference_spa` ([evaluation.py:1326-1331](../../evaluation.py#L1326-L1331))
calls greedy `model.generate()`:

```python
gen_kwargs = dict(
    **inputs_dev,
    max_new_tokens=max_new_tokens,        # 512 default
    do_sample=False,                       # greedy
    pad_token_id=processor.tokenizer.eos_token_id,
)
generated_ids = model.generate(**gen_kwargs)
trimmed       = generated_ids[0][inputs_dev["input_ids"].shape[1]:]
output        = processor.decode(
    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
)
```

`_make_result` parses the decoded text via three small helpers in
[evaluation.py](../../evaluation.py) (mirrored privately in
[xyz_validation.py](../../xyz_validation.py)):

```python
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE     = re.compile(r"(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?")

def extract_answer_content(text: str) -> str:
    """Raw inner content of the *last* <answer>...</answer> tag (or "")."""
    matches = _ANSWER_TAG_RE.findall(text or "")
    return matches[-1].strip() if matches else ""

def extract_answer_letter(text: str) -> str:
    """Leading MCQ letter inside <answer>...</answer> (or "")."""
    content = extract_answer_content(text)
    if not content: return ""
    m = re.match(r"\s*([A-Za-z])(?:\s|[.)]|$)", content)
    return m.group(1).upper() if m else ""

def extract_answer_number(text: str) -> str:
    """First numeric token inside <answer>...</answer>, with a last-number
    fallback over the full text. Lookbehind keeps `point-2` → 2 (not -2)."""
    content = extract_answer_content(text)
    if content:
        m = _NUMBER_RE.search(content)
        if m: return m.group(0)
    nums = _NUMBER_RE.findall(text or "")
    return nums[-1] if nums else ""
```

Behavior matrix (16/16 unit-tested cases):

| Model output | letter | number | content |
|---|---|---|---|
| `<answer>A</answer>` | A | "" | A |
| `<answer>C. left and forward</answer>` (VST style) | C | "" | C. left and forward |
| `<answer>D point-D</answer>` | D | "" | D point-D |
| `<answer>97 cm</answer>` | "" | 97 | 97 cm |
| `<answer>Distance[A,B]=0.7m\n...</answer>` | "" | 0.7 | (multi-line) |
| `<answer>point-2</answer>` | "" | 2 | point-2 |

The previous strict regex `<answer>\s*([A-Za-z])\s*</answer>` was tightened
during the 04-28 unification, then loosened on 04-30 to absorb the richer
content a model trained on VST may emit on MCQ benchmarks. Strict
single-letter outputs still parse correctly (full backward compatibility).

`compute_metrics` ([evaluation.py:1551-1575](../../evaluation.py#L1551-L1575))
finally compares case-insensitively:

```python
if pred.lower().strip() == gt.lower().strip():
    correct += 1
```

---

## 3. Why TF acc (training-time) ≡ Gen acc (deploy)

In greedy decoding with no causal-mask leak, the two paths emit the **same
letter token** because they share the same logits at the same absolute
sequence position.

```
TRAINING-TIME EVAL                                DEPLOY (evaluation.py)
─────────────────────                             ────────────────────────
forward(full_seq + GT, labels=...)                generate(prompt_only)
            │                                                 │
            ▼                                                 ▼
shift_logits[ans_start + 1].argmax              first prefill step:
            │                                   logits[L_prompt - 1].argmax
            ▼                                                 │
compare to labels[ans_start + 2]                              ▼
                                                emitted as new token, decoded,
                                                regex-extracted by the parser
```

Both `ans_start + 1` (TF view) and `L_prompt - 1` (gen view) refer to the
**same absolute position** in the underlying sequence: the position of the
`answer` token inside `<answer>{letter}</answer>`, predicting the
letter-fused token (`>A` / `>B` / `>C` / `>D`) as the next emission.
Identical weights, identical context, identical RoPE → identical logits.

Index arithmetic check:

```
labels: [..., -100, -100, <, answer, >X, </, answer, >, <|im_end|>, \n]
                          ↑
                          ans_start              (= L - 8, suffix start)
                          ↑       ↑       ↑
                          ↑       ↑       ans_start + 2 = letter token position
                          ↑       ↑
                          ↑       ans_start + 1 = "answer" token; its logits
                          ↑                       predict the next token (= letter)
                          ↑
        first non-(-100) label position
```

For the wrapper-based methods the indexing looks different but lands at the
same place: `_sl_m = shift_logits[0, mask]` collects rows where
`shift_labels[i] != -100`, the first such `i` is `ans_start - 1`, so
`_sl_m[2] = shift_logits[ans_start + 1]` — same logits.

Verified end-to-end by [tests/test_atten_no_leak.py](../../tests/test_atten_no_leak.py)
(cos sim = 1.0, max abs diff = 0.0 across 3 sampled MindCube questions).

---

## 4. lm_loss reasonableness (sanity numbers)

A single forward on the *untrained* base Qwen3.5-VL with the new format
(8-token suffix) gives a per-token loss profile of:

```
position  | token       | loss (nats)
──────────┼─────────────┼─────────────
0         | <           | ≈ 22         ← base model never saw "< after </think>\n\n"
1         | answer      | ≈ 2-6
2         | >X (letter) | ≈ 4-9        ← will drop to ln(4) ≈ 1.39 after training
3         | </          | ≈ 0.5-2
4         | answer      | ≈ 0
5         | >           | ≈ 0
6         | <|im_end|>  | ≈ 0
7         | \n          | ≈ 0

avg ≈ 4.5
```

Training drives positions 0/1 to near-zero quickly (the model learns the
fixed format in the first few hundred steps); position 2 (the actual
answer-letter prediction) becomes the dominant residual loss. Expected
converged value: **avg lm_loss ≈ 0.05 – 0.2** depending on letter
prediction accuracy.

The `eval/{ds}_acc` metric is **0% at step 0** (base model predicts `>` not
`>X`) and rises as soon as position 2 starts learning.

---

## 5. Cross-method extension table

The other 4 methods follow the same pattern as §1, with method-specific
differences in the wrapper class and the auxiliary spatial signal:

| Method | Train dataset | Train wrapper | Train-eval lm_loss | Train-eval acc | evaluation.py `--method` |
|---|---|---|---|---|---|
| atten | `MindCube_Train_Dataset` | `AnswerOnlyModel` | inline (eval block) | inline letter-pos argmax | `atten` |
| correspondence | `MindCube_Train_Dataset` | `AnswerOnlyModel` | inline (eval block) | inline letter-pos argmax | `vanilla` |
| coordinate | `MindCube_Train_Dataset_Coord` | `CoordinateModel` | wrapper `_ldict["lm_loss"]` | wrapper `_ldict["acc"]` | `coordinate` |
| alternate | `MindCube_Train_Dataset_Rotation` | `RotationRoPEModel` | wrapper `_ldict["lm_loss"]` | wrapper `_ldict["acc"]` | `rotation` |
| rl | `MindCube_Train_Dataset_Rotation` | `RotationRoPEModel` | wrapper `_ldict["lm_loss"]` | wrapper `_ldict["acc"]` | `rotation_rl` |

For wrapper-based methods, the acc is computed inside the wrapper's
`forward` (see [coordinate_llm.py:236-242](../../src/models/coordinate_llm.py#L236-L242)
and [rotation_rope_llm.py:1351-1356](../../src/models/rotation_rope_llm.py#L1351-L1356)),
using `self.letter_offset` set by the training script after model build:

```python
from src.dataset import compute_letter_offset
model.letter_offset = compute_letter_offset(processor.tokenizer)   # = 2
```

`load_spa_model` ([evaluation.py:507-541](../../evaluation.py#L507-L541))
branches per method but always: (1) wrap-then-PEFT-load for `atten` /
`decouple` / `polar`, (2) merge LoRA, (3) install method-specific
shims (`_eval_image_xyz` for atten, `_compute_xyz_pos` for decouple).

---

## 6. Backward compatibility

**MindCube-format checkpoints (`<answer>X</answer>`):** fully parseable by
the 04-30 lenient extractors — strict single-letter content matches the
leading-letter regex unchanged. No retraining required for stock MindCube
runs.

**Pre-04-28 bare-letter checkpoints (no `<answer>` wrapping at all):** still
broken — extractors require the tag pair. Such models must be retrained.

**VST-augmented checkpoints (jointly SFT'd on MindCube + VST):** parse on
all eval datasets via the lenient extractors. Free-form `<answer>{...}</answer>`
content from VST style is absorbed without scoring penalty.

The single-source-of-truth helper
[`src/dataset/answer_format.py`](../../src/dataset/answer_format.py) governs
the MCQ wrapping `<answer>{letter}</answer>` and `IM_END_NEWLINE`. The
extractor side ([evaluation.py](../../evaluation.py)
+ [xyz_validation.py](../../xyz_validation.py)) is now driven by
`_ANSWER_TAG_RE` and `_NUMBER_RE` constants. Format bumps require updating
both sides in lock-step.

---

## 7. Verification (status as of 2026-04-30)

Routine pipeline invariants (each verified at the linked location):

- `apply_chat_template(..., enable_thinking=False)` everywhere
  ([train_dataset_qwen35.py](../../src/dataset/train_dataset_qwen35.py),
  [eval_dataset_qwen35.py](../../src/dataset/eval_dataset_qwen35.py),
  [evaluation.py prepare_batch_spa](../../evaluation.py))
- MindCube assistant content via `format_answer(letter)` =
  `<answer>{letter}</answer>`; VST wraps raw GPT response in `<answer>...</answer>`
- `compute_letter_offset(tokenizer)` returns 2 for Qwen3.5-VL (BPE merges
  `>A`/`>B`/`>C`/`>D` into distinct single-token ids)
- All wrapper models compute lm_loss as
  `F.cross_entropy(shift_logits[masked], shift_labels[masked])`;
  inline eval uses `ignore_index=-100` (equivalent for B=1)
- All eval datasets parse correctly — all 10 benchmarks × ~200 samples
  pass extractor round-trip (1950/1950 via simulated `<answer>{gt}</answer>`
  inputs)
- Leak regression test passes:
  TF logits[L_p − 1] ≡ gen logits[L_p − 1]
  ([tests/test_atten_no_leak.py](../../tests/test_atten_no_leak.py))

After retraining: `evaluation.py:overall_accuracy ≈ wandb eval/{ds}_acc`
(within statistical noise from sample-set differences + bf16 drift).
