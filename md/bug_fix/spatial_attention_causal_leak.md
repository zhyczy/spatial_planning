# SpatialAttnWrapper Causal-Mask Leak — Postmortem

**Severity:** Critical — silently destroyed all training runs of `train_atten.py`
prior to 2026-04-27. Models reported `loss=0.0000 / acc=1.0000` on inline eval
but were actually learning a spurious cheat (read GT answer through unmasked
attention) that did not survive autoregressive generation (offline eval = 0.22,
near-random).

**Affected files:**
- [src/models/spatial_attention_llm.py](../../src/models/spatial_attention_llm.py)
- [src/models/answer_llm.py](../../src/models/answer_llm.py)
- [train_atten.py](../../train_atten.py)

**Root cause:** Qwen3.5's `create_causal_mask` returns `None` (so SDPA can use
its built-in `is_causal=True` flag for performance). Our `SpatialAttnWrapper`
had a fallback `else: new_mask = bias` that silently replaced the missing mask
with a zero tensor, which SDPA then interpreted as "explicit non-causal mask"
→ all causal masking disabled.

---

## 1. Symptoms

```
[eval] global_step=00050  mindcube_lm_loss=0.0000  mindcube_acc=1.0000  (n=1050)
[eval] global_step=00050  spinbench_lm_loss=0.0000  spinbench_acc=1.0000  (n=2742)
```

vs. offline `evaluation.py` on the same checkpoint:

```json
{ "overall_accuracy": 0.223, "correct_samples": 234, "total_samples": 1050 }
```

A 4-choice MCQ has 25% random baseline. Inline reports perfect, offline reports
below random. Two metrics on the *same model* differing by ~80 points = either
one of them is broken or there's a fundamental train/eval mismatch.

---

## 2. The Diagnostic Journey

This bug was buried under three other unrelated bugs. Each fix peeled back one
layer and exposed the next. Useful as a worked example of debugging deep ML
stacks: each stage produced *new* error messages, not "still failing the same
way."

### Stage 1 — DDP `find_unused_parameters` failure

```
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. ...
Parameter indices which did not receive grad for rank 1: 26 27 28 29 62 63 ...
                                                         (8 groups of 4)
```

**First diagnosis (wrong):** wrapped 32 layers but linear-attn layers don't
consume the additive mask → 24 layers' bias_module unused.

**What we did:** added `layer_type == "linear_attention"` skip in
`patch_attention_layers_spatial`.

**What was actually happening:** the original `hasattr(layer, "self_attn")`
check already excluded linear-attn layers (they have `linear_attn` attribute,
not `self_attn`). The fix was redundant — wrapping was already only happening
on 8 layers. The 32 unused params were 8 × 4 = the 8 full-attn bias_modules
themselves.

### Stage 2 — Same DDP failure persists

Same exact indices reported. This finally surfaced the right diagnosis:

```
mm_token_type_ids was never being forwarded from the dataset →
SpatialAttnVanillaModel.forward saw mm_token_type_ids=None →
_spatial_cache stayed None →
SpatialAttnWrapper bypassed (the `if flat_xyz is None ...` short-circuit) →
bias_module was never called →
its parameters never received grad → DDP fail.
```

**The fix:** plumb `mm_token_type_ids` through the call chain:
- `train_atten.py` extract from batch + forward to model
- `AnswerOnlyModel.forward` accept and forward to spa_model
- (`Qwen3_5ForConditionalGeneration` already had it as a named param)

See [qwen3.5.md §3](../model_design/qwen3.5.md#3-mm_token_type_ids--modality-labeling-contract)
for the modality-label-contract documentation that came out of this.

### Stage 3 — SDPA dtype mismatch

```
RuntimeError: invalid dtype for bias - should match query's dtype
```

`bias_module.mlp` was created in `patch_attention_layers_spatial` *after* the
backbone was cast to bf16, so the MLP linears were fresh fp32. Bias output was
fp32, query was bf16, SDPA refused.

**The fix:** in [build_model](../../train_atten.py), cast all
`SpatialAttentionBias` modules to `bf16` after patching. Plus a defensive
`new_mask.to(hidden_states.dtype)` at the end of the wrapper forward.

### Stage 4 — The real bug surfaces

After all the above, training ran cleanly. Inline eval reported perfect
accuracy after only 50 steps. Offline eval said 22%.

Investigation: load the checkpoint, run inline eval logic on a few samples,
look at top-5 predictions:

```
sample 0: gt='C'  pred='C'  ok=True  loss=0.0000  top5=['C', 'c', '_C', 'Ch', '(C']
sample 1: gt='A'  pred='A'  ok=True  loss=0.0000  top5=['A', 'a', 'А', '(A', ' A']
sample 2: gt='B'  pred='B'  ok=True  loss=0.0000  top5=['B', 'Б', '楚', 'Ｂ', '_B']
```

The model was outputting the EXACT correct answer with overwhelming
confidence on every sample, including ones where the model shouldn't have
that capability at step 50. Either the model had memorized everything (with
2400 samples? on a 4B model? in 50 steps? no), or something was leaking.

**The smoking gun test:** truncate input_ids before the answer letter, see
if model still predicts the answer:

```python
ids_truncated = ids[:, :ans_pos]   # drop the answer and everything after
out = spa(input_ids=ids_truncated, ...)
pred = out.logits[0, -1].argmax(-1).item()
```

Result:
```
sample 0: gt='C'  pred='<|im_end|>'  ok=False
sample 1: gt='A'  pred='A'           ok=True
sample 2: gt='B'  pred='<|im_end|>'  ok=False
sample 5: gt='D'  pred='A'           ok=False
```

1/4 ≈ random. So when the answer is NOT in input_ids, the model has no idea.
But when the answer IS in input_ids, it predicts perfectly. **Causal mask is
broken.**

### Stage 5 — Bisecting the leak

The "perturbation test" — replace `input_ids[ans_pos]` with a different
letter, check whether `logits[ans_pos-1]` changes (it shouldn't with a
correct causal mask):

```
[T1] stock Qwen3.5, no wrap                 logits diff = 0.0000  ✓ no leak
[T2] swap inner model only, no mm_token     logits diff = 0.0000  ✓ no leak
[T3] swap + mm_token (V↔V hole on)          logits diff = 0.0000  ✓ no leak
[T4] swap + mm_token + wrap (zero bias)     logits diff = 2.1133  ✗ LEAK
```

The wrapper itself caused the leak — even with `bias_module` zero-initialized
(producing all-zero bias). Stripped the wrapper down to a pure pass-through
and added a print:

```
attention_mask shape=None  dtype=None
[wrapper as pure pass-through] logits diff = 0.0000   ✓ no leak with passthrough
```

**`attention_mask` arrived at the wrapper as `None`.** Stock Qwen3.5
deliberately doesn't pass an explicit causal mask — it relies on SDPA's
`is_causal=True` flag for performance.

---

## 3. The Real Mechanism

### How HF's SDPA path normally works

```python
# transformers/integrations/sdpa_attention.py:92
if is_causal is None:
    is_causal = causal_mask is None and query.shape[2] > 1

attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,   # if None, SDPA uses its own
    is_causal=is_causal,     # True if no explicit mask given
)
```

Two valid configurations:
1. **`attn_mask=None, is_causal=True`** — SDPA generates causal pattern internally (fast path)
2. **`attn_mask=<real mask>, is_causal=False`** — caller provides explicit mask

Stock Qwen3.5 uses config 1. We assumed it was using config 2.

### Our wrapper's broken `else` branch

```python
def forward(self, ..., attention_mask=None, ...):
    bias = self.bias_module(flat_xyz, vision_mask)    # (B, H, L, L)
    if attention_mask is not None:
        bias = bias.to(attention_mask.dtype)
        new_mask = attention_mask + bias
        ...
    else:
        new_mask = bias                                # ← THE BUG
    return self.attn(..., attention_mask=new_mask, ...)
```

When `attention_mask` was `None`, we set `new_mask = bias` (a zero tensor at
init) and passed it as `attention_mask` to `self.attn`. Now SDPA sees:
- `attn_mask = zero_tensor` (not None)
- `is_causal = causal_mask is None and ...` = **False** (because mask is non-None)
- It uses our zero mask instead of generating its own causal pattern
- **Every position can attend to every other position, including the future**

The model could literally read `input_ids[k+1]` when computing `logits[k]`.
With supervised teacher-forcing, the GT answer was always in the input, so
the model trivially "predicted" it.

### Why the defenses didn't catch it

Both safety mechanisms were silently disabled by the same root cause:

```python
# V↔V prefix-mask hole in TextModel.forward
if (self._spatial_cache is not None
        and causal_mask is not None      # ← guard, never True (always None)
        and (past_key_values is None or ...)):
    causal_mask = torch.where(is_vv_pair, 0, causal_mask)
    # NEVER EXECUTED
```

```python
# torch.where defense in wrapper
if attention_mask is not None:           # ← guard, never True (always None)
    ...
    is_masked = attention_mask < -1e4
    new_mask = torch.where(is_masked, attention_mask, new_mask)
    # NEVER EXECUTED
else:
    new_mask = bias                       # ← THE actual code path taken
```

Both `if X is not None` guards looked like reasonable defensive coding. But
they were guarding against *the wrong null case*: they assumed "None means
the upstream skipped the mask for some unusual reason, fall back gracefully."
The actual semantic was "None means SDPA will handle causal masking itself —
do not synthesize a replacement." Replacing None with a zero tensor *broke
the contract*.

---

## 4. The Fix

Two changes to [spatial_attention_llm.py](../../src/models/spatial_attention_llm.py):

### 4.1 Materialise causal_mask in TextModel.forward

```python
causal_mask = create_causal_mask(...)     # may return None
linear_attn_mask = self._update_linear_attn_mask(...)

# NEW: materialize 4D causal mask whenever we will add spatial bias
if causal_mask is None and self._spatial_cache is not None:
    seq_len = inputs_embeds.shape[1]
    finfo_min = torch.finfo(inputs_embeds.dtype).min
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), finfo_min,
                   device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        diagonal=1,
    )[None, None, :, :]                    # (1, 1, L, L)
```

This guarantees `causal_mask` is a real 4D tensor before reaching the V↔V
hole code and the wrapper. The materialization only happens when
`_spatial_cache` is active (i.e. when bias will actually be added) — for
plain LM forward without spatial info we still get HF's None optimization.

### 4.2 Replace catastrophic `else` with `raise`

```python
if attention_mask is None:
    raise RuntimeError(
        "SpatialAttnWrapper received attention_mask=None while "
        "spatial bias is being applied. This would disable causal "
        "masking. Ensure SpatialAttnVanillaTextModel.forward "
        "materialises causal_mask whenever _spatial_cache is set."
    )
```

If §4.1 ever fails to materialize for some path we missed, fail *loudly* at
the wrapper instead of silently leaking. **Silent failure that produces
suspiciously good metrics is the worst kind.**

---

## 5. Verification

Same perturbation test, after fix:

```
[after fix, with our wrapping]   logits diff = 0.0000   ✓ no leak
sample 0: gt='C'  pred='To'  ok=False
sample 1: gt='A'  pred='To'  ok=False    ← all wrong, all 'To'
sample 2: gt='B'  pred='To'  ok=False
sample 5: gt='D'  pred='To'  ok=False    ← consistent with stock Qwen3.5
                                            on a fresh (untrained) model
```

Causal masking is restored. Untrained model now predicts the same `'To'`
that stock Qwen3.5 would output — no more cheating.

The two mask defenses now actually run. See
[../model_design/spatial_attention.md §5.4](../model_design/spatial_attention.md)
for the truth table of the *now-functioning* mask system.

---

## 6. Damage Assessment

All checkpoints saved before this fix (`train_records/atten_mindcube/step_*`)
are *unrecoverable*. The model trained against a leaky causal mask, so:
- Whatever weights the LoRA / bias_module learned encode "look at the
  unmasked answer position" rather than spatial reasoning
- Inference via `.generate()` (autoregressive — no future tokens to leak)
  reduces to ~random performance (0.22 on a 4-way MCQ)
- The training metric (loss=0, acc=1) was the *signature of the bug*, not
  evidence of progress

**Action:** delete pre-2026-04-27 ckpts, retrain from scratch.

---

## 7. Lessons Learned

1. **`if X is not None` guards can be dead code in practice.** The guard
   says "do something safe in the rare None case." But if None is the
   *common* case (HF's SDPA optimization), the body is what runs almost
   never, not what runs almost always.

2. **A `None`-replacement fallback should preserve the contract, not
   substitute a default value.** `else: new_mask = bias` looked like
   reasonable defensive coding. But the contract of `attention_mask=None`
   was "no explicit mask, SDPA handles it" — replacing None with a tensor
   silently *changes* the contract. The right `else` was either pass `None`
   through or raise.

3. **Suspicious-good metrics are bug signals.** A model going from 25%
   random to 100% accuracy in 50 steps with 2400 samples on hard spatial
   reasoning is not "fast learning" — it's a leak.

4. **Train and eval should share the same forward path** *and* be sanity-
   checked against an independent eval. Inline teacher-forced eval and
   offline `.generate()` eval gave the same numbers in stage 0 of the
   project (when there was no leak); divergence between them in stage 4
   was the only signal that finally caught the bug. Don't trust a single
   metric.

5. **Layered debugging compounds.** Each of stages 1-3 had a real bug, but
   none of them were the bug we cared about — they were just *prerequisites*
   for the real one to be reachable. Without fixing them in order, the
   causal leak couldn't surface as a measurable miscalibration. This is
   common in deep stacks: the first error you see is rarely the last.

6. **Read library code before assuming default behaviors.** The fact that
   Qwen3.5 returns `None` from `create_causal_mask` (rather than
   constructing the mask explicitly) is a non-obvious optimization that
   our extension implicitly relied against. Five minutes of reading
   `transformers/integrations/sdpa_attention.py` would have caught this in
   advance — but only if we'd thought to look.

---

## 8. Tests Worth Adding (TODO)

To prevent regression of this exact class of bug, consider:

- **Causal-mask probe in CI**: a unit test that runs `forward()` twice with
  `input_ids[k+1]` perturbed and asserts `logits[k]` is unchanged for any
  layer / config combination. Should be a one-liner against any future
  attention modification.
- **Inline-vs-offline eval drift alarm**: log both teacher-forced inline acc
  and a small `.generate()` autoregressive acc during training; alert if
  they diverge by > 5 points. The bug would have been caught in step 50 by
  this alone.
- **Mode-collapse detector for top-1 confidence**: if the model is producing
  loss < 1e-4 over a multi-task eval set within the first 100 steps, log a
  warning. Real training never gets there that fast.
